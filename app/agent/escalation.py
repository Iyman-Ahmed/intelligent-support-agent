"""
Escalation Engine — rule-based triggers + Gemini-based sentiment scoring.
Evaluates after every agent turn and flags conversations for human handoff.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

from google import genai
from google.genai import types

from app.agent.prompts import SENTIMENT_ANALYSIS_PROMPT
from app.models.conversation import Conversation, ConversationStatus

logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-2.0-flash"

# ---------------------------------------------------------------------------
# Escalation trigger patterns (rule-based)
# ---------------------------------------------------------------------------

LEGAL_KEYWORDS = [
    r"\blawsuit\b", r"\blitigation\b", r"\bsue\b", r"\bsuing\b",
    r"\bfraud\b", r"\bscam\b", r"\bchargeback\b", r"\bdispute\b",
    r"\battorney\b", r"\blawyer\b", r"\bbetter business bureau\b",
    r"\bfederal trade commission\b", r"\bFTC\b",
]

FRUSTRATION_KEYWORDS = [
    r"\bunacceptable\b", r"\bterrible\b", r"\bawful\b", r"\bhorrble\b",
    r"\bdisgusting\b", r"\boutrageous\b", r"\bnever.*again\b",
    r"\bworst.*ever\b", r"\bcancel.*account\b",
]

HUMAN_REQUEST_PATTERNS = [
    r"\bspeak.*human\b", r"\btalk.*human\b", r"\breal person\b",
    r"\bhuman agent\b", r"\blive agent\b", r"\bmanager\b",
    r"\bsupervisor\b", r"\bescalate\b",
]

MAX_TURNS_BEFORE_ESCALATION = 8
HIGH_REFUND_THRESHOLD       = 500.0


@dataclass
class EscalationDecision:
    should_escalate:  bool
    reason:           Optional[str]  = None
    urgency:          str            = "normal"    # normal | high | critical
    trigger_type:     str            = "none"      # rule | sentiment | llm | turn_limit
    sentiment_score:  Optional[float] = None
    confidence:       float          = 1.0


# ---------------------------------------------------------------------------
# Escalation Engine
# ---------------------------------------------------------------------------

class EscalationEngine:
    def __init__(self, gemini_client: Optional[genai.Client] = None):
        self.client          = gemini_client
        self._legal_re       = [re.compile(p, re.IGNORECASE) for p in LEGAL_KEYWORDS]
        self._frustration_re = [re.compile(p, re.IGNORECASE) for p in FRUSTRATION_KEYWORDS]
        self._human_re       = [re.compile(p, re.IGNORECASE) for p in HUMAN_REQUEST_PATTERNS]

    async def evaluate(
        self,
        conversation: Conversation,
        latest_user_message: str,
    ) -> EscalationDecision:
        # 1. Already escalated — skip
        if conversation.status == ConversationStatus.ESCALATED:
            return EscalationDecision(should_escalate=False)

        # 2. Human request patterns
        for pattern in self._human_re:
            if pattern.search(latest_user_message):
                return EscalationDecision(
                    should_escalate=True,
                    reason="Customer explicitly requested a human agent",
                    urgency="high",
                    trigger_type="rule",
                )

        # 3. Legal / fraud keywords
        for pattern in self._legal_re:
            if pattern.search(latest_user_message):
                return EscalationDecision(
                    should_escalate=True,
                    reason="Legal or fraud keyword detected in customer message",
                    urgency="critical",
                    trigger_type="rule",
                )

        # 4. VIP / enterprise customer — escalate after 3 turns
        if conversation.is_vip and conversation.turn_count >= 3:
            return EscalationDecision(
                should_escalate=True,
                reason="VIP customer with unresolved issue after 3 turns",
                urgency="high",
                trigger_type="rule",
            )

        # 5. Turn limit exceeded
        if conversation.turn_count >= MAX_TURNS_BEFORE_ESCALATION:
            return EscalationDecision(
                should_escalate=True,
                reason=f"Issue unresolved after {conversation.turn_count} turns",
                urgency="high",
                trigger_type="turn_limit",
            )

        # 6. Sentiment analysis via Gemini Flash
        sentiment = await self._analyse_sentiment(latest_user_message)
        if sentiment and sentiment.get("score", 0) < -0.7:
            return EscalationDecision(
                should_escalate=True,
                reason="Highly negative customer sentiment detected",
                urgency="high",
                trigger_type="sentiment",
                sentiment_score=sentiment.get("score"),
            )

        # 7. Frustration keywords (lighter rule)
        frustration_count = sum(
            1 for p in self._frustration_re if p.search(latest_user_message)
        )
        if frustration_count >= 2:
            return EscalationDecision(
                should_escalate=True,
                reason="Multiple frustration indicators in customer message",
                urgency="high",
                trigger_type="rule",
            )

        return EscalationDecision(should_escalate=False)

    async def _analyse_sentiment(self, text: str) -> Optional[dict]:
        """Use Gemini Flash to score sentiment (free tier)."""
        if not self.client:
            return None
        try:
            response = await self.client.aio.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    types.Content(
                        role="user",
                        parts=[types.Part(text=f"{SENTIMENT_ANALYSIS_PROMPT}\n\nMessage: {text}")],
                    )
                ],
                config=types.GenerateContentConfig(
                    max_output_tokens=256,
                    temperature=0.0,
                ),
            )
            raw = response.text.strip()
            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = re.sub(r"^```[a-z]*\n?", "", raw)
                raw = re.sub(r"\n?```$", "", raw)
            return json.loads(raw)
        except Exception as exc:
            logger.warning("Sentiment analysis failed: %s", exc)
            return None

    def check_refund_threshold(self, text: str) -> bool:
        """Check if message mentions a refund above the high threshold."""
        pattern = re.compile(
            r"refund.*?\$\s*(\d+(?:\.\d+)?)|(\$\s*\d+(?:\.\d+)?)\s*refund",
            re.IGNORECASE
        )
        for match in pattern.finditer(text):
            amount_str = match.group(1) or match.group(2)
            if amount_str:
                amount = float(re.sub(r"[^\d.]", "", amount_str))
                if amount > HIGH_REFUND_THRESHOLD:
                    return True
        return False
