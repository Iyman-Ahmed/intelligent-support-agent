"""
Amazon Q&A Router.

Decides whether Agent 1 (KBDirectAgent) or Agent 2 (ReasoningAgent)
should handle a given question, then returns the answer.

Routing logic:
  1. Agent 1 always runs first (zero API cost).
  2. If Agent 1 confidence >= CONFIDENCE_THRESHOLD *and* the query is NOT
     inferential → return Agent 1's answer.
  3. Otherwise hand off to Agent 2 with the top-5 products as context.

Inferential queries: comparisons, recommendations, "best for", pros/cons, etc.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

from app.agents.kb_agent import KBDirectAgent, load_kb_agent
from app.agents.reasoning_agent import ReasoningAgent

# Patterns that signal the question needs reasoning, not just lookup
_INFERENTIAL_RE = re.compile(
    r"\b("
    r"compare|comparison|vs\.?|versus|difference|which\s+is\s+better|"
    r"best\s+for|recommend|should\s+i|pros?\s+and\s+cons?|"
    r"worth\s+it|is\s+it\s+good|alternative|similar\s+to|"
    r"budget|affordable|cheap|value\s+for\s+money"
    r")\b",
    re.IGNORECASE,
)

DATA_FILE = Path(__file__).parent.parent.parent / "data" / "amazon_products.json"


class AmazonQARouter:
    """
    Orchestrates Agent 1 and Agent 2.

    Usage (async context):
        router = AmazonQARouter()
        answer, agent_label = await router.route("Which headphones have the best ANC?")
    """

    def __init__(self, data_file: Path = DATA_FILE) -> None:
        self._data_file = data_file
        self._kb_agent: KBDirectAgent | None = None
        self._reasoning_agent: ReasoningAgent | None = None
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def route(self, question: str) -> tuple[str, str]:
        """
        Returns (answer_text, agent_label).
        agent_label is "Agent 1 (KB Direct)" or "Agent 2 (Reasoning)".
        """
        await self._ensure_init()

        # Agent 1 — always try first
        kb_answer, confidence = self._kb_agent.answer(question)  # type: ignore[union-attr]

        inferential = bool(_INFERENTIAL_RE.search(question))

        if kb_answer and not inferential:
            return kb_answer, f"Agent 1 (KB Direct) — confidence {confidence:.2f}"

        # Fall through to Agent 2
        context_products = self._kb_agent.get_top_products(question, n=5)  # type: ignore[union-attr]
        answer, label = await self._reasoning_agent.answer(question, context_products)  # type: ignore[union-attr]
        return answer, label

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    async def _ensure_init(self) -> None:
        if self._kb_agent is not None:
            return
        async with self._lock:
            if self._kb_agent is None:
                self._kb_agent = load_kb_agent(self._data_file)
                self._reasoning_agent = ReasoningAgent()
