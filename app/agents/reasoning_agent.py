"""
Agent 2 — Reasoning Agent.

Activated when Agent 1's confidence is below threshold, or when the query
is inferential (comparison, recommendation, best-for, etc.).

Uses gemini-2.0-flash-lite to reason over the top-N products returned by
the KB agent and synthesise a helpful, grounded answer.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# Model preference — cheapest / lowest-quota model first
_GEMINI_LITE = "gemini-2.0-flash-lite"
_GEMINI_FLASH = "gemini-2.0-flash"

_SYSTEM_PROMPT = """You are a knowledgeable and friendly Amazon product assistant.
Your job is to help customers find the right product from the provided catalogue.

Guidelines:
- Base your answer ONLY on the product data supplied in the user message.
- Be concise but thorough; use bullet points for comparisons.
- Include price, rating, and key differentiators when comparing.
- If asked for a recommendation, pick ONE and explain why briefly.
- Never invent specs or prices not present in the data.
- Respond in plain text / Markdown. No JSON.
"""


class ReasoningAgent:
    """
    Agent 2: uses Gemini to reason over product context.

    answer() is async to allow non-blocking LLM calls inside Gradio.
    """

    def __init__(self) -> None:
        self._client = None          # lazy-initialised on first call
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def answer(self, question: str, context_products: list[dict]) -> tuple[str, str]:
        """
        Returns (formatted_answer, agent_label).
        agent_label is always "Agent 2 (Reasoning)".
        """
        client = await self._get_client()
        prompt = self._build_prompt(question, context_products)

        # Try lite first, then fall back to flash
        for model in (_GEMINI_LITE, _GEMINI_FLASH):
            try:
                response = await client.aio.models.generate_content(
                    model=model,
                    contents=prompt,
                    config={"system_instruction": _SYSTEM_PROMPT,
                            "max_output_tokens": 512,
                            "temperature": 0.4},
                )
                text = response.text or ""
                return text.strip(), "Agent 2 (Reasoning)"
            except Exception as exc:
                err = str(exc)
                if "429" in err or "RESOURCE_EXHAUSTED" in err:
                    if "per_day" in err.lower() or "daily" in err.lower():
                        return (
                            "⚠️ Daily Gemini quota exhausted. "
                            "Please try again tomorrow or check back later.",
                            "Agent 2 (Reasoning)",
                        )
                    # rate-limited on this model — try next
                    continue
                # non-rate-limit error — surface it
                return (
                    f"⚠️ Could not generate a response: {err[:120]}",
                    "Agent 2 (Reasoning)",
                )

        return (
            "⚠️ Gemini rate-limited on all models. Please wait ~30 s and retry.",
            "Agent 2 (Reasoning)",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _get_client(self):
        """Lazy-init the Gemini client (avoids import at module load time)."""
        if self._client is not None:
            return self._client
        async with self._lock:
            if self._client is None:
                import os
                from google import genai  # noqa: PLC0415

                api_key = os.environ.get("GEMINI_API_KEY", "")
                self._client = genai.Client(api_key=api_key)
        return self._client

    @staticmethod
    def _build_prompt(question: str, products: list[dict]) -> str:
        """Serialise product context + question into a single user prompt."""
        if not products:
            return (
                f"Question: {question}\n\n"
                "No specific products are available in the catalogue right now. "
                "Please answer as best you can with general knowledge."
            )

        lines = [f"Question: {question}\n", "Available products:\n"]
        for p in products:
            lines.append(
                f"- **{p['title']}** (${p['price']}, ⭐{p['rating']}/5, "
                f"{p['review_count']:,} reviews)\n"
                f"  Brand: {p.get('brand', 'N/A')} | "
                f"Category: {p.get('category', 'N/A')}\n"
                f"  Features: {'; '.join(p.get('features', [])[:4])}\n"
                f"  Description: {p.get('description', '')[:200]}\n"
            )
        lines.append("\nPlease answer the question based on the products above.")
        return "\n".join(lines)
