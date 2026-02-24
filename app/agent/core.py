"""
AI Agent Core — the agentic loop that powers multi-turn conversations.

Flow:
  1. Load conversation history from session store
  2. Append new user message
  3. Call Claude with tool_use enabled
  4. If Claude returns tool_use blocks → dispatch tools → inject results → loop
  5. When Claude returns end_turn (text only) → run escalation check → return
  6. Persist updated conversation
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import anthropic

from app.agent.escalation import EscalationEngine
from app.agent.prompts import (
    CONTEXT_SUMMARISATION_PROMPT,
    ESCALATION_SUMMARY_PROMPT,
    SUPPORT_AGENT_SYSTEM_PROMPT,
)
from app.agent.tools import TOOL_DEFINITIONS, ToolDispatcher
from app.models.conversation import (
    Conversation,
    ConversationStatus,
    MessageRole,
    ToolCallRecord,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PRIMARY_MODEL = "claude-sonnet-4-6-20250929"
FALLBACK_MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 1024
MAX_TOOL_ITERATIONS = 10          # Safety cap on agentic loop iterations
CONTEXT_WINDOW_TURNS = 20        # Rolling window for history sent to Claude
SUMMARISE_AFTER_TURNS = 30       # Compress history after this many turns


# ---------------------------------------------------------------------------
# Agent Core
# ---------------------------------------------------------------------------

class SupportAgentCore:
    """
    Orchestrates the full agentic loop:
    message → Claude → tool calls → Claude → … → final response
    """

    def __init__(
        self,
        anthropic_client: anthropic.AsyncAnthropic,
        tool_dispatcher: ToolDispatcher,
        escalation_engine: EscalationEngine,
        session_store=None,          # Optional: Redis-backed session store
    ):
        self.client = anthropic_client
        self.dispatcher = tool_dispatcher
        self.escalation = escalation_engine
        self.session_store = session_store

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def process_message(
        self,
        conversation: Conversation,
        user_message: str,
    ) -> Tuple[str, Conversation]:
        """
        Process a new user message through the full agent loop.
        Returns (assistant_reply_text, updated_conversation).
        """
        start_time = time.monotonic()

        # 1. Check for escalation BEFORE running the agent
        escalation_decision = await self.escalation.evaluate(
            conversation, user_message
        )

        # 2. Append user message to history
        conversation.add_message(MessageRole.USER, user_message)

        # 3. Handle pre-escalation (legal keywords, explicit human request)
        if escalation_decision.should_escalate and escalation_decision.trigger_type in (
            "rule",
        ):
            reply = await self._handle_escalation(
                conversation, escalation_decision
            )
            conversation.add_message(MessageRole.ASSISTANT, reply)
            self._record_timing(conversation, start_time)
            return reply, conversation

        # 4. Run agentic loop
        reply, tool_records = await self._agent_loop(conversation)

        # 5. Post-response escalation check (sentiment, turn limit)
        if not escalation_decision.should_escalate:
            escalation_decision = await self.escalation.evaluate(
                conversation, user_message
            )

        if escalation_decision.should_escalate:
            reply = await self._handle_escalation(
                conversation, escalation_decision, agent_reply=reply
            )

        # 6. Append final assistant reply
        conversation.add_message(
            MessageRole.ASSISTANT,
            reply,
            tool_calls=tool_records,
        )
        self._record_timing(conversation, start_time)

        # 7. Optionally compress history
        if conversation.turn_count >= SUMMARISE_AFTER_TURNS:
            await self._compress_history(conversation)

        return reply, conversation

    async def stream_message(
        self,
        conversation: Conversation,
        user_message: str,
    ) -> AsyncGenerator[str, None]:
        """
        Streaming variant — yields text chunks as they arrive.
        Full tool-use loop runs synchronously first, then streams final text.
        (True streaming with mid-loop tool use requires more complex handling.)
        """
        reply, updated_conv = await self.process_message(conversation, user_message)
        # Simulate streaming by yielding words
        for word in reply.split(" "):
            yield word + " "
            await asyncio.sleep(0.02)

    # ------------------------------------------------------------------
    # Agentic Loop
    # ------------------------------------------------------------------

    async def _agent_loop(
        self,
        conversation: Conversation,
    ) -> Tuple[str, List[ToolCallRecord]]:
        """
        Core agentic loop.
        Runs until Claude stops calling tools (stop_reason == "end_turn").
        """
        messages = conversation.to_claude_messages(max_turns=CONTEXT_WINDOW_TURNS)
        tool_records: List[ToolCallRecord] = []
        iterations = 0

        while iterations < MAX_TOOL_ITERATIONS:
            iterations += 1
            response = await self._call_claude(messages, use_fallback=(iterations > 1))

            stop_reason = response.stop_reason

            # --- Extract text and tool_use blocks ---
            text_blocks = []
            tool_use_blocks = []
            for block in response.content:
                if block.type == "text":
                    text_blocks.append(block.text)
                elif block.type == "tool_use":
                    tool_use_blocks.append(block)

            # --- No tool calls → final answer ---
            if stop_reason == "end_turn" and not tool_use_blocks:
                final_text = "\n".join(text_blocks).strip()
                return final_text, tool_records

            # --- Process tool calls ---
            if tool_use_blocks:
                # Append assistant message (may include partial text + tool_use blocks)
                messages.append({
                    "role": "assistant",
                    "content": response.content
                })

                # Dispatch all tool calls (in parallel)
                tool_results = await asyncio.gather(*[
                    self._dispatch_and_record(block, conversation.session_id, tool_records)
                    for block in tool_use_blocks
                ])

                # Append tool results back to messages
                messages.append({
                    "role": "user",
                    "content": tool_results
                })
                continue

            # --- Unexpected stop ---
            final_text = "\n".join(text_blocks).strip() or (
                "I'm sorry, I encountered an unexpected issue. "
                "Please try again or contact our support team directly."
            )
            return final_text, tool_records

        # Safety: exceeded max iterations
        logger.warning("Agent loop exceeded max iterations for session")
        return (
            "I've been working on your issue but need a moment more. "
            "Let me connect you with a human agent who can help right away.",
            tool_records,
        )

    async def _dispatch_and_record(
        self,
        tool_use_block,
        session_id: str,
        tool_records: List[ToolCallRecord],
    ) -> Dict[str, Any]:
        """Dispatch a single tool call, record it, and return the tool_result block."""
        result = await self.dispatcher.dispatch(
            tool_name=tool_use_block.name,
            tool_input=tool_use_block.input,
            session_id=session_id,
        )

        record = ToolCallRecord(
            tool_use_id=tool_use_block.id,
            tool_name=tool_use_block.name,
            inputs=tool_use_block.input,
            output=result,
            success=result.get("_success", True),
            latency_ms=result.get("_latency_ms"),
        )
        tool_records.append(record)

        return {
            "type": "tool_result",
            "tool_use_id": tool_use_block.id,
            "content": str(result),
        }

    # ------------------------------------------------------------------
    # Claude API call with fallback
    # ------------------------------------------------------------------

    async def _call_claude(
        self,
        messages: List[Dict],
        use_fallback: bool = False,
    ):
        """Call Claude API with primary model, falling back to Haiku on error."""
        model = FALLBACK_MODEL if use_fallback else PRIMARY_MODEL

        for attempt in range(3):
            try:
                response = await self.client.messages.create(
                    model=model,
                    max_tokens=MAX_TOKENS,
                    system=SUPPORT_AGENT_SYSTEM_PROMPT,
                    tools=TOOL_DEFINITIONS,
                    messages=messages,
                )
                return response

            except anthropic.RateLimitError:
                wait = 2 ** attempt
                logger.warning("Rate limit hit, retrying in %ss", wait)
                await asyncio.sleep(wait)

            except (anthropic.APITimeoutError, anthropic.APIConnectionError) as exc:
                if attempt == 2:
                    if model == PRIMARY_MODEL:
                        logger.warning("Primary model failed, switching to fallback")
                        model = FALLBACK_MODEL
                        attempt = -1     # Reset retry counter for fallback
                    else:
                        raise RuntimeError(
                            "Both primary and fallback models unavailable"
                        ) from exc
                wait = 2 ** attempt
                await asyncio.sleep(wait)

            except anthropic.APIError as exc:
                logger.error("Claude API error: %s", exc)
                raise

    # ------------------------------------------------------------------
    # Escalation handler
    # ------------------------------------------------------------------

    async def _handle_escalation(
        self,
        conversation: Conversation,
        decision,
        agent_reply: Optional[str] = None,
    ) -> str:
        """Update conversation state and build escalation response."""
        conversation.status = ConversationStatus.ESCALATED
        conversation.escalation_reason = decision.reason

        # Call the escalate_to_human tool to actually route to queue
        await self.dispatcher.dispatch(
            tool_name="escalate_to_human",
            tool_input={
                "reason": decision.reason,
                "urgency": decision.urgency,
                "summary": await self._generate_escalation_summary(conversation),
            },
            session_id=conversation.session_id,
        )

        prefix = (agent_reply + "\n\n") if agent_reply else ""
        return (
            f"{prefix}"
            "I want to make sure you get the best help possible. "
            "I'm connecting you with one of our human support specialists right now. "
            f"**Reason: {decision.reason}**\n\n"
            "They'll have the full context of our conversation and will be with you shortly. "
            "Your estimated wait time is 5–10 minutes. "
            "A ticket has been created to track your issue."
        )

    async def _generate_escalation_summary(self, conversation: Conversation) -> str:
        """Generate a concise handoff summary using Claude Haiku."""
        if not self.client:
            return f"Session {conversation.session_id} escalated after {conversation.turn_count} turns."
        try:
            history_text = "\n".join(
                f"{m.role.upper()}: {m.content}"
                for m in conversation.messages[-10:]
            )
            response = await self.client.messages.create(
                model=FALLBACK_MODEL,
                max_tokens=300,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"{ESCALATION_SUMMARY_PROMPT}\n\n"
                            f"--- CONVERSATION ---\n{history_text}"
                        )
                    }
                ]
            )
            return response.content[0].text.strip()
        except Exception as exc:
            logger.warning("Summary generation failed: %s", exc)
            return f"Session {conversation.session_id} — {conversation.turn_count} turns — {conversation.escalation_reason}"

    # ------------------------------------------------------------------
    # Context compression
    # ------------------------------------------------------------------

    async def _compress_history(self, conversation: Conversation) -> None:
        """Summarise old messages to save context window tokens."""
        try:
            old_messages = conversation.messages[:-10]
            history_text = "\n".join(
                f"{m.role.upper()}: {m.content}" for m in old_messages
            )
            response = await self.client.messages.create(
                model=FALLBACK_MODEL,
                max_tokens=400,
                messages=[
                    {
                        "role": "user",
                        "content": f"{CONTEXT_SUMMARISATION_PROMPT}\n\n{history_text}"
                    }
                ]
            )
            summary = response.content[0].text.strip()
            from app.models.conversation import Message
            summary_msg = Message(
                role=MessageRole.SYSTEM,
                content=f"[CONVERSATION SUMMARY]\n{summary}"
            )
            conversation.messages = [summary_msg] + conversation.messages[-10:]
            logger.info("Compressed history for session %s", conversation.session_id)
        except Exception as exc:
            logger.warning("History compression failed: %s", exc)

    @staticmethod
    def _record_timing(conversation: Conversation, start_time: float) -> None:
        elapsed_ms = round((time.monotonic() - start_time) * 1000, 2)
        conversation.metadata["last_response_time_ms"] = elapsed_ms
