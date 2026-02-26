"""
AI Agent Core — the agentic loop that powers multi-turn conversations.
Powered by Google Gemini 2.0 Flash (free tier).

Flow:
  1. Load conversation history from session store
  2. Append new user message
  3. Call Gemini with function_calling enabled
  4. If Gemini returns function_call parts → dispatch tools → inject results → loop
  5. When Gemini returns text only → run escalation check → return
  6. Persist updated conversation
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from google import genai
from google.genai import types

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

GEMINI_MODEL         = "gemini-2.0-flash"       # Free tier: 1,500 req/day
GEMINI_MODEL_LITE    = "gemini-2.0-flash-lite"  # Fallback 1: higher free quota
GEMINI_MODEL_LEGACY  = "gemini-1.5-flash"       # Fallback 2: separate quota pool
MAX_TOKENS      = 1024
MAX_TOOL_ITERATIONS  = 10
CONTEXT_WINDOW_TURNS = 20
SUMMARISE_AFTER_TURNS = 30


# ---------------------------------------------------------------------------
# Convert Anthropic-style tool schemas → Gemini FunctionDeclarations
# ---------------------------------------------------------------------------

def _build_gemini_tools() -> list:
    """Convert the shared TOOL_DEFINITIONS to google-genai Tool objects."""

    _TYPE_MAP = {
        "string":  types.Type.STRING,
        "integer": types.Type.INTEGER,
        "number":  types.Type.NUMBER,
        "boolean": types.Type.BOOLEAN,
        "array":   types.Type.ARRAY,
        "object":  types.Type.OBJECT,
    }

    def _convert_schema(schema: Dict[str, Any]) -> types.Schema:
        t = _TYPE_MAP.get(schema.get("type", "string"), types.Type.STRING)
        kwargs: Dict[str, Any] = {
            "type": t,
            "description": schema.get("description", ""),
        }
        if "enum" in schema:
            kwargs["enum"] = schema["enum"]
        if "properties" in schema:
            kwargs["properties"] = {
                k: _convert_schema(v) for k, v in schema["properties"].items()
            }
        if "required" in schema:
            kwargs["required"] = schema["required"]
        return types.Schema(**kwargs)

    declarations = []
    for tool in TOOL_DEFINITIONS:
        declarations.append(
            types.FunctionDeclaration(
                name=tool["name"],
                description=tool["description"],
                parameters=_convert_schema(tool["input_schema"]),
            )
        )
    return [types.Tool(function_declarations=declarations)]


GEMINI_TOOLS = _build_gemini_tools()


# ---------------------------------------------------------------------------
# Agent Core
# ---------------------------------------------------------------------------

class SupportAgentCore:
    """
    Orchestrates the full agentic loop:
    message → Gemini → function calls → Gemini → … → final response
    """

    def __init__(
        self,
        gemini_client: genai.Client,
        tool_dispatcher: ToolDispatcher,
        escalation_engine: EscalationEngine,
        session_store=None,
    ):
        self.client     = gemini_client
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
        start_time = time.monotonic()

        # 1. Pre-escalation check
        escalation_decision = await self.escalation.evaluate(
            conversation, user_message
        )

        # 2. Append user message
        conversation.add_message(MessageRole.USER, user_message)

        # 3. Immediate rule-based escalation
        if escalation_decision.should_escalate and escalation_decision.trigger_type == "rule":
            reply = await self._handle_escalation(conversation, escalation_decision)
            conversation.add_message(MessageRole.ASSISTANT, reply)
            self._record_timing(conversation, start_time)
            return reply, conversation

        # 4. Run agentic loop
        reply, tool_records = await self._agent_loop(conversation)

        # 5. Post-response escalation check (sentiment / turn limit)
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

    # ------------------------------------------------------------------
    # Agentic loop
    # ------------------------------------------------------------------

    async def _agent_loop(
        self,
        conversation: Conversation,
    ) -> Tuple[str, List[ToolCallRecord]]:
        contents     = self._build_contents(conversation)
        tool_records: List[ToolCallRecord] = []

        for _ in range(MAX_TOOL_ITERATIONS):
            response  = await self._call_gemini(contents)
            candidate = response.candidates[0]
            parts     = candidate.content.parts if candidate.content else []

            text_parts = [p.text for p in parts if getattr(p, "text", None)]
            fn_parts   = [
                p for p in parts
                if getattr(p, "function_call", None) is not None
            ]

            # No function calls → final answer
            if not fn_parts:
                return (
                    " ".join(text_parts).strip()
                    or "I'm sorry, I had trouble processing your request. Please try again.",
                    tool_records,
                )

            # Append model's response (including function_call parts) to history
            contents.append(candidate.content)

            # Dispatch all tool calls in parallel
            async def _run(part):
                fc     = part.function_call
                result = await self.dispatcher.dispatch(
                    tool_name=fc.name,
                    tool_input=dict(fc.args or {}),
                    session_id=conversation.session_id,
                )
                record = ToolCallRecord(
                    tool_use_id=str(uuid.uuid4()),
                    tool_name=fc.name,
                    inputs=dict(fc.args or {}),
                    output=result,
                    success=result.get("_success", True),
                    latency_ms=result.get("_latency_ms"),
                )
                tool_records.append(record)
                return types.Part(
                    function_response=types.FunctionResponse(
                        name=fc.name,
                        response=result,
                    )
                )

            fn_response_parts = await asyncio.gather(*[_run(p) for p in fn_parts])

            # Inject tool results back as a "user" turn
            contents.append(
                types.Content(role="user", parts=list(fn_response_parts))
            )

        # Exceeded max iterations
        logger.warning("Agent loop exceeded max iterations for session %s", conversation.session_id)
        return (
            "I've been working on your issue but it's taking longer than expected. "
            "Let me connect you with a human agent who can help right away.",
            tool_records,
        )

    # ------------------------------------------------------------------
    # Gemini API call
    # ------------------------------------------------------------------

    async def _call_gemini(self, contents: list):
        import re as _re

        # Try models in order — each has its own separate quota pool
        models_to_try = [GEMINI_MODEL, GEMINI_MODEL_LITE, GEMINI_MODEL_LEGACY]

        for model in models_to_try:
            for attempt in range(3):
                try:
                    response = await self.client.aio.models.generate_content(
                        model=model,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            system_instruction=SUPPORT_AGENT_SYSTEM_PROMPT,
                            tools=GEMINI_TOOLS,
                            max_output_tokens=MAX_TOKENS,
                            temperature=0.3,
                        ),
                    )
                    return response

                except Exception as exc:
                    err_str = str(exc)

                    # 401 / invalid key — fail immediately, no point retrying
                    if "400" in err_str and "API_KEY_INVALID" in err_str:
                        raise RuntimeError(
                            "❌ Invalid Gemini API key.\n"
                            "Go to https://aistudio.google.com/apikey, create a new key, "
                            "and update GEMINI_API_KEY in your HF Space secrets."
                        ) from exc

                    # 429 rate limit — respect retry-after, then try next model
                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                        m = _re.search(r"retry in ([\d.]+)s", err_str)
                        wait = float(m.group(1)) + 0.5 if m else (2 ** (attempt + 1))
                        logger.warning(
                            "Rate limited on %s (attempt %d), waiting %.1fs",
                            model, attempt + 1, wait,
                        )
                        await asyncio.sleep(min(wait, 15))  # cap at 15s
                        if attempt == 2:
                            # Exhausted retries on this model → try next model
                            logger.warning("Switching from %s to fallback model", model)
                            break
                        continue

                    # Other errors — retry with backoff
                    if attempt == 2:
                        logger.error("Gemini error on %s: %s", model, exc)
                        break  # try next model
                    await asyncio.sleep(2 ** attempt)

        raise RuntimeError(
            "⚠️ Gemini API is temporarily unavailable or rate limited. "
            "Please wait a moment and try again."
        )

    # ------------------------------------------------------------------
    # History → Gemini contents
    # ------------------------------------------------------------------

    def _build_contents(self, conversation: Conversation) -> list:
        """Convert the conversation's message history to Gemini Content objects."""
        history = [
            m for m in conversation.messages
            if m.role in (MessageRole.USER, MessageRole.ASSISTANT)
        ]
        history = history[-CONTEXT_WINDOW_TURNS:]

        contents = []
        for msg in history:
            role = "model" if msg.role == MessageRole.ASSISTANT else "user"
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part(text=msg.content or "…")],
                )
            )
        return contents

    # ------------------------------------------------------------------
    # Escalation handler
    # ------------------------------------------------------------------

    async def _handle_escalation(
        self,
        conversation: Conversation,
        decision,
        agent_reply: Optional[str] = None,
    ) -> str:
        conversation.status           = ConversationStatus.ESCALATED
        conversation.escalation_reason = decision.reason

        await self.dispatcher.dispatch(
            tool_name="escalate_to_human",
            tool_input={
                "reason":  decision.reason,
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
        """Generate a concise handoff summary using Gemini Flash."""
        try:
            history_text = "\n".join(
                f"{m.role.upper()}: {m.content}"
                for m in conversation.messages[-10:]
            )
            response = await self.client.aio.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    types.Content(
                        role="user",
                        parts=[types.Part(text=f"{ESCALATION_SUMMARY_PROMPT}\n\n--- CONVERSATION ---\n{history_text}")],
                    )
                ],
                config=types.GenerateContentConfig(max_output_tokens=300),
            )
            return response.text.strip()
        except Exception as exc:
            logger.warning("Summary generation failed: %s", exc)
            return (
                f"Session {conversation.session_id} — "
                f"{conversation.turn_count} turns — {conversation.escalation_reason}"
            )

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
            response = await self.client.aio.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    types.Content(
                        role="user",
                        parts=[types.Part(text=f"{CONTEXT_SUMMARISATION_PROMPT}\n\n{history_text}")],
                    )
                ],
                config=types.GenerateContentConfig(max_output_tokens=400),
            )
            summary = response.text.strip()
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
