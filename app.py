"""
HuggingFace Spaces — Gradio 5.x web interface.
Powered by Google Gemini 2.0 Flash (free tier — 1,500 req/day).
Compatible with Python 3.13.

Run locally:  python app.py
HF Spaces:    push to your Space repo — runs automatically
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import os
import sys
import time
import types as _types
from typing import List

# ---------------------------------------------------------------------------
# Compatibility patches — MUST run before `import gradio`
# ---------------------------------------------------------------------------
def _patch_compat() -> None:
    # audioop removed in Python 3.13 — pydub needs it
    for mod_name in ("audioop", "pyaudioop"):
        if mod_name not in sys.modules:
            sys.modules[mod_name] = _types.ModuleType(mod_name)

    # HfFolder removed in huggingface_hub >= 0.25 — gradio.oauth needs it
    try:
        import huggingface_hub as _hfhub
        if not hasattr(_hfhub, "HfFolder"):
            class _HfFolderStub:
                @staticmethod
                def get_token():
                    return os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
                @staticmethod
                def save_token(token: str) -> None: pass
                @staticmethod
                def delete_token() -> None: pass
            _hfhub.HfFolder = _HfFolderStub
    except ImportError:
        pass

_patch_compat()

# NOTE: `from google import genai` is intentionally NOT imported here at module
# level.  Importing it here can trigger SDK initialisation side-effects before
# any user interaction.  It is imported lazily inside _build_agent() instead,
# so the Google GenAI SDK is loaded only when the first user message arrives.
import gradio as gr

from app.agent.core import SupportAgentCore
from app.agent.escalation import EscalationEngine
from app.agent.tools import ToolDispatcher
from app.models.conversation import ChannelType, Conversation, ConversationStatus
from app.services.customer_db import CustomerDBService
from app.services.email_service import EmailService
from app.services.knowledge_base import KnowledgeBaseService
from app.services.ticket_service import TicketService
from app.session_store import SessionStore

# ---------------------------------------------------------------------------
# Global agent — lazily initialised inside Gradio's own event loop
# ---------------------------------------------------------------------------

_agent: SupportAgentCore | None = None
_store: SessionStore | None = None
_init_lock: asyncio.Lock | None = None

# ---------------------------------------------------------------------------
# Response cache — avoids repeat Gemini calls for identical FAQ questions.
# Only caches anonymous-user, generic questions (no personal data).
# TTL: 1 hour.  Max entries: 200 (auto-evicted oldest first).
# ---------------------------------------------------------------------------

_RESPONSE_CACHE: dict[str, dict] = {}   # key → {"reply": str, "expires": float}
_CACHE_TTL = 3600              # seconds
_CACHE_MAX = 200


def _cache_key(message: str) -> str:
    return hashlib.md5(message.lower().strip().encode()).hexdigest()


def _get_cached(message: str) -> str | None:
    key = _cache_key(message)
    entry = _RESPONSE_CACHE.get(key)
    if entry and time.time() < entry["expires"]:
        return entry["reply"]
    if entry:
        _RESPONSE_CACHE.pop(key, None)   # expired
    return None


def _set_cached(message: str, reply: str) -> None:
    if len(_RESPONSE_CACHE) >= _CACHE_MAX:
        # evict the oldest entry
        oldest = min(_RESPONSE_CACHE, key=lambda k: _RESPONSE_CACHE[k]["expires"])
        _RESPONSE_CACHE.pop(oldest, None)
    _RESPONSE_CACHE[_cache_key(message)] = {"reply": reply, "expires": time.time() + _CACHE_TTL}


def _is_cacheable(message: str, selected_customer: str) -> bool:
    """Only cache anonymous, short, generic queries — never personalised ones."""
    if "None (anonymous)" not in selected_customer:
        return False
    msg = message.lower()
    # Skip if message contains personal data indicators
    if "@" in msg or any(x in msg for x in ("order", "my account", "my subscription", "i can't", "i cannot", "i'm unable")):
        return False
    return len(message) < 120


async def _build_agent() -> tuple[SupportAgentCore, SessionStore]:
    """Build and initialise all services using Gemini Flash (free tier).

    Imported lazily here — the Google GenAI SDK is NOT loaded at module level,
    so no SDK initialisation (or any network call) happens until a user sends
    their first message.
    """
    from google import genai  # lazy import — keeps startup free of API calls

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not set.\n"
            "• Get a free key at: https://aistudio.google.com/apikey\n"
            "• On HF Spaces: Settings → Variables and secrets → New secret → GEMINI_API_KEY"
        )

    print("⚙️  First user message received — initialising Gemini agent (no prior API calls were made).")
    client   = genai.Client(api_key=api_key)
    kb       = KnowledgeBaseService()
    customer = CustomerDBService()
    tickets  = TicketService()
    email    = EmailService()
    store    = SessionStore()

    await asyncio.gather(
        kb.initialize(),
        customer.initialize(),
        tickets.initialize(),
        store.initialize(),
    )

    dispatcher = ToolDispatcher(
        knowledge_base_service=kb,
        customer_service=customer,
        ticket_service=tickets,
        email_service=email,
    )
    escalation = EscalationEngine(gemini_client=client)
    agent = SupportAgentCore(
        gemini_client=client,
        tool_dispatcher=dispatcher,
        escalation_engine=escalation,
        session_store=store,
    )
    return agent, store


async def _ensure_agent() -> None:
    """Initialise the agent on first call, inside Gradio's running event loop."""
    global _agent, _store, _init_lock

    if _agent is not None:
        return

    if _init_lock is None:
        _init_lock = asyncio.Lock()

    async with _init_lock:
        if _agent is not None:
            return
        try:
            _agent, _store = await _build_agent()
            print("✅ Gemini agent initialised successfully")
        except Exception as exc:
            print(f"⚠️  Agent init failed: {exc}")
            raise


# ---------------------------------------------------------------------------
# Demo presets
# ---------------------------------------------------------------------------

DEMO_PROMPTS = [
    "What is your refund policy?",
    "I can't log in — email is demo@example.com",
    "Where is my order? Order ID is ORD-002",
    "How do I cancel my subscription?",
    "I want to speak to a human agent",
    "I'm going to file a lawsuit against you",
]

DEMO_CUSTOMERS = {
    "None (anonymous)": (None, None),
    "Demo User — Pro plan  (demo@example.com)": ("demo@example.com", None),
    "VIP Enterprise Customer  (vip@enterprise.com)": ("vip@enterprise.com", None),
}

# ---------------------------------------------------------------------------
# Chat handler
# ---------------------------------------------------------------------------

async def chat(
    user_message: str,
    history: List[dict],
    session_state: dict,
    selected_customer: str,
) -> tuple:
    """Main chat handler — always returns a response, never hangs."""
    print(f"[CHAT] >>> Received: {user_message[:80]}")

    if not user_message.strip():
        return history, session_state, _status_html(session_state), _meta_html(session_state)

    # ── Cache check ────────────────────────────────────────────────────────
    # For anonymous, generic FAQ questions return the cached reply immediately
    # without touching Gemini — preserves free-tier quota.
    if _is_cacheable(user_message, selected_customer):
        cached_reply = _get_cached(user_message)
        if cached_reply:
            print(f"[CHAT] Cache hit for: {user_message[:60]}")
            history = history + [
                {"role": "user",      "content": user_message},
                {"role": "assistant", "content": cached_reply},
            ]
            return history, session_state, _status_html(session_state), _meta_html(session_state)

    reply = "⚠️ Something went wrong — no response generated."  # safety default

    try:
        await _ensure_agent()
    except Exception as exc:
        print(f"[CHAT] Agent init failed: {exc}")
        reply = (
            f"⚠️ **Agent failed to initialise:** {exc}\n\n"
            "Make sure **GEMINI_API_KEY** is set in your Space secrets.\n"
            "Get a free key at: https://aistudio.google.com/apikey"
        )
        history = history + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": reply},
        ]
        return history, session_state, _status_html(session_state), _meta_html(session_state)

    # Load or create conversation
    session_id   = session_state.get("session_id")
    conversation = None

    if session_id and _store:
        conversation = await _store.load(session_id)

    if not conversation:
        email_val, cid = DEMO_CUSTOMERS.get(selected_customer, (None, None))
        conversation   = Conversation(
            channel=ChannelType.CHAT,
            customer_email=email_val,
            customer_id=cid,
        )
        if "VIP" in selected_customer:
            conversation.is_vip = True

    # Run agent
    start = time.monotonic()
    try:
        print(f"[CHAT] Calling process_message...")
        reply, conversation = await _agent.process_message(conversation, user_message)
        snippet = (reply or "")[:120]
        print(f"[CHAT] Got reply: {snippet if snippet else '(empty)'}")
        if _store:
            await _store.save(conversation)
        # Save to cache for identical future queries (anonymous + generic only)
        if reply and _is_cacheable(user_message, selected_customer):
            _set_cached(user_message, reply)
    except Exception as exc:
        print(f"[CHAT] Agent error: {exc}")
        reply = f"⚠️ Agent error: {exc}"

    # Ensure reply is never empty
    if not reply or not reply.strip():
        reply = "I received your message but wasn't able to generate a response. Please try again."

    elapsed = round((time.monotonic() - start) * 1000)

    last_tools = []
    for m in reversed(conversation.messages):
        if m.role == "assistant" and m.tool_calls:
            last_tools = [t.tool_name for t in m.tool_calls]
            break

    session_state.update({
        "session_id": conversation.session_id,
        "status":     conversation.status,
        "turn_count": conversation.turn_count,
        "ticket_id":  conversation.ticket_id,
        "elapsed_ms": elapsed,
        "last_tools": last_tools,
    })

    history = history + [
        {"role": "user",      "content": user_message},
        {"role": "assistant", "content": reply},
    ]
    print(f"[CHAT] <<< Returning {len(history)} messages, {elapsed}ms")
    return history, session_state, _status_html(session_state), _meta_html(session_state)


async def reset_chat(selected_customer: str):
    return [], {}, _status_html({}), _meta_html({})


async def inject_prompt(prompt: str, history, session_state, selected_customer):
    return await chat(prompt, history, session_state, selected_customer)


# ---------------------------------------------------------------------------
# Status / metadata panels
# ---------------------------------------------------------------------------

STATUS_META = {
    ConversationStatus.OPEN:      ("#16a34a", "🟢 Open"),
    ConversationStatus.ESCALATED: ("#dc2626", "🔴 Escalated to Human"),
    ConversationStatus.RESOLVED:  ("#2563eb", "🔵 Resolved"),
    ConversationStatus.CLOSED:    ("#6b7280", "⚫ Closed"),
    ConversationStatus.PENDING:   ("#d97706", "🟡 Pending"),
}

_PANEL = (
    "border:1px solid #e5e7eb;border-radius:8px;"
    "padding:12px;font-family:monospace;font-size:13px;line-height:1.6"
)


def _status_html(state: dict) -> str:
    status       = state.get("status")
    color, label = STATUS_META.get(status, ("#6b7280", "⚪ No session yet"))
    turn         = state.get("turn_count", 0)
    sid          = state.get("session_id", "—")
    sid_short    = (sid[:8] + "…") if sid and sid != "—" else "—"
    return (
        f'<div style="{_PANEL}">'
        f'<div style="color:{color};font-weight:bold;margin-bottom:4px">{label}</div>'
        f'<div>Turns: <b>{turn}</b></div>'
        f'<div>Session: <b>{sid_short}</b></div>'
        f'</div>'
    )


def _meta_html(state: dict) -> str:
    ticket      = state.get("ticket_id") or "—"
    elapsed     = state.get("elapsed_ms", "—")
    tools       = state.get("last_tools", [])
    tools_str   = " · ".join(f"<code>{t}</code>" for t in tools) if tools else "<i>none</i>"
    elapsed_str = f"{elapsed} ms" if isinstance(elapsed, int) else "—"
    return (
        f'<div style="{_PANEL}">'
        f'<div>Response: <b>{elapsed_str}</b></div>'
        f'<div style="margin-top:4px">Ticket: <b>{ticket}</b></div>'
        f'<div style="margin-top:6px">Tools: {tools_str}</div>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Gradio 5 UI
# ---------------------------------------------------------------------------

CSS = """
#chatbot { height: 500px; }
footer { display: none !important; }
.api-docs { display: none !important; }
.built-with { display: none !important; }
"""

with gr.Blocks(
    title="Intelligent Customer Support Agent",
    theme=gr.themes.Soft(primary_hue="violet", neutral_hue="slate"),
    css=CSS,
) as demo:

    session_state = gr.State({})

    gr.Markdown("""
    # 🤖 Intelligent Customer Support Agent
    **Powered by Gemini 2.0 Flash (free tier)** &nbsp;|&nbsp; Multi-turn memory &nbsp;|&nbsp;
    Tool use &nbsp;|&nbsp; Auto-escalation &nbsp;|&nbsp; Ticket creation
    """)

    with gr.Row(equal_height=True):

        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
                elem_id="chatbot",
                label="Chat with Aria (AI Support Agent)",
                type="messages",
                show_copy_button=True,
            )
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Ask anything — refund, order status, account help...",
                    show_label=False,
                    scale=6,
                    container=False,
                    autofocus=True,
                )
                send_btn = gr.Button("Send ➤", variant="primary", scale=1, min_width=80)

        with gr.Column(scale=2, min_width=240):

            gr.Markdown("#### 👤 Demo Customer")
            customer_dd = gr.Dropdown(
                choices=list(DEMO_CUSTOMERS.keys()),
                value="None (anonymous)",
                label="",
                info="Agent looks up this customer when you give their email",
            )

            gr.Markdown("#### ⚡ Quick Prompts")
            prompt_btns = []
            for p in DEMO_PROMPTS:
                label = p[:44] + ("…" if len(p) > 44 else "")
                b = gr.Button(label, size="sm", variant="secondary")
                prompt_btns.append((b, p))

            gr.Markdown("#### 📊 Status")
            status_display = gr.HTML(value=_status_html({}))

            gr.Markdown("#### 🔧 Last Tool Calls")
            meta_display = gr.HTML(value=_meta_html({}))

            reset_btn = gr.Button("🔄 New Conversation", variant="secondary", size="sm")

    with gr.Accordion("ℹ️ How the agent works", open=False):
        gr.Markdown("""
        **On every message:**
        1. Escalation pre-check — legal keywords, human request, VIP flag
        2. Gemini 2.0 Flash processes message + full history with tools
        3. Tool loop — Gemini calls tools, reads results, continues until done
        4. Post-check — sentiment scored by Gemini Flash

        **Tools:** `search_knowledge_base` · `lookup_customer` · `create_ticket` ·
        `check_order_status` · `escalate_to_human` · `send_email_reply`

        **Auto-escalation:** legal words · "speak to human/manager" ·
        VIP customer · refund > $500 · sentiment < -0.7 · 8+ turns unresolved

        **Cost:** 100% free — uses Gemini 2.0 Flash free tier (1,500 requests/day).
        Get your free API key at [aistudio.google.com](https://aistudio.google.com/apikey).
        """)

    # ── Event wiring ────────────────────────────────────────────────────
    _inputs  = [msg_input, chatbot, session_state, customer_dd]
    _outputs = [chatbot, session_state, status_display, meta_display]

    # api_name=False on every handler → excluded from schema builder
    # (prevents gradio_client TypeError on bool JSON Schema values)
    send_btn.click(fn=chat, inputs=_inputs, outputs=_outputs,
                   api_name=False).then(
        fn=lambda: gr.update(value=""), outputs=msg_input
    )
    msg_input.submit(fn=chat, inputs=_inputs, outputs=_outputs,
                     api_name=False).then(
        fn=lambda: gr.update(value=""), outputs=msg_input
    )
    reset_btn.click(fn=reset_chat, inputs=[customer_dd], outputs=_outputs,
                    api_name=False)

    for btn, prompt_text in prompt_btns:
        btn.click(
            fn=functools.partial(inject_prompt, prompt_text),
            inputs=[chatbot, session_state, customer_dd],
            outputs=_outputs,
            api_name=False,
        )

    # NOTE: Removed demo.load() warm-up to prevent initialization on page load.
    # Agent initializes on first user message instead (lazy loading).
    # This prevents unnecessary API calls and respects rate limits.


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),
        show_error=True,
        show_api=False,   # disables gradio_client schema builder → no TypeError
    )
