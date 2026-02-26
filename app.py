"""
HuggingFace Spaces — Gradio 5.x web interface.
Compatible with Python 3.13 (no audioop/pydub dependency).

Run locally:  python app.py
HF Spaces:    push to your Space repo — runs automatically
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
import types
from typing import List

# ---------------------------------------------------------------------------
# Python 3.13 fix: audioop was removed; pydub (used by gradio) needs it.
# We don't use any audio features, so a no-op mock is safe.
# This MUST run before `import gradio`.
# ---------------------------------------------------------------------------
def _patch_audioop() -> None:
    for mod_name in ("audioop", "pyaudioop"):
        if mod_name not in sys.modules:
            sys.modules[mod_name] = types.ModuleType(mod_name)

_patch_audioop()

import anthropic
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
# Global agent (initialised once)
# ---------------------------------------------------------------------------

_agent: SupportAgentCore | None = None
_store: SessionStore | None = None


def _init_agent() -> tuple[SupportAgentCore, SessionStore]:
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY not set. "
            "On HF Spaces: Settings → Variables and secrets → New secret."
        )

    client     = anthropic.AsyncAnthropic(api_key=api_key)
    kb         = KnowledgeBaseService()
    customer   = CustomerDBService()
    tickets    = TicketService()
    email      = EmailService()
    store      = SessionStore()

    dispatcher = ToolDispatcher(
        knowledge_base_service=kb,
        customer_service=customer,
        ticket_service=tickets,
        email_service=email,
    )
    escalation = EscalationEngine(anthropic_client=client)
    agent = SupportAgentCore(
        anthropic_client=client,
        tool_dispatcher=dispatcher,
        escalation_engine=escalation,
        session_store=store,
    )

    loop = asyncio.new_event_loop()
    loop.run_until_complete(asyncio.gather(
        kb.initialize(),
        customer.initialize(),
        tickets.initialize(),
        store.initialize(),
    ))
    loop.close()
    return agent, store


def get_agent() -> tuple[SupportAgentCore, SessionStore]:
    global _agent, _store
    if _agent is None:
        _agent, _store = _init_agent()
    return _agent, _store


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
# Chat function  —  Gradio 5 uses List[dict] for history
# ---------------------------------------------------------------------------

def chat(
    user_message: str,
    history: List[dict],          # [{"role": "user"/"assistant", "content": "..."}]
    session_state: dict,
    selected_customer: str,
) -> tuple:
    if not user_message.strip():
        return history, session_state, _status_html(session_state), _meta_html(session_state)

    agent, store = get_agent()

    # Load or create conversation
    session_id   = session_state.get("session_id")
    conversation = None

    if session_id:
        loop         = asyncio.new_event_loop()
        conversation = loop.run_until_complete(store.load(session_id))
        loop.close()

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
        loop         = asyncio.new_event_loop()
        reply, conversation = loop.run_until_complete(
            agent.process_message(conversation, user_message)
        )
        loop.run_until_complete(store.save(conversation))
        loop.close()
    except Exception as exc:
        reply = f"⚠️ Agent error: {exc}"
        try:
            loop.close()
        except Exception:
            pass

    elapsed = round((time.monotonic() - start) * 1000)

    # Collect tool calls from last assistant message
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

    # Gradio 5: history is list of dicts
    history = history + [
        {"role": "user",      "content": user_message},
        {"role": "assistant", "content": reply},
    ]
    return history, session_state, _status_html(session_state), _meta_html(session_state)


def reset_chat(selected_customer: str):
    return [], {}, _status_html({}), _meta_html({})


def inject_prompt(prompt: str, history, session_state, selected_customer):
    return chat(prompt, history, session_state, selected_customer)


# ---------------------------------------------------------------------------
# Status / metadata panels
# ---------------------------------------------------------------------------

STATUS_META = {
    ConversationStatus.OPEN:      ("#22c55e", "🟢 Open"),
    ConversationStatus.ESCALATED: ("#ef4444", "🔴 Escalated to Human"),
    ConversationStatus.RESOLVED:  ("#3b82f6", "🔵 Resolved"),
    ConversationStatus.CLOSED:    ("#6b7280", "⚫ Closed"),
    ConversationStatus.PENDING:   ("#f59e0b", "🟡 Pending"),
}

_PANEL = (
    "background:#1e1e2e;border-radius:8px;color:#cdd6f4;"
    "padding:12px;font-family:monospace;font-size:13px"
)


def _status_html(state: dict) -> str:
    status      = state.get("status")
    color, label = STATUS_META.get(status, ("#6b7280", "⚪ No session yet"))
    turn        = state.get("turn_count", 0)
    sid         = state.get("session_id", "—")
    sid_short   = (sid[:8] + "…") if sid and sid != "—" else "—"
    return (
        f'<div style="{_PANEL}">'
        f'<div style="color:{color};font-weight:bold;margin-bottom:6px">{label}</div>'
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
        f'<div>Response time: <b>{elapsed_str}</b></div>'
        f'<div style="margin-top:4px">Ticket: <b>{ticket}</b></div>'
        f'<div style="margin-top:6px">Tools used:</div>'
        f'<div style="margin-top:2px">{tools_str}</div>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Gradio 5 UI
# ---------------------------------------------------------------------------

CSS = """
#chatbot { height: 500px; }
footer { display: none !important; }
"""

with gr.Blocks(
    title="Intelligent Customer Support Agent",
    theme=gr.themes.Soft(primary_hue="violet", neutral_hue="slate"),
    css=CSS,
) as demo:

    session_state = gr.State({})

    gr.Markdown("""
    # 🤖 Intelligent Customer Support Agent
    **Powered by Claude claude-sonnet-4-6** &nbsp;|&nbsp; Multi-turn memory &nbsp;|&nbsp;
    Tool use &nbsp;|&nbsp; Auto-escalation &nbsp;|&nbsp; Ticket creation
    """)

    with gr.Row(equal_height=True):

        # ── Chat panel ──────────────────────────────────────────────────
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
                elem_id="chatbot",
                label="Chat with Aria (AI Support Agent)",
                type="messages",          # Gradio 5 format
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

        # ── Sidebar ──────────────────────────────────────────────────────
        with gr.Column(scale=2, min_width=240):

            gr.Markdown("#### 👤 Demo Customer Profile")
            customer_dd = gr.Dropdown(
                choices=list(DEMO_CUSTOMERS.keys()),
                value="None (anonymous)",
                label="",
                info="Agent will auto-look up this customer when you give their email",
            )

            gr.Markdown("#### ⚡ Try These Prompts")
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

    # ── How it works ────────────────────────────────────────────────────
    with gr.Accordion("ℹ️ How the agent works", open=False):
        gr.Markdown("""
        **On every message:**
        1. Escalation pre-check — legal keywords, human request, VIP flag
        2. Claude claude-sonnet-4-6 processes message + full history with tools
        3. Tool loop — Claude calls tools, reads results, continues until done
        4. Post-check — sentiment scored by Claude Haiku

        **Tools available:** `search_knowledge_base` · `lookup_customer` ·
        `create_ticket` · `check_order_status` · `escalate_to_human` · `send_email_reply`

        **Auto-escalation triggers:** legal words · "speak to human/manager" ·
        VIP customer · refund > $500 · sentiment < -0.7 · 8+ turns unresolved
        """)

    # ── Event wiring ────────────────────────────────────────────────────
    _inputs  = [msg_input, chatbot, session_state, customer_dd]
    _outputs = [chatbot, session_state, status_display, meta_display]

    send_btn.click(fn=chat, inputs=_inputs, outputs=_outputs).then(
        fn=lambda: gr.update(value=""), outputs=msg_input
    )
    msg_input.submit(fn=chat, inputs=_inputs, outputs=_outputs).then(
        fn=lambda: gr.update(value=""), outputs=msg_input
    )
    reset_btn.click(fn=reset_chat, inputs=[customer_dd], outputs=_outputs)

    for btn, prompt_text in prompt_btns:
        btn.click(
            fn=inject_prompt,
            inputs=[
                gr.Textbox(value=prompt_text, visible=False),
                chatbot, session_state, customer_dd
            ],
            outputs=_outputs,
        )

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),
        show_error=True,
    )
