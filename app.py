"""
HuggingFace Spaces — Gradio web interface for the Intelligent Customer Support Agent.
Entry point: HF Spaces looks for app.py at the root level.

Run locally:  python app.py
HF Spaces:    push to your Space repo — runs automatically
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import List, Tuple

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
# Global agent instance (initialised once at startup)
# ---------------------------------------------------------------------------

_agent: SupportAgentCore | None = None
_session_store: SessionStore | None = None


def _init_agent() -> tuple[SupportAgentCore, SessionStore]:
    """Initialise the agent and all services synchronously for Gradio."""
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY is not set. "
            "On HuggingFace Spaces: Settings → Variables and secrets → New secret."
        )

    client = anthropic.AsyncAnthropic(api_key=api_key)

    kb       = KnowledgeBaseService()
    customer = CustomerDBService()
    tickets  = TicketService()
    email    = EmailService()
    store    = SessionStore()   # In-memory (no Redis needed for demo)

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
    global _agent, _session_store
    if _agent is None:
        _agent, _session_store = _init_agent()
    return _agent, _session_store


# ---------------------------------------------------------------------------
# Demo presets
# ---------------------------------------------------------------------------

DEMO_PROMPTS = [
    "What is your refund policy?",
    "I can't log into my account — email is demo@example.com",
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
# Core chat function
# ---------------------------------------------------------------------------

def chat(
    user_message: str,
    history: list,
    session_state: dict,
    selected_customer: str,
) -> tuple:
    if not user_message.strip():
        return history, session_state, _status_html(session_state), _meta_html(session_state)

    agent, store = get_agent()

    # Load or create conversation
    session_id = session_state.get("session_id")
    conversation = None

    if session_id:
        loop = asyncio.new_event_loop()
        conversation = loop.run_until_complete(store.load(session_id))
        loop.close()

    if not conversation:
        email_val, cid = DEMO_CUSTOMERS.get(selected_customer, (None, None))
        conversation = Conversation(
            channel=ChannelType.CHAT,
            customer_email=email_val,
            customer_id=cid,
        )
        if "VIP" in selected_customer:
            conversation.is_vip = True

    # Run the agent
    start = time.monotonic()
    error_reply = None
    try:
        loop = asyncio.new_event_loop()
        reply, conversation = loop.run_until_complete(
            agent.process_message(conversation, user_message)
        )
        loop.run_until_complete(store.save(conversation))
        loop.close()
    except Exception as exc:
        error_reply = f"⚠️ Agent error: {exc}"
        try:
            loop.close()
        except Exception:
            pass

    if error_reply:
        history = history + [(user_message, error_reply)]
        return history, session_state, _status_html(session_state), _meta_html(session_state)

    elapsed = round((time.monotonic() - start) * 1000)

    # Collect last tool calls
    last_tools = []
    if conversation.messages:
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

    history = history + [(user_message, reply)]
    return history, session_state, _status_html(session_state), _meta_html(session_state)


def reset_chat(selected_customer: str):
    return [], {}, _status_html({}), _meta_html({})


def inject_prompt(prompt: str, history, session_state, selected_customer):
    """Insert demo prompt into chat and run it immediately."""
    return chat(prompt, history, session_state, selected_customer)


# ---------------------------------------------------------------------------
# HTML status panels
# ---------------------------------------------------------------------------

STATUS_META = {
    ConversationStatus.OPEN:      ("#22c55e", "🟢 Open"),
    ConversationStatus.ESCALATED: ("#ef4444", "🔴 Escalated to Human"),
    ConversationStatus.RESOLVED:  ("#3b82f6", "🔵 Resolved"),
    ConversationStatus.CLOSED:    ("#6b7280", "⚫ Closed"),
    ConversationStatus.PENDING:   ("#f59e0b", "🟡 Pending"),
}

_PANEL = "background:#1e1e2e;border-radius:8px;color:#cdd6f4;padding:12px;font-family:monospace;font-size:13px"


def _status_html(state: dict) -> str:
    status = state.get("status", None)
    color, label = STATUS_META.get(status, ("#6b7280", "⚪ No session yet"))
    turn = state.get("turn_count", 0)
    sid  = state.get("session_id", "—")
    sid_short = (sid[:8] + "…") if sid and sid != "—" else "—"
    return (
        f'<div style="{_PANEL}">'
        f'<div style="color:{color};font-weight:bold;margin-bottom:6px">{label}</div>'
        f'<div>Turns: <b>{turn}</b></div>'
        f'<div>Session: <b>{sid_short}</b></div>'
        f'</div>'
    )


def _meta_html(state: dict) -> str:
    ticket  = state.get("ticket_id") or "—"
    elapsed = state.get("elapsed_ms", "—")
    tools   = state.get("last_tools", [])
    tools_str = " · ".join(f"<code>{t}</code>" for t in tools) if tools else "<i>none</i>"
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
# Gradio UI Layout
# ---------------------------------------------------------------------------

CSS = """
#chatbot { height: 500px; }
.gr-button-sm { font-size: 11px !important; }
footer { display: none !important; }
"""

with gr.Blocks(
    title="Intelligent Customer Support Agent",
    theme=gr.themes.Soft(primary_hue="violet", neutral_hue="slate"),
    css=CSS,
) as demo:

    session_state = gr.State({})

    # ── Header ──────────────────────────────────────────────────────────
    gr.Markdown("""
    # 🤖 Intelligent Customer Support Agent
    **Powered by Claude claude-sonnet-4-6** &nbsp;|&nbsp; Multi-turn memory &nbsp;|&nbsp; Tool use &nbsp;|&nbsp; Auto-escalation &nbsp;|&nbsp; Ticket creation
    """)

    with gr.Row(equal_height=True):

        # ── Chat panel ──────────────────────────────────────────────────
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
                elem_id="chatbot",
                label="Chat with Aria (AI Support Agent)",
                bubble_full_width=False,
                show_copy_button=True,
                avatar_images=(
                    None,
                    "https://api.dicebear.com/7.x/bottts-neutral/svg?seed=Aria",
                ),
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

        # ── Right sidebar ────────────────────────────────────────────────
        with gr.Column(scale=2, min_width=240):

            gr.Markdown("#### 👤 Demo Customer Profile")
            customer_dd = gr.Dropdown(
                choices=list(DEMO_CUSTOMERS.keys()),
                value="None (anonymous)",
                label="",
                info="Agent will auto-look up this customer when you mention their email",
            )

            gr.Markdown("#### ⚡ Try These Prompts")
            prompt_btns = []
            for p in DEMO_PROMPTS:
                label = p[:44] + ("…" if len(p) > 44 else "")
                b = gr.Button(label, size="sm", variant="secondary")
                prompt_btns.append((b, p))

            gr.Markdown("#### 📊 Conversation Status")
            status_display = gr.HTML(value=_status_html({}))

            gr.Markdown("#### 🔧 Last Tool Calls")
            meta_display = gr.HTML(value=_meta_html({}))

            reset_btn = gr.Button("🔄 Start New Conversation", variant="secondary", size="sm")

    # ── How it works accordion ───────────────────────────────────────────
    with gr.Accordion("ℹ️ How the agent works (expand)", open=False):
        gr.Markdown("""
        ### Agent Loop (runs on every message)
        1. **Escalation pre-check** — scans for legal/fraud keywords, human requests, VIP flag
        2. **Claude claude-sonnet-4-6** processes message + full conversation history with tools available
        3. **Tool dispatch loop** — if Claude calls a tool, result is fed back and Claude continues
        4. **Final response** returned when Claude emits `end_turn`
        5. **Post-check** — sentiment scored by Claude Haiku; escalates if very negative
        6. **Session persisted** in memory store (Redis in production)

        ### Available Tools
        | Tool | What triggers it |
        |---|---|
        | `search_knowledge_base` | Policy / how-to / FAQ questions |
        | `lookup_customer` | Customer provides their email address |
        | `create_ticket` | Issue can't be resolved in conversation |
        | `check_order_status` | Customer provides an order ID |
        | `escalate_to_human` | Legal keywords · explicit request · VIP · 8+ turns |
        | `update_ticket` | Follow-up on existing ticket |
        | `send_email_reply` | Email channel conversations |

        ### Escalation Triggers (automatic)
        - Legal keywords: *lawsuit, fraud, attorney, chargeback*
        - Customer says: *human, manager, supervisor, escalate*
        - VIP/Enterprise customer + unresolved after 3 turns
        - Refund request over **$500**
        - Sentiment score below **-0.7** (Claude Haiku classifier)
        - More than **8 conversation turns** unresolved
        """)

    # ── Event wiring ────────────────────────────────────────────────────
    _chat_inputs  = [msg_input, chatbot, session_state, customer_dd]
    _chat_outputs = [chatbot, session_state, status_display, meta_display]

    send_btn.click(fn=chat, inputs=_chat_inputs, outputs=_chat_outputs).then(
        fn=lambda: gr.update(value=""), outputs=msg_input
    )
    msg_input.submit(fn=chat, inputs=_chat_inputs, outputs=_chat_outputs).then(
        fn=lambda: gr.update(value=""), outputs=msg_input
    )

    for btn, prompt_text in prompt_btns:
        btn.click(
            fn=inject_prompt,
            inputs=[gr.Textbox(value=prompt_text, visible=False), chatbot, session_state, customer_dd],
            outputs=_chat_outputs,
        )

    reset_btn.click(fn=reset_chat, inputs=[customer_dd], outputs=_chat_outputs)


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),
        show_error=True,
        share=False,
    )
