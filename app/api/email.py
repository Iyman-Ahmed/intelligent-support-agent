"""
Email API endpoints — inbound webhook parsing + agent processing.
Supports SendGrid Inbound Parse and Mailgun webhooks.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import time
from typing import Any, Dict

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, Request

from app.models.conversation import (
    ChannelType,
    ChatRequest,
    Conversation,
    ConversationStatus,
)
from app.services.email_service import InboundEmail

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/email", tags=["email"])


def get_agent(request: Request):
    return request.app.state.agent

def get_session_store(request: Request):
    return request.app.state.session_store

def get_email_service(request: Request):
    return request.app.state.email_service

def get_metrics(request: Request):
    return request.app.state.metrics


# ---------------------------------------------------------------------------
# SendGrid Inbound Parse webhook
# ---------------------------------------------------------------------------

@router.post("/inbound/sendgrid")
async def sendgrid_inbound(
    request: Request,
    background_tasks: BackgroundTasks,
    agent=Depends(get_agent),
    session_store=Depends(get_session_store),
    email_service=Depends(get_email_service),
):
    """
    Receives inbound emails via SendGrid Inbound Parse.
    Processes them in background to return 200 quickly.
    """
    form_data = await request.form()
    payload = dict(form_data)

    inbound = InboundEmail.from_sendgrid_webhook(payload)
    logger.info("Inbound email from %s: %s", inbound.sender, inbound.subject)

    background_tasks.add_task(
        _process_inbound_email,
        inbound, agent, session_store, email_service
    )
    return {"status": "accepted"}


@router.post("/inbound/mailgun")
async def mailgun_inbound(
    request: Request,
    background_tasks: BackgroundTasks,
    agent=Depends(get_agent),
    session_store=Depends(get_session_store),
    email_service=Depends(get_email_service),
):
    """
    Receives inbound emails via Mailgun webhooks.
    """
    form_data = await request.form()
    payload = dict(form_data)

    inbound = InboundEmail.from_mailgun_webhook(payload)
    logger.info("Inbound email (Mailgun) from %s: %s", inbound.sender, inbound.subject)

    background_tasks.add_task(
        _process_inbound_email,
        inbound, agent, session_store, email_service
    )
    return {"status": "accepted"}


# ---------------------------------------------------------------------------
# Manual email processing (for testing)
# ---------------------------------------------------------------------------

@router.post("/process")
async def process_email_manually(
    payload: Dict[str, Any],
    request: Request,
    agent=Depends(get_agent),
    session_store=Depends(get_session_store),
    email_service=Depends(get_email_service),
):
    """Process an email message manually (useful for testing)."""
    inbound = InboundEmail(
        message_id=payload.get("message_id", "manual"),
        sender=payload["sender"],
        sender_name=payload.get("sender_name", ""),
        recipient=payload.get("recipient", "support@yourdomain.com"),
        subject=payload.get("subject", ""),
        body_plain=payload["body"],
    )
    reply = await _process_inbound_email(inbound, agent, session_store, email_service)
    return {"status": "processed", "reply": reply}


# ---------------------------------------------------------------------------
# Background task: process one email through the agent
# ---------------------------------------------------------------------------

async def _process_inbound_email(
    inbound: InboundEmail,
    agent,
    session_store,
    email_service,
) -> str:
    start = time.monotonic()

    # Use message_id to create a deterministic session_id for email threads
    session_id = f"email_{hashlib.md5(inbound.sender.encode()).hexdigest()[:12]}"

    # Load or create conversation for this email thread
    conversation = await session_store.load(session_id)
    if not conversation:
        conversation = Conversation(
            session_id=session_id,
            channel=ChannelType.EMAIL,
            customer_email=inbound.sender,
        )

    # Use subject + plain text as the user message
    user_message = (
        f"Subject: {inbound.subject}\n\n{inbound.body_plain.strip()}"
        if inbound.subject
        else inbound.body_plain.strip()
    )

    try:
        reply, conversation = await agent.process_message(conversation, user_message)
        await session_store.save(conversation)

        # Send reply email
        if email_service:
            reply_subject = (
                inbound.subject
                if inbound.subject.startswith("Re:")
                else f"Re: {inbound.subject}"
            )
            if conversation.ticket_id:
                reply_subject = f"[{conversation.ticket_id}] {reply_subject}"

            await email_service.send(
                to=inbound.sender,
                subject=reply_subject,
                body=reply,
                reply_to=inbound.recipient,
            )

        elapsed = round((time.monotonic() - start) * 1000, 2)
        logger.info(
            "Email processed: session=%s elapsed=%sms escalated=%s",
            session_id, elapsed,
            conversation.status == ConversationStatus.ESCALATED
        )
        return reply

    except Exception as exc:
        logger.exception("Failed to process inbound email from %s: %s", inbound.sender, exc)

        # Fallback: send a generic acknowledgment
        if email_service:
            await email_service.send(
                to=inbound.sender,
                subject=f"Re: {inbound.subject} — We've received your message",
                body=(
                    f"Hi,\n\n"
                    f"Thank you for contacting us. We've received your message and "
                    f"a support team member will get back to you within 24 hours.\n\n"
                    f"Best regards,\nSupport Team"
                )
            )
        return "fallback_sent"


# ---------------------------------------------------------------------------
# Ticket status update webhook (from Zendesk / Freshdesk)
# ---------------------------------------------------------------------------

@router.post("/ticket/webhook")
async def ticket_webhook(
    payload: Dict[str, Any],
    request: Request,
    x_webhook_secret: str = Header(None),
):
    """
    Receives ticket status update webhooks from external ticketing systems.
    Can be used to notify customers when their ticket is updated.
    """
    # Verify webhook signature (production: implement proper HMAC verification)
    secret = request.app.state.config.webhook_secret
    if secret and x_webhook_secret != secret:
        raise HTTPException(status_code=401, detail="Invalid webhook signature")

    ticket_id = payload.get("ticket_id") or payload.get("id")
    new_status = payload.get("status")

    logger.info("Ticket webhook: ticket=%s status=%s", ticket_id, new_status)

    ticket_service = request.app.state.ticket_service
    if ticket_service and ticket_id:
        await ticket_service.update(ticket_id=ticket_id, status=new_status)

    return {"status": "ok", "ticket_id": ticket_id}
