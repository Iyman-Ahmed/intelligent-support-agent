"""
Chat API endpoints — WebSocket real-time chat + REST fallback.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect, status
from fastapi.responses import StreamingResponse

from app.models.conversation import (
    ChatRequest,
    ChatResponse,
    ChannelType,
    Conversation,
    ConversationStatus,
    ConversationSummary,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


# ---------------------------------------------------------------------------
# Dependency: get agent from app state
# ---------------------------------------------------------------------------

def get_agent(request: Request):
    return request.app.state.agent


def get_session_store(request: Request):
    return request.app.state.session_store


def get_metrics(request: Request):
    return request.app.state.metrics


# ---------------------------------------------------------------------------
# REST endpoint — single message / response
# ---------------------------------------------------------------------------

@router.post("/message", response_model=ChatResponse)
async def send_message(
    payload: ChatRequest,
    request: Request,
    agent=Depends(get_agent),
    session_store=Depends(get_session_store),
    metrics=Depends(get_metrics),
):
    """
    Send a customer message and receive the agent's reply.
    Creates a new session if session_id is None.
    """
    start = time.monotonic()

    # Load or create conversation
    conversation = await _get_or_create_conversation(
        payload, session_store
    )

    try:
        reply, conversation = await agent.process_message(
            conversation, payload.message
        )
    except RuntimeError as exc:
        # Both primary and fallback models failed
        logger.exception("Agent failed for session %s", conversation.session_id)
        reply = (
            "We're experiencing technical difficulties right now. "
            "Your issue has been logged and a team member will contact you shortly."
        )
        conversation.status = ConversationStatus.ESCALATED

    # Persist updated conversation
    await session_store.save(conversation)

    # Record metrics
    elapsed_ms = round((time.monotonic() - start) * 1000, 2)
    if metrics:
        metrics.record_response_time(elapsed_ms)
        metrics.record_conversation_status(conversation.status)

    return ChatResponse(
        session_id=conversation.session_id,
        message=reply,
        status=conversation.status,
        ticket_id=conversation.ticket_id,
        escalated=(conversation.status == ConversationStatus.ESCALATED),
        response_time_ms=elapsed_ms,
    )


# ---------------------------------------------------------------------------
# Streaming endpoint (SSE)
# ---------------------------------------------------------------------------

@router.post("/stream")
async def stream_message(
    payload: ChatRequest,
    request: Request,
    agent=Depends(get_agent),
    session_store=Depends(get_session_store),
):
    """
    Streaming variant — returns text/event-stream with word-by-word chunks.
    """
    conversation = await _get_or_create_conversation(payload, session_store)

    async def event_generator():
        try:
            async for chunk in agent.stream_message(conversation, payload.message):
                yield f"data: {json.dumps({'text': chunk})}\n\n"
            await session_store.save(conversation)
            yield f"data: {json.dumps({'done': True, 'session_id': conversation.session_id})}\n\n"
        except Exception as exc:
            logger.exception("Streaming error: %s", exc)
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


# ---------------------------------------------------------------------------
# WebSocket endpoint — real-time bidirectional chat
# ---------------------------------------------------------------------------

@router.websocket("/ws/{session_id}")
async def websocket_chat(
    websocket: WebSocket,
    session_id: str,
    request: Request,
):
    agent = request.app.state.agent
    session_store = request.app.state.session_store

    await websocket.accept()
    logger.info("WebSocket connected: session=%s", session_id)

    # Load existing conversation or create new
    conversation = await session_store.load(session_id)
    if not conversation:
        conversation = Conversation(
            session_id=session_id,
            channel=ChannelType.CHAT,
        )

    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            user_message = data.get("message", "").strip()

            if not user_message:
                continue

            # Send "typing" indicator
            await websocket.send_json({"type": "typing", "typing": True})

            try:
                reply, conversation = await agent.process_message(
                    conversation, user_message
                )
                await session_store.save(conversation)
            except Exception as exc:
                logger.exception("Agent error in WebSocket: %s", exc)
                reply = "I'm sorry, something went wrong. Please try again."

            await websocket.send_json({
                "type": "message",
                "typing": False,
                "message": reply,
                "session_id": conversation.session_id,
                "status": conversation.status,
                "escalated": conversation.status == ConversationStatus.ESCALATED,
            })

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: session=%s", session_id)
        await session_store.save(conversation)


# ---------------------------------------------------------------------------
# Conversation history endpoints
# ---------------------------------------------------------------------------

@router.get("/conversation/{session_id}", response_model=ConversationSummary)
async def get_conversation(
    session_id: str,
    session_store=Depends(get_session_store),
):
    """Fetch summary of a conversation."""
    conversation = await session_store.load(session_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Session not found")
    return ConversationSummary(
        session_id=conversation.session_id,
        status=conversation.status,
        turn_count=conversation.turn_count,
        channel=conversation.channel,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        ticket_id=conversation.ticket_id,
        escalated=(conversation.status == ConversationStatus.ESCALATED),
    )


@router.get("/conversation/{session_id}/messages")
async def get_messages(
    session_id: str,
    session_store=Depends(get_session_store),
):
    """Fetch full message history for a conversation."""
    conversation = await session_store.load(session_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": session_id,
        "messages": [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "created_at": m.created_at.isoformat(),
            }
            for m in conversation.messages
        ]
    }


@router.delete("/conversation/{session_id}")
async def close_conversation(
    session_id: str,
    session_store=Depends(get_session_store),
):
    """Mark a conversation as closed."""
    conversation = await session_store.load(session_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Session not found")
    conversation.status = ConversationStatus.CLOSED
    await session_store.save(conversation)
    return {"message": "Conversation closed", "session_id": session_id}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

async def _get_or_create_conversation(
    payload: ChatRequest,
    session_store,
) -> Conversation:
    if payload.session_id:
        conversation = await session_store.load(payload.session_id)
        if conversation:
            return conversation

    # New conversation
    conversation = Conversation(
        channel=payload.channel,
        customer_email=payload.customer_email,
        customer_id=payload.customer_id,
        metadata=payload.metadata,
    )
    if payload.session_id:
        conversation.session_id = payload.session_id
    await session_store.save(conversation)
    return conversation
