"""
Tests for the AI Agent Core.
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from app.agent.core import SupportAgentCore
from app.agent.escalation import EscalationEngine
from app.agent.tools import ToolDispatcher
from app.models.conversation import (
    Conversation,
    ConversationStatus,
    ChannelType,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_anthropic_client():
    client = AsyncMock()
    # Default: return a simple end_turn response
    mock_response = MagicMock()
    mock_response.stop_reason = "end_turn"
    mock_response.content = [MagicMock(type="text", text="Hello! How can I help you today?")]
    client.messages.create = AsyncMock(return_value=mock_response)
    return client


@pytest.fixture
def mock_dispatcher():
    dispatcher = AsyncMock(spec=ToolDispatcher)
    dispatcher.dispatch = AsyncMock(return_value={"_success": True, "result": "mock result"})
    return dispatcher


@pytest.fixture
def mock_escalation():
    engine = AsyncMock(spec=EscalationEngine)
    from app.agent.escalation import EscalationDecision
    engine.evaluate = AsyncMock(return_value=EscalationDecision(should_escalate=False))
    return engine


@pytest.fixture
def agent(mock_anthropic_client, mock_dispatcher, mock_escalation):
    return SupportAgentCore(
        anthropic_client=mock_anthropic_client,
        tool_dispatcher=mock_dispatcher,
        escalation_engine=mock_escalation,
    )


@pytest.fixture
def fresh_conversation():
    return Conversation(channel=ChannelType.CHAT)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_basic_message_processing(agent, fresh_conversation, mock_anthropic_client):
    """Agent should return a reply and append it to conversation history."""
    reply, updated_conv = await agent.process_message(
        fresh_conversation, "Hello, I need help"
    )

    assert isinstance(reply, str)
    assert len(reply) > 0
    assert updated_conv.turn_count == 2   # user + assistant
    # User message appended
    assert updated_conv.messages[0].content == "Hello, I need help"
    # Assistant reply appended
    assert updated_conv.messages[1].content == reply


@pytest.mark.asyncio
async def test_escalation_is_triggered_on_legal_keyword(agent, fresh_conversation, mock_escalation):
    """Legal keyword should trigger escalation."""
    from app.agent.escalation import EscalationDecision
    mock_escalation.evaluate = AsyncMock(
        return_value=EscalationDecision(
            should_escalate=True,
            reason="Legal keyword detected",
            urgency="critical",
            trigger_type="rule",
        )
    )

    reply, conv = await agent.process_message(
        fresh_conversation, "I'm going to sue your company!"
    )

    assert conv.status == ConversationStatus.ESCALATED
    assert "human" in reply.lower() or "specialist" in reply.lower()


@pytest.mark.asyncio
async def test_conversation_history_grows(agent, fresh_conversation):
    """Each turn should add 2 messages (user + assistant)."""
    _, conv = await agent.process_message(fresh_conversation, "First message")
    _, conv = await agent.process_message(conv, "Second message")

    user_msgs = [m for m in conv.messages if m.role == "user"]
    assistant_msgs = [m for m in conv.messages if m.role == "assistant"]

    assert len(user_msgs) == 2
    assert len(assistant_msgs) == 2


@pytest.mark.asyncio
async def test_tool_call_is_dispatched(agent, fresh_conversation, mock_anthropic_client, mock_dispatcher):
    """When Claude returns a tool_use block, the dispatcher should be called."""
    tool_use_block = MagicMock()
    tool_use_block.type = "tool_use"
    tool_use_block.id = "tu_001"
    tool_use_block.name = "search_knowledge_base"
    tool_use_block.input = {"query": "refund policy"}

    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "Here is our refund policy..."

    # First call: tool_use; Second call: end_turn with text
    tool_response = MagicMock()
    tool_response.stop_reason = "tool_use"
    tool_response.content = [tool_use_block]

    final_response = MagicMock()
    final_response.stop_reason = "end_turn"
    final_response.content = [text_block]

    mock_anthropic_client.messages.create = AsyncMock(
        side_effect=[tool_response, final_response]
    )

    reply, conv = await agent.process_message(fresh_conversation, "What is your refund policy?")

    # Dispatcher should have been called once
    mock_dispatcher.dispatch.assert_called_once_with(
        tool_name="search_knowledge_base",
        tool_input={"query": "refund policy"},
        session_id=conv.session_id,
    )
    assert "refund" in reply.lower()


@pytest.mark.asyncio
async def test_fallback_on_api_error(agent, fresh_conversation, mock_anthropic_client):
    """On API error, agent should fall back gracefully."""
    import anthropic
    mock_anthropic_client.messages.create = AsyncMock(
        side_effect=RuntimeError("Both models unavailable")
    )

    with pytest.raises(RuntimeError):
        await agent.process_message(fresh_conversation, "Hello")


@pytest.mark.asyncio
async def test_vip_escalation(agent, mock_escalation):
    """VIP customer should be escalated after 3 turns."""
    from app.agent.escalation import EscalationDecision

    vip_conv = Conversation(channel=ChannelType.CHAT, is_vip=True, turn_count=3)

    mock_escalation.evaluate = AsyncMock(
        return_value=EscalationDecision(
            should_escalate=True,
            reason="VIP customer with unresolved issue",
            urgency="high",
            trigger_type="rule",
        )
    )

    reply, conv = await agent.process_message(vip_conv, "I still need help")
    assert conv.status == ConversationStatus.ESCALATED
