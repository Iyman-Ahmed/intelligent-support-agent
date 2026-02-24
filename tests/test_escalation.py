"""
Tests for the Escalation Engine.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock

from app.agent.escalation import EscalationEngine
from app.models.conversation import Conversation, ConversationStatus, ChannelType


@pytest.fixture
def engine():
    return EscalationEngine(anthropic_client=None)  # No LLM for unit tests


@pytest.fixture
def fresh_conv():
    return Conversation(channel=ChannelType.CHAT)


@pytest.mark.asyncio
async def test_human_request_triggers_escalation(engine, fresh_conv):
    decision = await engine.evaluate(fresh_conv, "I want to speak to a human agent please")
    assert decision.should_escalate is True
    assert decision.trigger_type == "rule"
    assert decision.urgency == "high"


@pytest.mark.asyncio
async def test_legal_keyword_triggers_escalation(engine, fresh_conv):
    decision = await engine.evaluate(fresh_conv, "I'm going to file a lawsuit against you")
    assert decision.should_escalate is True
    assert decision.urgency == "critical"


@pytest.mark.asyncio
async def test_fraud_keyword_triggers_escalation(engine, fresh_conv):
    decision = await engine.evaluate(fresh_conv, "This is fraud and I'm reporting it")
    assert decision.should_escalate is True
    assert decision.urgency == "critical"


@pytest.mark.asyncio
async def test_normal_message_no_escalation(engine, fresh_conv):
    decision = await engine.evaluate(fresh_conv, "How do I reset my password?")
    assert decision.should_escalate is False


@pytest.mark.asyncio
async def test_turn_limit_triggers_escalation(engine):
    conv = Conversation(channel=ChannelType.CHAT, turn_count=10)
    decision = await engine.evaluate(conv, "I still need help")
    assert decision.should_escalate is True
    assert decision.trigger_type == "turn_limit"


@pytest.mark.asyncio
async def test_vip_customer_escalates_after_3_turns(engine):
    conv = Conversation(channel=ChannelType.CHAT, is_vip=True, turn_count=3)
    decision = await engine.evaluate(conv, "I have another question")
    assert decision.should_escalate is True
    assert decision.urgency == "high"


@pytest.mark.asyncio
async def test_already_escalated_skips_check(engine):
    conv = Conversation(
        channel=ChannelType.CHAT,
        status=ConversationStatus.ESCALATED
    )
    decision = await engine.evaluate(conv, "I want to sue you")
    # Already escalated — should not re-escalate
    assert decision.should_escalate is False


@pytest.mark.asyncio
async def test_manager_request_triggers_escalation(engine, fresh_conv):
    decision = await engine.evaluate(fresh_conv, "Let me speak to your manager")
    assert decision.should_escalate is True


def test_refund_threshold_high_amount(engine):
    assert engine.check_refund_threshold("I want a refund of $600") is True


def test_refund_threshold_low_amount(engine):
    assert engine.check_refund_threshold("I want a $10 refund") is False


def test_refund_threshold_no_amount(engine):
    assert engine.check_refund_threshold("I want a refund") is False
