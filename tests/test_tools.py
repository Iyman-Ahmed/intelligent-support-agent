"""
Tests for the Tool Dispatcher and individual tool handlers.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from app.agent.tools import ToolDispatcher


@pytest.fixture
def dispatcher():
    return ToolDispatcher()   # No real services — uses mocks/fallbacks


@pytest.mark.asyncio
async def test_search_knowledge_base_returns_results(dispatcher):
    result = await dispatcher.dispatch("search_knowledge_base", {"query": "refund policy"})
    assert result["_success"] is True
    assert "results" in result
    assert isinstance(result["results"], list)


@pytest.mark.asyncio
async def test_lookup_customer_found(dispatcher):
    result = await dispatcher.dispatch(
        "lookup_customer",
        {"identifier": "demo@example.com", "identifier_type": "email"}
    )
    assert result["_success"] is True
    assert result["found"] is True
    assert "customer" in result


@pytest.mark.asyncio
async def test_create_ticket(dispatcher):
    result = await dispatcher.dispatch(
        "create_ticket",
        {
            "subject": "Cannot login to my account",
            "description": "Getting error 401 when logging in",
            "priority": "high",
        }
    )
    assert result["_success"] is True
    assert result["ticket_created"] is True
    assert result["ticket_id"].startswith("TKT-")


@pytest.mark.asyncio
async def test_check_order_status_found(dispatcher):
    result = await dispatcher.dispatch(
        "check_order_status",
        {"order_id": "ORD-002"}
    )
    assert result["_success"] is True


@pytest.mark.asyncio
async def test_escalate_to_human(dispatcher):
    result = await dispatcher.dispatch(
        "escalate_to_human",
        {
            "reason": "Customer requested a human",
            "urgency": "high",
            "summary": "Customer cannot access their account after password reset.",
        }
    )
    assert result["_success"] is True
    assert result["escalated"] is True


@pytest.mark.asyncio
async def test_unknown_tool_returns_error(dispatcher):
    result = await dispatcher.dispatch("nonexistent_tool", {})
    assert result["_success"] is False
    assert "error" in result


@pytest.mark.asyncio
async def test_tool_latency_recorded(dispatcher):
    result = await dispatcher.dispatch("search_knowledge_base", {"query": "billing"})
    assert "_latency_ms" in result
    assert result["_latency_ms"] >= 0


@pytest.mark.asyncio
async def test_send_email_console_fallback(dispatcher):
    result = await dispatcher.dispatch(
        "send_email_reply",
        {
            "to": "customer@example.com",
            "subject": "Your support request",
            "body": "Thank you for contacting us.",
        }
    )
    assert result["_success"] is True
    assert result["sent"] is True
