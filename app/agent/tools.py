"""
Tool definitions (Claude tool_use schema) + dispatcher.
Each handler calls the appropriate service layer.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool schemas — passed directly to the Claude API
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "name": "search_knowledge_base",
        "description": (
            "Search the internal knowledge base (FAQs, documentation, policies) "
            "to find answers to customer questions. Use this before answering "
            "any policy, how-to, or product question."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return (default 3)",
                    "default": 3
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "lookup_customer",
        "description": (
            "Look up a customer's account details, subscription status, order history, "
            "and support ticket history. Use when the customer provides their email or ID."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "identifier": {
                    "type": "string",
                    "description": "The customer's email address or customer ID"
                },
                "identifier_type": {
                    "type": "string",
                    "enum": ["email", "customer_id"],
                    "description": "Whether the identifier is an email or customer ID"
                }
            },
            "required": ["identifier", "identifier_type"]
        }
    },
    {
        "name": "create_ticket",
        "description": (
            "Create a support ticket to track this issue. Use for any issue that "
            "cannot be fully resolved in this conversation or requires follow-up."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "subject": {
                    "type": "string",
                    "description": "Brief subject line for the ticket"
                },
                "description": {
                    "type": "string",
                    "description": "Detailed description of the issue"
                },
                "priority": {
                    "type": "string",
                    "enum": ["low", "medium", "high", "urgent"],
                    "description": "Ticket priority level"
                },
                "category": {
                    "type": "string",
                    "description": "Issue category (e.g. billing, technical, shipping)"
                }
            },
            "required": ["subject", "description", "priority"]
        }
    },
    {
        "name": "update_ticket",
        "description": "Update an existing support ticket's status, notes, or assignment.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticket_id": {
                    "type": "string",
                    "description": "The ticket ID to update"
                },
                "status": {
                    "type": "string",
                    "enum": ["open", "in_progress", "pending_customer", "resolved", "closed"]
                },
                "note": {
                    "type": "string",
                    "description": "Note to add to the ticket"
                }
            },
            "required": ["ticket_id"]
        }
    },
    {
        "name": "escalate_to_human",
        "description": (
            "Escalate this conversation to a human support agent. Use when the issue "
            "is complex, the customer is very frustrated, legal/fraud keywords are used, "
            "or the customer explicitly requests a human."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Clear reason for escalation"
                },
                "urgency": {
                    "type": "string",
                    "enum": ["normal", "high", "critical"],
                    "description": "Urgency level for the human team"
                },
                "summary": {
                    "type": "string",
                    "description": "Concise summary of the issue and conversation for the human agent"
                }
            },
            "required": ["reason", "urgency", "summary"]
        }
    },
    {
        "name": "check_order_status",
        "description": (
            "Fetch real-time status of a customer order including shipping, "
            "tracking, and estimated delivery date."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The order ID to look up"
                }
            },
            "required": ["order_id"]
        }
    },
    {
        "name": "send_email_reply",
        "description": (
            "Send a follow-up email to the customer. Use for email channel conversations "
            "or when the customer needs a written record of the resolution."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": "Recipient email address"
                },
                "subject": {
                    "type": "string",
                    "description": "Email subject line"
                },
                "body": {
                    "type": "string",
                    "description": "Plain text email body"
                }
            },
            "required": ["to", "subject", "body"]
        }
    }
]


# ---------------------------------------------------------------------------
# Tool Dispatcher
# ---------------------------------------------------------------------------

class ToolDispatcher:
    """
    Routes tool_use blocks from Claude to the correct service handler.
    Each handler is async and returns a dict result.
    """

    def __init__(
        self,
        knowledge_base_service=None,
        customer_service=None,
        ticket_service=None,
        email_service=None,
    ):
        self.kb = knowledge_base_service
        self.customer_svc = customer_service
        self.ticket_svc = ticket_service
        self.email_svc = email_service

    async def dispatch(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Route a tool call to the appropriate handler.
        Returns a dict that is serialised back to Claude as a tool_result.
        """
        start = time.monotonic()
        try:
            result = await self._call_tool(tool_name, tool_input, session_id)
            result["_latency_ms"] = round((time.monotonic() - start) * 1000, 2)
            result["_success"] = True
            return result
        except Exception as exc:
            logger.exception("Tool %s failed: %s", tool_name, exc)
            return {
                "_success": False,
                "_latency_ms": round((time.monotonic() - start) * 1000, 2),
                "error": str(exc),
                "message": (
                    f"The {tool_name} tool encountered an error. "
                    "Please inform the customer that this information is temporarily unavailable."
                )
            }

    async def _call_tool(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        session_id: Optional[str],
    ) -> Dict[str, Any]:
        handlers = {
            "search_knowledge_base": self._search_knowledge_base,
            "lookup_customer": self._lookup_customer,
            "create_ticket": self._create_ticket,
            "update_ticket": self._update_ticket,
            "escalate_to_human": self._escalate_to_human,
            "check_order_status": self._check_order_status,
            "send_email_reply": self._send_email_reply,
        }
        handler = handlers.get(tool_name)
        if not handler:
            raise ValueError(f"Unknown tool: {tool_name}")
        return await handler(tool_input, session_id=session_id)

    # ------------------------------------------------------------------
    # Individual tool handlers
    # ------------------------------------------------------------------

    async def _search_knowledge_base(
        self, inputs: Dict[str, Any], **_
    ) -> Dict[str, Any]:
        query = inputs["query"]
        top_k = inputs.get("top_k", 3)
        if self.kb:
            results = await self.kb.search(query, top_k=top_k)
        else:
            # Mock fallback for local development
            results = [
                {
                    "title": "Refund Policy",
                    "content": (
                        "We offer a 30-day money-back guarantee on all plans. "
                        "Refunds are processed within 5–7 business days."
                    ),
                    "score": 0.92
                }
            ]
        return {"results": results, "query": query}

    async def _lookup_customer(
        self, inputs: Dict[str, Any], **_
    ) -> Dict[str, Any]:
        identifier = inputs["identifier"]
        id_type = inputs["identifier_type"]
        if self.customer_svc:
            customer = await self.customer_svc.lookup(identifier, id_type)
        else:
            # Mock customer for development
            customer = {
                "customer_id": "cust_demo_001",
                "email": identifier if id_type == "email" else "demo@example.com",
                "full_name": "Demo User",
                "tier": "pro",
                "is_vip": False,
                "subscription": {"plan_name": "Pro Monthly", "status": "active"},
                "recent_orders": [],
                "open_ticket_count": 1
            }
        if not customer:
            return {"found": False, "message": "No customer found with that identifier."}
        return {"found": True, "customer": customer}

    async def _create_ticket(
        self, inputs: Dict[str, Any], session_id: Optional[str] = None, **_
    ) -> Dict[str, Any]:
        if self.ticket_svc:
            ticket = await self.ticket_svc.create(
                subject=inputs["subject"],
                description=inputs["description"],
                priority=inputs["priority"],
                category=inputs.get("category"),
                session_id=session_id,
            )
        else:
            import uuid
            ticket = {
                "ticket_id": f"TKT-{uuid.uuid4().hex[:6].upper()}",
                "subject": inputs["subject"],
                "status": "open",
                "priority": inputs["priority"],
            }
        return {
            "ticket_created": True,
            "ticket_id": ticket["ticket_id"],
            "message": f"Ticket {ticket['ticket_id']} created successfully."
        }

    async def _update_ticket(
        self, inputs: Dict[str, Any], **_
    ) -> Dict[str, Any]:
        ticket_id = inputs["ticket_id"]
        if self.ticket_svc:
            await self.ticket_svc.update(
                ticket_id=ticket_id,
                status=inputs.get("status"),
                note=inputs.get("note"),
            )
        return {
            "updated": True,
            "ticket_id": ticket_id,
            "message": f"Ticket {ticket_id} updated."
        }

    async def _escalate_to_human(
        self, inputs: Dict[str, Any], session_id: Optional[str] = None, **_
    ) -> Dict[str, Any]:
        # In production: push to human queue (e.g., Zendesk, Intercom, internal queue)
        logger.info(
            "ESCALATION | session=%s reason=%s urgency=%s",
            session_id, inputs["reason"], inputs["urgency"]
        )
        if self.ticket_svc:
            ticket = await self.ticket_svc.create(
                subject=f"[ESCALATED] {inputs['reason']}",
                description=inputs["summary"],
                priority="urgent" if inputs["urgency"] == "critical" else "high",
                category="escalation",
                session_id=session_id,
            )
            ticket_id = ticket["ticket_id"]
        else:
            ticket_id = "ESC-DEMO"

        return {
            "escalated": True,
            "ticket_id": ticket_id,
            "queue_position": 1,
            "estimated_wait": "5–10 minutes",
            "message": (
                "The conversation has been escalated. A human agent will join shortly."
            )
        }

    async def _check_order_status(
        self, inputs: Dict[str, Any], **_
    ) -> Dict[str, Any]:
        order_id = inputs["order_id"]
        if self.customer_svc:
            order = await self.customer_svc.get_order(order_id)
        else:
            order = {
                "order_id": order_id,
                "status": "shipped",
                "tracking_number": "1Z999AA10123456784",
                "carrier": "UPS",
                "estimated_delivery": "2026-02-26",
                "items": [{"product_name": "Pro Subscription", "quantity": 1}]
            }
        if not order:
            return {"found": False, "message": f"Order {order_id} not found."}
        return {"found": True, "order": order}

    async def _send_email_reply(
        self, inputs: Dict[str, Any], **_
    ) -> Dict[str, Any]:
        if self.email_svc:
            sent = await self.email_svc.send(
                to=inputs["to"],
                subject=inputs["subject"],
                body=inputs["body"],
            )
        else:
            sent = True
        return {
            "sent": sent,
            "to": inputs["to"],
            "message": f"Email sent to {inputs['to']}" if sent else "Email delivery failed."
        }
