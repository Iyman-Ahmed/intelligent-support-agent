"""
Ticket management API — CRUD endpoints for support tickets.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request

from app.models.customer import (
    Ticket,
    TicketCreateRequest,
    TicketUpdateRequest,
    TicketStatus,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/tickets", tags=["tickets"])


def get_ticket_service(request: Request):
    return request.app.state.ticket_service


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/", response_model=dict)
async def create_ticket(
    payload: TicketCreateRequest,
    ticket_service=Depends(get_ticket_service),
):
    """Manually create a support ticket."""
    ticket = await ticket_service.create(
        subject=payload.subject,
        description=payload.description,
        priority=payload.priority,
        category=payload.category,
        customer_id=payload.customer_id,
        customer_email=payload.customer_email,
        session_id=payload.session_id,
        tags=payload.tags,
    )
    return {"message": "Ticket created", "ticket": ticket}


@router.get("/{ticket_id}")
async def get_ticket(
    ticket_id: str,
    ticket_service=Depends(get_ticket_service),
):
    """Fetch a single ticket by ID."""
    ticket = await ticket_service.get(ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} not found")
    return ticket


@router.patch("/{ticket_id}")
async def update_ticket(
    ticket_id: str,
    payload: TicketUpdateRequest,
    ticket_service=Depends(get_ticket_service),
):
    """Update a ticket's status, assignee, or resolution notes."""
    updated = await ticket_service.update(
        ticket_id=ticket_id,
        status=payload.status,
        assigned_to=payload.assigned_to,
        resolution_notes=payload.resolution_notes,
    )
    if not updated:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} not found")
    return {"message": "Ticket updated", "ticket_id": ticket_id}


@router.get("/customer/{customer_id}")
async def list_customer_tickets(
    customer_id: str,
    limit: int = 10,
    ticket_service=Depends(get_ticket_service),
):
    """List all tickets for a specific customer."""
    tickets = await ticket_service.list_by_customer(customer_id, limit=limit)
    return {"customer_id": customer_id, "tickets": tickets, "count": len(tickets)}


@router.get("/")
async def list_open_tickets(
    limit: int = 50,
    ticket_service=Depends(get_ticket_service),
):
    """List all open / in-progress tickets (for agent dashboard)."""
    tickets = await ticket_service.list_open(limit=limit)
    return {"tickets": tickets, "count": len(tickets)}


@router.post("/{ticket_id}/resolve")
async def resolve_ticket(
    ticket_id: str,
    resolution_notes: str,
    ticket_service=Depends(get_ticket_service),
):
    """Mark a ticket as resolved."""
    updated = await ticket_service.update(
        ticket_id=ticket_id,
        status="resolved",
        resolution_notes=resolution_notes,
    )
    if not updated:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} not found")
    return {"message": "Ticket resolved", "ticket_id": ticket_id}
