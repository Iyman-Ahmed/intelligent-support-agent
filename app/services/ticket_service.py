"""
Ticket Service — create, update, and query support tickets.
In production: integrates with Zendesk / Freshdesk or custom PostgreSQL store.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# In-memory ticket store for development
_TICKET_STORE: Dict[str, Dict] = {}


class TicketService:
    def __init__(
        self,
        db_url: Optional[str] = None,
        zendesk_api_key: Optional[str] = None,
        zendesk_subdomain: Optional[str] = None,
    ):
        self.db_url = db_url
        self.zendesk_api_key = zendesk_api_key
        self.zendesk_subdomain = zendesk_subdomain
        self._pool = None

    async def initialize(self) -> None:
        if self.db_url:
            try:
                import asyncpg
                self._pool = await asyncpg.create_pool(self.db_url)
                logger.info("TicketService: PostgreSQL connected")
            except Exception as exc:
                logger.error("TicketService DB init failed: %s", exc)
        elif self.zendesk_api_key:
            logger.info("TicketService: Zendesk mode")
        else:
            logger.info("TicketService: in-memory mock mode")

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    async def create(
        self,
        subject: str,
        description: str,
        priority: str = "medium",
        category: Optional[str] = None,
        customer_id: Optional[str] = None,
        customer_email: Optional[str] = None,
        session_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        ticket_id = f"TKT-{uuid.uuid4().hex[:6].upper()}"
        ticket = {
            "ticket_id": ticket_id,
            "subject": subject,
            "description": description,
            "priority": priority,
            "category": category,
            "customer_id": customer_id,
            "customer_email": customer_email,
            "session_id": session_id,
            "status": "open",
            "tags": tags or [],
            "assigned_to": None,
            "resolution_notes": None,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "resolved_at": None,
        }

        if self._pool:
            ticket = await self._db_create(ticket)
        elif self.zendesk_api_key:
            ticket = await self._zendesk_create(ticket)
        else:
            _TICKET_STORE[ticket_id] = ticket

        logger.info("Ticket created: %s (%s)", ticket_id, priority)
        return ticket

    async def get(self, ticket_id: str) -> Optional[Dict[str, Any]]:
        if self._pool:
            return await self._db_get(ticket_id)
        return _TICKET_STORE.get(ticket_id)

    async def update(
        self,
        ticket_id: str,
        status: Optional[str] = None,
        note: Optional[str] = None,
        assigned_to: Optional[str] = None,
        resolution_notes: Optional[str] = None,
    ) -> bool:
        updates: Dict[str, Any] = {"updated_at": datetime.utcnow().isoformat()}
        if status:
            updates["status"] = status
            if status in ("resolved", "closed"):
                updates["resolved_at"] = datetime.utcnow().isoformat()
        if assigned_to:
            updates["assigned_to"] = assigned_to
        if resolution_notes:
            updates["resolution_notes"] = resolution_notes

        if self._pool:
            return await self._db_update(ticket_id, updates)

        ticket = _TICKET_STORE.get(ticket_id)
        if ticket:
            ticket.update(updates)
            if note:
                ticket.setdefault("notes", []).append({
                    "content": note,
                    "created_at": datetime.utcnow().isoformat()
                })
            return True
        return False

    async def list_by_customer(
        self, customer_id: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        if self._pool:
            return await self._db_list_by_customer(customer_id, limit)
        return [
            t for t in _TICKET_STORE.values()
            if t.get("customer_id") == customer_id
        ][:limit]

    async def list_open(self, limit: int = 50) -> List[Dict[str, Any]]:
        if self._pool:
            return await self._db_list_open(limit)
        return [
            t for t in _TICKET_STORE.values()
            if t["status"] in ("open", "in_progress", "pending_customer")
        ][:limit]

    # ------------------------------------------------------------------
    # PostgreSQL
    # ------------------------------------------------------------------

    async def _db_create(self, ticket: Dict) -> Dict:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO tickets (
                    ticket_id, subject, description, priority, category,
                    customer_id, customer_email, session_id, status, tags,
                    created_at, updated_at
                ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
                """,
                ticket["ticket_id"], ticket["subject"], ticket["description"],
                ticket["priority"], ticket["category"], ticket["customer_id"],
                ticket["customer_email"], ticket["session_id"], ticket["status"],
                ticket["tags"], ticket["created_at"], ticket["updated_at"]
            )
        return ticket

    async def _db_get(self, ticket_id: str) -> Optional[Dict]:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM tickets WHERE ticket_id = $1", ticket_id
            )
        return dict(row) if row else None

    async def _db_update(self, ticket_id: str, updates: Dict) -> bool:
        if not updates:
            return True
        set_clauses = ", ".join(f"{k} = ${i+2}" for i, k in enumerate(updates))
        values = list(updates.values())
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"UPDATE tickets SET {set_clauses} WHERE ticket_id = $1",
                ticket_id, *values
            )
        return True

    async def _db_list_by_customer(self, customer_id: str, limit: int) -> List[Dict]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM tickets WHERE customer_id = $1 ORDER BY created_at DESC LIMIT $2",
                customer_id, limit
            )
        return [dict(r) for r in rows]

    async def _db_list_open(self, limit: int) -> List[Dict]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM tickets WHERE status IN ('open','in_progress','pending_customer') "
                "ORDER BY created_at DESC LIMIT $1",
                limit
            )
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Zendesk integration
    # ------------------------------------------------------------------

    async def _zendesk_create(self, ticket: Dict) -> Dict:
        try:
            import httpx
            url = f"https://{self.zendesk_subdomain}.zendesk.com/api/v2/tickets.json"
            payload = {
                "ticket": {
                    "subject": ticket["subject"],
                    "comment": {"body": ticket["description"]},
                    "priority": ticket["priority"],
                    "tags": ticket.get("tags", []),
                }
            }
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    url,
                    json=payload,
                    auth=("token", self.zendesk_api_key),
                )
                resp.raise_for_status()
                zd_ticket = resp.json()["ticket"]
                ticket["zendesk_id"] = zd_ticket["id"]
        except Exception as exc:
            logger.error("Zendesk ticket creation failed: %s", exc)
        return ticket
