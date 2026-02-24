"""
Customer Database Service — lookup, order status, and account management.
Uses PostgreSQL/Supabase in production; mock data for local development.
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mock customer data
# ---------------------------------------------------------------------------

MOCK_CUSTOMERS: Dict[str, Dict] = {
    "demo@example.com": {
        "customer_id": "cust_001",
        "email": "demo@example.com",
        "full_name": "Alex Johnson",
        "tier": "pro",
        "is_vip": False,
        "phone": "+1-555-0101",
        "company": None,
        "subscription": {
            "subscription_id": "sub_001",
            "plan_name": "Pro Monthly",
            "status": "active",
            "billing_cycle": "monthly",
            "amount": 49.0,
            "currency": "USD",
            "next_billing_date": (datetime.utcnow() + timedelta(days=15)).isoformat(),
        },
        "recent_orders": [
            {
                "order_id": "ORD-001",
                "status": "delivered",
                "total_amount": 49.0,
                "currency": "USD",
                "items": [{"product_name": "Pro Plan - Monthly", "quantity": 1, "unit_price": 49.0, "total_price": 49.0}],
                "created_at": (datetime.utcnow() - timedelta(days=30)).isoformat(),
            }
        ],
        "open_ticket_count": 0,
        "total_ticket_count": 2,
        "account_created_at": (datetime.utcnow() - timedelta(days=180)).isoformat(),
    },
    "vip@enterprise.com": {
        "customer_id": "cust_vip_001",
        "email": "vip@enterprise.com",
        "full_name": "Sarah Chen",
        "tier": "enterprise",
        "is_vip": True,
        "phone": "+1-555-0202",
        "company": "TechCorp Inc.",
        "subscription": {
            "subscription_id": "sub_vip_001",
            "plan_name": "Enterprise Annual",
            "status": "active",
            "billing_cycle": "annual",
            "amount": 5000.0,
            "currency": "USD",
            "next_billing_date": (datetime.utcnow() + timedelta(days=200)).isoformat(),
        },
        "recent_orders": [],
        "open_ticket_count": 1,
        "total_ticket_count": 15,
        "account_created_at": (datetime.utcnow() - timedelta(days=730)).isoformat(),
    },
}

MOCK_ORDERS: Dict[str, Dict] = {
    "ORD-001": {
        "order_id": "ORD-001",
        "customer_id": "cust_001",
        "status": "delivered",
        "items": [{"product_name": "Pro Plan - Monthly", "quantity": 1, "unit_price": 49.0, "total_price": 49.0}],
        "total_amount": 49.0,
        "currency": "USD",
        "tracking_number": None,
        "estimated_delivery": None,
        "created_at": (datetime.utcnow() - timedelta(days=30)).isoformat(),
        "updated_at": (datetime.utcnow() - timedelta(days=28)).isoformat(),
    },
    "ORD-002": {
        "order_id": "ORD-002",
        "customer_id": "cust_002",
        "status": "shipped",
        "items": [{"product_name": "Widget Pro X", "quantity": 2, "unit_price": 29.99, "total_price": 59.98}],
        "total_amount": 59.98,
        "currency": "USD",
        "tracking_number": "1Z999AA10123456784",
        "carrier": "UPS",
        "estimated_delivery": (datetime.utcnow() + timedelta(days=2)).strftime("%Y-%m-%d"),
        "created_at": (datetime.utcnow() - timedelta(days=3)).isoformat(),
        "updated_at": (datetime.utcnow() - timedelta(days=1)).isoformat(),
    },
}


# ---------------------------------------------------------------------------
# Customer DB Service
# ---------------------------------------------------------------------------

class CustomerDBService:
    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url
        self._pool = None

    async def initialize(self) -> None:
        if self.db_url:
            try:
                import asyncpg
                self._pool = await asyncpg.create_pool(self.db_url)
                logger.info("CustomerDB: PostgreSQL connected")
            except Exception as exc:
                logger.error("CustomerDB init failed: %s", exc)
        else:
            logger.info("CustomerDB: using mock data")

    async def lookup(
        self,
        identifier: str,
        identifier_type: str = "email",
    ) -> Optional[Dict[str, Any]]:
        """Look up a customer by email or customer_id."""
        if self._pool:
            return await self._db_lookup(identifier, identifier_type)

        # Mock lookup
        if identifier_type == "email":
            return MOCK_CUSTOMERS.get(identifier.lower())
        else:
            # Search by customer_id
            for customer in MOCK_CUSTOMERS.values():
                if customer["customer_id"] == identifier:
                    return customer
        return None

    async def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single order by order_id."""
        if self._pool:
            return await self._db_get_order(order_id)
        return MOCK_ORDERS.get(order_id.upper())

    async def get_customer_orders(
        self, customer_id: str, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Fetch recent orders for a customer."""
        if self._pool:
            return await self._db_customer_orders(customer_id, limit)
        return [
            o for o in MOCK_ORDERS.values()
            if o.get("customer_id") == customer_id
        ][:limit]

    async def update_customer(
        self, customer_id: str, updates: Dict[str, Any]
    ) -> bool:
        """Update customer fields."""
        if self._pool:
            async with self._pool.acquire() as conn:
                set_clauses = ", ".join(f"{k} = ${i+2}" for i, k in enumerate(updates))
                values = list(updates.values())
                await conn.execute(
                    f"UPDATE customers SET {set_clauses}, updated_at = NOW() WHERE customer_id = $1",
                    customer_id, *values
                )
            return True
        # Mock update
        for customer in MOCK_CUSTOMERS.values():
            if customer["customer_id"] == customer_id:
                customer.update(updates)
                return True
        return False

    # ------------------------------------------------------------------
    # PostgreSQL queries
    # ------------------------------------------------------------------

    async def _db_lookup(
        self, identifier: str, identifier_type: str
    ) -> Optional[Dict[str, Any]]:
        field = "email" if identifier_type == "email" else "customer_id"
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                SELECT c.*, s.plan_name, s.status as sub_status, s.amount,
                       s.next_billing_date, s.billing_cycle
                FROM customers c
                LEFT JOIN subscriptions s ON s.customer_id = c.customer_id
                WHERE c.{field} = $1
                """,
                identifier
            )
        return dict(row) if row else None

    async def _db_get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT o.*, array_agg(
                    json_build_object(
                        'product_name', oi.product_name,
                        'quantity', oi.quantity,
                        'unit_price', oi.unit_price
                    )
                ) as items
                FROM orders o
                LEFT JOIN order_items oi ON oi.order_id = o.order_id
                WHERE o.order_id = $1
                GROUP BY o.order_id
                """,
                order_id
            )
        return dict(row) if row else None

    async def _db_customer_orders(
        self, customer_id: str, limit: int
    ) -> List[Dict[str, Any]]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM orders WHERE customer_id = $1 ORDER BY created_at DESC LIMIT $2",
                customer_id, limit
            )
        return [dict(r) for r in rows]
