"""
Customer data models — account, orders, subscriptions.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, EmailStr, Field


class CustomerTier(str, Enum):
    FREE = "free"
    STARTER = "starter"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class OrderStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"


class SubscriptionStatus(str, Enum):
    ACTIVE = "active"
    TRIALING = "trialing"
    PAST_DUE = "past_due"
    CANCELLED = "cancelled"
    PAUSED = "paused"


# ---------------------------------------------------------------------------
# Order
# ---------------------------------------------------------------------------

class OrderItem(BaseModel):
    product_id: str
    product_name: str
    quantity: int
    unit_price: float
    total_price: float


class Order(BaseModel):
    order_id: str
    customer_id: str
    status: OrderStatus
    items: List[OrderItem] = Field(default_factory=list)
    total_amount: float
    currency: str = "USD"
    shipping_address: Optional[str] = None
    tracking_number: Optional[str] = None
    estimated_delivery: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        use_enum_values = True


# ---------------------------------------------------------------------------
# Subscription
# ---------------------------------------------------------------------------

class Subscription(BaseModel):
    subscription_id: str
    plan_name: str
    status: SubscriptionStatus
    billing_cycle: str = "monthly"   # monthly / annual
    amount: float
    currency: str = "USD"
    next_billing_date: Optional[datetime] = None
    trial_ends_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None

    class Config:
        use_enum_values = True


# ---------------------------------------------------------------------------
# Customer
# ---------------------------------------------------------------------------

class Customer(BaseModel):
    customer_id: str
    email: str
    full_name: str
    tier: CustomerTier = CustomerTier.FREE
    is_vip: bool = False
    phone: Optional[str] = None
    company: Optional[str] = None
    subscription: Optional[Subscription] = None
    recent_orders: List[Order] = Field(default_factory=list)
    open_ticket_count: int = 0
    total_ticket_count: int = 0
    account_created_at: Optional[datetime] = None
    last_contact_at: Optional[datetime] = None
    notes: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def display_name(self) -> str:
        return self.full_name or self.email

    @property
    def should_escalate(self) -> bool:
        """Auto-escalate VIP or enterprise customers."""
        return self.is_vip or self.tier == CustomerTier.ENTERPRISE

    class Config:
        use_enum_values = True


# ---------------------------------------------------------------------------
# Ticket model
# ---------------------------------------------------------------------------

class TicketStatus(str, Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    PENDING_CUSTOMER = "pending_customer"
    ESCALATED = "escalated"
    RESOLVED = "resolved"
    CLOSED = "closed"


class Ticket(BaseModel):
    ticket_id: str
    customer_id: Optional[str] = None
    customer_email: Optional[str] = None
    session_id: Optional[str] = None
    subject: str
    description: str
    status: TicketStatus = TicketStatus.OPEN
    priority: str = "medium"
    category: Optional[str] = None
    assigned_to: Optional[str] = None          # human agent name/id
    resolution_notes: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None

    class Config:
        use_enum_values = True


class TicketCreateRequest(BaseModel):
    subject: str
    description: str
    priority: str = "medium"
    category: Optional[str] = None
    customer_id: Optional[str] = None
    customer_email: Optional[str] = None
    session_id: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


class TicketUpdateRequest(BaseModel):
    status: Optional[TicketStatus] = None
    priority: Optional[str] = None
    assigned_to: Optional[str] = None
    resolution_notes: Optional[str] = None
    tags: Optional[List[str]] = None

    class Config:
        use_enum_values = True
