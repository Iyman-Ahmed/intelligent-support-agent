"""
Conversation & Message data models.
Covers session state, message history, and tool call records.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ConversationStatus(str, Enum):
    OPEN = "open"
    PENDING = "pending"         # Waiting on customer reply
    ESCALATED = "escalated"     # Handed off to human
    RESOLVED = "resolved"
    CLOSED = "closed"


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class ChannelType(str, Enum):
    CHAT = "chat"
    EMAIL = "email"
    SLACK = "slack"
    WHATSAPP = "whatsapp"


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


# ---------------------------------------------------------------------------
# Tool call record (persisted alongside messages)
# ---------------------------------------------------------------------------

class ToolCallRecord(BaseModel):
    tool_use_id: str
    tool_name: str
    inputs: Dict[str, Any]
    output: Optional[Dict[str, Any]] = None
    success: bool = True
    error: Optional[str] = None
    latency_ms: Optional[float] = None
    called_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Individual message
# ---------------------------------------------------------------------------

class Message(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: MessageRole
    content: str                        # Plain text content
    raw_content: Optional[Any] = None  # Full Claude content blocks (list)
    tool_calls: List[ToolCallRecord] = Field(default_factory=list)
    sentiment_score: Optional[float] = None   # -1.0 (negative) to 1.0 (positive)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        use_enum_values = True


# ---------------------------------------------------------------------------
# Conversation / Session
# ---------------------------------------------------------------------------

class Conversation(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    customer_id: Optional[str] = None
    customer_email: Optional[str] = None
    channel: ChannelType = ChannelType.CHAT
    status: ConversationStatus = ConversationStatus.OPEN
    messages: List[Message] = Field(default_factory=list)
    ticket_id: Optional[str] = None
    escalation_reason: Optional[str] = None
    turn_count: int = 0
    is_vip: bool = False
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None

    def add_message(self, role: MessageRole, content: str, **kwargs) -> Message:
        msg = Message(role=role, content=content, **kwargs)
        self.messages.append(msg)
        self.turn_count += 1
        self.updated_at = datetime.utcnow()
        return msg

    def to_claude_messages(self, max_turns: int = 20) -> List[Dict[str, Any]]:
        """Convert conversation history to Claude API message format."""
        history = [
            m for m in self.messages
            if m.role in (MessageRole.USER, MessageRole.ASSISTANT)
        ]
        # Rolling window — keep last max_turns messages
        history = history[-max_turns:]
        result = []
        for msg in history:
            if msg.raw_content is not None:
                result.append({"role": msg.role, "content": msg.raw_content})
            else:
                result.append({"role": msg.role, "content": msg.content})
        return result

    class Config:
        use_enum_values = True


# ---------------------------------------------------------------------------
# API Request / Response schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    session_id: Optional[str] = None        # None → new session
    message: str
    customer_email: Optional[str] = None
    customer_id: Optional[str] = None
    channel: ChannelType = ChannelType.CHAT
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    session_id: str
    message: str
    status: ConversationStatus
    ticket_id: Optional[str] = None
    escalated: bool = False
    response_time_ms: Optional[float] = None

    class Config:
        use_enum_values = True


class ConversationSummary(BaseModel):
    session_id: str
    status: ConversationStatus
    turn_count: int
    channel: ChannelType
    created_at: datetime
    updated_at: datetime
    ticket_id: Optional[str] = None
    escalated: bool = False

    class Config:
        use_enum_values = True
