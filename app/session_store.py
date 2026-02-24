"""
Session Store — persists Conversation objects across requests.
Uses Redis for hot storage (active sessions) and PostgreSQL for cold storage.
Falls back to an in-memory dict for local development.
"""

from __future__ import annotations

import json
import logging
import pickle
from typing import Dict, Optional

from app.models.conversation import Conversation

logger = logging.getLogger(__name__)

# In-memory fallback (development)
_MEMORY_STORE: Dict[str, str] = {}


class SessionStore:
    """
    Conversation session store backed by Redis.
    Falls back to in-memory if Redis is unavailable.
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        ttl_seconds: int = 86400,   # 24 hours
    ):
        self.redis_url = redis_url
        self.ttl = ttl_seconds
        self._redis = None

    async def initialize(self) -> None:
        if self.redis_url:
            try:
                import redis.asyncio as aioredis
                self._redis = await aioredis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                )
                await self._redis.ping()
                logger.info("SessionStore: Redis connected at %s", self.redis_url)
            except Exception as exc:
                logger.warning("Redis unavailable (%s) — using in-memory store", exc)
                self._redis = None
        else:
            logger.info("SessionStore: using in-memory store")

    async def load(self, session_id: str) -> Optional[Conversation]:
        key = f"session:{session_id}"
        try:
            if self._redis:
                data = await self._redis.get(key)
            else:
                data = _MEMORY_STORE.get(key)

            if not data:
                return None
            return Conversation.model_validate_json(data)
        except Exception as exc:
            logger.error("Failed to load session %s: %s", session_id, exc)
            return None

    async def save(self, conversation: Conversation) -> None:
        key = f"session:{conversation.session_id}"
        try:
            data = conversation.model_dump_json()
            if self._redis:
                await self._redis.setex(key, self.ttl, data)
            else:
                _MEMORY_STORE[key] = data
        except Exception as exc:
            logger.error("Failed to save session %s: %s", conversation.session_id, exc)

    async def delete(self, session_id: str) -> None:
        key = f"session:{session_id}"
        if self._redis:
            await self._redis.delete(key)
        else:
            _MEMORY_STORE.pop(key, None)

    async def exists(self, session_id: str) -> bool:
        key = f"session:{session_id}"
        if self._redis:
            return bool(await self._redis.exists(key))
        return key in _MEMORY_STORE

    async def close(self) -> None:
        if self._redis:
            await self._redis.close()
