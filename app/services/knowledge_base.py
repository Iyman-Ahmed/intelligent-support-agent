"""
Knowledge Base Service — semantic search over FAQs, docs, and policies.
Supports pgvector (PostgreSQL) or Pinecone as the vector store.
Falls back to in-memory mock data for local development.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mock knowledge base entries (used when no vector DB is configured)
# ---------------------------------------------------------------------------

MOCK_KB: List[Dict[str, Any]] = [
    {
        "id": "kb_001",
        "title": "Refund Policy",
        "content": (
            "We offer a full 30-day money-back guarantee on all plans. "
            "To request a refund, contact support with your order ID. "
            "Refunds are processed within 5–7 business days to the original payment method."
        ),
        "category": "billing",
        "tags": ["refund", "money-back", "billing"],
    },
    {
        "id": "kb_002",
        "title": "How to Cancel Your Subscription",
        "content": (
            "You can cancel your subscription at any time from the Account Settings page. "
            "Click on 'Billing' → 'Cancel Subscription'. "
            "Your access continues until the end of the current billing period. "
            "No partial refunds are given for unused time within a billing cycle."
        ),
        "category": "billing",
        "tags": ["cancel", "subscription", "billing"],
    },
    {
        "id": "kb_003",
        "title": "Password Reset",
        "content": (
            "To reset your password, click 'Forgot Password' on the login page. "
            "Enter your email address and we'll send a reset link within 2 minutes. "
            "The reset link expires after 1 hour. "
            "If you don't receive the email, check your spam folder."
        ),
        "category": "account",
        "tags": ["password", "reset", "login", "account"],
    },
    {
        "id": "kb_004",
        "title": "Shipping & Delivery Times",
        "content": (
            "Standard shipping: 5–7 business days. "
            "Express shipping: 2–3 business days. "
            "Overnight shipping: next business day (order before 2pm EST). "
            "Free standard shipping on orders over $50. "
            "International shipping available to 45+ countries."
        ),
        "category": "shipping",
        "tags": ["shipping", "delivery", "order"],
    },
    {
        "id": "kb_005",
        "title": "Upgrading or Downgrading Your Plan",
        "content": (
            "You can change your plan at any time from Account Settings → Billing. "
            "Upgrades take effect immediately and you are charged the prorated difference. "
            "Downgrades take effect at the next billing cycle. "
            "You will not lose any data when downgrading."
        ),
        "category": "billing",
        "tags": ["upgrade", "downgrade", "plan", "billing"],
    },
    {
        "id": "kb_006",
        "title": "Two-Factor Authentication (2FA)",
        "content": (
            "Enable 2FA from Account Settings → Security. "
            "We support authenticator apps (Google Authenticator, Authy) and SMS. "
            "Authenticator apps are recommended for better security. "
            "If you lose access to your 2FA device, use backup codes provided during setup."
        ),
        "category": "security",
        "tags": ["2fa", "security", "authentication"],
    },
    {
        "id": "kb_007",
        "title": "API Rate Limits",
        "content": (
            "Free plan: 100 API calls/day. "
            "Starter plan: 1,000 API calls/day. "
            "Pro plan: 10,000 API calls/day. "
            "Enterprise plan: unlimited (subject to fair use). "
            "Rate limit headers are included in every API response."
        ),
        "category": "technical",
        "tags": ["api", "rate limit", "technical"],
    },
    {
        "id": "kb_008",
        "title": "Data Export",
        "content": (
            "You can export all your data at any time from Account Settings → Data. "
            "Exports are available in JSON and CSV formats. "
            "Large exports may take up to 24 hours to prepare. "
            "You will receive an email when your export is ready."
        ),
        "category": "account",
        "tags": ["export", "data", "gdpr"],
    },
]


# ---------------------------------------------------------------------------
# Knowledge Base Service
# ---------------------------------------------------------------------------

class KnowledgeBaseService:
    """
    Semantic search over the knowledge base.

    In production: uses pgvector or Pinecone.
    In development: uses simple keyword matching on MOCK_KB.
    """

    def __init__(
        self,
        use_vector_db: bool = False,
        pinecone_api_key: Optional[str] = None,
        pinecone_index: Optional[str] = None,
        db_url: Optional[str] = None,
        anthropic_client=None,
    ):
        self.use_vector_db = use_vector_db and (pinecone_api_key or db_url)
        self.pinecone_api_key = pinecone_api_key
        self.pinecone_index = pinecone_index
        self.db_url = db_url
        self.anthropic_client = anthropic_client
        self._pinecone_index = None
        self._db_pool = None

    async def initialize(self) -> None:
        """Initialize the vector DB connection (called at app startup)."""
        if not self.use_vector_db:
            logger.info("KnowledgeBase: using mock data (no vector DB configured)")
            return
        if self.pinecone_api_key:
            await self._init_pinecone()
        elif self.db_url:
            await self._init_pgvector()

    async def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search the knowledge base for documents relevant to the query.
        Returns list of {title, content, score, category} dicts.
        """
        if self.use_vector_db and self._pinecone_index:
            return await self._pinecone_search(query, top_k)
        if self.use_vector_db and self._db_pool:
            return await self._pgvector_search(query, top_k)
        return self._mock_search(query, top_k)

    async def add_document(
        self,
        doc_id: str,
        title: str,
        content: str,
        category: str = "general",
        tags: Optional[List[str]] = None,
    ) -> bool:
        """Add or update a document in the knowledge base."""
        if self.use_vector_db:
            embedding = await self._embed(content)
            if self._pinecone_index:
                self._pinecone_index.upsert([
                    (doc_id, embedding, {"title": title, "content": content, "category": category})
                ])
            return True
        # For mock: just add to in-memory list
        MOCK_KB.append({
            "id": doc_id, "title": title, "content": content,
            "category": category, "tags": tags or []
        })
        return True

    # ------------------------------------------------------------------
    # Mock search (keyword-based)
    # ------------------------------------------------------------------

    def _mock_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        query_lower = query.lower()
        query_words = set(query_lower.split())
        scored = []
        for doc in MOCK_KB:
            score = 0
            text = (doc["title"] + " " + doc["content"] + " " + " ".join(doc.get("tags", []))).lower()
            for word in query_words:
                if len(word) > 2 and word in text:
                    score += text.count(word)
            if score > 0:
                scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "title": doc["title"],
                "content": doc["content"],
                "category": doc.get("category", "general"),
                "score": min(score / 10, 1.0),
            }
            for score, doc in scored[:top_k]
        ]

    # ------------------------------------------------------------------
    # Pinecone
    # ------------------------------------------------------------------

    async def _init_pinecone(self) -> None:
        try:
            import pinecone
            pc = pinecone.Pinecone(api_key=self.pinecone_api_key)
            self._pinecone_index = pc.Index(self.pinecone_index)
            logger.info("KnowledgeBase: Pinecone connected")
        except Exception as exc:
            logger.error("Pinecone init failed: %s", exc)

    async def _pinecone_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        embedding = await self._embed(query)
        results = self._pinecone_index.query(vector=embedding, top_k=top_k, include_metadata=True)
        return [
            {
                "title": m.metadata.get("title", ""),
                "content": m.metadata.get("content", ""),
                "category": m.metadata.get("category", ""),
                "score": m.score,
            }
            for m in results.matches
        ]

    # ------------------------------------------------------------------
    # pgvector
    # ------------------------------------------------------------------

    async def _init_pgvector(self) -> None:
        try:
            import asyncpg
            self._db_pool = await asyncpg.create_pool(self.db_url)
            logger.info("KnowledgeBase: pgvector connected")
        except Exception as exc:
            logger.error("pgvector init failed: %s", exc)

    async def _pgvector_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        embedding = await self._embed(query)
        embedding_str = "[" + ",".join(map(str, embedding)) + "]"
        async with self._db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT title, content, category,
                       1 - (embedding <=> $1::vector) AS score
                FROM knowledge_base
                ORDER BY embedding <=> $1::vector
                LIMIT $2
                """,
                embedding_str, top_k
            )
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # Embedding helper (uses Claude / OpenAI / local model)
    # ------------------------------------------------------------------

    async def _embed(self, text: str) -> List[float]:
        """Generate embeddings. Replace with your preferred embedding model."""
        try:
            import anthropic
            # Anthropic doesn't provide embeddings yet; use OpenAI or a local model
            # Placeholder — replace with actual embedding call
            import hashlib
            # Deterministic mock embedding (1536 dims)
            h = hashlib.sha256(text.encode()).hexdigest()
            return [int(h[i:i+2], 16) / 255.0 for i in range(0, min(len(h), 1536 * 2), 2)]
        except Exception:
            return [0.0] * 1536
