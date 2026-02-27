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

# ---------------------------------------------------------------------------
# TechFlow Solutions — mock knowledge base
#
# TechFlow Solutions is a SaaS company offering two products:
#   • TechFlow Analytics  — real-time business intelligence & dashboards
#   • TechFlow Projects   — agile project management & team collaboration
#
# Plans:  Free | Starter ($29/mo) | Pro ($79/mo) | Enterprise ($199/mo)
# ---------------------------------------------------------------------------

MOCK_KB: List[Dict[str, Any]] = [
    # ── Billing & Refunds ────────────────────────────────────────────────
    {
        "id": "kb_001",
        "title": "Refund Policy",
        "content": (
            "TechFlow Solutions offers a full 30-day money-back guarantee on all paid plans "
            "(Starter, Pro, and Enterprise). "
            "To request a refund, contact support at support@techflow.io with your account email "
            "and order/invoice ID. "
            "Refunds are processed within 5–7 business days to the original payment method. "
            "After 30 days, refunds are not available except in cases of billing errors or "
            "service outages lasting more than 24 hours. "
            "Annual plan refunds are prorated for the unused months if cancelled within 60 days."
        ),
        "category": "billing",
        "tags": ["refund", "money-back", "billing", "guarantee", "30-day"],
    },
    {
        "id": "kb_002",
        "title": "Subscription Plans & Pricing",
        "content": (
            "TechFlow Solutions offers four plans: "
            "Free ($0/mo): up to 3 users, 5 projects, 100 API calls/day, community support. "
            "Starter ($29/mo): up to 10 users, unlimited projects, 1,000 API calls/day, email support, "
            "basic analytics. "
            "Pro ($79/mo): up to 50 users, unlimited projects, 10,000 API calls/day, priority support, "
            "advanced analytics, custom dashboards, integrations with Slack/Jira/GitHub. "
            "Enterprise ($199/mo): unlimited users, unlimited API calls, dedicated account manager, "
            "SSO/SAML, custom SLA (99.99% uptime), on-premise deployment option, 24/7 phone support. "
            "All plans are billed monthly or annually (annual = 2 months free). "
            "Prices are in USD and exclude applicable taxes."
        ),
        "category": "billing",
        "tags": ["pricing", "plans", "starter", "pro", "enterprise", "free", "cost", "subscription"],
    },
    {
        "id": "kb_003",
        "title": "How to Cancel Your Subscription",
        "content": (
            "You can cancel your TechFlow subscription at any time — no cancellation fees. "
            "Go to Account Settings → Billing → Cancel Subscription and follow the prompts. "
            "Your access continues until the end of the current billing period; "
            "you will not be charged again after cancellation. "
            "No partial refunds are given for unused days within a billing cycle. "
            "Your data is retained for 90 days after cancellation so you can export it. "
            "After 90 days, all data is permanently deleted. "
            "To reactivate, simply choose a new plan — your historical data will be restored "
            "if you reactivate within the 90-day window."
        ),
        "category": "billing",
        "tags": ["cancel", "subscription", "billing", "terminate", "stop"],
    },
    {
        "id": "kb_004",
        "title": "Upgrading or Downgrading Your Plan",
        "content": (
            "You can change your plan at any time from Account Settings → Billing → Change Plan. "
            "Upgrades take effect immediately; you are charged the prorated difference for the "
            "remaining days in the current cycle. "
            "Downgrades take effect at the start of the next billing cycle. "
            "You will not lose any data when downgrading, but features unavailable on the lower plan "
            "will become read-only. "
            "To upgrade from Free to Starter or higher, a credit or debit card is required."
        ),
        "category": "billing",
        "tags": ["upgrade", "downgrade", "plan", "billing", "change plan"],
    },
    {
        "id": "kb_005",
        "title": "Billing & Invoice FAQ",
        "content": (
            "Invoices are emailed automatically on each billing date and available under "
            "Account Settings → Billing → Invoice History. "
            "We accept Visa, Mastercard, American Express, and PayPal. "
            "For Enterprise plans, we also accept bank wire transfers and purchase orders. "
            "To update your payment method, go to Account Settings → Billing → Payment Method. "
            "If a payment fails, we retry 3 times over 7 days and send email reminders. "
            "After 7 days of failed payment, the account is downgraded to the Free plan. "
            "All prices are in USD. VAT/GST is added for customers in applicable regions."
        ),
        "category": "billing",
        "tags": ["invoice", "payment", "billing", "credit card", "receipt"],
    },

    # ── Account & Security ───────────────────────────────────────────────
    {
        "id": "kb_006",
        "title": "Password Reset",
        "content": (
            "To reset your TechFlow password, click 'Forgot Password' on the login page at "
            "app.techflow.io/login. "
            "Enter your account email address and we will send a reset link within 2 minutes. "
            "The reset link is valid for 1 hour and can only be used once. "
            "If you do not receive the email within 5 minutes, check your spam/junk folder "
            "or try adding support@techflow.io to your contacts. "
            "If your account uses SSO (Single Sign-On), password reset is managed by your "
            "IT administrator — contact them directly."
        ),
        "category": "account",
        "tags": ["password", "reset", "login", "account", "forgot password"],
    },
    {
        "id": "kb_007",
        "title": "Two-Factor Authentication (2FA)",
        "content": (
            "Two-factor authentication (2FA) is available on all paid plans and strongly recommended. "
            "To enable: Account Settings → Security → Enable 2FA. "
            "Supported methods: authenticator apps (Google Authenticator, Authy, 1Password) and SMS. "
            "Authenticator apps are more secure and recommended over SMS. "
            "During setup, save your 8 backup codes in a safe place — each can be used once "
            "if you lose access to your 2FA device. "
            "To disable 2FA or recover a locked account, email security@techflow.io with your "
            "account email and a government-issued ID for verification."
        ),
        "category": "security",
        "tags": ["2fa", "two-factor", "security", "authentication", "mfa"],
    },
    {
        "id": "kb_008",
        "title": "Account Login Issues",
        "content": (
            "If you cannot log in to your TechFlow account, try these steps in order: "
            "1. Check you are using the correct email address. "
            "2. Clear your browser cache and cookies, or try an incognito/private window. "
            "3. Use 'Forgot Password' to reset your password. "
            "4. If your account uses Google/Microsoft SSO, try logging in via your SSO provider. "
            "5. If you see 'Account suspended', your subscription may have lapsed — check your "
            "billing email for payment failure notices. "
            "6. After 10 failed login attempts, your account is temporarily locked for 30 minutes "
            "as a security measure. "
            "If none of the above resolves the issue, contact support@techflow.io."
        ),
        "category": "account",
        "tags": ["login", "can't log in", "access", "account", "locked", "suspended"],
    },
    {
        "id": "kb_009",
        "title": "Data Export & GDPR",
        "content": (
            "You can export all your TechFlow data at any time from Account Settings → Data → Export. "
            "Exports include projects, tasks, analytics data, and team activity logs. "
            "Available formats: JSON, CSV, and Excel (.xlsx). "
            "Large exports (>500 MB) may take up to 24 hours; you will receive an email when ready. "
            "Under GDPR/CCPA, you have the right to request deletion of all your personal data. "
            "To submit a data deletion request, email privacy@techflow.io from your account email. "
            "Deletion is completed within 30 days and you will receive a confirmation."
        ),
        "category": "account",
        "tags": ["export", "data", "gdpr", "ccpa", "privacy", "delete data"],
    },

    # ── Product & Technical ──────────────────────────────────────────────
    {
        "id": "kb_010",
        "title": "API Rate Limits & Usage",
        "content": (
            "TechFlow API rate limits by plan: "
            "Free: 100 calls/day (no per-minute limit). "
            "Starter: 1,000 calls/day, 60 calls/minute. "
            "Pro: 10,000 calls/day, 300 calls/minute. "
            "Enterprise: unlimited (subject to fair-use policy, ~1M calls/day). "
            "All API responses include rate-limit headers: X-RateLimit-Limit, "
            "X-RateLimit-Remaining, X-RateLimit-Reset. "
            "If you exceed limits, the API returns HTTP 429 with a Retry-After header. "
            "To increase limits, upgrade your plan or contact enterprise@techflow.io for "
            "a custom quota."
        ),
        "category": "technical",
        "tags": ["api", "rate limit", "technical", "developer", "quota"],
    },
    {
        "id": "kb_011",
        "title": "Integrations — Slack, Jira, GitHub, Zapier",
        "content": (
            "TechFlow Pro and Enterprise plans include native integrations: "
            "Slack: receive project updates, task assignments, and alerts in Slack channels. "
            "Set up via Account Settings → Integrations → Slack. "
            "Jira: two-way sync of issues and tasks between TechFlow Projects and Jira. "
            "GitHub: link pull requests and commits to TechFlow tasks automatically. "
            "Zapier: connect TechFlow to 5,000+ apps via Zapier (Starter plan and above). "
            "Webhooks: all plans support outbound webhooks for custom integrations. "
            "For integration setup guides, visit docs.techflow.io/integrations."
        ),
        "category": "technical",
        "tags": ["integration", "slack", "jira", "github", "zapier", "webhook"],
    },
    {
        "id": "kb_012",
        "title": "TechFlow Analytics — Features Overview",
        "content": (
            "TechFlow Analytics provides real-time business intelligence for your data: "
            "• Connect data sources: PostgreSQL, MySQL, BigQuery, Snowflake, CSV upload, REST API. "
            "• Build interactive dashboards with 30+ chart types (line, bar, funnel, heatmap, etc.). "
            "• Schedule automated reports delivered by email or Slack. "
            "• Set threshold alerts — get notified when a metric crosses a defined value. "
            "• AI-powered anomaly detection (Pro and Enterprise). "
            "• Share dashboards publicly via link or embed in your product. "
            "Free plan: 1 dashboard, 1 data source. "
            "Starter: 5 dashboards, 3 data sources. "
            "Pro: unlimited dashboards and sources."
        ),
        "category": "product",
        "tags": ["analytics", "dashboard", "reports", "charts", "features", "product"],
    },
    {
        "id": "kb_013",
        "title": "TechFlow Projects — Features Overview",
        "content": (
            "TechFlow Projects is an agile project management tool: "
            "• Kanban boards, Gantt charts, and sprint planning views. "
            "• Task assignments, due dates, priorities, and labels. "
            "• Time tracking and workload management. "
            "• File attachments (up to 250 MB per file on Pro; 50 MB on Starter). "
            "• Recurring tasks and task dependencies. "
            "• Team chat and inline comments on tasks. "
            "• Guest access — invite clients or contractors without a paid seat. "
            "Free plan: 3 projects, 5 members. "
            "Starter: unlimited projects, 10 members. "
            "Pro: unlimited projects, 50 members, advanced reporting."
        ),
        "category": "product",
        "tags": ["projects", "kanban", "gantt", "tasks", "agile", "features", "product"],
    },

    # ── Shipping & Orders ────────────────────────────────────────────────
    {
        "id": "kb_014",
        "title": "Shipping & Delivery (Hardware Add-ons)",
        "content": (
            "TechFlow sells optional hardware add-ons (TechFlow Sync Dongle, printed onboarding kits). "
            "Shipping options within the US: "
            "Standard (5–7 business days): free on orders over $50, else $5.99. "
            "Express (2–3 business days): $12.99. "
            "Overnight (next business day, order before 2 pm EST): $24.99. "
            "International shipping: available to 45+ countries, $19.99 flat rate, 7–14 business days. "
            "You will receive a tracking number by email once your order ships. "
            "For damaged or missing shipments, contact support within 14 days of the delivery date."
        ),
        "category": "shipping",
        "tags": ["shipping", "delivery", "order", "hardware", "dongle", "tracking"],
    },
    {
        "id": "kb_015",
        "title": "Technical Support & SLA",
        "content": (
            "TechFlow support tiers: "
            "Free plan: community forum only (forum.techflow.io). "
            "Starter: email support, response within 2 business days. "
            "Pro: priority email support, response within 4 business hours; live chat Mon–Fri 9am–6pm EST. "
            "Enterprise: 24/7 phone and email support, dedicated account manager, "
            "1-hour response SLA for P1 (critical) issues, 4-hour SLA for P2 (high). "
            "For all plans, the status page is at status.techflow.io. "
            "To open a support ticket: email support@techflow.io or use the in-app chat widget."
        ),
        "category": "support",
        "tags": ["support", "help", "sla", "response time", "contact", "ticket"],
    },
    {
        "id": "kb_016",
        "title": "Frequently Asked Questions (General)",
        "content": (
            "Q: Is there a free trial for paid plans? "
            "A: Yes — all paid plans come with a 14-day free trial, no credit card required. "
            "Q: Can I use TechFlow offline? "
            "A: TechFlow is cloud-based. A PWA (progressive web app) with limited offline support "
            "is available for Pro and Enterprise users. "
            "Q: Is my data secure? "
            "A: Yes. TechFlow is SOC 2 Type II certified. All data is encrypted at rest (AES-256) "
            "and in transit (TLS 1.3). Servers are hosted on AWS (US-East, EU-West). "
            "Q: Can I import data from another tool? "
            "A: Yes — we support imports from Trello, Asana, Monday.com, Jira, and CSV. "
            "Q: What happens if TechFlow has downtime? "
            "A: We issue service credits automatically for outages exceeding our SLA. "
            "Check status.techflow.io for real-time status."
        ),
        "category": "general",
        "tags": ["faq", "trial", "free trial", "offline", "security", "import", "downtime"],
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
