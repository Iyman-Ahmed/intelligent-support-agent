"""
FastAPI application entrypoint.

Startup sequence:
  1. Load config
  2. Initialise services (DB, Redis, email, knowledge base)
  3. Wire up agent core with all services
  4. Register API routers
  5. Start serving
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any

import anthropic
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

from app.agent.core import SupportAgentCore
from app.agent.escalation import EscalationEngine
from app.agent.tools import ToolDispatcher
from app.api import chat_router, email_router, tickets_router
from app.config import settings
from app.monitoring.metrics import MetricsCollector
from app.monitoring.tracing import setup_tracing
from app.session_store import SessionStore
from app.services import (
    CustomerDBService,
    EmailService,
    KnowledgeBaseService,
    TicketService,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Application lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting %s", settings.app_name)

    # 1. Tracing
    setup_tracing(
        service_name="intelligent-support-agent",
        otlp_endpoint=settings.otlp_endpoint,
        datadog_enabled=settings.datadog_enabled,
    )

    # 2. Anthropic client
    anthropic_client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

    # 3. Services
    knowledge_base = KnowledgeBaseService(
        use_vector_db=bool(settings.pinecone_api_key or settings.database_url),
        pinecone_api_key=settings.pinecone_api_key,
        pinecone_index=settings.pinecone_index,
        db_url=settings.database_url,
        anthropic_client=anthropic_client,
    )
    await knowledge_base.initialize()

    customer_service = CustomerDBService(db_url=settings.database_url)
    await customer_service.initialize()

    ticket_service = TicketService(
        db_url=settings.database_url,
        zendesk_api_key=settings.zendesk_api_key,
        zendesk_subdomain=settings.zendesk_subdomain,
    )
    await ticket_service.initialize()

    email_service = EmailService(
        sendgrid_api_key=settings.sendgrid_api_key,
        mailgun_api_key=settings.mailgun_api_key,
        mailgun_domain=settings.mailgun_domain,
        from_email=settings.support_from_email,
        from_name=settings.support_from_name,
    )

    # 4. Session store
    session_store = SessionStore(redis_url=settings.redis_url)
    await session_store.initialize()

    # 5. Metrics
    metrics = MetricsCollector()

    # 6. Wire up agent
    dispatcher = ToolDispatcher(
        knowledge_base_service=knowledge_base,
        customer_service=customer_service,
        ticket_service=ticket_service,
        email_service=email_service,
    )
    escalation_engine = EscalationEngine(anthropic_client=anthropic_client)
    agent = SupportAgentCore(
        anthropic_client=anthropic_client,
        tool_dispatcher=dispatcher,
        escalation_engine=escalation_engine,
        session_store=session_store,
    )

    # 7. Store on app state (accessible from route handlers)
    app.state.agent = agent
    app.state.session_store = session_store
    app.state.ticket_service = ticket_service
    app.state.email_service = email_service
    app.state.metrics = metrics
    app.state.config = settings

    logger.info("✅ All services initialised — ready to serve")

    yield   # Application runs here

    # Shutdown
    logger.info("🛑 Shutting down...")
    await session_store.close()
    await anthropic_client.close()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title=settings.app_name,
    description=(
        "AI-powered customer support agent with multi-turn conversations, "
        "tool use, escalation, and real-time monitoring."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Auth middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """Simple API key authentication. Skip for health/metrics endpoints."""
    public_paths = {"/health", "/metrics", "/docs", "/redoc", "/openapi.json"}
    if request.url.path in public_paths:
        return await call_next(request)

    # Also skip WebSocket upgrades (auth handled in WebSocket handler)
    if request.headers.get("upgrade", "").lower() == "websocket":
        return await call_next(request)

    api_key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
    if settings.api_key and api_key != settings.api_key:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"detail": "Invalid or missing API key"}
        )
    return await call_next(request)


# ---------------------------------------------------------------------------
# Request timing middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    start = time.monotonic()
    response = await call_next(request)
    elapsed = round((time.monotonic() - start) * 1000, 2)
    response.headers["X-Response-Time-Ms"] = str(elapsed)
    return response


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(chat_router)
app.include_router(email_router)
app.include_router(tickets_router)


# ---------------------------------------------------------------------------
# Health & Metrics endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["system"])
async def health_check(request: Request):
    return {
        "status": "ok",
        "service": settings.app_name,
        "version": "1.0.0",
    }


@app.get("/metrics", tags=["system"], response_class=PlainTextResponse)
async def prometheus_metrics(request: Request):
    """Prometheus-compatible metrics endpoint."""
    metrics: MetricsCollector = request.app.state.metrics
    return metrics.prometheus_export()


@app.get("/metrics/summary", tags=["system"])
async def metrics_summary(request: Request):
    """Human-readable KPI summary."""
    metrics: MetricsCollector = request.app.state.metrics
    return metrics.summary()


@app.post("/metrics/csat", tags=["system"])
async def submit_csat(
    session_id: str,
    score: float,
    request: Request,
):
    """Submit a CSAT score for a completed conversation."""
    if not 1 <= score <= 5:
        raise HTTPException(status_code=400, detail="Score must be between 1 and 5")
    metrics: MetricsCollector = request.app.state.metrics
    metrics.record_csat(session_id, score)
    return {"message": "CSAT recorded", "session_id": session_id, "score": score}


# ---------------------------------------------------------------------------
# Global error handler
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred. Please try again."}
    )


# ---------------------------------------------------------------------------
# Entry point for local development
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info",
    )
