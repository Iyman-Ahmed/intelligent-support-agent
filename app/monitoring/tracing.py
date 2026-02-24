"""
OpenTelemetry tracing setup.
Provides distributed tracing across API → Agent → Tool calls.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Try to import OpenTelemetry; gracefully degrade if not installed
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    logger.info("OpenTelemetry not installed — tracing disabled")


# ---------------------------------------------------------------------------
# Tracer setup
# ---------------------------------------------------------------------------

_tracer = None


def setup_tracing(
    service_name: str = "intelligent-support-agent",
    otlp_endpoint: Optional[str] = None,
    datadog_enabled: bool = False,
) -> None:
    """
    Initialise the global tracer.
    Call this once at application startup.
    """
    global _tracer

    if not OTEL_AVAILABLE:
        _tracer = _NoopTracer()
        return

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    if otlp_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
            provider.add_span_processor(BatchSpanProcessor(exporter))
            logger.info("Tracing: OTLP exporter → %s", otlp_endpoint)
        except ImportError:
            logger.warning("OTLP exporter not available")

    if datadog_enabled:
        try:
            from opentelemetry.exporter.datadog import DatadogSpanExporter
            exporter = DatadogSpanExporter(service=service_name)
            provider.add_span_processor(BatchSpanProcessor(exporter))
            logger.info("Tracing: Datadog exporter enabled")
        except ImportError:
            logger.warning("Datadog exporter not available")

    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer(service_name)
    logger.info("Tracing: OpenTelemetry initialised for '%s'", service_name)


def get_tracer():
    global _tracer
    if _tracer is None:
        _tracer = _NoopTracer()
    return _tracer


# ---------------------------------------------------------------------------
# Span helpers
# ---------------------------------------------------------------------------

@contextmanager
def agent_span(operation: str, attributes: Optional[Dict[str, Any]] = None):
    """Context manager for tracing an agent operation."""
    tracer = get_tracer()
    with tracer.start_as_current_span(f"agent.{operation}") as span:
        if attributes:
            for k, v in attributes.items():
                span.set_attribute(k, str(v))
        try:
            yield span
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(exc)) if OTEL_AVAILABLE else None)
            raise


@contextmanager
def tool_span(tool_name: str, session_id: Optional[str] = None):
    """Context manager for tracing a tool call."""
    tracer = get_tracer()
    with tracer.start_as_current_span(f"tool.{tool_name}") as span:
        if session_id:
            span.set_attribute("session_id", session_id)
        span.set_attribute("tool.name", tool_name)
        try:
            yield span
        except Exception as exc:
            span.record_exception(exc)
            raise


@contextmanager
def api_span(endpoint: str, method: str = "POST"):
    """Context manager for tracing an API request."""
    tracer = get_tracer()
    with tracer.start_as_current_span(f"api.{endpoint}") as span:
        span.set_attribute("http.method", method)
        span.set_attribute("http.route", endpoint)
        try:
            yield span
        except Exception as exc:
            span.record_exception(exc)
            raise


# ---------------------------------------------------------------------------
# No-op tracer (when OTEL not installed)
# ---------------------------------------------------------------------------

class _NoopSpan:
    def set_attribute(self, *a, **kw): pass
    def record_exception(self, *a, **kw): pass
    def set_status(self, *a, **kw): pass
    def __enter__(self): return self
    def __exit__(self, *a): pass


class _NoopTracer:
    def start_as_current_span(self, name, **kw):
        return _NoopSpan()
