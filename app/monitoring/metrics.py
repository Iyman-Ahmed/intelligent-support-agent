"""
Metrics — custom KPI tracking for the support agent.
Tracks: response time, FCR, escalation rate, tool success rate, CSAT.
Exposes a /metrics endpoint compatible with Prometheus.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ConversationMetric:
    session_id: str
    channel: str
    start_time: float
    end_time: Optional[float] = None
    turn_count: int = 0
    escalated: bool = False
    resolved: bool = False
    ticket_created: bool = False
    tool_calls: int = 0
    tool_failures: int = 0
    final_status: Optional[str] = None
    csat_score: Optional[float] = None


class MetricsCollector:
    """
    In-memory metrics store with rolling windows.
    For production: push to Prometheus / Datadog / TimescaleDB.
    """

    def __init__(self, window_minutes: int = 60):
        self.window = timedelta(minutes=window_minutes)

        # Rolling response times (deque of float ms values)
        self._response_times: Deque[tuple] = deque()    # (timestamp, ms)

        # Conversation registry
        self._conversations: Dict[str, ConversationMetric] = {}

        # Running counters
        self._total_conversations: int = 0
        self._total_escalations: int = 0
        self._total_resolutions: int = 0
        self._total_tool_calls: int = 0
        self._total_tool_failures: int = 0
        self._csat_scores: Deque[float] = deque()

        # Per-tool latency tracking
        self._tool_latencies: Dict[str, Deque[float]] = defaultdict(deque)

        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Recording methods (called by agent & API layer)
    # ------------------------------------------------------------------

    def start_conversation(self, session_id: str, channel: str) -> None:
        self._conversations[session_id] = ConversationMetric(
            session_id=session_id,
            channel=channel,
            start_time=time.monotonic(),
        )
        self._total_conversations += 1

    def record_response_time(self, ms: float) -> None:
        now = time.time()
        self._response_times.append((now, ms))
        self._prune_response_times()

    def record_tool_call(
        self,
        tool_name: str,
        latency_ms: float,
        success: bool,
        session_id: Optional[str] = None,
    ) -> None:
        self._total_tool_calls += 1
        if not success:
            self._total_tool_failures += 1
        self._tool_latencies[tool_name].append(latency_ms)
        if len(self._tool_latencies[tool_name]) > 1000:
            self._tool_latencies[tool_name].popleft()

        if session_id and session_id in self._conversations:
            self._conversations[session_id].tool_calls += 1
            if not success:
                self._conversations[session_id].tool_failures += 1

    def record_escalation(self, session_id: str) -> None:
        self._total_escalations += 1
        if session_id in self._conversations:
            self._conversations[session_id].escalated = True

    def record_resolution(self, session_id: str) -> None:
        self._total_resolutions += 1
        if session_id in self._conversations:
            conv = self._conversations[session_id]
            conv.resolved = True
            conv.end_time = time.monotonic()

    def record_csat(self, session_id: str, score: float) -> None:
        self._csat_scores.append(score)
        if len(self._csat_scores) > 10000:
            self._csat_scores.popleft()
        if session_id in self._conversations:
            self._conversations[session_id].csat_score = score

    def record_conversation_status(self, status: str) -> None:
        """Called after each agent response to track current statuses."""
        pass   # Aggregated from conversation registry

    # ------------------------------------------------------------------
    # Computed KPIs
    # ------------------------------------------------------------------

    def avg_response_time_ms(self) -> float:
        """Average response time over the rolling window."""
        self._prune_response_times()
        if not self._response_times:
            return 0.0
        return sum(ms for _, ms in self._response_times) / len(self._response_times)

    def p95_response_time_ms(self) -> float:
        """95th percentile response time."""
        self._prune_response_times()
        if not self._response_times:
            return 0.0
        times = sorted(ms for _, ms in self._response_times)
        idx = int(len(times) * 0.95)
        return times[min(idx, len(times) - 1)]

    def escalation_rate(self) -> float:
        """Percentage of conversations escalated to human."""
        if not self._total_conversations:
            return 0.0
        return round(self._total_escalations / self._total_conversations * 100, 2)

    def first_contact_resolution_rate(self) -> float:
        """Percentage of conversations resolved without escalation."""
        if not self._total_conversations:
            return 0.0
        fcr = sum(
            1 for c in self._conversations.values()
            if c.resolved and not c.escalated
        )
        return round(fcr / self._total_conversations * 100, 2)

    def tool_success_rate(self) -> float:
        """Percentage of tool calls that succeeded."""
        if not self._total_tool_calls:
            return 100.0
        failures = self._total_tool_failures
        return round((1 - failures / self._total_tool_calls) * 100, 2)

    def avg_csat(self) -> float:
        """Average CSAT score (1–5)."""
        if not self._csat_scores:
            return 0.0
        return round(sum(self._csat_scores) / len(self._csat_scores), 2)

    def per_tool_avg_latency(self) -> Dict[str, float]:
        return {
            tool: round(sum(lats) / len(lats), 2)
            for tool, lats in self._tool_latencies.items()
            if lats
        }

    def summary(self) -> Dict[str, Any]:
        """Full KPI summary dict — returned by /metrics/summary endpoint."""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "kpis": {
                "avg_response_time_ms": self.avg_response_time_ms(),
                "p95_response_time_ms": self.p95_response_time_ms(),
                "escalation_rate_pct": self.escalation_rate(),
                "fcr_rate_pct": self.first_contact_resolution_rate(),
                "tool_success_rate_pct": self.tool_success_rate(),
                "avg_csat_score": self.avg_csat(),
            },
            "totals": {
                "conversations": self._total_conversations,
                "escalations": self._total_escalations,
                "resolutions": self._total_resolutions,
                "tool_calls": self._total_tool_calls,
                "tool_failures": self._total_tool_failures,
                "csat_responses": len(self._csat_scores),
            },
            "tool_latencies_ms": self.per_tool_avg_latency(),
        }

    def prometheus_export(self) -> str:
        """Export metrics in Prometheus text format."""
        s = self.summary()
        kpis = s["kpis"]
        totals = s["totals"]
        lines = [
            "# HELP support_agent_response_time_ms Average response time",
            "# TYPE support_agent_response_time_ms gauge",
            f"support_agent_response_time_ms {kpis['avg_response_time_ms']}",
            "",
            "# HELP support_agent_escalation_rate Escalation rate %",
            "# TYPE support_agent_escalation_rate gauge",
            f"support_agent_escalation_rate {kpis['escalation_rate_pct']}",
            "",
            "# HELP support_agent_fcr_rate First contact resolution rate %",
            "# TYPE support_agent_fcr_rate gauge",
            f"support_agent_fcr_rate {kpis['fcr_rate_pct']}",
            "",
            "# HELP support_agent_tool_success_rate Tool success rate %",
            "# TYPE support_agent_tool_success_rate gauge",
            f"support_agent_tool_success_rate {kpis['tool_success_rate_pct']}",
            "",
            "# HELP support_agent_conversations_total Total conversations",
            "# TYPE support_agent_conversations_total counter",
            f"support_agent_conversations_total {totals['conversations']}",
            "",
            "# HELP support_agent_csat_score Average CSAT score",
            "# TYPE support_agent_csat_score gauge",
            f"support_agent_csat_score {kpis['avg_csat_score']}",
        ]
        return "\n".join(lines)

    def _prune_response_times(self) -> None:
        cutoff = time.time() - self.window.total_seconds()
        while self._response_times and self._response_times[0][0] < cutoff:
            self._response_times.popleft()
