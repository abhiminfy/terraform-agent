# src/app/core/metrics.py
"""
Prometheus metrics wrapper with safe, no-fuss helpers.

Exposes a singleton `metrics` that the app can import.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Optional

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    push_to_gateway,
)

# Create a process-local registry so tests/instances don't collide
_registry = CollectorRegistry(auto_describe=True)

# Counters
request_count = Counter(
    "app_requests_total",
    "Total number of requests",
    ["endpoint"],
    registry=_registry,
)
errors = Counter(
    "app_errors_total",
    "Total number of errors",
    ["type"],
    registry=_registry,
)
model_requests = Counter(
    "model_requests_total",
    "Total number of model requests",
    ["model_name", "status"],
    registry=_registry,
)
model_tokens = Counter(
    "model_tokens_total",
    "Number of tokens sent/received",
    ["model_name", "type"],  # type: prompt|completion
    registry=_registry,
)
tool_executions = Counter(
    "tool_executions_total",
    "How many times a tool ran",
    ["tool_name", "status"],  # status: success|error
    registry=_registry,
)
terraform_plans = Counter(
    "terraform_plans_total",
    "Number of terraform plan executions",
    ["status"],
    registry=_registry,
)
chat_messages = Counter(
    "chat_messages_total",
    "Chat messages by role and category",
    ["role", "category"],
    registry=_registry,
)
http_requests = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
    registry=_registry,
)

# Gauges
token_budget_remaining = Gauge(
    "token_budget_remaining",
    "Remaining token budget (approx)",
    registry=_registry,
)
active_chats = Gauge(
    "active_chats",
    "Number of active chat sessions",
    registry=_registry,
)

# Histograms
model_latency = Histogram(
    "model_latency_seconds",
    "Latency of model calls",
    ["model_name"],
    registry=_registry,
    buckets=(0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10),
)
tool_latency = Histogram(
    "tool_latency_seconds",
    "Latency of tool executions",
    ["tool_name"],
    registry=_registry,
    buckets=(0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5),
)
terraform_plan_latency = Histogram(
    "terraform_plan_latency_seconds",
    "Latency of `terraform plan`",
    registry=_registry,
    buckets=(0.2, 0.5, 1, 2, 5, 10, 20, 60),
)
infra_monthly_cost_usd = Histogram(
    "infra_monthly_cost_usd",
    "Estimated monthly infra cost (USD)",
    ["provider"],  # e.g., infracost|fallback
    registry=_registry,
    buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000, 5000),
)
http_request_duration = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency",
    ["method", "endpoint"],
    registry=_registry,
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
)


class Metrics:
    def __init__(self) -> None:
        self._push_gateway: Optional[str] = os.getenv("PROMETHEUS_PUSHGATEWAY") or None
        self.request_count = request_count
        self.errors = errors
        self.model_requests = model_requests
        self.model_tokens = model_tokens
        self.tool_executions = tool_executions
        self.terraform_plans = terraform_plans
        self.chat_messages = chat_messages
        self.http_requests = http_requests

        self.token_budget_remaining = token_budget_remaining
        self.active_chats = active_chats

        self.model_latency = model_latency
        self.tool_latency = tool_latency
        self.terraform_plan_latency = terraform_plan_latency
        self.infra_monthly_cost_usd = infra_monthly_cost_usd
        self.http_request_duration = http_request_duration

    def _try_push(self) -> None:
        if not self._push_gateway:
            return
        try:
            push_to_gateway(self._push_gateway, job="strands_agent", registry=_registry)
        except Exception:
            # Swallow push errors (metrics are best-effort)
            pass

    # ---------- Convenience helpers ----------
    def record_chat_message(self, role: str, category: str = "general") -> None:
        try:
            self.chat_messages.labels(role=role, category=category).inc()
            self._try_push()
        except Exception:
            pass

    def record_cost_estimation(self, provider: str, monthly_cost: float) -> None:
        try:
            self.infra_monthly_cost_usd.labels(provider=provider).observe(
                max(0.0, float(monthly_cost))
            )
            self._try_push()
        except Exception:
            pass

    def get_metrics(self) -> str:
        """Return Prometheus formatted metrics."""
        try:
            return generate_latest(_registry).decode("utf-8")
        except Exception:
            return ""

    def get_metrics_summary(self) -> dict:
        """Return a summary of current metrics values."""
        try:
            return {
                "http_requests": self._get_counter_value(self.http_requests),
                "chat_messages": self._get_counter_value(self.chat_messages),
                "tool_executions": self._get_counter_value(self.tool_executions),
                "terraform_plans": self._get_counter_value(self.terraform_plans),
                "errors": self._get_counter_value(self.errors),
                "active_chats": self.active_chats._value._value,
            }
        except Exception:
            return {"error": "Failed to get metrics summary"}

    def _get_counter_value(self, counter) -> dict:
        """Helper to get counter values by labels."""
        try:
            return {str(sample.labels): sample.value for sample in counter.collect()[0].samples}
        except Exception:
            return {}

    # ---------- HTTP Request tracking ----------
    def track_request(self, method: str, endpoint: str):
        """
        Decorator to track HTTP requests with timing and status.

        Usage:
            @metrics.track_request("GET", "/api/endpoint")
            def my_endpoint():
                ...
        """

        def _decorator(func):
            def _wrapped(*args, **kwargs):
                start_time = time.perf_counter()
                status = "success"

                try:
                    # Increment request counter
                    self.request_count.labels(endpoint=endpoint).inc()

                    # Execute the function
                    result = func(*args, **kwargs)

                    return result

                except Exception as e:
                    status = "error"
                    # Record the error
                    try:
                        self.errors.labels(type=type(e).__name__).inc()
                    except Exception:
                        pass
                    raise

                finally:
                    # Record duration and final status
                    try:
                        duration = time.perf_counter() - start_time
                        self.http_request_duration.labels(method=method, endpoint=endpoint).observe(
                            duration
                        )
                        self.http_requests.labels(
                            method=method, endpoint=endpoint, status=status
                        ).inc()
                        self._try_push()
                    except Exception:
                        pass

            return _wrapped

        return _decorator

    # ---------- Decorators & timers ----------
    def track_tool(self, tool_name: str):
        """
        Decorator to time and count tool executions.

        Usage:
            @metrics.track_tool("terraform_timeout")
            def run_with_timeout(...):
                ...
        """

        def _decorator(func):
            def _wrapped(*args, **kwargs):
                start = time.perf_counter()
                try:
                    rv = func(*args, **kwargs)
                    dur = time.perf_counter() - start
                    try:
                        self.tool_latency.labels(tool_name=tool_name).observe(dur)
                        self.tool_executions.labels(tool_name=tool_name, status="success").inc()
                        self._try_push()
                    except Exception:
                        pass
                    return rv
                except Exception:
                    dur = time.perf_counter() - start
                    try:
                        self.tool_latency.labels(tool_name=tool_name).observe(dur)
                        self.tool_executions.labels(tool_name=tool_name, status="error").inc()
                        self._try_push()
                    except Exception:
                        pass
                    raise

            return _wrapped

        return _decorator

    @contextmanager
    def tool_timer(self, tool_name: str):
        """
        Context manager to time arbitrary tool runs.

        Usage:
            with metrics.tool_timer("tfsec"):
                ... external call ...
        """
        start = time.perf_counter()
        try:
            yield
            status = "success"
        except Exception:
            status = "error"
            raise
        finally:
            dur = time.perf_counter() - start
            try:
                self.tool_latency.labels(tool_name=tool_name).observe(dur)
                self.tool_executions.labels(tool_name=tool_name, status=status).inc()
                self._try_push()
            except Exception:
                pass


# Export singleton
metrics = Metrics()
