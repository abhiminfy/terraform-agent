# -*- coding: utf-8 -*-
# metrics.py - Prometheus metrics implementation
import time
import logging
from functools import wraps
from typing import Dict, Any

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    CollectorRegistry,
    generate_latest,
)
from prometheus_client.exposition import push_to_gateway

from backend.app.core.config import Settings

logger = logging.getLogger(__name__)
settings = Settings()


class MetricsCollector:
    """Prometheus metrics collector for the Terraform agent"""

    def __init__(self):
        self.registry = CollectorRegistry()

        # Request metrics
        self.request_count = Counter(
            'terraform_agent_requests_total',
            'Total number of requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )

        self.request_duration = Histogram(
            'terraform_agent_request_duration_seconds',
            'Request duration in seconds',
            ['method', 'endpoint'],
            buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0],
            registry=self.registry
        )

        # Tool metrics
        self.tool_execution_count = Counter(
            'terraform_agent_tool_executions_total',
            'Total tool executions',
            ['tool_name', 'status'],
            registry=self.registry
        )

        self.tool_execution_duration = Histogram(
            'terraform_agent_tool_duration_seconds',
            'Tool execution duration',
            ['tool_name'],
            buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0],
            registry=self.registry
        )

        # Model metrics
        self.model_requests = Counter(
            'terraform_agent_model_requests_total',
            'Total model requests',
            ['model_name', 'status'],
            registry=self.registry
        )

        self.model_latency = Histogram(
            'terraform_agent_model_latency_seconds',
            'Model response latency',
            ['model_name'],
            buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0],
            registry=self.registry
        )

        self.model_tokens = Counter(
            'terraform_agent_tokens_total',
            'Total tokens processed',
            ['model_name', 'type'],
            registry=self.registry
        )

        # NEW: Token budget remaining
        self.token_budget_remaining = Gauge(
            'terraform_agent_token_budget_remaining',
            'Remaining token budget',
            registry=self.registry
        )
        try:
            self.token_budget_remaining.set(float(settings.TOKEN_BUDGET))
        except Exception:
            self.token_budget_remaining.set(100000.0)

        # Terraform metrics
        self.terraform_plans = Counter(
            'terraform_agent_plans_total',
            'Total terraform plans',
            ['status'],
            registry=self.registry
        )

        self.terraform_plan_duration = Histogram(
            'terraform_agent_plan_duration_seconds',
            'Terraform plan duration',
            buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 120.0],
            registry=self.registry
        )

        # Error metrics
        self.errors = Counter(
            'terraform_agent_errors_total',
            'Total errors by type',
            ['error_type', 'endpoint'],
            registry=self.registry
        )

        # Active connections
        self.chat_messages = Counter(
            'terraform_agent_chat_messages_total',
            'Total chat messages',
            ['role', 'chat_type'],
            registry=self.registry
        )

        self.chat_sessions = Gauge(
            'terraform_agent_chat_sessions_active',
            'Active chat sessions',
            registry=self.registry
        )

        self.cost_estimations = Counter(
            'terraform_agent_cost_estimations_total',
            'Total cost estimations',
            ['method'],
            registry=self.registry
        )

        self.estimated_costs = Gauge(
            'terraform_agent_estimated_costs_usd',
            'Estimated costs in USD',
            registry=self.registry
        )

    # ---- Pushgateway helper (non-fatal) ----
    def _try_push(self, job: str):
        if settings.PROM_PUSHGATEWAY:
            try:
                push_to_gateway(settings.PROM_PUSHGATEWAY, job=job, registry=self.registry)
            except Exception as e:
                logger.debug(f"Pushgateway push failed (ignored): {e}")

    # ---- internal ----
    @staticmethod
    def _sum_counter(counter: Counter) -> float:
        total = 0.0
        for metric in counter.collect():
            for sample in metric.samples:
                total += float(sample.value or 0.0)
        return total

    # ---- decorators (unchanged interface) ----

        # Alias to keep older/newer code paths working
    def track_tool_execution(self, tool_name: str):
        # Reuse the existing decorator to avoid duplication
        return self.track_tool(tool_name)

    def track_request(self, method: str, endpoint: str):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                status = "success"
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    status = "error"
                    self.errors.labels(error_type=type(e).__name__, endpoint=endpoint).inc()
                    raise
                finally:
                    duration = time.time() - start_time
                    self.request_count.labels(method=method, endpoint=endpoint, status=status).inc()
                    self.request_duration.labels(method=method, endpoint=endpoint).observe(duration)
                    self._try_push("requests")
            return wrapper
        return decorator

    def track_tool(self, tool_name: str):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                status = "success"
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    status = "error"
                    self.errors.labels(error_type=type(e).__name__, endpoint=tool_name).inc()
                    raise
                finally:
                    duration = time.time() - start_time
                    self.tool_execution_count.labels(tool_name=tool_name, status=status).inc()
                    self.tool_execution_duration.labels(tool_name=tool_name).observe(duration)
                    self._try_push("tools")
            return wrapper
        return decorator

    def track_model_call(self, model_name: str):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                status = "success"
                prompt_tokens = int(kwargs.get("prompt_tokens", 0) or 0)
                completion_tokens = int(kwargs.get("completion_tokens", 0) or 0)
                try:
                    result = func(*args, **kwargs)
                    if prompt_tokens > 0:
                        self.model_tokens.labels(model_name=model_name, type="prompt").inc(prompt_tokens)
                        try:
                            self.token_budget_remaining.dec(float(prompt_tokens))
                        except Exception:
                            pass
                    if completion_tokens > 0:
                        self.model_tokens.labels(model_name=model_name, type="completion").inc(completion_tokens)
                        try:
                            self.token_budget_remaining.dec(float(completion_tokens))
                        except Exception:
                            pass
                    return result
                except Exception:
                    status = "error"
                    raise
                finally:
                    duration = time.time() - start_time
                    self.model_requests.labels(model_name=model_name, status=status).inc()
                    self.model_latency.labels(model_name=model_name).observe(duration)
                    self._try_push("models")
            return wrapper
        return decorator

    def track_terraform_plan(self):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                status = "success"
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception:
                    status = "error"
                    raise
                finally:
                    duration = time.time() - start_time
                    self.terraform_plans.labels(status=status).inc()
                    self.terraform_plan_duration.observe(duration)
                    self._try_push("terraform")
            return wrapper
        return decorator

    # ---- direct record helpers (unchanged interface) ----
    def record_chat_message(self, role: str, chat_type: str = "terraform"):
        self.chat_messages.labels(role=role, chat_type=chat_type).inc()
        self._try_push("chat")

    def record_cost_estimation(self, method: str, estimated_cost: float):
        self.cost_estimations.labels(method=method).inc()
        try:
            self.estimated_costs.set(float(estimated_cost))
        except Exception:
            pass
        self._try_push("cost")

    def update_active_sessions(self, count: int):
        self.chat_sessions.set(float(count))
        self._try_push("sessions")

    def get_metrics(self) -> str:
        return generate_latest(self.registry).decode('utf-8')

    def get_metrics_summary(self) -> Dict[str, Any]:
        try:
            active_sessions = 0
            cs = list(self.chat_sessions.collect())
            if cs and cs[0].samples:
                active_sessions = int(cs[0].samples[0].value)

            est = 0.0
            es = list(self.estimated_costs.collect())
            if es and es[0].samples:
                est = float(es[0].samples[0].value)

            budget = float(settings.TOKEN_BUDGET)
            tb = list(self.token_budget_remaining.collect())
            if tb and tb[0].samples:
                budget = float(tb[0].samples[0].value)

            return {
                "requests_total": int(self._sum_counter(self.request_count)),
                "errors_total": int(self._sum_counter(self.errors)),
                "tools_total": int(self._sum_counter(self.tool_execution_count)),
                "model_requests_total": int(self._sum_counter(self.model_requests)),
                "terraform_plans_total": int(self._sum_counter(self.terraform_plans)),
                "chat_messages_total": int(self._sum_counter(self.chat_messages)),
                "active_sessions": active_sessions,
                "estimated_costs_usd": est,
                "token_budget_remaining": budget,
            }
        except Exception as e:
            logger.error(f"Failed to get metrics summary: {str(e)}")
            return {"error": f"Metrics summary failed: {str(e)}"}


# Global metrics instance
metrics = MetricsCollector()
