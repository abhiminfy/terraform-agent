# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List

from celery import Celery

from src.app.core.config import Settings

# -----------------------------------------------------------------------------
# Settings & environment
# -----------------------------------------------------------------------------
settings = Settings()

# Ensure Infracost API key is available to the integration at import time.
# (_ICSettings in infracost_integration captures env on import.)
if settings.infracost_api_key and not os.getenv("INFRACOST_API_KEY"):
    os.environ["INFRACOST_API_KEY"] = settings.infracost_api_key

# Broker/backend: read from env (preferred) with safe Redis defaults.
BROKER_URL = os.getenv("CELERY_BROKER_URL", os.getenv("REDIS_URL", "redis://redis:6379/1"))
RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/2")

# Import AFTER setting env so any downstream module sees the key on import.
# Use the compat shim, which provides a stable API regardless of internal changes.
from src.app.services.compat import cost_diff as compat_cost_diff
from src.app.services.compat import (  # noqa: E402
    estimate_cost_async_v2 as compat_estimate_cost_async_v2,
)
from src.app.services.compat import estimate_cost_sync as compat_estimate_cost_sync
from src.app.services.compat import get_budget_status as compat_get_budget_status
from src.app.services.compat import update_budget as compat_update_budget

# -----------------------------------------------------------------------------
# Celery application
# -----------------------------------------------------------------------------
celery_app = Celery("tfagent", broker=BROKER_URL, backend=RESULT_BACKEND)
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)


# -----------------------------------------------------------------------------
# Tasks
# -----------------------------------------------------------------------------
@celery_app.task(name="infracost.estimate")
def infracost_estimate_task(tf_code: str, currency: str | None = "USD") -> Dict[str, Any]:
    """
    Asynchronous cost estimate via compat shim -> integration.
    """
    try:
        result = asyncio.run(compat_estimate_cost_async_v2(tf_code=tf_code, currency=currency))
        return result
    except Exception as e:
        return {"success": False, "data": {}, "error": f"Task execution failed: {e}"}


@celery_app.task(name="infracost.estimate_sync")
def infracost_estimate_sync_task(tf_code: str, workspace: str = "default") -> Dict[str, Any]:
    """
    Synchronous cost estimate via compat shim.
    """
    try:
        return compat_estimate_cost_sync(tf_code=tf_code, workspace=workspace)
    except Exception as e:
        return {
            "success": False,
            "error": f"Sync task execution failed: {e}",
            "fallback_available": True,
        }


@celery_app.task(name="infracost.cost_diff")
def infracost_cost_diff_task(
    old_tf_code: str, new_tf_code: str, workspace: str = "default"
) -> Dict[str, Any]:
    """
    Cost delta between two Terraform configs via compat shim.
    """
    try:
        return compat_cost_diff(old_tf=old_tf_code, new_tf=new_tf_code, workspace=workspace)
    except Exception as e:
        return {
            "success": False,
            "error": f"Cost diff task execution failed: {e}",
            "fallback_available": False,
        }


@celery_app.task(name="infracost.update_budget")
def infracost_update_budget_task(
    workspace: str, monthly_limit: float, alert_thresholds: List[float]
) -> Dict[str, Any]:
    """
    Update workspace budget settings via compat shim.
    """
    try:
        return compat_update_budget(
            workspace=workspace, monthly_limit=monthly_limit, alert_thresholds=alert_thresholds
        )
    except Exception as e:
        return {"success": False, "error": f"Budget update task execution failed: {e}"}


@celery_app.task(name="infracost.get_budget_status")
def infracost_get_budget_status_task(workspace: str = "default") -> Dict[str, Any]:
    """
    Retrieve current budget status for a workspace via compat shim.
    """
    try:
        return compat_get_budget_status(workspace=workspace)
    except Exception as e:
        return {
            "workspace": workspace,
            "budget_configured": False,
            "error": f"Failed to get budget status: {e}",
        }
