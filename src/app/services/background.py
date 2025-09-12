# -*- coding: utf-8 -*-
import asyncio

from celery import Celery

from src.app.core.config import Settings
from src.app.services.infracost_integration import (
    estimate_cost_async_v2,
    infracost_integration,
)

settings = Settings()

celery_app = Celery(
    "tfagent",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

# Configure Celery
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)


@celery_app.task(name="infracost.estimate")
def infracost_estimate_task(tf_code: str, currency: str = "USD"):
    """
    Celery task for estimating Terraform infrastructure costs.

    Args:
        tf_code (str): Terraform configuration code
        currency (str): Target currency for cost estimation (default: USD)

    Returns:
        dict: Cost estimation result with success status, data, and error info
    """
    try:
        # Run the async function using asyncio
        result = asyncio.run(estimate_cost_async_v2(tf_code=tf_code, currency=currency))
        return result
    except Exception as e:
        return {
            "success": False,
            "data": {},
            "error": f"Task execution failed: {str(e)}",
        }


@celery_app.task(name="infracost.estimate_sync")
def infracost_estimate_sync_task(tf_code: str, workspace: str = "default"):
    """
    Celery task for synchronous Terraform cost estimation using the class method.

    Args:
        tf_code (str): Terraform configuration code
        workspace (str): Workspace name for budget analysis (default: "default")

    Returns:
        dict: Cost estimation result from InfracostIntegration class
    """
    try:
        result = infracost_integration.generate_cost_estimate(
            terraform_code=tf_code, workspace=workspace
        )
        return result
    except Exception as e:
        return {
            "success": False,
            "error": f"Sync task execution failed: {str(e)}",
            "fallback_available": True,
        }


@celery_app.task(name="infracost.cost_diff")
def infracost_cost_diff_task(old_tf_code: str, new_tf_code: str, workspace: str = "default"):
    """
    Celery task for generating cost difference between two Terraform configurations.

    Args:
        old_tf_code (str): Original Terraform configuration
        new_tf_code (str): New Terraform configuration
        workspace (str): Workspace name (default: "default")

    Returns:
        dict: Cost difference analysis result
    """
    try:
        result = infracost_integration.generate_cost_diff(
            old_terraform=old_tf_code, new_terraform=new_tf_code, workspace=workspace
        )
        return result
    except Exception as e:
        return {
            "success": False,
            "error": f"Cost diff task execution failed: {str(e)}",
            "fallback_available": False,
        }


@celery_app.task(name="infracost.update_budget")
def infracost_update_budget_task(workspace: str, monthly_limit: float, alert_thresholds: list):
    """
    Celery task for updating budget configuration.

    Args:
        workspace (str): Workspace name
        monthly_limit (float): Monthly budget limit
        alert_thresholds (list): List of alert threshold values

    Returns:
        dict: Budget update result
    """
    try:
        result = infracost_integration.update_budget(
            workspace=workspace,
            monthly_limit=monthly_limit,
            alert_thresholds=alert_thresholds,
        )
        return result
    except Exception as e:
        return {
            "success": False,
            "error": f"Budget update task execution failed: {str(e)}",
        }


@celery_app.task(name="infracost.get_budget_status")
def infracost_get_budget_status_task(workspace: str = "default"):
    """
    Celery task for retrieving budget status.

    Args:
        workspace (str): Workspace name (default: "default")

    Returns:
        dict: Current budget status
    """
    try:
        result = infracost_integration.get_budget_status(workspace=workspace)
        return result
    except Exception as e:
        return {
            "workspace": workspace,
            "budget_configured": False,
            "error": f"Failed to get budget status: {str(e)}",
        }
