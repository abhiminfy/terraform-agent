# Compat shims so the app boots even when optional pieces are missing.
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

# -----------------------------
# Infracost integration (safe)
# -----------------------------
try:
    from src.app.services.infracost_integration import infracost_integration as _ic
except Exception as _ic_err:
    _ic = None  # type: ignore
else:
    _ic_err = None  # type: ignore


async def estimate_cost_async_v2(tf_code: str, currency: str = "USD") -> Dict[str, Any]:
    if _ic and hasattr(_ic, "estimate_cost_async_v2"):
        return await _ic.estimate_cost_async_v2(tf_code=tf_code, currency=currency)  # type: ignore[attr-defined]
    return {
        "success": False,
        "data": {},
        "error": f"Infracost unavailable: {_ic_err or 'missing method'}",
    }


def estimate_cost_sync(tf_code: str, workspace: str = "default") -> Dict[str, Any]:
    if _ic and hasattr(_ic, "generate_cost_estimate"):
        return _ic.generate_cost_estimate(terraform_code=tf_code, workspace=workspace)  # type: ignore[attr-defined]
    return {
        "success": False,
        "data": {},
        "error": f"Infracost unavailable: {_ic_err or 'missing method'}",
    }


def cost_diff(old_tf: str, new_tf: str, workspace: str = "default") -> Dict[str, Any]:
    if _ic and hasattr(_ic, "generate_cost_diff"):
        return _ic.generate_cost_diff(old_tf=old_tf, new_tf=new_tf, workspace=workspace)  # type: ignore[attr-defined]
    return {
        "success": False,
        "data": {},
        "error": f"Infracost unavailable: {_ic_err or 'missing method'}",
    }


def update_budget(
    workspace: str, monthly_limit: float, alert_thresholds: List[float]
) -> Dict[str, Any]:
    if _ic and hasattr(_ic, "update_budget"):
        return _ic.update_budget(workspace=workspace, monthly_limit=monthly_limit, alert_thresholds=alert_thresholds)  # type: ignore[attr-defined]
    return {
        "success": False,
        "data": {},
        "error": f"Infracost unavailable: {_ic_err or 'missing method'}",
    }


def get_budget_status(workspace: str = "default") -> Dict[str, Any]:
    if _ic and hasattr(_ic, "get_budget_status"):
        return _ic.get_budget_status(workspace=workspace)  # type: ignore[attr-defined]
    return {
        "success": False,
        "data": {},
        "error": f"Infracost unavailable: {_ic_err or 'missing method'}",
    }


# -----------------------------
# Policy engine (optional)
# -----------------------------
try:
    from src.app.services import policy_engine as _pe  # type: ignore
except Exception as _pe_err:
    _pe = None  # type: ignore
else:
    _pe_err = None  # type: ignore

_validate_fn = None
if _pe:
    for _name in (
        "validate_terraform_code_ast",
        "validate_terraform_ast",
        "validate_terraform_code",
    ):
        _validate_fn = getattr(_pe, _name, None)
        if _validate_fn:
            break


async def validate_terraform_code_ast(tf_code: str) -> Dict[str, Any]:
    """
    Soft wrapper: if a validator exists (any supported name), call it,
    otherwise return a friendly 'not available' response.
    """
    if callable(_validate_fn):
        res = _validate_fn(tf_code)  # may be sync or async
        if asyncio.iscoroutine(res):
            return await res
        return res
    return {"success": False, "data": {}, "error": "Policy validation not available in this build."}


def get_policy_summary() -> Dict[str, Any]:
    if _pe and hasattr(_pe, "get_policy_summary"):
        try:
            return _pe.get_policy_summary()
        except Exception as e:  # pragma: no cover
            return {"success": False, "error": str(e)}
    return {"success": False, "error": "Policy engine not available."}


__all__ = [
    "estimate_cost_async_v2",
    "estimate_cost_sync",
    "cost_diff",
    "update_budget",
    "get_budget_status",
    "validate_terraform_code_ast",
    "get_policy_summary",
]
