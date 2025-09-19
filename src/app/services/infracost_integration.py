# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import shutil

# Note: we keep stdlib subprocess to avoid extra deps and match your project
import subprocess
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# -----------------------
# Settings / environment
# -----------------------
@dataclass(frozen=True)
class _ICSettings:
    TF_BIN: str = os.getenv("TF_BIN", "terraform")
    INFRACOST_BIN: str = os.getenv("INFRACOST_BIN", "infracost")
    INFRACOST_API_KEY: Optional[str] = os.getenv("INFRACOST_API_KEY")

    # Default workspace + budgets
    DEFAULT_WORKSPACE: str = os.getenv("INFRACOST_WORKSPACE", "default")
    BUDGETS_FILE: str = os.getenv("INFRACOST_BUDGETS_FILE", "budgets.json")

    # Timeouts (seconds)
    TF_INIT_TIMEOUT: int = int(os.getenv("TF_INIT_TIMEOUT", "60"))
    TF_PLAN_TIMEOUT: int = int(os.getenv("TF_PLAN_TIMEOUT", "90"))
    TF_SHOW_TIMEOUT: int = int(os.getenv("TF_SHOW_TIMEOUT", "60"))
    IC_TIMEOUT: int = int(os.getenv("INFRACOST_TIMEOUT", "60"))

    # Extra flags
    TF_INPUT: str = os.getenv("TF_INPUT", "false")  # do not prompt in non-interactive
    TF_NO_COLOR: str = os.getenv("TF_NO_COLOR", "true")

    # Optional Terraform CLI args (var-files etc.) – applied only to *this* module
    TF_CLI_ARGS: str = os.getenv("TF_CLI_ARGS_INTEG", "")  # keep your global TF_CLI_ARGS untouched

    # Currency (default USD)
    CURRENCY: str = (os.getenv("INFRACOST_CURRENCY", "USD") or "USD").upper()


_S = _ICSettings()


# -----------------------
# Small in-memory cache
# -----------------------
class _LRUCache:
    """Tiny threadsafe LRU cache for cost breakdown results keyed by (hash(tf), currency)."""

    def __init__(self, maxsize: int = 64):
        self.maxsize = maxsize
        self._data: OrderedDict[str, Any] = OrderedDict()
        self._lock = RLock()

    def _mk(self, key: str) -> str:
        return key

    def get(self, key: str) -> Any:
        with self._lock:
            k = self._mk(key)
            if k in self._data:
                self._data.move_to_end(k)
                return self._data[k]
            return None

    def set(self, key: str, value: Any):
        with self._lock:
            k = self._mk(key)
            self._data[k] = value
            self._data.move_to_end(k)
            if len(self._data) > self.maxsize:
                self._data.popitem(last=False)


_cache = _LRUCache(maxsize=64)


# -----------------------
# Helpers
# -----------------------
def _run(
    cmd: List[str],
    cwd: Optional[str] = None,
    timeout: int = 60,
    env: Optional[Dict[str, str]] = None,
) -> Tuple[int, str, str]:
    try:
        logger.debug("Running: %s (cwd=%s)", " ".join(cmd), cwd or os.getcwd())
        p = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            check=False,
        )
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        return 124, "", f"Command timed out after {timeout}s: {e}"
    except FileNotFoundError:
        return 127, "", f"Binary not found: {cmd[0]}"
    except Exception as e:
        return 1, "", f"Command failed: {e}"


async def _run_async(
    *cmd: str, cwd: Optional[str] = None, timeout: int = 60, env: Optional[Dict[str, str]] = None
):
    try:
        logger.debug("Running (async): %s (cwd=%s)", " ".join(cmd), cwd or os.getcwd())
        p = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        try:
            out_b, err_b = await asyncio.wait_for(p.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            p.kill()
            return 124, "", f"Command timed out after {timeout}s"
        return p.returncode, (out_b or b"").decode(), (err_b or b"").decode()
    except FileNotFoundError:
        return 127, "", f"Binary not found: {cmd[0]}"
    except Exception as e:
        return 1, "", f"Command failed: {e}"


def _secure_tempdir(prefix: str = "ic_tf_"):
    return tempfile.TemporaryDirectory(prefix=prefix)


def _hash_tf(tf_code: str, usage_yaml: Optional[str], currency: str) -> str:
    h = hashlib.sha256()
    h.update(tf_code.encode("utf-8"))
    if usage_yaml:
        h.update(b"|usage|")
        h.update(usage_yaml.encode("utf-8"))
    h.update(f"|{currency}|".encode("utf-8"))
    return h.hexdigest()


def _write_tf_files(dirpath: str, tf_code: str, usage_yaml: Optional[str] = None):
    Path(dirpath, "main.tf").write_text(tf_code, encoding="utf-8")
    if usage_yaml:
        Path(dirpath, "usage.yml").write_text(usage_yaml, encoding="utf-8")


def _terraform_plan_json(dirpath: str) -> Tuple[bool, str, str]:
    env = dict(os.environ)
    # Apply only our module-local TF_CLI_ARGS to keep caller-global TF_CLI_ARGS intact.
    if _S.TF_CLI_ARGS:
        env["TF_CLI_ARGS"] = _S.TF_CLI_ARGS

    rc, out, err = _run(
        [_S.TF_BIN, "init", "-no-color", f"-input={_S.TF_INPUT}"],
        cwd=dirpath,
        timeout=_S.TF_INIT_TIMEOUT,
        env=env,
    )
    if rc != 0:
        return False, "", f"terraform init failed: {err or out}"

    # Create machine-readable plan
    plan_path = str(Path(dirpath, "plan.tfplan"))
    rc, out, err = _run(
        [_S.TF_BIN, "plan", "-no-color", f"-input={_S.TF_INPUT}", "-out", plan_path],
        cwd=dirpath,
        timeout=_S.TF_PLAN_TIMEOUT,
        env=env,
    )
    if rc != 0:
        return False, "", f"terraform plan failed: {err or out}"

    rc, out, err = _run(
        [_S.TF_BIN, "show", "-json", plan_path],
        cwd=dirpath,
        timeout=_S.TF_SHOW_TIMEOUT,
        env=env,
    )
    if rc != 0:
        return False, "", f"terraform show -json failed: {err or out}"

    plan_json_path = str(Path(dirpath, "plan.json"))
    Path(plan_json_path).write_text(out, encoding="utf-8")
    return True, plan_json_path, ""


def _parse_infracost_json(txt: str) -> Tuple[float, float, str, List[Dict[str, Any]]]:
    """
    Support both legacy and modern Infracost JSON formats.
    Returns: (monthly_cost, yearly_cost, currency, resource_breakdown[])
    """
    data = {}
    try:
        data = json.loads(txt or "{}")
    except Exception:
        return 0.0, 0.0, _S.CURRENCY, []

    currency = (data.get("currency") or _S.CURRENCY).upper()

    # Newer format: projects[*].breakdown.totalMonthlyCost and .resources[*]
    total_monthly = 0.0
    resources: List[Dict[str, Any]] = []
    projects = data.get("projects") or []

    if projects:
        for p in projects:
            b = p.get("breakdown") or {}
            # cost as string, e.g. "123.45"
            try:
                total_monthly += float(b.get("totalMonthlyCost") or 0)
            except Exception:
                pass
            for r in b.get("resources") or []:
                name = r.get("name") or r.get("address") or r.get("resourceType") or "resource"
                try:
                    rcost = float(r.get("monthlyCost") or 0)
                except Exception:
                    rcost = 0.0
                resources.append({"name": name, "monthly_cost": rcost})
    else:
        # Legacy format (top-level totalMonthlyCost)
        try:
            total_monthly = float(data.get("totalMonthlyCost") or 0)
        except Exception:
            total_monthly = 0.0
        # Try to build a minimal breakdown if present
        for r in data.get("resources") or []:
            name = r.get("name") or r.get("address") or r.get("resourceType") or "resource"
            try:
                rcost = float(r.get("monthlyCost") or 0)
            except Exception:
                rcost = 0.0
            resources.append({"name": name, "monthly_cost": rcost})

    yearly = total_monthly * 12.0
    # Top N resources by cost
    resources.sort(key=lambda x: x.get("monthly_cost", 0.0), reverse=True)
    return total_monthly, yearly, currency, resources[:25]


def _read_budgets() -> Dict[str, Any]:
    p = Path(_S.BUDGETS_FILE)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("budgets.json is invalid; recreating")
    return {}


def _write_budgets(b: Dict[str, Any]) -> bool:
    try:
        Path(_S.BUDGETS_FILE).write_text(
            json.dumps(b, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return True
    except Exception as e:
        logger.error("Failed to write budgets: %s", e)
        return False


def _budget_analysis(workspace: str, monthly_cost: float) -> Dict[str, Any]:
    budgets = _read_budgets()
    ws = budgets.get(workspace) or {}
    limit = float(ws.get("monthly_limit") or 0.0)
    thresholds = ws.get("alert_thresholds") or [0.5, 0.8, 1.0]

    alert_level = "GREEN"
    utilization_pct = 0.0
    if limit > 0:
        utilization_pct = (monthly_cost / limit) * 100.0
        if utilization_pct >= (thresholds[-1] * 100):
            alert_level = "RED"
        elif utilization_pct >= (thresholds[0] * 100):
            alert_level = "YELLOW"

    return {
        "workspace": workspace,
        "monthly_limit": limit,
        "thresholds": thresholds,
        "budget_utilization_percent": utilization_pct,
        "alert_level": alert_level,
    }


# -----------------------
# Public Integration API
# -----------------------
class InfracostIntegration:
    def __init__(self):
        self.infracost_available = self._check_infracost()

    def _check_infracost(self) -> bool:
        rc, _, _ = _run([_S.INFRACOST_BIN, "--version"], timeout=8)
        return rc == 0

    # ---- Breakdown (HCL) ----------------------------------------------------
    def generate_cost_estimate(
        self,
        terraform_code: str,
        workspace: str = _S.DEFAULT_WORKSPACE,
        usage_yaml: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        1) Writes TF to a temp dir (and optional usage.yml)
        2) Terraform init/plan/show -json
        3) infracost breakdown --path plan.json (preferred), else --path <dir>
        """
        if not terraform_code or not terraform_code.strip():
            return {"success": False, "error": "terraform_code required"}

        currency = _S.CURRENCY
        cache_key = f"breakdown:{_hash_tf(terraform_code, usage_yaml, currency)}"
        cached = _cache.get(cache_key)
        if cached:
            return cached

        if not self.infracost_available:
            return {"success": False, "error": "Infracost CLI not available"}

        env = dict(os.environ)
        if _S.INFRACOST_API_KEY:
            env["INFRACOST_API_KEY"] = _S.INFRACOST_API_KEY

        with _secure_tempdir("ic_hcl_") as d:
            _write_tf_files(d, terraform_code, usage_yaml=usage_yaml)

            ok, plan_json_path, err = _terraform_plan_json(d)
            if not ok:
                return {"success": False, "error": err}

            # Prefer pointing Infracost at the plan.json for accuracy
            cmd = [
                _S.INFRACOST_BIN,
                "breakdown",
                "--path",
                plan_json_path,
                "--format",
                "json",
                "--no-color",
                "--project-name",
                workspace,
                "--currency",
                currency,
            ]
            if usage_yaml:
                cmd += ["--usage-file", str(Path(d, "usage.yml"))]

            rc, out, err = _run(cmd, cwd=d, timeout=_S.IC_TIMEOUT, env=env)
            if rc != 0:
                # fallback to pointing at dir
                rc2, out2, err2 = _run(
                    [
                        _S.INFRACOST_BIN,
                        "breakdown",
                        "--path",
                        d,
                        "--format",
                        "json",
                        "--no-color",
                        "--project-name",
                        workspace,
                        "--currency",
                        currency,
                    ],
                    cwd=d,
                    timeout=_S.IC_TIMEOUT,
                    env=env,
                )
                if rc2 != 0:
                    return {"success": False, "error": f"infracost breakdown failed: {err or err2}"}
                out = out2

        monthly, yearly, currency, resources = _parse_infracost_json(out)
        result = {
            "success": True,
            "currency": currency,
            "cost_estimate": {
                "monthly_cost": monthly,
                "yearly_cost": yearly,
                "daily_cost": monthly / 30.0 if monthly else 0.0,
            },
            "resource_breakdown": resources,
            "budget_analysis": _budget_analysis(workspace, monthly),
            "generated_at": datetime.utcnow().isoformat() + "Z",
        }
        _cache.set(cache_key, result)
        return result

    # ---- Diff ---------------------------------------------------------------
    def generate_cost_diff(
        self, old_tf: str, new_tf: str, workspace: str = _S.DEFAULT_WORKSPACE
    ) -> Dict[str, Any]:
        if not (old_tf and new_tf):
            return {"success": False, "error": "Both old_terraform and new_terraform are required"}

        if not self.infracost_available:
            return {"success": False, "error": "Infracost CLI not available"}

        env = dict(os.environ)
        if _S.INFRACOST_API_KEY:
            env["INFRACOST_API_KEY"] = _S.INFRACOST_API_KEY

        with _secure_tempdir("ic_old_") as d1, _secure_tempdir("ic_new_") as d2:
            _write_tf_files(d1, old_tf)
            _write_tf_files(d2, new_tf)

            ok1, plan1, err1 = _terraform_plan_json(d1)
            ok2, plan2, err2 = _terraform_plan_json(d2)
            if not (ok1 and ok2):
                return {"success": False, "error": f"Plan failure: old=({err1}), new=({err2})"}

            cmd = [
                _S.INFRACOST_BIN,
                "diff",
                "--path",
                plan1,
                "--path2",
                plan2,
                "--format",
                "json",
                "--no-color",
                "--project-name",
                workspace,
                "--currency",
                _S.CURRENCY,
            ]
            rc, out, err = _run(cmd, timeout=_S.IC_TIMEOUT, env=env)
            if rc != 0:
                return {"success": False, "error": f"infracost diff failed: {err}"}

        # Parse diff: use the same parser; monthly cost delta is difference
        m, y, currency, _ = _parse_infracost_json(out)
        # When using diff, Infracost may output "totalMonthlyCost" as delta; if not, we just report it as "delta"
        return {
            "success": True,
            "currency": currency,
            "delta_monthly_cost": m,
            "delta_yearly_cost": y,
            "generated_at": datetime.utcnow().isoformat() + "Z",
        }

    # ---- Budgets ------------------------------------------------------------
    def update_budget(
        self, workspace: str, monthly_limit: float, alert_thresholds: List[float]
    ) -> Dict[str, Any]:
        try:
            if monthly_limit < 0:
                return {"success": False, "error": "monthly_limit must be >= 0"}
            if not alert_thresholds or any(t <= 0 for t in alert_thresholds):
                return {"success": False, "error": "alert_thresholds must be positive floats"}
            thresholds = sorted(alert_thresholds)
            b = _read_budgets()
            b[workspace] = {"monthly_limit": float(monthly_limit), "alert_thresholds": thresholds}
            if not _write_budgets(b):
                return {"success": False, "error": "Failed to persist budgets"}
            return {
                "success": True,
                "workspace": workspace,
                "monthly_limit": monthly_limit,
                "alert_thresholds": thresholds,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_budget_status(self, workspace: str) -> Dict[str, Any]:
        b = _read_budgets()
        ws = b.get(workspace) or {"monthly_limit": 0.0, "alert_thresholds": [0.5, 0.8, 1.0]}
        return {"success": True, "workspace": workspace, **ws}

    # ---- Async v2 (compatible with your routes) -----------------------------
    async def estimate_cost_async_v2(
        self, tf_code: str, currency: Optional[str] = None
    ) -> Dict[str, Any]:
        currency = (currency or _S.CURRENCY).upper()

        if not tf_code or not tf_code.strip():
            return {"success": False, "data": {}, "error": "terraform_code required"}

        if not self.infracost_available:
            return {"success": False, "data": {}, "error": "Infracost CLI not available"}

        env = dict(os.environ)
        if _S.INFRACOST_API_KEY:
            env["INFRACOST_API_KEY"] = _S.INFRACOST_API_KEY

        with _secure_tempdir("ic_v2_") as d:
            _write_tf_files(d, tf_code)

            ok, plan_json_path, err = _terraform_plan_json(d)
            if not ok:
                return {"success": False, "data": {}, "error": err}

            rc, out, err = await _run_async(
                _S.INFRACOST_BIN,
                "breakdown",
                "--path",
                plan_json_path,
                "--format",
                "json",
                "--no-color",
                "--project-name",
                _S.DEFAULT_WORKSPACE,
                "--currency",
                currency,
                cwd=d,
                timeout=_S.IC_TIMEOUT,
                env=env,
            )
            if rc != 0:
                return {"success": False, "data": {}, "error": f"infracost breakdown failed: {err}"}

        monthly, yearly, currency, _ = _parse_infracost_json(out)
        return {
            "success": True,
            "data": {
                "currency": currency,
                "total_monthly_cost": monthly,
                "total_yearly_cost": yearly,
                "total_monthly_cost_usd": (
                    monthly if currency == "USD" else monthly
                ),  # leave as-is; caller can convert
            },
            "error": "",
        }


# Singleton (as in your original)
infracost_integration = InfracostIntegration()
