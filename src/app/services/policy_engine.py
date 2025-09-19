# src/app/services/policy_engine.py
"""
Advanced Policy Engine for Terraform code.

- Runs external scanners when available (tfsec, checkov, conftest, tflint, terrascan)
- Adds lightweight built-in rules as a fallback
- Normalizes findings into a common schema
- Generates a concise, human-friendly Markdown report
- Designed to degrade gracefully when tools are missing

This file is self-contained and safe to import even if binaries are not installed.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Optional metrics (no-op fallback if unavailable)
try:
    from src.app.core.metrics import metrics
except Exception:  # pragma: no cover

    class _NoopMetrics:
        def __getattr__(self, name):
            def _inner(*args, **kwargs):
                class _Ctx:
                    def __enter__(self_):
                        return None

                    def __exit__(self_, exc_type, exc, tb):
                        return False

                return _Ctx() if name.endswith("_timer") else (lambda *a, **k: None)

            return _inner

    metrics = _NoopMetrics()  # type: ignore

logger = logging.getLogger(__name__)


# -----------------------------
# Data model
# -----------------------------
SEVERITY_ORDER = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]


@dataclass
class Violation:
    id: str
    message: str
    severity: str = "MEDIUM"
    source: str = "builtin"
    resource: Optional[str] = None
    file: Optional[str] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    guideline: Optional[str] = None

    def normalize(self) -> "Violation":
        sev = (self.severity or "MEDIUM").upper()
        if sev not in SEVERITY_ORDER:
            sev = "MEDIUM"
        self.severity = sev
        return self


# -----------------------------
# Helpers
# -----------------------------
def _run_cmd(cmd: List[str], cwd: Optional[str] = None, timeout: int = 60) -> Dict[str, Any]:
    """Run a command and capture output. Never raises."""
    try:
        result = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False
        )
        return {
            "success": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout or "",
            "stderr": result.stderr or "",
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "returncode": -1,
            "stdout": "",
            "stderr": f"timeout after {timeout}s",
        }
    except FileNotFoundError:
        return {
            "success": False,
            "returncode": 127,
            "stdout": "",
            "stderr": f"tool not found: {cmd[0]}",
        }
    except Exception as e:  # pragma: no cover
        return {"success": False, "returncode": 1, "stdout": "", "stderr": str(e)}


def _which(name: str) -> bool:
    return shutil.which(name) is not None


def _summarize_severity(violations: List[Violation]) -> Dict[str, int]:
    summary = {k: 0 for k in SEVERITY_ORDER}
    for v in violations:
        summary[v.severity] = summary.get(v.severity, 0) + 1
    summary["TOTAL"] = len(violations)
    return summary


def _mk_temp_tfdir(terraform_code: str) -> Tuple[str, Path]:
    tmpdir = tempfile.mkdtemp(prefix="policy_")
    tf_path = Path(tmpdir) / "main.tf"
    tf_path.write_text(terraform_code or "", encoding="utf-8")
    return tmpdir, tf_path


def _safe_json_load(s: str) -> Any:
    try:
        return json.loads(s)
    except Exception:
        return None


# -----------------------------
# Engine
# -----------------------------
class PolicyEngine:
    def __init__(self) -> None:
        self.enabled_scanners: Dict[str, bool] = {
            "tfsec": _which("tfsec"),
            "checkov": _which("checkov"),
            "conftest": _which("conftest"),
            "tflint": _which("tflint"),
            "terrascan": _which("terrascan"),
        }

    # -------- Built-in lightweight checks (regex) --------
    def _builtin_checks(self, code: str) -> List[Violation]:
        v: List[Violation] = []
        if not code:
            return v

        rules = [
            (
                r'cidr_blocks\s*=\s*\["0\.0\.0\.0/0"\]',
                "Security group open to the world (0.0.0.0/0).",
                "HIGH",
                "SG-0001",
            ),
            (
                r"publicly_accessible\s*=\s*true",
                "Database is publicly accessible.",
                "HIGH",
                "DB-0001",
            ),
            (
                r"force_destroy\s*=\s*true",
                "Destructive deletion enabled (e.g. S3 bucket).",
                "HIGH",
                "S3-0001",
            ),
            (
                r"skip_final_snapshot\s*=\s*true",
                "RDS final snapshot is skipped on deletion.",
                "MEDIUM",
                "RDS-0002",
            ),
            (
                r"deletion_protection\s*=\s*false",
                "Deletion protection disabled.",
                "MEDIUM",
                "GEN-0003",
            ),
            (
                r'access_key\s*=\s*".*"',
                "Hardcoded AWS access key detected.",
                "CRITICAL",
                "IAM-0001",
            ),
            (
                r'secret_key\s*=\s*".*"',
                "Hardcoded AWS secret key detected.",
                "CRITICAL",
                "IAM-0002",
            ),
            (r'password\s*=\s*".*"', "Hardcoded password detected.", "CRITICAL", "SEC-0003"),
        ]

        for pattern, msg, sev, rid in rules:
            if re.search(pattern, code, re.IGNORECASE):
                v.append(Violation(id=rid, message=msg, severity=sev, source="builtin").normalize())

        # S3 bucket public ACL
        if re.search(r'resource\s+"aws_s3_bucket(_acl)?"\s+"[^"]+"\s*{', code):
            if re.search(r'acl\s*=\s*"(public-read|public-read-write)"', code, re.IGNORECASE):
                v.append(
                    Violation(
                        id="S3-0002",
                        message="Public S3 ACL set.",
                        severity="HIGH",
                        source="builtin",
                    ).normalize()
                )

        # Missing required_providers/version pins
        if "required_providers" not in code:
            v.append(
                Violation(
                    id="TF-0001",
                    message="Pin provider versions via required_providers.",
                    severity="LOW",
                    source="builtin",
                ).normalize()
            )
        if "required_version" not in code:
            v.append(
                Violation(
                    id="TF-0002",
                    message="Pin Terraform required_version.",
                    severity="LOW",
                    source="builtin",
                ).normalize()
            )

        return v

    # -------- External scanners --------
    def _run_tfsec(self, tfdir: str) -> Tuple[List[Violation], Dict[str, Any]]:
        if not self.enabled_scanners.get("tfsec"):
            return [], {"available": False, "error": "tfsec not installed"}

        cmd = ["tfsec", "--no-colour", "--format", "json", "--soft-fail", tfdir]
        with metrics.tool_timer("tfsec"):
            r = _run_cmd(cmd, cwd=tfdir, timeout=120)
        if r["returncode"] == 127:
            return [], {"available": False, "error": r["stderr"]}
        data = _safe_json_load(r.get("stdout", "")) or {}
        violations: List[Violation] = []
        for item in data.get("results", []):
            loc = item.get("location", {}) or {}
            v = Violation(
                id=str(item.get("rule_id") or item.get("id") or "TFSEC"),
                message=str(
                    item.get("description") or item.get("long_description") or "tfsec finding"
                ),
                severity=str(item.get("severity") or "MEDIUM"),
                source="tfsec",
                resource=item.get("resource"),
                file=loc.get("filename"),
                start_line=loc.get("start_line") or loc.get("start") or None,
                end_line=loc.get("end_line") or None,
                guideline=(item.get("links") or [None])[0],
            ).normalize()
            violations.append(v)
        return violations, {"available": True, "raw": data, "stderr": r.get("stderr", "")}

    def _run_checkov(self, tfdir: str, tfpath: Path) -> Tuple[List[Violation], Dict[str, Any]]:
        if not self.enabled_scanners.get("checkov"):
            return [], {"available": False, "error": "checkov not installed"}

        # Prefer scanning the file to avoid network/etc.
        cmd = ["checkov", "-f", str(tfpath), "-o", "json", "--framework", "terraform"]
        with metrics.tool_timer("checkov"):
            r = _run_cmd(cmd, cwd=tfdir, timeout=180)
        if r["returncode"] == 127:
            return [], {"available": False, "error": r["stderr"]}
        data = _safe_json_load(r.get("stdout", "")) or {}

        violations: List[Violation] = []
        # Checkov JSON may put failed checks under 'results' or 'summary' structures
        for section in ("results",):
            res = data.get(section) or {}
            for check in res.get("failed_checks", []) or []:
                v = Violation(
                    id=str(check.get("check_id") or "CKV_UNKNOWN"),
                    message=str(check.get("check_name") or "checkov finding"),
                    severity=str(check.get("severity") or "MEDIUM"),
                    source="checkov",
                    resource=check.get("resource"),
                    file=check.get("file_path"),
                    start_line=check.get("file_line_range", [None, None])[0],
                    end_line=check.get("file_line_range", [None, None])[1],
                    guideline=(
                        (check.get("guideline") or (check.get("guidelines") or [None]))[0]
                        if isinstance(check.get("guidelines"), list)
                        else check.get("guideline")
                    ),
                ).normalize()
                violations.append(v)
        # Some versions place it in 'check_type' items
        if not violations:
            for item in data if isinstance(data, list) else []:
                failed = (item.get("summary") or {}).get("failed", 0)
                if failed and item.get("results", {}).get("failed_checks"):
                    for check in item["results"]["failed_checks"]:
                        v = Violation(
                            id=str(check.get("check_id") or "CKV_UNKNOWN"),
                            message=str(check.get("check_name") or "checkov finding"),
                            severity=str(check.get("severity") or "MEDIUM"),
                            source="checkov",
                            resource=check.get("resource"),
                            file=check.get("file_path"),
                            start_line=check.get("file_line_range", [None, None])[0],
                            end_line=check.get("file_line_range", [None, None])[1],
                            guideline=check.get("guideline"),
                        ).normalize()
                        violations.append(v)

        return violations, {"available": True, "raw": data, "stderr": r.get("stderr", "")}

    def _run_conftest(self, tfdir: str) -> Tuple[List[Violation], Dict[str, Any]]:
        if not self.enabled_scanners.get("conftest"):
            return [], {"available": False, "error": "conftest not installed"}
        # This assumes there are Rego policies in ./policy/ - if none, conftest returns 0
        # We still capture output for completeness.
        cmd = ["conftest", "test", "--no-color", "--output", "json", "."]
        with metrics.tool_timer("conftest"):
            r = _run_cmd(cmd, cwd=tfdir, timeout=90)
        data = _safe_json_load(r.get("stdout", "")) or []
        violations: List[Violation] = []
        for file_result in data:
            for res in file_result.get("failures", []):
                v = Violation(
                    id=str(res.get("code") or "REGO"),
                    message=str(res.get("msg") or "policy failure"),
                    severity="MEDIUM",
                    source="conftest",
                    file=file_result.get("filename"),
                ).normalize()
                violations.append(v)
        return violations, {"available": True, "raw": data, "stderr": r.get("stderr", "")}

    def _run_tflint(self, tfdir: str) -> Tuple[List[Violation], Dict[str, Any]]:
        if not self.enabled_scanners.get("tflint"):
            return [], {"available": False, "error": "tflint not installed"}
        cmd = ["tflint", "--format", "json"]
        with metrics.tool_timer("tflint"):
            r = _run_cmd(cmd, cwd=tfdir, timeout=90)
        data = _safe_json_load(r.get("stdout", "")) or {}
        violations: List[Violation] = []
        for diag in data.get("diagnostics", []):
            v = Violation(
                id=str(diag.get("rule") or "TFLINT"),
                message=str(diag.get("message") or "tflint issue"),
                severity=str(diag.get("severity") or "MEDIUM"),
                source="tflint",
                file=diag.get("range", {}).get("filename"),
                start_line=(diag.get("range", {}).get("start", {}) or {}).get("line"),
                end_line=(diag.get("range", {}).get("end", {}) or {}).get("line"),
            ).normalize()
            violations.append(v)
        return violations, {"available": True, "raw": data, "stderr": r.get("stderr", "")}

    def _run_terrascan(self, tfdir: str) -> Tuple[List[Violation], Dict[str, Any]]:
        if not self.enabled_scanners.get("terrascan"):
            return [], {"available": False, "error": "terrascan not installed"}
        cmd = ["terrascan", "scan", "-o", "json"]
        with metrics.tool_timer("terrascan"):
            r = _run_cmd(cmd, cwd=tfdir, timeout=180)
        data = _safe_json_load(r.get("stdout", "")) or {}
        violations: List[Violation] = []
        for res in (data.get("results", {}) or {}).get("violations", []):
            v = Violation(
                id=str(res.get("rule_id") or "TERRASCAN"),
                message=str(res.get("description") or "terrascan violation"),
                severity=str(res.get("severity") or "MEDIUM"),
                source="terrascan",
                file=(res.get("file") or {}).get("file"),
                start_line=(res.get("location", {}) or {}).get("start_line"),
                end_line=(res.get("location", {}) or {}).get("end_line"),
            ).normalize()
            violations.append(v)
        return violations, {"available": True, "raw": data, "stderr": r.get("stderr", "")}

    # -------- Public API --------
    def validate_with_policies(self, terraform_code: str) -> Dict[str, Any]:
        """
        Validate Terraform code with external scanners (when available) plus built-in rules.
        Returns a normalized result dict with violations and a summary.
        """
        if not terraform_code or not terraform_code.strip():
            return {
                "success": False,
                "violations": [],
                "warnings": ["empty terraform code"],
                "summary": {"TOTAL": 0},
            }

        tmpdir, tfpath = _mk_temp_tfdir(terraform_code)
        all_violations: List[Violation] = []
        scanner_results: Dict[str, Any] = {}
        try:
            # External scanners
            for name, runner in (
                ("tfsec", self._run_tfsec),
                ("checkov", lambda d: self._run_checkov(d, tfpath)),
                ("conftest", self._run_conftest),
                ("tflint", self._run_tflint),
                ("terrascan", self._run_terrascan),
            ):
                try:
                    vios, raw = runner(tmpdir)
                    scanner_results[name] = raw
                    all_violations.extend(vios)
                except Exception as e:  # never fail completely
                    logger.warning("Scanner %s failed: %s", name, e)
                    scanner_results[name] = {"available": False, "error": str(e)}

            # Built-in
            all_violations.extend(self._builtin_checks(terraform_code))

            # Normalize & summarize
            norm = [v.normalize() for v in all_violations]
            # sort by severity
            norm.sort(key=lambda v: (SEVERITY_ORDER.index(v.severity), v.id))
            summary = _summarize_severity(norm)

            return {
                "success": True,
                "violations": [asdict(v) for v in norm],
                "warnings": [],
                "summary": summary,
                "scanner_results": scanner_results,
            }
        finally:
            try:
                shutil.rmtree(tmpdir, ignore_errors=True)
            except Exception:
                pass

    def create_policy_report(self, policy_result: Dict[str, Any], terraform_code: str) -> str:
        """
        Build a Markdown report including a summary and top issues.
        """
        if not policy_result or not isinstance(policy_result, dict):
            return "No policy results available."

        summary = policy_result.get("summary") or {}
        vios = policy_result.get("violations") or []
        top = vios[: min(15, len(vios))]

        lines: List[str] = []
        lines.append("## Policy & Security Report")
        lines.append("")
        if summary:
            parts = [f"{k}: {v}" for k, v in summary.items() if k in SEVERITY_ORDER + ["TOTAL"]]
            lines.append("**Summary:** " + ", ".join(parts))
            lines.append("")

        if top:
            lines.append("### Top Findings")
            lines.append("")
            for v in top:
                sev = v.get("severity", "MEDIUM")
                src = v.get("source", "policy")
                vid = v.get("id", "ID")
                msg = v.get("message", "")
                file = v.get("file")
                line = v.get("start_line")
                where = f" ({file}:{line})" if file and line else ""
                lines.append(f"- **[{sev}] {vid}** ({src}) — {msg}{where}")
            lines.append("")

        # Hints
        if not vios:
            lines.append("✅ No violations detected by scanners or built-in rules.")
        else:
            lines.append(
                "_Consider addressing the issues above. Some scanners may be disabled or missing locally._"
            )

        return "\\n".join(lines)


# Singleton export used throughout the app
policy_engine = PolicyEngine()
