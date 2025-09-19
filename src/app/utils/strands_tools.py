# strands_tools.py — Advanced mode
import abc as _abc
import asyncio
import json
import logging
import os
import re
import subprocess
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from functools import wraps
from io import StringIO as _ST_StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hcl2 as _st_hcl2

from src.app.core.config import Settings as _ST_Settings
from src.app.utils.utils import run_cmd_async as _st_run_cmd_async
from src.app.utils.utils import secure_tempdir as _st_secure_tempdir

# -------------------------------------------------------------------
# Strands "Tool" shim (supports both `strands.tools.tool` and `strands.tool`)
# -------------------------------------------------------------------
try:
    from strands.tools import Tool
    from strands.tools import tool as strands_tool  # type: ignore
except Exception:
    try:
        from strands import Tool
        from strands import tool as strands_tool  # type: ignore
    except Exception:

        class Tool:  # type: ignore
            def __init__(self, name, description, parameters=None, function=None):
                self.name = name
                self.description = description
                self.parameters = parameters
                self.function = function

        def strands_tool(*args, **kwargs):
            def _wrap(f):
                return f

            return _wrap


# -----------------------------------------------------------------------------
# Logging + Config
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_st_cfg = _ST_Settings()


def _cfg(name: str, default: Any) -> Any:
    """Get config value from Settings or env (fallback)."""
    v = getattr(_st_cfg, name, None)
    if v is None:
        v = os.getenv(name, default)
    return v if v is not None else default


# Global knobs (overridable via Settings or env)
TF_VALIDATE_TIMEOUT = int(_cfg("TF_VALIDATE_TIMEOUT", 30))
TF_PLAN_TIMEOUT = int(_cfg("TF_PLAN_TIMEOUT", 120))
TF_FMT_TIMEOUT = int(_cfg("TF_FMT_TIMEOUT", 15))
STRICT_TF_VALIDATION = str(_cfg("STRICT_TF_VALIDATION", "true")).lower() == "true"

CB_FAILURE_THRESHOLD = int(_cfg("CB_FAILURE_THRESHOLD", 3))
CB_RESET_TIMEOUT = int(_cfg("CB_RESET_TIMEOUT", 60))

# ==================== SECURITY IMPLEMENTATIONS ====================


class SecurityException(Exception):
    pass


class DangerousConstructException(SecurityException):
    pass


class CircuitBreakerOpenException(SecurityException):
    pass


@dataclass
class ToolResult:
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "warnings": self.warnings,
        }


class CircuitBreaker:
    def __init__(
        self, failure_threshold: int = CB_FAILURE_THRESHOLD, reset_timeout: int = CB_RESET_TIMEOUT
    ):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def _is_circuit_open(self) -> bool:
        if self.state == "OPEN":
            if time.time() - (self.last_failure_time or 0) > self.reset_timeout:
                self.state = "HALF_OPEN"
                return False
            return True
        return False

    def record_success(self):
        self.failure_count = 0
        self.state = "CLOSED"
        self.last_failure_time = None

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"

    def execute(self, func, *args, **kwargs):
        if self._is_circuit_open():
            raise CircuitBreakerOpenException("Circuit breaker is open")
        try:
            result = func(*args, **kwargs)
            self.record_success()
            return result
        except Exception:
            self.record_failure()
            raise

    async def execute_async(self, coro_func, *args, **kwargs):
        if self._is_circuit_open():
            raise CircuitBreakerOpenException("Circuit breaker is open")
        try:
            result = await coro_func(*args, **kwargs)
            self.record_success()
            return result
        except Exception:
            self.record_failure()
            raise


def secure_command_execution(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            if hasattr(self, "circuit_breaker"):
                return self.circuit_breaker.execute(func, self, *args, **kwargs)
            return func(self, *args, **kwargs)
        except CircuitBreakerOpenException as e:
            return self._handle_exception(e, "Circuit breaker open")

    return wrapper


def secure_command_execution_async(func):
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        try:
            if hasattr(self, "circuit_breaker"):
                return await self.circuit_breaker.execute_async(func, self, *args, **kwargs)
            return await func(self, *args, **kwargs)
        except CircuitBreakerOpenException as e:
            return self._handle_exception(e, "Circuit breaker open")

    return wrapper


# ==================== BASE TOOL CLASSES ====================


class BaseTool(_abc.ABC):
    def __init__(self):
        self.circuit_breaker = CircuitBreaker()

    def _handle_exception(self, e: Exception, context: str = "") -> ToolResult:
        logger.error(f"Error in {self.__class__.__name__}: {str(e)}")
        return ToolResult(success=False, error=f"{context}: {str(e)}" if context else str(e))


class SecureTerraformMixin:
    BLOCKED_PROVISIONERS = ["local-exec", "remote-exec"]
    BLOCKED_FUNCTIONS = ["templatefile", "file"]

    def _scan_for_dangerous_constructs(self, code: str) -> List[str]:
        issues = []
        for provisioner in self.BLOCKED_PROVISIONERS:
            if re.search(rf'provisioner\s+"{re.escape(provisioner)}"', code):
                issues.append(f"Blocked provisioner: {provisioner}")
        for func in self.BLOCKED_FUNCTIONS:
            if re.search(rf"{func}\s*\(\s*var\.", code):
                issues.append(f"Potentially dangerous function {func} with user input")
        if re.search(r'data\s+"external"\s+"[^"]+"\s*\{', code) and "var." in code:
            issues.append("External data source with user input")
        return issues


# ==================== CALCULATOR TOOL ====================


class CalculatorTool(BaseTool):
    @staticmethod
    def add(a: float, b: float) -> ToolResult:
        try:
            return ToolResult(success=True, data={"result": a + b})
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    @staticmethod
    def subtract(a: float, b: float) -> ToolResult:
        try:
            return ToolResult(success=True, data={"result": a - b})
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    @staticmethod
    def multiply(a: float, b: float) -> ToolResult:
        try:
            return ToolResult(success=True, data={"result": a * b})
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    @staticmethod
    def divide(a: float, b: float) -> ToolResult:
        try:
            if b == 0:
                return ToolResult(success=False, error="Division by zero is not allowed")
            return ToolResult(success=True, data={"result": a / b})
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    @staticmethod
    def percentage(value: float, percentage: float) -> ToolResult:
        try:
            return ToolResult(success=True, data={"result": (value * percentage) / 100})
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    @staticmethod
    def compound_interest(principal: float, rate: float, time: float, n: int = 12) -> ToolResult:
        try:
            result = principal * (1 + rate / n) ** (n * time)
            return ToolResult(success=True, data={"result": result})
        except Exception as e:
            return ToolResult(success=False, error=str(e))


# ==================== TERRAFORM VALIDATOR TOOL ====================


class TerraformValidatorTool(BaseTool, SecureTerraformMixin):
    @secure_command_execution
    def validate_terraform_syntax(
        self, terraform_code: str, timeout: int = TF_VALIDATE_TIMEOUT
    ) -> ToolResult:
        try:
            if not terraform_code or not terraform_code.strip():
                return ToolResult(success=False, error="Empty Terraform code provided")

            security_issues = self._scan_for_dangerous_constructs(terraform_code)
            if security_issues and STRICT_TF_VALIDATION:
                return ToolResult(
                    success=False,
                    error="Security issues detected",
                    data={"security_issues": security_issues},
                )

            with _st_secure_tempdir("tf_validate_") as temp_dir:
                tf_file = Path(temp_dir) / "main.tf"
                tf_file.write_text(terraform_code, encoding="utf-8")

                try:
                    init_result = subprocess.run(
                        ["terraform", "init", "-input=false", "-no-color"],
                        cwd=temp_dir,
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                    )
                except subprocess.TimeoutExpired:
                    return ToolResult(
                        success=False, error=f"Terraform init timed out after {timeout} seconds"
                    )

                if init_result.returncode != 0:
                    return ToolResult(
                        success=False,
                        error="Terraform init failed",
                        data={"stderr": init_result.stderr, "stdout": init_result.stdout},
                    )

                try:
                    result = subprocess.run(
                        ["terraform", "validate", "-json", "-no-color"],
                        cwd=temp_dir,
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                    )
                except subprocess.TimeoutExpired:
                    return ToolResult(
                        success=False, error=f"Terraform validate timed out after {timeout} seconds"
                    )

                if result.returncode == 0:
                    try:
                        json_output = json.loads(result.stdout) if result.stdout else {}
                        return ToolResult(
                            success=True,
                            data={
                                "valid": json_output.get("valid", True) if json_output else True,
                                "output": "Terraform configuration is valid",
                                "json_output": json_output or None,
                            },
                        )
                    except json.JSONDecodeError:
                        return ToolResult(
                            success=True,
                            data={
                                "valid": True,
                                "output": result.stdout or "Terraform configuration is valid",
                            },
                        )
                else:
                    err = result.stderr or ""
                    try:
                        jo = json.loads(result.stdout) if result.stdout else {}
                        if "diagnostics" in jo:
                            msgs = []
                            for d in jo["diagnostics"]:
                                summary = d.get("summary", "Validation error")
                                detail = d.get("detail", "")
                                msgs.append(f"{summary}: {detail}" if detail else summary)
                            err = "\n".join(msgs)
                    except json.JSONDecodeError:
                        pass
                    return ToolResult(success=False, error=err, data={"stdout": result.stdout})

        except FileNotFoundError:
            return ToolResult(
                success=False,
                error="Terraform CLI not found. Please install Terraform.",
                data={"tool_missing": True},
            )
        except Exception as e:
            return self._handle_exception(e, "Validation failed")

    @secure_command_execution
    def format_terraform_code(
        self, terraform_code: str, timeout: int = TF_FMT_TIMEOUT
    ) -> ToolResult:
        try:
            if not terraform_code or not terraform_code.strip():
                return ToolResult(success=True, data={"formatted_code": terraform_code})

            with _st_secure_tempdir("tf_fmt_") as temp_dir:
                tf_file = Path(temp_dir) / "main.tf"
                tf_file.write_text(terraform_code, encoding="utf-8")

                try:
                    subprocess.run(
                        ["terraform", "fmt"],
                        cwd=temp_dir,
                        check=True,
                        capture_output=True,
                        timeout=timeout,
                    )
                except subprocess.TimeoutExpired:
                    logger.warning(
                        f"Terraform fmt timed out after {timeout} seconds, returning original"
                    )
                    return ToolResult(success=True, data={"formatted_code": terraform_code})

                formatted_code = tf_file.read_text(encoding="utf-8")
                return ToolResult(success=True, data={"formatted_code": formatted_code})

        except FileNotFoundError:
            logger.warning("Terraform CLI not found, returning unformatted code")
            return ToolResult(success=True, data={"formatted_code": terraform_code})
        except Exception as e:
            return self._handle_exception(e, "Failed to format Terraform code")

    @staticmethod
    def normalize_unicode(text: str) -> str:
        if not text:
            return text
        normalized = unicodedata.normalize("NFKC", text)
        translation_table = {
            0x2018: "'",
            0x2019: "'",
            0x201C: '"',
            0x201D: '"',
            0x2013: "-",
            0x2014: "-",
            0x00A0: " ",
        }
        return normalized.translate(translation_table)

    @staticmethod
    def autofix_common_hcl(code: str, region: Optional[str] = None) -> Tuple[str, List[str]]:
        notes: List[str] = []
        if not code:
            return code, notes

        c2 = re.sub(r'("amzn2-ami-hvm-[^"]*?)-gp2(")', r"\1-gp3\2", code)
        if c2 != code:
            notes.append("AMI filter: gp2 → gp3")
            code = c2

        c2 = re.sub(
            r'data\s+"aws_default_vpc"\s+"default"\s*{[^}]*}',
            'data "aws_vpc" "default" { default = true }',
            code,
            flags=re.S,
        )
        if c2 != code:
            notes.append("Replaced aws_default_vpc with aws_vpc.default")
            code = c2
        code = re.sub(r"data\.aws_default_vpc\.default", "data.aws_vpc.default", code)

        code = re.sub(r'filter\s*{\s*name\s*=\s*"tag:aws-cdk:subnet-type"[^}]*}', "", code)

        m = re.search(r'resource\s+"aws_security_group"\s+"(\w+)"', code)
        if m:
            sg = m.group(1)
            code = re.sub(r"aws_security_group\.\w+\.id", f"aws_security_group.{sg}.id", code)

        if region:
            code = re.sub(
                r'provider\s+"aws"\s*{[^}]*}',
                f'provider "aws" {{ region = "{region}" }}',
                code,
                flags=re.S,
            )

        return code, notes

    @secure_command_execution
    def run_fmt_validate_plan_from_code(
        self, terraform_code: str, timeout: int = TF_PLAN_TIMEOUT
    ) -> ToolResult:
        """Synchronous path used by existing call sites."""
        try:
            if not terraform_code or not terraform_code.strip():
                return ToolResult(
                    success=False,
                    error="Empty Terraform code provided",
                    data={"errors": ["empty terraform code"]},
                )

            def run(cmd, cwd):
                try:
                    p = subprocess.run(
                        cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout
                    )
                    return {"rc": p.returncode, "out": p.stdout, "err": p.stderr}
                except FileNotFoundError:
                    return {"rc": 127, "out": "", "err": f"tool not found: {cmd[0]}"}
                except subprocess.TimeoutExpired:
                    return {
                        "rc": 124,
                        "out": "",
                        "err": f"timeout after {timeout}s: {' '.join(cmd)}",
                    }

            with _st_secure_tempdir("tf_ops_") as work:
                Path(work, "main.tf").write_text(terraform_code, encoding="utf-8", errors="ignore")

                init = run(["terraform", "init", "-input=false", "-no-color"], work)
                fmt = run(["terraform", "fmt", "-check", "-no-color"], work)
                val = run(["terraform", "validate", "-no-color"], work)
                plan = run(
                    ["terraform", "plan", "-detailed-exitcode", "-no-color", "-input=false"], work
                )

                result_data = {
                    "fmt_ok": fmt["rc"] == 0,
                    "validate_ok": val["rc"] == 0,
                    "plan_exit_code": plan["rc"],
                    "plan_output": (plan["out"] or "") + (plan["err"] or ""),
                    "errors": [],
                }

                if init["rc"] != 0:
                    result_data["errors"].append(f"init: {init['err']}")
                if fmt["rc"] != 0:
                    result_data["errors"].append(f"fmt: {fmt['err']}")
                if val["rc"] != 0:
                    result_data["errors"].append(f"validate: {val['err']}")
                if plan["rc"] not in (0, 2):
                    result_data["errors"].append(f"plan: {plan['err']}")

                m = re.search(
                    r"Plan:\s+\d+\s+to\s+add,\s+\d+\s+to\s+change,\s+\d+\s+to\s+destroy",
                    result_data["plan_output"],
                )
                result_data["plan_summary"] = m.group(0) if m else ""

                return ToolResult(success=True, data=result_data)

        except Exception as e:
            return self._handle_exception(e, "Failed to run Terraform operations")

    @secure_command_execution_async
    async def run_fmt_validate_plan_from_code_async(
        self, terraform_code: str, timeout: int = TF_PLAN_TIMEOUT
    ) -> ToolResult:
        """Async variant using run_cmd_async to avoid blocking FastAPI event loop."""
        try:
            if not terraform_code or not terraform_code.strip():
                return ToolResult(
                    success=False,
                    error="Empty Terraform code provided",
                    data={"errors": ["empty terraform code"]},
                )

            async def run(cwd, *cmd):
                rc, out, err = await _st_run_cmd_async(*cmd, cwd=cwd, timeout=timeout)
                return {"rc": rc, "out": out, "err": err}

            with _st_secure_tempdir("tf_ops_async_") as work:
                Path(work, "main.tf").write_text(terraform_code, encoding="utf-8", errors="ignore")

                init = await run(work, "terraform", "init", "-input=false", "-no-color")
                fmt = await run(work, "terraform", "fmt", "-check", "-no-color")
                val = await run(work, "terraform", "validate", "-no-color")
                plan = await run(
                    work, "terraform", "plan", "-detailed-exitcode", "-no-color", "-input=false"
                )

                result_data = {
                    "fmt_ok": fmt["rc"] == 0,
                    "validate_ok": val["rc"] == 0,
                    "plan_exit_code": plan["rc"],
                    "plan_output": (plan["out"] or "") + (plan["err"] or ""),
                    "errors": [],
                }
                if init["rc"] != 0:
                    result_data["errors"].append(f"init: {init['err']}")
                if fmt["rc"] != 0:
                    result_data["errors"].append(f"fmt: {fmt['err']}")
                if val["rc"] != 0:
                    result_data["errors"].append(f"validate: {val['err']}")
                if plan["rc"] not in (0, 2):
                    result_data["errors"].append(f"plan: {plan['err']}")

                m = re.search(
                    r"Plan:\s+\d+\s+to\s+add,\s+\d+\s+to\s+change,\s+\d+\s+to\s+destroy",
                    result_data["plan_output"],
                )
                result_data["plan_summary"] = m.group(0) if m else ""

                return ToolResult(success=True, data=result_data)

        except Exception as e:
            return self._handle_exception(e, "Failed to run Terraform operations (async)")


# ==================== AWS PRICING TOOL ====================


class AWSPricingTool(BaseTool):
    FALLBACK_PRICING_DATA = {
        "ec2": {
            "t2.nano": {"hourly": 0.0058, "monthly": 4.23},
            "t2.micro": {"hourly": 0.0116, "monthly": 8.47},
            "t3.micro": {"hourly": 0.0104, "monthly": 7.59},
            "t3.small": {"hourly": 0.0208, "monthly": 15.18},
            "m5.large": {"hourly": 0.096, "monthly": 70.08},
            "c5.large": {"hourly": 0.085, "monthly": 62.05},
        },
        "rds": {
            "db.t3.micro": {"hourly": 0.017, "monthly": 12.41},
            "db.t3.small": {"hourly": 0.034, "monthly": 24.82},
        },
        "s3": {
            "standard": {"per_gb_month": 0.023},
            "standard_ia": {"per_gb_month": 0.0125},
        },
        "vpc": {
            "nat_gateway": {"hourly": 0.045, "monthly": 32.85},
            "data_processing_per_gb": 0.045,
        },
    }

    def __init__(self, use_live_pricing: bool = True):
        super().__init__()
        self.use_live_pricing = use_live_pricing

    def _fetch_live_pricing(self, service: str, resource_type: str, region: str) -> Optional[float]:
        # TODO: integrate the AWS Price List API if desired
        return None

    def get_pricing(self, service: str, resource_type: str, region: str = "us-east-1") -> float:
        if self.use_live_pricing:
            live_price = self._fetch_live_pricing(service, resource_type, region)
            if live_price is not None:
                return live_price

        service_data = self.FALLBACK_PRICING_DATA.get(service, {})
        resource_data = service_data.get(resource_type, {})
        hourly = resource_data.get("hourly")
        monthly = resource_data.get("monthly")
        if monthly is not None:
            return monthly
        if hourly is not None:
            return hourly * 730
        return 0.0

    def estimate_ec2_cost(
        self,
        instance_type: str,
        count: int = 1,
        hours_per_month: int = 730,
        region: str = "us-east-1",
    ) -> ToolResult:
        try:
            monthly_unit = self.get_pricing("ec2", instance_type, region)
            monthly_cost = monthly_unit * count * (hours_per_month / 730)
            hourly_rate = (monthly_unit / 730) if monthly_unit else 0.0
            return ToolResult(
                success=True,
                data={
                    "instance_type": instance_type,
                    "count": count,
                    "hourly_rate": hourly_rate,
                    "hours_per_month": hours_per_month,
                    "monthly_cost": round(monthly_cost, 2),
                    "yearly_cost": round(monthly_cost * 12, 2),
                    "source": "live" if self.use_live_pricing else "fallback",
                },
            )
        except Exception as e:
            return self._handle_exception(e, "Failed to estimate EC2 cost")

    def estimate_rds_cost(
        self,
        db_instance_class: str,
        storage_gb: int = 20,
        storage_type: str = "gp2",
        region: str = "us-east-1",
    ) -> ToolResult:
        try:
            monthly_instance = self.get_pricing("rds", db_instance_class, region)
            hourly_rate = (monthly_instance / 730) if monthly_instance else 0.0

            storage_rates = {"gp2": 0.115, "gp3": 0.092, "io1": 0.125}
            storage_rate = storage_rates.get(storage_type, 0.115)
            monthly_storage_cost = storage_gb * storage_rate
            total_monthly = monthly_instance + monthly_storage_cost

            return ToolResult(
                success=True,
                data={
                    "db_instance_class": db_instance_class,
                    "storage_gb": storage_gb,
                    "storage_type": storage_type,
                    "monthly_instance_cost": round(monthly_instance, 2),
                    "monthly_storage_cost": round(monthly_storage_cost, 2),
                    "total_monthly_cost": round(total_monthly, 2),
                    "yearly_cost": round(total_monthly * 12, 2),
                    "hourly_rate": hourly_rate,
                    "source": "live" if self.use_live_pricing else "fallback",
                },
            )
        except Exception as e:
            return self._handle_exception(e, "Failed to estimate RDS cost")


# ==================== GIT INTEGRATION TOOL ====================


class GitIntegrationTool(BaseTool):
    @secure_command_execution
    def initialize_git_repo(self, timeout: int = 15) -> ToolResult:
        try:
            if not Path(".git").exists():
                try:
                    result = subprocess.run(
                        ["git", "init"], check=True, capture_output=True, text=True, timeout=timeout
                    )
                    logger.info("Git repository initialized")
                    return ToolResult(
                        success=True,
                        data={"message": "Git repository initialized", "output": result.stdout},
                    )
                except subprocess.TimeoutExpired:
                    return ToolResult(
                        success=False, error=f"Git init timed out after {timeout} seconds"
                    )
            else:
                return ToolResult(success=True, data={"message": "Git repository already exists"})
        except subprocess.CalledProcessError as e:
            return ToolResult(
                success=False, error=f"Failed to initialize git: {e.stderr or str(e)}"
            )
        except FileNotFoundError:
            return ToolResult(
                success=False,
                error="Git not found. Please install Git.",
                data={"tool_missing": True},
            )

    def create_gitignore(self, project_type: str = "terraform") -> ToolResult:
        try:
            gitignore_templates = {
                "terraform": """# Local .terraform directories
**/.terraform/*

# .tfstate files
*.tfstate
*.tfstate.*

# Crash log files
crash.log
crash.*.log

# Exclude all .tfvars files, which are likely to contain sensitive data
*.tfvars
*.tfvars.json

# Override files
override.tf
override.tf.json
*_override.tf
*_override.tf.json

# Plans
*tfplan*

# CLI config
.terraformrc
terraform.rc""",
                "general": """# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Python
__pycache__/
*.pyc
*.pyo
*.pyd
env/
venv/
.env
.env.local

# Node
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# Logs / temp
*.log
*.tmp
*.temp""",
            }

            terraform_ignore = gitignore_templates.get("terraform", "")
            general_ignore = gitignore_templates.get("general", "")
            gitignore_content = f"{terraform_ignore}\n\n{general_ignore}"

            Path(".gitignore").write_text(gitignore_content.strip(), encoding="utf-8")
            return ToolResult(
                success=True,
                data={"message": f".gitignore created successfully for {project_type} project"},
            )
        except Exception as e:
            return self._handle_exception(e, "Failed to create .gitignore")

    @secure_command_execution
    def commit_changes_with_validation(self, message: str = None, timeout: int = 30) -> ToolResult:
        try:
            if not message:
                message = f"Terraform code update - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

            if not Path(".git").exists():
                return ToolResult(
                    success=False, error="Git repository not initialized. Run git init first."
                )

            tf_files = list(Path(".").glob("*.tf"))
            if tf_files:
                logger.info(f"Validating {len(tf_files)} Terraform files before commit")
                validation_errors: List[str] = []
                for tf_file in tf_files:
                    try:
                        content = tf_file.read_text(encoding="utf-8", errors="ignore")
                        if content.strip() and content.count("{") != content.count("}"):
                            validation_errors.append(f"{tf_file.name}: Mismatched braces")
                    except Exception as e:
                        validation_errors.append(f"{tf_file.name}: {str(e)}")
                if validation_errors:
                    return ToolResult(
                        success=False,
                        error=f"Pre-commit validation failed: {'; '.join(validation_errors)}",
                    )

            try:
                subprocess.run(
                    ["git", "add", "."], check=True, capture_output=True, timeout=timeout
                )
            except subprocess.TimeoutExpired:
                return ToolResult(success=False, error=f"Git add timed out after {timeout} seconds")

            try:
                result = subprocess.run(
                    ["git", "status", "--porcelain"],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired:
                return ToolResult(
                    success=False, error=f"Git status check timed out after {timeout} seconds"
                )

            if not result.stdout.strip():
                return ToolResult(success=True, data={"message": "No changes to commit"})

            try:
                commit_result = subprocess.run(
                    ["git", "commit", "-m", message],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                return ToolResult(
                    success=True,
                    data={
                        "message": f"Changes committed: {message}",
                        "output": commit_result.stdout,
                    },
                )
            except subprocess.TimeoutExpired:
                return ToolResult(
                    success=False, error=f"Git commit timed out after {timeout} seconds"
                )

        except subprocess.CalledProcessError as e:
            return ToolResult(
                success=False, error=f"Failed to commit changes: {e.stderr if e.stderr else str(e)}"
            )
        except FileNotFoundError:
            return ToolResult(
                success=False,
                error="Git not found. Please install Git.",
                data={"tool_missing": True},
            )

    def commit_changes(self, message: str = None, timeout: int = 30) -> ToolResult:
        return self.commit_changes_with_validation(message, timeout)

    @secure_command_execution
    def push_to_github(self, repo_url: str, branch: str = "main", timeout: int = 60) -> ToolResult:
        try:
            try:
                subprocess.run(
                    ["git", "remote", "add", "origin", repo_url],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout // 4,
                )
            except subprocess.CalledProcessError:
                subprocess.run(
                    ["git", "remote", "set-url", "origin", repo_url],
                    check=True,
                    timeout=timeout // 4,
                )
            except subprocess.TimeoutExpired:
                return ToolResult(success=False, error="Remote setup timed out")

            try:
                push_result = subprocess.run(
                    ["git", "push", "-u", "origin", branch],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                return ToolResult(
                    success=True,
                    data={
                        "message": f"Successfully pushed to GitHub: {repo_url}",
                        "output": push_result.stdout,
                    },
                )
            except subprocess.TimeoutExpired:
                return ToolResult(
                    success=False, error=f"Git push timed out after {timeout} seconds"
                )

        except subprocess.CalledProcessError as e:
            return ToolResult(
                success=False, error=f"Failed to push to GitHub: {e.stderr if e.stderr else str(e)}"
            )
        except FileNotFoundError:
            return ToolResult(
                success=False,
                error="Git not found. Please install Git.",
                data={"tool_missing": True},
            )


# ==================== INFRASTRUCTURE ANALYZER (AST-based) ====================


class InfrastructureAnalyzer(BaseTool, SecureTerraformMixin):
    def _parse_hcl(self, terraform_code: str) -> Dict[str, Any]:
        """Parse HCL using python-hcl2 (via StringIO to avoid writing to disk)."""
        try:
            return _st_hcl2.load(_ST_StringIO(terraform_code))
        except Exception as e:
            logger.debug(f"HCL parse failed: {e}")
            return {}

    def _extract_from_ast(self, ast: Dict[str, Any]) -> Dict[str, Any]:
        resources: List[Dict[str, str]] = []
        providers: List[str] = []
        variables: List[str] = []
        outputs: List[str] = []

        for pblk in ast.get("provider", []) or []:
            if isinstance(pblk, dict):
                providers.extend(pblk.keys())

        for vblk in ast.get("variable", []) or []:
            if isinstance(vblk, dict):
                variables.extend(vblk.keys())

        for oblk in ast.get("output", []) or []:
            if isinstance(oblk, dict):
                outputs.extend(oblk.keys())

        for rblk in ast.get("resource", []) or []:
            if not isinstance(rblk, dict):
                continue
            for rtype, entries in rblk.items():
                if isinstance(entries, list):
                    for entry in entries:
                        if isinstance(entry, dict):
                            for rname in entry.keys():
                                resources.append({"type": rtype, "name": rname})

        return {
            "resources": resources,
            "providers": list(dict.fromkeys(providers)),
            "variables": variables,
            "outputs": outputs,
        }

    def _regex_fallback_scan(self, code: str) -> Dict[str, Any]:
        resources = []
        for rtype, rname in re.findall(
            r'resource\s+"([^"]+)"\s+"([^"]+)"\s*\{', code, re.MULTILINE
        ):
            resources.append({"type": rtype, "name": rname})
        providers = list(set(re.findall(r'provider\s+"([^"]+)"\s*\{', code, re.MULTILINE)))
        variables = re.findall(r'variable\s+"([^"]+)"\s*\{', code, re.MULTILINE)
        outputs = re.findall(r'output\s+"([^"]+)"\s*\{', code, re.MULTILINE)
        return {
            "resources": resources,
            "providers": providers,
            "variables": variables,
            "outputs": outputs,
        }

    def analyze_terraform_resources(self, terraform_code: str) -> ToolResult:
        try:
            if not terraform_code or not terraform_code.strip():
                return ToolResult(
                    success=False,
                    error="Empty Terraform code provided",
                    data={"resources": [], "providers": [], "variables": [], "outputs": []},
                )

            security_issues = self._scan_for_dangerous_constructs(terraform_code)

            ast = self._parse_hcl(terraform_code)
            parsed = (
                self._extract_from_ast(ast) if ast else self._regex_fallback_scan(terraform_code)
            )

            analysis: Dict[str, Any] = {
                **parsed,
                "estimated_monthly_cost": 0,
                "security_concerns": [],
                "optimization_suggestions": [],
                "resource_count": len(parsed["resources"]),
                "compliance_checks": [],
                "blast_radius_score": 0,
                "security_issues": security_issues,
            }

            security_checks = [
                (
                    r'password\s*=\s*"[^"]*"',
                    "🔐 Hardcoded password detected - use variables or a secrets manager",
                    "HIGH",
                ),
                (
                    r'secret\s*=\s*"[^"]*"',
                    "🔐 Hardcoded secret detected - use a secrets manager",
                    "HIGH",
                ),
                (
                    r'access_key\s*=\s*"[^"]*"',
                    "🔐 Hardcoded access key - prefer IAM roles",
                    "CRITICAL",
                ),
                (
                    r'secret_key\s*=\s*"[^"]*"',
                    "🔐 Hardcoded secret key - prefer IAM roles",
                    "CRITICAL",
                ),
                (
                    r'cidr_blocks\s*=\s*\["0\.0\.0\.0/0"\]',
                    "🌐 Open security group rule (0.0.0.0/0) - restrict access",
                    "HIGH",
                ),
                (
                    r"publicly_accessible\s*=\s*true",
                    "🌐 Publicly accessible resource - verify necessity",
                    "HIGH",
                ),
                (
                    r"force_destroy\s*=\s*true",
                    "⚠️ Destructive S3 behavior (force_destroy) - double check",
                    "MEDIUM",
                ),
            ]
            for pattern, message, severity in security_checks:
                if re.search(pattern, terraform_code, re.IGNORECASE | re.DOTALL):
                    analysis["security_concerns"].append(
                        {"message": message, "severity": severity, "category": severity}
                    )

            critical = len(
                [c for c in analysis["security_concerns"] if c.get("severity") == "CRITICAL"]
            )
            high = len([c for c in analysis["security_concerns"] if c.get("severity") == "HIGH"])
            medium = len(
                [c for c in analysis["security_concerns"] if c.get("severity") == "MEDIUM"]
            )
            analysis["blast_radius_score"] = min(100, critical * 30 + high * 15 + medium * 5)

            analysis["estimated_monthly_cost"] = self._estimate_enhanced_cost(
                analysis["resources"], terraform_code
            )
            analysis["optimization_suggestions"] = self._generate_optimization_suggestions(
                terraform_code, analysis["resources"]
            )

            return ToolResult(success=True, data=analysis)

        except Exception as e:
            return self._handle_exception(e, "Analysis failed")

    def _estimate_enhanced_cost(self, resources: List[dict], terraform_code: str) -> float:
        total_cost = 0.0
        cost_mapping: Dict[str, Any] = {
            "aws_instance": {
                "base": 30,
                "patterns": [
                    (r"t2\.micro", 8.5),
                    (r"t2\.small", 17),
                    (r"t3\.micro", 7.5),
                    (r"t3\.small", 15),
                    (r"t3\.medium", 30),
                    (r"t3\.large", 60),
                    (r"m5\.large", 70),
                    (r"m5\.xlarge", 140),
                    (r"c5\.large", 65),
                    (r"r5\.large", 90),
                ],
            },
            "aws_rds_instance": {
                "base": 50,
                "patterns": [
                    (r"db\.t3\.micro", 12),
                    (r"db\.t3\.small", 25),
                    (r"db\.t3\.medium", 50),
                    (r"db\.t3\.large", 100),
                    (r"db\.m5\.large", 140),
                    (r"multi_az\s*=\s*true", 2.0),
                ],
            },
            "aws_s3_bucket": 5,
            "aws_nat_gateway": 33,
            "aws_lb": 16,
            "aws_application_load_balancer": 18,
            "aws_network_load_balancer": 16,
            "aws_elasticache_cluster": 25,
            "aws_eks_cluster": 73,
            "aws_lambda_function": 2,
        }

        for r in resources:
            rtype = r.get("type", "")
            base_cost = 0.0
            if rtype in cost_mapping:
                info = cost_mapping[rtype]
                if isinstance(info, dict):
                    base_cost = info["base"]
                    for pattern, adjust in info.get("patterns", []):
                        if re.search(pattern, terraform_code, re.IGNORECASE):
                            base_cost = adjust if adjust > 10 else base_cost * adjust
                            break
                else:
                    base_cost = float(info)
            total_cost += base_cost

        region_multipliers = {
            r"us-east-1": 1.0,
            r"us-west-2": 1.05,
            r"eu-west-1": 1.1,
            r"ap-southeast-1": 1.15,
            r"ap-northeast-1": 1.2,
        }
        for region_pattern, mult in region_multipliers.items():
            if re.search(f"region.*{region_pattern}", terraform_code, re.IGNORECASE):
                total_cost *= mult
                break
        return round(total_cost, 2)

    def _generate_optimization_suggestions(
        self, terraform_code: str, resources: List[dict]
    ) -> List[str]:
        suggestions: List[str] = []
        if "required_version" not in terraform_code:
            suggestions.append("📌 Pin Terraform version: add required_version constraint")
        if "required_providers" not in terraform_code:
            suggestions.append("📌 Pin provider versions: add required_providers block")
        if "tags" not in terraform_code and any(
            res["type"].startswith("aws_") for res in resources
        ):
            suggestions.append("🏷️ Add consistent resource tags for cost tracking and management")
        if "backend" not in terraform_code:
            suggestions.append(
                "💾 Configure remote state backend (S3 + DynamoDB) for team collaboration"
            )
        return suggestions

    def suggest_best_practices(self, terraform_code: str) -> ToolResult:
        try:
            if not terraform_code:
                return ToolResult(
                    success=False,
                    error="Provide Terraform code for analysis",
                    data={"suggestions": []},
                )

            suggestions: List[str] = []
            if "required_version" not in terraform_code:
                suggestions.append("Add Terraform version constraints using required_version")
            if "required_providers" not in terraform_code:
                suggestions.append("Pin provider versions using required_providers block")
            if "tags" not in terraform_code and "aws_" in terraform_code:
                suggestions.append(
                    "Add consistent tags to AWS resources for better management and cost tracking"
                )
            if "backend" not in terraform_code:
                suggestions.append(
                    "Configure remote state backend (S3 + DynamoDB) for team collaboration"
                )

            return ToolResult(success=True, data={"suggestions": suggestions})
        except Exception as e:
            return self._handle_exception(e, "Failed to generate best practices")


# ==================== FILE OPERATIONS ====================


class FileOperations(BaseTool):
    def save_terraform_file(self, content: str, filename: str = "main.tf") -> ToolResult:
        try:
            if not content or not content.strip():
                return ToolResult(success=False, error="Cannot save empty content")

            file_path = Path(filename)
            file_path.write_text(content, encoding="utf-8")
            file_size = file_path.stat().st_size
            return ToolResult(
                success=True,
                data={
                    "message": f"Terraform code saved to {filename}",
                    "file_path": str(file_path.absolute()),
                    "file_size": file_size,
                },
            )
        except Exception as e:
            return self._handle_exception(e, "Failed to save file")

    def create_terraform_structure(self, project_name: str = "terraform-project") -> ToolResult:
        try:
            base_dir = Path(project_name)
            base_dir.mkdir(exist_ok=True)

            directories = [
                "modules",
                "environments/dev",
                "environments/staging",
                "environments/prod",
            ]
            for directory in directories:
                (base_dir / directory).mkdir(parents=True, exist_ok=True)

            files = {
                "main.tf": """# Main infrastructure resources
terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}
""",
                "variables.tf": """# Input variables
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "dev"
}

variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "my-project"
}
""",
                "outputs.tf": """# Output values
output "region" {
  description = "AWS region"
  value       = var.aws_region
}
""",
                "versions.tf": """terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}
""",
                "README.md": f"""# {project_name}

Terraform infrastructure project

## Structure

- `main.tf` - Main infrastructure resources
- `variables.tf` - Input variables
- `outputs.tf` - Output values
- `versions.tf` - Terraform and provider version constraints
- `modules/` - Reusable modules
- `environments/` - Environment-specific configurations
""",
            }

            for filename, content in files.items():
                (base_dir / filename).write_text(content, encoding="utf-8")

            return ToolResult(
                success=True,
                data={
                    "message": f"Terraform project structure created in {project_name}/",
                    "created": {
                        "directories": directories,
                        "files": list(files.keys()),
                        "project_path": str(base_dir.absolute()),
                    },
                },
            )
        except Exception as e:
            return self._handle_exception(e, "Failed to create structure")

    def read_terraform_file(self, filename: str = "main.tf") -> ToolResult:
        try:
            file_path = Path(filename)
            if not file_path.exists():
                return ToolResult(success=False, error=f"File {filename} does not exist")

            content = file_path.read_text(encoding="utf-8")
            file_size = file_path.stat().st_size
            return ToolResult(
                success=True,
                data={
                    "content": content,
                    "file_path": str(file_path.absolute()),
                    "file_size": file_size,
                },
            )
        except Exception as e:
            return self._handle_exception(e, "Failed to read file")


# ==================== TERRAFORM PLAN ANALYZER ====================


class TerraformPlanAnalyzer(BaseTool):
    def analyze_plan_output(self, plan_output: str) -> ToolResult:
        try:
            if not plan_output:
                return ToolResult(
                    success=False,
                    error="Empty plan output provided",
                    data={
                        "resources_to_create": 0,
                        "resources_to_update": 0,
                        "resources_to_destroy": 0,
                        "changes": [],
                        "warnings": [],
                        "errors": [],
                    },
                )

            analysis = {
                "resources_to_create": 0,
                "resources_to_update": 0,
                "resources_to_destroy": 0,
                "changes": [],
                "warnings": [],
                "errors": [],
            }

            creates = re.findall(r"#\s+(.+?)\s+will be created", plan_output)
            updates = re.findall(r"#\s+(.+?)\s+will be updated", plan_output)
            destroys = re.findall(r"#\s+(.+?)\s+will be destroyed", plan_output)

            analysis["resources_to_create"] = len(creates)
            analysis["resources_to_update"] = len(updates)
            analysis["resources_to_destroy"] = len(destroys)

            summary_match = re.search(
                r"Plan:\s+(\d+)\s+to\s+add,\s+(\d+)\s+to\s+change,\s+(\d+)\s+to\s+destroy",
                plan_output,
            )
            if summary_match:
                analysis["summary"] = {
                    "add": int(summary_match.group(1)),
                    "change": int(summary_match.group(2)),
                    "destroy": int(summary_match.group(3)),
                }

            analysis["warnings"] = [line for line in plan_output.split("\n") if "Warning:" in line]
            analysis["errors"] = [line for line in plan_output.split("\n") if "Error:" in line]

            return ToolResult(success=True, data=analysis)
        except Exception as e:
            return self._handle_exception(e, "Plan analysis failed")


# ==================== STRANDS TOOLS AGGREGATE ====================


class StrandsTools:
    def __init__(self):
        self.calculator = CalculatorTool()
        self.terraform_validator = TerraformValidatorTool()
        self.aws_pricing = AWSPricingTool()
        self.git_integration = GitIntegrationTool()
        self.infrastructure_analyzer = InfrastructureAnalyzer()
        self.file_operations = FileOperations()
        self.plan_analyzer = TerraformPlanAnalyzer()

    def get_all_tools(self) -> Dict[str, Any]:
        return {
            "calculator": self.calculator,
            "terraform_validator": self.terraform_validator,
            "aws_pricing": self.aws_pricing,
            "git_integration": self.git_integration,
            "infrastructure_analyzer": self.infrastructure_analyzer,
            "file_operations": self.file_operations,
            "plan_analyzer": self.plan_analyzer,
        }

    def get_tool_info(self) -> Dict[str, str]:
        return {
            "calculator": "Basic mathematical operations for cost calculations",
            "terraform_validator": "Validate and format Terraform code with enhanced security",
            "aws_pricing": "Estimate AWS infrastructure costs with regional pricing",
            "git_integration": "Git repository management and GitHub integration with validation",
            "infrastructure_analyzer": "Comprehensive Terraform analysis with security and compliance checks",
            "file_operations": "File system operations for Terraform projects",
            "plan_analyzer": "Analyze Terraform plan outputs for changes and impacts",
        }

    def get_health_status(self) -> ToolResult:
        try:
            health = {"overall_status": "healthy", "tools": {}, "issues": []}
            checks = [("terraform", ["terraform", "--version"]), ("git", ["git", "--version"])]

            for tool_name, command in checks:
                try:
                    result = subprocess.run(command, capture_output=True, text=True, timeout=5)
                    health["tools"][tool_name] = {
                        "available": result.returncode == 0,
                        "version": (
                            result.stdout.split("\n")[0] if result.returncode == 0 else None
                        ),
                    }
                    if result.returncode != 0:
                        health["issues"].append(f"{tool_name} not working properly")
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    health["tools"][tool_name] = {"available": False, "version": None}
                    health["issues"].append(f"{tool_name} not installed or not in PATH")

            terraform_available = health["tools"].get("terraform", {}).get("available", False)
            if not terraform_available:
                health["overall_status"] = "degraded"
            if len(health["issues"]) > 2:
                health["overall_status"] = "unhealthy"

            return ToolResult(success=True, data=health)
        except Exception as e:
            return ToolResult(success=False, error=str(e))


# ==================== UTILS & EXPORTS ====================

st_tools = StrandsTools()
calculator = st_tools.calculator
terraform_validator = st_tools.terraform_validator
aws_pricing = st_tools.aws_pricing
git_integration = st_tools.git_integration
infrastructure_analyzer = st_tools.infrastructure_analyzer
file_operations = st_tools.file_operations
plan_analyzer = st_tools.plan_analyzer


def calculate(operation: str, a: float, b: float = None) -> float:
    try:
        if operation == "add" and b is not None:
            return calculator.add(a, b).data["result"]
        elif operation == "subtract" and b is not None:
            return calculator.subtract(a, b).data["result"]
        elif operation == "multiply" and b is not None:
            return calculator.multiply(a, b).data["result"]
        elif operation == "divide" and b is not None:
            res = calculator.divide(a, b)
            return res.data["result"] if res.success else 0
        else:
            raise ValueError(f"Unsupported operation: {operation} or missing parameter")
    except Exception as e:
        logger.error(f"Calculation error: {str(e)}")
        raise


def static_sanity_checks(hcl_text: str) -> List[str]:
    issues: List[str] = []
    txt = hcl_text or ""

    if re.search(r"\baws_alb(?:_listener|_target_group)?\b", txt):
        issues.append("Use aws_lb/aws_lb_listener/aws_lb_target_group; aws_alb* is not supported.")
    if not re.search(r'\bresource\s+"aws_lb"\b', txt):
        issues.append('Missing Application Load Balancer: add resource "aws_lb".')
    if not re.search(r'\bresource\s+"aws_lb_target_group"\b', txt):
        issues.append(
            'Missing target group: add resource "aws_lb_target_group" with target_type="ip" for Fargate.'
        )
    if not re.search(r'\bresource\s+"aws_lb_listener"\b', txt):
        issues.append('Missing listener: add resource "aws_lb_listener" on port 80.')
    if re.search(r'\bresource\s+"aws_lb_listener"\s+"[^"]+"\s*{[^}]*\btags\s*=', txt, re.S):
        issues.append("aws_lb_listener does not support tags.")

    if re.search(r'\bresource\s+"aws_ecs_task_definition"\b', txt):
        if 'requires_compatibilities = ["FARGATE"]' not in txt:
            issues.append("Task definition must require FARGATE.")
        if 'network_mode = "awsvpc"' not in txt:
            issues.append('Task definition must set network_mode="awsvpc".')
    if re.search(r'\bresource\s+"aws_ecs_service"\b', txt):
        if 'launch_type = "FARGATE"' not in txt:
            issues.append('ECS service must set launch_type="FARGATE".')
        m = re.search(
            r'resource\s+"aws_ecs_service"\s+"[^"]+"\s*{[^}]*network_configuration\s*{(?P<body>[^}]*)}',
            txt,
            re.S,
        )
        if m:
            net = m.group("body")
            if "assign_public_ip = false" not in net:
                issues.append("ECS service must set assign_public_ip = false (NAT-less design).")
            if re.search(r"aws_subnet\.public", net):
                issues.append(
                    "ECS service should use private subnets (not public) in network_configuration.subnets."
                )
        else:
            issues.append("ECS service missing network_configuration block.")
    if re.search(r'\bresource\s+"aws_lb_target_group"\b', txt) and 'target_type = "ip"' not in txt:
        issues.append('LB target group must set target_type="ip" for Fargate.')
    return issues


def extract_best_terraform_block(response_text: str) -> str:
    if not response_text:
        return ""
    labelled = re.findall(
        r"```(?:terraform|hcl|tf)\s*\n(.*?)```", response_text, re.DOTALL | re.IGNORECASE
    )
    candidates = labelled[:]
    any_fenced = re.findall(r"```\s*\n(.*?)```", response_text, re.DOTALL)
    candidates.extend(any_fenced)
    hcl_spans = re.findall(
        r"((?:terraform|provider|variable|output|data|module|resource)\s+[^{]*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})",
        response_text,
        re.DOTALL | re.IGNORECASE,
    )
    candidates.extend(hcl_spans)
    if not candidates:
        return ""

    def score(txt: str) -> int:
        tokens = sum(
            1
            for t in ["resource ", "provider ", "variable ", "output ", "module ", "terraform "]
            if t in txt
        )
        return len(txt) + 200 * tokens

    best = max(candidates, key=score)
    return best.strip()


def ensure_minimal_scaffold(terraform_code: str, default_region: str = "us-east-1") -> str:
    code = terraform_code or ""
    needs_tf = "terraform {" not in code
    needs_provider = 'provider "aws"' not in code
    if not (needs_tf or needs_provider):
        return code
    scaffold_parts = []
    if needs_tf:
        scaffold_parts.append(
            "terraform {\n"
            '  required_version = ">= 1.3.0"\n'
            "  required_providers {\n"
            "    aws = {\n"
            '      source  = "hashicorp/aws"\n'
            '      version = "~> 5.0"\n'
            "    }\n"
            "  }\n"
            "}\n"
        )
    if needs_provider:
        scaffold_parts.append('provider "aws" {\n' f'  region = "{default_region}"\n' "}\n")
    return ("\n".join(scaffold_parts) + "\n" + code).strip()


# -------------------- Strands-compatible wrappers --------------------


@strands_tool(name="terraform_format", description="Format Terraform code")
def st_terraform_format(code: str) -> dict:
    tr = terraform_validator.format_terraform_code(code)
    if getattr(tr, "success", False) and tr.data:
        return tr.data  # {"formatted_code": "..."}
    return {"error": getattr(tr, "error", "format failed")}


@strands_tool(name="terraform_validate", description="Validate Terraform syntax")
def st_terraform_validate(code: str) -> dict:
    tr = terraform_validator.validate_terraform_syntax(code)
    if getattr(tr, "success", False) and tr.data:
        return tr.data  # {"valid": bool, "output": "..."}
    return {"valid": False, "error": getattr(tr, "error", "validation failed")}


@strands_tool(name="terraform_plan", description="Run fmt/validate/plan from Terraform code")
def st_terraform_plan(code: str, timeout: int = TF_PLAN_TIMEOUT) -> dict:
    tr = terraform_validator.run_fmt_validate_plan_from_code(code, timeout=timeout)
    if getattr(tr, "success", False) and tr.data:
        return tr.data
    return {"error": getattr(tr, "error", "plan failed")}


@strands_tool(name="analyze_resources", description="Analyze Terraform resources & risks")
def st_analyze_resources(code: str) -> dict:
    tr = infrastructure_analyzer.analyze_terraform_resources(code)
    if getattr(tr, "success", False) and tr.data:
        return tr.data
    return {"error": getattr(tr, "error", "analyze failed")}


@strands_tool(name="best_practices", description="Suggest Terraform best practices")
def st_best_practices(code: str) -> dict:
    tr = infrastructure_analyzer.suggest_best_practices(code)
    if getattr(tr, "success", False) and tr.data:
        return tr.data  # {"suggestions":[...]}
    return {"suggestions": [], "error": getattr(tr, "error", "no suggestions")}


@strands_tool(name="save_file", description="Save Terraform code to a file")
def st_save_file(code: str, path: str = "main.tf") -> dict:
    tr = file_operations.save_terraform_file(code, path)
    if getattr(tr, "success", False):
        return {"success": True, "path": path}
    return {"success": False, "error": getattr(tr, "error", "save failed")}


@strands_tool(name="static_sanity", description="Static sanity checks for Terraform")
def st_static_sanity(code: str) -> dict:
    try:
        issues = static_sanity_checks(code)
        return {"issues": issues or []}
    except Exception as e:
        return {"issues": [], "error": str(e)}


@strands_tool(name="plan_analyze", description="Analyze 'terraform plan' output")
def st_plan_analyze(plan_output: str) -> dict:
    tr = plan_analyzer.analyze_plan_output(plan_output)
    if getattr(tr, "success", False) and tr.data:
        return tr.data
    return {"error": getattr(tr, "error", "plan analyze failed")}


@strands_tool(name="calc_multiply", description="Multiply two numbers")
def st_calc_multiply(a: float, b: float) -> dict:
    res = calculator.multiply(a, b)
    if getattr(res, "success", False) and res.data:
        return {"result": res.data.get("result")}
    return {"error": getattr(res, "error", "multiply failed")}


# Export the list Strands Agent expects
tools_for_strands = [
    st_terraform_format,
    st_terraform_validate,
    st_terraform_plan,
    st_analyze_resources,
    st_best_practices,
    st_save_file,
    st_static_sanity,
    st_plan_analyze,
    st_calc_multiply,
]

# Keep a compat alias some code may import
tools = tools_for_strands

__all__ = [
    "tools_for_strands",
    "tools",
    "st_tools",
    "calculator",
    "terraform_validator",
    "aws_pricing",
    "git_integration",
    "infrastructure_analyzer",
    "file_operations",
    "plan_analyzer",
    "calculate",
    "StrandsTools",
    "CalculatorTool",
    "TerraformValidatorTool",
    "AWSPricingTool",
    "GitIntegrationTool",
    "InfrastructureAnalyzer",
    "FileOperations",
    "TerraformPlanAnalyzer",
    "static_sanity_checks",
    "extract_best_terraform_block",
    "ensure_minimal_scaffold",
    "ToolResult",
    "SecurityException",
    "DangerousConstructException",
    "CircuitBreakerOpenException",
]


if __name__ == "__main__":
    # Debug tool registration
    print("Available tools:")
    for tool_func in tools_for_strands:
        print(f"  - {tool_func.__name__}: {getattr(tool_func, '_tool_name', 'unnamed')}")

    try:
        from strands.tools.registry import ToolRegistry

        registry = ToolRegistry()
        print(f"Registry tools: {len(registry.get_all_tools())}")
    except Exception as e:
        print(f"Registry error: {e}")
