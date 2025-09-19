# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
import difflib
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# -----------------------------------------------------------------------------
# Safe import of sanitize_user_text (with fallback)
# -----------------------------------------------------------------------------
try:
    from src.app.utils.utils import sanitize_user_text as _sanitize_user_text
except Exception:
    try:
        from app.utils.utils import (
            sanitize_user_text as _sanitize_user_text,  # legacy path
        )
    except Exception:

        def _sanitize_user_text(s: str) -> str:
            try:
                return (s or "").strip()
            except Exception:
                return str(s)


sanitize_user_text = _sanitize_user_text

# -----------------------------------------------------------------------------
# Gemini
# -----------------------------------------------------------------------------
import google.generativeai as genai
from cachetools import TTLCache
from dotenv import load_dotenv

from src.app.core.config import Settings
from src.app.core.metrics import metrics
from src.app.services import strands_compat as sc
from src.app.services.github_integration import github_integration
from src.app.services.guardrails import TerraformGuardrails
from src.app.services.infracost_integration import infracost_integration
from src.app.services.policy_engine import policy_engine
from src.app.utils import strands_tools as st
from src.app.utils.chat_memory import ChatMemory
from src.app.utils.strands_tools import (  # noqa: F401 (imported for side effects / availability)
    calculator,
    file_operations,
    infrastructure_analyzer,
    plan_analyzer,
    static_sanity_checks,
    terraform_validator,
)

# -----------------------------------------------------------------------------
# Strands SDK (optional) – tolerate multiple layouts/APIs
# -----------------------------------------------------------------------------
_StrandsAgent = None
for candidate in (
    "strands.core.agent",  # some releases
    "strands.agent",  # others/legacy
    "strands",  # very old / flat export
):
    try:
        mod = __import__(candidate, fromlist=["Agent"])
        _StrandsAgent = getattr(mod, "Agent", None)
        if _StrandsAgent:
            break
    except Exception:
        continue

try:
    from transformers import pipeline
except ImportError:
    pipeline = None

# --- UTF-8 bootstrap (console safe emojis/logs on Windows)
try:
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# -----------------------------------------------------------------------------
# Env
# -----------------------------------------------------------------------------
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
FORCE_GEMINI_FALLBACK = os.getenv("FORCE_GEMINI_FALLBACK", "").lower() in {"1", "true", "yes", "on"}

if API_KEY:
    genai.configure(api_key=API_KEY)
else:
    print("⚠️  GEMINI_API_KEY not found; will degrade gracefully.")


# -----------------------------------------------------------------------------
# Credential Error Detection and Handling
# -----------------------------------------------------------------------------
def _is_credential_error(error: Exception) -> bool:
    error_msg = str(error).lower()
    credential_patterns = [
        "unable to locate credentials",
        "could not automatically determine credentials",
        "default credentials",
        "application default credentials",
        "adc",
        "google.auth.exceptions",
        "credentials not found",
        "authentication failed",
        "invalid credentials",
        "no credentials provided",
    ]
    return any(pattern in error_msg for pattern in credential_patterns)


def _log_strands_fallback(reason: str, error: Exception = None):
    if error:
        print(f"🔄 Strands fallback triggered: {reason} - Error: {str(error)}")
    else:
        print(f"🔄 Strands fallback triggered: {reason}")
    print("🔄 Using direct Gemini API instead...")


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
async def _maybe_await(x: Any) -> Any:
    return await x if asyncio.iscoroutine(x) else x


def infer_cloud(s: str) -> Optional[str]:
    s = (s or "").lower()
    aws_k = ["s3", "ec2", "iam", "vpc", "route53", "aws", "cloudwatch", "alb", "eks", "lambda"]
    gcp_k = [
        "gcs",
        "gke",
        "bigquery",
        "pubsub",
        "gcp",
        "google cloud",
        "compute engine",
        "cloud run",
    ]
    azure_k = ["azure", "azurerm", "resource group", "storage account", "vnet", "aks"]
    if any(k in s for k in aws_k):
        return "aws"
    if any(k in s for k in gcp_k):
        return "gcp"
    if any(k in s for k in azure_k):
        return "azure"
    return None


# -----------------------------------------------------------------------------
# Terraform helper utilities
# -----------------------------------------------------------------------------
class TerraformTools:
    """Custom helpers for Terraform operations with safety and resilience."""

    @staticmethod
    def highlight_placeholders(terraform_code: str) -> str:
        if not terraform_code:
            return terraform_code
        banner = (
            "# ⚠️  DEMO CREDENTIALS - DO NOT USE IN PRODUCTION!\n"
            "# Replace demo values with real ones and use a secrets manager.\n\n"
        )
        subs = {
            r'ami\s*=\s*"[^"]*"': 'ami = "ami-09ac0b140f63d3458"  # Demo AMI',
            r'username\s*=\s*"[^"]*"': 'username = "demo_admin"  # Demo',
            r'password\s*=\s*"[^"]*"': 'password = "REPLACE_WITH_SECURE_PASSWORD"',
            r'db_name\s*=\s*"[^"]*"': 'db_name = "demo_database"  # Demo',
            r'identifier\s*=\s*"[^"]*"': 'identifier = "demo-db-instance"  # Demo',
            r'key_name\s*=\s*"[^"]*"': 'key_name = "demo-keypair"  # Replace',
            r'subnet_id\s*=\s*"[^"]*"': "subnet_id = data.aws_subnet.selected.id  # example data source",
            r"vpc_security_group_ids\s*=\s*\[[^\]]*\]": "vpc_security_group_ids = [aws_security_group.demo.id]",
            r'availability_zone\s*=\s*"[^"]*"': "availability_zone = data.aws_availability_zones.available.names[0]",
            r'file\("~/.ssh/id_rsa.pub"\)': (
                '"ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQD... # replace with your real public key"'
            ),
        }
        for pat, repl in subs.items():
            terraform_code = re.sub(pat, repl, terraform_code)
        return banner + terraform_code

    @staticmethod
    def extract_terraform_code(response_text: str) -> str:
        if not response_text:
            return ""
        fenced = re.findall(r"```(?:terraform|hcl|tf)?\n?(.*?)```", response_text, re.DOTALL)
        blocks = [b.strip() for b in fenced if b and b.strip()]
        if blocks:
            return "\n\n".join(blocks)
        hcl_block_pattern = (
            r"((?:resource|provider|variable|output|data|module|terraform)\s+[^{]*"
            r"{[^{}]*(?:{[^{}]*}[^{}]*)*})"
        )
        matches = re.findall(hcl_block_pattern, response_text, re.DOTALL | re.IGNORECASE)
        if matches:
            return "\n\n".join([m.strip() for m in matches if m.strip()])
        patterns = [
            r'(resource\s+"[^"]+"\s+"[^"]+"\s*{[^}]*})',
            r'(provider\s+"[^"]+"\s*{[^}]*})',
            r'(variable\s+"[^"]+"\s*{[^}]*})',
            r'(output\s+"[^"]+"\s*{[^}]*})',
            r'(data\s+"[^"]+"\s+"[^"]+"\s*{[^}]*})',
            r'(module\s+"[^"]+"\s*{[^}]*})',
            r"(terraform\s*{[^}]*})",
        ]
        out: List[str] = []
        for p in patterns:
            for m in re.findall(p, response_text, re.DOTALL | re.IGNORECASE):
                if m and m.strip():
                    out.append(m.strip())
        return "\n\n".join(out) if out else ""

    @staticmethod
    def ensure_minimal_scaffold(terraform_code: str, cloud: Optional[str] = "aws") -> str:
        code = (terraform_code or "").strip()
        if not code:
            return code

        def has(regex: str) -> bool:
            return re.search(regex, code, re.IGNORECASE | re.DOTALL) is not None

        pieces: List[str] = []

        if cloud == "gcp":
            if not has(r"\bterraform\b\s*{"):
                pieces.append(
                    "terraform {\n"
                    '  required_version = ">= 1.4.0"\n'
                    "  required_providers {\n"
                    "    google = {\n"
                    '      source  = "hashicorp/google"\n'
                    '      version = "~> 5.0"\n'
                    "    }\n"
                    "  }\n"
                    "}\n"
                )
            if not has(r'\bprovider\s+"google"\b\s*{'):
                pieces.append(
                    'provider "google" {\n'
                    "  project = var.project\n"
                    "  region  = var.region\n"
                    "}\n"
                )
            if not has(r'\bvariable\s+"project"\b'):
                pieces.append(
                    'variable "project" {\n'
                    '  description = "GCP project ID"\n'
                    "  type        = string\n"
                    "}\n"
                )
            if not has(r'\bvariable\s+"region"\b'):
                pieces.append(
                    'variable "region" {\n'
                    '  description = "GCP region"\n'
                    "  type        = string\n"
                    '  default     = "us-central1"\n'
                    "}\n"
                )

        elif cloud == "azure":
            if not has(r"\bterraform\b\s*{"):
                pieces.append(
                    "terraform {\n"
                    '  required_version = ">= 1.4.0"\n'
                    "  required_providers {\n"
                    "    azurerm = {\n"
                    '      source  = "hashicorp/azurerm"\n'
                    '      version = "~> 3.100"\n'
                    "    }\n"
                    "  }\n"
                    "}\n"
                )
            if not has(r'\bprovider\s+"azurerm"\b\s*{'):
                pieces.append('provider "azurerm" {\n  features {}\n}\n')
            if not has(r'\bvariable\s+"location"\b'):
                pieces.append(
                    'variable "location" {\n'
                    '  description = "Azure location"\n'
                    "  type        = string\n"
                    '  default     = "eastus"\n'
                    "}\n"
                )

        else:  # default AWS
            if not has(r"\bterraform\b\s*{"):
                pieces.append(
                    "terraform {\n"
                    '  required_version = ">= 1.4.0"\n'
                    "  required_providers {\n"
                    "    aws = {\n"
                    '      source  = "hashicorp/aws"\n'
                    '      version = "~> 5.0"\n'
                    "    }\n"
                    "  }\n"
                    "}\n"
                )
            if not has(r'\bprovider\s+"aws"\b\s*{'):
                pieces.append('provider "aws" {\n  region = var.region\n}\n')
            if not has(r'\bvariable\s+"region"\b'):
                pieces.append(
                    'variable "region" {\n'
                    '  description = "AWS region"\n'
                    "  type        = string\n"
                    '  default     = "us-east-1"\n'
                    "}\n"
                )

        pieces.append(code)
        return "\n\n".join(pieces)

    @staticmethod
    @metrics.track_tool("terraform_timeout")
    def run_with_timeout(command: List[str], timeout: int = 30, cwd: str = ".") -> Dict[str, Any]:
        try:
            result = subprocess.run(
                command, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False
            )
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Command timed out after {timeout} seconds",
                "returncode": -1,
                "timeout": True,
            }
        except FileNotFoundError:
            tool_name = command[0] if command else "unknown"
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Tool '{tool_name}' not found - please install it",
                "returncode": -2,
                "tool_missing": True,
            }
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Command failed: {str(e)}",
                "returncode": -3,
            }

    @staticmethod
    def estimate_infracost_with_fallback(terraform_code: str = "") -> str:
        try:
            if infracost_integration.infracost_available:
                result = infracost_integration.generate_cost_estimate(terraform_code, "default")
                if result.get("success"):
                    cost_data = result.get("cost_estimate", {})
                    monthly_cost = float(cost_data.get("monthly_cost", 0) or 0)
                    lines = [
                        "📊 INFRACOST ESTIMATE",
                        "=" * 50,
                        f"Monthly Cost: ${monthly_cost:.2f}",
                        f"Yearly Cost: ${float(cost_data.get('yearly_cost', 0) or 0):.2f}",
                        f"Daily Cost: ${float(cost_data.get('daily_cost', 0) or 0):.2f}",
                        "",
                    ]
                    resources = result.get("resource_breakdown", [])
                    if resources:
                        lines.append("Top Cost Contributors:")
                        for resource in resources[:5]:
                            name = resource.get("name", "unknown")
                            monthly = float(resource.get("monthly_cost", 0) or 0)
                            lines.append(f"  • {name}: ${monthly:.2f}/month")
                        lines.append("")
                    budget = result.get("budget_analysis", {})
                    if budget:
                        alert = budget.get("alert_level", "GREEN")
                        if alert != "GREEN":
                            lines.append(f"⚠️ Budget Alert: {alert}")
                            lines.append(
                                f"Budget Utilization: {float(budget.get('budget_utilization_percent', 0) or 0):.1f}%"
                            )
                            lines.append("")
                    lines.append("Generated by Infracost CLI")
                    return "\n".join(lines)
            # fallback rough
            return TerraformTools.get_rough_cost_estimate(terraform_code)
        except Exception as e:
            return f"Cost estimation error: {str(e)}"

    @staticmethod
    def get_rough_cost_estimate(terraform_code: str) -> str:
        if not terraform_code:
            return "No Terraform code provided for cost estimation"
        try:
            tr = infrastructure_analyzer.analyze_terraform_resources(terraform_code)
            analysis = tr.data if getattr(tr, "success", False) else {}
            if not analysis:
                return f"Cost estimation failed: {getattr(tr, 'error', 'unknown error')}"
            estimated_cost = float(analysis.get("estimated_monthly_cost", 0) or 0)
            resource_count = int(analysis.get("resource_count", 0) or 0)
            lines: List[str] = []
            lines.append("📊 ROUGH COST ESTIMATE (Not exact - for planning only)")
            lines.append("=" * 60)
            lines.append(f"Resources analyzed: {resource_count}")
            lines.append("")
            lines.append(f"💰 Estimated monthly cost: ~${estimated_cost:.2f}")
            lines.append(f"💰 Estimated yearly cost: ~${(estimated_cost * 12):.2f}")
            lines.append("")
            lines.append("⚠️  This is a rough estimate. Install Infracost CLI for accurate pricing.")
            return "\n".join(lines)
        except Exception as e:
            return f"Rough cost estimation failed: {str(e)}"

    @staticmethod
    def push_to_github_with_resilience(terraform_code: str = "") -> str:
        if not github_integration.available:
            return "GitHub push skipped: Missing GITHUB_TOKEN or GITHUB_REPO environment variables"
        try:
            result = github_integration.create_branch_and_pr(
                terraform_code,
                f"Terraform update - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            )
            if result.get("success"):
                pr_number = result.get("pr_number")
                pr_url = result.get("pr_url")
                checks_passed = result.get("checks_passed")
                lines = [
                    f"✅ Created PR #{pr_number}: {pr_url}",
                    f"Branch: {result.get('branch_name')}",
                    "",
                ]
                if checks_passed:
                    lines.append("✅ All automated checks passed")
                    lines.append("PR is ready for review and merge")
                else:
                    lines.append("⚠️ Some automated checks failed")
                    lines.append("Review the PR for details and required fixes")
                lines.append(f"Check results: {result.get('checks_summary', '')}")
                return "\n".join(lines)
            else:
                return f"GitHub PR creation failed: {result.get('error', 'Unknown error')}"
        except Exception as e:
            return f"GitHub operation failed: {str(e)}"

    @staticmethod
    def create_diff(old_content: str, new_content: str, filename: str = "main.tf") -> str:
        if not old_content and not new_content:
            return "No changes"
        if not old_content:
            return f"New file created: {filename} ({len(new_content.splitlines())} lines)"
        if not new_content:
            return f"File deleted: {filename}"
        old_lines = old_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)
        diff = list(
            difflib.unified_diff(
                old_lines,
                new_lines,
                fromfile=f"{filename} (before)",
                tofile=f"{filename} (after)",
                lineterm="",
            )
        )
        return "".join(diff) or "No changes detected"

    @staticmethod
    def analyze_blast_radius(terraform_code: str) -> List[str]:
        if not terraform_code:
            return []
        warnings = []
        patterns = [
            (
                r"skip_final_snapshot\s*=\s*true",
                "⚠️ Database will be deleted without final snapshot",
            ),
            (r"deletion_protection\s*=\s*false", "⚠️ Deletion protection is disabled"),
            (r"force_destroy\s*=\s*true", "⚠️ S3 bucket will be destroyed even if not empty"),
            (
                r'cidr_blocks\s*=\s*\["0\.0\.0\.0/0"\]',
                "⚠️ Security group allows access from anywhere (0.0.0.0/0)",
            ),
            (r"publicly_accessible\s*=\s*true", "⚠️ Database will be publicly accessible"),
            (r"associate_public_ip_address\s*=\s*true", "⚠️ Instance will have public IP"),
        ]
        for pattern, warn in patterns:
            if re.search(pattern, terraform_code, re.IGNORECASE):
                warnings.append(warn)
        return warnings

    @staticmethod
    def is_infra_prompt(user_prompt: str) -> bool:
        if not user_prompt:
            return False
        lower = user_prompt.lower()
        infra_keywords = [
            "vpc",
            "ec2",
            "s3",
            "rds",
            "terraform",
            "provision",
            "infra",
            "deploy",
            "aws",
            "cloud",
            "instance",
            "database",
            "server",
            "load balancer",
            "security group",
            "subnet",
            "internet gateway",
            "elastic",
            "lambda",
            "cloudformation",
            "kubernetes",
            "docker",
            "create",
            "build",
            "setup",
            "configure",
            "gcp",
            "azure",
            "azurerm",
            "google",
        ]
        return any(word in lower for word in infra_keywords)

    @staticmethod
    def mentions_cost_estimate(user_prompt: str) -> bool:
        if not user_prompt:
            return False
        lower = user_prompt.lower()
        cost_keywords = [
            "cost",
            "estimate",
            "infracost",
            "price",
            "pricing",
            "budget",
            "expensive",
            "cheap",
            "money",
        ]
        return any(word in lower for word in cost_keywords)

    @staticmethod
    def mentions_github_push(user_prompt: str) -> bool:
        if not user_prompt:
            return False
        lower = user_prompt.lower()
        git_keywords = [
            "github",
            "push",
            "commit",
            "repository",
            "repo",
            "git",
            "pull request",
            "pr",
        ]
        return any(word in lower for word in git_keywords)


# -----------------------------------------------------------------------------
# TerraformAgent – Strands + Gemini (with compat shim wiring)
# -----------------------------------------------------------------------------
class TerraformAgent:
    """Terraform Agent with chat memory, guardrails, and integrations."""

    def __init__(self):
        self.strands_available = False  # Track if Strands is actually working
        self.force_gemini = FORCE_GEMINI_FALLBACK

        try:
            # Initialize Strands agent if available and not forced to use Gemini
            if _StrandsAgent and not self.force_gemini:
                model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
                try:
                    try:
                        self.agent = _StrandsAgent(model=model_name)
                    except TypeError:
                        self.agent = _StrandsAgent(model_name)

                    # Minimal connection test via compat shim
                    try:
                        test_ok = self._test_strands_connection()
                        if test_ok:
                            print("✅ Terraform Agent initialized with Strands integration")
                            self.strands_available = True
                        else:
                            print(
                                "⚠️  Strands agent created but credential/API test failed - using Gemini fallback"
                            )
                            self.agent = None
                    except Exception as e:
                        if _is_credential_error(e):
                            _log_strands_fallback("Strands credential test failed", e)
                            self.agent = None
                        else:
                            print(f"⚠️  Strands test failed with non-credential error: {e}")
                            # Keep agent around; allow best-effort use
                            self.strands_available = True

                    # Only register tools if Strands is working
                    if self.strands_available and self.agent:
                        self._register_strands_tools()

                except Exception as e:
                    if _is_credential_error(e):
                        _log_strands_fallback("Strands initialization failed due to credentials", e)
                        self.agent = None
                    else:
                        print(f"⚠️  Strands initialization failed: {str(e)}")
                        self.agent = None
            else:
                self.agent = None
                reason = (
                    "FORCE_GEMINI_FALLBACK enabled"
                    if self.force_gemini
                    else "Strands SDK not available"
                )
                print(f"⚠️ {reason}; using Gemini fallback for all operations")

            # Initialize other components
            self.tools = TerraformTools()
            self.chat_memory = ChatMemory()
            self.guardrails = TerraformGuardrails()
            self.policy_engine = policy_engine
            self.infracost_integration = infracost_integration
            self.github_integration = github_integration
            self.metrics = metrics

        except Exception as e:
            print(f"❌ Error initializing TerraformAgent: {str(e)}")
            print(traceback.format_exc())
            raise

    def _test_strands_connection(self) -> bool:
        """
        Test if Strands agent can actually make calls (no credential/API mismatch).
        Uses compat shim which prefers __call__ on strands-agents==1.9.0.
        """
        if not self.agent:
            return False
        try:
            out = sc.call_agent(self.agent, "Hello")
            # handle possible coroutine
            if asyncio.iscoroutine(out):
                # run a quick event loop step if we're not already in one
                # (in constructor, we're sync; just treat coroutine as "works")
                return True
            # strings or response objects
            text = getattr(out, "text", out)
            return bool(str(text).strip())
        except Exception as e:
            # Credentials → definitely false; other errors we allow best-effort use
            return False if _is_credential_error(e) else True

    def _register_strands_tools(self):
        """Register tools with Strands via compat shim (multi-API support)."""
        tool_fns = [
            st.st_terraform_format,
            st.st_terraform_validate,
            st.st_terraform_plan,
            st.st_analyze_resources,
            st.st_best_practices,
            st.st_save_file,
            st.st_static_sanity,
            st.st_plan_analyze,
            st.st_calc_multiply,
        ]

        # Optional: allow skipping registration entirely
        if os.getenv("STRANDS_SKIP_TOOL_REG", "").lower() in {"1", "true", "yes", "on"}:
            print("ℹ️  Skipping Strands tool registration by request (STRANDS_SKIP_TOOL_REG=1).")
            return

        try:
            api_used = sc.register_tools(self.agent, tool_fns)
            print(f"✅ Registered Strands tools via {api_used}")
        except Exception as e:
            if _is_credential_error(e):
                print(
                    "⚠️  Strands tool registration failed due to credential issues; continuing without tools"
                )
            else:
                print(f"⚠️  Strands tools not registered: {e} (continuing without)")

    # --- Strands invocation via compat shim + credential fallback
    async def _call_strands(self, prompt: str) -> str:
        """
        Call Strands through the compat shim. Falls back to Gemini on credential errors.
        """
        if not self.agent or self.force_gemini:
            raise RuntimeError("Strands agent not available")

        try:
            out = sc.call_agent(self.agent, prompt)
            out = await _maybe_await(out)
            result = str(getattr(out, "text", out) or "").strip()
            if result:
                return result
            # Empty result: treat as non-credential error and try Gemini
            return await self._llm_call(prompt)
        except Exception as e:
            if _is_credential_error(e):
                _log_strands_fallback("Credential error during Strands call", e)
                self.strands_available = False
                return await self._llm_call(prompt)
            # Non-credential issue → try Gemini but keep Strands available
            print(f"⚠️  Strands call failed (non-credential): {e} — trying Gemini fallback")
            return await self._llm_call(prompt)

    async def _llm_call(self, prompt: str) -> str:
        """Direct Gemini API call as fallback."""
        if not API_KEY:
            return ""

        def _run():
            m = genai.GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-1.5-pro"))
            r = m.generate_content(prompt)
            return getattr(r, "text", "") or ""

        return await asyncio.to_thread(_run)

    async def get_reasoned_response(
        self,
        user_prompt: str,
        chat_id: str = None,
        stream_thinking: bool = False,
        cloud_hint: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Enhanced get_reasoned_response with automatic fallback on credential errors."""
        if not user_prompt or not user_prompt.strip():
            return "Empty prompt provided", "Please provide a valid request."

        context = ""
        chat_settings: Dict[str, Any] = {}
        if chat_id:
            context = self.chat_memory.get_context_for_prompt(chat_id, context_length=5)
            chat_settings = self.chat_memory.get_chat_settings(chat_id)

        sys_prompt = """You are a Terraform expert and helpful assistant. You can help with both infrastructure code and general questions.

For infrastructure requests:
- Generate valid, secure Terraform code
- Include proper resource configurations
- Add security best practices
- Prefer minimal examples that apply cleanly

For general chat:
- Be helpful and concise
- Maintain context from previous messages"""

        if cloud_hint:
            sys_prompt += f"\n\nTarget cloud: {cloud_hint}."

        if chat_settings:
            info = []
            if chat_settings.get("cloud"):
                info.append(f"Target Cloud: {chat_settings['cloud']}")
            if chat_settings.get("region"):
                info.append(f"Region: {chat_settings['region']}")
            if chat_settings.get("environment"):
                info.append(f"Environment: {chat_settings['environment']}")
            if chat_settings.get("budget"):
                info.append(f"Budget Limit: ${chat_settings['budget']}")
            if info:
                sys_prompt += f"\n\nChat Settings: {', '.join(info)}"

        if context and context != "No previous conversation context.":
            full_prompt = f"""{sys_prompt}

Previous conversation context:
{context}

Current request: {user_prompt}

Respond in this format:
THINKING: <brief analysis>
FINAL RESPONSE: <your response or Terraform code>"""
        else:
            full_prompt = f"""{sys_prompt}

Request: {user_prompt}

Respond in this format:
THINKING: <brief analysis>
FINAL RESPONSE: <your response or Terraform code>"""

        try:
            # Try Strands first if available, with automatic fallback
            if self.agent and self.strands_available and not self.force_gemini:
                try:
                    txt = await self._call_strands(full_prompt)
                except Exception as e:
                    if _is_credential_error(e):
                        _log_strands_fallback("get_reasoned_response Strands call failed", e)
                        txt = await self._llm_call(full_prompt)
                    else:
                        print(
                            f"⚠️  Strands call failed with non-credential error, trying Gemini fallback: {e}"
                        )
                        txt = await self._llm_call(full_prompt)
            else:
                # Use direct Gemini API
                txt = await self._llm_call(full_prompt)

            if txt:
                full_text = txt.strip()
                filtered_text, _ = self.guardrails.filter_hallucinations(full_text, user_prompt)
                if "FINAL RESPONSE:" in filtered_text:
                    parts = filtered_text.split("FINAL RESPONSE:", 1)
                    thinking = parts[0].replace("THINKING:", "").strip()
                    final = parts[1].strip()
                    return thinking, final
                elif "```" in filtered_text:
                    return "Generated code without structured format", filtered_text
                else:
                    code = self.tools.extract_terraform_code(filtered_text)
                    if code:
                        return "Extracted Terraform code from response", code
                    else:
                        return "No structured format found", filtered_text
            else:
                return (
                    "Could not get response from AI.",
                    "I'm having trouble generating a response right now. Please try again.",
                )
        except Exception as e:
            if _is_credential_error(e):
                print(f"❌ Credential error in get_reasoned_response: {str(e)}")
                try:
                    txt = await self._llm_call(full_prompt)
                    if txt:
                        return "Response generated via Gemini fallback", txt.strip()
                except Exception as gemini_e:
                    print(f"❌ Gemini fallback also failed: {str(gemini_e)}")
                    return (
                        "Both Strands and Gemini failed due to credential issues",
                        "I'm having trouble connecting to the AI services. Please check your API configuration.",
                    )
            else:
                print(f"❌ Error in get_reasoned_response: {str(e)}")
            return (
                f"Error generating response: {str(e)}",
                "I encountered an error while processing your request. Please try again.",
            )

    async def ask_for_clarity(self, user_prompt: str) -> str:
        """Ask for clarification with credential error handling."""
        clarification_prompt = f"""The user request needs clarification: "{user_prompt}"

Ask 2-3 specific technical questions about:
- Cloud & services needed
- Sizes/types and regions
- Network setup
- Environment (dev/prod)

Return numbered questions only."""
        try:
            if self.agent and self.strands_available and not self.force_gemini:
                try:
                    txt = await self._call_strands(clarification_prompt)
                except Exception as e:
                    if _is_credential_error(e):
                        txt = await self._llm_call(clarification_prompt)
                    else:
                        print(f"⚠️  Strands clarity call failed, using Gemini: {e}")
                        txt = await self._llm_call(clarification_prompt)
            else:
                txt = await self._llm_call(clarification_prompt)

            return (
                txt.strip()
                if txt
                else "Could you provide more details about your infrastructure requirements?"
            )
        except Exception as e:
            if _is_credential_error(e):
                print(f"❌ Credential error in ask_for_clarity: {str(e)}")
                try:
                    txt = await self._llm_call(clarification_prompt)
                    return (
                        txt.strip()
                        if txt
                        else "Could you provide more details about your infrastructure requirements?"
                    )
                except Exception:
                    return "Could you provide more details about what you'd like to deploy?"
            else:
                print(f"❌ Error in ask_for_clarity: {str(e)}")
                return "Could you provide more details about what you'd like to deploy?"

    def check_tool_availability(self) -> Dict[str, bool]:
        tools_status: Dict[str, Any] = {}
        terraform_result = TerraformTools.run_with_timeout(["terraform", "--version"], timeout=5)
        tools_status["terraform"] = terraform_result["success"]
        tools_status["infracost"] = self.infracost_integration.infracost_available
        git_result = TerraformTools.run_with_timeout(["git", "--version"], timeout=5)
        tools_status["git"] = git_result["success"]
        tools_status["github"] = self.github_integration.available
        tools_status["policy_scanners"] = self.policy_engine.enabled_scanners
        tools_status["strands"] = self.strands_available
        tools_status["gemini_fallback"] = bool(API_KEY)
        return tools_status

    async def process_message(self, user_message: str, chat_id: str = "default") -> Dict[str, Any]:
        print(f"🔄 Processing message: {user_message}")

        result: Dict[str, Any] = {
            "type": "",
            "content": "",
            "terraform_code": "",
            "formatted_code": "",
            "validation_output": "",
            "security_findings": [],
            "best_practices": [],
            "cost_estimate": "Skipped",
            "github_status": "Skipped",
            "thinking_trace": "",
            "questions": "",
            "diff": "",
            "tool_status": self.check_tool_availability(),
            "blast_radius_warnings": [],
            "chat_id": chat_id,
            "confidence_score": 0.0,
            "policy_report": {},
            "infracost_data": {},
            "artifacts": {},
            "analysis": {},
            "ai_backend": "unknown",
        }

        try:
            input_validation = self.guardrails.validate_user_input(user_message)
            if not input_validation["valid"]:
                questions_list = [s for s in input_validation.get("suggestions", []) if s]
                result.update(
                    {
                        "type": "error",
                        "content": input_validation["reason"],
                        "questions": questions_list,
                        "thinking_trace": "Input validation failed",
                    }
                )
                return result

            user_message = user_message.strip()
            self.chat_memory.save_message(chat_id, "user", user_message)

            should_clarify, clarification_questions = self.guardrails.should_ask_for_clarification(
                user_message
            )
            if should_clarify and self.guardrails.is_infrastructure_request(user_message):
                print("❓ Need clarification for infrastructure request")
                questions_text = "\n".join(clarification_questions)
                questions_list = [
                    q.strip() for q in re.split(r"\r?\n+", questions_text) if q.strip()
                ]
                self.chat_memory.save_message(
                    chat_id, "assistant", f"I need more information: {questions_text}"
                )
                result.update(
                    {
                        "type": "clarify",
                        "content": "I need more information to help you better.",
                        "questions": questions_list,
                        "thinking_trace": "Asking for clarification based on guardrails",
                        "ai_backend": (
                            "gemini"
                            if not self.strands_available
                            else "strands_with_gemini_fallback"
                        ),
                    }
                )
                return result

            if self.guardrails.is_infrastructure_request(user_message):
                print("🗗️ Detected infrastructure prompt")
                cloud_hint = infer_cloud(user_message) or "aws"

                thinking, model_response = await self.get_reasoned_response(
                    user_message, chat_id, cloud_hint=cloud_hint
                )

                backend_used = (
                    "gemini" if not self.strands_available or self.force_gemini else "strands"
                )
                result["ai_backend"] = backend_used

                raw_code = self.tools.extract_terraform_code(model_response)

                if not raw_code or not re.search(
                    r"\b(resource|provider|module)\b", raw_code, re.IGNORECASE
                ):
                    print("❓ Need clarification for infrastructure request")
                    questions = await self.ask_for_clarity(user_message)
                    self.chat_memory.save_message(
                        chat_id, "assistant", f"I need clarification: {questions}"
                    )
                    q_list = [q.strip() for q in re.split(r"\r?\n+", questions) if q.strip()]
                    result.update(
                        {
                            "type": "clarify",
                            "content": "I need more information to help you better.",
                            "questions": q_list,
                            "thinking_trace": thinking,
                            "ai_backend": backend_used,
                        }
                    )
                else:
                    print("⚙️ Generating and processing Terraform code")

                    terraform_validation = self.guardrails.validate_terraform_response(
                        model_response
                    )
                    if (not terraform_validation["valid"]) or (
                        terraform_validation["confidence"] < 0.3
                    ):
                        result.update(
                            {
                                "type": "error",
                                "content": (
                                    "Generated response doesn't meet quality standards: "
                                    f"{terraform_validation['reason']}"
                                ),
                                "thinking_trace": thinking,
                                "confidence_score": terraform_validation["confidence"],
                                "ai_backend": backend_used,
                            }
                        )
                        return result

                    raw_code = self.tools.ensure_minimal_scaffold(raw_code, cloud=cloud_hint)
                    processed_code = self.tools.highlight_placeholders(raw_code)

                    security_issues = self.guardrails.check_security_issues(processed_code)

                    policy_result = self.policy_engine.validate_with_policies(processed_code)
                    result["policy_report"] = policy_result
                    if policy_result.get("violations"):
                        comprehensive_report = self.policy_engine.create_policy_report(
                            policy_result, processed_code
                        )
                        result["policy_report"]["comprehensive_report"] = comprehensive_report

                    old_content = ""
                    main_tf_path = Path("main.tf")
                    if main_tf_path.exists():
                        try:
                            old_content = main_tf_path.read_text(encoding="utf-8")
                        except Exception:
                            old_content = ""

                    if main_tf_path.exists():
                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=".backup.tf"
                        ) as backup_file:
                            shutil.copy2(main_tf_path, backup_file.name)
                            result["backup_path"] = backup_file.name

                    save_tr = file_operations.save_terraform_file(processed_code, "main.tf")
                    if not getattr(save_tr, "success", False):
                        result.update(
                            {
                                "type": "error",
                                "content": "Failed to save Terraform file: "
                                + getattr(save_tr, "error", "Unknown error"),
                                "ai_backend": backend_used,
                            }
                        )
                        return result

                    processed_code = terraform_validator.normalize_unicode(processed_code)
                    region_hint = None
                    fixed_code, fix_notes = terraform_validator.autofix_common_hcl(
                        processed_code, region_hint
                    )
                    processed_code = fixed_code
                    try:
                        result.setdefault("notes", [])
                        if fix_notes:
                            result["notes"].extend(fix_notes)
                    except Exception:
                        pass

                    fmt_tr = terraform_validator.format_terraform_code(processed_code)
                    formatted_code = (
                        (fmt_tr.data or {}).get("formatted_code")
                        if getattr(fmt_tr, "success", False)
                        else processed_code
                    )

                    val_tr = terraform_validator.validate_terraform_syntax(formatted_code)
                    val_ok = getattr(val_tr, "success", False)
                    val_data = val_tr.data or {}
                    valid_flag = bool(val_data.get("valid", val_ok))
                    validation_output = ""
                    if not val_ok:
                        validation_output = getattr(val_tr, "error", "") or ""
                    else:
                        validation_output = (val_data.get("output") or "").strip()

                    want_plan = any(
                        k in user_message.lower()
                        for k in (
                            "plan",
                            "show the plan",
                            "review changes",
                            "diff",
                            "terraform plan",
                        )
                    )
                    cli_checks: Dict[str, Any] = {}
                    if want_plan:
                        try:
                            plan_tr = terraform_validator.run_fmt_validate_plan_from_code(
                                formatted_code, timeout=120
                            )
                            plan_data = plan_tr.data or {}
                            cli_checks = plan_data
                            ptxt = plan_data.get("plan_output", "")
                            if ptxt:
                                pa_tr = plan_analyzer.analyze_plan_output(ptxt)
                                pa_data = pa_tr.data or {}
                                result["plan_analysis"] = pa_data
                        except Exception as e:
                            cli_checks = {
                                "errors": [f"plan-exec failed: {e}"],
                                "plan_output": "",
                                "plan_exit_code": 1,
                            }

                    try:
                        sanity_issues = static_sanity_checks(formatted_code)
                    except Exception:
                        sanity_issues = []

                    if not valid_flag:
                        result.update(
                            {
                                "type": "terraform",
                                "needs_fixes": True,
                                "content": "⚠️ Generated Terraform code has validation errors - fixes needed:",
                                "terraform_code": processed_code,
                                "formatted_code": formatted_code,
                                "validation_output": validation_output
                                or (val_data.get("output") or ""),
                                "sanity_issues": sanity_issues,
                                "thinking_trace": thinking,
                                "ai_backend": backend_used,
                                "artifacts": {
                                    "formatted_code": formatted_code or processed_code,
                                    "validation_output": validation_output
                                    or (val_data.get("output") or ""),
                                    "diff": (
                                        ""
                                        if not old_content
                                        else self.tools.create_diff(old_content, formatted_code)
                                    ),
                                },
                                "analysis": {
                                    "security_findings": [],
                                    "best_practices": [],
                                    "blast_radius_warnings": [],
                                },
                            }
                        )
                        return result

                    sa_tr = infrastructure_analyzer.analyze_terraform_resources(formatted_code)
                    sa_data = sa_tr.data or {}
                    security_concerns = list(sa_data.get("security_concerns", []))
                    if sanity_issues:
                        security_concerns.extend([f"Static sanity: {s}" for s in sanity_issues])

                    blast_warnings = self.tools.analyze_blast_radius(formatted_code)

                    best_practices: List[str] = []
                    if valid_flag:
                        bp_tr = infrastructure_analyzer.suggest_best_practices(formatted_code)
                        bp_data = bp_tr.data or {}
                        best_practices = list(bp_data.get("suggestions", []))[:3]

                    diff_output = ""
                    if old_content:
                        diff_output = self.tools.create_diff(old_content, formatted_code)

                    cost_estimate: Any = "Skipped"
                    if self.tools.mentions_cost_estimate(user_message):
                        print("💰 Running cost estimation...")
                        if self.infracost_integration.infracost_available:
                            cost_result = self.infracost_integration.generate_cost_estimate(
                                formatted_code, "default"
                            )
                            if cost_result.get("success"):
                                cost_estimate = cost_result
                                result["infracost_data"] = cost_result
                                monthly_cost = float(
                                    cost_result.get("cost_estimate", {}).get("monthly_cost", 0) or 0
                                )
                                self.metrics.record_cost_estimation("infracost", monthly_cost)
                            else:
                                cost_estimate = self.tools.estimate_infracost_with_fallback(
                                    formatted_code
                                )
                        else:
                            cost_estimate = self.tools.estimate_infracost_with_fallback(
                                formatted_code
                            )

                    git_result = "Skipped"
                    if self.tools.mentions_github_push(user_message):
                        print("📤 Creating GitHub PR...")
                        git_result = self.tools.push_to_github_with_resilience(formatted_code)

                    try:
                        file_operations.save_terraform_file(formatted_code, "main.tf")
                    except Exception:
                        pass

                    confidence = self.guardrails.get_confidence_score(model_response, user_message)
                    response_content = f"✅ Generated Terraform code with {len(sa_data.get('resources', []))} resources"
                    self.chat_memory.save_message(
                        chat_id,
                        "assistant",
                        response_content,
                        {
                            "type": "terraform",
                            "resource_count": len(sa_data.get("resources", [])),
                            "has_security_issues": len(security_concerns) > 0,
                            "policy_violations": len(policy_result.get("violations", [])),
                            "ai_backend": backend_used,
                        },
                    )

                    result.update(
                        {
                            "type": "terraform",
                            "content": "✅ Terraform code generated successfully!",
                            "terraform_code": processed_code,
                            "formatted_code": formatted_code,
                            "validation_output": (
                                (cli_checks.get("plan_output") or "").strip()
                                if cli_checks
                                else (validation_output or "Valid")
                            ),
                            "security_findings": security_concerns
                            + [
                                {"message": issue["description"], "severity": issue["severity"]}
                                for issue in security_issues
                            ],
                            "best_practices": best_practices,
                            "cost_estimate": cost_estimate,
                            "github_status": git_result,
                            "terraform_cli": cli_checks or {},
                            "thinking_trace": thinking,
                            "diff": diff_output,
                            "blast_radius_warnings": blast_warnings,
                            "confidence_score": confidence,
                            "ai_backend": backend_used,
                            "artifacts": {
                                "formatted_code": formatted_code or processed_code,
                                "validation_output": (
                                    (cli_checks.get("plan_output") or "").strip()
                                    if cli_checks
                                    else (validation_output or "")
                                ),
                                "diff": diff_output,
                            },
                            "analysis": {
                                "security_findings": security_concerns,
                                "best_practices": best_practices,
                                "blast_radius_warnings": blast_warnings,
                            },
                        }
                    )
            else:
                print("💬 Handling regular chat")
                thinking, response = await self.get_reasoned_response(user_message, chat_id)
                backend_used = (
                    "gemini" if not self.strands_available or self.force_gemini else "strands"
                )
                confidence = self.guardrails.get_confidence_score(response, user_message)
                self.chat_memory.save_message(
                    chat_id,
                    "assistant",
                    response,
                    {"type": "chat", "confidence": confidence, "ai_backend": backend_used},
                )
                result.update(
                    {
                        "type": "chat",
                        "content": response,
                        "thinking_trace": thinking,
                        "confidence_score": confidence,
                        "ai_backend": backend_used,
                    }
                )

        except Exception as e:
            print(f"❌ Error in process_message: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            print(traceback.format_exc())

            error_message = str(e)
            if _is_credential_error(e):
                error_message = (
                    "I encountered credential authentication issues. "
                    "The system has automatically fallen back to direct Gemini API mode. "
                    f"Original error: {str(e)}"
                )

            result.update(
                {
                    "type": "error",
                    "content": f"I encountered an error while processing your request: {error_message}",
                    "thinking_trace": f"Error occurred during processing: {str(e)}",
                    "ai_backend": "error",
                }
            )

        print(f"✅ Returning result type: {result.get('type')}")
        return result

    def get_chat_history(self, chat_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        return self.chat_memory.get_chat_history(chat_id, limit)

    def delete_chat_history(self, chat_id: str) -> bool:
        return self.chat_memory.delete_chat(chat_id)

    def list_all_chats(self) -> List[Dict[str, Any]]:
        return self.chat_memory.list_chats()

    def search_chats(self, query: str, include_archived: bool = False) -> List[Dict[str, Any]]:
        return self.chat_memory.search_chats(query, include_archived)

    def get_chat_stats(self) -> Dict[str, Any]:
        return self.chat_memory.get_chat_stats()

    async def health_check(self) -> Dict[str, Any]:
        try:
            tool_status = self.check_tool_availability()
            health = {
                "status": "healthy",
                "agent_initialized": True,
                "tools": tool_status,
                "gemini_configured": bool(os.getenv("GEMINI_API_KEY")),
                "strands_available": self.strands_available,
                "force_gemini_fallback": self.force_gemini,
                "working_directory": str(Path.cwd()),
                "main_tf_exists": Path("main.tf").exists(),
                "chat_memory_initialized": self.chat_memory is not None,
                "guardrails_initialized": self.guardrails is not None,
                "chat_data_dir": (str(self.chat_memory.storage_dir) if self.chat_memory else None),
                "policy_scanners": self.policy_engine.enabled_scanners,
                "chat_stats": self.chat_memory.get_chat_stats(),
                "ai_backends": {
                    "strands": self.strands_available,
                    "gemini": bool(API_KEY),
                    "force_gemini": self.force_gemini,
                },
            }

            if not tool_status.get("terraform", False):
                health["status"] = "degraded"
                health["warning"] = "Terraform CLI not available - validation/formatting disabled"

            if not self.strands_available and not API_KEY:
                health["status"] = "unhealthy"
                health["error"] = (
                    "No AI backend available - neither Strands nor Gemini API key configured"
                )
            elif not self.strands_available and API_KEY:
                health["status"] = "healthy"
                health["info"] = "Running in Gemini-only mode due to Strands credential issues"

            critical_failures = []
            if not self.infracost_integration.infracost_available:
                critical_failures.append("Infracost CLI not configured")
            if not self.github_integration.available:
                critical_failures.append("GitHub integration not configured")
            if not self.strands_available and not self.force_gemini:
                critical_failures.append("Strands integration failed - using Gemini fallback")

            if critical_failures:
                health["warnings"] = critical_failures

            return health
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "agent_initialized": False,
                "ai_backends": {
                    "strands": False,
                    "gemini": bool(API_KEY),
                    "force_gemini": self.force_gemini if hasattr(self, "force_gemini") else False,
                },
            }

    def _detect_tf_cli_version(self) -> Optional[str]:
        r = TerraformTools.run_with_timeout(["terraform", "--version"], timeout=5)
        out = (r.get("stdout") or "") + " " + (r.get("stderr") or "")
        m = re.search(r"Terraform v(\d+\.\d+\.\d+)", out)
        return m.group(1) if m else None

    def _auto_pin_required_version(self, files: dict) -> dict:
        cli = self._detect_tf_cli_version() or "1.5.7"
        pin = f">= {cli}"

        def inject_or_update(content: str) -> str:
            if re.search(r'required_version\s*=\s*".*?"', content):
                return re.sub(
                    r'required_version\s*=\s*".*?"', f'required_version = "{pin}"', content
                )
            if re.search(r"\bterraform\s*{", content):
                return re.sub(
                    r"(\bterraform\s*{)",
                    r'\1\n  required_version = "' + pin + '"',
                    content,
                    count=1,
                )
            return (
                "terraform {\n"
                f'  required_version = "{pin}"\n'
                "  required_providers {\n"
                '    aws = { source = "hashicorp/aws", version = "~> 5.60" }\n'
                "  }\n"
                "}\n\n" + content
            )

        updated = False
        for path in list(files.keys()):
            if path.endswith("versions.tf"):
                files[path] = inject_or_update(files[path])
                updated = True
                break
        if not updated:
            files["envs/dev/versions.tf"] = (
                "terraform {\n"
                f'  required_version = "{pin}"\n'
                "  required_providers {\n"
                '    aws = { source = "hashicorp/aws", version = "~> 5.60" }\n'
                "  }\n"
                "}\n"
            )
        return files


# -----------------------------------------------------------------------------
# Small, cached chat helpers (used by /secure endpoints)
# -----------------------------------------------------------------------------
_sa_settings = Settings()
_sa_cache = TTLCache(maxsize=512, ttl=3600)


def _sa_append(chat_id: str, role: str, text: str):
    msgs = _sa_cache.get(chat_id) or []
    msgs.append((role, text))
    _sa_cache[chat_id] = msgs


def _sa_context(chat_id: str, n: int = 8) -> str:
    msgs = _sa_cache.get(chat_id) or []
    msgs = msgs[-n:]
    return (
        "\n".join([f"{(r.upper() if hasattr(r, 'upper') else r)}: {t}" for r, t in msgs])
        if msgs
        else "(no prior context)"
    )


def _sa_fallback(prompt: str) -> str:
    if not pipeline:
        return "Fallback model unavailable."
    gen = pipeline("text-generation", model=_sa_settings.HF_MODEL, max_new_tokens=256)
    return gen(prompt, do_sample=False)[0]["generated_text"]


# -----------------------------------------------------------------------------
# Module-level helpers used by routes.py
# -----------------------------------------------------------------------------
async def chat_secure(chat_id: str, user_text: str):
    clean = sanitize_user_text(user_text)
    ctx = _sa_context(chat_id)
    prompt = f"[Context]\n{ctx}\n\n[User]\n{clean}\n\n[Assistant]"
    metrics.record_chat_message("user", "terraform")

    pt = len(prompt.split())
    metrics.model_tokens.labels(model_name="gemini-pro", type="prompt").inc(pt)
    try:
        metrics.token_budget_remaining.dec(float(pt))
    except Exception:
        pass

    try:
        if API_KEY:

            def _run():
                m = genai.GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-1.5-pro"))
                r = m.generate_content(prompt)
                return getattr(r, "text", "") or ""

            answer = await asyncio.to_thread(_run)
            status = "success"
        else:
            raise Exception("No API key available")
    except Exception as e:
        if _is_credential_error(e):
            answer = "I'm experiencing authentication issues. Please check your API configuration."
            status = "credential_error"
        else:
            answer = _sa_fallback(prompt)
            status = "fallback"

    ct = len(answer.split())
    metrics.model_tokens.labels(model_name="gemini-pro", type="completion").inc(ct)
    try:
        metrics.token_budget_remaining.dec(float(ct))
    except Exception:
        pass
    metrics.model_requests.labels(model_name="gemini-pro", status=status).inc()

    _sa_append(chat_id, "user", clean)
    _sa_append(chat_id, "assistant", answer)
    return {"success": True, "data": {"answer": answer, "backend": status}, "error": ""}


async def generate_tf_unit_tests(tf_code: str):
    clean = sanitize_user_text(tf_code)
    ideas = [
        "Validate 'terraform validate' passes.",
        "Check variables have defaults/validations.",
        "No SG rules with 0.0.0.0/0 on sensitive ports.",
        "Required tags on all resources.",
        "Drift check: plan empty after apply.",
    ]
    return {"success": True, "data": {"prompt": clean, "suggestions": ideas}, "error": ""}
