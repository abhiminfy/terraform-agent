# src/app/services/helpers.py
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import google.generativeai as genai
from dotenv import load_dotenv

from src.app.core.metrics import metrics
from src.app.services.infracost_integration import infracost_integration

# ---------- logging ----------
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

# ---------- env / paths ----------
load_dotenv()
WORKDIR = Path(os.getenv("WORKING_DIRECTORY", "/app"))
CHAT_DATA_DIR = Path(os.getenv("CHAT_DATA_DIR", "chat_data"))
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "200000"))


# ---------- Gemini config ----------
def configure_gemini() -> None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY not set; Gemini fallback/repair disabled")
        return
    try:
        genai.configure(api_key=api_key)
        logger.info("Gemini configured")
    except Exception as e:
        logger.warning("Gemini configure failed: %s", e)


# ---------- Simple utils ----------
def chunk(s: str, n: int = 160):
    for i in range(0, len(s), n):
        yield s[i : i + n]


def spill_if_too_long(text: str) -> Tuple[str, Optional[str]]:
    if not isinstance(text, str):
        return "", None
    if len(text) <= MAX_INPUT_CHARS:
        return text, None
    CHAT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    p = CHAT_DATA_DIR / f"incoming_{hashlib.sha1(text.encode()).hexdigest()[:10]}.txt"
    p.write_text(text, encoding="utf-8")
    return (f"Load and process the file at {p}. Summarize actions/results only.", str(p))


# ---------- HCL helpers ----------
_CODE_FENCE = re.compile(r"```(?:hcl|terraform)?\s*([\s\S]*?)```", re.I)


def extract_hcl(t: str) -> str:
    if not isinstance(t, str):
        return ""
    m = _CODE_FENCE.search(t)
    return (m.group(1) if m else t).strip()


_EMPTY_OR_ERR = re.compile(
    r"^(?:no content generated\.?|i need more information|encountered an error|error:)",
    re.I,
)


def looks_empty_or_error(text: str) -> bool:
    return not text or not text.strip() or bool(_EMPTY_OR_ERR.search(text.strip()))


_BAD_S3_INLINE = re.compile(
    r'resource\s+"aws_s3_bucket"\s+"[^"]+"\s*{[^}]*?(server_side_encryption_configuration|public_access_block|lifecycle_rule)\s*{',
    re.I | re.S,
)


def has_bad_s3_inline(code: str) -> bool:
    return bool(code and _BAD_S3_INLINE.search(code))


def has_bad_local_names(code: str) -> bool:
    # Local names should not contain '-'
    return bool(re.search(r'resource\s+"[^"]+"\s+"[^"]*-[^"]*"\s*{', code or "", re.I))


# ---------- disk I/O ----------
def save_main_tf(code: str, workdir: Path = WORKDIR) -> Optional[str]:
    try:
        workdir.mkdir(parents=True, exist_ok=True)
        path = workdir / "main.tf"
        path.write_text(code or "", encoding="utf-8")
        return str(path)
    except Exception as e:
        logger.warning("save_main_tf failed: %s", e)
        return None


def read_main_tf(workdir: Path = WORKDIR) -> str:
    try:
        p = workdir / "main.tf"
        if p.exists():
            return p.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning("read_main_tf failed: %s", e)
    return ""


# ---------- cost ----------
_COST_CACHE: Dict[str, Tuple[float, dict]] = {}


def _cache_get(key: str, ttl: int = 300) -> Optional[dict]:
    v = _COST_CACHE.get(key)
    if not v:
        return None
    ts, data = v
    if time.time() - ts > ttl:
        _COST_CACHE.pop(key, None)
        return None
    return data


def _cache_set(key: str, data: dict) -> None:
    _COST_CACHE[key] = (time.time(), data)


def infracost_breakdown_hcl(
    terraform_code: str, usage_yaml: Optional[str] = None
) -> Dict[str, Any]:
    if not os.environ.get("INFRACOST_API_KEY"):
        return {"success": False, "error": "INFRACOST_API_KEY not set"}

    if not infracost_integration.infracost_available or shutil.which("infracost") is None:
        return {"success": False, "error": "infracost CLI not available"}

    key = hashlib.sha1(
        (terraform_code or "").encode("utf-8") + b"::" + (usage_yaml or "").encode("utf-8")
    ).hexdigest()
    cached = _cache_get(key)
    if cached:
        return cached

    tmp = tempfile.mkdtemp(prefix="infracost_hcl_")
    try:
        main_tf = Path(tmp) / "main.tf"
        main_tf.write_text(terraform_code or "", encoding="utf-8")

        cmd = ["infracost", "breakdown", "--path", tmp, "--format", "json", "--no-color"]
        if usage_yaml:
            usage_path = Path(tmp) / "infracost-usage.yml"
            usage_path.write_text(usage_yaml, encoding="utf-8")
            cmd += ["--usage-file", str(usage_path)]

        env = {**os.environ}
        env.pop("TF_CLI_ARGS", None)
        env.setdefault("TF_CLI_ARGS_init", "-no-color -input=false")
        env.setdefault("TF_CLI_ARGS_plan", "-no-color -input=false")
        env.setdefault("TF_CLI_ARGS_apply", "-no-color -input=false")

        out = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
        if out.returncode != 0:
            return {
                "success": False,
                "error": (out.stderr or out.stdout or "Infracost failed").strip(),
            }

        data = json.loads(out.stdout or "{}")
        currency = data.get("currency") or "USD"
        total = data.get("totalMonthlyCost")
        monthly = float(total) if total is not None else 0.0
        result = {
            "success": True,
            "integration": "infracost",
            "currency": currency,
            "cost_estimate": {"monthly_cost": monthly, "currency": currency},
            "raw": data,
        }
        _cache_set(key, result)
        try:
            metrics.record_cost_estimation("infracost", float(monthly))
        except Exception:
            pass
        return result
    except Exception as e:
        return {"success": False, "error": f"Infracost error: {e}"}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------- response formatting ----------
def _mk_cost_block(cost: Optional[dict]) -> str:
    if not isinstance(cost, dict) or not cost.get("success"):
        return ""
    cur = cost.get("currency", "USD")
    monthly = float(cost.get("cost_estimate", {}).get("monthly_cost", 0) or 0)
    return (
        "### Cost (Infracost)\n"
        f"- Estimated monthly cost: **{cur} {monthly:,.2f}**\n"
        "- Static price lookup from HCL (no cloud creds).\n"
    ).strip()


def format_infra_reply(
    status: str,
    code: Optional[str],
    diagnostics: Optional[str],
    cost_estimate: Optional[dict] = None,
    saved_to: Optional[str] = "main.tf",
) -> str:
    header = "✅ Terraform code generated successfully."
    if status == "validation_failed":
        header = "⚠️ Generated Terraform code has validation errors — fixes needed:"
    parts = [header]
    if diagnostics:
        parts.append("### Diagnostics\n```\n" + diagnostics.strip() + "\n```")
    cb = _mk_cost_block(cost_estimate)
    if cb:
        parts.append(cb)
    if code:
        parts.append("### Terraform (HCL)\n```hcl\n" + code.strip() + "\n```")
    nxt = (
        f"- Saved to `{saved_to}`\n- You can run `terraform init && terraform validate`"
        if saved_to
        else "- You can run `terraform init && terraform validate`"
    )
    parts.append("### What I did next\n" + nxt)
    return "\n\n".join(parts).strip()


# ---------- Gemini fallback / repair ----------
async def _gemini_call(prompt: str) -> str:
    def _run():
        resp = genai.GenerativeModel(
            os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        ).generate_content(prompt)
        return getattr(resp, "text", "") or ""

    return await asyncio.to_thread(_run)


async def gemini_fallback(prompt: str, code_preferred: bool = False, retries: int = 1) -> str:
    if not os.getenv("GEMINI_API_KEY"):
        return ""
    try:
        if code_preferred:
            sys = (
                "You are a Terraform assistant. Output ONLY valid Terraform HCL (no backticks, no prose). "
                "For AWS provider v5, split S3 controls into separate resources."
            )
        else:
            sys = "You are a concise cloud/IaC assistant. Answer clearly."

        full = f"{sys}\n\nUser request:\n{prompt}"
        text = ""
        for i in range(retries + 1):
            text = (await _gemini_call(full)).strip()
            if text:
                break
            await asyncio.sleep(0.3)
        return text
    except Exception as e:
        logger.warning("Gemini fallback failed: %s", e)
        return ""


async def ai_repair_terraform(user_prompt: str, bad_code: str) -> str:
    if not os.getenv("GEMINI_API_KEY"):
        return ""
    rules = """
You are a Terraform fixer. Rewrite the given Terraform into VALID HCL with these HARD RULES:
- Never use '-' in local resource names; use '_' instead.
- For AWS provider v5, DO NOT inline S3 config in aws_s3_bucket. Split into:
  * aws_s3_bucket_versioning
  * aws_s3_bucket_server_side_encryption_configuration
  * aws_s3_bucket_lifecycle_configuration
  * aws_s3_bucket_public_access_block
- Prefer aws_s3_bucket_ownership_controls { object_ownership = "BucketOwnerEnforced" } when appropriate.
- Use jsonencode(...) for policy JSON.
Return ONLY HCL. No backticks. No prose.
"""
    prompt = f"{rules}\n\nUser intent:\n{user_prompt}\n\nInvalid/old HCL:\n{bad_code}\n\nRewritten HCL:\n"
    try:
        text = await _gemini_call(prompt)
        return extract_hcl(text)
    except Exception as e:
        logger.warning("AI repair failed: %s", e)
        return ""
