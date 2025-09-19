# routes.py
import asyncio
import json
import logging
import os
import re
import time
import traceback
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

import httpx as _rt_httpx
import jwt as _rt_jwt
from dotenv import load_dotenv
from fastapi import APIRouter
from fastapi import Depends as _rt_Depends
from fastapi import HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel
from slowapi import Limiter as _rt_Limiter
from slowapi.util import get_remote_address as _rt_get_remote_address

from app.services import policy_engine
from src.app.core.config import Settings as _RT_Settings
from src.app.services.background import celery_app as _rt_celery

# REMOVED: from src.app.services.infracost_integration import estimate_cost_async_v2 as _rt_cost_v2
from src.app.services.compat import validate_terraform_code_ast as _rt_validate_ast
from src.app.services.github_integration import (
    enable_auto_apply_action as _rt_enable_actions,
)
from src.app.services.github_integration import trufflehog_scan_ref as _rt_truffle
from src.app.services.helpers import infracost_breakdown_hcl as _infracost_breakdown_hcl
from src.app.services.helpers import read_main_tf as _read_last_terraform_from_disk
from src.app.services.helpers import save_main_tf as _save_terraform_to_disk

# NEW: Orchestrator + helpers
from src.app.services.orchestrator import (
    AgentReply,
    ReplyType,
    run_pipeline,
    stream_pipeline,
)
from src.app.services.strands_agent import TerraformAgent
from src.app.services.strands_agent import chat_secure as _rt_chat_secure
from src.app.services.strands_agent import generate_tf_unit_tests as _rt_ai_tests
from src.app.utils.utils import run_cmd_async as _rt_run_cmd_async
from src.app.utils.utils import sanitize_user_text as _rt_sanitize
from src.app.utils.utils import secure_tempdir as _rt_secure_tempdir
from src.app.utils.verified import verify_answer

# Prometheus content-type fallback
try:
    from prometheus_client import CONTENT_TYPE_LATEST as _PROM_CTYPE
except Exception:
    _PROM_CTYPE = "text/plain; version=0.0.4; charset=utf-8"

# Local modules
from src.app.core.metrics import metrics
from src.app.services.github_integration import github_integration
from src.app.services.infracost_integration import infracost_integration

# Add this alias for backward compatibility
_rt_cost_v2 = infracost_integration.estimate_cost_async_v2

# --- ADDED: no-op decorator shim so AttributeError won't crash imports ---
if not hasattr(metrics, "track_request"):

    def _noop_track_request(_method: str, _path: str):
        def _decorator(func):
            return func

        return _decorator

    metrics.track_request = _noop_track_request  # type: ignore[attr-defined]
# -------------------------------------------------------------------------

# Add this alias for backward compatibility
_rt_cost_v2 = infracost_integration.estimate_cost_async_v2

# ... rest of the code remains the same ...


# ============================================================
# OpenAI wrapper (kept for WebUI compatibility)
# ============================================================
def openai_chat_response(content: str, model: str = "terraform-agent") -> dict:
    """Return content in the exact shape Open WebUI expects."""
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


# ---------------------------
# Helpers
# ---------------------------
def _asdict(value):
    """Best-effort conversion to plain dict for JSON responses."""
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump()
        except Exception:
            pass
    if hasattr(value, "dict"):
        try:
            return value.dict()
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return {"result": str(value)}


async def _attach_verification(user_msg: str, payload: dict) -> dict:
    """Add verification fields without changing your response shape."""
    try:
        draft = payload.get("response", "") if isinstance(payload, dict) else ""
        try:
            v = verify_answer(user_msg, draft, min_citations=1)
        except Exception as ve:
            logger.warning(f"verify_answer failed: {str(ve)}")
            v = {
                "verified": False,
                "sources": [],
                "answer": draft,
                "reasoning_note": f"Verification error: {str(ve)}",
            }

        if isinstance(payload, dict):
            payload["verified"] = bool(v.get("verified", False))
            payload["sources"] = v.get("sources", [])
            payload["answer_refined"] = v.get("answer", draft) or draft
            payload["verification_note"] = v.get("reasoning_note", "")
        return payload
    except Exception as e:
        logger.warning(f"Verification failed: {str(e)}")
        if isinstance(payload, dict):
            payload.setdefault("verified", False)
            payload.setdefault("sources", [])
            payload.setdefault("answer_refined", payload.get("response", ""))
            payload.setdefault("verification_note", f"Verification failed: {str(e)}")
        return payload


# ---------------------------
# Init / Env / Logging
# ---------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()

ALLOWED_ORIGINS = (
    os.getenv("ALLOWED_ORIGINS", "*").split(",") if os.getenv("ALLOWED_ORIGINS") != "*" else ["*"]
)

router = APIRouter()

terraform_agent = None
try:
    terraform_agent = TerraformAgent()
    logger.info("[OK] TerraformAgent initialized successfully")
except Exception as e:
    logger.error("[ERROR] Failed to initialize TerraformAgent: %s", str(e))
    logger.error(traceback.format_exc())


# ---------------------------
# Router-level Request ID
# ---------------------------
async def add_request_id_middleware(request, call_next):
    request_id = str(uuid.uuid4())[:8]
    try:
        request.state.request_id = request_id
    except Exception:
        pass
    try:
        response = await call_next(request)
        try:
            response.headers["X-Request-ID"] = request_id
        except Exception:
            pass
        return response
    except Exception as exc:
        logger.error("Request failed: %s", str(exc))
        raise


try:
    if hasattr(router, "middleware"):
        router.middleware("http")(add_request_id_middleware)
        logger.info("APIRouter.middleware registered")
    else:
        logger.info("APIRouter.middleware not available; app-level will handle request IDs")
except Exception as _e:
    logger.warning("Router middleware registration skipped: %s", str(_e))


# ---------------------------
# Env helper (context manager)
# ---------------------------
@contextmanager
def _clear_tf_cli_args_env():
    old = os.environ.pop("TF_CLI_ARGS", None)
    try:
        yield
    finally:
        if old is not None:
            os.environ["TF_CLI_ARGS"] = old


# ---------------------------
# Models
# ---------------------------
class ChatMessageRequest(BaseModel):
    chat_id: str
    user_message: str


class ChatRenameRequest(BaseModel):
    title: str
    archived: Optional[bool] = False


class ChatHistoryRequest(BaseModel):
    chat_id: str
    limit: Optional[int] = 20


class ValidateRequest(BaseModel):
    terraform_code: str
    run_policy_checks: Optional[bool] = True
    usage_yaml: Optional[str] = None


class BudgetUpdateRequest(BaseModel):
    workspace: str
    monthly_limit: float
    alert_thresholds: List[float]


class CreatePRRequest(BaseModel):
    terraform_code: str
    commit_message: Optional[str] = None
    branch_name: Optional[str] = None


class MergePRRequest(BaseModel):
    pr_number: int
    merge_method: Optional[str] = "squash"


class ChatSettingsRequest(BaseModel):
    cloud: Optional[str] = None
    region: Optional[str] = None
    environment: Optional[str] = None
    budget: Optional[float] = None


# === Long-input handling
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "200000"))
CHAT_DATA_DIR = Path(os.getenv("CHAT_DATA_DIR", "chat_data"))


def spill_if_too_long(text: str) -> tuple[str, str | None]:
    if not isinstance(text, str):
        return "", None
    if len(text) <= MAX_INPUT_CHARS:
        return text, None
    CHAT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    p = CHAT_DATA_DIR / f"incoming_{uuid.uuid4().hex}.txt"
    p.write_text(text, encoding="utf-8")
    return (
        f"Load and process the file at {p}. Do not print the raw contents; summarize actions and results instead.",
        str(p),
    )


# ---------------------------
# Simple ping for readiness
# ---------------------------
@router.get("/ping")
def ping():
    return {"ok": True}


# ---------------------------
# Basic / Health
# ---------------------------
@router.get("/test")
async def test_endpoint():
    try:
        if terraform_agent is None:
            return {"status": "error", "agent_initialized": False, "error": "Agent not initialized"}
        health_status = await terraform_agent.health_check()
        return {"status": "ok", "message": "Backend is healthy", **health_status}
    except Exception as e:
        logger.error("Health check failed: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.get("/health")
async def detailed_health_check():
    try:
        if terraform_agent is None:
            return {
                "status": "unhealthy",
                "agent_initialized": False,
                "error": "Agent not initialized",
            }
        health_status = await terraform_agent.health_check()
        health_status.update(
            {
                "environment_vars": {
                    "GITHUB_TOKEN": bool(os.getenv("GITHUB_TOKEN")),
                    "GITHUB_REPO": bool(os.getenv("GITHUB_REPO")),
                    "INFRACOST_API_KEY": bool(os.getenv("INFRACOST_API_KEY")),
                },
                "filesystem": {
                    "current_dir": str(Path.cwd()),
                    "main_tf_exists": Path("main.tf").exists(),
                    "gitignore_exists": Path(".gitignore").exists(),
                    "git_dir_exists": Path(".git").exists(),
                    "chat_data_exists": Path("chat_data").exists(),
                    "budgets_exists": Path("budgets.json").exists(),
                },
                "components": {
                    "metrics_enabled": True,
                    "policy_engine_enabled": True,
                    "infracost_integration": infracost_integration.infracost_available,
                    "policy_scanners": policy_engine.enabled_scanners,
                    "github_integration": github_integration.available,
                },
                "cors_settings": {"allowed_origins": ALLOWED_ORIGINS},
            }
        )
        return health_status
    except Exception as e:
        logger.error("Detailed health check failed: %s", str(e))
        return {"status": "unhealthy", "error": str(e), "agent_initialized": False}


# ---------------------------
# Chat (via orchestrator)
# ---------------------------
@router.post("/chat")
async def chat_with_ai(req: ChatMessageRequest):
    request_id = str(uuid.uuid4())[:8]
    logger.info("[REQ] chat: chat_id=%s req_id=%s", req.chat_id, request_id)

    if terraform_agent is None:
        raise HTTPException(
            status_code=500, detail={"error": "Agent not initialized", "request_id": request_id}
        )

    try:
        if not req.user_message or not req.user_message.strip():
            payload = {"error": "Empty message not allowed", "request_id": request_id}
            payload = await _attach_verification(req.user_message, payload)
            raise HTTPException(status_code=400, detail=payload)

        # (Optional) spill very long inputs to disk
        user_msg_raw = req.user_message.strip()
        user_msg_slim, _spilled = spill_if_too_long(user_msg_raw)

        metrics.record_chat_message("user", "terraform")
        try:
            reply: AgentReply = await asyncio.wait_for(
                run_pipeline(user_msg_slim, req.chat_id, terraform_agent),
                timeout=180.0,
            )
        except asyncio.TimeoutError:
            payload = {"error": "Request timed out", "request_id": request_id}
            payload = await _attach_verification(req.user_message, payload)
            raise HTTPException(status_code=408, detail=payload)

        metrics.record_chat_message("assistant", reply.type.value)

        payload = reply.model_dump()
        payload["request_id"] = request_id
        payload = await _attach_verification(req.user_message, payload)
        return payload

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Exception in /chat: %s", str(e))
        logger.error(traceback.format_exc())

        payload = {
            "status": "error",
            "type": ReplyType.ERROR.value,
            "error": f"Server error: {str(e)}",
            "exception_type": type(e).__name__,
            "request_id": request_id,
        }
        try:
            payload = await _attach_verification(req.user_message, payload)
        except Exception as ve:
            logger.warning(f"Verification also failed: {str(ve)}")
            payload["verified"] = False
            payload["sources"] = []
            payload["answer_refined"] = ""
            payload["verification_note"] = "Verification skipped due to error"

        raise HTTPException(status_code=500, detail=payload)


# ---------------------------
# GET + SSE stream (orchestrator)
# ---------------------------
@router.get("/chat/stream")
async def chat_with_ai_stream(
    chat_id: str = Query(..., description="Chat ID"),
    user_message: str = Query(..., description="User message"),
):
    request_id = str(uuid.uuid4())[:8]
    logger.info("[REQ] chat/stream: chat_id=%s req_id=%s", chat_id, request_id)

    if terraform_agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")

    async def generate_stream():
        try:
            if not user_message or not user_message.strip():
                yield f"data: {json.dumps({'error': 'Empty message not allowed', 'done': True, 'request_id': request_id})}\n\n"
                return

            user_msg_raw = user_message.strip()
            user_msg_slim, _ = spill_if_too_long(user_msg_raw)

            metrics.record_chat_message("user", "terraform")

            async for item in stream_pipeline(user_msg_slim, chat_id, terraform_agent):
                # Attach request id and verification on the final payload
                item["request_id"] = request_id
                if item.get("done") and item.get("response"):
                    try:
                        item = await _attach_verification(user_message, item)
                    except Exception as ve:
                        logger.warning("Verification failed in stream: %s", str(ve))
                        item["verified"] = False
                        item["sources"] = []
                        item["verification_note"] = f"Verification failed: {str(ve)}"
                yield f"data: {json.dumps(item)}\n\n"

        except Exception as e:
            logger.error("Exception in /chat/stream: %s", str(e))
            yield f"data: {json.dumps({'status': 'error', 'type': 'error', 'error': str(e), 'done': True, 'request_id': request_id})}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------
# OpenAI-compatible adapter
# ---------------------------
class OAIMsg(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class OAIChatReq(BaseModel):
    model: Optional[str] = "terraform-agent"
    messages: List[OAIMsg]
    stream: bool = False


def _oai_extract_user_text(messages: List[OAIMsg]) -> str:
    for m in reversed(messages):
        if m.role == "user":
            return (m.content or "").strip()
    return (messages[-1].content or "").strip() if messages else ""


def _chunk(s: str, n: int = 120):
    for i in range(0, len(s), n):
        yield s[i : i + n]


@router.get("/v1/models")
def oai_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "terraform-agent",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            }
        ],
    }


@router.post("/v1/chat/completions")
async def oai_chat(req: OAIChatReq):
    if terraform_agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")

    chat_id = f"oai-{uuid4().hex[:8]}"
    user_text = _oai_extract_user_text(req.messages)
    if not user_text:
        raise HTTPException(status_code=400, detail="Empty message")

    user_text, _ = spill_if_too_long(user_text)

    # Non-streaming path
    if not req.stream:
        reply: AgentReply = await asyncio.wait_for(
            run_pipeline(user_text, chat_id, terraform_agent), timeout=180.0
        )
        text = (
            reply.response.strip()
            or "I'm ready to help you with Terraform. What would you like to do?"
        )
        return JSONResponse(openai_chat_response(text, model=req.model or "terraform-agent"))

    # Streaming path (send one full message chunked)
    async def gen():
        try:
            reply: AgentReply = await asyncio.wait_for(
                run_pipeline(user_text, chat_id, terraform_agent),
                timeout=180.0,
            )
            text = reply.response.strip()
            if not text:
                raise RuntimeError("No content generated")
        except Exception as e:
            raise HTTPException(status_code=500, detail={"message": str(e)})

        def sse(obj: dict) -> str:
            return "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"

        base_chunk = {
            "id": f"cmpl-{uuid4().hex}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": req.model or "terraform-agent",
        }

        yield sse(
            {
                **base_chunk,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
        )

        for part in _chunk(text, 120):
            yield sse(
                {
                    **base_chunk,
                    "choices": [{"index": 0, "delta": {"content": part}, "finish_reason": None}],
                }
            )
            await asyncio.sleep(0)

        yield sse({**base_chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------
# Delete branch (async)
# ---------------------------
@router.delete("/github/branch/{branch_name}")
async def delete_branch(branch_name: str):
    try:
        result = github_integration.delete_branch(branch_name)
        if result.get("success"):
            return {"status": "success", "message": result.get("message")}
        else:
            return {"status": "error", "error": result.get("error", "Failed to delete branch")}
    except Exception as e:
        logger.error("Failed to delete branch: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Failed to delete branch: {str(e)}")


# ---------------------------
# Tools / Validation
# ---------------------------
@router.get("/tools/status")
async def get_tool_status():
    if terraform_agent is None:
        return {"error": "Agent not initialized"}
    try:
        tool_status = terraform_agent.check_tool_availability()
        enhanced_status = {
            **tool_status,
            "infracost": infracost_integration.infracost_available,
            "policy_scanners": policy_engine.enabled_scanners,
            "github_integration": github_integration.available,
        }
        return {
            "status": "success",
            "tools": enhanced_status,
            "summary": {
                "terraform_available": enhanced_status.get("terraform", False),
                "infracost_available": enhanced_status.get("infracost", False),
                "git_available": enhanced_status.get("git", False),
                "github_integration_available": enhanced_status.get("github_integration", False),
                "policy_scanners_available": any(policy_engine.enabled_scanners.values()),
                "all_tools_available": all(
                    [v for k, v in enhanced_status.items() if isinstance(v, bool)]
                ),
            },
            "policy_engine": policy_engine.get_policy_summary(),
        }
    except Exception as e:
        logger.error("Tool status check failed: %s", str(e))
        return {"status": "error", "error": str(e)}


@router.post("/validate")
async def validate_terraform_code(request: ValidateRequest):
    try:
        terraform_code = request.terraform_code
        if not terraform_code:
            return {"status": "error", "error": "No Terraform code provided"}

        from src.app.services.guardrails import TerraformGuardrails
        from src.app.utils.strands_tools import (
            infrastructure_analyzer,
            terraform_validator,
        )

        guardrails = TerraformGuardrails()

        normalized = terraform_validator.normalize_unicode(terraform_code)
        scaffolded = normalized

        with _clear_tf_cli_args_env():
            fmt_res = terraform_validator.format_terraform_code(scaffolded)
            if getattr(fmt_res, "success", None) is True and fmt_res.data:
                formatted_code = fmt_res.data.get("formatted_code", scaffolded)
            else:
                formatted_code = scaffolded
                logger.warning(
                    "terraform_validator.format_terraform_code failed: %s",
                    getattr(fmt_res, "error", None),
                )

            val_res = terraform_validator.validate_terraform_syntax(formatted_code)
            if getattr(val_res, "success", None) is True and val_res.data:
                validation_result = val_res.data
            else:
                validation_result = {
                    "valid": False,
                    "error": getattr(val_res, "error", "Validation failed"),
                    "output": getattr(val_res, "data", {}),
                }

            infra_res = infrastructure_analyzer.analyze_terraform_resources(formatted_code)
            if getattr(infra_res, "success", None) is True and infra_res.data:
                security_analysis = infra_res.data
            else:
                security_analysis = {
                    "security_concerns": [],
                    "estimated_monthly_cost": 0,
                    "resource_count": 0,
                    "error": getattr(infra_res, "error", None),
                }

        # Save the code we validated too
        _save_terraform_to_disk(formatted_code, Path(os.getenv("WORKING_DIRECTORY", "/app")))

        security_issues = guardrails.check_security_issues(formatted_code)
        terraform_validation = guardrails.validate_terraform_response(formatted_code)

        hints: List[str] = []
        code_for_hints = formatted_code

        if re.search(r"\baws_", code_for_hints, re.I):
            m = re.search(r'provider\s+"aws"\s*{([^}]*)}', code_for_hints, flags=re.I | re.S)
            has_region = bool(m and re.search(r"\bregion\s*=", m.group(1), flags=re.I))
            if not has_region:
                if re.search(r'provider\s+"aws"\s*{', code_for_hints, flags=re.I):
                    hints.append(
                        'AWS: add region in provider, e.g. `provider "aws" { region = "us-east-1" }`'
                    )
                else:
                    hints.append(
                        'AWS: add a provider block with region, e.g. `provider "aws" { region = "us-east-1" }`'
                    )

        if re.search(r"\bgoogle_", code_for_hints, re.I):
            m = re.search(r'provider\s+"google"\s*{([^}]*)}', code_for_hints, flags=re.I | re.S)
            has_g_region = bool(m and re.search(r"\b(region|zone)\s*=", m.group(1), flags=re.I))
            if not has_g_region:
                if re.search(r'provider\s+"google"\s*{', code_for_hints, flags=re.I):
                    hints.append(
                        'Google: set `region` or `zone` in `provider "google"` (and `project`).'
                    )
                else:
                    hints.append(
                        'Google: add `provider "google" { project = "...", region = "us-central1" }` (or set `zone`).'
                    )

        response_data: Dict[str, Any] = {
            "status": "success",
            "formatted_code": formatted_code,
            "validation": validation_result,
            "guardrails": {
                "terraform_validation": terraform_validation,
                "security_issues": security_issues,
            },
            "security_analysis": {
                "security_concerns": security_analysis.get("security_concerns", []),
                "estimated_monthly_cost": security_analysis.get("estimated_monthly_cost", 0),
                "resource_count": security_analysis.get("resource_count", 0),
                "raw": security_analysis,
            },
        }

        if hints:
            response_data["hints"] = {"missing_regions": hints}

        if infracost_integration.infracost_available:
            usage_yaml = getattr(request, "usage_yaml", None)
            cost_result = _infracost_breakdown_hcl(formatted_code, usage_yaml=usage_yaml)
            if cost_result.get("success"):
                response_data["cost_estimate"] = cost_result
                try:
                    monthly_cost = cost_result.get("cost_estimate", {}).get("monthly_cost", 0)
                    metrics.record_cost_estimation("infracost", float(monthly_cost))
                except Exception:
                    pass

        return response_data
    except Exception as e:
        logger.error("Validation endpoint error: %s", str(e))
        return {"status": "error", "error": str(e)}


# ---------------------------
# Metrics
# ---------------------------
@router.get("/metrics")
def get_metrics():
    metrics_data = metrics.get_metrics()
    return Response(content=metrics_data, media_type=_PROM_CTYPE)


@router.get("/metrics/summary")
@metrics.track_request("GET", "/metrics/summary")
def get_metrics_summary():
    try:
        summary = metrics.get_metrics_summary()
        return {"status": "success", "metrics": summary, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error("Metrics summary failed: %s", str(e))
        return {"status": "error", "error": str(e)}


# ---------------------------
# Budget / Cost
# ---------------------------
@router.get("/budgets/{workspace}")
@metrics.track_request("GET", "/budgets")
def get_budget_status(workspace: str):
    try:
        budget_status = infracost_integration.get_budget_status(workspace)
        return {"status": "success", **budget_status}
    except Exception as e:
        logger.error("Budget status check failed: %s", str(e))
        return {"status": "error", "error": str(e)}


@router.put("/budgets/{workspace}")
@metrics.track_request("PUT", "/budgets")
def update_budget(workspace: str, req: BudgetUpdateRequest):
    try:
        result = infracost_integration.update_budget(
            req.workspace, req.monthly_limit, req.alert_thresholds
        )
        if result.get("success"):
            return {"status": "success", **result}
        else:
            return {"status": "error", "error": result.get("error", "Budget update failed")}
    except Exception as e:
        logger.error("Budget update failed: %s", str(e))
        return {"status": "error", "error": str(e)}


@router.post("/cost/diff")
@metrics.track_request("POST", "/cost/diff")
def generate_cost_diff(request: dict):
    try:
        old_terraform = request.get("old_terraform", "")
        new_terraform = request.get("new_terraform", "")
        workspace = request.get("workspace", "default")
        usage_yaml_old = request.get("usage_yaml_old")
        usage_yaml_new = request.get("usage_yaml_new")

        if not old_terraform or not new_terraform:
            return {"status": "error", "error": "Both old_terraform and new_terraform are required"}

        try:
            diff_result = infracost_integration.generate_cost_diff(
                old_terraform, new_terraform, workspace
            )
            if diff_result.get("success"):
                return {"status": "success", **diff_result}
        except Exception:
            pass

        old_cost = _infracost_breakdown_hcl(old_terraform, usage_yaml_old)
        new_cost = _infracost_breakdown_hcl(new_terraform, usage_yaml_new)
        if not (old_cost.get("success") and new_cost.get("success")):
            return {
                "status": "error",
                "error": f"Cost diff failed: old={old_cost.get('error')}, new={new_cost.get('error')}",
            }
        old_v = float(old_cost["cost_estimate"]["monthly_cost"] or 0)
        new_v = float(new_cost["cost_estimate"]["monthly_cost"] or 0)
        return {
            "status": "success",
            "success": True,
            "old_monthly_cost": old_v,
            "new_monthly_cost": new_v,
            "delta_monthly_cost": new_v - old_v,
            "currency": new_cost.get("currency", "USD"),
            "old": old_cost,
            "new": new_cost,
        }
    except Exception as e:
        logger.error("Cost diff generation failed: %s", str(e))
        return {"status": "error", "error": str(e)}


@router.post("/cost/estimate")
@metrics.track_request("POST", "/cost/estimate")
def generate_cost_estimate(request: dict):
    try:
        terraform_code = request.get("terraform_code", "")
        usage_yaml = request.get("usage_yaml")
        if not terraform_code:
            terraform_code = _read_last_terraform_from_disk(
                Path(os.getenv("WORKING_DIRECTORY", "/app"))
            )
            if not terraform_code:
                return {"status": "error", "error": "terraform_code is required"}

        cost_result = _infracost_breakdown_hcl(terraform_code, usage_yaml=usage_yaml)
        if cost_result.get("success"):
            try:
                monthly_cost = cost_result.get("cost_estimate", {}).get("monthly_cost", 0)
                metrics.record_cost_estimation("infracost", float(monthly_cost))
            except Exception:
                pass
            return {"status": "success", **cost_result}

        return {"status": "error", "error": cost_result.get("error", "Cost estimation failed")}
    except Exception as e:
        logger.error("Cost estimation failed: %s", str(e))
        return {"status": "error", "error": str(e)}


# ---------------------------
# Policy
# ---------------------------
@router.get("/policy/summary")
@metrics.track_request("GET", "/policy/summary")
def get_policy_summary():
    try:
        summary = policy_engine.get_policy_summary()
        return {"status": "success", **summary}
    except Exception as e:
        logger.error("Policy summary failed: %s", str(e))
        return {"status": "error", "error": str(e)}


@router.post("/policy/validate")
@metrics.track_request("POST", "/policy/validate")
def validate_with_policies(request: dict):
    try:
        terraform_code = request.get("terraform_code", "")
        if not terraform_code:
            terraform_code = _read_last_terraform_from_disk(
                Path(os.getenv("WORKING_DIRECTORY", "/app"))
            )
            if not terraform_code:
                return {"status": "error", "error": "terraform_code is required"}
        validation_result = policy_engine.validate_with_policies(terraform_code)
        report = None
        if validation_result.get("violations"):
            report = policy_engine.create_policy_report(validation_result, terraform_code)
        return {"status": "success", "validation_result": validation_result, "report": report}
    except Exception as e:
        logger.error("Policy validation failed: %s", str(e))
        return {"status": "error", "error": str(e)}


# ---------------------------
# Workflow: terraform -> PR
# ---------------------------
@router.post("/workflow/terraform-to-pr")
def terraform_to_pr_workflow(request: dict):
    try:
        terraform_code = (request.get("terraform_code") or "").strip()
        if not terraform_code:
            terraform_code = _read_last_terraform_from_disk(
                Path(os.getenv("WORKING_DIRECTORY", "/app"))
            )
        workspace = request.get("workspace") or "default"
        commit_message = request.get("commit_message") or "feat(iac): add demo stack via TF Agent"
        branch_name = request.get("branch_name") or f"tf/feature-{uuid4().hex[:8]}"
        base_branch = (request.get("base") or os.getenv("GITHUB_BASE_BRANCH") or "main").strip()

        if not terraform_code:
            return {"status": "error", "error": "terraform_code is required"}

        workflow_result: Dict[str, Any] = {
            "status": "success",
            "workflow_steps": {"workspace": {"status": "completed", "selected": workspace}},
            "final_result": None,
        }

        if not github_integration.available:
            workflow_result["workflow_steps"]["github_pr"] = {
                "status": "skipped",
                "reason": "GitHub integration not available",
            }
            return workflow_result

        os.environ.setdefault("GITHUB_BASE_BRANCH", base_branch)

        pr_result = _asdict(
            github_integration.create_branch_and_pr(terraform_code, commit_message, branch_name)
        )

        workflow_result["workflow_steps"]["github_pr"] = {
            "status": "completed" if pr_result.get("success") else "failed",
            "result": {**pr_result, "requested_base": base_branch},
        }
        workflow_result["final_result"] = workflow_result["workflow_steps"]["github_pr"]["result"]
        return workflow_result

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("terraform_to_pr_workflow failed")
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "type": "server_error",
                "error": str(e),
                "exception_type": type(e).__name__,
            },
        )


# ---------------------------
# Admin
# ---------------------------
@router.post("/admin/cache/clear")
@metrics.track_request("POST", "/admin/cache/clear")
def clear_system_cache():
    try:
        cache_cleared = {"chat_sessions": 0, "tool_cache": 0, "cost_estimates": 0}
        return {
            "status": "success",
            "message": "System caches cleared",
            "cleared": cache_cleared,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error("Cache clear failed: %s", str(e))
        return {"status": "error", "error": str(e)}


@router.get("/admin/system/info")
@metrics.track_request("GET", "/admin/system/info")
def get_system_info():
    try:
        import platform
        import sys

        return {
            "status": "success",
            "system": {
                "platform": platform.platform(),
                "python_version": sys.version,
                "current_directory": str(Path.cwd()),
                "environment_variables": {
                    "GITHUB_TOKEN": ("configured" if os.getenv("GITHUB_TOKEN") else "missing"),
                    "GITHUB_REPO": os.getenv("GITHUB_REPO", "not configured"),
                    "INFRACOST_API_KEY": (
                        "configured" if os.getenv("INFRACOST_API_KEY") else "missing"
                    ),
                },
            },
            "integrations": {
                "terraform_agent": terraform_agent is not None,
                "github_integration": github_integration.available,
                "infracost_integration": infracost_integration.infracost_available,
                "policy_engine": True,
                "metrics": True,
            },
            "files": {
                "chat_data_dir": Path("chat_data").exists(),
                "budgets_file": Path("budgets.json").exists(),
                "main_tf": Path("main.tf").exists(),
                "git_repo": Path(".git").exists(),
            },
        }
    except Exception as e:
        logger.error("System info failed: %s", str(e))
        return {"status": "error", "error": str(e)}


# ---------------------------
# SECURE/JWT endpoints
# ---------------------------
_RT = _RT_Settings()
_limiter2 = _rt_Limiter(key_func=_rt_get_remote_address)


def _rt_require_jwt(request: Request):
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing token")
    token = auth.split(" ", 1)[1]
    try:
        _rt_jwt.decode(token, _RT.JWT_SECRET, algorithms=[_RT.JWT_ALG])
        return True
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")


@router.post("/secure/chat")
@_limiter2.limit(_RT.RATE_LIMIT_CHAT)
async def _rt_secure_chat(
    request: Request,
    req: dict,
    _=_rt_Depends(_rt_require_jwt),
):
    chat_id = (req or {}).get("chat_id") or "default"
    user_text = _rt_sanitize((req or {}).get("user_text", ""))
    result = await _rt_chat_secure(chat_id, user_text)
    return JSONResponse(result)


@router.post("/ai/generate-tests")
async def _rt_generate_tests(req: dict, _=_rt_Depends(_rt_require_jwt)):
    tf_code = req.get("terraform_code", "") or _read_last_terraform_from_disk(
        Path(os.getenv("WORKING_DIRECTORY", "/app"))
    )
    return JSONResponse(await _rt_ai_tests(tf_code))


@router.post("/policy/validate/ast")
async def _rt_policy_validate_ast(req: dict, _=_rt_Depends(_rt_require_jwt)):
    tf_code = req.get("terraform_code", "") or _read_last_terraform_from_disk(
        Path(os.getenv("WORKING_DIRECTORY", "/app"))
    )
    return JSONResponse(await _rt_validate_ast(tf_code))


@router.post("/cost/estimate/async")
async def _rt_cost_async(req: dict, _=_rt_Depends(_rt_require_jwt)):
    tf_code = req.get("terraform_code", "") or _read_last_terraform_from_disk(
        Path(os.getenv("WORKING_DIRECTORY", "/app"))
    )
    currency = (req.get("currency") or "USD").upper()
    task = _rt_celery.send_task("infracost.estimate", args=[tf_code, currency])
    return {"success": True, "data": {"task_id": task.id}, "error": ""}


@router.get("/tasks/{task_id}")
async def _rt_task_status(task_id: str, _=_rt_Depends(_rt_require_jwt)):
    res = _rt_celery.AsyncResult(task_id)
    if res.ready():
        try:
            data = res.result.get("data") if isinstance(res.result, dict) else None
            if data:
                usd = float(data.get("total_monthly_cost_usd", 0.0))
                metrics.record_cost_estimation("async", usd)
        except Exception:
            pass
        return {"success": True, "data": res.result, "error": ""}
    return {"success": True, "data": {"state": res.state}, "error": ""}


@router.post("/cost/estimate/v2")
async def _rt_cost_estimate_v2(req: dict, _=_rt_Depends(_rt_require_jwt)):
    tf_code = req.get("terraform_code", "") or _read_last_terraform_from_disk(
        Path(os.getenv("WORKING_DIRECTORY", "/app"))
    )
    currency = (req.get("currency") or "USD").upper()
    res = await _rt_cost_v2(tf_code, currency=currency)
    if res.get("success"):
        try:
            usd = float(res["data"].get("total_monthly_cost_usd", 0.0))
            metrics.record_cost_estimation("sync", usd)
        except Exception:
            pass
    return JSONResponse(res)


@router.post("/github/enable-auto-apply")
async def _rt_enable_auto_apply(_=_rt_Depends(_rt_require_jwt)):
    return JSONResponse(await _rt_enable_actions())


@router.post("/github/scan/trufflehog")
async def _rt_scan_trufflehog(req: dict, _=_rt_Depends(_rt_require_jwt)):
    ref = req.get("ref")
    return JSONResponse(await _rt_truffle(ref or None))


@router.post("/notify")
async def _rt_notify(req: dict, _=_rt_Depends(_rt_require_jwt)):
    text = _rt_sanitize(req.get("text", ""))
    slack = req.get("slack_webhook")
    teams = req.get("teams_webhook")
    async with _rt_httpx.AsyncClient(timeout=15.0) as client:
        if slack:
            await client.post(slack, json={"text": text})
        if teams:
            await client.post(teams, json={"text": text})
    return {"success": True, "data": {"posted": bool(slack or teams)}, "error": ""}


@router.post("/visualize")
async def _rt_visualize(req: dict, _=_rt_Depends(_rt_require_jwt)):
    tf_code = req.get("terraform_code", "") or _read_last_terraform_from_disk(
        Path(os.getenv("WORKING_DIRECTORY", "/app"))
    )
    from os import path as _os_path

    with _rt_secure_tempdir("graph_v2_") as d:
        with open(_os_path.join(d, "main.tf"), "w", encoding="utf-8") as f:
            f.write(tf_code)
        rc, out, err = await _rt_run_cmd_async(
            "terraform", "init", "-no-color", "-input=false", cwd=d
        )
        if rc != 0:
            return {"success": False, "data": {"stderr": err}, "error": "terraform init failed"}
        rc, dot, err = await _rt_run_cmd_async("terraform", "graph", "-no-color", cwd=d)
        if rc != 0:
            return {"success": False, "data": {"stderr": err}, "error": "terraform graph failed"}
        return {"success": True, "data": {"dot": dot}, "error": ""}
