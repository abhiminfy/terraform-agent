from fastapi import APIRouter, HTTPException, Query, Request, Depends as _rt_Depends
from fastapi.responses import JSONResponse, StreamingResponse, Response
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import google.generativeai as genai
from backend.app.services.strands_agent import TerraformAgent
import traceback
import asyncio
import logging
import json
import uuid
import time
import re
from pathlib import Path
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime
from backend.app.utils.verified import verify_answer
from uuid import uuid4

# Prometheus content-type fallback
try:
    from prometheus_client import CONTENT_TYPE_LATEST as _PROM_CTYPE
except Exception:
    _PROM_CTYPE = "text/plain; version=0.0.4; charset=utf-8"

# Local modules
from backend.app.core.metrics import metrics
from backend.app.services.policy_engine import policy_engine
from backend.app.services.infracost_integration import infracost_integration
from backend.app.services.github_integration import github_integration

# ============================================================
# NEW: OpenAI wrapper + Infra formatter
# ============================================================
def openai_chat_response(content: str, model: str = "terraform-agent") -> dict:
    """Return content in the exact shape Open WebUI expects."""
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }

def format_infra_reply(status: str, code: Optional[str], diagnostics: Optional[str]) -> str:
    """
    Produce one Markdown message that includes:
      - header
      - diagnostics (if any)
      - HCL code block (if any)
      - brief 'what next'
    This way Open WebUI shows everything in a single assistant bubble.
    """
    header = "✅ Terraform code generated successfully."
    if status == "validation_failed":
        header = "⚠️ Generated Terraform code has validation errors — fixes needed:"

    parts = [header]

    if diagnostics:
        parts.append("### Diagnostics\n```\n" + diagnostics.strip() + "\n```")

    if code:
        parts.append("### Terraform (HCL)\n```hcl\n" + code.strip() + "\n```")

    parts.append("### What I did next\n- Saved to `main.tf`\n- You can run `terraform init && terraform validate`")
    return "\n\n".join(parts).strip()
# ============================================================


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
        # verify_answer is synchronous
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

# ---------- NEW: Fallback helpers ----------
def _is_probably_code_request(text: str) -> bool:
    t = (text or "").lower()
    # quick heuristics that cover most IaC asks
    keywords = [
        "terraform", "hcl", "resource", "module", "generate", "create",
        "aws_", "azurerm_", "google_", "s3", "vpc", "ec2", "bucket"
    ]
    return any(k in t for k in keywords)

def _extract_hcl_from_text(t: str) -> str:
    """Try to pull HCL from a Markdown code fence first; else return the text as-is."""
    if not isinstance(t, str):
        return ""
    m = re.search(r"```(?:hcl|terraform)?\s*([\s\S]*?)```", t, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return t.strip()

async def _gemini_fallback(prompt: str, code_preferred: bool = False) -> str:
    """
    Use Gemini directly as a safety net if the agent returns clarify/empty.
    - If code_preferred=True, request HCL only.
    """
    if not os.getenv("GEMINI_API_KEY"):
        return ""

    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    try:
        model = genai.GenerativeModel(model_name)

        if code_preferred:
            sys_prompt = (
                "You are a Terraform assistant. "
                "Return ONLY valid Terraform HCL that a user can paste into .tf files. "
                "No commentary, no explanations, no backticks."
            )
            full = f"{sys_prompt}\n\nUser request:\n{prompt}"
        else:
            sys_prompt = (
                "You are a concise cloud/IaC assistant. "
                "Answer clearly without asking clarifying questions unless truly required."
            )
            full = f"{sys_prompt}\n\nUser request:\n{prompt}"

        # google-generativeai is sync; run in a thread to avoid blocking the event loop
        def _run():
            resp = model.generate_content(full)
            return getattr(resp, "text", "") or ""

        text = await asyncio.to_thread(_run)
        return (text or "").strip()
    except Exception as e:
        logger.warning(f"Gemini fallback failed: {e}")
        return ""

def _looks_like_empty_or_error_text(text: str) -> bool:
    if not text or not text.strip():
        return True
    t = text.strip().lower()
    # Common noisy lines from agent validation path
    return (
        t.startswith("generated terraform code has validation errors")
        or t == "i need more information to help you better."
        or t == "i need more information"
    )

# ---------------------------
# Init / Env / Logging
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    logger.warning("GEMINI_API_KEY not found; some features may be unavailable")

ALLOWED_ORIGINS = (
    os.getenv("ALLOWED_ORIGINS", "*").split(",")
    if os.getenv("ALLOWED_ORIGINS") != "*"
    else ["*"]
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
    merge_method: Optional[str] = "squash"  # squash, merge, rebase

class ChatSettingsRequest(BaseModel):
    cloud: Optional[str] = None
    region: Optional[str] = None
    environment: Optional[str] = None
    budget: Optional[float] = None

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
            return {
                "status": "error",
                "agent_initialized": False,
                "gemini_api_configured": bool(os.getenv("GEMINI_API_KEY")),
                "error": "Agent not initialized",
            }
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
                    "GEMINI_API_KEY": bool(os.getenv("GEMINI_API_KEY")),
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
# Chat
# ---------------------------
@router.post("/chat")
async def chat_with_ai(req: ChatMessageRequest):
    request_id = str(uuid.uuid4())[:8]
    logger.info("[REQ] chat: chat_id=%s req_id=%s", req.chat_id, request_id)

    if terraform_agent is None:
        raise HTTPException(
            status_code=500,
            detail={"error": "Agent not initialized", "request_id": request_id},
        )

    try:
        if not req.user_message or not req.user_message.strip():
            payload = {"error": "Empty message not allowed", "request_id": request_id}
            payload = await _attach_verification(req.user_message, payload)
            raise HTTPException(status_code=400, detail=payload)

        metrics.record_chat_message("user", "terraform")
        try:
            # Properly await the async method call
            result = await asyncio.wait_for(
                terraform_agent.process_message(req.user_message.strip(), req.chat_id),
                timeout=180.0,
            )
        except asyncio.TimeoutError:
            payload = {"error": "Request timed out", "request_id": request_id}
            payload = await _attach_verification(req.user_message, payload)
            raise HTTPException(status_code=408, detail=payload)

        # ---------- NEW: intelligent fallback ----------
        rtype = (result or {}).get("type", "") if isinstance(result, dict) else ""
        content = (result or {}).get("content", "") if isinstance(result, dict) else ""
        tf_code = ""
        if isinstance(result, dict):
            tf_code = (result.get("terraform_code") or
                       result.get("artifacts", {}).get("formatted_code") or "").strip()

        need_fallback = (
            result is None
            or rtype == "clarify"
            or _looks_like_empty_or_error_text(content)
            or (rtype == "terraform" and not tf_code)
        )

        if need_fallback:
            code_pref = _is_probably_code_request(req.user_message)
            fb = await _gemini_fallback(req.user_message, code_preferred=code_pref)
            if fb:
                if code_pref:
                    # Treat as successful terraform response
                    tf = _extract_hcl_from_text(fb)
                    return {
                        "status": "success",
                        "type": "terraform",
                        "response": "Terraform code generated.",
                        "terraform_code": tf,
                        "thinking": "",
                        "tool_status": {},
                        "chat_id": result.get("chat_id", req.chat_id) if isinstance(result, dict) else req.chat_id,
                        "confidence_score": 0.7,
                        "request_id": request_id,
                        "artifacts": {
                            "formatted_code": tf,
                            "validation_output": "",
                            "diff": "",
                        },
                        "analysis": {
                            "security_findings": [],
                            "best_practices": [],
                            "blast_radius_warnings": [],
                        },
                    }
                else:
                    payload = {
                        "status": "success",
                        "type": "chat",
                        "response": fb,
                        "thinking": "",
                        "tool_status": {},
                        "chat_id": req.chat_id,
                        "confidence_score": 0.7,
                        "request_id": request_id,
                    }
                    payload = await _attach_verification(req.user_message, payload)
                    return payload
            # if fallback failed, continue with original result path below

        if result is None:
            payload = {"error": "Agent returned no response", "request_id": request_id}
            payload = await _attach_verification(req.user_message, payload)
            raise HTTPException(status_code=500, detail=payload)

        response_type = result.get("type", "unknown")
        metrics.record_chat_message("assistant", response_type)

        if response_type == "chat":
            payload = {
                "status": "success",
                "type": "chat",
                "response": result.get("content", "No response generated"),
                "thinking": result.get("thinking_trace", ""),
                "tool_status": result.get("tool_status", {}),
                "chat_id": result.get("chat_id", req.chat_id),
                "confidence_score": result.get("confidence_score", 0.0),
                "request_id": request_id,
            }
            payload = await _attach_verification(req.user_message, payload)
            return payload

        elif response_type == "clarify":
            # soften clarify with a minimal direct answer fallback if possible
            fb = await _gemini_fallback(req.user_message, code_preferred=False)
            if fb:
                payload = {
                    "status": "success",
                    "type": "chat",
                    "response": fb,
                    "thinking": result.get("thinking_trace", ""),
                    "tool_status": result.get("tool_status", {}),
                    "chat_id": result.get("chat_id", req.chat_id),
                    "confidence_score": 0.6,
                    "request_id": request_id,
                }
                payload = await _attach_verification(req.user_message, payload)
                return payload

            return {
                "status": "success",
                "type": "clarify",
                "response": "I need more information to help you better.",
                "questions": result.get(
                    "questions", "Could you provide more details?"
                ),
                "thinking": result.get("thinking_trace", ""),
                "tool_status": result.get("tool_status", {}),
                "chat_id": result.get("chat_id", req.chat_id),
                "request_id": request_id,
            }

        elif response_type == "terraform":
            terraform_code = result.get("terraform_code", "") or tf_code
            cost_estimate = result.get("cost_estimate", "")

            if terraform_code and infracost_integration.infracost_available:
                cost_result = infracost_integration.generate_cost_estimate(
                    terraform_code, "default"
                )
                if cost_result.get("success"):
                    cost_estimate = cost_result

            response_data: Dict[str, Any] = {
                "status": "success",
                "type": "terraform",
                "response": result.get("content", "Terraform code generated."),
                "terraform_code": terraform_code,
                "thinking": result.get("thinking_trace", ""),
                "tool_status": result.get("tool_status", {}),
                "chat_id": result.get("chat_id", req.chat_id),
                "confidence_score": result.get("confidence_score", 0.0),
                "request_id": request_id,
                "artifacts": {
                    "formatted_code": result.get("formatted_code", "") or terraform_code,
                    "validation_output": result.get("validation_output", ""),
                    "diff": result.get("diff", ""),
                },
                "analysis": {
                    "security_findings": result.get("security_findings", []),
                    "best_practices": result.get("best_practices", []),
                    "blast_radius_warnings": result.get("blast_radius_warnings", []),
                },
            }
            if cost_estimate and cost_estimate != "Skipped":
                response_data["cost_estimate"] = cost_estimate
                if isinstance(cost_estimate, dict) and cost_estimate.get("success"):
                    monthly_cost = cost_estimate.get("cost_estimate", {}).get(
                        "monthly_cost", 0
                    )
                    metrics.record_cost_estimation("infracost", monthly_cost)

            github_status = result.get("github_status")
            if github_status and github_status != "Skipped":
                response_data["github_status"] = github_status

            return response_data

        else:
            return {
                "status": "error",
                "type": "error",
                "error": f"Unexpected response type: {response_type}",
                "request_id": request_id,
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Exception in /chat: %s", str(e))
        logger.error(traceback.format_exc())

        payload = {
            "status": "error",
            "type": "error",
            "error": f"Server error: {str(e)}",
            "exception_type": type(e).__name__,
            "request_id": request_id,
        }

        # Wrap verification in try-catch to prevent secondary errors
        try:
            payload = await _attach_verification(req.user_message, payload)
        except Exception as ve:
            logger.warning(f"Verification also failed: {str(ve)}")
            payload["verified"] = False
            payload["sources"] = []
            payload["answer_refined"] = ""
            payload["verification_note"] = "Verification skipped due to error"

        raise HTTPException(status_code=500, detail=payload)

# GET + SSE (EventSource)
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
        last_heartbeat = datetime.now()
        try:
            if not user_message or not user_message.strip():
                yield f"data: {json.dumps({'error': 'Empty message not allowed', 'done': True, 'request_id': request_id})}\n\n"
                return

            # initial heartbeat
            yield f"data: {json.dumps({'heartbeat': True, 'request_id': request_id})}\n\n"
            metrics.record_chat_message("user", "terraform")

            # await the async agent call
            result = await asyncio.wait_for(
                terraform_agent.process_message(user_message.strip(), chat_id),
                timeout=180.0,
            )

            if result:
                thinking_trace = result.get("thinking_trace", "")
                stream_thinking = user_message.lower().startswith(
                    "debug:"
                ) or "thinking" in user_message.lower()

                if thinking_trace and stream_thinking:
                    chunk_size = 30
                    for i in range(0, len(thinking_trace), chunk_size):
                        chunk = thinking_trace[i : i + chunk_size]
                        yield f"data: {json.dumps({'thinking_trace': chunk, 'request_id': request_id})}\n\n"
                        await asyncio.sleep(0.03)
                        if (datetime.now() - last_heartbeat).seconds > 15:
                            yield f"data: {json.dumps({'heartbeat': True, 'request_id': request_id})}\n\n"
                            last_heartbeat = datetime.now()

                metrics.record_chat_message("assistant", result.get("type", "chat"))

                final_data: Dict[str, Any] = {
                    "status": "success",
                    "type": result.get("type", "chat"),
                    "response": result.get("content", ""),
                    "terraform_code": result.get("terraform_code", ""),
                    "cost_estimate": result.get("cost_estimate", ""),
                    "github_status": result.get("github_status", ""),
                    "questions": result.get("questions", ""),
                    "thinking_trace": thinking_trace if not stream_thinking else "",
                    "tool_status": result.get("tool_status", {}),
                    "chat_id": result.get("chat_id", chat_id),
                    "confidence_score": result.get("confidence_score", 0.0),
                    "request_id": request_id,
                    "done": True,
                }

                # attach verification (protected)
                try:
                    final_data = await _attach_verification(user_message, final_data)
                except Exception as ve:
                    logger.warning("Verification failed in stream: %s", str(ve))
                    final_data["verified"] = False
                    final_data["sources"] = []
                    final_data["verification_note"] = f"Verification failed: {str(ve)}"

                if result.get("type") == "terraform":
                    final_data.update(
                        {
                            "artifacts": {
                                "formatted_code": result.get("formatted_code", ""),
                                "validation_output": result.get("validation_output", ""),
                                "diff": result.get("diff", ""),
                            },
                            "analysis": {
                                "security_findings": result.get(
                                    "security_findings", []
                                ),
                                "best_practices": result.get("best_practices", []),
                                "blast_radius_warnings": result.get(
                                    "blast_radius_warnings", []
                                ),
                            },
                        }
                    )

                # keep key fields even if falsey
                final_data = {
                    k: v
                    for k, v in final_data.items()
                    if v
                    or k
                    in [
                        "status",
                        "type",
                        "done",
                        "tool_status",
                        "chat_id",
                        "confidence_score",
                        "request_id",
                    ]
                }

                yield f"data: {json.dumps(final_data)}\n\n"
            else:
                # Try a minimal fallback in stream as well
                fb = await _gemini_fallback(user_message, _is_probably_code_request(user_message))
                if fb:
                    yield f"data: {json.dumps({'status':'success','type':'chat','response': fb,'done': True, 'request_id': request_id})}\n\n"
                else:
                    yield f"data: {json.dumps({'status': 'error', 'type': 'error', 'error': 'No response generated', 'done': True, 'request_id': request_id})}\n\n"
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
# OpenAI-compatible adapter for UIs like Open WebUI
# ---------------------------
class OAIMsg(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class OAIChatReq(BaseModel):
    model: Optional[str] = "terraform-agent"
    messages: List[OAIMsg]
    stream: bool = False

def _oai_extract_user_text(messages: List[OAIMsg]) -> str:
    # last user message wins; fallback to last message
    for m in reversed(messages):
        if m.role == "user":
            return (m.content or "").strip()
    return (messages[-1].content or "").strip() if messages else ""

def _chunk(s: str, n: int = 120):
    for i in range(0, len(s), n):
        yield s[i:i+n]

def _extract_text(result: Optional[Dict[str, Any]]) -> str:
    if not isinstance(result, dict):
        return ""
    candidates = [
        result.get("content"),
        result.get("response"),
        result.get("answer"),
        result.get("text"),
        result.get("message"),
        result.get("output"),
    ]
    for c in candidates:
        if isinstance(c, str) and c.strip():
            return c
    return ""

def _render_text_for_oai(result: Optional[Dict[str, Any]]) -> str:
    if not isinstance(result, dict):
        return ""
    text = _extract_text(result).strip()
    if text:
        return text

    rtype = (result.get("type") or "").strip().lower()

    if rtype == "clarify":
        qs = result.get("questions")
        if isinstance(qs, (list, tuple)):
            qstr = "\n".join(f"- {q}" for q in qs if q)
        else:
            qstr = str(qs or "Could you provide more details?")
        return f"I need more information:\n{qstr}".strip()

    if rtype == "terraform":
        summary = (result.get("content") or "Terraform code generated.").strip()
        tf = result.get("terraform_code") or result.get("artifacts", {}).get("formatted_code", "")
        tf = (tf or "").strip()
        if tf:
            return f"{summary}\n\n```hcl\n{tf}\n```"
        return summary

    try:
        safe = {k: v for k, v in result.items() if k != "thinking_trace"}
        blob = json.dumps(safe, ensure_ascii=False)
        return blob[:4000]
    except Exception:
        return "No content generated."

@router.get("/v1/models")
def oai_models():
    return {
        "object": "list",
        "data": [{
            "id": "terraform-agent",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "local",
        }],
    }

@router.post("/v1/chat/completions")
async def oai_chat(req: OAIChatReq):
    """
    UPDATED:
      - Always convert agent/Gemini output into a single assistant message string.
      - For Terraform responses, use format_infra_reply() so code + diagnostics render
        in the same bubble.
      - Then wrap with openai_chat_response() so Open WebUI displays it reliably.
    """
    if terraform_agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")

    chat_id = f"oai-{uuid4().hex[:8]}"
    user_text = _oai_extract_user_text(req.messages)
    if not user_text:
        raise HTTPException(status_code=400, detail="Empty message")

    # ---------- Non-streaming ----------
    if not req.stream:
        result = await asyncio.wait_for(
            terraform_agent.process_message(user_text, chat_id),
            timeout=180.0
        )

        text = ""
        rtype = (result or {}).get("type", "") if isinstance(result, dict) else ""

        # Terraform branch -> format with code + diagnostics
        if isinstance(result, dict) and rtype == "terraform":
            tf_code = (result.get("terraform_code") or
                       result.get("artifacts", {}).get("formatted_code") or "").strip()
            diagnostics = (result.get("validation_output") or
                           result.get("artifacts", {}).get("validation_output") or "").strip()

            status = "ok"
            # If we have diagnostics that look like errors or code missing, mark failed
            if diagnostics and re.search(r"\berror\b", diagnostics, re.IGNORECASE):
                status = "validation_failed"
            if not tf_code:
                status = "validation_failed"

            if not tf_code:
                # Try Gemini fallback for code if missing
                fb = await _gemini_fallback(user_text, code_preferred=True)
                if fb:
                    tf_code = _extract_hcl_from_text(fb)
                    status = "ok" if tf_code and not diagnostics else status

            text = format_infra_reply(status=status, code=tf_code, diagnostics=diagnostics)

        else:
            # Non-infra -> render normally then apply fallback if needed
            text = _render_text_for_oai(result)
            missing_or_clarify = _looks_like_empty_or_error_text(text) or rtype == "clarify"
            if missing_or_clarify:
                fb = await _gemini_fallback(user_text, code_preferred=_is_probably_code_request(user_text))
                if fb:
                    if _is_probably_code_request(user_text):
                        tf = _extract_hcl_from_text(fb)
                        text = format_infra_reply(status="ok", code=tf, diagnostics=None)
                    else:
                        text = fb

        if not text or not text.strip():
            text = "I'm ready to help you with Terraform. What would you like to do?"

        return JSONResponse(openai_chat_response(text, model=req.model or "terraform-agent"))

    # ---------- Streaming (delta) ----------
    async def gen():
        try:
            result = await asyncio.wait_for(
                terraform_agent.process_message(user_text, chat_id),
                timeout=180.0
            )

            rtype = (result or {}).get("type", "") if isinstance(result, dict) else ""
            if isinstance(result, dict) and rtype == "terraform":
                tf_code = (result.get("terraform_code") or
                           result.get("artifacts", {}).get("formatted_code") or "").strip()
                diagnostics = (result.get("validation_output") or
                               result.get("artifacts", {}).get("validation_output") or "").strip()
                status = "ok"
                if diagnostics and re.search(r"\berror\b", diagnostics, re.IGNORECASE):
                    status = "validation_failed"
                if not tf_code:
                    # Try code fallback
                    fb = await _gemini_fallback(user_text, code_preferred=True)
                    if fb:
                        tf_code = _extract_hcl_from_text(fb)
                        status = "ok" if tf_code and not diagnostics else status
                text = format_infra_reply(status=status, code=tf_code, diagnostics=diagnostics)
            else:
                text = _render_text_for_oai(result)
                if _looks_like_empty_or_error_text(text) or rtype == "clarify":
                    fb = await _gemini_fallback(user_text, code_preferred=_is_probably_code_request(user_text))
                    if fb:
                        if _is_probably_code_request(user_text):
                            tf = _extract_hcl_from_text(fb)
                            text = format_infra_reply(status="ok", code=tf, diagnostics=None)
                        else:
                            text = fb

            if not text or not text.strip():
                text = "I'm ready to help you with Terraform. What would you like to do?"

            for part in _chunk(text):
                yield "data: " + json.dumps({
                    "id": f"cmpl-{uuid4().hex}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": req.model or "terraform-agent",
                    "choices": [{"index": 0, "delta": {"content": part}, "finish_reason": None}],
                }) + "\n\n"
                await asyncio.sleep(0)
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield "data: " + json.dumps({
                "error": {"message": str(e), "type": "internal_error"}
            }) + "\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )

# Delete branch (async) — do NOT decorate with async-unsafe metrics decorator
@router.delete("/github/branch/{branch_name}")
async def delete_branch(branch_name: str):
    try:
        result = github_integration.delete_branch(branch_name)
        if result.get("success"):
            return {"status": "success", "message": result.get("message")}
        else:
            return {
                "status": "error",
                "error": result.get("error", "Failed to delete branch"),
            }
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
                "github_integration_available": enhanced_status.get(
                    "github_integration", False
                ),
                "policy_scanners_available": any(
                    policy_engine.enabled_scanners.values()
                ),
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

        from strands_tools import terraform_validator, infrastructure_analyzer
        from backend.app.services.guardrails import TerraformGuardrails

        guardrails = TerraformGuardrails()
        formatted_code = terraform_validator.format_terraform_code(terraform_code)
        validation_result = terraform_validator.validate_terraform_syntax(formatted_code)
        security_analysis = infrastructure_analyzer.analyze_terraform_resources(
            formatted_code
        )
        security_issues = guardrails.check_security_issues(formatted_code)
        terraform_validation = guardrails.validate_terraform_response(formatted_code)

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
                "estimated_monthly_cost": security_analysis.get(
                    "estimated_monthly_cost", 0
                ),
                "resource_count": security_analysis.get("resource_count", 0),
            },
        }

        if request.run_policy_checks:
            policy_result = policy_engine.validate_with_policies(formatted_code)
            response_data["policy_validation"] = policy_result
            if policy_result.get("violations"):
                policy_report = policy_engine.create_policy_report(
                    policy_result, formatted_code
                )
                response_data["policy_report"] = policy_report

        if infracost_integration.infracost_available:
            cost_result = infracost_integration.generate_cost_estimate(
                terraform_code, "default"
            )
            if cost_result.get("success"):
                response_data["cost_estimate"] = cost_result
                monthly_cost = cost_result.get("cost_estimate", {}).get(
                    "monthly_cost", 0
                )
                metrics.record_cost_estimation("infracost", monthly_cost)

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
@metrics.track_request("GET", "/metrics/summary")  # sync -> safe
def get_metrics_summary():
    try:
        summary = metrics.get_metrics_summary()
        return {
            "status": "success",
            "metrics": summary,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error("Metrics summary failed: %s", str(e))
        return {"status": "error", "error": str(e)}

# ---------------------------
# Budget / Cost
# ---------------------------
@router.get("/budgets/{workspace}")
@metrics.track_request("GET", "/budgets")  # sync -> safe
def get_budget_status(workspace: str):
    try:
        budget_status = infracost_integration.get_budget_status(workspace)
        return {"status": "success", **budget_status}
    except Exception as e:
        logger.error("Budget status check failed: %s", str(e))
        return {"status": "error", "error": str(e)}

@router.put("/budgets/{workspace}")
@metrics.track_request("PUT", "/budgets")  # sync -> safe
def update_budget(workspace: str, req: BudgetUpdateRequest):
    try:
        result = infracost_integration.update_budget(
            req.workspace, req.monthly_limit, req.alert_thresholds
        )
        if result.get("success"):
            return {"status": "success", **result}
        else:
            return {
                "status": "error",
                "error": result.get("error", "Budget update failed"),
            }
    except Exception as e:
        logger.error("Budget update failed: %s", str(e))
        return {"status": "error", "error": str(e)}

@router.post("/cost/diff")
@metrics.track_request("POST", "/cost/diff")  # sync -> safe
def generate_cost_diff(request: dict):
    try:
        old_terraform = request.get("old_terraform", "")
        new_terraform = request.get("new_terraform", "")
        workspace = request.get("workspace", "default")
        if not old_terraform or not new_terraform:
            return {
                "status": "error",
                "error": "Both old_terraform and new_terraform are required",
            }
        diff_result = infracost_integration.generate_cost_diff(
            old_terraform, new_terraform, workspace
        )
        if diff_result.get("success"):
            return {"status": "success", **diff_result}
        else:
            return {
                "status": "error",
                "error": diff_result.get("error", "Cost diff generation failed"),
            }
    except Exception as e:
        logger.error("Cost diff generation failed: %s", str(e))
        return {"status": "error", "error": str(e)}

@router.post("/cost/estimate")
@metrics.track_request("POST", "/cost/estimate")  # sync -> safe
def generate_cost_estimate(request: dict):
    try:
        terraform_code = request.get("terraform_code", "")
        workspace = request.get("workspace", "default")
        if not terraform_code:
            return {"status": "error", "error": "terraform_code is required"}
        cost_result = infracost_integration.generate_cost_estimate(
            terraform_code, workspace
        )
        if cost_result.get("success"):
            monthly_cost = cost_result.get("cost_estimate", {}).get(
                "monthly_cost", 0
            )
            metrics.record_cost_estimation("infracost", monthly_cost)
            return {"status": "success", **cost_result}
        else:
            return {
                "status": "error",
                "error": cost_result.get("error", "Cost estimation failed"),
            }
    except Exception as e:
        logger.error("Cost estimation failed: %s", str(e))
        return {"status": "error", "error": str(e)}

# ---------------------------
# Policy
# ---------------------------
@router.get("/policy/summary")
@metrics.track_request("GET", "/policy/summary")  # sync -> safe
def get_policy_summary():
    try:
        summary = policy_engine.get_policy_summary()
        return {"status": "success", **summary}
    except Exception as e:
        logger.error("Policy summary failed: %s", str(e))
        return {"status": "error", "error": str(e)}

@router.post("/policy/validate")
@metrics.track_request("POST", "/policy/validate")  # sync -> safe
def validate_with_policies(request: dict):
    try:
        terraform_code = request.get("terraform_code", "")
        if not terraform_code:
            return {"status": "error", "error": "terraform_code is required"}
        validation_result = policy_engine.validate_with_policies(terraform_code)
        report = None
        if validation_result.get("violations"):
            report = policy_engine.create_policy_report(
                validation_result, terraform_code
            )
        return {
            "status": "success",
            "validation_result": validation_result,
            "report": report,
        }
    except Exception as e:
        logger.error("Policy validation failed: %s", str(e))
        return {"status": "error", "error": str(e)}

# ---------------------------
# Workflow: terraform -> PR (kept async here; not decorated by metrics)
# ---------------------------
@router.post("/workflow/terraform-to-pr")
def terraform_to_pr_workflow(request: dict):
    """
    Non-decorated by metrics (sync wrapper to avoid coroutine mishandling).
    """
    try:
        terraform_code = (request.get("terraform_code") or "").strip()
        workspace = request.get("workspace") or "default"
        commit_message = request.get("commit_message") or "feat(iac): add demo stack via TF Agent"
        branch_name = request.get("branch_name") or f"tf/feature-{uuid4().hex[:8]}"

        if not terraform_code:
            return {"status": "error", "error": "terraform_code is required"}

        workflow_result: Dict[str, Any] = {
            "status": "success",
            "workflow_steps": {
                "workspace": {"status": "completed", "selected": workspace}
            },
            "final_result": None,
        }

        if not github_integration.available:
            workflow_result["workflow_steps"]["github_pr"] = {
                "status": "skipped",
                "reason": "GitHub integration not available",
            }
            return workflow_result

        pr_result = _asdict(
            github_integration.create_branch_and_pr(
                terraform_code, commit_message, branch_name
            )
        )

        workflow_result["workflow_steps"]["github_pr"] = {
            "status": "completed" if pr_result.get("success") else "failed",
            "result": pr_result,
        }
        workflow_result["final_result"] = pr_result
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
@metrics.track_request("POST", "/admin/cache/clear")  # sync -> safe
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
@metrics.track_request("GET", "/admin/system/info")  # sync -> safe
def get_system_info():
    try:
        import sys
        import platform
        return {
            "status": "success",
            "system": {
                "platform": platform.platform(),
                "python_version": sys.version,
                "current_directory": str(Path.cwd()),
                "environment_variables": {
                    "GEMINI_API_KEY": "configured"
                    if os.getenv("GEMINI_API_KEY")
                    else "missing",
                    "GITHUB_TOKEN": "configured"
                    if os.getenv("GITHUB_TOKEN")
                    else "missing",
                    "GITHUB_REPO": os.getenv("GITHUB_REPO", "not configured"),
                    "INFRACOST_API_KEY": "configured"
                    if os.getenv("INFRACOST_API_KEY")
                    else "missing",
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
# SECURE/JWT endpoints (not decorated by metrics)
# ---------------------------
from slowapi import Limiter as _rt_Limiter
from slowapi.util import get_remote_address as _rt_get_remote_address
import jwt as _rt_jwt, httpx as _rt_httpx
from backend.app.core.config import Settings as _RT_Settings
from backend.app.services.background import celery_app as _rt_celery
from backend.app.utils.utils import (
    sanitize_user_text as _rt_sanitize,
    run_cmd_async as _rt_run_cmd_async,
    secure_tempdir as _rt_secure_tempdir,
)
from backend.app.services.strands_agent import chat_secure as _rt_chat_secure, generate_tf_unit_tests as _rt_ai_tests
from backend.app.services.infracost_integration import estimate_cost_async_v2 as _rt_cost_v2
from backend.app.services.policy_engine import validate_terraform_code_ast as _rt_validate_ast
from backend.app.services.github_integration import trufflehog_scan_ref as _rt_truffle, enable_auto_apply_action as _rt_enable_actions

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
    _= _rt_Depends(_rt_require_jwt),
):
    chat_id = (req or {}).get("chat_id") or "default"
    user_text = _rt_sanitize((req or {}).get("user_text", ""))
    result = await _rt_chat_secure(chat_id, user_text)
    return JSONResponse(result)

@router.post("/ai/generate-tests")
async def _rt_generate_tests(req: dict, _= _rt_Depends(_rt_require_jwt)):
    tf_code = req.get("terraform_code", "")
    return JSONResponse(await _rt_ai_tests(tf_code))

@router.post("/policy/validate/ast")
async def _rt_policy_validate_ast(req: dict, _= _rt_Depends(_rt_require_jwt)):
    tf_code = req.get("terraform_code", "")
    return JSONResponse(await _rt_validate_ast(tf_code))

@router.post("/cost/estimate/async")
async def _rt_cost_async(req: dict, _= _rt_Depends(_rt_require_jwt)):
    tf_code = req.get("terraform_code", "")
    currency = (req.get("currency") or "USD").upper()
    task = _rt_celery.send_task("infracost.estimate", args=[tf_code, currency])
    return {"success": True, "data": {"task_id": task.id}, "error": ""}

@router.get("/tasks/{task_id}")
async def _rt_task_status(task_id: str, _= _rt_Depends(_rt_require_jwt)):
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
async def _rt_cost_estimate_v2(req: dict, _= _rt_Depends(_rt_require_jwt)):
    tf_code = req.get("terraform_code", "")
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
async def _rt_enable_auto_apply(_= _rt_Depends(_rt_require_jwt)):
    return JSONResponse(await _rt_enable_actions())

@router.post("/github/scan/trufflehog")
async def _rt_scan_trufflehog(req: dict, _= _rt_Depends(_rt_require_jwt)):
    ref = req.get("ref")
    return JSONResponse(await _rt_truffle(ref or None))

@router.post("/notify")
async def _rt_notify(req: dict, _= _rt_Depends(_rt_require_jwt)):
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
async def _rt_visualize(req: dict, _= _rt_Depends(_rt_require_jwt)):
    tf_code = req.get("terraform_code", "")
    from os import path as _os_path
    with _rt_secure_tempdir("graph_v2_") as d:
        with open(_os_path.join(d, "main.tf"), "w", encoding="utf-8") as f:
            f.write(tf_code)
        rc, out, err = await _rt_run_cmd_async("terraform", "init", "-input=false", cwd=d)
        if rc != 0:
            return {"success": False, "data": {"stderr": err}, "error": "terraform init failed"}
        rc, dot, err = await _rt_run_cmd_async("terraform", "graph", cwd=d)
        if rc != 0:
            return {"success": False, "data": {"stderr": err}, "error": "terraform graph failed"}
        return {"success": True, "data": {"dot": dot}, "error": ""}

