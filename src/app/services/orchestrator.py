# src/app/services/orchestrator.py
from __future__ import annotations

import asyncio
import re
import time
from enum import Enum
from typing import Any, AsyncGenerator, Callable, Dict, Optional

from pydantic import BaseModel

from src.app.services.strands_agent import TerraformAgent

from .helpers import (
    ai_repair_terraform,
    configure_gemini,
    extract_hcl,
    format_infra_reply,
    gemini_fallback,
    has_bad_local_names,
    has_bad_s3_inline,
    infracost_breakdown_hcl,
    looks_empty_or_error,
    read_main_tf,
    save_main_tf,
)
from .intents import Intent, detect_intent


class ReplyType(str, Enum):
    CHAT = "chat"
    TERRAFORM = "terraform"
    CLARIFY = "clarify"
    ERROR = "error"


class AgentReply(BaseModel):
    status: str = "success"
    type: ReplyType = ReplyType.CHAT
    response: str = ""
    terraform_code: str = ""
    diagnostics: str = ""
    cost_estimate: Optional[dict] = None
    github_status: Optional[dict] = None
    questions: Any = None
    thinking: str = ""
    tool_status: Dict[str, Any] = {}
    chat_id: str = ""
    confidence_score: float = 0.0
    artifacts: Dict[str, Any] = {}
    analysis: Dict[str, Any] = {}
    request_id: Optional[str] = None
    # Optional verification fields (populated if verify_fn provided)
    verified: Optional[bool] = None
    sources: Optional[list] = None
    answer_refined: Optional[str] = None
    verification_note: Optional[str] = None


configure_gemini()


def _with_three_sections(user_text: str) -> str:
    # Your “A+” prefix — keeps output tidy when generating code.
    FORMAT_DIRECTIVE = (
        "Respond in exactly three sections:\n"
        "## Summary\n- 1–3 bullets.\n\n"
        "## Steps You Will Run\n1. ...\n\n"
        "## Result\nCode or outputs, if any."
    )
    return f"{FORMAT_DIRECTIVE}\n\n{user_text}"


async def _agent_call(agent: TerraformAgent, prompt: str, chat_id: str, timeout: float = 180.0):
    return await asyncio.wait_for(agent.process_message(prompt, chat_id), timeout=timeout)


async def _verify(
    user_text: str, reply: AgentReply, verify_fn: Optional[Callable[[str, str], dict]]
):
    if not verify_fn or not reply.response:
        return reply
    try:
        v = verify_fn(user_text, reply.response)
        reply.verified = bool(v.get("verified", False))
        reply.sources = v.get("sources", [])
        reply.answer_refined = v.get("answer", reply.response)
        reply.verification_note = v.get("reasoning_note", "")
    except Exception as e:
        reply.verified = False
        reply.sources = []
        reply.answer_refined = reply.response
        reply.verification_note = f"Verification error: {e}"
    return reply


async def run_pipeline(
    user_text: str,
    chat_id: str,
    agent: TerraformAgent,
    verify_fn: Optional[Callable[[str, str], dict]] = None,
) -> AgentReply:
    intent = detect_intent(user_text)
    code_pref = intent in {Intent.GENERATE_TF, Intent.VALIDATE_ONLY, Intent.VISUALIZE}
    prompt = _with_three_sections(user_text) if code_pref else user_text

    # Fast COST
    if intent == Intent.ASK_COST:
        last = read_main_tf().strip()
        if last:
            cost = infracost_breakdown_hcl(last)
            if cost.get("success"):
                body = (
                    "## Summary\n- Estimated cost for your current Terraform (`main.tf`).\n\n"
                    "## Steps You Will Run\n1. Load `main.tf` from disk.\n2. Run Infracost breakdown.\n\n"
                    f"## Result\n{format_infra_reply('ok', None, None, cost_estimate=cost, saved_to='main.tf')}"
                )
                reply = AgentReply(
                    type=ReplyType.CHAT,
                    response=body,
                    cost_estimate=cost,
                    chat_id=chat_id,
                    confidence_score=0.9,
                )
                return await _verify(user_text, reply, verify_fn)

    # Normal agent call
    try:
        result = await _agent_call(agent, prompt, chat_id)
    except asyncio.TimeoutError:
        return AgentReply(
            status="error", type=ReplyType.ERROR, response="Request timed out", chat_id=chat_id
        )

    rtype = (result or {}).get("type", "") if isinstance(result, dict) else ""
    content = (result or {}).get("content", "") if isinstance(result, dict) else ""
    tf_code = ""
    if isinstance(result, dict):
        tf_code = (
            result.get("terraform_code") or result.get("artifacts", {}).get("formatted_code") or ""
        ).strip()

    need_fallback = (
        result is None
        or rtype == "clarify"
        or looks_empty_or_error(content)
        or (rtype == "terraform" and not tf_code)
    )

    if need_fallback:
        fb = await gemini_fallback(prompt, code_preferred=code_pref, retries=1)
        if fb:
            if code_pref:
                tf = extract_hcl(fb)
                if has_bad_s3_inline(tf) or has_bad_local_names(tf):
                    repaired = await ai_repair_terraform(user_text, tf)
                    tf = repaired or tf
                saved = save_main_tf(tf)
                body = format_infra_reply("ok", tf, None, saved_to=saved)
                reply = AgentReply(
                    type=ReplyType.TERRAFORM,
                    response=body,
                    terraform_code=tf,
                    chat_id=chat_id,
                    confidence_score=0.7,
                )
                return await _verify(user_text, reply, verify_fn)
            else:
                reply = AgentReply(
                    type=ReplyType.CHAT, response=fb, chat_id=chat_id, confidence_score=0.7
                )
                return await _verify(user_text, reply, verify_fn)
        return AgentReply(
            status="error", type=ReplyType.ERROR, response="No response generated", chat_id=chat_id
        )

    # We have a result
    if rtype == "chat":
        reply = AgentReply(
            type=ReplyType.CHAT,
            response=result.get("content", ""),
            chat_id=result.get("chat_id", chat_id),
            confidence_score=result.get("confidence_score", 0.0),
        )
        return await _verify(user_text, reply, verify_fn)

    if rtype == "clarify":
        fb = await gemini_fallback(prompt, code_preferred=False)
        if fb:
            reply = AgentReply(
                type=ReplyType.CHAT,
                response=fb,
                chat_id=result.get("chat_id", chat_id),
                confidence_score=0.6,
            )
            return await _verify(user_text, reply, verify_fn)
        return AgentReply(
            type=ReplyType.CLARIFY,
            response="I need more information to help you better.",
            questions=result.get("questions", "Could you provide more details?"),
            chat_id=result.get("chat_id", chat_id),
        )

    if rtype == "terraform":
        terraform_code = tf_code
        diagnostics = (
            result.get("validation_output")
            or result.get("artifacts", {}).get("validation_output")
            or ""
        ).strip()

        if terraform_code and (
            has_bad_s3_inline(terraform_code) or has_bad_local_names(terraform_code)
        ):
            rep = await ai_repair_terraform(user_text, terraform_code)
            terraform_code = rep or terraform_code

        saved = save_main_tf(terraform_code)

        # Attach cost (A+ UX)
        cost_estimate = result.get("cost_estimate", "")
        if not cost_estimate or cost_estimate == "Skipped":
            c = infracost_breakdown_hcl(terraform_code)
            if c.get("success"):
                cost_estimate = c

        status = "ok"
        if diagnostics and re.search(r"\berror\b", diagnostics, re.I):
            status = "validation_failed"

        body = format_infra_reply(
            status, terraform_code, diagnostics, cost_estimate=cost_estimate, saved_to=saved
        )
        reply = AgentReply(
            type=ReplyType.TERRAFORM,
            response=body,
            terraform_code=terraform_code,
            cost_estimate=cost_estimate if isinstance(cost_estimate, dict) else None,
            artifacts={
                "formatted_code": result.get("formatted_code", "") or terraform_code,
                "validation_output": result.get("validation_output", ""),
                "diff": result.get("diff", ""),
            },
            analysis={
                "security_findings": result.get("security_findings", []),
                "best_practices": result.get("best_practices", []),
                "blast_radius_warnings": result.get("blast_radius_warnings", []),
            },
            chat_id=result.get("chat_id", chat_id),
            confidence_score=result.get("confidence_score", 0.0),
        )
        return await _verify(user_text, reply, verify_fn)

    return AgentReply(
        status="error",
        type=ReplyType.ERROR,
        response=f"Unexpected response type: {rtype}",
        chat_id=chat_id,
    )


# --- Streaming variant (SSE-friendly) ---
async def stream_pipeline(
    user_text: str,
    chat_id: str,
    agent: TerraformAgent,
    verify_fn: Optional[Callable[[str, str], dict]] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    yield {"heartbeat": True}
    reply = await run_pipeline(user_text, chat_id, agent, verify_fn=verify_fn)
    payload = {k: v for k, v in reply.model_dump().items() if v or k in {"status", "type"}}
    payload["done"] = True
    yield payload
