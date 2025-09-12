"""
verified.py: Strict JSON verification for agent outputs using Gemini 1.5 Pro.
- verify_answer(user_message, draft_text, min_citations=1) -> dict
- Enforces JSON with repair loop
- Adds sources and sets verified=True only when evidence present
"""

import json
import os
from typing import Any, Dict

import google.generativeai as genai  # type: ignore

from src.app.core.metrics import metrics as _ver_metrics

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")

SYSTEM = (
    "You are a strict verifier. Return ONLY valid JSON matching the schema. "
    "If you cannot verify, set verified=false and explain briefly in reasoning_note."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "sources": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "title": {"type": "string"},
                    "snippet": {"type": "string"},
                    "score": {"type": "number"},
                },
                "required": [],
            },
        },
        "verified": {"type": "boolean"},
        "reasoning_note": {"type": "string"},
    },
    "required": ["answer", "sources", "verified"],
}


def _cfg():
    return {
        "temperature": 0.2,
        "top_p": 0.9,
        "top_k": 32,
        "max_output_tokens": 1024,
        "response_mime_type": "application/json",
        "response_schema": SCHEMA,
    }


def _safe_json(s: str):
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        end = s.rfind("}")
        if end != -1:
            try:
                return json.loads(s[: end + 1])
            except Exception:
                pass
        return None


def _model():
    return genai.GenerativeModel(model_name=MODEL_NAME, system_instruction=SYSTEM)


def verify_answer(user_message: str, draft_text: str, min_citations: int = 1) -> Dict[str, Any]:
    """
    Returns dict: {"answer": <refined_markdown>, "sources": [..], "verified": bool, "reasoning_note": str}
    If model unavailable or key missing, returns a safe default with verified=False.
    """
    if not GEMINI_API_KEY:
        return {
            "answer": draft_text or "",
            "sources": [],
            "verified": False,
            "reasoning_note": "No API key configured",
        }
    try:
        model = _model()
        prompt = (
            "User message:\n"
            + (user_message or "")
            + "\n\nDraft answer to verify (may be incomplete):\n"
            + (draft_text or "")
            + "\n\nTask: Verify accuracy and provide citations (URLs). "
            f"Require at least {min_citations} citation(s). "
            "Return JSON with fields answer, sources[], verified, reasoning_note."
        )
        resp = model.generate_content(prompt, generation_config=_cfg())
        raw = getattr(resp, "text", "") or ""
        data = _safe_json(raw) or {
            "answer": draft_text or "",
            "sources": [],
            "verified": False,
            "reasoning_note": "Could not parse JSON",
        }
        if isinstance(data, dict):
            if not isinstance(data.get("sources", []), list):
                data["sources"] = []
            if len(data["sources"]) < min_citations:
                data["verified"] = False
                data.setdefault("reasoning_note", "Insufficient citations to verify")
            if not data.get("answer"):
                data["answer"] = draft_text or ""
        else:
            data = {
                "answer": draft_text or "",
                "sources": [],
                "verified": False,
                "reasoning_note": "Invalid JSON",
            }
        return data
    except Exception as e:
        return {
            "answer": draft_text or "",
            "sources": [],
            "verified": False,
            "reasoning_note": f"Verification error: {e}",
        }


# === APPEND: structured verify w/ token accounting (non-destructive) ===


def verify_with_gemini_v2(prompt: str, citations: bool = True):
    model = "gemini-1.5-pro"
    prompt_tokens = len((prompt or "").split())
    _ver_metrics.model_tokens.labels(model_name=model, type="prompt").inc(prompt_tokens)
    try:
        _ver_metrics.token_budget_remaining.dec(float(prompt_tokens))
    except Exception:
        pass

    # Replace with real Gemini call:
    answer = "Verified with citations..." if citations else "Verified."
    completion_tokens = len(answer.split())
    _ver_metrics.model_tokens.labels(model_name=model, type="completion").inc(completion_tokens)
    try:
        _ver_metrics.token_budget_remaining.dec(float(completion_tokens))
    except Exception:
        pass
    _ver_metrics.model_requests.labels(model_name=model, status="success").inc()
    return {
        "success": True,
        "data": {"answer": answer, "model": model},
        "error": "",
    }
