# src/app/services/strands_compat.py
from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List


# --------------------------------------------
# Unified Strands caller (tuned for 1.9.0)
# --------------------------------------------
def call_agent(agent: Any, prompt: str, **kwargs) -> Any:
    """
    Call Strands Agent regardless of which method it exposes.
    Prefers __call__ because strands-agents==1.9.0 implements that.
    """
    # Method preference order tuned for your installed SDK
    candidates = ["__call__", "run", "respond", "invoke", "generate", "ask", "ainvoke"]
    for name in candidates:
        fn = getattr(agent, name, None)
        if callable(fn):
            # Many APIs accept just the prompt; pass kwargs if supported
            try:
                return fn(prompt, **kwargs)
            except TypeError:
                # Fallback if the signature doesn't accept kwargs
                return fn(prompt)
    raise RuntimeError("Strands Agent has no supported inference method")


# --------------------------------------------
# Tool schema helpers
# --------------------------------------------
def _py_to_schema(t: Any) -> str:
    if t in (int, float):
        return "number"
    if t is bool:
        return "boolean"
    return "string"


def _make_tool_spec(fn: Callable) -> Dict[str, Any]:
    sig = inspect.signature(fn)
    params = []
    for name, p in sig.parameters.items():
        ann = p.annotation if p.annotation is not inspect._empty else str
        params.append(
            {
                "name": name,
                "type": _py_to_schema(ann),
                "required": (p.default is inspect._empty),
                "description": "",  # keep simple; Strands can accept empty descriptions
            }
        )
    return {
        "name": fn.__name__,
        "description": (fn.__doc__ or fn.__name__).strip(),
        "parameters": params,
        # v2-style specs typically carry a callable handler
        "handler": fn,
    }


# --------------------------------------------
# Universal tool registrar (multi-API support)
# --------------------------------------------
def register_tools(agent: Any, tools: List[Callable]) -> str:
    """
    Try multiple registration styles. Return a short string telling which API worked.
    Raises RuntimeError if none matched (caller can decide to continue without tools).
    """
    specs = [_make_tool_spec(t) for t in tools]

    # 1) Agent-level bulk register
    reg = getattr(agent, "register_tools", None)
    if callable(reg):
        reg(specs)
        return "agent.register_tools"

    # 2) Callable .tools([...])
    tools_attr = getattr(agent, "tools", None)
    if callable(tools_attr):
        tools_attr(specs)
        return "agent.tools(callable)"

    # 3) Per-tool add_tool(...)
    add_tool = getattr(agent, "add_tool", None)
    if callable(add_tool):
        for t in tools:
            add_tool(t)
        return "agent.add_tool"

    # 4) Dict-like .tools[...] (rare but seen)
    if isinstance(tools_attr, dict):
        for t in tools:
            name = getattr(t, "__name__", getattr(t, "name", None)) or f"tool_{id(t)}"
            tools_attr[name] = t
        return "agent.tools(dict)"

    # 5) Last-ditch: module registry (older/legacy shapes)
    try:
        from strands import tools as st_mod  # may not exist in 1.9.0

        registry = getattr(st_mod, "registry", None)
        if registry and hasattr(registry, "register") and callable(registry.register):
            for t in tools:
                registry.register(t)
            return "module.registry.register"
    except Exception:
        pass

    raise RuntimeError("No known Strands tool registration API in this SDK")
