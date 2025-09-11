# -*- coding: utf-8 -*-
import asyncio
import contextlib
import html
import logging
import os
import shutil
import tempfile
from typing import Any, Dict, Optional, Tuple

import httpx
from tenacity import (retry, retry_if_exception_type, stop_after_attempt,
                      wait_exponential)

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def secure_tempdir(prefix: str = "tfagent_"):
    d = tempfile.mkdtemp(prefix=prefix)
    try:
        yield d
    finally:
        with contextlib.suppress(Exception):
            shutil.rmtree(d, ignore_errors=True)


def sanitize_user_text(text: str) -> str:
    return html.escape(text or "", quote=True)


async def run_cmd_async(
    *args: str,
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    timeout: Optional[int] = 600,
) -> Tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        *args,
        cwd=cwd,
        env={**os.environ, **(env or {})},
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError as e:
        proc.kill()
        raise RuntimeError(f"Command timed out: {' '.join(args)}") from e
    return (
        proc.returncode,
        stdout.decode("utf-8", "replace"),
        stderr.decode("utf-8", "replace"),
    )


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((httpx.ConnectError, httpx.ReadTimeout)),
)
async def http_json_get(
    url: str, headers: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.get(url, headers=headers)
        resp.raise_for_status()
        return resp.json()
