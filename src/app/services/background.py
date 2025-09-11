# -*- coding: utf-8 -*-
import asyncio
from celery import Celery

from backend.app.core.config import Settings
from backend.app.services.infracost_integration import (
    infracost_integration,
)  # existing object if any
from backend.app.utils.utils import run_cmd_async, secure_tempdir
from backend.app.services.infracost_integration import (
    estimate_cost_async_v2,
)  # appended below

settings = Settings()

celery_app = Celery(
    "tfagent",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)


@celery_app.task(name="infracost.estimate")
def infracost_estimate_task(tf_code: str, currency: str = "USD"):
    return asyncio.run(estimate_cost_async_v2(tf_code=tf_code, currency=currency))
