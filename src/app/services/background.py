# -*- coding: utf-8 -*-
import asyncio

from backend.app.core.config import Settings
from backend.app.services.infracost_integration import (  # existing object if any; appended below
    estimate_cost_async_v2, infracost_integration)
from backend.app.utils.utils import run_cmd_async, secure_tempdir
from celery import Celery

settings = Settings()

celery_app = Celery(
    "tfagent",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)


@celery_app.task(name="infracost.estimate")
def infracost_estimate_task(tf_code: str, currency: str = "USD"):
    return asyncio.run(estimate_cost_async_v2(tf_code=tf_code, currency=currency))
