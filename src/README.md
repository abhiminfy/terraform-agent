# Terraform Agent (FastAPI)

Additive improvements without removing your existing features.

## Setup
- Copy `.env` (JWT_SECRET, GEMINI_API_KEY, REDIS/CELERY, GITHUB_TOKEN/REPO, PROM_PUSHGATEWAY)
- `pip install -r requirements.txt`
- Start Redis & Celery: `celery -A background.celery_app worker -l info`
- `uvicorn main:app --reload`
- Prometheus scrapes `/metrics`; optional Pushgateway; Alerts in `prometheus_rules.yml`.

## Added Endpoints (non-breaking)
- `/metrics` & `/metrics/summary`
- `/cost/estimate/async`, `/tasks/{task_id}`
- `/github/enable-auto-apply`, `/github/token-scopes`
- `/notify` (Slack/Teams)
- `/visualize` (Terraform DOT)
- `/ai/generate-tests` (TF unit-test ideas)
