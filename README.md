
# Terraform Agent

> A FastAPI + Celery “agent” that can generate, validate, plan, cost, and policy-check Terraform projects from natural-language prompts, with GitHub PR workflows, Infracost estimates, policy scanners (Checkov/policy engine), and Prometheus observability.

Built for a local/dev workflow first: run an API, queue background jobs, talk to it with HTTP or an OpenAI-compatible `/v1/chat/completions` endpoint, and wire it into CI.

---

## ✨ Features

* **FastAPI HTTP API** with typed responses and helpful errors
* **LLM endpoints** (`/chat`, `/v1/chat/completions`) for “NL → Terraform” assistance
* **Terraform validation** (`/validate`) + **PR workflows** (`/workflow/terraform-to-pr`)
* **Costs with Infracost** (`/cost/estimate`, `.../diff`, async variant & v2)
* **Policy scan/AST validate** (`/policy/validate`, `/policy/validate/ast`) and summaries
* **GitHub integration** (delete branches, enable auto-apply, PR flows)
* **Prometheus metrics** (`/metrics`, `/metrics/summary`)
* **Background jobs** with Celery + Redis
* **Pre-commit linting**: Black, isort, Flake8 (config in `setup.cfg`)

The API surface (routes) is defined in `src/app/api/routes.py`.

---

## 🚀 Quickstart

### 1) Prerequisites

* Python 3.11
* Redis (for Celery): `docker run -p 6379:6379 redis:7`
* Terraform CLI
* Infracost CLI + API key
* (Optional) Checkov for policy scanning
* Git installed & a GitHub token if you’ll use repo/PR features

### 2) Install (Poetry)

```bash
# from repo root
poetry install
poetry run pre-commit install
````

Dependencies come from `pyproject.toml`. Dev tools include Black, isort, pytest, mypy.

### 3\) Environment

Create `.env` (or export vars) — as referenced in `routes.py` and config:

```env
# CORS
ALLOWED_ORIGINS=*

# Working dirs / chat scratch
WORKING_DIRECTORY=/tmp/terraform-agent
CHAT_DATA_DIR=.cache/chat

# Infracost
INFRACOST_API_KEY=your_infracost_key

# GitHub (optional but needed for PR flows)
GITHUB_TOKEN=ghp_xxx
GITHUB_REPO=owner/repo
GITHUB_BASE_BRANCH=main

# Input guardrail
MAX_INPUT_CHARS=8000
```

These are the variables read directly in the API layer. Additional service modules (e.g., GitHub, policy, or Celery) may use their own settings—check `src/app/core/config.py` if you add more.

### 4\) Run the API

```bash
# tell uvicorn to use `src` as the app dir
poetry run uvicorn main:app --app-dir src --reload --port 8000
```

  * **Swagger:** `http://localhost:8000/docs`
  * **OpenAPI JSON:** `http://localhost:8000/openapi.json`
  * **Prometheus:** `http://localhost:8000/metrics`

### 5\) Run workers (background jobs)

Open another terminal:

```bash
# example (adjust module/path to your Celery app if needed)
poetry run celery -A app.services.background worker --loglevel=INFO --workdir src
```

-----

## 🔌 API overview

Paths and verbs below come from `src/app/api/routes.py`. The agent glues together Terraform, Infracost, policy tools, GitHub, and LLMs.

### Health & meta

| Method | Path | Notes |
| :--- | :--- | :--- |
| `GET` | `/ping` | Liveness |
| `GET` | `/test` | Simple echo/smoke |
| `GET` | `/health` | Health info |

### Chat / LLM

| Method | Path | Notes |
| :--- | :--- | :--- |
| `POST` | `/chat` | NL prompt → assistant response |
| `POST` | `/chat/stream` | Server-sent events/streaming (chunked) |
| `GET` | `/v1/models` | Model list (OpenAI-style) |
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat completions |

### Tools / status

| Method | Path | Notes |
| :--- | :--- | :--- |
| `GET` | `/tools/status` | Tooling availability/status snapshot |

### Terraform validation

| Method | Path | Notes |
| :--- | :--- | :--- |
| `POST` | `/validate` | Validate Terraform snippets/project |

### Costs (Infracost)

| Method | Path | Notes |
| :--- | :--- | :--- |
| `POST` | `/cost/estimate` | Synchronous estimate |
| `POST` | `/cost/estimate/v2` | Extended estimate v2 |
| `POST` | `/cost/diff` | Cost diff between states/commits |
| `POST` | `/cost/estimate/async` | Queue async estimate (returns task id) |
| `GET` | `/tasks/{task_id}` | Poll async estimate task result |

### Policy & security

| Method | Path | Notes |
| :--- | :--- | :--- |
| `POST` | `/policy/validate` | Policy scan (Checkov / custom rules) |
| `POST` | `/policy/validate/ast`| AST-based policy validation |
| `GET` | `/policy/summary` | Policy summary |

### GitHub operations

| Method | Path | Notes |
| :--- | :--- | :--- |
| `DELETE`| `/github/branch/{owner}/{repo}/{branch}` | Delete a remote branch |
| `POST` | `/github/enable-auto-apply` | Enable auto-apply for workflows/PRs |
| `POST` | `/workflow/terraform-to-pr` | Generate plan → open/update PR workflow |

### Admin & metrics

| Method | Path | Notes |
| :--- | :--- | :--- |
| `GET` | `/metrics` | Prometheus metrics endpoint |
| `GET` | `/metrics/summary` | Human-friendly metrics summary |
| `POST` | `/admin/cache/clear` | Clear agent caches |
| `GET` | `/system/info` | System & toolchain info |

### Extras

| Method | Path | Notes |
| :--- | :--- | :--- |
| `POST` | `/secure/chat` | Chat with extra secrets/guardrails |
| `POST` | `/ai/generate-tests`| Generate tests from code/spec |
| `POST` | `/scan/trufflehog` | Secret scan on a repo/path |
| `POST` | `/notify` | Send a notification (channel-agnostic) |
| `POST` | `/visualize` | Turn JSON/state into a diagram/image |

-----

## 🧪 Usage examples

### Health

```bash
curl http://localhost:8000/health
```

### Validate Terraform

```bash
curl -X POST http://localhost:8000/validate \
  -H "Content-Type: application/json" \
  -d '{"files": [{"path":"main.tf","content":"terraform { required_version = \">= 1.6.0\" }"}]}'
```

### Cost estimate (sync)

```bash
curl -X POST http://localhost:8000/cost/estimate \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $INFRACOST_API_KEY" \
  -d '{"path": "/path/to/terraform/project"}'
```

### Cost estimate (async)

```bash
# queue
TASK=$(curl -s -X POST http://localhost:8000/cost/estimate/async \
  -H "Content-Type: application/json" \
  -d '{"path": "/path/to/terraform/project"}' | jq -r .task_id)

# poll
curl "http://localhost:8000/tasks/$TASK"
```

### OpenAI-compatible chat

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "gpt-4o-mini",
        "messages": [
          {"role":"system","content":"You are a Terraform assistant."},
          {"role":"user","content":"Create an S3 bucket and output its ARN."}
        ]
      }'
```

-----

## ⚙️ Configuration & environment

The API reads the following environment variables (directly referenced in the HTTP layer):

  * `ALLOWED_ORIGINS` – CORS allowlist (e.g., `*` or a CSV)
  * `WORKING_DIRECTORY` – filesystem workspace for operations
  * `CHAT_DATA_DIR` – optional cache/scratch for chat memory
  * `INFRACOST_API_KEY` – required for cost endpoints
  * `GITHUB_TOKEN`, `GITHUB_REPO`, `GITHUB_BASE_BRANCH` – for GitHub operations
  * `MAX_INPUT_CHARS` – guardrail to reject overly large prompts

Service modules under `src/app/services/` may use additional vars (e.g., policy toggles). Check `src/app/core/config.py` if you extend the app.

-----

## 🔍 Linting & pre-commit

Hooks are configured to run on every commit:

  * trailing whitespace / EOF newline
  * YAML checks
  * Black, isort, Flake8

Install & run:

```bash
poetry run pre-commit install
poetry run pre-commit run --all-files
```

-----

## 🧱 CI

A “lint-test” workflow runs on pull requests and pushes. Keep your local hooks green before pushing to avoid CI failures.

-----

## 🧭 Development tips

  * Start Redis before Celery.
  * If `uvicorn` can’t find the app, include `--app-dir src`.
  * Infracost endpoints require a valid `INFRACOST_API_KEY`.
  * GitHub endpoints require `GITHUB_TOKEN` with `repo` scope.
  * For large chats or files, tune `MAX_INPUT_CHARS`.

-----

## 📜 License

MIT © 2025 Abhinav Mishra. See [LICENSE](./LICENSE) for details.

-----

## 🙌 Acknowledgements

  * [FastAPI](https://fastapi.tiangolo.com/)
  * [Celery](https://docs.celeryq.dev/en/stable/)
  * [Infracost](https://www.infracost.io/)
  * [Checkov](https://www.checkov.io/)
  * [Prometheus Python client](https://github.com/prometheus/client_python)

<!-- end list -->

```
```
