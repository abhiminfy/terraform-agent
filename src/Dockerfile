FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl unzip ca-certificates git graphviz \
 && rm -rf /var/lib/apt/lists/*

# Terraform
RUN curl -fsSL https://releases.hashicorp.com/terraform/1.7.5/terraform_1.7.5_linux_amd64.zip -o tf.zip \
 && unzip tf.zip -d /usr/local/bin && rm tf.zip

# Infracost
RUN curl -fsSL https://raw.githubusercontent.com/infracost/infracost/master/scripts/install.sh | sh

# trufflehog
RUN curl -sSfL https://raw.githubusercontent.com/trufflesecurity/trufflehog/main/scripts/install.sh \
 | sh -s -- -b /usr/local/bin

WORKDIR /app

# --- dependency install (cache-friendly) ---
# Copy only reqs first so Docker can cache this layer if code changes
COPY requirements.txt /app/requirements.txt

# Remove hash checking (Windows hashes won't match Linux wheels)
# Also switch torch to a CPU-only wheel if present to avoid CUDA downloads
ARG TORCH_CPU=1
RUN set -eux; \
    # 1) kill hash enforcement safely (handles inline and separate hash lines)
    sed -i -E 's/--require-hashes//g' requirements.txt || true; \
    sed -i -E '/^\s*--hash=sha256:[a-f0-9]{64}\s*$/d' requirements.txt || true; \
    sed -i -E 's/\s*--hash=sha256:[a-f0-9]{64}//g' requirements.txt || true; \
    sed -i -E 's/\\$//g' requirements.txt || true; \
    sed -i -E 's/[[:space:]]+$//' requirements.txt || true; \
    # 2) make torch CPU-only (optional but recommended for API containers)
    if [ "$TORCH_CPU" = "1" ] && grep -qiE '^torch([<=>].*)?$' requirements.txt; then \
        # If torch is unpinned or pinned, rewrite to CPU wheel variant
        # Feel free to change version if you need a specific one
        sed -i -E 's/^torch([<>=].*)?$/torch==2.8.0+cpu/' requirements.txt; \
        PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu" pip install -r requirements.txt; \
    else \
        pip install -r requirements.txt; \
    fi

# --- app code ---
COPY . /app

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
