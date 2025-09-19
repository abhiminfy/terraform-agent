FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Make BOTH import styles work: "src.*" and "app.*"
    PYTHONPATH=/app:/app/src

# OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl gnupg unzip git bash build-essential \
 && rm -rf /var/lib/apt/lists/*

# Terraform CLI
RUN mkdir -p /etc/apt/keyrings \
 && curl -fsSL https://apt.releases.hashicorp.com/gpg | gpg --dearmor -o /etc/apt/keyrings/hashicorp.gpg \
 && echo "deb [signed-by=/etc/apt/keyrings/hashicorp.gpg] https://apt.releases.hashicorp.com $(. /etc/os-release; echo $VERSION_CODENAME) main" \
    > /etc/apt/sources.list.d/hashicorp.list \
 && apt-get update && apt-get install -y --no-install-recommends terraform \
 && rm -rf /var/lib/apt/lists/*

# Infracost CLI
RUN curl -fsSL https://raw.githubusercontent.com/infracost/infracost/master/scripts/install.sh \
 | sh -s -- -b /usr/local/bin

WORKDIR /app

# Install Python deps first for layer cache
COPY src/requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip wheel setuptools \
 && python -m pip install -r /app/requirements.txt

# App code
COPY src /app/src

# Non-root
RUN adduser --disabled-password --gecos "" app && chown -R app:app /app
USER app

EXPOSE 8000
# Command is supplied by docker-compose
