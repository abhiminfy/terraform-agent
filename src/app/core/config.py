# -*- coding: utf-8 -*-
from __future__ import annotations

from pydantic import Field, AliasChoices
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central app settings.
    - Loads from .env (UTF-8)
    - Ignores unknown env keys (prevents crashes if extra vars are present)
    - Uses explicit aliases so BOTH lower/UPPER env names work
    - Keeps runtime attribute compatibility via UPPERCASE properties
    """

    # Read from .env, respect exact case (we provide aliases), ignore extras
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,        # respect case; we rely on aliases below
        extra="ignore",             # <- prevents "extra inputs are not permitted"
    )

    # ----- Core app defaults -----
    APP_VERSION: str = "1.0.0"
    JWT_SECRET: str = "change-me"
    JWT_ALG: str = "HS256"

    # ----- Celery/Redis -----
    CELERY_BROKER_URL: str = "redis://redis:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://redis:6379/2"

    # ----- CLI binaries / misc -----
    TF_BIN: str = "terraform"
    INFRACOST_BIN: str = "infracost"
    TRUFFLEHOG_BIN: str = "trufflehog"
    GITHUB_DEFAULT_BRANCH: str = "main"

    # ----- Budgets / tokens -----
    TOKEN_BUDGET: int = 100_000

    # ----- Prometheus Pushgateway -----
    prom_pushgateway: str | None = Field(
        default=None,
        validation_alias=AliasChoices("prom_pushgateway", "PROM_PUSHGATEWAY"),
    )

    # ----- External creds/repos (accept lower/UPPER env names) -----
    # Example: `mongo_uri=...` or `MONGO_URI=...`
    mongo_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("mongo_uri", "MONGO_URI"),
    )
    infracost_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("infracost_api_key", "INFRACOST_API_KEY"),
    )
    github_username: str | None = Field(
        default=None,
        validation_alias=AliasChoices("github_username", "GITHUB_USERNAME"),
    )
    github_token: str | None = Field(
        default=None,
        validation_alias=AliasChoices("github_token", "GITHUB_TOKEN"),
    )
    github_repo: str | None = Field(
        default=None,
        validation_alias=AliasChoices("github_repo", "GITHUB_REPO"),
    )

    # ----- Rate limiting (env-overridable; both lower/UPPER accepted) -----
    # Matches your routes usage: @_limiter2.limit(_RT.RATE_LIMIT_CHAT)
    rate_limit_chat: str = Field(
        default="60/minute",
        validation_alias=AliasChoices("rate_limit_chat", "RATE_LIMIT_CHAT"),
    )
    rate_limit_secure: str = Field(
        default="30/minute",
        validation_alias=AliasChoices("rate_limit_secure", "RATE_LIMIT_SECURE"),
    )
    rate_limit_tf_to_pr: str = Field(
        default="10/minute",
        validation_alias=AliasChoices("rate_limit_tf_to_pr", "RATE_LIMIT_TF_TO_PR"),
    )

    # =========================
    # Back-compat runtime properties
    # (Expose UPPERCASE names your code references at runtime)
    # =========================

    # Rate limits
    @property
    def RATE_LIMIT_CHAT(self) -> str:        # used by routes decorators
        return self.rate_limit_chat

    @property
    def RATE_LIMIT_SECURE(self) -> str:
        return self.rate_limit_secure

    @property
    def RATE_LIMIT_TF_TO_PR(self) -> str:
        return self.rate_limit_tf_to_pr

    # Prometheus Pushgateway
    @property
    def PROM_PUSHGATEWAY(self) -> str | None:
        return self.prom_pushgateway

    # Optional convenience: expose UPPERCASE accessors if any code uses them
    @property
    def MONGO_URI(self) -> str | None:
        return self.mongo_url

    @property
    def INFRACOST_API_KEY(self) -> str | None:
        return self.infracost_api_key

    @property
    def GITHUB_USERNAME(self) -> str | None:
        return self.github_username

    @property
    def GITHUB_TOKEN(self) -> str | None:
        return self.github_token

    @property
    def GITHUB_REPO(self) -> str | None:
        return self.github_repo