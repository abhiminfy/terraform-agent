# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central app settings.

    - Loads from .env
    - Case-insensitive env keys via explicit aliases
    - Extra env vars are ignored (no crashes)
    - Provides UPPERCASE property aliases for backward-compat
    """

    # --- Models & APIs ---
    gemini_api_key: Optional[str] = Field(
        default=None, validation_alias=AliasChoices("GEMINI_API_KEY", "gemini_api_key")
    )
    gemini_model: str = Field(
        default="gemini-1.5-pro", validation_alias=AliasChoices("GEMINI_MODEL", "gemini_model")
    )
    hf_model: str = Field(
        default="distilgpt2", validation_alias=AliasChoices("HF_MODEL", "hf_model")
    )

    # --- Costs/Infra ---
    infracost_api_key: Optional[str] = Field(
        default=None, validation_alias=AliasChoices("INFRACOST_API_KEY", "infracost_api_key")
    )

    # --- GitHub ---
    github_username: Optional[str] = Field(
        default=None, validation_alias=AliasChoices("GITHUB_USERNAME", "github_username")
    )
    github_token: Optional[str] = Field(
        default=None, validation_alias=AliasChoices("GITHUB_TOKEN", "github_token")
    )
    github_repo: Optional[str] = Field(
        default=None, validation_alias=AliasChoices("GITHUB_REPO", "github_repo")
    )

    # --- Metrics ---
    prometheus_pushgateway: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("PROMETHEUS_PUSHGATEWAY", "prometheus_pushgateway"),
    )

    # --- Security & Rate Limiting ---
    jwt_secret: str = Field(
        default="your-super-secret-jwt-key-change-this-in-production",
        validation_alias=AliasChoices("JWT_SECRET", "jwt_secret"),
    )
    jwt_algorithm: str = Field(
        default="HS256",
        validation_alias=AliasChoices("JWT_ALG", "JWT_ALGORITHM", "jwt_algorithm", "jwt_alg"),
    )
    rate_limit_chat: str = Field(
        default="10/minute", validation_alias=AliasChoices("RATE_LIMIT_CHAT", "rate_limit_chat")
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- UPPERCASE aliases (back-compat) ----
    @property
    def GEMINI_API_KEY(self) -> Optional[str]:
        return self.gemini_api_key

    @property
    def GEMINI_MODEL(self) -> str:
        return self.gemini_model

    @property
    def HF_MODEL(self) -> str:
        return self.hf_model

    @property
    def INFRACOST_API_KEY(self) -> Optional[str]:
        return self.infracost_api_key

    @property
    def GITHUB_USERNAME(self) -> Optional[str]:
        return self.github_username

    @property
    def GITHUB_TOKEN(self) -> Optional[str]:
        return self.github_token

    @property
    def GITHUB_REPO(self) -> Optional[str]:
        return self.github_repo

    @property
    def PROMETHEUS_PUSHGATEWAY(self) -> Optional[str]:
        return self.prometheus_pushgateway

    @property
    def JWT_SECRET(self) -> str:
        return self.jwt_secret

    @property
    def JWT_ALG(self) -> str:
        return self.jwt_algorithm

    @property
    def RATE_LIMIT_CHAT(self) -> str:
        return self.rate_limit_chat
