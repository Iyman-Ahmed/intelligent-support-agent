"""
Application configuration — loaded from environment variables.
"""

from __future__ import annotations

import os
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # === Anthropic ===
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")

    # === App ===
    app_name: str = "Intelligent Customer Support Agent"
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    api_key: str = os.getenv("API_KEY", "dev-secret-key")   # For auth middleware
    webhook_secret: Optional[str] = os.getenv("WEBHOOK_SECRET")

    # === Redis ===
    redis_url: Optional[str] = os.getenv("REDIS_URL")

    # === Database ===
    database_url: Optional[str] = os.getenv("DATABASE_URL")

    # === Email ===
    sendgrid_api_key: Optional[str] = os.getenv("SENDGRID_API_KEY")
    mailgun_api_key: Optional[str] = os.getenv("MAILGUN_API_KEY")
    mailgun_domain: Optional[str] = os.getenv("MAILGUN_DOMAIN")
    support_from_email: str = os.getenv("SUPPORT_FROM_EMAIL", "support@yourdomain.com")
    support_from_name: str = os.getenv("SUPPORT_FROM_NAME", "Support Team")

    # === Ticketing ===
    zendesk_api_key: Optional[str] = os.getenv("ZENDESK_API_KEY")
    zendesk_subdomain: Optional[str] = os.getenv("ZENDESK_SUBDOMAIN")

    # === Vector DB ===
    pinecone_api_key: Optional[str] = os.getenv("PINECONE_API_KEY")
    pinecone_index: Optional[str] = os.getenv("PINECONE_INDEX", "support-kb")

    # === Monitoring ===
    otlp_endpoint: Optional[str] = os.getenv("OTLP_ENDPOINT")
    datadog_enabled: bool = os.getenv("DATADOG_ENABLED", "false").lower() == "true"

    # === CORS ===
    allowed_origins: list = ["*"]   # Restrict in production

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Singleton
settings = Settings()
