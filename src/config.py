"""
src/config.py
─────────────────────────────────────────────────────────────
Centralised configuration loaded from environment variables.
Uses pydantic-settings so every field is validated on startup.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── watsonx credentials ───────────────────────────────────
    watsonx_apikey: str = Field(..., description="IBM Cloud API key")
    watsonx_url: str = Field(
        "https://us-south.ml.cloud.ibm.com",
        description="watsonx.ai endpoint",
    )
    watsonx_project_id: str = Field(..., description="watsonx project GUID")

    # ── Granite model IDs ─────────────────────────────────────
    granite_llm_model: str = Field(
        "ibm/granite-3-8b-instruct",
        description="Text generation model",
    )
    granite_vision_model: str = Field(
        "ibm/granite-vision-3-2-2b",
        description="Vision / multimodal model",
    )
    granite_guardian_model: str = Field(
        "ibm/granite-guardian-3-8b",
        description="Safety guardrail model",
    )
    granite_embedding_model: str = Field(
        "ibm/granite-embedding-278m-multilingual",
        description="Embedding model for RAG",
    )

    # ── Vector store ──────────────────────────────────────────
    chroma_persist_dir: str = Field(
        "./data/chroma_db",
        description="ChromaDB persistence directory",
    )
    chroma_collection_foods: str = "usda_foods"
    chroma_collection_guidelines: str = "disease_guidelines"

    # ── Speech-to-Text ────────────────────────────────────────
    whisper_model_size: Literal["tiny", "base", "small", "medium", "large"] = "base"
    watson_stt_apikey: str | None = None
    watson_stt_url: str | None = None

    # ── Application ───────────────────────────────────────────
    app_env: Literal["development", "staging", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    @field_validator("watsonx_url")
    @classmethod
    def _strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")

    # ── LLM generation defaults ───────────────────────────────
    @property
    def llm_params(self) -> dict:
        return {
            "decoding_method": "greedy",
            "max_new_tokens": 2048,
            "min_new_tokens": 1,
            "temperature": 0.7,
            "repetition_penalty": 1.1,
        }

    @property
    def embedding_params(self) -> dict:
        return {"truncate_input_tokens": 512}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
