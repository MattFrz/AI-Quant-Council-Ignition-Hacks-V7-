"""Single source of configuration for the whole backend.

Every module imports `settings` from here. Do not call os.getenv anywhere else -
scattered env reads are how the four lanes end up disagreeing about paths.
"""
from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- LLM ---------------------------------------------------------------
    llm_provider: str = "openai"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    llm_budget_usd: float = 40.0

    # ---- SEC ---------------------------------------------------------------
    sec_user_agent: str = ""

    # ---- Universe ----------------------------------------------------------
    universe_size: int = 1000
    min_market_cap: float = 1_000_000_000
    min_adv_usd: float = 5_000_000

    # ---- Paths -------------------------------------------------------------
    data_cache_dir: str = "data/cache"
    fixtures_dir: str = "data/fixtures"

    # ---- Backtest ----------------------------------------------------------
    backtest_start: date = date(2015, 1, 1)
    backtest_end: date = date(2024, 12, 31)
    benchmark_ticker: str = "SPY"
    commission_bps: float = 1.0
    max_position_pct: float = 5.0
    max_adv_participation: float = 0.05

    # ---- Runtime -----------------------------------------------------------
    log_level: str = "INFO"
    offline_mode: bool = False
    frontend_origin: str = "http://localhost:3000"
    # Comma-separated extra origins allowed to call this API - e.g. the live
    # Base44 app URL. Kept separate from frontend_origin so nothing that
    # already reads that field breaks. Set in .env: EXTRA_CORS_ORIGINS=...
    extra_cors_origins: str = ""

    @property
    def cors_origins(self) -> list[str]:
        origins = [self.frontend_origin]
        origins += [o.strip() for o in self.extra_cors_origins.split(",") if o.strip()]
        return origins

    # ---- Derived paths -----------------------------------------------------
    @property
    def root(self) -> Path:
        return PROJECT_ROOT

    @property
    def cache_path(self) -> Path:
        p = PROJECT_ROOT / self.data_cache_dir
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def fixtures_path(self) -> Path:
        p = PROJECT_ROOT / self.fixtures_dir
        p.mkdir(parents=True, exist_ok=True)
        return p

    def require_llm_key(self) -> str:
        if not self.llm_api_key or self.llm_api_key.startswith("sk-replace"):
            raise RuntimeError(
                "LLM_API_KEY is not set. Copy .env.example to .env and fill it in."
            )
        return self.llm_api_key

    def require_sec_user_agent(self) -> str:
        if not self.sec_user_agent or "@" not in self.sec_user_agent:
            raise RuntimeError(
                "SEC_USER_AGENT must contain a real name and email or EDGAR will "
                "reject every request. Set it in .env."
            )
        return self.sec_user_agent


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings: Settings = get_settings()
