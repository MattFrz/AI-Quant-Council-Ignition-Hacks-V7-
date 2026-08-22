"""FastAPI entrypoint.

Run with:  uvicorn backend.main:app --reload

Routers are mounted in Phase 3 as each one starts returning real data. Until
then this serves /health and /api/fixture so the frontend has something live to
talk to.
"""
from __future__ import annotations

import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.core.logging import get_logger

log = get_logger(__name__)

app = FastAPI(
    title="Autonomous Alpha",
    description="Autonomous quant research platform - IgnitionHacks V7",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "offline_mode": settings.offline_mode,
        "model": settings.llm_model,
        "universe_size": settings.universe_size,
    }


@app.get("/api/fixture")
def fixture() -> dict:
    """Serve the Phase 1 sample TradeIdea so the frontend has a live endpoint
    returning the exact contract shape before any real pipeline exists."""
    path = settings.fixtures_path / "sample_trade_idea.json"
    if not path.exists():
        raise HTTPException(404, f"fixture not found at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


# --- Phase 3: uncomment as each router starts returning real data ------------
# from backend.api.routes import thesis, scan, research, backtest, risk, portfolio, stream
# app.include_router(thesis.router,    prefix="/api")
# app.include_router(scan.router,      prefix="/api")
# app.include_router(research.router,  prefix="/api")
# app.include_router(backtest.router,  prefix="/api")
# app.include_router(risk.router,      prefix="/api")
# app.include_router(portfolio.router, prefix="/api")
# app.include_router(stream.router,    prefix="/api")


@app.on_event("startup")
def _startup() -> None:
    log.info("Autonomous Alpha backend up - model=%s offline=%s",
             settings.llm_model, settings.offline_mode)
