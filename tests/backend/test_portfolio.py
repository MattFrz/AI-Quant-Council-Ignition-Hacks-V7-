"""D14 smoke test - build a portfolio from the Phase 1 fixture and a couple
of synthetic variants, confirm the section 13 pipeline runs end-to-end and
produces an explainable result. Run with: pytest tests/backend/test_portfolio.py
"""
from __future__ import annotations

import json
from pathlib import Path

from backend.portfolio.construction import build_portfolio
from data.schemas.trade_idea import TradeIdea

FIXTURE = Path(__file__).resolve().parents[2] / "data" / "fixtures" / "sample_trade_idea.json"


def _load_fixture() -> TradeIdea:
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    raw.pop("_README", None)
    return TradeIdea.model_validate(raw)


def test_build_portfolio_from_fixture():
    idea = _load_fixture()
    portfolio = build_portfolio([idea])

    assert portfolio.positions, "the Phase 1 fixture should survive constraints and get sized"
    pos = portfolio.positions[0]
    assert pos.ticker == idea.ticker
    assert 0 < pos.position_size_pct <= 5.0  # settings.max_position_pct default
    assert portfolio.total_invested_pct == pos.position_size_pct
    assert not portfolio.excluded


def test_rejected_verdict_is_excluded():
    idea = _load_fixture()
    idea.validator_verdict = "rejected"

    portfolio = build_portfolio([idea])

    assert not portfolio.positions
    assert portfolio.excluded[0].ticker == idea.ticker
    assert any("rejected" in r for r in portfolio.excluded[0].reasons)


def test_low_confidence_is_excluded():
    idea = _load_fixture()
    idea.confidence = 0.30

    portfolio = build_portfolio([idea])

    assert not portfolio.positions
    assert any("confidence" in r for r in portfolio.excluded[0].reasons)


def test_position_size_respects_custom_cap():
    idea = _load_fixture()
    portfolio = build_portfolio([idea], max_position_pct=2.0)

    assert portfolio.positions[0].position_size_pct <= 2.0
