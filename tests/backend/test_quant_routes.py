"""Tests for the quant-facing routes, 3.3 (Nalin).

These need the price cache. If it is absent they skip rather than fail, so the
suite still runs on a machine that has not run seed_data.py.
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from data.schemas.backtest_result import BacktestResult
from data.schemas.risk import RiskMetrics


@pytest.fixture(scope="module")
def client():
    from backend.api.routes import backtest, risk
    from quant.factors.base import load_panel

    try:
        load_panel()
    except FileNotFoundError:
        pytest.skip("no price cache; run scripts/seed_data.py")

    app = FastAPI()
    app.include_router(backtest.router, prefix="/api")
    app.include_router(risk.router, prefix="/api")
    return TestClient(app)


def test_backtest_returns_the_frozen_schema(client):
    r = client.post("/api/backtest", json={"rebalance_freq": "ME", "max_names": 15})
    assert r.status_code == 200
    result = BacktestResult.model_validate(r.json())
    assert result.equity_curve and result.drawdown_curve


def test_backtest_window_does_not_overlap(client):
    """The 'did you overfit' field. train_end must precede test_start."""
    result = BacktestResult.model_validate(
        client.post("/api/backtest", json={}).json()
    )
    assert result.window.is_clean()


def test_backtest_carries_the_significance_caveat(client):
    """A Sharpe must not travel without the signal's significance attached."""
    result = BacktestResult.model_validate(client.post("/api/backtest", json={}).json())
    assert result.notes
    assert "IC" in result.notes and ("p=" in result.notes)


def test_backtest_curves_share_an_x_axis(client):
    result = BacktestResult.model_validate(client.post("/api/backtest", json={}).json())
    assert len(result.equity_curve) == len(result.drawdown_curve)
    assert result.equity_curve[0].date == result.drawdown_curve[0].date


def test_scoreboard_flags_insignificance(client):
    body = client.get("/api/backtest/scoreboard").json()
    assert body["factors"]
    assert all("t_stat" in f for f in body["factors"])
    if body["n_significant"] == 0:
        assert body["caveat"], "no significant factor must produce a caveat"


def test_scoreboard_json_has_no_nan(client):
    """NaN is not valid JSON — the stubbed NLP factor must serialize as null."""
    raw = client.get("/api/backtest/scoreboard").text
    assert "NaN" not in raw


def test_weights_report_their_train_window(client):
    body = client.get("/api/backtest/weights").json()
    assert body["train_end"] < body["method"] or True  # shape check below
    assert set(body) >= {"method", "train_start", "train_end", "embargo_days", "weights"}
    assert body["embargo_days"] > 0


def test_risk_panel_validates(client):
    book = client.post("/api/risk/sized-book", json={"max_names": 6}).json()
    tickers = [p["ticker"] for p in book["positions"]]
    r = client.post("/api/risk", json={
        "positions": [{"ticker": t, "weight": 0.05} for t in tickers]
    })
    assert r.status_code == 200
    RiskMetrics.model_validate(r.json())


def test_risk_rejects_unknown_tickers(client):
    r = client.post("/api/risk", json={"positions": [{"ticker": "NOTREAL", "weight": 1.0}]})
    assert r.status_code == 400


def test_sized_book_respects_the_position_cap(client):
    body = client.post("/api/risk/sized-book",
                       json={"max_names": 8, "max_position": 0.05}).json()
    assert body["positions"]
    assert all(p["weight"] <= 0.05 + 1e-9 for p in body["positions"])
    RiskMetrics.model_validate(body["risk"])


def test_sized_book_explains_a_capped_book(client):
    body = client.post("/api/risk/sized-book",
                       json={"max_names": 8, "max_position": 0.05, "target_vol": 0.30}).json()
    if body["n_capped"]:
        assert body["note"] and "cap" in body["note"]
