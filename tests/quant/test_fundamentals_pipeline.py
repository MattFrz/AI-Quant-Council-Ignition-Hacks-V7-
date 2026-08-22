"""Tests for the EDGAR companyfacts pipeline. Offline - no network."""
from __future__ import annotations

import pandas as pd
import pytest

from data.pipelines.fundamentals import _first_filed, _quarterly, _raw_facts


def _payload(tag, unit, facts):
    return {"facts": {"us-gaap": {tag: {"units": {unit: facts}}}}}


def test_raw_facts_reads_the_named_unit():
    payload = _payload("Revenues", "USD", [
        {"end": "2023-03-31", "start": "2023-01-01", "val": 100.0,
         "filed": "2023-05-01", "form": "10-Q"},
    ])
    got = _raw_facts(payload, ["Revenues"], "USD")
    assert len(got) == 1 and got.iloc[0]["val"] == 100.0
    assert _raw_facts(payload, ["Revenues"], "shares").empty


def test_raw_facts_falls_back_through_tag_candidates():
    payload = _payload("SalesRevenueNet", "USD", [
        {"end": "2023-03-31", "start": "2023-01-01", "val": 7.0,
         "filed": "2023-05-01", "form": "10-Q"}])
    got = _raw_facts(payload, ["RevenueFromContractWithCustomerExcludingAssessedTax",
                               "Revenues", "SalesRevenueNet"], "USD")
    assert len(got) == 1


def test_first_filed_keeps_the_original_not_the_restatement():
    """A later 10-K re-reports an old quarter. The market traded the first one."""
    frame = pd.DataFrame({
        "end": pd.to_datetime(["2022-03-31", "2022-03-31"]),
        "val": [100.0, 999.0],
        "filed": pd.to_datetime(["2022-05-02", "2024-02-01"]),
        "form": ["10-Q", "10-K"],
    })
    kept = _first_filed(frame)
    assert len(kept) == 1
    assert kept.iloc[0]["val"] == 100.0
    assert kept.iloc[0]["filed"] == pd.Timestamp("2022-05-02")


def test_q4_is_derived_from_the_annual_figure():
    """Issuers fold Q4 into the 10-K. Without deriving it, 'four quarters ago'
    silently becomes five."""
    rows = [
        {"start": "2022-01-01", "end": "2022-03-31", "val": 10.0, "filed": "2022-05-01", "form": "10-Q"},
        {"start": "2022-04-01", "end": "2022-06-30", "val": 12.0, "filed": "2022-08-01", "form": "10-Q"},
        {"start": "2022-07-01", "end": "2022-09-30", "val": 14.0, "filed": "2022-11-01", "form": "10-Q"},
        {"start": "2022-01-01", "end": "2022-12-31", "val": 50.0, "filed": "2023-02-01", "form": "10-K"},
    ]
    frame = _raw_facts(_payload("Revenues", "USD", rows), ["Revenues"], "USD")
    out = _quarterly(frame, "duration")

    assert len(out) == 4
    q4 = out[out["end"] == pd.Timestamp("2022-12-31")].iloc[0]
    assert q4["val"] == pytest.approx(50.0 - (10.0 + 12.0 + 14.0))
    assert q4["filed"] == pd.Timestamp("2023-02-01")


def test_q4_is_not_invented_when_quarters_are_missing():
    rows = [
        {"start": "2022-01-01", "end": "2022-03-31", "val": 10.0, "filed": "2022-05-01", "form": "10-Q"},
        {"start": "2022-01-01", "end": "2022-12-31", "val": 50.0, "filed": "2023-02-01", "form": "10-K"},
    ]
    frame = _raw_facts(_payload("Revenues", "USD", rows), ["Revenues"], "USD")
    out = _quarterly(frame, "duration")
    assert list(out["end"]) == [pd.Timestamp("2022-03-31")]


def test_annual_rows_are_not_mistaken_for_quarters():
    rows = [{"start": "2022-01-01", "end": "2022-12-31", "val": 50.0,
             "filed": "2023-02-01", "form": "10-K"}]
    frame = _raw_facts(_payload("Revenues", "USD", rows), ["Revenues"], "USD")
    assert _quarterly(frame, "duration").empty


# ---- against the real cache, when it exists -------------------------------

@pytest.fixture(scope="module")
def cached():
    from data.pipelines.fundamentals import load_fundamentals
    try:
        return load_fundamentals()
    except (FileNotFoundError, OSError):
        pytest.skip("no EDGAR cache")


def test_cache_never_reports_before_the_period_ended(cached):
    assert (cached["report_date"] >= cached["period_end"]).all()


def test_cache_reporting_lag_is_realistic(cached):
    lag = (cached["report_date"] - cached["period_end"]).dt.days
    assert 15 <= lag.median() <= 75


def test_cache_has_no_duplicate_periods(cached):
    assert not cached.duplicated(subset=["ticker", "period_end"]).any()


def test_panel_accepts_the_cache(cached):
    from quant.factors.base import Panel, load_panel
    panel = load_panel()
    assert panel.fundamentals is not None
    from quant.factors.fundamental import RevenueGrowth
    out = RevenueGrowth().compute(panel, panel.dates[-1])
    assert out.notna().sum() > 0
