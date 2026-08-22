"""Tests for the Lane B foundation, B1-B5.

The one that matters most is `test_factor_cannot_see_the_future`. Leakage does
not raise an exception — it just makes every number better than it should be,
right up until someone asks how it was handled. So it gets tested directly:
compute a factor at date t, append the future to the panel, compute again, and
demand the two are identical.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.alpha.statistical_tests import (
    cross_sectional_ic,
    factor_scoreboard,
    information_coefficient,
    signal_decay,
    summarize_ic,
)
from quant.factors.base import Factor, Panel
from quant.factors.market import (
    Momentum12_1,
    RealizedVolatility,
    RelativeStrength,
    VolumeTrend,
    default_market_factors,
)
from quant.signals.cross_sectional import (
    demean_by_group,
    factor_panel,
    forward_returns,
    normalize_panel,
    rebalance_dates,
)
from quant.signals.normalization import normalize, rank_transform, winsorize, zscore


# ---------------------------------------------------------------- B1: Panel

def test_panel_rejects_unsorted_dates():
    dates = pd.to_datetime(["2024-01-03", "2024-01-02"])
    df = pd.DataFrame(1.0, index=dates, columns=["A"])
    with pytest.raises(ValueError, match="sorted ascending"):
        Panel(adj_close=df, volume=df)


def test_panel_rejects_misaligned_frames():
    dates = pd.bdate_range("2024-01-02", periods=3)
    px = pd.DataFrame(1.0, index=dates, columns=["A", "B"])
    vol = pd.DataFrame(1.0, index=dates, columns=["A"])
    with pytest.raises(ValueError, match="ticker column set"):
        Panel(adj_close=px, volume=vol)


def test_as_of_truncates_at_the_decision_date(synthetic_panel):
    cut = synthetic_panel.dates[500]
    window = synthetic_panel.as_of(cut, lag_days=0)
    assert window.dates[-1] == cut
    assert window.n_days == 501


def test_as_of_lag_is_in_trading_days(synthetic_panel):
    cut = synthetic_panel.dates[500]
    window = synthetic_panel.as_of(cut, lag_days=3)
    assert window.dates[-1] == synthetic_panel.dates[497]


def test_as_of_before_history_returns_empty(synthetic_panel):
    window = synthetic_panel.as_of("1990-01-01")
    assert window.n_days == 0


# ------------------------------------------------- B1: the leakage guarantee

def test_factor_cannot_see_the_future(synthetic_panel):
    """A signal computed at t must not change when the future is appended."""
    as_of = synthetic_panel.dates[900]
    truncated = synthetic_panel.as_of(as_of)

    for factor in default_market_factors():
        early = factor.compute(truncated, as_of)
        late = factor.compute(synthetic_panel, as_of)
        pd.testing.assert_series_equal(early, late)


def test_min_lag_excludes_the_decision_day_bar(synthetic_panel):
    """With min_lag_days=1 the signal for date t uses data through t-1 only."""
    as_of = synthetic_panel.dates[800]
    factor = RealizedVolatility(window=20, min_lag_days=1)

    tampered = synthetic_panel.adj_close.copy()
    tampered.loc[as_of] = tampered.loc[as_of] * 5.0  # a violent move ON the decision day
    poisoned = Panel(
        adj_close=tampered,
        volume=synthetic_panel.volume,
        close=synthetic_panel.close,
        securities=synthetic_panel.securities,
        universe=synthetic_panel.universe,
    )

    pd.testing.assert_series_equal(
        factor.compute(synthetic_panel, as_of),
        factor.compute(poisoned, as_of),
    )


# --------------------------------------------------------------- B2: factors

def test_momentum_measures_the_right_window():
    dates = pd.bdate_range("2020-01-01", periods=300)
    ramp = pd.DataFrame({"A": np.linspace(100.0, 400.0, 300)}, index=dates)
    panel = Panel(adj_close=ramp, volume=pd.DataFrame(1e6, index=dates, columns=["A"]))

    factor = Momentum12_1(lookback=252, skip=21, min_lag_days=0)
    got = factor.compute(panel, dates[-1])["A"]

    px = ramp["A"].values
    expected = px[-1 - 21] / px[-1 - 252] - 1.0
    assert got == pytest.approx(expected)


def test_realized_vol_is_annualized():
    dates = pd.bdate_range("2020-01-01", periods=120)
    rng = np.random.default_rng(3)
    daily = 0.02
    px = pd.DataFrame({"A": 100 * np.exp(np.cumsum(rng.normal(0, daily, 120)))}, index=dates)
    panel = Panel(adj_close=px, volume=pd.DataFrame(1e6, index=dates, columns=["A"]))

    got = RealizedVolatility(window=100, min_lag_days=0).compute(panel, dates[-1])["A"]
    assert got == pytest.approx(daily * np.sqrt(252), rel=0.35)


def test_factors_return_one_value_per_ticker(synthetic_panel):
    as_of = synthetic_panel.dates[1000]
    for factor in default_market_factors():
        out = factor.compute(synthetic_panel, as_of)
        assert list(out.index) == list(synthetic_panel.tickers)
        assert out.notna().sum() > 0.9 * synthetic_panel.n_tickers
        assert out.name == factor.name


def test_factor_returns_nan_without_enough_history(synthetic_panel):
    early = synthetic_panel.dates[10]
    out = Momentum12_1().compute(synthetic_panel, early)
    assert out.isna().all()


# --------------------------------------------------- B3: normalization

def test_zscore_standardizes():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    z = zscore(s)
    assert z.mean() == pytest.approx(0.0, abs=1e-12)
    assert z.std(ddof=1) == pytest.approx(1.0)


def test_zscore_preserves_nan_and_index():
    s = pd.Series([1.0, np.nan, 3.0, 4.0, 5.0, 6.0], index=list("abcdef"))
    z = zscore(s)
    assert z.isna().loc["b"]
    assert list(z.index) == list(s.index)


def test_zscore_refuses_degenerate_cross_section():
    assert zscore(pd.Series([2.0] * 10)).isna().all()   # no dispersion
    assert zscore(pd.Series([1.0, 2.0])).isna().all()   # too few names


def test_winsorize_clips_the_outlier():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 1000.0])
    assert winsorize(s, 0.0, 0.75).max() < 1000.0


def test_winsorize_before_zscore_protects_the_rest():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 500.0])
    raw = zscore(s).abs().iloc[:5].max()
    clipped = normalize(s, winsor=(0.0, 0.8)).abs().iloc[:5].max()
    assert clipped > raw  # the good names get real dispersion back


def test_rank_transform_is_monotonic():
    s = pd.Series([10.0, 30.0, 20.0])
    assert list(rank_transform(s).sort_values().index) == [0, 2, 1]


# ------------------------------------------------ B4: cross-sectional panels

def test_factor_panel_shape(synthetic_panel):
    dates = rebalance_dates(synthetic_panel, warmup=300)[:12]
    raw = factor_panel(Momentum12_1(), synthetic_panel, dates)
    assert raw.shape == (12, synthetic_panel.n_tickers)


def test_normalize_panel_is_per_date(synthetic_panel):
    dates = rebalance_dates(synthetic_panel, warmup=300)[:6]
    scores = normalize_panel(factor_panel(Momentum12_1(), synthetic_panel, dates))
    row_means = scores.mean(axis=1)
    assert np.allclose(row_means.values, 0.0, atol=1e-9)


def test_demean_by_group_removes_sector_means(synthetic_panel):
    dates = rebalance_dates(synthetic_panel, warmup=300)[:4]
    raw = factor_panel(Momentum12_1(), synthetic_panel, dates)
    neutral = demean_by_group(raw, synthetic_panel.securities["sector"])
    sector_means = neutral.T.groupby(synthetic_panel.securities["sector"]).mean()
    assert np.allclose(sector_means.fillna(0.0).values, 0.0, atol=1e-9)


def test_forward_returns_look_forward_not_back(synthetic_panel):
    dates = synthetic_panel.dates
    fwd = forward_returns(synthetic_panel, horizon=5)
    px = synthetic_panel.adj_close
    expected = px.iloc[105]["SYN000"] / px.iloc[100]["SYN000"] - 1.0
    assert fwd.iloc[100]["SYN000"] == pytest.approx(expected)
    assert fwd.iloc[-1].isna().all()  # nothing after the last bar


# ------------------------------------------------------- B5: the scoreboard

def test_ic_recovers_the_designed_correlation(designed_ic):
    """The arithmetic check: a factor built to have IC 0.05 must score 0.05."""
    factor_df, returns_df = designed_ic
    ic = cross_sectional_ic(factor_df, returns_df, method="pearson")
    summary = summarize_ic(ic, factor="designed", horizon=21, method="pearson")

    # The estimate is a sample mean, so judge it against its own standard
    # error rather than a tolerance picked to make one seed pass.
    standard_error = summary.std_ic / np.sqrt(summary.n_periods)
    assert abs(summary.mean_ic - 0.05) < 3 * standard_error
    assert summary.n_periods == len(factor_df)
    assert summary.is_significant()


def test_ic_of_pure_noise_is_not_significant():
    factor_df, _ = __import__("tests.conftest", fromlist=["x"]).make_factor_and_returns(ic=0.0, seed=99)
    _, returns_df = __import__("tests.conftest", fromlist=["x"]).make_factor_and_returns(ic=0.0, seed=1234)
    ic = cross_sectional_ic(factor_df, returns_df)
    summary = summarize_ic(ic, factor="noise", horizon=21)
    assert abs(summary.mean_ic) < 0.02
    assert not summary.is_significant()


def test_momentum_recovers_its_embedded_edge(synthetic_panel, synthetic_truth):
    """The end-to-end check: Panel -> Factor -> normalize -> IC on a market
    where momentum was deliberately built into the return process."""
    dates = rebalance_dates(synthetic_panel, warmup=synthetic_truth["first_signal_row"])
    scores = normalize_panel(factor_panel(Momentum12_1(), synthetic_panel, dates))
    summary = information_coefficient(scores, synthetic_panel, horizon=21,
                                      method="spearman", name="momentum_12_1")

    assert summary.mean_ic > 0.0
    assert summary.t_stat > 2.0
    assert summary.hit_rate > 0.5
    assert not summary.overlapping   # monthly dates, 21-day horizon


def test_scoreboard_ranks_every_factor(synthetic_panel):
    board = factor_scoreboard(default_market_factors(), synthetic_panel, horizon=21)
    assert len(board) == 4
    assert set(board.columns) >= {"factor", "mean_ic", "t_stat", "p_value", "hit_rate"}
    abs_ic = board["mean_ic"].abs()
    assert abs_ic.is_monotonic_decreasing


def test_signal_decay_covers_every_horizon(synthetic_panel, synthetic_truth):
    dates = rebalance_dates(synthetic_panel, warmup=synthetic_truth["first_signal_row"])
    scores = normalize_panel(factor_panel(Momentum12_1(), synthetic_panel, dates))
    decay = signal_decay(scores, synthetic_panel, horizons=(1, 5, 21, 63))
    assert list(decay.index) == [1, 5, 21, 63]
    assert decay["mean_ic"].notna().all()


def test_overlapping_windows_are_flagged(synthetic_panel, synthetic_truth):
    """Weekly dates with a 63-day horizon overlap — the t-stat is optimistic
    and the summary has to say so rather than quietly reporting it."""
    dates = rebalance_dates(synthetic_panel, freq="W-FRI",
                            warmup=synthetic_truth["first_signal_row"])
    scores = normalize_panel(factor_panel(Momentum12_1(), synthetic_panel, dates))
    summary = information_coefficient(scores, synthetic_panel, horizon=63, method="spearman")
    assert summary.overlapping


# ==========================================================================
# B8 - B13: the alpha model
# ==========================================================================

from datetime import date as _Date  # noqa: E402

from data.schemas.catalyst import Catalyst, Direction, SourceType  # noqa: E402
from data.schemas.signal import AlphaBreakdown, FactorCategory  # noqa: E402
from quant.alpha.composite import CompositeModel, equal_weight_model  # noqa: E402
from quant.alpha.weighting import (  # noqa: E402
    fit_ic_weights,
    fit_ridge_weights,
    split_train_test,
)
from quant.eventstudy.study import (  # noqa: E402
    event_study,
    events_from_catalysts,
)
from quant.factors.nlp import CatalystSentiment  # noqa: E402
from quant.optimization.vol_scaling import size_positions  # noqa: E402


@pytest.fixture(scope="module")
def panels(synthetic_panel, synthetic_truth):
    """Normalized factor panels for the four market factors."""
    from quant.signals.cross_sectional import build_factor_panels
    dates = rebalance_dates(synthetic_panel, warmup=synthetic_truth["first_signal_row"])
    return build_factor_panels(default_market_factors(), synthetic_panel, dates)


# ---------------------------------------------------------------- B9

def test_contributions_sum_to_composite_exactly(panels, synthetic_panel):
    """AlphaBreakdown.check_sums() is a gate in verify_contract.py."""
    model = equal_weight_model(panels)
    as_of = list(panels.values())[0].index[-1]
    ticker = synthetic_panel.tickers[0]

    breakdown = model.breakdown(panels, ticker, as_of)
    assert isinstance(breakdown, AlphaBreakdown)
    assert breakdown.check_sums(1e-9)


def test_every_breakdown_on_a_date_validates(panels):
    model = equal_weight_model(panels)
    as_of = list(panels.values())[0].index[-1]
    ideas = model.rank_date(panels, as_of, top_n=15)

    assert len(ideas) == 15
    assert all(b.check_sums(1e-9) for b in ideas)
    assert all(0.0 <= b.alpha_score <= 10.0 for b in ideas)
    # rank_date returns best-first
    assert ideas[0].composite_alpha >= ideas[-1].composite_alpha


def test_missing_factor_rescales_instead_of_diluting(panels):
    """A NaN factor must not drag a name's score toward zero."""
    model = equal_weight_model(panels)
    as_of = list(panels.values())[0].index[-1]
    ticker = panels[list(panels)[0]].loc[as_of].dropna().index[0]

    full = model.breakdown(panels, ticker, as_of)

    holed = {k: v.copy() for k, v in panels.items()}
    dropped = list(holed)[0]
    holed[dropped].loc[as_of, ticker] = np.nan
    partial = model.breakdown(holed, ticker, as_of)

    assert len(partial.contributions) == len(full.contributions) - 1
    assert partial.check_sums(1e-9)
    # surviving weights were scaled back up to full strength
    assert sum(abs(c.weight) for c in partial.contributions) == pytest.approx(
        sum(abs(c.weight) for c in full.contributions), rel=1e-9
    )


def test_composite_score_panel_matches_breakdown(panels):
    model = equal_weight_model(panels)
    as_of = list(panels.values())[0].index[-1]
    scores = model.score_panel(panels)
    ticker = scores.loc[as_of].dropna().index[0]
    assert model.breakdown(panels, ticker, as_of).composite_alpha == pytest.approx(
        scores.at[as_of, ticker], rel=1e-9
    )


def test_composite_rejects_unsupplied_factor(panels):
    model = CompositeModel(weights={"nonexistent_factor": 1.0})
    with pytest.raises(KeyError, match="nonexistent_factor"):
        model.score_panel(panels)


# ---------------------------------------------------------------- B10

def test_train_test_split_leaves_an_embargo_gap(panels):
    dates = list(panels.values())[0].index
    train, test = split_train_test(dates, train_frac=0.6, embargo=21)

    assert train[-1] < test[0]
    assert (test[0] - train[-1]).days >= 21


def test_weights_are_fitted_on_train_only(panels, synthetic_panel):
    """Mutate everything after the train window's reach; weights must not move."""
    dates = list(panels.values())[0].index
    train, test = split_train_test(dates, train_frac=0.6, embargo=21)

    before = fit_ic_weights(panels, synthetic_panel, train, horizon=21)

    last_train_pos = int(synthetic_panel.dates.searchsorted(train[-1]))
    poison_from = last_train_pos + 21 + 1
    tampered = synthetic_panel.adj_close.copy()
    tampered.iloc[poison_from:] *= 3.0
    poisoned = Panel(
        adj_close=tampered,
        volume=synthetic_panel.volume,
        close=synthetic_panel.close,
        securities=synthetic_panel.securities,
        universe=synthetic_panel.universe,
    )

    after = fit_ic_weights(panels, poisoned, train, horizon=21)
    assert before.weights == pytest.approx(after.weights)


def test_ic_weighting_prefers_the_factor_with_the_real_edge(panels, synthetic_panel):
    dates = list(panels.values())[0].index
    train, _ = split_train_test(dates, train_frac=0.7, embargo=21)
    fitted = fit_ic_weights(panels, synthetic_panel, train, horizon=21)

    best = max(fitted.weights, key=lambda k: fitted.weights[k])
    assert best == "momentum_12_1"
    assert sum(abs(w) for w in fitted.weights.values()) == pytest.approx(1.0)


def test_min_abs_t_zeroes_out_factors_that_did_not_earn_their_place(panels, synthetic_panel):
    dates = list(panels.values())[0].index
    train, _ = split_train_test(dates, train_frac=0.7, embargo=21)
    strict = fit_ic_weights(panels, synthetic_panel, train, horizon=21, min_abs_t=1.5)
    assert any(w == 0.0 for w in strict.weights.values())


def test_ridge_fit_produces_usable_weights(panels, synthetic_panel):
    dates = list(panels.values())[0].index
    train, _ = split_train_test(dates, train_frac=0.7, embargo=21)
    fitted = fit_ridge_weights(panels, synthetic_panel, train, horizon=21)

    assert set(fitted.weights) == set(panels)
    assert sum(abs(w) for w in fitted.weights.values()) == pytest.approx(1.0)
    assert fitted.to_model().score_panel(panels).notna().any().any()


# ---------------------------------------------------------------- B8

def test_nlp_factor_is_stubbed_until_catalysts_arrive(synthetic_panel):
    factor = CatalystSentiment()
    assert factor.is_stubbed
    out = factor.compute(synthetic_panel, synthetic_panel.dates[-1])
    assert out.isna().all()   # absence of evidence, not a measured zero


def test_stubbed_nlp_factor_does_not_break_the_composite(panels, synthetic_panel):
    dates = list(panels.values())[0].index
    stub = CatalystSentiment()
    with_nlp = dict(panels)
    with_nlp[stub.name] = factor_panel(stub, synthetic_panel, dates)

    model = equal_weight_model(with_nlp)
    scores = model.score_panel(with_nlp)
    assert scores.notna().any().any()

    as_of = dates[-1]
    ticker = scores.loc[as_of].dropna().index[0]
    breakdown = model.breakdown(with_nlp, ticker, as_of)
    assert breakdown.check_sums(1e-9)
    assert all(c.factor != stub.name for c in breakdown.contributions)


def test_nlp_factor_respects_the_as_of_rule(synthetic_panel):
    ticker = synthetic_panel.tickers[0]
    published = synthetic_panel.dates[-10]
    catalyst = Catalyst(
        catalyst_id="c1", ticker=ticker, headline="Guidance raised",
        quote="We are raising full-year capex guidance.",
        source_type=SourceType.SEC_FILING,
        source_url="https://www.sec.gov/example",
        source_date=published.date(), direction=Direction.BULLISH, confidence=0.9,
    )
    factor = CatalystSentiment(catalysts=[catalyst])

    # Long before it was published: invisible.
    assert np.isnan(factor.compute(synthetic_panel, synthetic_panel.dates[100])[ticker])

    # ON the publication date it is still invisible, because min_lag_days=1
    # means the signal for date t is built from evidence available through t-1.
    assert np.isnan(factor.compute(synthetic_panel, published)[ticker])

    # The next trading day it counts.
    assert factor.compute(synthetic_panel, synthetic_panel.dates[-9])[ticker] > 0
    assert factor.compute(synthetic_panel, synthetic_panel.dates[-1])[ticker] > 0


def test_nlp_sentiment_decays_with_age(synthetic_panel):
    ticker = synthetic_panel.tickers[0]
    published = synthetic_panel.dates[-200]
    catalyst = Catalyst(
        catalyst_id="c3", ticker=ticker, headline="Capex raised", quote="q",
        source_type=SourceType.SEC_FILING, source_url="https://www.sec.gov/e",
        source_date=published.date(), direction=Direction.BULLISH, confidence=0.9,
    )
    factor = CatalystSentiment(catalysts=[catalyst], half_life_days=45)

    fresh = factor.compute(synthetic_panel, synthetic_panel.dates[-199])[ticker]
    stale = factor.compute(synthetic_panel, synthetic_panel.dates[-60])[ticker]
    assert 0 < stale < fresh


def test_events_from_catalysts_uses_source_date(synthetic_panel):
    c = Catalyst(
        catalyst_id="c2", ticker="SYN000", headline="h", quote="q",
        source_type=SourceType.NEWS, source_url="https://example.com/x",
        source_date=_Date(2022, 6, 15), event_date=_Date(2022, 5, 1),
        direction=Direction.BULLISH, confidence=0.5,
    )
    frame = events_from_catalysts([c])
    assert frame.loc[0, "event_date"] == pd.Timestamp("2022-06-15")


# ---------------------------------------------------------------- B11

def _panel_with_injected_events(n_events=40, bump=0.004, hold=10, seed=5):
    """A market where a known abnormal drift follows each event date."""
    rng = np.random.default_rng(seed)
    n_days, n_tickers = 900, 30
    dates = pd.bdate_range("2019-01-02", periods=n_days)
    tickers = [f"EVT{i:02d}" for i in range(n_tickers)]

    rets = rng.normal(0.0, 0.012, (n_days, n_tickers))
    events = []
    for k in range(n_events):
        t = int(rng.integers(200, n_days - 100))
        i = int(rng.integers(0, n_tickers))
        rets[t + 1: t + 1 + hold, i] += bump
        events.append({"ticker": tickers[i], "event_date": dates[t]})

    px = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=dates, columns=tickers)
    panel = Panel(adj_close=px, volume=pd.DataFrame(1e6, index=dates, columns=tickers))
    return panel, pd.DataFrame(events)


def test_event_study_recovers_an_injected_drift():
    panel, events = _panel_with_injected_events()
    result = event_study(panel, events, pre=10, post=30, label="injected events")

    assert result.n_events > 30
    assert result.car_post_event > 0.0
    assert result.t_post_event > 2.0
    assert result.is_significant()
    assert len(result.to_curve()) == 41


def test_event_study_finds_nothing_when_there_is_nothing():
    panel, events = _panel_with_injected_events(bump=0.0)
    result = event_study(panel, events, pre=10, post=30)
    assert not result.is_significant()


def test_event_study_refuses_a_tiny_sample():
    panel, events = _panel_with_injected_events(n_events=40)
    with pytest.raises(ValueError, match="anecdote"):
        event_study(panel, events.head(2), pre=10, post=30, min_events=5)


def test_event_study_summary_quotes_the_sample_size():
    panel, events = _panel_with_injected_events()
    line = event_study(panel, events, pre=10, post=30).summary_line()
    assert "historical instances" in line and "p=" in line


# ---------------------------------------------------------------- B13

def test_sizing_respects_the_position_cap(panels, synthetic_panel):
    model = equal_weight_model(panels)
    as_of = list(panels.values())[0].index[-1]
    alpha = model.score_panel(panels).loc[as_of]

    book = size_positions(alpha, synthetic_panel, as_of, max_position=0.05, max_names=8)
    assert (book.weights.abs() <= 0.05 + 1e-9).all()
    assert book.n_positions <= 8


def test_sizing_is_inverse_to_volatility(panels, synthetic_panel):
    model = equal_weight_model(panels)
    as_of = list(panels.values())[0].index[-1]
    alpha = model.score_panel(panels).loc[as_of]

    book = size_positions(alpha, synthetic_panel, as_of, max_position=1.0, max_names=20)
    frame = book.to_frame().dropna()
    # higher vol must not receive a larger weight than lower vol at similar alpha
    corr = frame["weight"].corr(frame["realized_vol"])
    assert corr < 0


def test_sizing_hits_the_vol_target_when_uncapped(panels, synthetic_panel):
    model = equal_weight_model(panels)
    as_of = list(panels.values())[0].index[-1]
    alpha = model.score_panel(panels).loc[as_of]

    book = size_positions(alpha, synthetic_panel, as_of, target_vol=0.10,
                          max_position=1.0, max_names=15)
    assert book.est_portfolio_vol == pytest.approx(0.10, rel=1e-6)


def test_sizing_handles_an_empty_signal(synthetic_panel):
    empty = pd.Series(np.nan, index=synthetic_panel.tickers)
    book = size_positions(empty, synthetic_panel, synthetic_panel.dates[-1])
    assert book.n_positions == 0


# ==========================================================================
# The A3 seam: Panel.from_wide against Matt's load_wide() interface
# ==========================================================================

class _FakeLoadWide:
    """Stand-in for Matt's A3 `load_wide()` result, matching the attributes he
    specified: .close (adjusted), .raw_close, .volume, .adv, .returns."""

    def __init__(self, dates, tickers, misalign=False):
        rng = np.random.default_rng(2)
        n, m = len(dates), len(tickers)
        px = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (n, m)), axis=0)),
                          index=dates, columns=tickers)
        self.close = px
        self.raw_close = px * 0.98
        self.volume = pd.DataFrame(1e6, index=dates, columns=tickers)
        self.adv = pd.DataFrame(5e7, index=dates, columns=tickers)
        self.returns = px.pct_change()
        if misalign:
            # a ticker present in prices but missing from volume
            self.volume = self.volume.drop(columns=[tickers[-1]])


def test_from_wide_builds_a_working_panel():
    dates = pd.bdate_range("2021-01-04", periods=400)
    tickers = [f"AAA{i}" for i in range(12)]
    panel = Panel.from_wide(_FakeLoadWide(dates, tickers))

    assert panel.n_days == 400
    assert panel.n_tickers == 12
    assert panel.adv is not None
    out = Momentum12_1(lookback=252, skip=21).compute(panel, dates[-1])
    assert out.notna().all()


def test_from_wide_aligns_rather_than_trusting_upstream():
    """A ticker in one frame and not another must not become a NaN column."""
    dates = pd.bdate_range("2021-01-04", periods=300)
    tickers = [f"BBB{i}" for i in range(8)]
    panel = Panel.from_wide(_FakeLoadWide(dates, tickers, misalign=True))

    assert panel.n_tickers == 7
    assert tickers[-1] not in panel.tickers
    assert panel.adj_close.columns.equals(panel.volume.columns)


def test_from_wide_slices_adv_with_everything_else():
    dates = pd.bdate_range("2021-01-04", periods=300)
    panel = Panel.from_wide(_FakeLoadWide(dates, ["CCC0", "CCC1"]))
    window = panel.as_of(dates[100], lag_days=1)
    assert window.adv is not None
    assert len(window.adv) == len(window.adj_close) == 100


def test_from_wide_demands_the_two_required_frames():
    class Bare:
        close = pd.DataFrame(1.0, index=pd.bdate_range("2021-01-04", periods=5), columns=["X"])

    with pytest.raises(AttributeError, match="volume"):
        Panel.from_wide(Bare())


# ==========================================================================
# Optional lane work: B12 matching, B14 regime, optimizers, SignalEngine
# ==========================================================================

from quant.eventstudy.matching import (  # noqa: E402
    analogue_study,
    find_similar_setups,
    study_matched_setups,
)
from quant.optimization.mean_variance import (  # noqa: E402
    alpha_to_expected_returns,
    optimize_book,
)
from quant.optimization.risk_parity import (  # noqa: E402
    covariance_from_panel,
    equal_risk_contribution,
    risk_contributions,
    risk_parity_book,
)
from quant.signals.generation import SignalEngine  # noqa: E402
from quant.signals.regime import (  # noqa: E402
    Regime,
    classify_regimes,
    ic_by_regime,
    market_index,
    regime_at,
    regime_summary,
)


# ---------------------------------------------------------------- B12

def test_matches_come_only_from_the_past(panels, synthetic_panel):
    """Every match, AND its whole outcome window, must precede the decision."""
    target_date = list(panels.values())[0].index[-1]
    ticker = panels["momentum_12_1"].loc[target_date].dropna().index[0]

    post = 40
    result = find_similar_setups(panels, ticker, target_date, k=20, post=post)

    assert result.n_matches > 0
    cutoff = target_date - pd.Timedelta(days=int(np.ceil(post * 7 / 5)))
    assert all(m.date <= cutoff for m in result.matches)


def test_matches_are_separated_to_avoid_double_counting(panels):
    target_date = list(panels.values())[0].index[-1]
    ticker = panels["momentum_12_1"].loc[target_date].dropna().index[0]

    result = find_similar_setups(panels, ticker, target_date, k=30,
                                 min_separation_days=90)
    by_ticker = {}
    for m in result.matches:
        by_ticker.setdefault(m.ticker, []).append(m.date)
    for dates in by_ticker.values():
        dates = sorted(dates)
        gaps = [(b - a).days for a, b in zip(dates, dates[1:])]
        assert all(g >= 90 for g in gaps)


def test_matches_are_ordered_by_closeness(panels):
    target_date = list(panels.values())[0].index[-1]
    ticker = panels["momentum_12_1"].loc[target_date].dropna().index[0]
    result = find_similar_setups(panels, ticker, target_date, k=15)
    distances = [m.distance for m in result.matches]
    assert distances == sorted(distances)


def test_analogue_study_produces_a_quotable_sentence(panels, synthetic_panel):
    target_date = list(panels.values())[0].index[-1]
    ticker = panels["momentum_12_1"].loc[target_date].dropna().index[0]

    result = analogue_study(synthetic_panel, panels, ticker, target_date, k=25)
    assert result.study is not None
    line = result.summary_line()
    assert "historical setups resembling" in line and "market-adjusted" in line
    assert len(result.to_frame()) == result.n_matches


def test_matching_rejects_an_unscored_target(panels):
    target_date = list(panels.values())[0].index[-1]
    with pytest.raises(KeyError, match="no complete factor vector"):
        find_similar_setups(panels, "NOT_A_TICKER", target_date)


# ---------------------------------------------------------------- B14

def test_regime_labels_cover_the_sample(synthetic_panel):
    regimes = classify_regimes(synthetic_panel)
    assert len(regimes) == synthetic_panel.n_days
    known = regimes[regimes != Regime.UNKNOWN.value]
    assert len(known) > 0.5 * synthetic_panel.n_days
    assert set(known.unique()) <= {r.value for r in Regime}


def test_regime_threshold_does_not_use_the_future(synthetic_panel):
    """Truncate the panel and the labels on the shared dates must not move."""
    cut = synthetic_panel.dates[1200]
    full = classify_regimes(synthetic_panel)
    early = classify_regimes(synthetic_panel.as_of(cut, lag_days=0))

    shared = early.index
    pd.testing.assert_series_equal(full.loc[shared], early.loc[shared])


def test_regime_summary_shares_sum_to_one(synthetic_panel):
    summary = regime_summary(classify_regimes(synthetic_panel))
    assert summary["share"].sum() == pytest.approx(1.0)


def test_ic_by_regime_splits_the_scoreboard(panels, synthetic_panel):
    table = ic_by_regime(panels["momentum_12_1"], synthetic_panel,
                         horizon=21, name="momentum_12_1", min_periods=4)
    if not table.empty:
        assert {"mean_ic", "t_stat", "n"} <= set(table.columns)
        assert table["n"].min() >= 4


def test_regime_at_returns_a_label(synthetic_panel):
    regimes = classify_regimes(synthetic_panel)
    assert isinstance(regime_at(regimes, synthetic_panel.dates[-1]), Regime)
    assert regime_at(regimes, pd.Timestamp("1990-01-01")) is Regime.UNKNOWN


# ---------------------------------------------------- optimizers

def test_equal_risk_contribution_actually_equalises_risk(synthetic_panel):
    tickers = list(synthetic_panel.tickers[:8])
    cov = covariance_from_panel(synthetic_panel, synthetic_panel.dates[-1], tickers)
    w = equal_risk_contribution(cov)

    contrib = risk_contributions(w.to_numpy(), np.asarray(cov))
    share = contrib / contrib.sum()
    assert w.sum() == pytest.approx(1.0)
    assert share.max() - share.min() < 0.01   # all within a percentage point


def test_risk_parity_differs_from_naive_inverse_vol_when_correlated(synthetic_panel):
    tickers = list(synthetic_panel.tickers[:6])
    cov = covariance_from_panel(synthetic_panel, synthetic_panel.dates[-1], tickers,
                                shrinkage=0.0)
    erc = equal_risk_contribution(cov)
    inverse_vol = 1.0 / np.sqrt(np.diag(cov.values))
    inverse_vol = pd.Series(inverse_vol / inverse_vol.sum(), index=cov.index)
    # they agree only if correlations are zero; on real-ish data they should not
    assert not np.allclose(erc.values, inverse_vol.values, atol=1e-6)


def test_covariance_shrinkage_pulls_correlations_toward_zero(synthetic_panel):
    tickers = list(synthetic_panel.tickers[:10])
    as_of = synthetic_panel.dates[-1]
    raw = covariance_from_panel(synthetic_panel, as_of, tickers, shrinkage=0.0)
    shrunk = covariance_from_panel(synthetic_panel, as_of, tickers, shrinkage=0.5)

    off_raw = np.abs(raw.values[~np.eye(len(tickers), dtype=bool)]).mean()
    off_shrunk = np.abs(shrunk.values[~np.eye(len(tickers), dtype=bool)]).mean()
    assert off_shrunk < off_raw
    assert np.allclose(np.diag(raw.values), np.diag(shrunk.values))


def test_alpha_to_expected_returns_states_a_bounded_spread():
    alpha = pd.Series([3.0, 1.0, -1.0, -3.0], index=list("ABCD"))
    er = alpha_to_expected_returns(alpha, spread=0.12)
    assert er.max() - er.min() <= 0.12 + 1e-9
    assert er.idxmax() == "A" and er.idxmin() == "D"


def test_mean_variance_respects_caps_and_budget(panels, synthetic_panel):
    as_of = list(panels.values())[0].index[-1]
    engine_alpha = equal_weight_model(panels).score_panel(panels).loc[as_of]

    book = optimize_book(engine_alpha, synthetic_panel, as_of,
                         max_names=8, max_position=0.05)
    assert book.converged
    assert (book.weights <= 0.05 + 1e-8).all()
    assert (book.weights >= -1e-8).all()
    assert book.weights.sum() == pytest.approx(min(1.0, 0.05 * 8), abs=1e-6)


def test_risk_parity_book_reports_risk_shares(panels, synthetic_panel):
    as_of = list(panels.values())[0].index[-1]
    alpha = equal_weight_model(panels).score_panel(panels).loc[as_of]
    book = risk_parity_book(alpha, synthetic_panel, as_of, max_names=6)
    assert len(book) == 6
    assert book["pct_of_risk"].sum() == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------- SignalEngine

def test_engine_runs_the_whole_lane(synthetic_panel, synthetic_truth):
    engine = SignalEngine.default().fit(synthetic_panel, train_frac=0.6)

    assert engine.fitted is not None
    assert engine.model is not None
    scores = engine.scores()
    assert scores.notna().any().any()

    train, test = engine.train_test_dates(train_frac=0.6)
    assert train[-1] < test[0]

    ideas = engine.rank(engine.dates[-1], top_n=5)
    assert len(ideas) == 5
    assert all(i.check_sums(1e-9) for i in ideas)


def test_engine_evaluates_out_of_sample(synthetic_panel):
    engine = SignalEngine.default().fit(synthetic_panel, train_frac=0.6)
    _, test = engine.train_test_dates(train_frac=0.6)
    result = engine.evaluate(synthetic_panel, dates=test)
    assert result.n_periods > 5
    assert np.isfinite(result.mean_ic)


def test_engine_refuses_to_score_without_weights(synthetic_panel):
    engine = SignalEngine.default().build(synthetic_panel)
    with pytest.raises(RuntimeError, match="never invented"):
        engine.scores()


def test_engine_equal_weight_baseline(synthetic_panel):
    engine = SignalEngine.default().use_equal_weights(synthetic_panel)
    assert engine.fitted is None
    assert engine.scores().notna().any().any()


def test_engine_carries_the_nlp_stub_without_breaking(synthetic_panel):
    engine = SignalEngine.default().fit(synthetic_panel)
    assert "catalyst_sentiment" in engine.panels
    assert engine.panels["catalyst_sentiment"].isna().all().all()
    assert engine.scores().notna().any().any()


def test_ridge_tolerates_a_stubbed_factor(panels, synthetic_panel):
    """The B8 stub is all-NaN; ridge must exclude it, not refuse to fit."""
    from quant.factors.nlp import CatalystSentiment
    dates = list(panels.values())[0].index
    stub = CatalystSentiment()
    with_stub = dict(panels)
    with_stub[stub.name] = factor_panel(stub, synthetic_panel, dates)

    train, _ = split_train_test(dates, train_frac=0.7, embargo=21)
    fitted = fit_ridge_weights(with_stub, synthetic_panel, train, horizon=21)

    assert fitted.weights[stub.name] == 0.0
    assert sum(abs(w) for w in fitted.weights.values()) == pytest.approx(1.0)
