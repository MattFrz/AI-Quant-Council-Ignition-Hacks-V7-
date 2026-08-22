# Test Coverage

What is actually tested, mapped to the build-order steps. One row per step, with
an honest status - a step marked **none** has working code and no test proving
it stays working.

**Run everything:**

```bash
.venv\Scripts\activate
python -m pytest tests/ -v
```

**Current: 36 tests, all passing, ~6s, no network required.**

| Status | Meaning |
|---|---|
| **unit** | Has assertions in `tests/`. Breaks loudly in CI. |
| **smoke** | Exercised end-to-end by `scripts/run_backtest.py`, but nothing asserts the output. A silent regression would pass. |
| **none** | No automated coverage. Verified by hand once. |

| Lane | Owner | State |
|---|---|---|
| **A** - data, backtest, risk, C++ | **Matt** | steps A1-A18 built, documented below |
| B - factors, signals, alpha | Nalin | not started |
| C - agents, RAG, NLP | Zain | not started |
| D - frontend, portfolio, demo | Cecile | not started |

---

## Phase 0 / 1 - Foundation and contract

| Step | File | Status | Test |
|---|---|---|---|
| 1.1-1.12 | `data/schemas/*`, `frontend/lib/types.ts`, fixture | **unit** | `scripts/verify_contract.py` - 7 checks: fixture validates, audit trail clickable, alpha contributions sum to composite, train/test windows disjoint, every catalyst predates the decision date, risk attached |

```bash
python scripts/verify_contract.py
```

---

## Phase 2 - Lane A (Matt)

### Data layer

| Step | File | Status | Notes |
|---|---|---|---|
| A1 | `data/sources/base.py` | **none** | Retry, rate limiter and disk cache verified only by the seed run succeeding. Worth a test that a second call hits cache instead of the network. |
| A2 | `data/sources/yahoo.py` | **none** | Verified by hand: 25 tickers x 6 years -> 37,725 rows cached. |
| A3 | `data/pipelines/prices.py` | **smoke** | `to_wide`, `daily_returns`, `average_dollar_volume`, `align` all run inside `run_backtest.py`. No assertions on their output. |
| A4 | `scripts/seed_data.py` | **none** | Manual. Prints row counts and date range for eyeballing. |

### Universe

| Step | File | Status | Notes |
|---|---|---|---|
| A5 | `quant/universe/filters.py` | **none** | Four filters, no tests. Each returns a count the UI depends on - a wrong count is a visibly wrong funnel. |
| A6 | `quant/universe/builder.py` | **none** | `funnel()` output shape is what Cecile's `FunnelStage` consumes. Untested. |

### Backtester

| Step | File | Status | Tests |
|---|---|---|---|
| A7 | `quant/backtest/events.py` | **smoke** | `FillEvent` construction exercised by every engine test; no direct assertions. |
| A8 | `quant/backtest/leakage_guards.py` | **unit** (15) | The strongest coverage in the repo - see below |
| A9 | `quant/backtest/costs.py` | **unit** (2) | `test_illiquid_names_cost_more_to_trade`, `test_cost_scales_with_notional` |
| A10 | `quant/backtest/slippage.py` | **unit** (3) | `test_slippage_grows_with_participation`, `test_slippage_is_sublinear`, `test_slippage_always_works_against_you` |
| A11 | `quant/backtest/engine.py` | **unit** (6) | `test_engine_runs_and_produces_a_valid_result`, `test_position_cap_binds_at_rebalance`, `test_positions_drift_only_modestly_between_rebalances`, `test_liquidity_cap_limits_trade_size`, `test_costs_reduce_returns`, `test_engine_applies_execution_lag` |
| A12 | `quant/backtest/metrics.py` | **unit** (8) | Sharpe (2), max drawdown (2), Sortino (2), win rate, turnover |
| A13 | `quant/backtest/benchmark.py` | **smoke** | `compare()` prints beta / CAPM alpha / info ratio in `run_backtest.py`. No assertions. |
| A14 | `quant/backtest/walk_forward.py` | **unit** (2) | `test_splits_never_overlap`, `test_splits_move_forward_in_time` |

### Risk engine

| Step | File | Status | Notes |
|---|---|---|---|
| A15 | `quant/risk/metrics.py` | **smoke** | `build_risk_metrics` assembles the full panel in `run_backtest.py`. Nothing asserts the numbers. |
| A16 | `quant/risk/var.py` | **smoke** | VaR/CVaR print. An easy, valuable test: CVaR must always be <= VaR. |
| A17 | `quant/risk/exposures.py` | **smoke** | `exposure_report` runs; concentration and effective-positions untested. |
| A18 | `quant/risk/correlation.py`, `liquidity.py` | **smoke** | Average pairwise correlation and days-to-liquidate print only. |

---

## The tests that matter most

### `test_leakage.py` - 15 tests

If any of these fail, every performance number downstream is fiction. Two are
worth understanding rather than just running:

**`test_lookahead_signal_fails_the_stability_check`** - builds a signal that
peeks at tomorrow's price and asserts the guard *catches* it. This test found a
real bug: `check_stability` was comparing frames with arithmetic that let pandas
skip NaN mismatches, so a genuine look-ahead signal passed the leak detector.
A leak detector that cannot fail is worse than none.

**`test_available_at_uses_report_date_not_period_end`** - on 2023-04-15 the Q1
period has ended but nothing has been published. Filtering on `period_end`
instead of `report_date` is the classic look-ahead bug and this pins it.

The rest cover: as-of joins, `latest_available`, execution lag enforcement,
zero-lag rejection, future-data assertions, window overlap, and same-day
correlation detection.

### `test_engine_applies_execution_lag`

Feeds the engine tomorrow's return as a signal and asserts Sharpe stays below 8.
Without the lag this would print an absurd number. This is the test that proves
the engine is honest.

### `test_costs_reduce_returns`

Same signal, run twice: once with zero costs, once with 10bps commission and
50bps slippage. Asserts the second is worse. If this ever passes trivially,
costs have stopped being charged.

---

## Known gaps, ranked

1. **Universe filters (A5, A6) have no tests.** They produce the counts the demo
   funnel animates. A wrong count is visible on screen to a judge.
2. **Risk engine (A15-A18) is smoke-only.** Cheap wins available: CVaR <= VaR,
   beta of a series against itself is 1.0, days-to-liquidate scales linearly
   with position size.
3. **No cache test (A1).** The offline-mode guarantee - the thing the whole demo
   depends on - is unproven. A test that seeds, flips `offline=True`, and asserts
   the second read succeeds without network would cover it.
4. **`test_backtest.py` runs on synthetic data only.** Correct for unit tests, but
   nothing catches a regression in the real Yahoo path.

---

## Running subsets

```bash
python -m pytest tests/quant/test_leakage.py -v
```

```bash
python -m pytest tests/ -k "slippage or cost" -v
```

```bash
python scripts/run_backtest.py --walk-forward
```

The last one is the end-to-end check: load -> universe -> signal -> leakage guard ->
engine -> metrics -> benchmark -> risk. It prints numbers rather than asserting
them, so read the output - it flags `beta > 1.1` itself when returns are mostly
market exposure rather than alpha.

---

## Lane B - Nalin

_Not started. Factors, signals, alpha model, event study._

## Lane C - Zain

_Not started. LLM client, RAG, catalyst extraction, the five agents._

## Lane D - Cecile

_Not started. Frontend components, charts, portfolio construction._
