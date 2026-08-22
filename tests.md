# Test Coverage

What is actually tested, mapped to the build-order steps. One row per step, with
an honest status. A step marked **none** has working code and no test proving it
stays working.

**Run everything:**

```bash
.venv\Scripts\activate
python -m pytest tests/ -v
```

**Current: 89 tests, all passing, ~15s, no network required.**

| Status | Meaning |
|---|---|
| **unit** | Has assertions in `tests/`. Breaks loudly. |
| **smoke** | Exercised end-to-end by `scripts/run_backtest.py`, but nothing asserts the output. A silent regression would pass. |
| **none** | No automated coverage. Verified by hand. |

| Lane | Owner | State |
|---|---|---|
| **A** - data, backtest, risk, C++ | **Matt** | A1-A19 built and covered, documented below |
| B - factors, signals, alpha | Nalin | not started |
| C - agents, RAG, NLP | Zain | not started |
| D - frontend, portfolio, demo | Cecile | D1/D2 pushed, no tests yet |

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

| Step | File | Status | Tests |
|---|---|---|---|
| A1 | `data/sources/base.py` | **unit** (13) | `test_cache.py` - cache hit/miss, refresh, offline serves warm cache, offline fails loudly on cold cache, retry with backoff, gives up after max retries, key stability and filename safety |
| A2 | `data/sources/yahoo.py` | **none** | Verified by hand: 504 tickers requested, 500 returned, 1.21M rows. Delisted names (JNPR, FDXF, HONA, Q, SNDK) skipped without crashing. |
| A3 | `data/pipelines/prices.py` | **smoke** | `to_wide`, `daily_returns`, `average_dollar_volume`, `align`, `load_wide` all run inside `run_backtest.py`. No assertions on their output. |
| A4 | `scripts/seed_data.py` | **none** | Manual. Prints row counts, date range and the universe funnel. |

### Universe

| Step | File | Status | Tests |
|---|---|---|---|
| A5 | `quant/universe/filters.py` | **unit** (7) | `test_universe.py` - each of the four filters, missing market cap dropped not kept, filters respect the `universe` argument so stages stay cumulative, count matches survivors |
| A6 | `quant/universe/builder.py` | **unit** (5) | Funnel is monotonic (counts never rise), funnel shape validates as `FunnelStage`, survivors satisfy *every* filter not just the last, `max_size` caps by liquidity, scanned count is the full input |

### Backtester

| Step | File | Status | Tests |
|---|---|---|---|
| A7 | `quant/backtest/events.py` | **smoke** | `FillEvent` construction exercised by every engine test; no direct assertions. |
| A8 | `quant/backtest/leakage_guards.py` | **unit** (15) | The strongest coverage in the repo - see below |
| A9 | `quant/backtest/costs.py` | **unit** (2) | Illiquid names cost more, cost scales with notional |
| A10 | `quant/backtest/slippage.py` | **unit** (3) | Impact grows with participation, is sublinear, always works against you |
| A11 | `quant/backtest/engine.py` | **unit** (6) | Valid result, position cap binds at rebalance, drift stays bounded, liquidity cap limits trade size, costs reduce returns, execution lag neutralises foresight |
| A12 | `quant/backtest/metrics.py` | **unit** (8) | Sharpe (2), max drawdown (2), Sortino (2), win rate, turnover |
| A13 | `quant/backtest/benchmark.py` | **smoke** | `compare()` prints beta / CAPM alpha / info ratio. No assertions. |
| A14 | `quant/backtest/walk_forward.py` | **unit** (2) | Splits never overlap, splits move forward in time |

### Risk engine

| Step | File | Status | Tests |
|---|---|---|---|
| A15 | `quant/risk/metrics.py` | **unit** (7) | `test_risk.py` - beta against itself is 1.0, beta of 2x benchmark is 2.0, risk-band thresholds, assembler populates the panel, assembler degrades gracefully on missing inputs |
| A16 | `quant/risk/var.py` | **unit** (5) | CVaR never worse than VaR, VaR negative, higher confidence is worse, None on thin history, stress scenarios find the real worst day |
| A17 | `quant/risk/exposures.py` | **unit** (7) | Sector exposure sums to gross, unknown sector labelled not dropped, concentration, Herfindahl of equal weights, effective positions reveals hidden concentration, factor exposure, long/short nets to zero |
| A18 | `quant/risk/correlation.py`, `liquidity.py` | **unit** (9) | Correlation in range, correlated pair ranks first, diversification ratio >= 1, days-to-liquidate scales linearly and falls with liquidity, horizon set by slowest name, capacity grows with liquidity |
| A19 | `tests/quant/test_leakage.py` | **unit** | See below |

---

## The tests that matter most

### `test_leakage.py` - 15 tests

If any of these fail, every performance number downstream is fiction.

**`test_lookahead_signal_fails_the_stability_check`** - builds a signal that
peeks at tomorrow's price and asserts the guard *catches* it. This test found a
real bug: `check_stability` compared frames with arithmetic that let pandas skip
NaN mismatches, so a genuine look-ahead signal passed the leak detector. A leak
detector that cannot fail is worse than none.

**`test_available_at_uses_report_date_not_period_end`** - on 2023-04-15 the Q1
period has ended but nothing has been published. Filtering on `period_end`
instead of `report_date` is the classic look-ahead bug and this pins it.

### `test_engine_applies_execution_lag`

Feeds the engine tomorrow's return as a signal and asserts Sharpe stays below 8.
Without the lag this would print an absurd number. This is the test that proves
the engine is honest.

### `test_costs_reduce_returns`

Same signal, zero costs versus 10bps commission plus 50bps slippage. Asserts the
second is worse. If this passes trivially, costs have stopped being charged.

### `test_offline_mode_serves_a_warm_cache` / `..._fails_loudly_on_a_cold_cache`

Seeds online, flips offline, asserts the read succeeds without reaching the
loader, and that a missing key raises rather than silently fetching. The demo
reliability story rests on exactly this pair.

### `test_beta_against_itself_is_one`

One line, catches a swapped covariance/variance denominator instantly.

---

## Bugs these tests caught

1. **`check_stability` let a look-ahead signal pass.** NaN mismatches were
   silently skipped by pandas' `.max()`. Fixed with an explicit NaN-mask compare.
2. **Sharpe returned 7.28e16** on a near-constant series. Floating-point std of
   `[0.001]*100` is 2.2e-19, so the `== 0` guard never fired. Now floored at
   1e-12 across sharpe, sortino, beta and information ratio.
3. **Cost drag reported 82% of capital.** The metric divided cumulative dollar
   costs by *initial* capital, so a book compounding over 10 years looked like it
   paid 82% when it paid 1.09%/yr. Now measured against average equity.

---

## Known gaps, ranked

1. **A2 (`yahoo.py`) has no tests.** The download path is verified only by the
   seed run succeeding. A test with a mocked yfinance response would cover the
   batch-splitting and MultiIndex-unpacking logic.
2. **A3 and A13 are smoke-only.** `to_wide` / `align` and the benchmark
   comparison run but nothing asserts their output.
3. **`test_backtest.py` runs on synthetic data.** Correct for unit tests, but
   nothing catches a regression in the real Yahoo path.
4. **Survivorship bias is warned about, not fixed.** `run_backtest.py` now
   prints the caveat, but the universe is still current index membership.

---

## Running subsets

```bash
python -m pytest tests/quant/test_leakage.py -v
```

```bash
python -m pytest tests/ -k "offline or cache" -v
```

```bash
python scripts/run_backtest.py --walk-forward
```

The last one is the end-to-end check: load -> universe -> signal -> leakage
guard -> engine -> metrics -> benchmark -> risk. It prints numbers rather than
asserting them, and ends with a `READ BEFORE QUOTING THESE NUMBERS` block that
flags survivorship bias, implausible alpha and excessive concentration
automatically.

---

## Lane B - Nalin

_Not started. Factors, signals, alpha model, event study._

Entry point is ready: `from data.pipelines.prices import load_wide`.

## Lane C - Zain

_Not started. LLM client, RAG, catalyst extraction, the five agents._

Note: EDGAR `companyfacts` is the only free source of `report_date`. Watch the
restatement trap - 10-Ks repeat prior periods, so take `min(filed)` per
(concept, period_start, period_end) and that filing's value.

## Lane D - Cecile

_D1/D2 pushed (`api.ts`, `stream.ts`, `globals.css`). No tests yet._
