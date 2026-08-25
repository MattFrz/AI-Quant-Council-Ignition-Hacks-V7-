# Test Coverage

What is actually tested. One row per area, with an honest status, so a gap is
something you decided to accept rather than something you discovered late.

```bash
.venv\Scripts\activate
python -m pytest tests/ -q
```

**233 tests, 229 passing and 4 skipped, roughly 8 minutes. No network required.**

The 4 skips are the end-to-end pipeline tests, which skip automatically when
`data/cache/` is empty so the suite still passes on a fresh clone.

| Status | Meaning |
|---|---|
| **unit** | Has assertions. Breaks loudly. |
| **smoke** | Runs inside `scripts/run_backtest.py` but nothing asserts the output. A silent regression would pass. |
| **none** | No automated coverage. Verified by hand. |

---

## Where the tests are

| File | Tests | Covers |
|---|---|---|
| `tests/quant/test_factors.py` | 93 | Factor construction, signals, alpha model, event study |
| `tests/quant/test_risk.py` | 28 | VaR, CVaR, beta, exposures, correlation, liquidity |
| `tests/backend/test_pipeline.py` | 22 | End-to-end pipeline, result cache, job runner |
| `tests/quant/test_backtest.py` | 21 | Engine, costs, slippage, metrics, walk-forward splits |
| `tests/quant/test_leakage.py` | 15 | Point-in-time correctness |
| `tests/backend/test_quant_routes.py` | 15 | Backtest, risk and portfolio endpoints |
| `tests/quant/test_cache.py` | 13 | Disk cache, offline mode, retry |
| `tests/quant/test_universe.py` | 12 | Universe filters and the funnel |
| `tests/quant/test_fundamentals_pipeline.py` | 10 | EDGAR point-in-time fundamentals |
| `tests/backend/test_portfolio.py` | 4 | Sizing and constraints |
| `tests/cpp/test_orderbook.cpp` | 16 groups | Order book, ITCH parser, execution sim |

---

## The tests that matter most

### `test_leakage.py`

If any of these fail, every performance number downstream is fiction.

**`test_lookahead_signal_fails_the_stability_check`** builds a signal that peeks
at tomorrow's price and asserts the guard *catches* it. This found a real bug:
`check_stability` compared frames with arithmetic that let pandas skip NaN
mismatches, so a genuine look-ahead signal passed the leak detector. A leak
detector that cannot fail is worse than none.

**`test_available_at_uses_report_date_not_period_end`** pins the classic
mistake. On 2023-04-15 the Q1 period has ended but nothing has been published.

### `test_engine_applies_execution_lag`

Feeds the engine tomorrow's return as a signal and asserts Sharpe stays below 8.
Without the lag this prints an absurd number. This is the test that proves the
engine is honest.

### `test_costs_reduce_returns`

Same signal, run twice: zero costs versus 10bps commission and 50bps slippage.
Asserts the second is worse. If it ever passes trivially, costs have stopped
being charged.

### `test_offline_mode_serves_a_warm_cache`

Seeds online, flips offline, asserts the read succeeds without reaching the
loader. The demo-reliability story rests on this.

### `test_beta_against_itself_is_one`

One line, catches a swapped covariance/variance denominator instantly.

---

## Bugs these tests caught

1. **`check_stability` let a look-ahead signal pass.** NaN mismatches were
   silently skipped by pandas' `.max()`.
2. **Sharpe returned 7.28e16** on a near-constant series. Floating-point std of
   `[0.001]*100` is 2.2e-19, so the `== 0` guard never fired. Now floored at
   1e-12 across sharpe, sortino, beta and information ratio.
3. **An event study reported t = 2.2e16.** Twenty catalysts from one filing are
   one event, not twenty; identical paths collapsed the standard error. Events
   now deduplicate on (ticker, date).
4. **Cost drag reported 82% of capital.** Cumulative dollars were divided by
   *initial* capital, so a book compounding over ten years looked like it paid
   82% when it paid 1.09% a year.

---

## Known gaps, ranked

1. **`data/sources/yahoo.py` has no unit tests.** The download path is verified
   only by the seed run succeeding. Mocked responses would cover batch splitting
   and MultiIndex unpacking.
2. **The reshape helpers and benchmark comparison are smoke-only.** `to_wide`,
   `align` and `compare()` all run but nothing asserts their output.
3. **`test_backtest.py` runs on synthetic data.** Correct for unit tests, but
   nothing catches a regression in the real Yahoo path.
4. **Survivorship bias is warned about, not fixed.** The universe is the current
   S&P 500.

---

## Running subsets

```bash
python -m pytest tests/quant/test_leakage.py -v
python -m pytest tests/ -k "offline or cache" -v
python scripts/verify_contract.py
python scripts/run_backtest.py --walk-forward
```

The last one is the end-to-end check: load, universe, signal, leakage guard,
engine, metrics, benchmark, risk. It prints numbers rather than asserting them,
and ends with a `READ BEFORE QUOTING THESE NUMBERS` block flagging survivorship
bias, implausible alpha and excessive concentration automatically.

## C++ tests

```bash
cmake -S cpp -B cpp/build && cmake --build cpp/build
./cpp/build/test_orderbook
```

No test framework, one binary, exits non-zero on failure. Three behaviours worth
knowing:

- **ITCH replace sends an order to the back of the queue.** Modelling it as an
  in-place edit hands every replaced order a position it did not earn.
- **Passive orders do not fill just because you posted.** 500 shares trading
  against a 1000-share queue ahead of you fills nothing.
- **Getting picked off is charged as a cost.** Booking it at your limit price is
  how passive backtests manufacture free money.
