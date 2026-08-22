# Architecture

## The one design decision everything follows from

The language model may **propose**. It may not **measure**.

Every AI finance tool fails the same way: the model is asked to be the source of
truth, so when you ask it for a Sharpe ratio it gives you one, and the number is
invented. Here that boundary is structural rather than a matter of prompting.

| The LLM does | The quant engine does |
|---|---|
| Decompose a thesis into screening criteria | Build the universe and run the funnel |
| Read filings and transcripts | Compute factors and z-score them cross-sectionally |
| Extract catalysts with verbatim quotes | Fit weights on a training window |
| Argue the bull case, then attack it | Backtest with costs, slippage and liquidity caps |
| Write the final rationale | Compute risk, VaR and exposure |

No agent computes a statistic. The quant validator calls into `quant/` and
reports what came back, `backend/agents/quant_validator/validator.py` exists to
make that boundary a file rather than a promise.

## The pipeline

```
thesis (natural language)
  -> decompose            LLM   backend/research/thesis/decomposer.py
  -> screening criteria   LLM   backend/research/thesis/criteria.py
  -> universe + funnel    QUANT quant/universe/builder.py
  -> retrieve filings     LLM   backend/rag/retrieval/retriever.py
  -> extract catalysts    LLM   backend/research/catalysts/extractor.py
  -> factors              QUANT quant/factors/
  -> composite alpha      QUANT quant/alpha/composite.py
  -> backtest             QUANT quant/backtest/engine.py
  -> risk panel           QUANT quant/risk/
  -> bull / bear debate   LLM   backend/agents/{bull,bear}/
  -> quant validation     QUANT via backend/agents/quant_validator/
  -> TradeIdea            
```

`backend/services/pipeline.py` runs the chain as a plain function. The HTTP
layer wraps it; it is not required by it.

## The frozen contract

Before any product code was written, the team froze the objects that pass
between layers as pydantic models in `data/schemas/`. `TradeIdea` is the one
everything converges on:

```python
TradeIdea(
    ticker, company_name, side, as_of,
    alpha_score, confidence, expected_alpha,   # computed, never authored
    catalysts=[Catalyst, ...],                 # each with a resolvable URL
    alpha=AlphaBreakdown,                      # per-factor contributions
    backtest=BacktestResult,                   # curves, Sharpe, drawdown
    risk=RiskMetrics,                          # beta, VaR, exposures
    validator_verdict, bull_case, bear_case, pm_rationale,
)
```

Two consequences worth naming:

**Four people could build in parallel.** The contract was agreed in hour one, so
the data engineer, the quant, the ML engineer and the product lead never had to
wait on each other's internals. The parts fit on first contact.

**The frontend is contract-driven.** FastAPI publishes the same schemas as an
OpenAPI spec at `/openapi.json`, 14 endpoints, 36 typed schemas. The Base44
dashboard is built directly against that spec, so a no-code UI stayed in lockstep
with a 14,000-line quantitative backend without a single coordination meeting.
The contract freeze is what made a no-code frontend viable against an engine
this size.

## Look-ahead defence

Look-ahead bias never raises an exception. It just makes every number better
than it should be, until someone asks how it was handled. Four guards, each with
a regression test:

**Structural truncation**, `quant/factors/base.py`. A factor never receives the
full history. `Factor.compute()` slices the panel to the decision date before
`_compute()` sees it, so a careless factor *cannot* read the future, it is not
in the object it was handed.

**report_date, not period_end**, `data/pipelines/fundamentals.py`. Fundamentals
join on the date a figure became public, never the quarter it describes.
Restatements de-duplicate to the first filing, because that is the number the
market actually traded on. AAPL reports 2007 periods that were not filed until
2009; joining on the period would hand the backtest two years of hindsight.

**Train/test embargo**, `quant/alpha/weighting.py`. The 21-day forward return of
the last training date would otherwise reach into the test window. The split
drops it, so fitted weights never see the period they are judged on.

**Retrieval time filter**, `backend/rag/retrieval/retriever.py`. Top-k with a
hard `filed_date <= as_of` filter, so an agent testing a historical date cannot
read a document that did not exist yet.

The test that matters is `test_factor_cannot_see_the_future`: compute a signal at
date *t*, append the future to the panel, compute again, and assert the two are
identical.

## Data

| Source | What | Where |
|---|---|---|
| Yahoo Finance | Daily OHLCV, 80 US large caps, 2015-2024 | `data/pipelines/prices.py` |
| SEC EDGAR companyfacts | 5,629 point-in-time quarters, 79 tickers, back to 2006 | `data/pipelines/fundamentals.py` |
| SEC EDGAR documents | 10-K / 10-Q / 8-K text for retrieval | `data/pipelines/edgar.py` |

Everything is cached to disk. No demo path depends on a live API call.

## The quantitative engine

**Factors** (`quant/factors/`), market (12-1 momentum, realised volatility,
volume trend, relative strength), fundamental (growth, margins, FCF yield, ROIC,
leverage, valuation), event (standardised unexpected earnings, post-earnings
drift, acceleration), and NLP (catalyst sentiment, guidance changes).

Earnings surprise is measured as SUE, a seasonal-random-walk expectation scaled
by the company's own history of surprises, because no free source provides
analyst estimates. It is named as SUE rather than dressed up as a consensus
surprise.

**Signals** (`quant/signals/`), winsorize, z-score, cross-sectional ranking,
optional sector neutralisation. The model is cross-sectional: the question is
never "is this stock's momentum high" but "is it high against every other name
in the universe today".

**Alpha** (`quant/alpha/`), information coefficient, rank-IC, t-statistics and
signal decay were built *before* the factors they judge, so every factor has to
earn its place with a number. Weights are IC-fitted on the training window only.

**Backtest** (`quant/backtest/`), event-driven, with commission, participation-
rate slippage, position limits and a hard cap on trading more than 5% of a
name's average daily volume.

**Execution** (`cpp/`), a price-level order book and a Nasdaq ITCH parser in
C++, exposed through pybind11, so slippage can be derived from book mechanics
rather than a basis-point assumption. Routed behind a config flag with the pure
Python path still working: a compiler that fails on the demo laptop degrades to
a less impressive number, never to a broken demo.

## Honest limits

- No factor is statistically significant on an 80-name universe. Cross-sectional
  models need breadth; 80 large caps is not enough.
- Out-of-sample information coefficient drops roughly 65% from in-sample.
- Eighteen factors were tested and one cleared p < 0.05. That is what chance
  alone predicts, and we do not claim it.
- The backtest is long-only, monthly, and ignores borrow and taxes.

Every backtest response carries its own significance in `BacktestResult.notes`,
so a Sharpe ratio cannot leave the system without the evidence for it attached.
