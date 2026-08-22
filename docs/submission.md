# AI Quant Council — submission

**Track: Fintech**

## The one-liner

Ask any AI chatbot whether a stock is a good buy and it will hand you a Sharpe
ratio. That number is invented. We built the layer the entire AI-finance
category is missing: a quantitative engine that tests the AI's idea against
historical evidence and is allowed to say **no**.

## The problem

AI stock advice is now everywhere, and it is confidently wrong. A language model
asked for a risk-adjusted return will produce one — fluent, specific, and
fabricated, because nothing in the system ever computed it. Retail investors act
on that output. There is no falsification step anywhere in the category.

The failure is structural, not a prompting problem. If the model is the source
of truth, there is nothing to check it against.

## What we built

Give the system an investment thesis in plain English:

> *"Find companies benefiting from the AI data-center buildout that the market
> may be underpricing."*

It decomposes the thesis into measurable screening criteria, scans a universe of
US equities, retrieves the actual SEC filings for the survivors, extracts
catalysts with verbatim quotes and clickable source links, builds bull and bear
cases from that evidence, then hands the whole thing to a quantitative engine
that scores it, backtests it with realistic costs, measures the risk — and
rejects it if it does not survive.

**The language model proposes. The quantitative engine decides.**

That boundary is structural. The model never emits a Sharpe ratio, an alpha
score, a factor weight or a VaR. Those come from 14,000 lines of Python and
1,500 lines of C++ that the model cannot reach into. The quant validator agent
exists specifically to call that engine and report what came back, never to
estimate a number itself.

## Why this is different

Three things no chatbot does:

**Every claim carries a source.** A catalyst without a resolvable URL fails
schema validation and cannot ship. Click one in the demo and you land on the
actual 10-Q. The quote is extracted verbatim from the retrieved document — the
model never writes it.

**The system is allowed to reject its own ideas.** Candidates that fail
quantitative validation are excluded from the portfolio *with the reasons
attached*:

```
ACCEPTED  NGSM   5.0%   alpha 8.7   confidence 0.82
REJECTED  WEAK   -> quant validator rejected the idea
                    confidence 0.22 below minimum 0.55
```

**It tells you when it doesn't know.** Our own strategy returns +9.78%
annualised excess return against SPY at a Sharpe of 1.03 — and an out-of-sample
information coefficient of 0.02, t = 0.67. **That is not statistically
significant, and every backtest response says so in its own payload.** A Sharpe
ratio cannot leave this system without the evidence for it attached.

We are reporting a number that does not flatter us because that is the entire
product. A tool that only ever agrees with you is the thing we built this to
replace.

## Technical work

**Look-ahead defence, enforced structurally.** Factors are handed a panel already
truncated to the decision date, so a careless factor *cannot* read the future.
Fundamentals join on the date a figure became public, not the quarter it
describes — AAPL reports 2007 periods that were not filed until 2009, and
joining on the period would hand the backtest two years of hindsight.
Restatements de-duplicate to the first filing. Train and test are separated by
an embargo the length of the return horizon. The document retriever refuses
anything filed after the decision date. Each guard has a test that appends the
future to the data and asserts the signal does not move.

**Point-in-time fundamentals from SEC EDGAR.** 5,629 quarters across 79 tickers
back to 2006, pulled from the XBRL companyfacts API — the only free source
carrying both the fiscal period and the publication date. Q4 is derived from the
annual filing, because most issuers fold it in and without that correction
"four quarters ago" silently becomes five.

**A statistical scoreboard built before the factors.** Information coefficient,
rank-IC, t-statistics and signal decay, so every factor has to earn its place
with a number rather than an argument.

**C++ execution simulation.** A price-level order book and a Nasdaq ITCH parser
behind pybind11, so slippage derives from book mechanics rather than a
basis-point assumption — routed behind a config flag with the pure-Python path
still working.

**233 tests.** Including one that plants a known effect in a synthetic market
and asserts the engine recovers it at the right magnitude — proving the
measurement is correct independently of whether the alpha is.

## Built with Base44

The data contract was frozen in the first hour as pydantic models. FastAPI
publishes it as an OpenAPI specification — 14 endpoints, 36 typed schemas — and
the Base44 dashboard is built directly against that spec.

That is what made a four-person parallel build work. A no-code frontend stayed
in lockstep with a 14,000-line quantitative backend and 1,500 lines of C++,
with no coordination overhead, while three engineers built the engine
underneath it. Base44 was not where we made the pages; it was the decision that
let the product surface move as fast as the backend.

## Honest limits

- 80 names is too narrow a universe for a cross-sectional model. Momentum needs
  breadth and we did not have it.
- Out-of-sample IC drops about 65% from in-sample.
- Eighteen factors tested, one cleared p < 0.05 — which is what chance alone
  predicts. We are not claiming it.
- Long-only, monthly rebalance, no borrow costs or taxes.

With another week: the full S&P 500, twenty years of history, and walk-forward
refitting instead of a single split. That is the honest path from *the engine
works* to *the strategy works*.

## Stack

Python · FastAPI · NumPy / pandas / SciPy · FAISS · pydantic · C++ / pybind11 ·
Base44 · SEC EDGAR · Yahoo Finance. Total spend under $40.

---

*Matt · Nalin · Zain · Cecile — IgnitionHacks V7*
