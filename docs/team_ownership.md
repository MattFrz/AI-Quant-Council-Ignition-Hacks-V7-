# Who built what

Four people, four genuinely different technical layers. The build ran as four
parallel lanes against a data contract frozen in the first hour.

## Matt — data, backtester, risk, C++

Market data pipeline and disk cache. Universe construction and the screening
funnel. The event-driven backtester with commission, participation-rate
slippage, position limits and an ADV cap. The risk engine: beta, volatility,
drawdown, historical VaR and CVaR, sector exposure, concentration, correlation,
days-to-liquidate. Integration: `services/pipeline.py`, the job runner and the
result cache. Phase 4 in C++ — a price-level order book, a Nasdaq ITCH parser
and a fill simulator with queue position, behind pybind11.

`data/sources/` · `data/pipelines/prices.py` · `quant/universe/` ·
`quant/backtest/` · `quant/risk/` · `cpp/` · `backend/services/`

## Nalin — factors, alpha model, statistical testing

The research engine. A `Factor` base class that structurally prevents
look-ahead by truncating the panel before a factor ever sees it. Market,
fundamental, event and NLP factors. Cross-sectional normalisation and ranking.
The statistical scoreboard — information coefficient, rank-IC, t-statistics,
signal decay — built deliberately *before* the factors it judges. The composite
alpha model returning per-factor contributions rather than a single opaque
score, with IC and ridge weight fitting on the training window under an embargo.
Event studies and historical analogue matching. Volatility-scaled position
sizing, equal-risk-contribution and mean-variance optimisers.

Also the SEC EDGAR companyfacts pipeline — 5,629 point-in-time quarters keyed on
publication date rather than fiscal period — and the quant-facing API routes.

`quant/factors/` · `quant/signals/` · `quant/alpha/` · `quant/eventstudy/` ·
`quant/optimization/` · `data/pipelines/fundamentals.py` ·
`backend/api/routes/{backtest,risk}.py` · `notebooks/`

## Zain — retrieval, research agent, the debate

The intelligence layer. One LLM client with retry, token counting and a running
cost log against a $40 ceiling. Thesis decomposition into machine-executable
screening criteria. SEC EDGAR document retrieval, section-aware chunking,
embeddings cached by content hash, a FAISS index, and a retriever with a hard
`filed_date <= as_of` filter so an agent testing a historical date cannot read a
document that did not exist yet. Event extraction from filings, transcripts and
news. Catalyst objects carrying verbatim quotes and resolvable source URLs. The
bull and bear analysts, the quant validator, the portfolio manager, and the
orchestrator that sequences them while emitting timeline events.

`backend/agents/` · `backend/rag/` · `backend/research/` ·
`data/sources/sec_edgar.py` · `data/pipelines/edgar.py`

## Cecile — product, portfolio construction, integration

The product surface. Portfolio construction: alpha rank, risk filter,
correlation, position sizing — with every excluded candidate carrying the
reasons it was excluded, so a rejection is explainable rather than silent. The
API client and the research event stream. The Base44 dashboard built against the
published OpenAPI spec. Deployment, CORS, and the demo workflow.

`backend/portfolio/` · `backend/api/routes/portfolio.py` ·
`backend/api/deps.py` · `frontend/lib/` · the Base44 app

## Shared, by agreement

`data/schemas/` — the frozen data contract. Twelve pydantic models written as a
group in one sitting before any lane started. Changed only by agreement.

## Scale

| | |
|---|---|
| Python | 14,880 lines |
| C++ | 1,527 lines |
| TypeScript | 666 lines |
| Tests | 233 passing |
| Commits | 56 |
| Files with content | 151 of 244 |
| Budget | under $40 |
