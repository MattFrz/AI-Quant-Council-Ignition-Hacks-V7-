# AI Quant Council

### [Try it: ai-quant-council.vercel.app](https://ai-quant-council.vercel.app)

**Ask any AI chatbot whether a stock is a good buy and it will hand you a Sharpe
ratio. That number is invented.** We built the layer the AI-finance category is
missing: a quantitative engine that tests the AI's idea against historical
evidence and is allowed to say no.

> "Find companies benefiting from the AI data-center buildout that the market
> may be underpricing."

Give it a thesis in plain English. It decomposes that into measurable criteria,
scans the universe, retrieves the actual SEC filings, extracts catalysts with
verbatim quotes and clickable sources, argues both sides, then runs the whole
thing through a backtest that is allowed to reject it.

**The LLM proposes. The quantitative engine decides.** The model never emits a
Sharpe ratio, an alpha score, a factor weight or a VaR.

Our own strategy returns **+6.0% annualised excess against SPY at Sharpe 1.23**,
after commission and participation-rate slippage. The event study on the
catalysts we extract typically finds **too few independent events to be
statistically significant**, and reports that rather than a number. We say so
because it is the entire product: a tool that only ever agrees with you is the
thing this replaces.

| | |
|---|---|
| [Live app](https://ai-quant-council.vercel.app) | Eighteen pre-computed theses, no setup |
| [Submission description](docs/submission.md) | What it is and why it matters |
| [Architecture](docs/architecture.md) | The pipeline, the contract, the look-ahead defence |
| [Deployment](docs/deploy.md) | How the hosted version is put together |
| [Who built what](docs/team_ownership.md) | Four lanes, four owners |

**IgnitionHacks V7 - Fintech**, Matt, Nalin, Zain, Cecile

---

## Setup

Requires Python 3.9+ (3.11+ recommended) and Node 18+.

```bash
bash scripts/setup_env.sh
```

Then fill in your keys in `.env` - the LLM key, and `SEC_USER_AGENT` with a real
name and email (EDGAR rejects requests without one).

`.env` is gitignored and has never been committed. Only `.env.example` is
tracked, and it carries placeholders. Keep it that way: a key in git history
stays there after you delete the line.

## Run

Backend:

```bash
uvicorn backend.main:app --reload
```

Frontend:

```bash
cd frontend && npm run dev
```

Check <http://localhost:8000/health> and <http://localhost:3000>.

## Verify the data contract

Every lane's output has to fit `TradeIdea`. This checks the fixture still does:

```bash
python scripts/verify_contract.py
```

---

## Layout

| Path | Owner | What lives here |
|---|---|---|
| `data/schemas/` | **shared** | The data contract. Change by agreement only. |
| `data/sources/`, `data/pipelines/` | Matt | Yahoo, EDGAR, price and fundamental pipelines |
| `quant/universe/`, `quant/backtest/`, `quant/risk/` | Matt | Universe, event-driven backtester, risk engine |
| `quant/factors/`, `quant/signals/`, `quant/alpha/`, `quant/eventstudy/` | Nalin | Factor construction, alpha model, statistical testing |
| `backend/agents/`, `backend/research/`, `backend/rag/` | Zain | Research agent, bull/bear debate, retrieval |
| `frontend/`, `backend/portfolio/` | Cecile | Dashboard, portfolio construction, demo |
| `cpp/` | Matt | Order book, execution simulation (stretch) |
| `backend/api/` | **shared** | HTTP layer |

## Ground rules

1. **Cache everything to disk.** No demo path may depend on a live API call.
2. **As-of everything.** No fundamental row before its `report_date`, no filing
   before its `filed_date`. Both are enforced in the schemas.
3. **The LLM proposes, the quant decides.** The model never produces a Sharpe
   ratio, an alpha score or a factor weight.
4. **Every claim carries a URL.** A `Catalyst` without a clickable `source_url`
   will not validate.
5. **One LLM provider**, cost logged per call, against a $40 ceiling.
6. **Report the real numbers.** Whatever the backtest returns is what the slide
   says.

## Thesis syntax

Plain English. Three things the system reads beyond the topic:

| You write | It does |
|---|---|
| `Is NVIDIA underpriced given AI capex?` | researches NVDA specifically |
| `...AI data-center spending that isn't Vertiv` | excludes VRT from candidates |
| `Find companies benefiting from...` | open discovery, ranks the universe |

Naming a company steers which one gets researched. It does not change the score:
a directed name still receives whatever alpha rank the quant model gives it.

## Demo cache

A full run takes 60 to 90 seconds. Warm it first and it replays in under a
second, with the same data.

```bash
python -m backend.core.cache warm
python -m backend.core.cache            # list what is warmed
```

**After any backend change that alters what a run produces, clear and re-warm.**
A cached run stores its own event list, so a stale entry replays older output
and looks like a bug that only affects some prompts.

```bash
python -m backend.core.cache clear && python -m backend.core.cache warm
```

## Hosted version

[ai-quant-council.vercel.app](https://ai-quant-council.vercel.app) runs the same
code with nothing on a laptop: the frontend on Vercel, the backend in a
container on Render with the index and warmed results on a persistent disk.

Four pages: **Research** runs the thesis and shows the live timeline, evidence,
debate, backtest and risk. **Opportunities** shows the universe funnel and the
ranked shortlist. **Portfolio** sizes the book, and states which names the risk
layer refused and why. **Execution** drives the C++ order book directly: size an
order and watch what crossing the spread actually costs.

Two settings separate the hosted instance from a local checkout, both in
[docs/deploy.md](docs/deploy.md):

- `LIVE_RUNS_PER_HOUR` and `LIVE_RUNS_PER_DAY` bound how many uncached theses
  it will research, because an open URL with a live LLM key behind it bills the
  owner for every visitor who types something new. Warmed theses replay free and
  are never counted. (`PUBLIC_DEMO_MODE` is the stricter alternative: warmed
  theses only, zero spend, no live research at all.)
- `PINNED_AS_OF` freezes the date a run treats as today. The cache key includes
  that date, so without it every warmed entry expires at midnight and the next
  visitor pays for a full run.

Run it locally and neither applies: any thesis, researched for real, unlimited.

## Status

- [x] Phase 0 - foundation
- [x] Phase 1 - data contract frozen
- [x] Phase 2 - factors, agents, retrieval, backtester, risk engine
- [x] Phase 3 - integration, pipeline live end to end
- [x] Phase 4 - C++ order book and execution simulation
- [ ] Phase 5 - demo hardening

## Known limits

Stated here rather than left for a reader to discover.

- The retrieval corpus covers **17 companies**; the quant model scores **499**.
  Deep research only runs where filings exist to cite.
- The backtest runs on the **7 ranked candidates**, not the full universe.
- **No walk-forward split in the demo path.** `run_walk_forward` exists and is
  tested, but the live pipeline runs a single in-sample window.
- The **C++ execution simulator is built, tested and runnable from the app, but
  it is not wired into the backtest.** The Execution page calls it live through
  `/api/execution`, so its fills are real; the backtest still charges slippage
  with the Python participation-rate model.
- The universe is the **current** S&P 500, so backtests carry survivorship bias.
  `scripts/run_backtest.py` prints that warning itself.
- Event studies usually have **too few independent events** to reach
  significance, and report the count instead of a number.

**Not financial advice.** A research tool that produces auditable evidence, not
trade instructions.
