# Autonomous Quant Research - IgnitionHacks

## Project: AI Hedge Fund / Autonomous Alpha Researcher

### One-line pitch

> Give the AI an investment thesis, and it autonomously discovers, researches, tests, challenges, and ranks trades.

Instead of asking:

> "Will NVDA go up?"

the user asks:

> **"Find companies benefiting from the AI data-center buildout that the market may be underpricing."**

The system does the rest.

---

# 1. Core Pipeline

```text
User Investment Thesis
        ↓
   AI Research Agent
        ↓
 ┌──────────────────────┐
 │ SEC filings          │
 │ Earnings transcripts │
 │ News                 │
 │ Market data          │
 │ Alternative data     │
 └──────────────────────┘
        ↓
Event / Signal Extraction
        ↓
Quant Factor Engine
        ↓
Alpha Model
        ↓
Portfolio Construction
        ↓
Realistic Backtester
        ↓
Risk Analysis
        ↓
   Bull / Bear Debate
        ↓
     FINAL TRADE
```

---

# 2. Killer Feature: Autonomous Discovery

The user does **not** select a stock.

### Example

**User:**

> Find mispriced beneficiaries of increasing AI infrastructure spending.

### AI:

**Scanning the universe...**

↓

Finds **Company A**

AI discovers:

- Capex guidance increased
- Hyperscaler spending accelerating
- Supplier orders increasing
- Management commentary changed
- Analyst estimates have not fully adjusted

↓

### Quant model

```text
Fundamental signal       +2.1
Earnings revision        +1.7
News sentiment           +1.2
Momentum                 +0.8
Valuation                -0.3
Risk                     -0.5
────────────────────────────
Composite Alpha          +5.0
```

↓

### Backtester

> Historical instances of this signal produced +14.7% annualized excess return, Sharpe 1.31, max drawdown 8.4%.

↓

### AI conclusion

**LONG**

Confidence: **82%**

Expected alpha: **+9.8%**

*Numbers above are illustrative for the demo; the actual system must calculate them from its data.*

---

# 3. Audit Trail

The AI should never simply say:

> "AI says BUY."

Instead, show exactly why.

### Catalyst #1

> Company raised data-center capex guidance.

**Source:** SEC filing / earnings release

### Catalyst #2

> Major customers increased infrastructure spending.

**Source:** earnings calls

### Catalyst #3

> Supplier lead times are increasing.

**Source:** industry / company data

Then:

> **"These events historically preceded positive earnings revisions."**

The system should show the underlying statistical evidence and link each important claim to its source.

This is a major differentiator from a generic LLM stock screener.

---

# 4. Bull vs Bear Debate

Use multiple agents to challenge the thesis.

```text
             RESEARCH AGENT
                  │
        ┌─────────┴─────────┐
        ↓                   ↓
   🐂 BULL CASE         🐻 BEAR CASE
        │                   │
        └─────────┬─────────┘
                  ↓
          QUANT VALIDATOR
                  ↓
          PORTFOLIO MANAGER
                  ↓
             TRADE IDEA
```

### 🐂 Bull Analyst

Builds the strongest long thesis.

### 🐻 Bear Analyst

Attempts to destroy it:

- valuation too high
- catalyst already priced in
- weak margins
- deteriorating fundamentals
- crowded trade
- historical false positives

### 📊 Quant Validator

Tests whether the thesis survives historical data.

### 🧠 Portfolio Manager

Combines:

- fundamental evidence
- quantitative evidence
- risk
- valuation
- bull/bear arguments

Then produces the final trade idea.

---

# 5. Team Architecture


**Matt - Quant / Market Infrastructure**
Core trading engine
Market-data pipeline
C++ quantitative components
Backtesting engine
Execution simulation
Order-book / microstructure
Transaction costs & slippage
Portfolio mechanics
Risk engine
Realistic execution using your Nasdaq ITCH infrastructure
Integrate the quantitative engine with the rest of the system

**Quant / Data / ML - Nalin**
Core research engine
Factor construction
Signal generation
Statistical testing
Feature engineering
Alpha weighting
Portfolio optimization
Sharpe / Sortino / drawdown
Regime detection
Cross-sectional analysis
Backtest validation
Research datasets and pipelines

**ML / AI - Zain**
Autonomous Research Agent
Core intelligence
LLM orchestration
Research agent
Earnings/filing analysis
News/NLP
RAG + embeddings
VLM where useful
Agentic tool calling
Automated hypothesis generation
Agent evaluation
Model/inference optimization

**Product + Integration - Cecile**
Instead of making #4 a full-stack developer:
Use Base44 to rapidly build the dashboard/UI
Connect the frontend to your APIs
Build the demo workflow
Make the system visually impressive
Integrate the quant + ML components
Handle deployment
Polish the final presentation

---

# 6. Technology Stack

## Frontend (Base44) - Cecile

**Next.js + React + TypeScript**

Use for:

- Investment thesis input
- Opportunity scanner
- Research timeline
- Bull/bear debate
- Quant metrics
- Portfolio view
- Interactive charts

### Visualization - Cecile + Matt + Nalin

**Plotly.js** or **Recharts**

Show:

- Equity curves
- Drawdowns
- Factor contributions
- Signal strength
- Risk
- Historical event studies

---

## Backend - Everyone

**Python + FastAPI**

Use Python for:

- Agent orchestration
- Data processing
- Quant research
- ML
- Backtesting orchestration
- API layer

FastAPI exposes the system to the frontend.

---

## Quant Engine - Nalin + Matt

**Python + NumPy + pandas + SciPy**

For:

- Factors
- Statistics
- Event studies
- Signal generation
- Portfolio construction
- Risk analytics

### Optional C++ - Matt

Use **C++** where it provides meaningful performance or demonstrates your existing expertise:

- Fast backtesting components
- Execution simulation
- Order-book reconstruction
- Slippage / queue-position simulation
- Monte Carlo / computationally intensive calculations

Connect C++ to Python with **pybind11**.

Do **not** force C++ into every component. Use it where it makes the system technically impressive.

---

# 7. AI / Agent Layer - Zain, Cecile

## Primary LLM

Use a strong API model for:

- Thesis decomposition
- Research planning
- Filing interpretation
- Bull/bear reasoning
- Final synthesis

Possible providers:

- OpenAI
- Google Gemini
- Cohere

For a $40 budget, choose **one primary model provider** rather than paying for several.

## Agent Framework - Cecile

Keep the architecture simple.

Possible options:

- LangGraph
- Lightweight custom Python agent orchestration

For a hackathon, a custom orchestration layer may be easier to control and debug.

---

# 8. Retrieval / RAG - Zain, Cecile

Use a retrieval system so the AI can ground claims in actual documents.

### Components - Zain, Cecile

**Embeddings**

- OpenAI embeddings or another low-cost embedding model

**Vector database**

Use a free/local option:

- FAISS
- Chroma

### Retrieval sources

- SEC filings
- 10-K
- 10-Q
- 8-K
- Earnings releases
- Earnings transcripts
- Company presentations
- Relevant news

The system retrieves relevant passages before the LLM generates its conclusion.

---

# 9. Market Data - Matt, Nalin

## Primary hackathon data approach

Use a combination of:

### Free / low-cost

**Yahoo Finance**

Useful for:

- Historical OHLCV
- Basic company information
- Prototyping

### SEC EDGAR

Free source for:

- 10-K
- 10-Q
- 8-K
- Company filings

### Company / earnings sources

Use publicly accessible earnings releases and transcripts where permitted.

### Optional paid API

If required, use a low-cost market-data provider such as:

- Polygon/Massive
- Alpha Vantage
- Finnhub
- Financial Modeling Prep

The project does **not** require expensive Bloomberg access.

---

# 10. Data Universe - Matt, Nalin

Do not attempt to scan every security in the world.

For the hackathon MVP:

### Target universe

**500-1,500 liquid US equities**

Filter using:

- Market capitalization
- Average daily volume
- Listing status
- Data availability

This is enough to make the system feel like it is scanning the market while keeping costs and compute manageable.

---

# 11. Alpha Model - Nalin, Matt

Combine multiple signals.

### Fundamental

- Revenue growth
- EPS revisions
- Margin expansion
- Free cash flow
- ROIC
- Debt
- Valuation

### Market

- Momentum
- Volatility
- Volume
- Relative strength

### Event

- Earnings surprises
- Guidance changes
- Insider activity where data is available
- Corporate announcements

### NLP

- Management sentiment
- Guidance language changes
- Topic changes
- Earnings-call sentiment
- News sentiment

### Composite

```text
Alpha Score =
    Fundamental
  + Earnings Revision
  + Event Signal
  + NLP Signal
  + Momentum
  - Risk
  - Valuation Penalty
```

Weights should be learned, optimized, or explicitly validated rather than invented by the LLM.

---

# 12. Backtesting - Matt, Nalin

This is one of the most important parts.

Avoid:

> "Our strategy made 200%."

Instead build a realistic backtest.

Include:

- Train/test separation
- Walk-forward validation
- Transaction costs
- Slippage
- Position limits
- Liquidity constraints
- Benchmark comparison
- Maximum drawdown
- Sharpe ratio
- Sortino ratio
- Turnover
- Win rate

### Important

Prevent:

- Look-ahead bias
- Survivorship bias where possible
- Data leakage
- Overfitting
- Using future information in historical decisions

The LLM should **not** be allowed to see future prices when generating historical signals.

---

# 13. Portfolio Construction, Cecile

After ranking opportunities:

```text
Candidate Universe
       ↓
Alpha Ranking
       ↓
Risk Filtering
       ↓
Correlation Analysis
       ↓
Position Sizing
       ↓
Portfolio
```

Potential methodology:

- Equal risk contribution
- Volatility scaling
- Mean-variance optimization
- Simple constrained optimizer

For the hackathon, a transparent methodology is preferable to an unnecessarily complicated optimizer.

---

# 14. Risk Engine, Matt, Nalin

Show:

- Beta
- Volatility
- Maximum drawdown
- VaR / CVaR
- Sector exposure
- Concentration
- Correlation
- Factor exposure
- Liquidity

Example:

```text
Expected Return       +18.2%
Expected Alpha         +9.4%
Volatility             21.3%
Sharpe                  1.28
Max Drawdown            9.7%
Beta                    0.94
Position Size           4.2%
```

---

# 15. Dashboard (Base 44), Cecile

## Main screen

### 🔎 Find Opportunities

Input:

> "Find mispriced beneficiaries of AI infrastructure spending."

Button:

**RUN AUTONOMOUS RESEARCH**

---

## Opportunity screen

```text
AI Infrastructure Theme

1,247 companies scanned

↓ 183 passed liquidity filter
↓ 74 passed fundamental filter
↓ 21 passed event filter
↓ 7 high-alpha candidates
↓ 3 survived quant validation

TOP IDEA

Company XYZ
LONG

Alpha Score       8.7/10
Confidence        82%
Expected Alpha    +9.8%
Sharpe             1.31
Risk               Medium
```

---

# 16. Research Timeline, Zain, Cecile 

Show the AI's work live:

```text
x Parsed investment thesis
x Defined screening criteria
x Scanned 1,247 companies
x Identified 7 candidates
x Retrieved 10-K / 10-Q filings
x Analyzed earnings transcripts
x Extracted catalysts
x Generated bull thesis
x Generated bear thesis
x Ran historical event study
x Backtested signal
x Calculated portfolio risk
x Generated final recommendation
```

This makes the demo feel autonomous.

---

# 17. Recommended Demo, Everyone

Use a thesis with a clear narrative.

### Example

> **"Find companies benefiting from accelerating AI data-center spending that the market may be underpricing."**

Then run the entire system live.

### Demo sequence

**0-10 sec**

Enter thesis.

**10-25 sec**

AI decomposes thesis into measurable criteria.

**25-40 sec**

Universe scan.

**40-55 sec**

Top candidates appear.

**55-75 sec**

AI researches the top candidate.

**75-90 sec**

Bull vs Bear debate.

**90-110 sec**

Quant validator displays historical evidence.

**110-130 sec**

Backtest and risk analysis.

**130-150 sec**

Final trade idea.

The judges should be able to understand the entire system in approximately 2-3 minutes.

---

# 18. What NOT to Build

## ❌ Generic AI stock predictor

Too common.

## ❌ ChatGPT financial advisor

Too easy to replicate.

## ❌ Simple sentiment analyzer

Low differentiation.

## ❌ LSTM predicting SPY

Does not demonstrate the team's full capabilities.

## ❌ Fully autonomous live trading

Unnecessary complexity and risk.

Instead build:

> **Autonomous research + quantitative validation + paper trading/backtesting.**

---

# 19. $40 Budget

## Target budget: **<= $40**

The goal is to keep almost everything free and spend the majority of the budget on LLM/API usage.

### Proposed budget

| Component | Technology | Cost |
|---|---|---:|
| LLM API | OpenAI | **$20** |
| Market data | Free APIs / Yahoo Finance | **$0** |
| SEC filings | SEC EDGAR | **$0** |
| Embeddings | Local / low-cost API | **$0-3** |
| Vector DB | FAISS / Chroma | **$0** |
| Backend | FastAPI | **$0** |
| Frontend | Next.js | **$0** |
| Hosting | Free tier / local demo | **$0** |
| Database | Supabase free tier | **$0** |
| Quant libraries | NumPy / pandas / SciPy | **$0** |
| C++ | GCC / Clang | **$0** |
| Charts | Plotly / Recharts | **$0** |
| GitHub | GitHub | **$0** |
| Buffer | Extra API usage | **$17-20** |
| **TOTAL** | | **<= $40** |

### Spending strategy

Do **not** spend the $40 upfront.

Start with free/local infrastructure.

Only pay for:

1. LLM inference
2. Any market-data endpoint that becomes necessary
3. Extra API usage near the final demo

A realistic target is **$10-25 total**, leaving the rest as emergency budget.

---

# 20. Deployment

For the hackathon, prioritize reliability over production infrastructure.

### Local development

```text
Next.js
   ↓
FastAPI
   ↓
Python Agents
   ↓
Quant Engine
   ↓
Data + RAG
```

### Optional deployment

Frontend:

- Vercel

Backend:

- Render / Railway / similar free or low-cost tier

Database:

- Supabase

If deployment becomes unreliable, run the full system locally and use a polished demo environment.

---

# 21. GitHub Architecture

```text
autonomous-alpha/
│
├── frontend/
│   ├── app/
│   ├── components/
│   └── charts/
│
├── backend/
│   ├── api/
│   ├── agents/
│   ├── research/
│   ├── rag/
│   └── portfolio/
│
├── quant/
│   ├── factors/
│   ├── signals/
│   ├── backtest/
│   ├── risk/
│   └── optimization/
│
├── cpp/
│   ├── orderbook/
│   ├── execution/
│   └── simulation/
│
├── data/
│   ├── schemas/
│   └── pipelines/
│
├── tests/
│
├── notebooks/
│
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

# 22. MVP vs Stretch Goals

## MVP - Must Have

- Natural-language investment thesis
- Universe scanning
- Candidate ranking
- SEC/document retrieval
- LLM research agent
- Bull/bear analysis
- Quant signals
- Backtest
- Risk metrics
- Final trade recommendation
- Audit trail
- Polished dashboard

## Stretch Goal #1

**Autonomous hypothesis generation**

Instead of only testing the user's thesis, the AI discovers its own hypotheses.

## Stretch Goal #2

**Event-study engine**

Automatically identifies historical situations similar to the current setup.

## Stretch Goal #3

**Execution simulator**

Use your C++ infrastructure for:

- Slippage
- Market impact
- Order-book execution
- Queue position

## Stretch Goal #4

**Paper portfolio**

Track AI-generated trades over time.

## Stretch Goal #5

**Self-improving research loop**

After a strategy fails validation:

```text
Failed hypothesis
       ↓
AI analyzes failure
       ↓
Modifies signal
       ↓
Re-tests
       ↓
Accept / Reject
```

This would push the project toward a genuinely autonomous research system.

---

# 23. Competitive Differentiation

The project is not:

> **LLM + stocks**

It is:

> **LLM + financial research + quantitative signal discovery + adversarial reasoning + realistic backtesting + portfolio construction + explainability**

The key differentiator is that the LLM is not the source of truth.

The architecture is:

```text
LLM
 ↓
Hypothesis
 ↓
Evidence
 ↓
Quantification
 ↓
Historical Validation
 ↓
Risk
 ↓
Decision
```

The AI can **propose** an idea.

The quantitative system determines whether the idea actually survives the evidence.

---

# 24. Winning Demo Statement

> **"Most AI finance tools answer questions about stocks. Ours autonomously finds questions worth asking."**

Then:

> **"Give it an investment thesis, and it scans the market, finds candidates, researches filings and earnings, builds bull and bear cases, quantitatively tests the hypothesis, backtests the signal, measures risk, and produces an auditable trade idea."**

That is the core story.

---

# 25. Success Criteria

The project should feel like a **mini autonomous quant research platform**, not a chatbot.

A judge should be able to see:

### AI

The system reasons over unstructured financial information.

### Quant

The system converts qualitative ideas into measurable signals.

### Engineering

The system handles real financial data and realistic backtesting.

### Product

The entire workflow is understandable through a polished UI.

### Explainability

Every major recommendation can be traced back to evidence and quantitative validation.

### Team differentiation

Each team member contributes a genuinely different technical layer.

---

# Final Technology Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js, React, TypeScript |
| Backend | Python, FastAPI |
| AI | OpenAI / Gemini API |
| Agent orchestration | LangGraph or custom Python |
| RAG | FAISS / Chroma |
| Embeddings | Local / low-cost API |
| NLP | LLM + Python NLP tooling |
| Quant | NumPy, pandas, SciPy |
| Backtesting | Custom event-driven engine |
| Optimization | SciPy / cvxpy |
| Risk | Python quantitative stack |
| C++ | Execution / order book / simulation |
| Python ↔ C++ | pybind11 |
| Market data | Yahoo Finance + optional low-cost API |
| Fundamental data | SEC EDGAR |
| Database | PostgreSQL / Supabase |
| Charts | Plotly / Recharts |
| Frontend hosting | Vercel |
| Backend hosting | Render / Railway |
| Version control | GitHub |
| Containerization | Docker |
| Budget | **<= $40** |

## The goal

Build something that makes the judge think:

> **"This isn't an AI stock picker. They built an autonomous quant research workflow."**
