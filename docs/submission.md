# AI Quant Council

**Track: Fintech**

## The short version

Ask ChatGPT if a stock is a good buy and it will give you a Sharpe ratio. It
made that number up. Nothing in the system calculated it.

We built the part that actually checks. You give it an investment thesis, it
does the research, and then a quantitative engine tests whether the idea holds
up against historical data. If it doesn't hold up, the system says so.

## The problem

AI stock advice is everywhere now and most of it is confidently wrong. Ask a
language model for a risk adjusted return and you get one. It sounds specific,
it sounds authoritative, and it is fiction, because no part of the process ever
computed anything.

People act on this. They put money behind it.

The reason it happens isn't bad prompting. It's that the model is the only thing
in the loop, so there is nothing to check it against.

## What we built

You type a thesis in plain English:

> "Find companies benefiting from the AI data-center buildout that the market
> may be underpricing."

The system breaks that into measurable screening criteria, scans a universe of
US stocks, pulls the actual SEC filings for the ones that survive, and extracts
catalysts with real quotes and working links back to the source document. Two
agents then argue it out, one building the case for, one attacking it using the
quant results.

Then the quantitative engine scores it, backtests it with real trading costs,
measures the risk, and decides whether the idea survives.

The language model proposes. The quant engine decides.

That split is enforced in code, not in a prompt. The model cannot produce a
Sharpe ratio, an alpha score, a factor weight or a VaR. Those come out of 14,000
lines of Python and 1,500 lines of C++ that it has no access to. There is an
agent whose entire job is to call that engine and report what came back, and it
is not allowed to estimate anything itself.

## Three things a chatbot won't do

**Every claim has a source you can click.** A catalyst with no working URL fails
validation and never ships. Click one during the demo and you land on the real
10-Q. The quote is lifted word for word from the document. The model doesn't
write it.

**It rejects its own ideas.** Candidates that fail validation get thrown out of
the portfolio, and you get told why:

```
EXCLUDED  VRT           risk_band=high requires confidence >= 0.75, got 0.73
```

That is a real run. Vertiv scored 9.4 out of 10 on the alpha model and survived
the backtest, and the portfolio layer still refused to fund it: a
60%-volatility name needs stronger conviction than the evidence supported. The
system is allowed to say no to its own recommendation.

**It tells you when it doesn't know.** Our strategy returns 5.0% annualised
excess return over SPY at a Sharpe of 1.22, after commission and
participation-rate slippage. The event study we run over the extracted
catalysts usually finds too few independent events to reach significance, and
reports that count rather than a number. A Sharpe ratio cannot leave this system
without the evidence sitting next to it.

We are showing you a number that makes us look worse because that is the whole
point of the thing. A tool that always agrees with you is what we built this to
replace.

## The technical work

**Stopping look-ahead bias, structurally.** This is the bug that quietly ruins
backtests. It never throws an error, it just makes everything look better than
it was.

Factors get handed a dataset that has already been cut off at the decision date,
so a factor physically cannot read the future. It isn't in the object it
receives. Fundamentals join on the date a number became public, not the quarter
it covers. Apple reports 2007 figures that weren't filed until 2009, so joining
on the quarter would hand the backtest two years of hindsight. Restatements get
resolved back to the original filing, because that's the number people actually
traded on. Training and test periods are separated by a gap the length of the
return horizon. The document retriever refuses anything filed after the decision
date.

Every one of those has a test that appends future data and checks the signal
doesn't budge.

**Point in time fundamentals from SEC EDGAR.** 5,629 quarters across 79
companies going back to 2006, pulled from the XBRL companyfacts API. It's the
only free source that gives you both the fiscal period and the date it was
published. Q4 gets derived from the annual filing, because most companies fold
it in, and without that fix "four quarters ago" quietly becomes five.

**A scoreboard built before the factors.** Information coefficient, rank IC,
t stats, signal decay. We built the measuring stick first so every factor had to
justify itself with a number instead of an argument.

**C++ execution simulation.** A price level order book and a Nasdaq ITCH parser
wired in through pybind11, so slippage comes from actual order book mechanics
instead of a flat assumption. It sits behind a config flag with the Python
version still working, so a compiler that fails on the demo laptop gives you a
worse number rather than a broken demo.

**233 tests.** One of them plants a known effect into a fake market and checks
the engine finds it at the right size. That proves the measurement works even
when the alpha doesn't.

## Built with Base44

We froze the data contract in the first hour, as pydantic models. FastAPI
publishes that same contract as an OpenAPI spec, 14 endpoints and 36 typed
schemas, and the Base44 dashboard is built straight against it.

That's what made four people building at once actually work. The frontend stayed
in sync with a 14,000 line backend without anyone needing to coordinate, while
three of us built the engine underneath. Base44 wasn't just where the pages got
made. It's the reason the product surface could move as fast as the backend did.

## What's not finished

80 companies is too small a universe for this kind of model. Cross sectional
strategies need breadth and we didn't have it.

Out of sample IC drops about 65% from in sample.

We tested 18 factors and one came out significant at 5%. With 18 tests you'd
expect about one by pure chance, so we're not claiming it.

Long only, monthly rebalancing, no borrow costs or taxes.

Give us another week and we'd run the full S&P 500 over twenty years with
walk forward refitting instead of one split. That's the honest path from "the
engine works" to "the strategy works."

## Stack

Python, FastAPI, NumPy, pandas, SciPy, FAISS, pydantic, C++ with pybind11,
Base44, SEC EDGAR, Yahoo Finance. Under $40 total.

Matt, Nalin, Zain, Cecile. IgnitionHacks V7.
