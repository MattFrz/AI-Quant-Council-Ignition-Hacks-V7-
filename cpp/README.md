# Execution engine

A limit order book, a Nasdaq ITCH 5.0 replay parser and an execution simulator,
in C++17, exposed to Python through pybind11 as `aqc_exec`.

It answers one question: what does an order actually cost? A backtest that
assumes you trade at the price on the screen is measuring a market that does not
exist. Walk the depth instead and the cost grows with size, large orders do not
complete, and passive orders sometimes do not fill at all.

Run it from the app's Execution page, or from Python directly.

## Build

Needs a C++17 compiler.

```bash
python cpp/bindings/setup.py build_ext --inplace
```

On Windows that means MSVC Build Tools:

```bash
winget install --id Microsoft.VisualStudio.2022.BuildTools --override "--quiet --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```

The extension is optional everywhere it is used. `quant/backtest/slippage.py`
falls back to an analytic model and logs that it did; `/api/execution` returns
503 rather than inventing fills. A machine without a compiler still runs the
whole project.

## Use

```python
import aqc_exec as ax

book = ax.OrderBook()
ref = 1
for i in range(5):
    book.add(ref, ax.Side.BUY, 100.00 - i * 0.01, 1000); ref += 1
    book.add(ref, ax.Side.SELL, 100.01 + i * 0.01, 1000); ref += 1

sim = ax.ExecutionSimulator()
print(sim.market_order(book, ax.Side.BUY, 2500))
# <ExecutionResult market filled=2500/2500 slippage=1.30bps>
```

## Three behaviours worth knowing

**An ITCH replace sends an order to the back of the queue.** Modelling it as an
in-place edit hands every replaced order a queue position it did not earn.

**Passive orders do not fill because you posted them.** They fill when the
queue in front of them trades. Five hundred shares behind a thousand-share queue
fills nothing, and a simulator that fills it anyway is manufacturing free money.

**Getting picked off is charged as a cost.** If the price moves through your
resting limit you did trade, but only because someone knew something you did
not. Booking that at your limit price is how passive backtests print alpha that
never existed.

## Tests

```bash
cmake -S cpp -B cpp/build && cmake --build cpp/build
./cpp/build/test_orderbook
```

No framework, one binary, non-zero exit on failure.

## Layout

| Path | What it holds |
|---|---|
| `include/` | Shared types: `Side`, `Quote`, `ExecutionResult` |
| `orderbook/` | Price-time-priority book and the ITCH 5.0 parser |
| `execution/` | Market, sliced and passive fill simulation |
| `bindings/` | pybind11 module and its build script |

## Not wired into the backtest

The backtest charges slippage with a participation-rate model in Python. This
engine is more faithful and it is not what produces the performance figures
elsewhere in the project. Saying otherwise would misattribute every one of them.
