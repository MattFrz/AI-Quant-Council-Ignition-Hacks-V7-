from __future__ import annotations

# Tool schemas exposed to the LLM via LLMClient.complete_with_tools (C1).
# Keep this surface small (2-3 functions) - only what the debate agents
# genuinely need to call, not every quant function that exists.

TOOLS = [
    {
        "type": "function",
        "name": "run_backtest",
        "description": (
            "Runs a backtest for a given ticker/universe and returns real "
            "performance metrics (Sharpe, drawdown, win rate). Use this "
            "instead of estimating performance yourself."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "as_of": {"type": "string", "description": "ISO date, e.g. 2024-06-01"},
            },
            "required": ["ticker", "as_of"],
        },
    },
    {
        "type": "function",
        "name": "get_risk_metrics",
        "description": "Returns real risk metrics (beta, VaR, volatility) for a ticker.",
        "parameters": {
            "type": "object",
            "properties": {"ticker": {"type": "string"}},
            "required": ["ticker"],
        },
    },
]


def build_tool_impls(quant_validator) -> dict:
    """
    Maps tool name -> actual python callable, passed into
    LLMClient.complete_with_tools(tool_impls=...). Binds against a live
    QuantValidator instance so calls go through the real quant code, never
    a fabricated number (same DO NOT FAKE rule as C20).
    """
    return {
        "run_backtest": lambda ticker, as_of: quant_validator.run_backtest_for_ticker(ticker, as_of),
        "get_risk_metrics": lambda ticker: quant_validator.get_risk_metrics_for_ticker(ticker),
    }