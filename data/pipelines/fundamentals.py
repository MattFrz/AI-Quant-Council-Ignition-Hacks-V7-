"""Point-in-time fundamentals from EDGAR companyfacts. Step A-lane / unblocks B6.

The only free source that carries both the fiscal period (`end`) and the date
the figure became public (`filed`). yfinance gives the first and not the second,
which is why it cannot support an honest fundamental factor.

Three traps, each handled below:

  Restatements   A later 10-K re-reports old quarters. We keep the FIRST filing
                 of each period, because that is the number the market traded on.
  Missing Q4     Most issuers report Q1-Q3 quarterly and fold Q4 into the annual
                 figure. Left alone, "four quarters ago" silently becomes five.
                 Q4 is derived as FY minus Q1+Q2+Q3.
  Units          Facts nest under units (USD, USD/shares, shares). Reading the
                 wrong one returns share counts where dollars were wanted.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from backend.config import settings
from backend.core.logging import get_logger
from data.sources.sec_edgar import EdgarClient

log = get_logger(__name__)

FUNDAMENTALS_FILE = "fundamentals.parquet"
TAX_RATE = 0.21

QUARTER_DAYS = (80, 100)
ANNUAL_DAYS = (340, 380)

# field -> (candidate us-gaap tags in priority order, unit key, duration|instant)
CONCEPTS: Dict[str, Tuple[List[str], str, str]] = {
    "revenue": (["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues",
                 "SalesRevenueNet", "RevenueFromContractWithCustomerIncludingAssessedTax"],
                "USD", "duration"),
    "gross_profit": (["GrossProfit"], "USD", "duration"),
    "operating_income": (["OperatingIncomeLoss"], "USD", "duration"),
    "net_income": (["NetIncomeLoss", "ProfitLoss"], "USD", "duration"),
    "eps_diluted": (["EarningsPerShareDiluted", "IncomeLossFromContinuingOperationsPerDilutedShare"],
                    "USD/shares", "duration"),
    "operating_cash_flow": (["NetCashProvidedByUsedInOperatingActivities",
                             "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
                            "USD", "duration"),
    "capex": (["PaymentsToAcquirePropertyPlantAndEquipment",
               "PaymentsToAcquireProductiveAssets"], "USD", "duration"),
    "shares_diluted": (["WeightedAverageNumberOfDilutedSharesOutstanding"], "shares", "duration"),
    "total_debt": (["LongTermDebt", "LongTermDebtNoncurrent", "DebtLongtermAndShorttermCombinedAmount"],
                   "USD", "instant"),
    "cash_and_equivalents": (["CashAndCashEquivalentsAtCarryingValue",
                              "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
                             "USD", "instant"),
    "total_equity": (["StockholdersEquity",
                      "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
                     "USD", "instant"),
}


def _raw_facts(payload: dict, tags: Sequence[str], unit: str) -> pd.DataFrame:
    """Pull every reported fact for the first tag that has data."""
    us_gaap = payload.get("facts", {}).get("us-gaap", {})
    for tag in tags:
        block = us_gaap.get(tag, {}).get("units", {}).get(unit)
        if not block:
            continue
        frame = pd.DataFrame(block)
        if frame.empty or "end" not in frame or "filed" not in frame:
            continue
        frame["end"] = pd.to_datetime(frame["end"], errors="coerce")
        frame["filed"] = pd.to_datetime(frame["filed"], errors="coerce")
        if "start" in frame:
            frame["start"] = pd.to_datetime(frame["start"], errors="coerce")
        frame["tag"] = tag
        return frame.dropna(subset=["end", "filed", "val"])
    return pd.DataFrame()


def _first_filed(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per period: the original filing, not any later restatement."""
    if frame.empty:
        return frame
    idx = frame.sort_values("filed").groupby("end", as_index=False).head(1).index
    return frame.loc[idx].sort_values("end")


def _quarterly(frame: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Quarterly observations, with Q4 derived from the annual figure."""
    if frame.empty:
        return frame

    if kind == "instant":
        return _first_filed(frame)[["end", "val", "filed", "form"]]

    span = (frame["end"] - frame["start"]).dt.days
    quarters = _first_filed(frame[span.between(*QUARTER_DAYS)])
    annuals = _first_filed(frame[span.between(*ANNUAL_DAYS)])

    derived = []
    for _, year in annuals.iterrows():
        window = quarters[(quarters["end"] > year["start"]) & (quarters["end"] <= year["end"])]
        if len(window) != 3:
            continue  # Q4 only derivable when exactly three quarters are present
        derived.append({
            "end": year["end"],
            "val": float(year["val"]) - float(window["val"].sum()),
            "filed": year["filed"],
            "form": year.get("form", "10-K"),
        })

    out = pd.concat(
        [quarters[["end", "val", "filed", "form"]], pd.DataFrame(derived)],
        ignore_index=True,
    ) if derived else quarters[["end", "val", "filed", "form"]]

    return out.sort_values("end").drop_duplicates(subset="end", keep="first")


def fundamentals_for_ticker(ticker: str, client: EdgarClient) -> pd.DataFrame:
    """One ticker's point-in-time quarterly history."""
    try:
        payload = client.get_company_facts(ticker)
    except Exception as exc:  # noqa: BLE001 - one bad ticker must not stop the run
        log.warning("companyfacts failed for %s: %s", ticker, str(exc)[:90])
        return pd.DataFrame()

    series: Dict[str, pd.DataFrame] = {}
    for field, (tags, unit, kind) in CONCEPTS.items():
        got = _quarterly(_raw_facts(payload, tags, unit), kind)
        if not got.empty:
            series[field] = got.set_index("end")

    if "revenue" not in series and "net_income" not in series:
        log.warning("%s: no usable income-statement facts", ticker)
        return pd.DataFrame()

    periods = sorted(set().union(*(set(s.index) for s in series.values())))
    rows = []
    for period_end in periods:
        row: Dict[str, object] = {"ticker": ticker, "period_end": period_end}
        filed_dates = []
        for field, frame in series.items():
            if period_end not in frame.index:
                continue
            row[field] = float(frame.at[period_end, "val"])
            filed_dates.append(frame.at[period_end, "filed"])
        if not filed_dates:
            continue
        # The row is only fully public once its LAST field was published.
        row["report_date"] = max(filed_dates)
        row["fiscal_period"] = f"Q{((period_end.month - 1) // 3) + 1}"
        rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    # Derived fields, only where their inputs exist. No filling in blanks.
    rev = frame.get("revenue")
    if rev is not None:
        denom = rev.abs().where(rev.abs() > 0)
        if "gross_profit" in frame:
            frame["gross_margin"] = frame["gross_profit"] / denom
        if "operating_income" in frame:
            frame["operating_margin"] = frame["operating_income"] / denom
    if {"operating_cash_flow", "capex"} <= set(frame.columns):
        frame["free_cash_flow"] = frame["operating_cash_flow"] - frame["capex"]
    if {"operating_income", "total_debt", "total_equity"} <= set(frame.columns):
        invested = frame["total_debt"] + frame["total_equity"] - frame.get(
            "cash_and_equivalents", 0.0)
        invested = invested.where(invested > 0)
        frame["roic"] = (frame["operating_income"] * 4 * (1 - TAX_RATE)) / invested

    return frame.sort_values("period_end")


def build_fundamentals(
    tickers: Sequence[str],
    client: Optional[EdgarClient] = None,
    write: bool = True,
) -> pd.DataFrame:
    """Fetch every ticker, assemble the long frame, persist it."""
    client = client or EdgarClient(settings.require_sec_user_agent())

    frames = []
    for i, ticker in enumerate(tickers, 1):
        got = fundamentals_for_ticker(ticker, client)
        if not got.empty:
            frames.append(got)
        if i % 20 == 0:
            log.info("fundamentals: %d/%d tickers", i, len(tickers))

    if not frames:
        raise RuntimeError("EDGAR returned no usable fundamentals for any ticker.")

    panel = pd.concat(frames, ignore_index=True)
    panel = panel[panel["report_date"] >= panel["period_end"]]  # schema invariant

    if write:
        out = settings.cache_path / FUNDAMENTALS_FILE
        panel.to_parquet(out, index=False)
        log.info("fundamentals: %d rows, %d tickers -> %s",
                 len(panel), panel["ticker"].nunique(), out)
    return panel


def load_fundamentals(path: Optional[str] = None) -> pd.DataFrame:
    """Read the cached frame. Shaped for Panel.fundamentals."""
    target = path or (settings.cache_path / FUNDAMENTALS_FILE)
    frame = pd.read_parquet(target)
    frame["period_end"] = pd.to_datetime(frame["period_end"])
    frame["report_date"] = pd.to_datetime(frame["report_date"])
    return frame
