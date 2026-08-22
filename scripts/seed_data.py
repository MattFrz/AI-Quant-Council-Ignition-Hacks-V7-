"""Pull everything the demo needs and dump it to disk. Step A4.

Run this ONCE, early - by hour 4 of the hackathon. After it completes, set
OFFLINE_MODE=true in .env and the whole system reads from cache. Hackathon wifi
and Yahoo rate limits have killed more demos than bad code.

    python scripts/seed_data.py                 # default S&P-ish universe
    python scripts/seed_data.py --tickers AAPL,MSFT,NVDA
    python scripts/seed_data.py --refresh       # force re-download
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from backend.config import settings  # noqa: E402
from backend.core.logging import get_logger  # noqa: E402
from data.pipelines.prices import (  # noqa: E402
    build_price_panel,
    build_profiles,
    average_dollar_volume,
    to_wide,
)
from quant.universe.builder import build_universe  # noqa: E402

log = get_logger("seed_data")

#: Wikipedia's S&P 500 table. Free, no key, and a reasonable liquid universe.
SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

#: Used when the network is unavailable, so the script still produces something.
FALLBACK_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AVGO", "TSLA", "AMD", "INTC",
    "MU", "QCOM", "TXN", "ADI", "LRCX", "AMAT", "KLAC", "ASML", "ARM", "SMCI",
    "DELL", "HPE", "ANET", "CSCO", "JNPR", "VRT", "ETN", "PWR", "NVT", "GEV",
    "JPM", "BAC", "WFC", "GS", "MS", "V", "MA", "AXP", "BLK", "SCHW",
    "JNJ", "UNH", "LLY", "PFE", "MRK", "ABBV", "TMO", "DHR", "ABT", "AMGN",
    "XOM", "CVX", "COP", "SLB", "EOG", "PSX", "VLO", "MPC", "OXY", "HAL",
    "PG", "KO", "PEP", "COST", "WMT", "HD", "MCD", "NKE", "SBUX", "TGT",
    "CAT", "DE", "BA", "HON", "GE", "LMT", "RTX", "UPS", "UNP", "MMM",
    "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL", "ED", "PEG",
    "SPY",
]


def fetch_sp500() -> List[str]:
    try:
        tables = pd.read_html(SP500_URL)
        tickers = tables[0]["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
        log.info("universe source: S&P 500 constituents (%d names)", len(tickers))
        return sorted(set(tickers + [settings.benchmark_ticker]))
    except Exception as exc:  # noqa: BLE001
        log.warning("could not fetch S&P 500 list (%s) - using fallback", exc)
        return sorted(set(FALLBACK_TICKERS + [settings.benchmark_ticker]))


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed the local data cache")
    parser.add_argument("--tickers", type=str, default=None, help="comma-separated override")
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--refresh", action="store_true", help="force re-download")
    parser.add_argument("--skip-profiles", action="store_true",
                        help="profiles are a slow per-ticker loop; skip for a quick refresh")
    args = parser.parse_args()

    if settings.offline_mode:
        log.error("OFFLINE_MODE is true - set it to false in .env before seeding.")
        return 1

    tickers = (
        [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        if args.tickers else fetch_sp500()
    )
    start = date.fromisoformat(args.start) if args.start else settings.backtest_start
    end = date.fromisoformat(args.end) if args.end else settings.backtest_end

    log.info("seeding %d tickers, %s to %s", len(tickers), start, end)

    panel = build_price_panel(tickers, start, end, refresh=args.refresh)
    if panel.empty:
        log.error("no price data returned - check the network and try again")
        return 1

    if args.skip_profiles:
        log.info("skipping profiles")
        profiles = pd.DataFrame(columns=["ticker", "name", "sector", "industry", "exchange", "market_cap"])
    else:
        profiles = build_profiles(sorted(panel["ticker"].unique()), refresh=args.refresh)

    close = to_wide(panel, "adj_close")
    volume = to_wide(panel, "volume")
    adv = average_dollar_volume(close, volume)

    print()
    print(f"  tickers with data : {panel['ticker'].nunique()}")
    print(f"  rows              : {len(panel):,}")
    print(f"  date range        : {panel['date'].min()} to {panel['date'].max()}")
    print(f"  median 20d ADV    : ${adv.ffill().iloc[-1].median()/1e6:,.1f}M")
    print(f"  cache             : {settings.cache_path}")

    if not profiles.empty:
        universe = build_universe(panel, profiles)
        print()
        print("  Universe funnel:")
        for stage in universe.funnel():
            print(f"    {stage['count']:>5}  {stage['label']:<20} {stage['description']}")

    print()
    print("Seeding complete. Set OFFLINE_MODE=true in .env before the demo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
