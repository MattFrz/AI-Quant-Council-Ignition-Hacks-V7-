"""Re-fetch profile fields that came back empty on the seed run.

Yahoo's `get_info()` fails silently under rate limiting, and the seed run
records the miss as a null rather than retrying. A null market cap is not a
small company, but `filter_market_cap` cannot tell the difference, so the name
is dropped from the universe entirely - Micron disappeared this way despite
having 1,465 indexed filings, and "Analyze MU" came back about another company.

Fixing the data keeps the universe invariant intact: every survivor has a
market cap we can actually state.

    python scripts/backfill_profiles.py --dry-run
    python scripts/backfill_profiles.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from backend.config import settings  # noqa: E402
from backend.core.logging import get_logger  # noqa: E402
from data.pipelines.prices import PROFILE_FILE, load_profiles  # noqa: E402

log = get_logger("backfill_profiles")

#: Fields worth a second attempt. `name` is excluded: the seeder falls back to
#: the ticker rather than leaving it null, so a miss there is not detectable.
REFETCH = ("market_cap", "sector", "industry", "exchange")


def main() -> int:
    p = argparse.ArgumentParser(description="Re-fetch missing profile fields")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--field", default="market_cap", help="field that must be present")
    args = p.parse_args()

    profiles = load_profiles()
    missing = profiles[profiles[args.field].isna()]["ticker"].tolist()

    if not missing:
        print(f"No profiles missing {args.field}. Nothing to do.")
        return 0

    print(f"{len(missing)} profiles missing {args.field}: {', '.join(sorted(missing))}\n")
    if args.dry_run:
        return 0

    import yfinance as yf

    from data.pipelines.prices import load_prices

    last_close = (
        load_prices().sort_values("date").groupby("ticker")["adj_close"].last()
    )

    fixed, still_missing = 0, []
    for n, ticker in enumerate(sorted(missing), 1):
        tk = yf.Ticker(ticker)
        try:
            info = tk.get_info() or {}
        except Exception as exc:  # noqa: BLE001 - one bad ticker must not stop the run
            log.debug("no profile for %s (%s)", ticker, exc)
            info = {}

        row = profiles["ticker"] == ticker
        got = {}
        for field in REFETCH:
            key = {"market_cap": "marketCap"}.get(field, field)
            value = info.get(key)
            if value is not None and pd.isna(profiles.loc[row, field]).all():
                profiles.loc[row, field] = value
                got[field] = value

        # Derive the cap Yahoo would not give us.
        #
        # `marketCap` comes back null for names whose shares outstanding is
        # served fine, so multiply it by OUR last close rather than trusting a
        # second Yahoo field - it is the same price series the backtest runs
        # on, so the universe and the returns agree on what a share is worth.
        if "market_cap" not in got:
            shares = info.get("sharesOutstanding")
            if shares is None:
                try:
                    shares = tk.fast_info.get("shares")
                except Exception:  # noqa: BLE001
                    shares = None
            price = last_close.get(ticker)
            if shares and price and not pd.isna(price):
                cap = float(shares) * float(price)
                profiles.loc[row, "market_cap"] = cap
                got["market_cap"] = cap

        if args.field in got:
            fixed += 1
            cap = got.get("market_cap")
            shown = f"${cap/1e9:.1f}B" if isinstance(cap, (int, float)) else str(cap)
            print(f"  [{n}/{len(missing)}] {ticker:<6} {shown}")
        else:
            still_missing.append(ticker)
            print(f"  [{n}/{len(missing)}] {ticker:<6} still missing")

    if fixed:
        path = settings.cache_path / PROFILE_FILE
        profiles.to_parquet(path, index=False)
        print(f"\nwrote {fixed} repaired profiles to {path}")

    if still_missing:
        print(f"\n{len(still_missing)} still missing: {', '.join(sorted(still_missing))}")
        print("These stay out of the universe, which is the correct outcome for")
        print("a name whose size we genuinely cannot establish.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
