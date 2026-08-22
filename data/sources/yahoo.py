"""Yahoo Finance via yfinance. Step A2.

Batch every request. yfinance throttles hard on single-ticker loops, and a
1,000-name universe pulled one at a time will take longer than the hackathon.
"""
from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional, Sequence

import pandas as pd

from backend.core.logging import get_logger
from data.sources.base import DataSource, cache_key

log = get_logger(__name__)

PRICE_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
PROFILE_COLUMNS = ["ticker", "name", "sector", "industry", "exchange", "market_cap"]

#: How many tickers to request per yfinance call.
BATCH_SIZE = 100


class YahooSource(DataSource):
    """Historical OHLCV and light company metadata."""

    min_request_interval_s = 0.2

    @property
    def name(self) -> str:
        return "yahoo"

    # ----------------------------------------------------------------- prices

    def fetch_prices(
        self,
        tickers: Sequence[str],
        start: date,
        end: date,
        refresh: bool = False,
    ) -> pd.DataFrame:
        """Long-format OHLCV: one row per (date, ticker).

        adj_close is split/dividend adjusted and is what returns are computed
        from. close is the raw price, used for notional and liquidity checks.
        """
        tickers = sorted({t.upper() for t in tickers})
        frames: List[pd.DataFrame] = []

        for i in range(0, len(tickers), BATCH_SIZE):
            batch = tickers[i : i + BATCH_SIZE]
            key = cache_key("prices", start, end, len(batch), batch[0], batch[-1])
            frames.append(
                self.cached_frame(
                    key,
                    lambda b=batch: self._download_batch(b, start, end),
                    refresh=refresh,
                )
            )

        if not frames:
            return pd.DataFrame(columns=PRICE_COLUMNS)

        out = pd.concat(frames, ignore_index=True)
        out = out.dropna(subset=["adj_close"])
        out["date"] = pd.to_datetime(out["date"]).dt.date
        return out.sort_values(["ticker", "date"]).reset_index(drop=True)

    def _download_batch(self, tickers: List[str], start: date, end: date) -> pd.DataFrame:
        import yfinance as yf

        log.info("yahoo: downloading %d tickers %s to %s", len(tickers), start, end)
        raw = yf.download(
            tickers=tickers,
            start=str(start),
            end=str(end),
            auto_adjust=False,
            actions=False,
            progress=False,
            group_by="ticker",
            threads=True,
        )

        if raw is None or raw.empty:
            return pd.DataFrame(columns=PRICE_COLUMNS)

        rows: List[pd.DataFrame] = []
        for ticker in tickers:
            try:
                sub = raw[ticker] if isinstance(raw.columns, pd.MultiIndex) else raw
            except KeyError:
                continue
            if sub is None or sub.empty:
                continue

            frame = pd.DataFrame(
                {
                    "date": sub.index,
                    "ticker": ticker,
                    "open": _col(sub, "Open"),
                    "high": _col(sub, "High"),
                    "low": _col(sub, "Low"),
                    "close": _col(sub, "Close"),
                    "adj_close": _col(sub, "Adj Close", fallback="Close"),
                    "volume": _col(sub, "Volume"),
                }
            )
            rows.append(frame.dropna(subset=["close"]))

        if not rows:
            return pd.DataFrame(columns=PRICE_COLUMNS)
        return pd.concat(rows, ignore_index=True)[PRICE_COLUMNS]

    # --------------------------------------------------------------- profiles

    def fetch_profiles(
        self, tickers: Sequence[str], refresh: bool = False
    ) -> pd.DataFrame:
        """Sector, industry, exchange, market cap. Used by the universe filters.

        yfinance has no batch metadata endpoint, so this is a per-ticker loop -
        the reason it is cached aggressively and pulled once by seed_data.py.
        """
        tickers = sorted({t.upper() for t in tickers})
        key = cache_key("profiles", len(tickers), tickers[0] if tickers else "", tickers[-1] if tickers else "")
        return self.cached_frame(key, lambda: self._download_profiles(tickers), refresh=refresh)

    def _download_profiles(self, tickers: List[str]) -> pd.DataFrame:
        import yfinance as yf

        log.info("yahoo: fetching profiles for %d tickers", len(tickers))
        records: List[Dict[str, object]] = []

        for ticker in tickers:
            info: Dict[str, object] = {}
            try:
                self._limiter.wait()
                info = yf.Ticker(ticker).get_info() or {}
            except Exception as exc:  # noqa: BLE001 - one bad ticker must not stop the pull
                log.debug("yahoo: no profile for %s (%s)", ticker, exc)

            records.append(
                {
                    "ticker": ticker,
                    "name": info.get("longName") or info.get("shortName") or ticker,
                    "sector": info.get("sector"),
                    "industry": info.get("industry"),
                    "exchange": info.get("exchange"),
                    "market_cap": info.get("marketCap"),
                }
            )

        return pd.DataFrame(records, columns=PROFILE_COLUMNS)


def _col(df: pd.DataFrame, name: str, fallback: Optional[str] = None) -> pd.Series:
    if name in df.columns:
        return df[name].to_numpy()
    if fallback and fallback in df.columns:
        return df[fallback].to_numpy()
    return pd.Series([float("nan")] * len(df)).to_numpy()
