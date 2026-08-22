"""Base class for every external data source. Step A1.

Two jobs:
  1. Retry with backoff, so one flaky request does not kill a pipeline run.
  2. Cache to disk, so the demo never depends on a live API call.

Rule 1 of the build plan: every external call writes through this cache.
"""
from __future__ import annotations

import hashlib
import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

import pandas as pd

from backend.config import settings
from backend.core.logging import get_logger

log = get_logger(__name__)

T = TypeVar("T")


class DataSourceError(RuntimeError):
    """Raised when a source cannot satisfy a request after retries."""


class OfflineError(DataSourceError):
    """Raised when offline_mode is on and the answer is not already cached.

    This is deliberately loud: during a demo we want to know immediately that
    something is missing from the cache, not silently fetch it over the network.
    """


class RateLimiter:
    """Minimum wall-clock interval between calls.

    EDGAR blocks above ~10 requests/second and a block during the demo is not
    recoverable, so the limiter lives here rather than in one source.
    """

    def __init__(self, min_interval_s: float) -> None:
        self.min_interval_s = min_interval_s
        self._last_call = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < self.min_interval_s:
            time.sleep(self.min_interval_s - elapsed)
        self._last_call = time.monotonic()


def cache_key(*parts: Any) -> str:
    """Stable filename-safe key from arbitrary arguments."""
    raw = "|".join(str(p) for p in parts)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    head = "".join(c if c.isalnum() or c in "-_" else "_" for c in raw)[:60]
    return f"{head}__{digest}"


class DataSource(ABC):
    """Subclass this for Yahoo, EDGAR, news, anything external."""

    #: Seconds between outbound requests. Override per source.
    min_request_interval_s: float = 0.0

    #: How many times to retry a failed fetch before giving up.
    max_retries: int = 3

    def __init__(self, cache_dir: Optional[Path] = None, offline: Optional[bool] = None) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else settings.cache_path
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.offline = settings.offline_mode if offline is None else offline
        self._limiter = RateLimiter(self.min_request_interval_s)

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used as the cache subdirectory."""

    # ------------------------------------------------------------------ cache

    def _dir(self) -> Path:
        d = self.cache_dir / self.name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def cached_frame(
        self,
        key: str,
        loader: Callable[[], pd.DataFrame],
        refresh: bool = False,
    ) -> pd.DataFrame:
        """Return a DataFrame from disk, or fetch it and write it there."""
        path = self._dir() / f"{key}.parquet"

        if path.exists() and not refresh:
            log.debug("cache hit  %s/%s", self.name, key)
            return pd.read_parquet(path)

        if self.offline:
            raise OfflineError(
                f"offline_mode is on and {self.name}/{key} is not cached. "
                f"Run scripts/seed_data.py before the demo."
            )

        df = self._with_retries(loader, key)
        if df is not None and not df.empty:
            df.to_parquet(path, index=False)
            log.info("cached     %s/%s  (%d rows)", self.name, key, len(df))
        return df

    def cached_json(
        self,
        key: str,
        loader: Callable[[], Any],
        refresh: bool = False,
    ) -> Any:
        path = self._dir() / f"{key}.json"

        if path.exists() and not refresh:
            return json.loads(path.read_text(encoding="utf-8"))

        if self.offline:
            raise OfflineError(
                f"offline_mode is on and {self.name}/{key} is not cached."
            )

        payload = self._with_retries(loader, key)
        path.write_text(json.dumps(payload, default=str), encoding="utf-8")
        return payload

    # ------------------------------------------------------------------ retry

    def _with_retries(self, fn: Callable[[], T], label: str) -> T:
        last: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                self._limiter.wait()
                return fn()
            except Exception as exc:  # noqa: BLE001 - we re-raise below
                last = exc
                backoff = 2 ** (attempt - 1)
                log.warning(
                    "%s/%s failed (attempt %d/%d): %s - retrying in %ds",
                    self.name, label, attempt, self.max_retries, exc, backoff,
                )
                if attempt < self.max_retries:
                    time.sleep(backoff)
        raise DataSourceError(
            f"{self.name}/{label} failed after {self.max_retries} attempts: {last}"
        ) from last

    def is_cached(self, key: str, ext: str = "parquet") -> bool:
        return (self._dir() / f"{key}.{ext}").exists()
