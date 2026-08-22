"""Disk cache and offline mode. Step A1.

Rule 1 of the build plan is that no demo path depends on a live API call. This
is the test that proves it, using a fake source so nothing touches the network.
"""
from __future__ import annotations

import pandas as pd
import pytest

from data.sources.base import DataSource, DataSourceError, OfflineError, cache_key


class CountingSource(DataSource):
    """Fake source that records how many times it actually fetched."""

    def __init__(self, *args, fail_times: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = 0
        self.fail_times = fail_times

    @property
    def name(self) -> str:
        return "counting"

    def load(self) -> pd.DataFrame:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError(f"simulated failure {self.calls}")
        return pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})


@pytest.fixture
def cache_dir(tmp_path):
    return tmp_path / "cache"


# ------------------------------------------------------------------- cache

def test_first_call_fetches_and_writes_to_disk(cache_dir):
    s = CountingSource(cache_dir=cache_dir, offline=False)
    df = s.cached_frame("k", s.load)

    assert len(df) == 3
    assert s.calls == 1
    assert s.is_cached("k")


def test_second_call_reads_cache_without_fetching(cache_dir):
    """The whole point: a warm cache must not hit the network."""
    s = CountingSource(cache_dir=cache_dir, offline=False)
    s.cached_frame("k", s.load)
    df = s.cached_frame("k", s.load)

    assert s.calls == 1, "second call re-fetched instead of using the cache"
    assert len(df) == 3


def test_refresh_forces_a_refetch(cache_dir):
    s = CountingSource(cache_dir=cache_dir, offline=False)
    s.cached_frame("k", s.load)
    s.cached_frame("k", s.load, refresh=True)
    assert s.calls == 2


def test_separate_keys_do_not_collide(cache_dir):
    s = CountingSource(cache_dir=cache_dir, offline=False)
    s.cached_frame("one", s.load)
    s.cached_frame("two", s.load)
    assert s.calls == 2
    assert s.is_cached("one") and s.is_cached("two")


# ------------------------------------------------------------------ offline

def test_offline_mode_serves_a_warm_cache(cache_dir):
    """Seed online, then flip offline. This is the demo-day path."""
    warm = CountingSource(cache_dir=cache_dir, offline=False)
    warm.cached_frame("k", warm.load)

    cold = CountingSource(cache_dir=cache_dir, offline=True)
    df = cold.cached_frame("k", cold.load)

    assert len(df) == 3
    assert cold.calls == 0, "offline mode reached the loader"


def test_offline_mode_fails_loudly_on_a_cold_cache(cache_dir):
    """Silence here would mean discovering a missing file mid-demo."""
    s = CountingSource(cache_dir=cache_dir, offline=True)
    with pytest.raises(OfflineError, match="not cached"):
        s.cached_frame("never_seeded", s.load)
    assert s.calls == 0


def test_offline_error_names_the_seed_script(cache_dir):
    s = CountingSource(cache_dir=cache_dir, offline=True)
    with pytest.raises(OfflineError, match="seed_data"):
        s.cached_frame("missing", s.load)


# ------------------------------------------------------------------- retry

def test_transient_failures_are_retried(cache_dir):
    s = CountingSource(cache_dir=cache_dir, offline=False, fail_times=2)
    s.max_retries = 3
    df = s.cached_frame("k", s.load)

    assert s.calls == 3
    assert len(df) == 3


def test_gives_up_after_max_retries(cache_dir):
    s = CountingSource(cache_dir=cache_dir, offline=False, fail_times=99)
    s.max_retries = 2
    with pytest.raises(DataSourceError):
        s.cached_frame("k", s.load)
    assert s.calls == 2


# --------------------------------------------------------------- cache keys

def test_cache_key_is_stable():
    assert cache_key("prices", "2024-01-01", 100) == cache_key("prices", "2024-01-01", 100)


def test_cache_key_changes_with_arguments():
    assert cache_key("prices", "2024-01-01") != cache_key("prices", "2024-01-02")


def test_cache_key_is_filename_safe():
    key = cache_key("prices", "2024-01-01/2024-12-31", "A:B")
    assert not set(key) & set('/\\:*?"<>|')


def test_json_cache_roundtrips(cache_dir):
    s = CountingSource(cache_dir=cache_dir, offline=False)
    payload = {"AAPL": 320193, "MSFT": 789019}
    s.cached_json("map", lambda: payload)

    cold = CountingSource(cache_dir=cache_dir, offline=True)
    assert cold.cached_json("map", lambda: {}) == payload
