"""Caps on how many live pipeline runs a public instance will perform.

`PUBLIC_DEMO_MODE` is the blunt version: answer warmed theses, refuse the rest.
That protects the bill completely but makes the deployed app unable to research
anything, which is most of what the project does.

This is the middle setting. Live runs are allowed, but only so many per hour and
per day, so a stranger, a crawler, or a refresh-happy visitor cannot turn an
open URL into an open tab on the owner's LLM account.

Counters live in process memory. A restart forgets them, which is acceptable:
the limit exists to bound a runaway, not to bill anyone accurately. It also
assumes one worker, which the Dockerfile enforces for the same reason job state
does.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Deque, Optional

_HOUR = 3600.0
_DAY = 86400.0

_lock = threading.Lock()
_runs: Deque[float] = deque()


def _prune(now: float) -> None:
    while _runs and now - _runs[0] > _DAY:
        _runs.popleft()


def check(per_hour: int, per_day: int) -> Optional[str]:
    """Reason the run should be refused, or None to allow it.

    A limit of 0 means unlimited, matching the settings default so a local
    checkout is never rate limited.
    """
    if per_hour <= 0 and per_day <= 0:
        return None

    now = time.monotonic()
    with _lock:
        _prune(now)
        if per_day > 0 and len(_runs) >= per_day:
            return (
                "This deployment has hit its daily limit for live research runs. "
                "The pre-computed theses still work, or run the project locally "
                "for unlimited research."
            )
        if per_hour > 0:
            recent = sum(1 for t in _runs if now - t <= _HOUR)
            if recent >= per_hour:
                return (
                    "This deployment has hit its hourly limit for live research "
                    "runs. Try again later, pick one of the pre-computed theses, "
                    "or run the project locally."
                )
    return None


def record() -> None:
    """Count a live run. Call only when one is actually starting."""
    now = time.monotonic()
    with _lock:
        _prune(now)
        _runs.append(now)


def state() -> dict:
    """Current usage, for /health and for tests."""
    now = time.monotonic()
    with _lock:
        _prune(now)
        return {
            "last_hour": sum(1 for t in _runs if now - t <= _HOUR),
            "last_day": len(_runs),
        }


def reset() -> None:
    """Test hook."""
    with _lock:
        _runs.clear()
