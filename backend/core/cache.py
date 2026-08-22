"""Pipeline result cache. Step 3.7 - demo insurance.

A full run costs minutes and a fistful of LLM calls. Doing that live, in front
of judges, on hackathon wifi, is a bet you do not need to take.

Warm the cache once before the demo:

    python -m backend.core.cache warm "Find mispriced beneficiaries of AI
    infrastructure spending."

Then the live run replays a genuine, previously computed result in about a
second. Nothing is faked - it is the same TradeIdea the pipeline produced, with
the same catalysts, the same backtest and the same timeline. Only the waiting
is removed.

Keyed on (thesis, as_of, max_candidates, universe_size), so changing the thesis
misses the cache and runs for real.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.config import settings
from backend.core.logging import get_logger
from backend.services.events import ResearchEvent
from backend.services.pipeline import PipelineResult
from data.schemas.trade_idea import TradeIdea

log = get_logger(__name__)

#: Bump whenever the SHAPE of a stored run changes - new timeline steps, new
#: TradeIdea fields, changed event semantics. Entries written under an older
#: version are ignored rather than replayed.
#:
#: This is not bookkeeping. A cached run stores its own event list, so after
#: the timeline went from 11 steps to 13, every previously cached thesis kept
#: replaying 11 forever while a fresh thesis returned 13 - which looked exactly
#: like a bug that only affected "some prompts".
#:
#: 2: 13-step timeline, per-stock risk, populated position size and verdict bar
CACHE_VERSION = 2
CACHE_SUBDIR = "pipeline"


def cache_key(
    thesis: str,
    as_of: Optional[date] = None,
    max_candidates: int = 7,
    universe_size: Optional[int] = None,
) -> str:
    """Stable key. Normalised so trivial edits to the thesis still hit.

    `as_of=None` resolves to today, matching what the pipeline does with it.
    Without this, a warmed run (which stores its resolved date) and a live
    request (which passes None) produce different keys and the cache never
    hits - silently, which is the worst way for demo insurance to fail.
    """
    normalized = " ".join(thesis.lower().split())
    payload = "|".join([
        str(CACHE_VERSION),
        normalized,
        str(as_of or date.today()),
        str(max_candidates),
        str(universe_size or ""),
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _cache_dir() -> Path:
    d = settings.cache_path / CACHE_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    return d


# ------------------------------------------------------------- serialisation

def to_dict(result: PipelineResult) -> Dict[str, Any]:
    return {
        "cache_version": CACHE_VERSION,
        "cached_at": datetime.utcnow().isoformat(),
        "thesis": result.thesis,
        "as_of": result.as_of.isoformat(),
        "top_idea": result.top_idea.model_dump(mode="json") if result.top_idea else None,
        "runners_up": [i.model_dump(mode="json") for i in result.runners_up],
        "funnel_stages": result.funnel(),
        "events": [e.model_dump(mode="json") for e in result.events],
        "llm_cost_usd": result.llm_cost_usd,
        "elapsed_s": result.elapsed_s,
        "degraded": result.degraded,
    }


def from_dict(payload: Dict[str, Any]) -> PipelineResult:
    return PipelineResult(
        thesis=payload["thesis"],
        as_of=date.fromisoformat(payload["as_of"]),
        top_idea=TradeIdea.model_validate(payload["top_idea"]) if payload.get("top_idea") else None,
        runners_up=[TradeIdea.model_validate(i) for i in payload.get("runners_up", [])],
        events=[ResearchEvent.model_validate(e) for e in payload.get("events", [])],
        funnel_stages=payload.get("funnel_stages", []),
        llm_cost_usd=payload.get("llm_cost_usd"),
        elapsed_s=payload.get("elapsed_s", 0.0),
        degraded=payload.get("degraded", []),
    )


# --------------------------------------------------------------------- api

def get(
    thesis: str,
    as_of: Optional[date] = None,
    max_candidates: int = 7,
    universe_size: Optional[int] = None,
) -> Optional[PipelineResult]:
    """Cached result, or None. Never raises - a corrupt entry is a miss."""
    path = _cache_dir() / f"{cache_key(thesis, as_of, max_candidates, universe_size)}.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("cache_version") != CACHE_VERSION:
            log.info("cache: stale version for %s, ignoring", path.name)
            return None
        result = from_dict(payload)
        log.info("cache HIT  %s (%s, cached %s)",
                 path.name, result.thesis[:40], payload.get("cached_at", "?")[:19])
        return result
    except Exception as exc:  # noqa: BLE001
        log.warning("cache: unreadable entry %s (%s) - treating as miss", path.name, exc)
        return None


def put(
    result: PipelineResult,
    max_candidates: int = 7,
    universe_size: Optional[int] = None,
) -> Path:
    """Store a completed run.

    Refuses to cache a result with no top_idea: caching a failure would make
    every later run replay that failure instantly, which is the opposite of
    demo insurance.
    """
    if result.top_idea is None:
        raise ValueError("refusing to cache a run that produced no TradeIdea")

    key = cache_key(result.thesis, result.as_of, max_candidates, universe_size)
    path = _cache_dir() / f"{key}.json"
    path.write_text(json.dumps(to_dict(result), indent=2), encoding="utf-8")
    log.info("cache PUT  %s (%s)", path.name, result.thesis[:40])
    return path


def entries() -> List[Dict[str, Any]]:
    """What is warmed, for a pre-demo check."""
    out: List[Dict[str, Any]] = []
    for p in sorted(_cache_dir().glob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            out.append({"key": p.stem, "thesis": "<unreadable>", "ok": False})
            continue
        idea = d.get("top_idea") or {}
        out.append({
            "key": p.stem,
            "thesis": d.get("thesis", ""),
            "ticker": idea.get("ticker"),
            "catalysts": len(idea.get("catalysts", [])),
            "degraded": len(d.get("degraded", [])),
            "elapsed_s": d.get("elapsed_s"),
            "cached_at": d.get("cached_at", "")[:19],
            "ok": True,
        })
    return out


def clear() -> int:
    d = _cache_dir()
    n = len(list(d.glob("*.json")))
    shutil.rmtree(d, ignore_errors=True)
    log.info("cache: cleared %d entries", n)
    return n


DEMO_THESIS = (
    "Find companies benefiting from accelerating AI data-center spending "
    "that the market may be underpricing."
)


def warm(thesis: str = DEMO_THESIS, force: bool = False, **kwargs) -> PipelineResult:
    """Run the pipeline for real and store the result."""
    from backend.services.pipeline import run_pipeline

    if not force:
        hit = get(thesis, **kwargs)
        if hit is not None:
            log.info("cache: already warm, use --force to recompute")
            return hit

    log.info("cache: warming (this runs the full pipeline)")
    result = run_pipeline(thesis=thesis, **kwargs)
    if result.top_idea is None:
        log.error("cache: run produced no TradeIdea, nothing stored")
        return result
    put(result, **{k: v for k, v in kwargs.items() if k in ("max_candidates", "universe_size")})
    return result


if __name__ == "__main__":
    import sys

    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"

    if cmd == "warm":
        # Strip flags so "--force" is not mistaken for the thesis text.
        words = [a for a in sys.argv[2:] if not a.startswith("-")]
        thesis = " ".join(words) or DEMO_THESIS
        res = warm(thesis, force="--force" in sys.argv)
        print(f"\n  {res.thesis}")
        print(f"  -> {res.top_idea.ticker if res.top_idea else 'no idea'} "
              f"in {res.elapsed_s:.1f}s, {len(res.degraded)} degraded\n")

    elif cmd == "clear":
        print(f"cleared {clear()} entries")

    else:
        rows = entries()
        if not rows:
            print("\n  cache is EMPTY - warm it before the demo:")
            print("    python -m backend.core.cache warm\n")
        else:
            print(f"\n  {len(rows)} cached run(s):\n")
            for r in rows:
                flag = "" if r.get("catalysts") else "   <- no catalysts"
                print(f"    {r['ticker'] or '?':<6} {r['cached_at']}  "
                      f"{r.get('catalysts', 0)} catalysts, "
                      f"{r.get('degraded', 0)} degraded{flag}")
                print(f"           {r['thesis'][:66]}")
            print()
