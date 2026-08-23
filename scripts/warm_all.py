"""Warm a cached run for every company in the retrieval index.

A cold run is 60 to 90 seconds. During a demo that is a long time to stand
still, and a live LLM call is a dependency you do not need in front of judges.
This pre-computes one real run per indexed company so any of them replays in
under a second.

    python scripts/warm_all.py              # the demo thesis + all indexed names
    python scripts/warm_all.py --only NVDA,VRT
    python scripts/warm_all.py --force      # recompute entries already warmed
    python scripts/warm_all.py --dry-run    # show what it would do

Roughly 70 seconds and a few cents of LLM spend per company. Every result is a
genuine run: nothing here fabricates or edits output, it just computes it early.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config import settings  # noqa: E402
from backend.core import cache  # noqa: E402
from backend.core.logging import get_logger  # noqa: E402

log = get_logger("warm_all")

#: Phrasing that reliably targets one company. `_named_tickers` matches on the
#: ticker, so this steers candidate selection to that name.
COMPANY_THESIS = (
    "Analyze {ticker} and whether the market is underpricing its exposure to "
    "accelerating AI data-center spending."
)


def indexed_tickers() -> list:
    """Companies with filings in the retrieval index."""
    from backend.rag.index.build_index import CHUNK_LOOKUP_PATH

    sqlite_path = CHUNK_LOOKUP_PATH.parent / "chunks.sqlite"
    if sqlite_path.exists():
        from backend.rag.index.chunk_store import ChunkStore

        store = ChunkStore(sqlite_path)
        try:
            return store.tickers()
        finally:
            store.close()

    if not CHUNK_LOOKUP_PATH.exists():
        return []
    lookup = json.loads(CHUNK_LOOKUP_PATH.read_text(encoding="utf-8"))
    return sorted({v.get("ticker") for v in lookup.values() if v.get("ticker")})


def drop_stale() -> int:
    """Delete cache entries written under an older schema version.

    They would be ignored on read anyway, but leaving them around makes
    `cache list` misleading about what is actually warm.
    """
    removed = 0
    for path in (settings.cache_path / "pipeline").glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            path.unlink()
            removed += 1
            continue
        if payload.get("cache_version") != cache.CACHE_VERSION:
            path.unlink()
            removed += 1
    return removed


def main() -> int:
    p = argparse.ArgumentParser(description="Warm a cached run per indexed company")
    p.add_argument("--only", type=str, default=None, help="comma-separated tickers")
    p.add_argument("--force", action="store_true", help="recompute existing entries")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-demo", action="store_true", help="skip the main demo thesis")
    args = p.parse_args()

    stale = drop_stale()
    if stale:
        print(f"dropped {stale} stale cache entries\n")

    tickers = (
        [t.strip().upper() for t in args.only.split(",") if t.strip()]
        if args.only else indexed_tickers()
    )
    if not tickers:
        print("No indexed companies found. Build the retrieval index first.")
        return 1

    jobs = []
    if not args.skip_demo:
        jobs.append(("DEMO", cache.DEMO_THESIS))
    jobs += [(t, COMPANY_THESIS.format(ticker=t)) for t in tickers]

    print(f"{len(jobs)} theses to warm ({len(tickers)} companies)\n")

    if args.dry_run:
        for label, thesis in jobs:
            state = "warm" if cache.get(thesis) else "cold"
            print(f"  {label:<6} {state:<5} {thesis[:64]}")
        return 0

    started = time.monotonic()
    rows = []

    for n, (label, thesis) in enumerate(jobs, 1):
        if not args.force and cache.get(thesis) is not None:
            print(f"  [{n}/{len(jobs)}] {label:<6} already warm, skipping")
            continue

        print(f"  [{n}/{len(jobs)}] {label:<6} running...", flush=True)
        try:
            result = cache.warm(thesis, force=args.force)
        except Exception as exc:  # noqa: BLE001 - one bad name must not stop the rest
            print(f"           FAILED: {str(exc)[:70]}")
            rows.append((label, "FAILED", "", "", "", ""))
            continue

        idea = result.top_idea
        if idea is None:
            rows.append((label, "no idea", "", "", "", str(len(result.degraded))))
            continue

        rows.append((
            label,
            idea.ticker,
            f"{idea.alpha_score}",
            f"{idea.confidence:.0%}",
            str(len(idea.catalysts)),
            str(len(result.degraded)),
        ))

    elapsed = time.monotonic() - started

    print()
    print(f"  {'THESIS':<7} {'PICK':<6} {'ALPHA':>6} {'CONF':>6} {'CATALYSTS':>10} {'DEGRADED':>9}")
    print(f"  {'-'*7} {'-'*6} {'-'*6} {'-'*6} {'-'*10} {'-'*9}")
    for label, ticker, alpha, conf, cats, degraded in rows:
        print(f"  {label:<7} {ticker:<6} {alpha:>6} {conf:>6} {cats:>10} {degraded:>9}")

    print()
    print(f"warmed {len(rows)} runs in {elapsed/60:.1f} min")
    print("Check with:  python -m backend.core.cache")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
