"""Bundle the data a deployed instance actually needs.

The full `data/cache/` is about 1.4 GB, but most of it exists to BUILD things
rather than serve them: the embedding cache is consumed when the FAISS index is
written, the raw filings and Yahoo pulls are already distilled into the index
and the price panel. A server only needs what it reads at request time.

    python scripts/package_data.py                 # writes dist/aqc-data.tar.gz
    python scripts/package_data.py --list          # show what would go in

Upload the result as a GitHub release asset (2 GB limit, free) and point
DATA_BUNDLE_URL at it. `scripts/fetch_data.py` pulls it down on first boot.
"""
from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config import settings  # noqa: E402

#: Paths, relative to data/cache, that the server reads at request time.
#: Anything not listed here is regenerable from a seed run and stays local.
RUNTIME_PATHS = [
    "index/filings.faiss",       # vector store
    "index/chunk_lookup.json",   # chunk text, tickers, filed dates
    "index/filings.ids.json",    # faiss row -> chunk id
    "price_panel.parquet",       # prices for universe, backtest, risk
    "profiles.parquet",          # names, sectors, market caps
    "pipeline",                  # warmed results - the whole point of the demo
]


def collect(base: Path):
    found, missing = [], []
    for rel in RUNTIME_PATHS:
        p = base / rel
        (found if p.exists() else missing).append((rel, p))
    return found, missing


def size_of(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def main() -> int:
    ap = argparse.ArgumentParser(description="Bundle runtime data for deployment")
    ap.add_argument("--out", default="dist/aqc-data.tar.gz")
    ap.add_argument("--list", action="store_true", help="show contents and exit")
    args = ap.parse_args()

    base = settings.cache_path
    found, missing = collect(base)

    total = 0
    for rel, p in found:
        n = size_of(p)
        total += n
        print(f"  {rel:<28} {n/1e6:8.1f} MB")
    for rel, _ in missing:
        print(f"  {rel:<28}   MISSING")

    print(f"\n  {'total':<28} {total/1e6:8.1f} MB")

    if missing:
        print("\nMissing paths above will not be in the bundle. Run "
              "scripts/seed_data.py and scripts/warm_all.py first if the server "
              "needs them.")

    if args.list:
        return 0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nwriting {out} ...")
    with tarfile.open(out, "w:gz") as tar:
        for rel, p in found:
            tar.add(p, arcname=rel)

    print(f"done: {out} ({out.stat().st_size/1e6:.1f} MB compressed)")
    print("\nNext: upload it as a GitHub release asset, then set")
    print("  DATA_BUNDLE_URL=<the asset's browser_download_url>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
