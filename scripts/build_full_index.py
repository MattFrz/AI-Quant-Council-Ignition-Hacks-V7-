"""Build a retrieval index across the whole universe, a few tickers at a time.

`build_index()` loads every chunk, every vector and the entire lookup into
memory at once and writes one large JSON. That is fine for seventeen companies
and impossible for five hundred: roughly 774k chunks, 4.75 GB of vectors and a
1.6 GB serialisation, on a machine with 15.7 GB of RAM.

This does the same job in batches, and can be stopped and resumed. Each batch
pulls filings, chunks them, embeds what is not already cached, appends the
vectors to the FAISS index and writes the chunk rows to SQLite. State is saved
after every batch, so an interrupted run picks up where it left off rather than
starting the EDGAR fetch again.

    python scripts/build_full_index.py --dry-run       # what it would do
    python scripts/build_full_index.py                 # start, or resume
    python scripts/build_full_index.py --limit 20      # try a slice first
    python scripts/build_full_index.py --reset         # forget progress

Expect hours, mostly waiting on EDGAR, and a few dollars of embedding spend.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config import settings  # noqa: E402
from backend.core.logging import get_logger  # noqa: E402

log = get_logger("build_full_index")

INDEX_DIR = Path(settings.data_cache_dir) / "index"
FAISS_PATH = INDEX_DIR / "filings.faiss"
STORE_PATH = INDEX_DIR / "chunks.sqlite"
STATE_PATH = INDEX_DIR / "build_state.json"

#: Tickers per batch. Small enough that a crash loses little work, large enough
#: that the FAISS save (which rewrites the whole file) is not the bottleneck.
BATCH_SIZE = 10

#: Matches build_index.py, so the corpus stays consistent with what is already
#: indexed rather than mixing two different lookback windows.
LOOKBACK_DAYS = 730
FORMS = ["10-K", "10-Q", "8-K"]


def load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - a corrupt state file is just a restart
            pass
    return {"done": [], "failed": {}, "chunks": 0}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=1), encoding="utf-8")


def seed_from_json(state: dict) -> None:
    """Import an existing chunk_lookup.json into SQLite, once.

    Without this the new store would hold only the companies this script
    indexed, and `Retriever.load_default` prefers SQLite when it exists - so
    the seventeen companies already in the FAISS index would vanish from
    retrieval while their vectors sat in the index unreachable.

    The embedding_id on each chunk already points at its FAISS row, so the
    imported rows stay aligned with the vectors that are there.
    """
    from backend.rag.index.build_index import CHUNK_LOOKUP_PATH
    from backend.rag.index.chunk_store import ChunkStore
    from data.schemas.filing import FilingChunk

    if not CHUNK_LOOKUP_PATH.exists():
        return

    store = ChunkStore(STORE_PATH)
    store.create()
    if store.count() > 0:
        store.close()
        return

    print(f"importing existing {CHUNK_LOOKUP_PATH.name} into SQLite ...")
    raw = json.loads(CHUNK_LOOKUP_PATH.read_text(encoding="utf-8"))
    chunks = [FilingChunk(**data) for data in raw.values()]
    written = store.add(chunks)
    existing = sorted({c.ticker for c in chunks})
    store.close()

    for ticker in existing:
        if ticker not in state["done"]:
            state["done"].append(ticker)
    state["chunks"] = state.get("chunks", 0) + written
    save_state(state)
    print(f"imported {written:,} chunks covering {len(existing)} companies")


def universe_tickers() -> list:
    """Every name with price data, which is what the alpha model scores."""
    from data.pipelines.prices import load_profiles

    profiles = load_profiles()
    return sorted({str(t) for t in profiles["ticker"].dropna() if str(t).strip()})


def process_batch(tickers: list, since: date) -> int:
    """Fetch, chunk, embed and store one batch. Returns chunks added."""
    from backend.rag.chunking import chunk_filing
    from backend.rag.embeddings import EmbeddingCache
    from backend.rag.index.chunk_store import ChunkStore
    from backend.rag.vectorstore import VectorStore
    from data.pipelines.edgar import load_cached_filing_text, pull_filings_for_tickers
    from data.sources.sec_edgar import EdgarClient

    client = EdgarClient(user_agent=settings.require_sec_user_agent())
    filings_by_ticker = pull_filings_for_tickers(
        tickers, forms=FORMS, client=client, since=since
    )

    chunks = []
    for ticker, filings in filings_by_ticker.items():
        for filing in filings:
            try:
                raw = load_cached_filing_text(ticker, filing.accession_no)
            except Exception as exc:  # noqa: BLE001 - one bad document is not fatal
                log.warning("skipping %s %s: %s", ticker, filing.accession_no, exc)
                continue
            chunks.extend(chunk_filing(filing, raw))

    if not chunks:
        return 0

    embeddings = EmbeddingCache().get_or_embed(chunks)

    # Append to the existing index rather than rebuilding it. The row a vector
    # lands on is its embedding_id, and those rows must keep matching the ids
    # file, so the store is loaded, extended and saved as a unit.
    dim = len(next(iter(embeddings.values())))
    store = VectorStore(dim=dim, index_path=FAISS_PATH)
    if FAISS_PATH.exists():
        store.load()

    ordered_ids = [c.chunk_id for c in chunks if c.chunk_id in embeddings]
    base = len(store.ids)
    store.add(ids=ordered_ids, vectors=[embeddings[cid] for cid in ordered_ids])
    store.save()

    position = {cid: base + i for i, cid in enumerate(ordered_ids)}
    for chunk in chunks:
        chunk.embedding_id = position.get(chunk.chunk_id)

    chunk_store = ChunkStore(STORE_PATH)
    chunk_store.create()
    written = chunk_store.add(c for c in chunks if c.chunk_id in position)
    chunk_store.close()
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the retrieval index in batches")
    ap.add_argument("--limit", type=int, default=None, help="only this many tickers")
    ap.add_argument("--only", type=str, default=None, help="comma-separated tickers")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reset", action="store_true", help="forget progress and start over")
    args = ap.parse_args()

    if args.reset and STATE_PATH.exists():
        STATE_PATH.unlink()
        print("progress reset (the index and embedding cache are left alone)")

    state = load_state()
    seed_from_json(state)
    done = set(state["done"])

    tickers = (
        [t.strip().upper() for t in args.only.split(",") if t.strip()]
        if args.only
        else universe_tickers()
    )
    todo = [t for t in tickers if t not in done]
    if args.limit:
        todo = todo[: args.limit]

    print(f"universe {len(tickers)}  done {len(done)}  remaining {len(todo)}")
    if state["chunks"]:
        print(f"chunks indexed so far: {state['chunks']:,}")

    if args.dry_run:
        print("next batch:", ", ".join(todo[: args.batch_size]) or "(nothing)")
        return 0

    if not todo:
        print("Nothing to do. Everything in the universe is indexed.")
        return 0

    since = date.today() - timedelta(days=LOOKBACK_DAYS)
    started = time.monotonic()

    for i in range(0, len(todo), args.batch_size):
        batch = todo[i : i + args.batch_size]
        n = i // args.batch_size + 1
        total_batches = (len(todo) + args.batch_size - 1) // args.batch_size
        print(f"\n[{n}/{total_batches}] {', '.join(batch)}", flush=True)

        try:
            added = process_batch(batch, since)
        except KeyboardInterrupt:
            print("\ninterrupted - progress is saved, rerun to resume")
            return 130
        except Exception as exc:  # noqa: BLE001 - one bad batch must not end the run
            log.exception("batch failed")
            for t in batch:
                state["failed"][t] = str(exc)[:200]
            save_state(state)
            continue

        state["done"].extend(batch)
        state["chunks"] += added
        save_state(state)

        elapsed = time.monotonic() - started
        rate = (i + len(batch)) / elapsed if elapsed else 0
        remaining = (len(todo) - i - len(batch)) / rate if rate else 0
        print(
            f"      +{added:,} chunks  ({state['chunks']:,} total)  "
            f"eta {remaining/60:.0f} min",
            flush=True,
        )

    print(f"\ndone: {state['chunks']:,} chunks across {len(state['done'])} companies")
    if state["failed"]:
        print(f"{len(state['failed'])} failed: {', '.join(sorted(state['failed']))}")
        print("Rerun to retry them.")
    print(f"index  {FAISS_PATH}")
    print(f"chunks {STORE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
