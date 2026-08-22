from __future__ import annotations
import json
from pathlib import Path

from backend.config import settings
from backend.rag.chunking import chunk_filing
from backend.rag.embeddings import EmbeddingCache
from backend.rag.vectorstore import VectorStore
from data.pipelines.edgar import pull_filings_for_tickers, load_cached_filing_text
from data.sources.sec_edgar import EdgarClient

INDEX_PATH = Path(settings.DATA_CACHE_DIR) / "index" / "filings.faiss"
CHUNK_LOOKUP_PATH = Path(settings.DATA_CACHE_DIR) / "index" / "chunk_lookup.json"


def build_index(tickers: list[str], forms: list[str] = None) -> None:
    """
    Run once (or whenever the filing universe changes) as:
        python -m backend.rag.index.build_index
    Loads cached filings, chunks them, embeds, and writes a persisted FAISS index.
    """
    forms = forms or ["10-K", "10-Q", "8-K"]
    client = EdgarClient(user_agent=settings.SEC_USER_AGENT)

    filings_by_ticker = pull_filings_for_tickers(tickers, forms=forms, client=client)

    all_chunks = []
    for ticker, filings in filings_by_ticker.items():
        for filing in filings:
            raw_text = load_cached_filing_text(ticker, filing.accession)
            all_chunks.extend(chunk_filing(filing, raw_text))

    print(f"Chunked {len(all_chunks)} pieces from {sum(len(f) for f in filings_by_ticker.values())} filings")

    cache = EmbeddingCache()
    embeddings = cache.get_or_embed(all_chunks)

    dim = len(next(iter(embeddings.values())))
    store = VectorStore(dim=dim, index_path=INDEX_PATH)
    store.add(ids=list(embeddings.keys()), vectors=list(embeddings.values()))
    store.save()

    # persist chunk metadata separately so the retriever can look up full
    # FilingChunk objects by id after a FAISS search returns ids
    chunk_lookup = {c.chunk_id: c.__dict__ for c in all_chunks}
    CHUNK_LOOKUP_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHUNK_LOOKUP_PATH.write_text(json.dumps(chunk_lookup, default=str))

    print(f"Index built: {len(all_chunks)} vectors, saved to {INDEX_PATH}")


if __name__ == "__main__":
    # adjust to your actual candidate ticker list, or pull from the universe builder
    build_index(tickers=["AAPL", "MSFT", "NVDA"])