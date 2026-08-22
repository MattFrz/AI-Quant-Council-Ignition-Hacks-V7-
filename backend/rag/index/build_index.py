from __future__ import annotations
import json
from pathlib import Path

from backend.config import settings
from backend.rag.chunking import chunk_filing
from backend.rag.embeddings import EmbeddingCache
from backend.rag.vectorstore import VectorStore
from data.pipelines.edgar import pull_filings_for_tickers, load_cached_filing_text
from data.sources.sec_edgar import EdgarClient

INDEX_PATH = Path(settings.data_cache_dir) / "index" / "filings.faiss"
CHUNK_LOOKUP_PATH = Path(settings.data_cache_dir) / "index" / "chunk_lookup.json"


def build_index(tickers: list[str], forms: list[str] = None) -> None:
    """
    Run once as: python -m backend.rag.index.build_index
    """
    forms = forms or ["10-K", "10-Q", "8-K"]
    client = EdgarClient(user_agent=settings.sec_user_agent)

    filings_by_ticker = pull_filings_for_tickers(tickers, forms=forms, client=client)

    all_chunks = []
    for ticker, filings in filings_by_ticker.items():
        for filing in filings:
            raw_text = load_cached_filing_text(ticker, filing.accession_no)
            all_chunks.extend(chunk_filing(filing, raw_text))

    print(f"Chunked {len(all_chunks)} pieces")

    cache = EmbeddingCache()
    embeddings = cache.get_or_embed(all_chunks)

    dim = len(next(iter(embeddings.values())))
    store = VectorStore(dim=dim, index_path=INDEX_PATH)

    chunk_ids_in_order = list(embeddings.keys())
    store.add(ids=chunk_ids_in_order, vectors=[embeddings[cid] for cid in chunk_ids_in_order])
    store.save()

    chunk_by_id = {c.chunk_id: c for c in all_chunks}
    for row_index, chunk_id in enumerate(store.ids):
        chunk_by_id[chunk_id].embedding_id = row_index

    chunk_lookup_json = {c.chunk_id: c.model_dump(mode="json") for c in all_chunks}
    CHUNK_LOOKUP_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHUNK_LOOKUP_PATH.write_text(json.dumps(chunk_lookup_json))

    print(f"Index built: {len(all_chunks)} vectors, saved to {INDEX_PATH}")


if __name__ == "__main__":
    build_index(tickers=["AAPL", "MSFT", "NVDA"])