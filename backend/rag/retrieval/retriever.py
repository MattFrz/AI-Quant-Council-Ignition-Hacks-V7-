from __future__ import annotations
import json
from datetime import date

from backend.rag.vectorstore import VectorStore
from backend.rag.embeddings import EmbeddingCache
from data.schemas.filing import Filing, FilingChunk
from backend.rag.index.build_index import INDEX_PATH, CHUNK_LOOKUP_PATH

_OVERFETCH_MULTIPLIER = 3


class Retriever:
    def __init__(self, store: VectorStore, chunk_lookup: dict[str, FilingChunk]):
        self.store = store
        self.chunk_lookup = chunk_lookup
        self._embedder = EmbeddingCache()
        self._filing_lookup: dict[str, Filing] | None = None

    @property
    def filing_lookup(self) -> dict[str, Filing]:
        """{accession_no: Filing}, as to_citations() expects.

        Derived from the chunks rather than stored separately: every chunk
        already carries accession_no, ticker, form_type, filed_date and
        source_url, which is exactly a Filing. Keeping one source of truth
        means a citation can never disagree with the chunk it came from.
        """
        if self._filing_lookup is None:
            filings: dict[str, Filing] = {}
            for chunk in self.chunk_lookup.values():
                if chunk.accession_no in filings:
                    continue
                filings[chunk.accession_no] = Filing(
                    accession_no=chunk.accession_no,
                    ticker=chunk.ticker,
                    form_type=chunk.form_type,
                    filed_date=chunk.filed_date,
                    url=chunk.source_url,
                )
            self._filing_lookup = filings
        return self._filing_lookup

    @classmethod
    def load_default(cls) -> "Retriever":
        chunk_raw = json.loads(CHUNK_LOOKUP_PATH.read_text())
        chunk_lookup = {
            chunk_id: FilingChunk(**data) for chunk_id, data in chunk_raw.items()
        }
        store = VectorStore(dim=0, index_path=INDEX_PATH)
        store.load()
        return cls(store=store, chunk_lookup=chunk_lookup)

    def retrieve(self, query: str, as_of: date, k: int = 10) -> list[FilingChunk]:
        """
        CRITICAL: filed_date now lives directly on FilingChunk (confirmed by
        the runtime schema, not the earlier paste) - filter reads it off the
        chunk directly, no join needed.
        """
        query_vector = self._embedder.embed_query(query)
        raw_results = self.store.search(query_vector, k=k * _OVERFETCH_MULTIPLIER)

        chunks = [self.chunk_lookup[chunk_id] for chunk_id, _ in raw_results]
        filtered = [c for c in chunks if c.is_known_at(as_of)]

        return filtered[:k]