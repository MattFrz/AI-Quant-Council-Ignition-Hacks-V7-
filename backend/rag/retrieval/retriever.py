from __future__ import annotations
import json
from datetime import date

from backend.rag.vectorstore import VectorStore
from backend.rag.embeddings import EmbeddingCache
from data.schemas.filing import Filing, FilingChunk
from backend.rag.index.build_index import INDEX_PATH, CHUNK_LOOKUP_PATH

_OVERFETCH_MULTIPLIER = 3


class Retriever:
    def __init__(
        self,
        store: VectorStore,
        chunk_lookup: dict[str, FilingChunk] | None = None,
        chunk_store=None,
    ):
        """Chunks come from a dict, or from SQLite for a large corpus.

        The dict is simplest and is what a seventeen-company index uses. Past a
        few hundred thousand chunks it stops being viable - the JSON alone is
        gigabytes before it becomes Python objects - so `chunk_store` reads the
        handful of rows a query actually returns instead.
        """
        self.store = store
        self.chunk_lookup = chunk_lookup if chunk_lookup is not None else {}
        self.chunk_store = chunk_store
        self._embedder = EmbeddingCache()
        self._filing_lookup: dict[str, Filing] | None = None

    def _chunks_for(self, chunk_ids: list[str]) -> dict[str, FilingChunk]:
        if self.chunk_store is not None:
            return self.chunk_store.get_many(chunk_ids)
        return {cid: self.chunk_lookup[cid] for cid in chunk_ids if cid in self.chunk_lookup}

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

            if self.chunk_store is not None:
                # One row per filing from SQL, rather than walking every chunk.
                # There are orders of magnitude fewer filings than chunks, and
                # loading all the chunk text here would undo the whole point.
                from datetime import date as _date

                for accession_no, row in self.chunk_store.filings().items():
                    filings[accession_no] = Filing(
                        accession_no=accession_no,
                        ticker=row["ticker"],
                        form_type=row["form_type"],
                        filed_date=_date.fromisoformat(row["filed_date"]),
                        url=row["source_url"],
                    )
            else:
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
        """Prefer the SQLite store, fall back to the JSON lookup.

        Both describe the same corpus. SQLite exists because a large index
        cannot be held in memory; the JSON is what a small one still ships as,
        including the deployment bundle, so neither path is going away.
        """
        from backend.rag.index.chunk_store import ChunkStore

        store = VectorStore(dim=0, index_path=INDEX_PATH)
        store.load()

        sqlite_path = INDEX_PATH.parent / "chunks.sqlite"
        if sqlite_path.exists():
            return cls(store=store, chunk_store=ChunkStore(sqlite_path))

        chunk_raw = json.loads(CHUNK_LOOKUP_PATH.read_text())
        chunk_lookup = {
            chunk_id: FilingChunk(**data) for chunk_id, data in chunk_raw.items()
        }
        return cls(store=store, chunk_lookup=chunk_lookup)

    def retrieve(self, query: str, as_of: date, k: int = 10) -> list[FilingChunk]:
        """
        CRITICAL: filed_date now lives directly on FilingChunk (confirmed by
        the runtime schema, not the earlier paste) - filter reads it off the
        chunk directly, no join needed.
        """
        query_vector = self._embedder.embed_query(query)
        raw_results = self.store.search(query_vector, k=k * _OVERFETCH_MULTIPLIER)

        # Keep the ranking FAISS returned: _chunks_for is a lookup, not a sort,
        # and a dict does not promise to hand them back in score order.
        ranked_ids = [chunk_id for chunk_id, _ in raw_results]
        found = self._chunks_for(ranked_ids)
        chunks = [found[cid] for cid in ranked_ids if cid in found]
        filtered = [c for c in chunks if c.is_known_at(as_of)]

        return filtered[:k]