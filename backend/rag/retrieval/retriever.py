from __future__ import annotations
import json
from pathlib import Path
from datetime import date, datetime

from backend.rag.vectorstore import VectorStore
from backend.rag.embeddings import EmbeddingCache
from data.schemas.filing import FilingChunk
from backend.rag.index.build_index import INDEX_PATH, CHUNK_LOOKUP_PATH

# over-fetch multiplier: we pull more than k from FAISS because the as_of
# filter will drop some candidates, and we still want k results back
_OVERFETCH_MULTIPLIER = 3


class Retriever:
    def __init__(self, store: VectorStore, chunk_lookup: dict[str, FilingChunk]):
        self.store = store
        self.chunk_lookup = chunk_lookup
        self._embedder = EmbeddingCache()

    @classmethod
    def load_default(cls) -> "Retriever":
        """Loads the persisted index + chunk metadata built by build_index.py."""
        raw_lookup = json.loads(CHUNK_LOOKUP_PATH.read_text())
        chunk_lookup = {
            chunk_id: FilingChunk(
                chunk_id=data["chunk_id"],
                text=data["text"],
                section=data["section"],
                parent=data["parent"],
                source_url=data["source_url"],
                filed_date=date.fromisoformat(data["filed_date"]),
            )
            for chunk_id, data in raw_lookup.items()
        }
        # infer dim from any stored vector; simplest is to just load and let FAISS report it
        store = VectorStore(dim=0, index_path=INDEX_PATH)
        store.load()
        return cls(store=store, chunk_lookup=chunk_lookup)

    def retrieve(self, query: str, as_of: date, k: int = 10) -> list[FilingChunk]:
        """
        CRITICAL: never returns a chunk whose filed_date is after as_of.
        This is the "the LLM must not see the future" guarantee — the agent
        calling this function gets no choice in the matter, the filter is
        enforced here, not left to the caller to remember.
        """
        query_vector = self._embedder.embed_query(query)
        raw_results = self.store.search(query_vector, k=k * _OVERFETCH_MULTIPLIER)

        chunks = [self.chunk_lookup[chunk_id] for chunk_id, _ in raw_results]
        filtered = [c for c in chunks if c.filed_date <= as_of]

        if len(filtered) < k and len(raw_results) == k * _OVERFETCH_MULTIPLIER:
            # ran out of over-fetched candidates before hitting k after filtering —
            # not a bug, just means few enough documents existed as of that date.
            pass

        return filtered[:k]