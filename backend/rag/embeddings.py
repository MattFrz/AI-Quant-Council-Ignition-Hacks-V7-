from __future__ import annotations
import hashlib
import json
from pathlib import Path
from openai import OpenAI

from data.schemas.filing import FilingChunk
from backend.config import settings

EMBEDDING_MODEL = "text-embedding-3-small"  # cheap + good enough; confirm w/ team budget
BATCH_SIZE = 100


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EmbeddingCache:
    def __init__(self, cache_dir: Path | None = None, api_key: str | None = None):
        self.cache_dir = Path(cache_dir or settings.data_cache_dir) / "embeddings"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.client = OpenAI(api_key=api_key or settings.llm_api_key)

    def _cache_file(self, text_hash: str) -> Path:
        return self.cache_dir / f"{text_hash}.json"

    def get_or_embed(self, chunks: list[FilingChunk]) -> dict[str, list[float]]:
        """
        Returns {chunk_id: embedding_vector}. Checks disk cache by content hash
        before calling the API — never re-embeds identical text.
        """
        result: dict[str, list[float]] = {}
        to_embed: list[FilingChunk] = []

        for chunk in chunks:
            text_hash = _hash_text(chunk.text)
            cache_file = self._cache_file(text_hash)
            if cache_file.exists():
                result[chunk.chunk_id] = json.loads(cache_file.read_text())
            else:
                to_embed.append(chunk)

        for i in range(0, len(to_embed), BATCH_SIZE):
            batch = to_embed[i : i + BATCH_SIZE]
            vectors = embed_batch([c.text for c in batch])
            for chunk, vector in zip(batch, vectors):
                text_hash = _hash_text(chunk.text)
                self._cache_file(text_hash).write_text(json.dumps(vector))
                result[chunk.chunk_id] = vector

        return result

    def embed_query(self, query: str) -> list[float]:
        """No caching for one-off query embeddings — cache is keyed for chunk reuse."""
        return embed_batch([query])[0]


def embed_batch(texts: list[str]) -> list[list[float]]:
    client = OpenAI(api_key=settings.llm_api_key)
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    # response.data is returned in the same order as the input list
    return [item.embedding for item in response.data]