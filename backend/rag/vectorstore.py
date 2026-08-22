from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import faiss


class VectorStore:
    """
    FAISS wrapper using inner product over L2-normalized vectors, which is
    equivalent to cosine similarity. Persists both the index and the
    id -> position mapping to disk (FAISS itself only stores vectors + ints).
    """

    def __init__(self, dim: int, index_path: Path):
        self.dim = dim
        self.index_path = Path(index_path)
        self.index = faiss.IndexFlatIP(dim)
        self.ids: list[str] = []  # position i in self.ids <-> vector at row i

    def add(self, ids: list[str], vectors: list[list[float]]) -> None:
        arr = np.array(vectors, dtype="float32")
        faiss.normalize_L2(arr)  # required for IP to behave like cosine similarity
        self.index.add(arr)
        self.ids.extend(ids)

    def search(self, query_vector: list[float], k: int) -> list[tuple[str, float]]:
        query = np.array([query_vector], dtype="float32")
        faiss.normalize_L2(query)
        scores, positions = self.index.search(query, k)

        results = []
        for pos, score in zip(positions[0], scores[0]):
            if pos == -1:  # FAISS pads with -1 if fewer than k results exist
                continue
            results.append((self.ids[pos], float(score)))
        return results

    def save(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        ids_path = self.index_path.with_suffix(".ids.json")
        ids_path.write_text(json.dumps(self.ids))

    def load(self) -> None:
        self.index = faiss.read_index(str(self.index_path))
        ids_path = self.index_path.with_suffix(".ids.json")
        self.ids = json.loads(ids_path.read_text())

    @classmethod
    def load_from(cls, dim: int, index_path: Path) -> "VectorStore":
        store = cls(dim=dim, index_path=index_path)
        store.load()
        return store