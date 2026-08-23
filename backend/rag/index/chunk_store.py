"""SQLite-backed chunk lookup, for indexes too large to hold as JSON.

`chunk_lookup.json` is read with a single `json.loads`, which means the whole
corpus is parsed into memory before the first query. At 17 companies that is
48 MB and merely wasteful. At 500 it is about 1.6 GB of JSON that becomes
several GB of Python objects, which is the difference between a server that
starts and one that is killed.

This stores the same fields in SQLite instead. A retrieval touches the handful
of rows it actually returns, so memory no longer scales with the corpus.

The JSON path is left in place: a small index still works exactly as before,
and `Retriever` prefers whichever exists.
"""
from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from data.schemas.filing import FilingChunk

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id     TEXT PRIMARY KEY,
    accession_no TEXT NOT NULL,
    ticker       TEXT NOT NULL,
    form_type    TEXT NOT NULL,
    section      TEXT,
    text         TEXT NOT NULL,
    source_url   TEXT NOT NULL,
    filed_date   TEXT NOT NULL,
    char_start   INTEGER,
    char_end     INTEGER,
    embedding_id INTEGER
);
CREATE INDEX IF NOT EXISTS idx_chunks_ticker ON chunks(ticker);
CREATE INDEX IF NOT EXISTS idx_chunks_filed  ON chunks(filed_date);
"""

_COLUMNS = (
    "chunk_id, accession_no, ticker, form_type, section, text, "
    "source_url, filed_date, char_start, char_end, embedding_id"
)


def _row_to_chunk(row: sqlite3.Row) -> FilingChunk:
    return FilingChunk(
        chunk_id=row["chunk_id"],
        accession_no=row["accession_no"],
        ticker=row["ticker"],
        form_type=row["form_type"],
        section=row["section"],
        text=row["text"],
        source_url=row["source_url"],
        filed_date=date.fromisoformat(row["filed_date"]),
        char_start=row["char_start"],
        char_end=row["char_end"],
        embedding_id=row["embedding_id"],
    )


class ChunkStore:
    """Read/write access to the chunk table. Cheap to construct."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            # WAL keeps a long write from blocking reads, which matters while
            # a multi-hour build is running and something else wants to query.
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    def create(self) -> None:
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def add(self, chunks: Iterable[FilingChunk]) -> int:
        rows = [
            (
                c.chunk_id,
                c.accession_no,
                c.ticker,
                c.form_type.value if hasattr(c.form_type, "value") else str(c.form_type),
                c.section,
                c.text,
                c.source_url,
                c.filed_date.isoformat(),
                c.char_start,
                c.char_end,
                c.embedding_id,
            )
            for c in chunks
        ]
        if not rows:
            return 0
        # INSERT OR REPLACE so a resumed build that reprocesses a ticker does
        # not fail on the primary key.
        self.conn.executemany(
            f"INSERT OR REPLACE INTO chunks ({_COLUMNS}) VALUES ({','.join('?' * 11)})",
            rows,
        )
        self.conn.commit()
        return len(rows)

    def get_many(self, chunk_ids: List[str]) -> Dict[str, FilingChunk]:
        """Only the rows asked for. This is what keeps memory flat."""
        if not chunk_ids:
            return {}
        out: Dict[str, FilingChunk] = {}
        # SQLite caps variables per statement, so page through the ids.
        for i in range(0, len(chunk_ids), 500):
            page = chunk_ids[i : i + 500]
            cur = self.conn.execute(
                f"SELECT {_COLUMNS} FROM chunks WHERE chunk_id IN ({','.join('?' * len(page))})",
                page,
            )
            for row in cur:
                out[row["chunk_id"]] = _row_to_chunk(row)
        return out

    def tickers(self) -> List[str]:
        cur = self.conn.execute("SELECT DISTINCT ticker FROM chunks ORDER BY ticker")
        return [r["ticker"] for r in cur]

    def count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"])

    def filings(self) -> Dict[str, dict]:
        """One row per filing, for building the citation lookup.

        DISTINCT over accession_no rather than loading every chunk: the filing
        metadata repeats on each chunk, and there are far fewer filings.
        """
        cur = self.conn.execute(
            "SELECT accession_no, ticker, form_type, source_url, "
            "MIN(filed_date) AS filed_date FROM chunks GROUP BY accession_no"
        )
        return {r["accession_no"]: dict(r) for r in cur}

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
