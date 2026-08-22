"""SEC filings and their retrievable chunks. Step 1.4 of the data contract.

Every chunk carries source_url and filed_date all the way through retrieval.
Strip them at any stage and the audit trail becomes unrecoverable downstream.
"""
from __future__ import annotations

from datetime import date as Date
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class FormType(str, Enum):
    TEN_K = "10-K"
    TEN_Q = "10-Q"
    EIGHT_K = "8-K"
    DEF_14A = "DEF 14A"
    EARNINGS_RELEASE = "EARNINGS_RELEASE"
    TRANSCRIPT = "TRANSCRIPT"
    PRESENTATION = "PRESENTATION"
    OTHER = "OTHER"


class Filing(BaseModel):
    accession_no: str
    ticker: str
    cik: Optional[str] = None
    form_type: FormType
    filed_date: Date = Field(..., description="Date EDGAR received it - the as-of key")
    period_of_report: Optional[Date] = None
    url: str = Field(..., description="Public URL a judge can click")
    title: Optional[str] = None


class FilingChunk(BaseModel):
    chunk_id: str
    accession_no: str
    ticker: str
    form_type: FormType
    section: Optional[str] = Field(None, description="e.g. 'Item 7 - MD&A'")
    text: str

    # Provenance - never drop these two.
    source_url: str
    filed_date: Date

    char_start: Optional[int] = None
    char_end: Optional[int] = None
    embedding_id: Optional[int] = Field(None, description="Row index in the FAISS index")

    def is_known_at(self, as_of: Date) -> bool:
        """Retrieval must filter on this. See plan step C11."""
        return self.filed_date <= as_of
