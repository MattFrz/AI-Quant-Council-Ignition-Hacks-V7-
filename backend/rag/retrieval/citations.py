from __future__ import annotations
from dataclasses import dataclass
from datetime import date

from data.schemas.filing import FilingChunk


@dataclass
class Citation:
    text: str
    source_url: str
    filed_date: date
    form_type: str
    section: str


def to_citation(chunk: FilingChunk, form_type: str) -> Citation:
    """
    Thin formatting layer. Field names here map 1:1 onto what Catalyst
    (schema 1.5) expects for source_url / source_type / event_date, so C15
    can build a Catalyst directly from this with no reshaping.

    form_type isn't on FilingChunk itself (it's on the parent Filing) —
    pass it through from wherever you have the Filing object, or look it up
    via chunk.parent (the accession number) against your filings cache/index.
    """
    return Citation(
        text=chunk.text,
        source_url=chunk.source_url,
        filed_date=chunk.filed_date,
        form_type=form_type,
        section=chunk.section,
    )


def to_citations(chunks: list[FilingChunk], form_type_lookup: dict[str, str]) -> list[Citation]:
    """
    Batch version. form_type_lookup maps accession (chunk.parent) -> form_type,
    e.g. built once from your filings cache: {f.accession: f.form_type for f in filings}
    """
    return [
        to_citation(chunk, form_type_lookup.get(chunk.parent, "UNKNOWN"))
        for chunk in chunks
    ]