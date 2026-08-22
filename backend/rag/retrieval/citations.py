from __future__ import annotations
from dataclasses import dataclass
from datetime import date

from data.schemas.filing import FilingChunk, Filing


@dataclass
class Citation:
    text: str
    source_url: str
    filed_date: date
    form_type: str
    section: str | None


def to_citation(chunk: FilingChunk, filing: Filing) -> Citation:
    """
    filed_date and source_url come from the parent Filing (chunk doesn't
    carry either). form_type technically lives on both Filing and
    FilingChunk — read it off the chunk since that's the object callers
    already have in hand.
    """
    form_type_value = chunk.form_type.value if hasattr(chunk.form_type, "value") else chunk.form_type
    return Citation(
        text=chunk.text,
        source_url=filing.url,
        filed_date=filing.filed_date,
        form_type=form_type_value,
        section=chunk.section,
    )


def to_citations(chunks: list[FilingChunk], filing_lookup: dict[str, Filing]) -> list[Citation]:
    """
    filing_lookup: {accession_no: Filing} — pass retriever.filing_lookup
    directly from the call site (e.g. research_agent.py).
    """
    citations = []
    for chunk in chunks:
        filing = filing_lookup.get(chunk.accession_no)
        if filing is None:
            continue  # same rule as retriever.py: no verifiable source, no citation
        citations.append(to_citation(chunk, filing))
    return citations