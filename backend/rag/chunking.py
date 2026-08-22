from __future__ import annotations
import re
import uuid

from data.pipelines.edgar import Filing
from data.schemas.filing import FilingChunk  # adjust import if your schema module differs

# SEC 10-K / 10-Q filings use consistent "Item N" headers. This regex catches
# "Item 1.", "Item 1A.", "Item 7.", "ITEM 7A.", etc. at the start of a line.
_ITEM_HEADER_RE = re.compile(
    r"^\s*(ITEM\s+\d+[A-Z]?\.?.*)$", re.IGNORECASE | re.MULTILINE
)

MAX_CHUNK_TOKENS = 500
OVERLAP_TOKENS = 50
_APPROX_CHARS_PER_TOKEN = 4  # rough heuristic, fine for chunk sizing


def chunk_filing(filing: Filing, raw_text: str) -> list[FilingChunk]:
    """
    Splits filing text into section-aware chunks. Every chunk carries
    source_url, filed_date, section, and parent — never drop these.
    """
    sections = _split_by_sections(raw_text)
    chunks: list[FilingChunk] = []

    for section_name, section_text in sections:
        for piece in split_section(section_text):
            chunks.append(FilingChunk(
                chunk_id=str(uuid.uuid4()),
                text=piece,
                section=section_name,
                parent=filing.accession,
                source_url=filing.url,
                filed_date=filing.filed_date,
            ))

    return chunks


def _split_by_sections(raw_text: str) -> list[tuple[str, str]]:
    """
    Splits raw filing text at 'Item N' headers. Falls back to a single
    'FULL_DOCUMENT' section if no headers are found (e.g. 8-Ks are often short
    and unstructured).
    """
    matches = list(_ITEM_HEADER_RE.finditer(raw_text))
    if not matches:
        return [("FULL_DOCUMENT", raw_text)]

    sections = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
        header = match.group(1).strip()[:80]  # keep header label short
        sections.append((header, raw_text[start:end]))

    return sections


def split_section(
    section_text: str,
    max_tokens: int = MAX_CHUNK_TOKENS,
    overlap: int = OVERLAP_TOKENS,
) -> list[str]:
    """
    Sliding-window split within a section that's too long for one chunk.
    Uses a char-count approximation for tokens — fine for sizing, not exact.
    """
    max_chars = max_tokens * _APPROX_CHARS_PER_TOKEN
    overlap_chars = overlap * _APPROX_CHARS_PER_TOKEN

    text = section_text.strip()
    if len(text) <= max_chars:
        return [text] if text else []

    pieces = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        # try not to cut mid-sentence: back up to the last period if there is one nearby
        if end < len(text):
            last_period = text.rfind(". ", start, end)
            if last_period != -1 and last_period > start:
                end = last_period + 1
        pieces.append(text[start:end].strip())
        start = end - overlap_chars if end - overlap_chars > start else end

    return [p for p in pieces if p]