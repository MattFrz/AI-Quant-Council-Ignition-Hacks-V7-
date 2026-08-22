from __future__ import annotations
import re
import uuid

from data.schemas.filing import Filing, FilingChunk

_ITEM_HEADER_RE = re.compile(
    r"^\s*(ITEM\s+\d+[A-Z]?\.?.*)$", re.IGNORECASE | re.MULTILINE
)

MAX_CHUNK_TOKENS = 500
OVERLAP_TOKENS = 50
_APPROX_CHARS_PER_TOKEN = 4

# Table-of-contents "Item 7. ..." lines match the same regex as real
# section headers but sit only a few dozen characters apart from each
# other. Real sections are, at minimum, a few hundred characters of body
# text. Anything shorter than this is almost certainly a TOC entry, not
# a real section, and gets folded into the following section instead of
# becoming its own near-empty chunk.
_MIN_SECTION_CHARS = 300


def chunk_filing(filing: Filing, raw_text: str) -> list[FilingChunk]:
    """
    embedding_id is deliberately left None here - it's the chunk's row index
    in the FAISS index, which doesn't exist until build_index.py actually
    builds it. build_index.py sets it after the fact, not this function.
    """
    sections = _split_by_sections(raw_text)
    chunks: list[FilingChunk] = []

    for section_name, section_offset, section_text in sections:
        for piece, local_start, local_end in split_section(section_text):
            chunks.append(FilingChunk(
                chunk_id=str(uuid.uuid4()),
                accession_no=filing.accession_no,
                ticker=filing.ticker,
                form_type=filing.form_type,
                section=section_name,
                text=piece,
                source_url=filing.url,
                filed_date=filing.filed_date,
                char_start=section_offset + local_start,
                char_end=section_offset + local_end,
            ))

    return chunks


def _split_by_sections(raw_text: str) -> list[tuple[str, int, str]]:
    """Returns (section_label, offset_into_raw_text, section_text)."""
    matches = list(_ITEM_HEADER_RE.finditer(raw_text))
    if not matches:
        return [("FULL_DOCUMENT", 0, raw_text)]

    raw_sections = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
        header = match.group(1).strip()[:80]
        raw_sections.append((header, start, raw_text[start:end]))

    return _merge_short_sections(raw_sections)


def _merge_short_sections(
    raw_sections: list[tuple[str, int, str]]
) -> list[tuple[str, int, str]]:
    if not raw_sections:
        return raw_sections

    merged: list[tuple[str, int, str]] = []
    pending_label = None
    pending_offset = None
    pending_text_parts: list[str] = []

    def flush() -> None:
        nonlocal pending_label, pending_offset, pending_text_parts
        if pending_text_parts:
            merged.append((pending_label, pending_offset, "".join(pending_text_parts)))
        pending_label, pending_offset, pending_text_parts = None, None, []

    for label, offset, text in raw_sections:
        if len(text) >= _MIN_SECTION_CHARS:
            # A real section on its own - flush any accumulated TOC junk
            # first, then keep this section standalone with its own label.
            flush()
            merged.append((label, offset, text))
            continue

        if pending_label is None:
            pending_label, pending_offset = label, offset
        pending_text_parts.append(text)

    flush()
    return merged


def split_section(
    section_text: str,
    max_tokens: int = MAX_CHUNK_TOKENS,
    overlap: int = OVERLAP_TOKENS,
) -> list[tuple[str, int, int]]:
    """
    Returns (piece_text, local_start, local_end) tuples - offsets relative
    to section_text, not the raw filing. chunk_filing adds the section's
    own offset to get char_start/char_end relative to the whole document.
    """
    max_chars = max_tokens * _APPROX_CHARS_PER_TOKEN
    overlap_chars = overlap * _APPROX_CHARS_PER_TOKEN

    text = section_text.strip()
    leading_ws = len(section_text) - len(section_text.lstrip())

    if len(text) <= max_chars:
        return [(text, leading_ws, leading_ws + len(text))] if text else []

    pieces = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            last_period = text.rfind(". ", start, end)
            if last_period != -1 and last_period > start:
                end = last_period + 1
        piece = text[start:end].strip()
        if piece:
            pieces.append((piece, leading_ws + start, leading_ws + end))
        start = end - overlap_chars if end - overlap_chars > start else end

    return pieces