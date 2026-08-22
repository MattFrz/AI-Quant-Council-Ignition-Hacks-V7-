from __future__ import annotations

from backend.agents.llm_client import LLMClient
from backend.research.event_extraction.filings import (
    extract_events,
    ExtractedEvent,
)

# Reuses the same extraction logic/prompt as filings.py — transcripts are
# noisier text but the extraction shape (event_type + description) is
# identical. Build this only after filings.py (C13) is solid and tested;
# this is lower priority per the build order (filings are free, structured,
# and impossible to argue with — transcripts are neither).


def extract_events_from_transcript(
    transcript_chunk_text: str, chunk_id: str, llm: LLMClient
) -> list[ExtractedEvent]:
    """
    Thin pass-through for now. If transcript-specific noise (filler words,
    Q&A cross-talk, analyst names) turns out to hurt extraction quality,
    swap in a transcript-specific prompt in prompts.py rather than editing
    the shared EVENT_EXTRACTION_PROMPT used by filings.py.
    """
    return extract_events(transcript_chunk_text, chunk_id, llm)