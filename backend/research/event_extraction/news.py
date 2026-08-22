from __future__ import annotations

from backend.agents.llm_client import LLMClient
from backend.research.event_extraction.filings import (
    extract_events,
    ExtractedEvent,
)

# Same note as transcripts.py: reuses filings.py's extraction logic. News
# articles are NOT filings, so they don't have a filed_date/source_url in the
# same schema shape as FilingChunk — whatever news source you pull from
# (C14 build order doesn't specify one), make sure you wrap its output into
# something carrying a real URL and a publish date before this function runs,
# or C15's Catalyst objects downstream will be missing required fields.


def extract_events_from_news(
    article_chunk_text: str, chunk_id: str, llm: LLMClient
) -> list[ExtractedEvent]:
    return extract_events(article_chunk_text, chunk_id, llm)