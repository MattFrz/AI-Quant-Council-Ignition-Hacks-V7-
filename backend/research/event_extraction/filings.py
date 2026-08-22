from __future__ import annotations
import json
from dataclasses import dataclass

from backend.agents.llm_client import LLMClient
from backend.agents.prompts import EVENT_EXTRACTION_PROMPT
from backend.rag.retrieval.citations import Citation

VALID_EVENT_TYPES = {
    "guidance_change",
    "capex_commentary",
    "segment_shift",
    "margin_commentary",
    "buyback_or_dividend",
    "management_change",
    "other",
}


@dataclass
class ExtractedEvent:
    event_type: str
    description: str
    chunk_id: str  # traces back to the source chunk/citation, never drop this
    event_date: str | None = None  # ISO date string if the text names when the event happened;
    # None if it's not stated. This is DIFFERENT from source_date/filed_date —
    # e.g. a 10-Q filed March 15 can describe a guidance change that happened
    # in February. Leave None rather than guessing; extractor.py falls back
    # to the filing date when this is absent.


def extract_events(chunk_text: str, chunk_id: str, llm: LLMClient) -> list[ExtractedEvent]:
    """
    One LLM call per chunk. Forces structured JSON output: a list of events,
    each with a type (from a fixed vocabulary) and a plain description.
    Returns [] if the chunk contains nothing extraction-worthy — that's a
    valid, common outcome, not an error.
    """
    messages = [
        {"role": "system", "content": EVENT_EXTRACTION_PROMPT},
        {
            "role": "user",
            "content": (
                f"Filing excerpt:\n{chunk_text}\n\n"
                "Extract any notable events as a JSON array. Each item: "
                '{"event_type": one of ' + str(sorted(VALID_EVENT_TYPES)) + ', '
                '"description": "one sentence, factual, no speculation", '
                '"event_date": "ISO date if the text explicitly states when this '
                'happened (e.g. a specific quarter or date), otherwise null — '
                'do not guess or infer a date that is not stated"}. '
                "If there is nothing notable, return an empty array []."
            ),
        },
    ]

    response = llm.complete(messages)
    raw_events = _parse_json_array(response.text)

    events = []
    for item in raw_events:
        event_type = item.get("event_type", "other")
        if event_type not in VALID_EVENT_TYPES:
            event_type = "other"
        events.append(ExtractedEvent(
            event_type=event_type,
            description=item.get("description", "").strip(),
            chunk_id=chunk_id,
            event_date=item.get("event_date") or None,
        ))
    return events


def extract_events_batch(
    chunks: list[tuple[str, str]], llm: LLMClient
) -> list[ExtractedEvent]:
    """chunks: list of (chunk_id, chunk_text). Runs extraction per chunk and flattens."""
    all_events = []
    for chunk_id, text in chunks:
        all_events.extend(extract_events(text, chunk_id, llm))
    return all_events


def _parse_json_array(text: str) -> list[dict]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()

    if not cleaned:
        return []

    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        # extraction found nothing parseable — treat as no events rather than crashing
        return []