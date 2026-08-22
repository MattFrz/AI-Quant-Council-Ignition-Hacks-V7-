from __future__ import annotations

from data.schemas.catalyst import Catalyst  # adjust import if your schema module differs
from backend.research.event_extraction.filings import ExtractedEvent
from backend.rag.retrieval.citations import Citation

# Maps our internal event_type vocabulary to Catalyst's expected "direction"
# hint. Adjust as your team's factor model (Nalin's B8) settles on what it
# actually wants — this is a reasonable starting default, not gospel.
_EVENT_TYPE_DIRECTION = {
    "guidance_change": None,  # ambiguous without reading the description — leave to LLM/manual review
    "capex_commentary": None,
    "segment_shift": None,
    "margin_commentary": None,
    "buyback_or_dividend": "up",
    "management_change": None,
    "other": None,
}


def events_to_catalysts(
    events: list[ExtractedEvent],
    citation_lookup: dict[str, Citation],
    ticker: str,
) -> list[Catalyst]:
    """
    Converts extracted events into Catalyst objects, pulling the verbatim
    quote directly from the source chunk (never let the LLM paraphrase the
    quote field — that's how an unsupported claim slips in).

    citation_lookup: {chunk_id: Citation}, built from C12's to_citations()
    over the same chunks that were fed into extract_events.

    Raises if a citation is missing for an event's chunk_id — a Catalyst
    without a source_url must never ship, so fail loudly here rather than
    silently dropping the source.
    """
    catalysts: list[Catalyst] = []

    for event in events:
        citation = citation_lookup.get(event.chunk_id)
        if citation is None:
            raise ValueError(
                f"No citation found for chunk_id={event.chunk_id!r} — "
                "every Catalyst must carry a real source_url, refusing to "
                "fabricate one."
            )

        catalyst = Catalyst(
            ticker=ticker,
            headline=event.description,
            quote=_extract_verbatim_snippet(citation.text, event.description),
            source_type=citation.form_type,
            source_url=citation.source_url,
            event_date=citation.filed_date,
            direction=_EVENT_TYPE_DIRECTION.get(event.event_type),
            confidence=_confidence_for_event_type(event.event_type),
        )
        catalysts.append(catalyst)

    return catalysts


def _extract_verbatim_snippet(source_text: str, description: str, max_len: int = 300) -> str:
    """
    Pulls a verbatim snippet from the actual chunk text rather than trusting
    the LLM's paraphrase in `description`. This is a naive best-effort:
    returns the first max_len chars of the source chunk. If your team wants
    tighter quote-to-claim matching later, replace this with a sentence
    boundary search around keywords from `description`.
    """
    snippet = source_text.strip()
    if len(snippet) > max_len:
        snippet = snippet[:max_len].rsplit(" ", 1)[0] + "..."
    return snippet


def _confidence_for_event_type(event_type: str) -> float:
    # filings-sourced events get higher default confidence than the vaguer
    # 'other' bucket — tune this once you see real extraction output
    return 0.5 if event_type == "other" else 0.75


if __name__ == "__main__":
    # After this runs cleanly end-to-end on real data, message Nalin —
    # his nlp.py (B8) is stubbed returning zeros until this lands.
    print("Wire this up against a real chunk + extraction run to smoke-test.")