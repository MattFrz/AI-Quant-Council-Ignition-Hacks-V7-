from __future__ import annotations
from datetime import date

from data.schemas.catalyst import Catalyst, SourceType  # adjust import if your schema module differs
from backend.research.event_extraction.filings import ExtractedEvent
from backend.rag.retrieval.citations import Citation

# SEC form types that map to SourceType.SEC_FILING. Extend this if you add
# forms beyond 10-K/10-Q/8-K in C7's pull_filings_for_tickers.
_SEC_FORM_TYPES = {"10-K", "10-Q", "8-K", "S-1", "DEF 14A"}


def _source_type_for(citation: Citation) -> SourceType:
    """
    Maps a citation's form_type to the Catalyst enum. Right now every
    citation flowing through here comes from C12's to_citations() over SEC
    filings, so this only handles that path — extend when C14's
    transcript/news citations start flowing through the same function
    (they'll need their own form_type values, e.g. "TRANSCRIPT" / "NEWS",
    set at the point those citations are built).
    """
    if citation.form_type in _SEC_FORM_TYPES:
        return SourceType.SEC_FILING
    if citation.form_type == "TRANSCRIPT":
        return SourceType.TRANSCRIPT
    if citation.form_type == "NEWS":
        return SourceType.NEWS
    return SourceType.OTHER

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

    for i, event in enumerate(events):
        citation = citation_lookup.get(event.chunk_id)
        if citation is None:
            raise ValueError(
                f"No citation found for chunk_id={event.chunk_id!r} — "
                "every Catalyst must carry a real source_url, refusing to "
                "fabricate one."
            )

        # event_date = when the thing actually happened (from extraction, if
        # the text stated it). source_date = when the filing became public —
        # THIS is the field is_known_at() checks for the as-of leakage rule,
        # so it must be citation.filed_date, never left empty. Falls back to
        # source_date when the model couldn't find an explicit event date.
        event_date = (
            date.fromisoformat(event.event_date) if event.event_date else citation.filed_date
        )

        # index suffix guards against two events extracted from the same
        # chunk colliding on catalyst_id
        catalyst = Catalyst(
            catalyst_id=f"{ticker}-{event.chunk_id}-{i}",
            ticker=ticker,
            headline=event.description,
            quote=_extract_verbatim_snippet(citation.text, event.description),
            source_type=_source_type_for(citation),
            source_url=citation.source_url,
            source_date=citation.filed_date,
            event_date=event_date,
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