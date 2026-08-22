from __future__ import annotations
import re
from datetime import date

from data.schemas.catalyst import Catalyst, SourceType, Direction  # adjust import if your schema module differs
from backend.research.event_extraction.filings import ExtractedEvent
from backend.rag.retrieval.citations import Citation

# SEC form types that map to SourceType.SEC_FILING. Extend this if you add
# forms beyond 10-K/10-Q/8-K in C7's pull_filings_for_tickers.
_SEC_FORM_TYPES = {"10-K", "10-Q", "8-K", "S-1", "DEF 14A"}


def _parse_event_date(raw, fallback: date) -> date:
    """Parse a model-supplied event date, tolerantly.

    `event_date` comes out of an LLM, so it is untrusted input: we have seen
    "2025-08" (no day), and full prose is equally possible. A strict
    fromisoformat() raises and kills the whole run - which showed up as the
    pipeline randomly producing zero catalysts depending on which chunks were
    retrieved.

    Anything unparseable falls back to the filing date, which is always real.
    That is the conservative direction: event_date is cosmetic, while
    source_date drives the as-of leakage check and is never touched here.
    """
    if not raw:
        return fallback

    text = str(raw).strip()
    try:
        return date.fromisoformat(text)
    except ValueError:
        pass

    # "2025-08" -> first of that month.
    m = re.fullmatch(r"(\d{4})-(\d{1,2})", text)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), 1)
        except ValueError:
            return fallback

    # "2025" -> start of that year.
    if re.fullmatch(r"\d{4}", text):
        return date(int(text), 1, 1)

    return fallback


def _source_type_for(citation: Citation) -> SourceType:
    """
    Maps a citation's form_type to the Catalyst enum. Right now every
    citation flowing through here comes from C12's to_citations() over SEC
    filings, so this only handles that path - extend when C14's
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
# actually wants - this is a reasonable starting default, not gospel.
_EVENT_TYPE_DIRECTION = {
    "guidance_change": Direction.NEUTRAL,  # ambiguous without reading the description - leave to LLM/manual review
    "capex_commentary": Direction.NEUTRAL,
    "segment_shift": Direction.NEUTRAL,
    "margin_commentary": Direction.NEUTRAL,
    "buyback_or_dividend": Direction.BULLISH,
    "management_change": Direction.NEUTRAL,
    "other": Direction.NEUTRAL,
}


def events_to_catalysts(
    events: list[ExtractedEvent],
    citation_lookup: dict[str, Citation],
    ticker: str,
) -> list[Catalyst]:
    """
    Converts extracted events into Catalyst objects, pulling the verbatim
    quote directly from the source chunk (never let the LLM paraphrase the
    quote field - that's how an unsupported claim slips in).

    citation_lookup: {chunk_id: Citation}, built from C12's to_citations()
    over the same chunks that were fed into extract_events.

    Raises if a citation is missing for an event's chunk_id - a Catalyst
    without a source_url must never ship, so fail loudly here rather than
    silently dropping the source.
    """
    catalysts: list[Catalyst] = []

    for i, event in enumerate(events):
        citation = citation_lookup.get(event.chunk_id)
        if citation is None:
            raise ValueError(
                f"No citation found for chunk_id={event.chunk_id!r} - "
                "every Catalyst must carry a real source_url, refusing to "
                "fabricate one."
            )

        # event_date = when the thing actually happened (from extraction, if
        # the text stated it). source_date = when the filing became public -
        # THIS is the field is_known_at() checks for the as-of leakage rule,
        # so it must be citation.filed_date, never left empty. Falls back to
        # source_date when the model couldn't find an explicit event date.
        event_date = _parse_event_date(event.event_date, citation.filed_date)

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
            direction=_EVENT_TYPE_DIRECTION.get(event.event_type, Direction.NEUTRAL),
            confidence=_confidence_for_event_type(event.event_type),
        )
        catalysts.append(catalyst)

    return catalysts


#: Words too common to identify a passage.
_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "was", "were", "has",
    "have", "had", "are", "its", "our", "their", "will", "would", "been",
    "company", "quarter", "year", "revenue", "increase", "increased",
}


def _sentences(text: str) -> list:
    """Split into candidate sentences, discarding table debris.

    Filing text extracted from HTML is full of stripped table cells - runs of
    numbers, dollar signs and percent signs with no prose. Quoting those makes
    the audit trail look broken, so they are filtered out here rather than
    shown to a judge.
    """
    import re

    raw = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
    out = []
    for s in raw:
        s = " ".join(s.split())

        # Strip HTML-extraction furniture that prefixes a real sentence:
        # a leading page number, "Table of contents", running headers. Left
        # in, a quote reads "37 Table of contents We expanded our..." which
        # looks like a scraping bug rather than a citation.
        s = re.sub(
            r"^\s*\d{0,4}\s*(table of contents|index to financial statements|"
            r"form 10-[kq]|part [ivx]+|item \d+[a-z]?\.?)\s*",
            "", s, flags=re.IGNORECASE,
        ).strip()
        s = re.sub(r"^\d{1,4}\s+(?=[A-Z])", "", s).strip()

        if len(s) < 40:
            continue
        letters = sum(c.isalpha() or c.isspace() for c in s)
        if letters / len(s) < 0.80:
            continue  # mostly digits and symbols: a table row
        out.append(s)
    return out


def _extract_verbatim_snippet(source_text: str, description: str, max_len: int = 300) -> str:
    """Pull the sentence from the source that best matches the claim.

    Never trusts the LLM's paraphrase for the quote field - the quote must be
    text that actually appears in the filing, or the audit trail is fiction.
    What this adds over taking the chunk head is relevance: it scores real
    sentences by keyword overlap with the extracted claim, so each catalyst
    quotes the passage it was drawn from instead of five catalysts all
    repeating the same opening table.
    """
    sentences = _sentences(source_text)
    if not sentences:
        # No prose worth quoting. Fall back to the raw head rather than
        # returning nothing, since Catalyst requires a non-empty quote.
        snippet = " ".join(source_text.split())
        return snippet[:max_len].rsplit(" ", 1)[0] + "..." if len(snippet) > max_len else snippet

    keywords = {
        w.strip(".,;:()%$").lower()
        for w in description.split()
        if len(w) > 3 and w.strip(".,;:()%$").lower() not in _STOPWORDS
    }

    def score(sentence: str) -> int:
        low = sentence.lower()
        return sum(1 for k in keywords if k in low)

    best = max(sentences, key=score)
    if score(best) == 0:
        best = sentences[0]  # nothing matched; at least quote real prose

    return best[:max_len].rsplit(" ", 1)[0] + "..." if len(best) > max_len else best


def _confidence_for_event_type(event_type: str) -> float:
    # filings-sourced events get higher default confidence than the vaguer
    # 'other' bucket - tune this once you see real extraction output
    return 0.5 if event_type == "other" else 0.75


if __name__ == "__main__":
    # After this runs cleanly end-to-end on real data, message Nalin -
    # his nlp.py (B8) is stubbed returning zeros until this lands.
    print("Wire this up against a real chunk + extraction run to smoke-test.")