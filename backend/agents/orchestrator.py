from __future__ import annotations
from datetime import date

from backend.agents.llm_client import LLMClient
from backend.agents.state import ResearchState
from backend.agents.research.planner import ResearchPlanner
from backend.agents.research.research_agent import ResearchAgent
from backend.agents.quant_validator.validator import QuantValidator
from backend.agents.bull.bull_analyst import BullAnalyst
from backend.agents.bear.bear_analyst import BearAnalyst
from backend.agents.portfolio_manager.manager import PortfolioManager
from backend.rag.retrieval.retriever import Retriever
from data.schemas.trade_idea import TradeIdea

# NOTE on sequencing: BearAnalyst (C19) reads state.backtest_result, so
# QuantValidator MUST run before it. BullAnalyst doesn't need quant results,
# so it can run either before or after QuantValidator - placed before here
# so the bull case forms independent of the numbers, matching the spec's
# "grounded only in retrieved evidence" framing for the bull side.


#: Chunks sent for event extraction. One LLM call per chunk, so this is the
#: main cost dial on a run. 12 is enough to find catalysts across a handful of
#: filings without turning a demo into a four-minute wait.
_MAX_EXTRACTION_CHUNKS = 12


def _company_name(ticker: str) -> str:
    """Real company name from the cached profiles - the model must not guess.

    It guessed "Verint" for VRT, which is a different company (VRNT).
    """
    try:
        from data.pipelines.prices import load_profiles

        profiles = load_profiles()
        match = profiles.loc[profiles["ticker"] == ticker, "name"]
        if len(match) and match.iloc[0]:
            return str(match.iloc[0])
    except Exception:  # noqa: BLE001
        pass
    return ticker


def _extract_catalysts(state, retriever, llm, universe):
    """Retrieved chunks -> Catalyst objects. The section 3 audit trail.

    Sits between the researcher and the analysts because both bull and bear
    argue from catalysts, and the PortfolioManager cannot name a primary
    ticker without them.

    Every catalyst keeps a verbatim quote and a real source_url; a chunk whose
    parent filing is missing is skipped rather than cited to nothing.
    """
    from backend.rag.retrieval.citations import to_citation
    from backend.research.catalysts.extractor import events_to_catalysts
    from backend.research.event_extraction.filings import extract_events_batch

    chunks = state.retrieved_chunks or []
    if not chunks:
        state.emit("extract_catalysts", "Extracting catalysts", "done", "no chunks retrieved")
        return state

    state.emit("extract_catalysts", "Extracting catalysts", "in_progress")

    filing_lookup = retriever.filing_lookup
    citation_lookup = {}
    seen = set()
    ordered = []
    for chunk in chunks:
        if chunk.chunk_id in seen:
            continue
        seen.add(chunk.chunk_id)
        filing = filing_lookup.get(chunk.accession_no)
        if filing is None:
            continue  # no verifiable source, no citation
        citation_lookup[chunk.chunk_id] = to_citation(chunk, filing)
        ordered.append(chunk)

    # Extract from the CANDIDATES first.
    #
    # Retrieval queries the thesis, not the shortlist, so it happily returns
    # chunks about companies the alpha model never ranked. Extracting from
    # those wastes LLM calls and produces catalysts that get filtered out
    # later - one run found 22 catalysts across 2 companies and kept 1,
    # because the evidence was about a name that was not a candidate.
    if universe:
        preferred = [c for c in ordered if c.ticker in set(universe)]
        if preferred:
            ordered = preferred + [c for c in ordered if c.ticker not in set(universe)]

    ordered = ordered[:_MAX_EXTRACTION_CHUNKS]
    if not ordered:
        state.emit("extract_catalysts", "Extracting catalysts", "done", "no citable chunks")
        return state

    events = extract_events_batch([(c.chunk_id, c.text) for c in ordered], llm)

    # A catalyst must carry the ticker of the filing it came from, NOT the
    # candidate we happened to be researching. Stamping one ticker across every
    # event produces an audit trail that says "PLTR" and links to an NVDA
    # 10-Q - which a judge spots the moment they click, and which discredits
    # every other claim on the page.
    chunk_ticker = {c.chunk_id: c.ticker for c in ordered}
    catalysts = []
    by_ticker = {}
    for event in events:
        by_ticker.setdefault(chunk_ticker.get(event.chunk_id, "UNKNOWN"), []).append(event)

    for tkr, group in by_ticker.items():
        if tkr == "UNKNOWN":
            continue  # unattributable evidence is dropped, not relabelled
        catalysts.extend(events_to_catalysts(group, citation_lookup, tkr))

    state.catalysts = catalysts

    # Decide the subject BEFORE the analysts run.
    #
    # Previously the manager chose it afterwards, so bull and bear argued the
    # thesis in general - producing a "VRT" card whose bull case was about
    # NVDA and AMD, and which called Vertiv "Verint". Naming the company up
    # front makes both sides argue the thing we are actually recommending.
    if catalysts:
        counts = {}
        for c in catalysts:
            counts[c.ticker] = counts.get(c.ticker, 0) + 1
        state.primary_ticker = max(counts, key=counts.get)
        state.primary_name = _company_name(state.primary_ticker)

    state.emit(
        "extract_catalysts", "Extracting catalysts", "done",
        f"{len(catalysts)} catalysts across {len(by_ticker)} companies",
    )
    return state


def run_pipeline(
    thesis: str,
    as_of: date,
    llm: LLMClient,
    retriever: Retriever,
    form_type_lookup: dict[str, str] | None = None,
    universe: list | None = None,
    factor_scores: dict | None = None,
    on_event=None,
) -> TradeIdea:
    """
    This is what Phase 3's shared services/pipeline.py (3.1) calls into for
    the "research + debate" portion of the §1 chain. Confirm this exact
    signature - (thesis: str, as_of: date) -> TradeIdea - against whoever
    writes pipeline.py before Phase 3 integration.

    Universe and factor_scores are expected to already be populated on the
    state by Lane A/B code upstream of this call - see the TODO below for
    where that handoff needs to be wired once pipeline.py exists.
    """
    state = ResearchState(thesis=thesis, as_of=as_of, on_event=on_event)

    # Lane A/B handoff (the TODO this file carried until 3.1 landed).
    # services/pipeline.py builds the universe and scores it, then passes both
    # in here. QuantValidator raises rather than fabricating if they are
    # missing, which is the behaviour we want - a validator that invents a
    # Sharpe ratio is worse than one that refuses to run.
    state.universe = universe
    state.factor_scores = factor_scores

    planner = ResearchPlanner(llm)
    # ResearchAgent takes (llm, retriever) only - it resolves form types via
    # retriever.filing_lookup, so form_type_lookup is not threaded through here.
    researcher = ResearchAgent(llm, retriever)
    validator = QuantValidator(llm)
    bull = BullAnalyst(llm)
    bear = BearAnalyst(llm)
    manager = PortfolioManager(llm)

    state = planner.run(state)
    state = researcher.run(state)
    state = _extract_catalysts(state, retriever, llm, universe)
    state = bull.run(state)
    state = validator.run(state)
    state = bear.run(state)

    trade_idea = manager.run(state)
    return trade_idea


if __name__ == "__main__":
    import os
    llm = LLMClient(api_key=os.environ["LLM_API_KEY"])
    retriever = Retriever.load_default()
    result = run_pipeline(
        thesis="NVDA data center revenue growth is underpriced",
        as_of=date.today(),
        llm=llm,
        retriever=retriever,
        form_type_lookup={},  # populate from your filings cache in real use
    )
    print(result)