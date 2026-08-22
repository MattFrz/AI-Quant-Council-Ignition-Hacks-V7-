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
# so it can run either before or after QuantValidator — placed before here
# so the bull case forms independent of the numbers, matching the spec's
# "grounded only in retrieved evidence" framing for the bull side.


def run_pipeline(
    thesis: str,
    as_of: date,
    llm: LLMClient,
    retriever: Retriever,
    form_type_lookup: dict[str, str],
) -> TradeIdea:
    """
    This is what Phase 3's shared services/pipeline.py (3.1) calls into for
    the "research + debate" portion of the §1 chain. Confirm this exact
    signature — (thesis: str, as_of: date) -> TradeIdea — against whoever
    writes pipeline.py before Phase 3 integration.

    Universe and factor_scores are expected to already be populated on the
    state by Lane A/B code upstream of this call — see the TODO below for
    where that handoff needs to be wired once pipeline.py exists.
    """
    state = ResearchState(thesis=thesis, as_of=as_of)

    # TODO: wire in Matt/Nalin's universe + factor scoring here, e.g.
    # state.universe = build_universe(criteria)
    # state.factor_scores = score_universe(state.universe)
    # This orchestrator assumes those are already on state by the time
    # QuantValidator runs — until pipeline.py exists, populate them manually
    # for local testing.

    planner = ResearchPlanner(llm)
    researcher = ResearchAgent(llm, retriever, form_type_lookup)
    validator = QuantValidator(llm)
    bull = BullAnalyst(llm)
    bear = BearAnalyst(llm)
    manager = PortfolioManager(llm)

    state = planner.run(state)
    state = researcher.run(state)
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