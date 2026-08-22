"""The whole section 1 chain, front to back. Step 3.1.

    thesis -> criteria -> universe -> screen -> retrieve -> catalysts
           -> factors -> alpha -> backtest -> risk -> debate -> TradeIdea

Deliberately a plain function with no HTTP anywhere. Routes (3.2-3.4) call this;
it never imports FastAPI. That means the whole system can be exercised from a
terminal, which is how you debug it at 3am and how test_pipeline.py drives it.

Run it directly:

    python -m backend.services.pipeline "Find mispriced beneficiaries of AI
    data-center spending"
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from backend.config import settings
from backend.core.logging import get_logger
from backend.services.events import ResearchEvent
from data.pipelines.prices import load_prices, load_profiles, load_wide
from data.schemas.trade_idea import Side, TradeIdea, Verdict
from quant.universe.builder import UniverseResult, build_universe

log = get_logger(__name__)

EmitFn = Callable[[ResearchEvent], None]


@dataclass
class PipelineResult:
    """Everything one run produced. The route layer projects this into the
    Phase 1 response schemas; nothing here knows about HTTP."""

    thesis: str
    as_of: date
    top_idea: Optional[TradeIdea] = None
    runners_up: List[TradeIdea] = field(default_factory=list)
    universe: Optional[UniverseResult] = None
    criteria: Any = None
    events: List[ResearchEvent] = field(default_factory=list)
    llm_cost_usd: Optional[float] = None
    elapsed_s: float = 0.0
    degraded: List[str] = field(default_factory=list)

    #: Funnel stages, captured at run time. Held separately from `universe` so
    #: a cached result can be rebuilt without serialising the whole
    #: UniverseResult (which carries every dropped ticker at every stage).
    funnel_stages: List[dict] = field(default_factory=list)

    def funnel(self) -> List[dict]:
        if self.funnel_stages:
            return self.funnel_stages
        return self.universe.funnel() if self.universe else []


class Pipeline:
    """One run of the research chain.

    Every stage emits a ResearchEvent so the section 16 timeline animates
    live. Stages that depend on an unavailable lane degrade explicitly and
    record why in `degraded` - the run continues, but nothing is fabricated to
    fill the gap.
    """

    def __init__(self, emit: Optional[EmitFn] = None) -> None:
        self._emit = emit or (lambda e: None)
        self.events: List[ResearchEvent] = []
        self.degraded: List[str] = []

    # ---------------------------------------------------------------- events

    def emit(self, step_id: str, status: str = "running", detail: Optional[str] = None) -> None:
        from backend.services.events import STEP_LABELS, StepStatus

        event = ResearchEvent(
            step_id=step_id,
            label=STEP_LABELS.get(step_id, step_id),
            status=StepStatus(status),
            detail=detail,
        )
        self.events.append(event)
        self._emit(event)
        log.info("[%s] %s%s", status, event.label, f" - {detail}" if detail else "")

    def _degrade(self, stage: str, reason: str) -> None:
        """Record a missing capability instead of inventing output for it."""
        msg = f"{stage}: {reason}"
        self.degraded.append(msg)
        log.warning("degraded - %s", msg)

    # ------------------------------------------------------------------ run

    def run(
        self,
        thesis: str,
        as_of: Optional[date] = None,
        max_candidates: int = 7,
        universe_size: Optional[int] = None,
    ) -> PipelineResult:
        started = time.monotonic()
        as_of = as_of or date.today()

        result = PipelineResult(thesis=thesis, as_of=as_of)

        criteria = self._parse_thesis(thesis)
        result.criteria = criteria

        universe = self._scan_universe(criteria, universe_size)
        result.universe = universe
        result.funnel_stages = universe.funnel() if universe else []

        candidates = self._select_candidates(universe, max_candidates)

        idea = self._research_and_debate(thesis, as_of, universe, candidates)
        result.top_idea = idea

        result.events = self.events
        result.degraded = self.degraded
        result.elapsed_s = time.monotonic() - started

        self.emit(
            "final_recommendation",
            "done",
            f"{idea.ticker} {idea.side.value}" if idea else "no idea produced",
        )
        log.info("pipeline finished in %.1fs (%d degraded stages)",
                 result.elapsed_s, len(self.degraded))
        return result

    # --------------------------------------------------------------- stages

    def _parse_thesis(self, thesis: str) -> Any:
        self.emit("parse_thesis", "running")
        try:
            from backend.agents.llm_client import LLMClient
            from backend.research.thesis.decomposer import decompose_thesis

            criteria = decompose_thesis(thesis, LLMClient(api_key=settings.require_llm_key()))
            self.emit("parse_thesis", "done", "decomposed into structured criteria")
            self.emit("define_criteria", "done")
            return criteria
        except Exception as exc:  # noqa: BLE001
            self._degrade("parse_thesis", str(exc)[:120])
            self.emit("parse_thesis", "failed", str(exc)[:80])
            return None

    def _scan_universe(self, criteria: Any, universe_size: Optional[int]) -> Optional[UniverseResult]:
        self.emit("scan_universe", "running")
        try:
            panel = load_prices()
            profiles = load_profiles()
        except FileNotFoundError as exc:
            self._degrade("scan_universe", f"{exc} - run scripts/seed_data.py")
            self.emit("scan_universe", "failed", "no cached market data")
            return None

        universe = build_universe(panel, profiles, max_size=universe_size or settings.universe_size)
        self.emit(
            "scan_universe", "done",
            f"{universe.scanned} scanned, {len(universe.tickers)} in universe",
        )
        return universe

    def _select_candidates(self, universe: Optional[UniverseResult], max_candidates: int) -> List[str]:
        self.emit("identify_candidates", "running")
        if universe is None or not universe.tickers:
            self._degrade("identify_candidates", "no universe to rank")
            self.emit("identify_candidates", "failed")
            return []

        scores = self._score_universe(universe.tickers)
        if scores is None or scores.empty:
            ranked = list(universe.tickers)
            self._degrade("identify_candidates", "no alpha model - took first N by liquidity")
        else:
            ranked = scores.sort_values(ascending=False).index.tolist()

        # Only research companies we can actually cite.
        #
        # The alpha model ranks across the whole universe, but the RAG index
        # holds filings for a handful of names. Researching an unindexed
        # candidate produces an audit trail built from some OTHER company's
        # filings - which is worse than no audit trail, because it looks
        # authoritative and is wrong.
        citable = self._indexed_tickers()
        if citable:
            preferred = [t for t in ranked if t in citable]
            if preferred:
                candidates = preferred[:max_candidates]
                self.emit(
                    "identify_candidates", "done",
                    f"{len(candidates)} candidates with indexed filings "
                    f"(of {len(citable)} covered companies)",
                )
                return candidates
            self._degrade(
                "identify_candidates",
                f"no ranked candidate has indexed filings; index covers "
                f"{sorted(citable)}",
            )

        candidates = ranked[:max_candidates]
        self.emit("identify_candidates", "done", f"{len(candidates)} candidates")
        return candidates

    def _indexed_tickers(self) -> set:
        """Tickers with filings in the RAG index, or an empty set if none."""
        try:
            import json

            from backend.rag.index.build_index import CHUNK_LOOKUP_PATH

            lookup = json.loads(CHUNK_LOOKUP_PATH.read_text(encoding="utf-8"))
            return {v.get("ticker") for v in lookup.values() if v.get("ticker")}
        except Exception as exc:  # noqa: BLE001
            log.debug("no chunk lookup available (%s)", exc)
            return set()

    def _score_universe(self, tickers: List[str]) -> Optional[pd.Series]:
        """Cross-sectional alpha scores from Lane B, latest date."""
        try:
            from quant.factors.base import Panel
            from quant.signals.generation import SignalEngine

            # Panel.from_wide takes the WidePanel object itself - Nalin built it
            # against load_wide()'s return type, not against separate frames.
            wide = load_wide(tickers=tickers)
            panel = Panel.from_wide(wide)
            engine = SignalEngine.default().build(panel).use_equal_weights(panel)
            scores = engine.scores()          # method, not a property
            if scores is None or scores.empty:
                return None
            return scores.iloc[-1].dropna()
        except Exception as exc:  # noqa: BLE001
            self._degrade("alpha_model", str(exc)[:120])
            return None

    def _research_and_debate(
        self,
        thesis: str,
        as_of: date,
        universe: Optional[UniverseResult],
        candidates: List[str],
    ) -> Optional[TradeIdea]:
        """Retrieval, catalysts, bull/bear, quant validation, final synthesis."""
        if not candidates:
            self._degrade("research", "no candidates to research")
            return None

        try:
            from backend.agents.llm_client import LLMClient
            from backend.agents.orchestrator import run_pipeline as run_agents
            from backend.rag.retrieval.retriever import Retriever

            llm = LLMClient(api_key=settings.require_llm_key())
            retriever = Retriever.load_default()

            self.emit("retrieve_filings", "running")

            # QuantValidator needs the universe and its scores on state - it
            # calls into quant/ and refuses to run without them rather than
            # inventing numbers. Hand over the candidates we just ranked.
            scores = self._score_universe(candidates)
            factor_scores = (
                {t: float(v) for t, v in scores.items()}
                if scores is not None and not scores.empty
                else {t: 1.0 for t in candidates}
            )

            idea = run_agents(
                thesis=thesis,
                as_of=as_of,
                llm=llm,
                retriever=retriever,
                universe=candidates,
                factor_scores=factor_scores,
            )
            self.emit("extract_catalysts", "done")
            self.emit("generate_bull", "done")
            self.emit("generate_bear", "done")
            self.emit("backtest_signal", "done")
            self.emit("calculate_risk", "done")
            return idea

        except Exception as exc:  # noqa: BLE001
            self._degrade("research_and_debate", str(exc)[:160])
            self.emit("retrieve_filings", "failed", str(exc)[:80])
            return self._quant_only_idea(as_of, universe, candidates)

    def _quant_only_idea(
        self,
        as_of: date,
        universe: Optional[UniverseResult],
        candidates: List[str],
    ) -> Optional[TradeIdea]:
        """Fallback when the agent layer is unavailable.

        Produces a real backtested, risk-measured idea with NO catalysts and
        NO bull/bear case, and says so. Degrading to an honest partial result
        beats either crashing the demo or inventing evidence: `validator_verdict`
        stays INCONCLUSIVE and `has_audit_trail()` returns False, so nothing
        downstream can mistake this for a researched idea.
        """
        self.emit("backtest_signal", "running", "agent layer unavailable, quant only")
        try:
            from quant.api import compute_metrics, compute_risk_metrics, run_backtest
            from quant.risk.metrics import build_risk_metrics

            ticker = candidates[0]
            run = run_backtest(candidates, {t: 1.0 for t in candidates}, as_of)
            result = compute_metrics(run)
            risk = build_risk_metrics(run.returns)

            self.emit("backtest_signal", "done", f"sharpe {result.sharpe:.2f}")
            self.emit("calculate_risk", "done")

            profiles = load_profiles()
            name = profiles.loc[profiles["ticker"] == ticker, "name"]

            return TradeIdea(
                idea_id=f"quant-only-{as_of.isoformat()}-{ticker}",
                ticker=ticker,
                company_name=str(name.iloc[0]) if len(name) else ticker,
                side=Side.LONG,
                as_of=as_of,
                alpha_score=0.0,
                confidence=0.0,
                catalysts=[],
                backtest=result,
                risk=risk,
                validator_verdict=Verdict.INCONCLUSIVE,
                pm_rationale=(
                    "QUANT ONLY. The research and debate layer was unavailable, so "
                    "this idea carries no catalysts, no bull case and no bear case. "
                    "Backtest and risk figures are real; the recommendation is not "
                    "supported by evidence and must not be presented as researched. "
                    f"Degraded stages: {'; '.join(self.degraded)}"
                ),
            )
        except Exception as exc:  # noqa: BLE001
            self._degrade("quant_only_fallback", str(exc)[:160])
            self.emit("backtest_signal", "failed", str(exc)[:80])
            return None


def run_pipeline(
    thesis: str,
    as_of: Optional[date] = None,
    max_candidates: int = 7,
    universe_size: Optional[int] = None,
    emit: Optional[EmitFn] = None,
) -> PipelineResult:
    """Module-level entry point. This is what routes and tests call."""
    return Pipeline(emit=emit).run(
        thesis=thesis, as_of=as_of, max_candidates=max_candidates, universe_size=universe_size
    )


if __name__ == "__main__":
    import sys

    thesis_arg = " ".join(sys.argv[1:]) or (
        "Find companies benefiting from accelerating AI data-center spending "
        "that the market may be underpricing."
    )

    res = run_pipeline(thesis_arg)

    print(f"\n{'=' * 66}")
    print(f"  {res.thesis}")
    print(f"{'=' * 66}")
    if res.universe:
        for stage in res.funnel():
            print(f"  {stage['count']:>6}  {stage['label']}")
    if res.top_idea:
        i = res.top_idea
        print(f"\n  TOP IDEA  {i.ticker} {i.side.value}")
        print(f"    alpha score     {i.alpha_score}/10")
        print(f"    confidence      {i.confidence:.0%}")
        print(f"    catalysts       {len(i.catalysts)}")
        print(f"    verdict         {i.validator_verdict.value}")
        print(f"    audit trail     {'yes' if i.has_audit_trail() else 'NO'}")
        if i.backtest:
            print(f"    sharpe          {i.backtest.sharpe:.2f}")
    else:
        print("\n  no idea produced")
    if res.degraded:
        print(f"\n  DEGRADED STAGES ({len(res.degraded)}):")
        for d in res.degraded:
            print(f"    - {d}")
    print(f"\n  {res.elapsed_s:.1f}s\n")
