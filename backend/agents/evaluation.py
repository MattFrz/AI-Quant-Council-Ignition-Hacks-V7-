from __future__ import annotations
from dataclasses import dataclass
from datetime import date

from backend.agents.llm_client import LLMClient
from backend.agents.orchestrator import run_pipeline
from backend.rag.retrieval.retriever import Retriever

# A small fixed set of theses. Keep this list short (3-5) and stable —
# the point is catching regressions after a prompt tweak, not broad coverage.
EVAL_THESES = [
    "NVDA data center revenue growth is underpriced",
    "AAPL services margin expansion is stalling",
    "TSLA energy storage segment is undervalued relative to automotive",
]


@dataclass
class EvalResult:
    thesis: str
    passed: bool
    failure_reason: str | None
    catalyst_count: int
    has_bull_case: bool
    has_bear_case: bool


@dataclass
class EvalReport:
    results: list[EvalResult]

    @property
    def pass_count(self) -> int:
        return sum(1 for r in self.results if r.passed)

    def __str__(self) -> str:
        lines = [f"{self.pass_count}/{len(self.results)} passed"]
        for r in self.results:
            status = "PASS" if r.passed else f"FAIL ({r.failure_reason})"
            lines.append(f"  [{status}] {r.thesis}")
        return "\n".join(lines)


def run_eval(
    llm: LLMClient, retriever: Retriever, form_type_lookup: dict[str, str]
) -> EvalReport:
    """
    Doesn't grade quality — checks structural health: did every stage
    produce non-null output, does every catalyst carry a source_url. That
    catches most regressions (e.g. "I tweaked the bull prompt and now
    bear_case is empty") without needing a human grader in the loop.
    """
    results = []

    for thesis in EVAL_THESES:
        try:
            trade_idea = run_pipeline(
                thesis=thesis,
                as_of=date.today(),
                llm=llm,
                retriever=retriever,
                form_type_lookup=form_type_lookup,
            )
        except Exception as e:
            results.append(EvalResult(
                thesis=thesis, passed=False, failure_reason=str(e),
                catalyst_count=0, has_bull_case=False, has_bear_case=False,
            ))
            continue

        failure_reason = _check_health(trade_idea)
        results.append(EvalResult(
            thesis=thesis,
            passed=failure_reason is None,
            failure_reason=failure_reason,
            catalyst_count=len(trade_idea.catalysts),
            has_bull_case=bool(trade_idea.bull_case),
            has_bear_case=bool(trade_idea.bear_case),
        ))

    return EvalReport(results=results)


def _check_health(trade_idea) -> str | None:
    if not trade_idea.bull_case:
        return "empty bull_case"
    if not trade_idea.bear_case:
        return "empty bear_case"
    if not trade_idea.catalysts:
        return "no catalysts produced"
    for catalyst in trade_idea.catalysts:
        if not catalyst.source_url:
            return f"catalyst missing source_url: {catalyst.headline!r}"
    if trade_idea.backtest is None:
        return "missing backtest result"
    return None


if __name__ == "__main__":
    # run with: python -m backend.agents.evaluation
    import os
    llm = LLMClient(api_key=os.environ["LLM_API_KEY"])
    retriever = Retriever.load_default()
    report = run_eval(llm, retriever, form_type_lookup={})
    print(report)