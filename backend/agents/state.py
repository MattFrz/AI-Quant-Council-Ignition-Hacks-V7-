from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date, datetime

from backend.services.events import ResearchEvent  # schema 1.10


#: Agent-facing status words -> the StepStatus values frozen in schema 1.10.
_STATUS_ALIASES = {
    "in_progress": "running",
    "error": "failed",
    "complete": "done",
}


@dataclass
class ResearchState:
    thesis: str
    as_of: date

    # populated progressively as agents run
    criteria: object | None = None          # ThesisCriteria, from C4
    plan: list | None = None                # list[PlanStep], from C17 planner
    universe: list | None = None            # list[Security], from Lane A
    factor_scores: dict | None = None       # from Lane B
    retrieved_chunks: list = field(default_factory=list)   # list[FilingChunk]
    citations: list = field(default_factory=list)          # list[Citation]
    catalysts: list = field(default_factory=list)          # list[Catalyst]
    backtest_result: object | None = None   # BacktestResult, from Lane A via C20
    risk_metrics: dict | None = None        # from Lane A via C20
    bull_case: str | None = None
    bear_case: str | None = None

    events: list[ResearchEvent] = field(default_factory=list)

    def emit(self, step_id: str, label: str, status: str, detail: str | None = None) -> None:
        """
        Appends a ResearchEvent to the running timeline. This is what
        Cecile's D6 live-timeline component streams — every agent should
        call this at the start AND end of any meaningful step, not just
        once at the end, so the UI reads as continuously active rather than
        jumping straight to "done".

        status convention: "in_progress" | "done" | "error"

        Those strings are normalised onto the StepStatus enum frozen in
        Phase 1 (pending/running/done/failed). Without this mapping every
        emit() raises a pydantic ValidationError, because "in_progress" and
        "error" are not members. Mapping here keeps all six agent call sites
        and Cecile's TypeScript mirror unchanged.
        """
        self.events.append(ResearchEvent(
            step_id=step_id,
            label=label,
            status=_STATUS_ALIASES.get(status, status),
            detail=detail,
            timestamp=datetime.utcnow(),
        ))