"""The live research timeline. Step 1.10 of the data contract.

One event type drives the entire section 16 checklist - the component that makes
the system read as autonomous rather than as a form that returns JSON.

Every agent step emits one of these. The frontend consumes them over SSE.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


# The canonical pipeline, in order. Section 16 of the plan.
# The UI renders this list greyed out on load, then lights each row as events arrive.
PIPELINE_STEPS: List[str] = [
    "parse_thesis",
    "define_criteria",
    "scan_universe",
    "identify_candidates",
    "retrieve_filings",
    "analyze_transcripts",
    "extract_catalysts",
    "generate_bull",
    "generate_bear",
    "run_event_study",
    "backtest_signal",
    "calculate_risk",
    "final_recommendation",
]

STEP_LABELS: Dict[str, str] = {
    "parse_thesis": "Parsed investment thesis",
    "define_criteria": "Defined screening criteria",
    "scan_universe": "Scanned universe",
    "identify_candidates": "Identified candidates",
    "retrieve_filings": "Retrieved 10-K / 10-Q filings",
    "analyze_transcripts": "Analyzed earnings transcripts",
    "extract_catalysts": "Extracted catalysts",
    "generate_bull": "Generated bull thesis",
    "generate_bear": "Generated bear thesis",
    "run_event_study": "Ran historical event study",
    "backtest_signal": "Backtested signal",
    "calculate_risk": "Calculated portfolio risk",
    "final_recommendation": "Generated final recommendation",
}


class ResearchEvent(BaseModel):
    step_id: str
    label: str = Field(..., description="Shown verbatim in the timeline")
    status: StepStatus = StepStatus.PENDING
    detail: Optional[str] = Field(None, description="e.g. scanned 1,247 companies")
    progress: Optional[float] = Field(None, ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def start(cls, step_id: str, detail: Optional[str] = None) -> "ResearchEvent":
        return cls(
            step_id=step_id,
            label=STEP_LABELS.get(step_id, step_id),
            status=StepStatus.RUNNING,
            detail=detail,
        )

    @classmethod
    def done(cls, step_id: str, detail: Optional[str] = None) -> "ResearchEvent":
        return cls(
            step_id=step_id,
            label=STEP_LABELS.get(step_id, step_id),
            status=StepStatus.DONE,
            detail=detail,
        )

    @classmethod
    def failed(cls, step_id: str, detail: str) -> "ResearchEvent":
        return cls(
            step_id=step_id,
            label=STEP_LABELS.get(step_id, step_id),
            status=StepStatus.FAILED,
            detail=detail,
        )


def initial_timeline() -> List[ResearchEvent]:
    """All steps pending - what the UI shows before the run starts."""
    return [
        ResearchEvent(step_id=s, label=STEP_LABELS[s], status=StepStatus.PENDING)
        for s in PIPELINE_STEPS
    ]
