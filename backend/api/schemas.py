"""Request/response wrappers for the HTTP layer. Step 1.9.

These wrap the domain objects in data/schemas - they never redefine them.
"""
from __future__ import annotations

from datetime import date as Date
from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

from data.schemas.trade_idea import TradeIdea


class ThesisRequest(BaseModel):
    thesis: str = Field(
        ...,
        min_length=10,
        description=(
            "Natural language. Example: Find mispriced beneficiaries of "
            "accelerating AI data-center spending."
        ),
    )
    as_of: Optional[Date] = Field(None, description="Backdate the run. None = today.")
    max_candidates: int = Field(7, ge=1, le=50)
    universe_size: Optional[int] = None


class ScreeningCriterion(BaseModel):
    """One measurable criterion the thesis decomposed into."""

    label: str
    metric: str = Field(..., description="Machine-executable field name")
    operator: str = Field(..., description="gt | lt | gte | lte | in | contains")
    value: str
    rationale: Optional[str] = None


class FunnelStage(BaseModel):
    """One row of the section 15 narrowing cascade."""

    label: str
    count: int
    description: Optional[str] = None


class CandidateSummary(BaseModel):
    ticker: str
    company_name: str
    sector: Optional[str] = None
    alpha_score: float
    headline_catalyst: Optional[str] = None


class ScanResponse(BaseModel):
    thesis: str
    as_of: Date
    criteria: List[ScreeningCriterion] = Field(default_factory=list)
    funnel: List[FunnelStage] = Field(default_factory=list)
    candidates: List[CandidateSummary] = Field(default_factory=list)


class ResearchResponse(BaseModel):
    job_id: str
    thesis: str
    as_of: Date
    top_idea: Optional[TradeIdea] = None
    runners_up: List[TradeIdea] = Field(default_factory=list)
    scan: Optional[ScanResponse] = None
    llm_cost_usd: Optional[float] = Field(None, description="Spend for this run")


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class JobHandle(BaseModel):
    job_id: str
    status: JobStatus = JobStatus.QUEUED
    created_at: datetime
    stream_url: str = Field(..., description="SSE endpoint for ResearchEvent")
    from_cache: bool = False
