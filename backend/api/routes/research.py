from __future__ import annotations
from fastapi import APIRouter, HTTPException

from backend.api.schemas import (
    CandidateSummary,
    JobHandle,
    JobStatus,
    ResearchResponse,
    ScanResponse,
    ThesisRequest,
)
from backend.config import settings
from backend.services.job_runner import get_job, start_job

router = APIRouter(prefix="/research", tags=["research"])


@router.post("", response_model=JobHandle)
def start_research(request: ThesisRequest):
    """
    Third stage of the §1 pipeline order (3.2). start_job never blocks -
    it either replays a cached result instantly or spins up a background
    thread and returns right away (see job_runner.py's own docstring).
    """
    # A public instance answers from the warm cache only.
    #
    # Every uncached thesis is a real LLM run charged to whoever owns the key,
    # so an open URL without this is a bill that grows with traffic. Refusing
    # with a list of what IS available is friendlier than a rate limit and
    # keeps the demo instant.
    if settings.public_demo_mode:
        from backend.core import cache

        hit = cache.get(
            request.thesis,
            request.as_of,
            request.max_candidates,
            request.universe_size,
        )
        if hit is None:
            raise HTTPException(
                status_code=409,
                detail=(
                    "This deployment answers pre-computed theses only. Pick one of "
                    "the suggested theses, or run the project locally to research "
                    "anything you like."
                ),
            )

    job = start_job(
        thesis=request.thesis,
        as_of=request.as_of,
        max_candidates=request.max_candidates,
        universe_size=request.universe_size,
    )

    return JobHandle(
        job_id=job.job_id,
        status=job.status,
        created_at=job.created_at,
        # Must include the /api mount prefix. stream.ts prepends API_BASE to a
        # relative URL but does not know about the mount point, so "/research/..."
        # resolves to http://localhost:8000/research/... and 404s.
        stream_url=f"/api/research/stream/{job.job_id}",
        from_cache=job.from_cache,
    )


@router.get("/{job_id}", response_model=ResearchResponse)
def get_research(job_id: str):
    """Full result for a finished job.

    Cecile's api.ts calls this as fetchResearchResult(jobId) once the stream
    reports final_recommendation/done.

    Returns 409 rather than a partial body while a job is still running: a
    half-filled ResearchResponse renders as a finished idea with missing
    evidence, which is exactly the confusion the audit trail exists to prevent.
    """
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"unknown job {job_id}")

    if job.status == JobStatus.FAILED:
        raise HTTPException(status_code=500, detail=job.error or "pipeline failed")

    if not job.is_finished or job.result is None:
        raise HTTPException(
            status_code=409, detail=f"job {job_id} is still {job.status.value}"
        )

    result = job.result
    ideas = ([result.top_idea] if result.top_idea else []) + result.runners_up

    return ResearchResponse(
        job_id=job_id,
        thesis=result.thesis,
        as_of=result.as_of,
        top_idea=result.top_idea,
        runners_up=result.runners_up,
        scan=ScanResponse(
            thesis=result.thesis,
            as_of=result.as_of,
            criteria=[],
            funnel=result.funnel(),
            candidates=[
                CandidateSummary(
                    ticker=i.ticker,
                    company_name=i.company_name,
                    sector=i.risk.sector if i.risk else None,
                    alpha_score=i.alpha_score,
                    headline_catalyst=i.catalysts[0].headline if i.catalysts else None,
                )
                for i in ideas
            ],
        ),
        llm_cost_usd=result.llm_cost_usd,
    )