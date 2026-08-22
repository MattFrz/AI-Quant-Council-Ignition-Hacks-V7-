from __future__ import annotations
from fastapi import APIRouter

from backend.api.schemas import ThesisRequest, JobHandle
from backend.services.job_runner import start_job

router = APIRouter(prefix="/research", tags=["research"])


@router.post("", response_model=JobHandle)
def start_research(request: ThesisRequest):
    """
    Third stage of the §1 pipeline order (3.2). start_job never blocks —
    it either replays a cached result instantly or spins up a background
    thread and returns right away (see job_runner.py's own docstring).
    """
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
        stream_url=f"/research/stream/{job.job_id}",
        from_cache=job.from_cache,
    )