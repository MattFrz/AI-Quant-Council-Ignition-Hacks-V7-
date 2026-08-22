"""Background pipeline runs. Step 3.6.

A full run takes minutes once retrieval is live, which is far longer than any
sane HTTP timeout. So the route starts a job, returns a job id immediately, and
the client streams events against that id while the work continues.

    handle = start_job("Find mispriced beneficiaries of AI infra spending")
    for event in stream_events(handle.job_id):   # blocks, yields as they land
        ...
    result = get_job(handle.job_id).result

Cache-first: if step 3.7 already has a warmed result for this thesis, the job
completes instantly and replays the stored timeline. Same data, no waiting.
"""
from __future__ import annotations

import queue
import threading
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Dict, Iterator, List, Optional

from backend.api.schemas import JobStatus
from backend.core import cache
from backend.core.logging import get_logger
from backend.services.events import ResearchEvent, StepStatus, initial_timeline
from backend.services.pipeline import PipelineResult, run_pipeline

log = get_logger(__name__)

#: Sentinel pushed onto a job's queue to close its event stream.
_DONE = object()

#: How long a consumer waits for the next event before re-checking job state.
_POLL_TIMEOUT_S = 1.0

#: Completed jobs kept in memory. A hackathon demo will make a handful; this is
#: deliberately not an LRU because losing a finished result mid-demo would be
#: worse than the memory it costs.
_jobs: Dict[str, "Job"] = {}
_lock = threading.Lock()


@dataclass
class Job:
    job_id: str
    thesis: str
    as_of: Optional[date] = None
    status: JobStatus = JobStatus.QUEUED
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: Optional[datetime] = None

    result: Optional[PipelineResult] = None
    error: Optional[str] = None
    from_cache: bool = False

    events: List[ResearchEvent] = field(default_factory=list)
    _queue: "queue.Queue" = field(default_factory=queue.Queue, repr=False)

    @property
    def is_finished(self) -> bool:
        return self.status in (JobStatus.DONE, JobStatus.FAILED)

    @property
    def elapsed_s(self) -> float:
        end = self.finished_at or datetime.now(timezone.utc)
        return (end - self.created_at).total_seconds()

    def _push(self, event: ResearchEvent) -> None:
        self.events.append(event)
        self._queue.put(event)


# --------------------------------------------------------------------- api

def start_job(
    thesis: str,
    as_of: Optional[date] = None,
    max_candidates: int = 7,
    universe_size: Optional[int] = None,
    use_cache: bool = True,
) -> Job:
    """Queue a run and return immediately. Never blocks."""
    job = Job(job_id=uuid.uuid4().hex[:12], thesis=thesis, as_of=as_of)

    with _lock:
        _jobs[job.job_id] = job

    if use_cache:
        hit = cache.get(thesis, as_of, max_candidates, universe_size)
        if hit is not None:
            _complete_from_cache(job, hit)
            return job

    thread = threading.Thread(
        target=_run,
        args=(job, thesis, as_of, max_candidates, universe_size),
        name=f"pipeline-{job.job_id}",
        daemon=True,
    )
    thread.start()
    log.info("job %s started: %s", job.job_id, thesis[:60])
    return job


def get_job(job_id: str) -> Optional[Job]:
    with _lock:
        return _jobs.get(job_id)


def list_jobs() -> List[Job]:
    with _lock:
        return sorted(_jobs.values(), key=lambda j: j.created_at, reverse=True)


def stream_events(job_id: str, include_backlog: bool = True) -> Iterator[ResearchEvent]:
    """Yield events as they occur, then stop when the job finishes.

    `include_backlog` replays everything already emitted first, so a client
    that connects late still renders a complete timeline rather than joining
    mid-run with a half-empty checklist.
    """
    job = get_job(job_id)
    if job is None:
        raise KeyError(f"unknown job {job_id}")

    if include_backlog:
        for event in list(job.events):
            yield event
        if job.is_finished:
            return

    while True:
        try:
            item = job._queue.get(timeout=_POLL_TIMEOUT_S)
        except queue.Empty:
            if job.is_finished:
                return
            continue

        if item is _DONE:
            return
        yield item


def pending_timeline() -> List[ResearchEvent]:
    """All steps in `pending` - what the UI renders before a run starts."""
    return initial_timeline()


def clear_jobs() -> int:
    with _lock:
        n = len(_jobs)
        _jobs.clear()
    return n


# ---------------------------------------------------------------- internals

def _run(
    job: Job,
    thesis: str,
    as_of: Optional[date],
    max_candidates: int,
    universe_size: Optional[int],
) -> None:
    job.status = JobStatus.RUNNING
    try:
        result = run_pipeline(
            thesis=thesis,
            as_of=as_of,
            max_candidates=max_candidates,
            universe_size=universe_size,
            emit=job._push,
        )
        job.result = result
        job.status = JobStatus.DONE

        # Store successful runs so the next request for this thesis is instant.
        if result.top_idea is not None:
            try:
                cache.put(result, max_candidates, universe_size)
            except Exception as exc:  # noqa: BLE001
                log.warning("job %s: could not cache result (%s)", job.job_id, exc)

        log.info("job %s done in %.1fs (%d degraded)",
                 job.job_id, job.elapsed_s, len(result.degraded))

    except Exception as exc:  # noqa: BLE001
        job.status = JobStatus.FAILED
        job.error = f"{type(exc).__name__}: {exc}"
        job._push(ResearchEvent(
            step_id="final_recommendation",
            label="Pipeline failed",
            status=StepStatus.FAILED,
            detail=job.error[:160],
        ))
        log.exception("job %s failed", job.job_id)

    finally:
        job.finished_at = datetime.now(timezone.utc)
        job._queue.put(_DONE)


def _complete_from_cache(job: Job, hit: PipelineResult) -> None:
    """Replay a stored run. The events are the real ones from when it ran."""
    job.from_cache = True
    job.result = hit
    job.events = list(hit.events)
    job.status = JobStatus.DONE
    job.finished_at = datetime.now(timezone.utc)
    job._queue.put(_DONE)
    log.info("job %s served from cache (%d events)", job.job_id, len(job.events))
