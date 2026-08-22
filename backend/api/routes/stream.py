from __future__ import annotations
import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from backend.services.job_runner import stream_events, get_job

router = APIRouter(prefix="/research", tags=["research"])


@router.get("/stream/{job_id}")
async def stream_research(job_id: str):
    """
    stream_events() (3.6) is a plain sync generator — it blocks on
    queue.get(timeout=1.0) internally. Passing a sync generator directly as
    StreamingResponse's content is the correct move here: Starlette detects
    it's not an async generator and automatically wraps it with
    iterate_in_threadpool, so the blocking queue.get() runs on a worker
    thread instead of the event loop. This is what avoids the timeout
    problem — no manual `async for` needed, and manually iterating it
    inside an `async def` body would reintroduce exactly the blocking bug
    being avoided here.
    """
    if get_job(job_id) is None:
        raise HTTPException(status_code=404, detail=f"unknown job {job_id}")

    def event_stream():
        for event in stream_events(job_id):
            payload = json.dumps({
                "step_id": event.step_id,
                "label": event.label,
                "status": event.status.value if hasattr(event.status, "value") else event.status,
                "detail": event.detail,
                "timestamp": event.timestamp.isoformat(),
            })
            yield f"data: {payload}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )