"""
SSE Router — ``GET /sse/progress/{task_id}``
============================================

Returns a ``text/event-stream`` that pushes progress events in real time.
Works for every agent type (sales, marketing, websearch, job_scraping, etc.).

The endpoint complements (does NOT replace) existing polling endpoints.
"""

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.services.sse.sse_manager import sse_manager

router = APIRouter(tags=["SSE - Progress Streaming"])


@router.get(
    "/sse/progress/{task_id}",
    summary="Stream progress via SSE",
    description=(
        "Server-Sent Events stream for a running task.  "
        "Each event is a JSON object with: task_id, phase, progress, total, "
        "percent, message, status, data, timestamp.  "
        "The stream ends when status is completed / failed / cancelled."
    ),
    responses={
        200: {
            "description": "SSE event stream",
            "content": {"text/event-stream": {}},
        }
    },
)
async def stream_progress(task_id: str):
    return StreamingResponse(
        sse_manager.subscribe(task_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get(
    "/sse/result/{task_id}",
    summary="Get cached result for a completed task",
    description="Returns the terminal event (completed/failed) if available. Useful as a polling fallback when SSE is unavailable.",
)
async def get_task_result(task_id: str):
    cached = sse_manager._last_events.get(task_id)
    if not cached:
        try:
            redis = await sse_manager._get_redis()
            if redis:
                cached = await redis.get(f"sse:result:{task_id}")
        except Exception:
            pass
    if cached:
        import json
        return json.loads(cached)
    return {"status": "pending", "task_id": task_id, "message": "Task still processing or not found."}


@router.get(
    "/sse/health",
    summary="SSE service health check",
)
async def sse_health():
    redis_ok = False
    try:
        redis = await sse_manager._get_redis()
        if redis:
            await redis.ping()
            redis_ok = True
    except Exception:
        pass
    return {
        "sse_available": True,
        "redis_connected": redis_ok,
        "fallback_mode": not sse_manager._use_redis,
    }
