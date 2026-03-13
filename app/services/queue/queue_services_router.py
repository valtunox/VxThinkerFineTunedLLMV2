"""
Queue Services Router
=====================

FastAPI router for queue service: health checks, optional outbox trigger,
and pipeline consumer control. Follows same naming and pattern as
notifications_services_router, celery_service_router, messaging_services_router.
"""

from datetime import datetime
from fastapi import APIRouter, HTTPException

from app.core.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["queue"])


@router.get("/health")
async def queue_health():
    """Health check for queue service (broker config, outbox processor, pipeline)."""
    from app.services.queue.broker import get_broker_url
    from app.services.queue.outbox_processor import is_processor_running
    try:
        from app.services.queue.queue_pipeline_service import pipeline_health
        pipeline = await pipeline_health()
    except Exception as e:
        logger.warning("Queue pipeline health check failed: %s", e)
        pipeline = {"redis": False, "postgres": False, "consumer_running": False}
    return {
        "status": "healthy",
        "service": "queue",
        "broker_url_configured": bool(get_broker_url()),
        "outbox_processor_running": is_processor_running(),
        "pipeline": pipeline,
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.post("/outbox/process")
async def trigger_outbox_process(limit: int = 20):
    """Manually trigger one outbox processing batch (for admin or testing)."""
    from app.services.queue.outbox_processor import process_outbox_manually
    try:
        result = await process_outbox_manually(limit=limit)
        return {"success": True, "result": result}
    except Exception as e:
        logger.error("Outbox process trigger failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/outbox/processor/start")
async def start_outbox_processor(interval: int = 5):
    """Start the outbox processor background task."""
    from app.services.queue.outbox_processor import start_outbox_processor as start
    await start(interval=interval)
    return {"success": True, "message": "Outbox processor started", "interval_seconds": interval}


@router.post("/outbox/processor/stop")
async def stop_outbox_processor():
    """Stop the outbox processor background task."""
    from app.services.queue.outbox_processor import stop_outbox_processor as stop
    await stop()
    return {"success": True, "message": "Outbox processor stopped"}
