"""
FastAPI routes for Celery task queue (document processing, verification, and business analytics).
"""
import asyncio
from fastapi import APIRouter, Body, Depends
from .celery_service import CeleryService

router = APIRouter()


async def get_celery_service() -> CeleryService:
    return CeleryService()


@router.get(
    "/celery/health",
    summary="Celery Worker Health Check",
    description="Check if Celery workers are reachable via the broker (Redis). Returns worker status and count.",
    responses={
        200: {"description": "Health check result (always 200, status field indicates health)"},
    },
)
async def celery_health():
    """Ping Celery workers via the broker and return their availability."""
    try:
        from app.core.celery_app import get_celery_app, CELERY_AVAILABLE
        if not CELERY_AVAILABLE:
            return {
                "status": "unavailable",
                "workers": 0,
                "worker_names": [],
                "detail": "Celery package not installed",
            }

        app = get_celery_app()
        if app is None:
            return {
                "status": "unavailable",
                "workers": 0,
                "worker_names": [],
                "detail": "Celery app not configured",
            }

        ping_result = await asyncio.to_thread(
            app.control.ping, timeout=2.0
        )

        if ping_result:
            worker_names = [
                list(entry.keys())[0] for entry in ping_result if isinstance(entry, dict)
            ]
            return {
                "status": "connected",
                "workers": len(worker_names),
                "worker_names": worker_names,
            }
        else:
            return {
                "status": "disconnected",
                "workers": 0,
                "worker_names": [],
                "detail": "No workers responded to ping",
            }
    except Exception as e:
        return {
            "status": "disconnected",
            "workers": 0,
            "worker_names": [],
            "detail": str(e)[:200],
        }


@router.post(
    "/celery/task/submit",
    summary="Submit Celery Task",
    description="Submit a task to the Celery distributed task queue for asynchronous processing with optional arguments",
    responses={
        200: {"description": "Task submitted successfully, returns task ID"},
        400: {"description": "Invalid task name or parameters"},
        500: {"description": "Celery connection error or internal server error"},
    },
)
async def submit_task(
    task_name: str = Body(...),
    args: list = Body(default=[]),
    kwargs: dict = Body(default={}),
    service: CeleryService = Depends(get_celery_service),
):
    """Submit a task to the Celery queue."""
    return await service.submit_task(task_name, args, kwargs)


@router.get(
    "/celery/task/status",
    summary="Get Celery Task Status",
    description="Get the current status and result of a submitted Celery task by its task ID",
    responses={
        200: {"description": "Task status retrieved successfully"},
        404: {"description": "Task not found"},
        500: {"description": "Celery connection error or internal server error"},
    },
)
async def get_task_status(
    task_id: str,
    service: CeleryService = Depends(get_celery_service),
):
    """Get the status of a submitted task."""
    return await service.get_task_status(task_id)


@router.get(
    "/celery/task/list",
    summary="List All Celery Tasks",
    description="List all submitted Celery tasks with their current status and metadata",
    responses={
        200: {"description": "Tasks list retrieved successfully"},
        500: {"description": "Celery connection error or internal server error"},
    },
)
async def list_tasks(
    service: CeleryService = Depends(get_celery_service),
):
    """List all submitted tasks."""
    return await service.list_tasks()
