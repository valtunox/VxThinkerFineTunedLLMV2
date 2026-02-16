"""
FastAPI routes for Celery cloud operations.
"""
from fastapi import APIRouter, Body, Depends
from .celery_service import CeleryService

router = APIRouter()


async def get_celery_service() -> CeleryService:
    return CeleryService()


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
