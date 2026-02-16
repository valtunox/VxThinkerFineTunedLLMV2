"""
FastAPI routes for RabbitMQ message queue operations.
"""
import asyncio
from typing import Any, Dict

from fastapi import APIRouter, Body, Depends

from .rabbitmq_service import RabbitMQService, get_rabbitmq_service

router = APIRouter()


async def get_rabbitmq_service_dep() -> RabbitMQService:
    return get_rabbitmq_service()


@router.post(
    "/rabbitmq/publish",
    summary="Publish message to RabbitMQ queue",
    description="Publish a message to a RabbitMQ queue. Body can be string or JSON object.",
    responses={
        200: {"description": "Message published successfully"},
        400: {"description": "Invalid queue or body"},
        503: {"description": "RabbitMQ connection error"},
    },
)
async def publish_message(
    queue: str = Body(..., embed=True),
    body: Dict[str, Any] | str = Body(..., embed=True),
    durable: bool = Body(True, embed=True),
    service: RabbitMQService = Depends(get_rabbitmq_service_dep),
):
    """Publish a message to a queue. Runs sync pika call in executor."""
    loop = asyncio.get_event_loop()
    ok = await loop.run_in_executor(None, lambda: service.publish(queue, body, durable=durable))
    if not ok:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="Failed to publish to RabbitMQ (broker may be unavailable)")
    return {"published": True, "queue": queue}


@router.post(
    "/rabbitmq/queue/declare",
    summary="Declare RabbitMQ queue",
    description="Declare a queue (idempotent). Creates the queue if it does not exist.",
    responses={
        200: {"description": "Queue declared successfully"},
        503: {"description": "RabbitMQ connection error"},
    },
)
async def declare_queue(
    queue: str = Body(..., embed=True),
    durable: bool = Body(True, embed=True),
    service: RabbitMQService = Depends(get_rabbitmq_service_dep),
):
    """Declare a queue."""
    loop = asyncio.get_event_loop()
    ok = await loop.run_in_executor(None, lambda: service.declare_queue(queue, durable=durable))
    if not ok:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="Failed to declare queue (broker may be unavailable)")
    return {"declared": True, "queue": queue}


@router.get(
    "/rabbitmq/queue/{queue_name}",
    summary="Get RabbitMQ queue info",
    description="Get queue info (e.g. message count) if available.",
    responses={
        200: {"description": "Queue info retrieved"},
        404: {"description": "Queue not found or info unavailable"},
        503: {"description": "RabbitMQ connection error"},
    },
)
async def get_queue_info(
    queue_name: str,
    service: RabbitMQService = Depends(get_rabbitmq_service_dep),
):
    """Get queue info."""
    loop = asyncio.get_event_loop()
    info = await loop.run_in_executor(None, lambda: service.get_queue_info(queue_name))
    if info is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Queue not found or info unavailable")
    return info


@router.get(
    "/rabbitmq/health",
    summary="RabbitMQ health check",
    description="Check RabbitMQ connection health.",
    responses={
        200: {"description": "RabbitMQ is healthy"},
        503: {"description": "RabbitMQ is unavailable"},
    },
)
async def rabbitmq_health(service: RabbitMQService = Depends(get_rabbitmq_service_dep)):
    """Health check for RabbitMQ."""
    loop = asyncio.get_event_loop()
    ok = await loop.run_in_executor(None, service.health_check)
    if not ok:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="RabbitMQ unavailable")
    return {"status": "healthy", "service": "rabbitmq"}
