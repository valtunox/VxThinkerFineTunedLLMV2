"""
FastAPI routes for Redis cache (sessions, documents, verification, analytics data).
"""
from fastapi import APIRouter, Body, Depends
from .redis_service import RedisService
from app.core.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

async def get_redis_service() -> RedisService:
    return RedisService()

@router.post(
    "/redis/set",
    summary="Set Redis Key-Value",
    description="Set a key-value pair in Redis cache with optional TTL support",
    responses={
        200: {"description": "Key-value pair set successfully"},
        400: {"description": "Invalid key or value"},
        500: {"description": "Redis connection error or internal server error"}
    }
)
async def set_key(
    key: str = Body(...),
    value: str = Body(...),
    service: RedisService = Depends(get_redis_service)
):
    """Set a key-value pair in Redis."""
    logger.info("set_key called with key=%s", key)
    result = await service.set_key(key, value)
    logger.info("set_key completed for key=%s", key)
    return result

@router.get(
    "/redis/get",
    summary="Get Redis Value by Key",
    description="Retrieve a value from Redis cache by its key",
    responses={
        200: {"description": "Value retrieved successfully"},
        404: {"description": "Key not found"},
        500: {"description": "Redis connection error or internal server error"}
    }
)
async def get_key(
    key: str,
    service: RedisService = Depends(get_redis_service)
):
    """Get a value by key from Redis."""
    logger.info("get_key called with key=%s", key)
    result = await service.get_key(key)
    logger.info("get_key completed for key=%s", key)
    return result

@router.delete(
    "/redis/delete",
    summary="Delete Redis Key",
    description="Delete a key and its associated value from Redis cache",
    responses={
        200: {"description": "Key deleted successfully"},
        404: {"description": "Key not found"},
        500: {"description": "Redis connection error or internal server error"}
    }
)
async def delete_key(
    key: str,
    service: RedisService = Depends(get_redis_service)
):
    """Delete a key from Redis."""
    logger.info("delete_key called with key=%s", key)
    result = await service.delete_key(key)
    logger.info("delete_key completed for key=%s", key)
    return result

@router.get(
    "/redis/list",
    summary="List All Redis Keys",
    description="List all keys currently stored in Redis cache. Use with caution on large datasets",
    responses={
        200: {"description": "Keys list retrieved successfully"},
        500: {"description": "Redis connection error or internal server error"}
    }
)
async def list_keys(
    service: RedisService = Depends(get_redis_service)
):
    """List all keys in Redis."""
    logger.info("list_keys called")
    result = await service.list_keys()
    logger.info("list_keys completed, count=%d", len(result))
    return result
