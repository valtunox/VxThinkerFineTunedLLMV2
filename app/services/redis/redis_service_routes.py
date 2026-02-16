"""
FastAPI routes for Redis cloud operations.
"""
from fastapi import APIRouter, Body, Depends
from .redis_service import RedisService

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
    return await service.set_key(key, value)

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
    return await service.get_key(key)

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
    return await service.delete_key(key)

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
    return await service.list_keys()
