"""
FastAPI routes for Kafka cloud service with Redis and Postgres integration.
"""
from fastapi import APIRouter, Body, Depends
from typing import List, Dict, Any
from .cloud_service import KafkaCloudService, CloudEventKafka

router = APIRouter()

# Dependency for KafkaCloudService (mock clients for demo)
async def get_kafka_service() -> KafkaCloudService:
    return KafkaCloudService(redis_client=None, postgres_client=None)

@router.post(
    "/kafka/tenant/event/publish",
    summary="Publish Tenant Event to Kafka",
    description="Publish a cloud event for a tenant. Events are cached in Redis, logged to Postgres, and produced to Kafka topics for distributed processing",
    responses={
        200: {"description": "Event published successfully"},
        400: {"description": "Invalid event data"},
        500: {"description": "Kafka connection error or internal server error"}
    }
)
async def publish_tenant_event(
    tenant_id: str = Body(...),
    event: Dict[str, Any] = Body(...),
    service: KafkaCloudService = Depends(get_kafka_service)
):
    """Publish a cloud event for a tenant (caches in Redis, logs to Postgres, produces to Kafka)."""
    event_obj = CloudEventKafka(**event)
    return await service.publish_tenant_event(tenant_id, event_obj)

@router.post(
    "/kafka/tenant/events/batch",
    summary="Batch Publish Events to Kafka",
    description="Batch publish multiple cloud events for a tenant. Optimized for high-throughput event streaming",
    responses={
        200: {"description": "Events published successfully"},
        400: {"description": "Invalid event data in batch"},
        500: {"description": "Kafka connection error or internal server error"}
    }
)
async def batch_publish_events(
    tenant_id: str = Body(...),
    events: List[Dict[str, Any]] = Body(...),
    service: KafkaCloudService = Depends(get_kafka_service)
):
    """Batch publish cloud events for a tenant."""
    event_objs = [CloudEventKafka(**e) for e in events]
    return await service.batch_publish_events(tenant_id, event_objs)

@router.get(
    "/kafka/event/redis",
    summary="Get Cached Event from Redis",
    description="Retrieve a previously published event from Redis cache by its event ID",
    responses={
        200: {"description": "Event retrieved from cache successfully"},
        404: {"description": "Event not found in cache"},
        500: {"description": "Redis connection error or internal server error"}
    }
)
async def get_event_from_redis(
    event_id: str,
    service: KafkaCloudService = Depends(get_kafka_service)
):
    """Retrieve a cached event from Redis."""
    return await service.get_event_from_redis(event_id)

@router.get(
    "/kafka/tenant/events",
    summary="Get All Tenant Events",
    description="Retrieve all events for a specific tenant from Postgres database. Returns event history and audit trail",
    responses={
        200: {"description": "Events retrieved successfully"},
        404: {"description": "No events found for tenant"},
        500: {"description": "Database connection error or internal server error"}
    }
)
async def get_events_for_tenant(
    tenant_id: str,
    service: KafkaCloudService = Depends(get_kafka_service)
):
    """Retrieve all events for a tenant from Postgres."""
    return await service.get_events_for_tenant(tenant_id)
