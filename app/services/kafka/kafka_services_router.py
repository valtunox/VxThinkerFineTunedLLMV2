"""
FastAPI routes for Kafka event streaming (cloudsystem, sales, email campaigns, resumes) with Redis and Postgres integration.
"""
from fastapi import APIRouter, Body, Depends
from typing import List, Dict, Any
from .kafka_service import KafkaService, KafkaEvent
from app.core.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

# Dependency for KafkaService (mock clients for demo)
async def get_kafka_service() -> KafkaService:
    return KafkaService(redis_client=None, postgres_client=None)

@router.post(
    "/kafka/tenant/event/publish",
    summary="Publish Tenant Event to Kafka",
    description="Publish an event for a tenant (sales, campaigns, resumes). Events are cached in Redis, logged to Postgres, and produced to Kafka topics for distributed processing",
    responses={
        200: {"description": "Event published successfully"},
        400: {"description": "Invalid event data"},
        500: {"description": "Kafka connection error or internal server error"}
    }
)
async def publish_tenant_event(
    tenant_id: str = Body(...),
    event: Dict[str, Any] = Body(...),
    service: KafkaService = Depends(get_kafka_service)
):
    """Publish an event for a tenant (caches in Redis, logs to Postgres, produces to Kafka)."""
    logger.info("publish_tenant_event called for tenant_id=%s", tenant_id)
    event_obj = KafkaEvent(**event)
    result = await service.publish_tenant_event(tenant_id, event_obj)
    logger.info("publish_tenant_event completed for tenant_id=%s", tenant_id)
    return result

@router.post(
    "/kafka/tenant/events/batch",
    summary="Batch Publish Events to Kafka",
    description="Batch publish multiple events for a tenant. Optimized for high-throughput event streaming (campaigns, resumes)",
    responses={
        200: {"description": "Events published successfully"},
        400: {"description": "Invalid event data in batch"},
        500: {"description": "Kafka connection error or internal server error"}
    }
)
async def batch_publish_events(
    tenant_id: str = Body(...),
    events: List[Dict[str, Any]] = Body(...),
    service: KafkaService = Depends(get_kafka_service)
):
    """Batch publish events for a tenant."""
    logger.info("batch_publish_events called for tenant_id=%s, count=%d", tenant_id, len(events))
    event_objs = [KafkaEvent(**e) for e in events]
    result = await service.batch_publish_events(tenant_id, event_objs)
    logger.info("batch_publish_events completed for tenant_id=%s", tenant_id)
    return result

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
    service: KafkaService = Depends(get_kafka_service)
):
    """Retrieve a cached event from Redis."""
    logger.info("get_event_from_redis called for event_id=%s", event_id)
    result = await service.get_event_from_redis(event_id)
    logger.info("get_event_from_redis completed for event_id=%s", event_id)
    return result

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
    service: KafkaService = Depends(get_kafka_service)
):
    """Retrieve all events for a tenant from Postgres."""
    logger.info("get_events_for_tenant called for tenant_id=%s", tenant_id)
    result = await service.get_events_for_tenant(tenant_id)
    logger.info("get_events_for_tenant completed for tenant_id=%s", tenant_id)
    return result


@router.get(
    "/kafka/health",
    summary="Kafka Health Check",
    description="Check Kafka broker connectivity, producer status, and registered topics.",
    responses={
        200: {"description": "Kafka health status"},
    }
)
async def kafka_health():
    """Return Kafka connectivity status and topic list."""
    from .kafka_service import kafka_service
    return kafka_service.health_check()
