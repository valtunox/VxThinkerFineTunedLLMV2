"""
Kafka Service (kafka_service.py)
================================

Purpose: Central Kafka integration for the VaLLM Specialist Model platform.
Provides event streaming for:
  - Document processing and verification (upload, parse, verify)
  - Financial transactions and reconciliation
  - Business analytics and recommendations
  - Notifications and outbox processing (delivery, retries)

Uses Apache Kafka for message production/consumption, with optional Redis caching
and Postgres audit logging. Exposes KafkaService and KafkaEvent for API use.
"""

import json
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
import os
from pathlib import Path
from pydantic import BaseModel
from enum import Enum
import uuid

try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent.parent.parent / '.env'
    if _env_path.exists():
        load_dotenv(dotenv_path=_env_path, override=False)
    else:
        load_dotenv(override=False)
except ImportError:
    pass

from app.core.logger import get_logger

logger = get_logger(__name__)

def _import_kafka_python():
    """Import from the real kafka-python package, not our local services/kafka/ dir.

    The local ``services/kafka/`` directory shadows the ``kafka-python`` pip
    package because ``app/`` and ``app/services/`` are both on ``sys.path``.
    We work around this by temporarily stripping those paths *and* purging any
    cached ``kafka`` module entries so that Python resolves ``kafka`` to the
    site-packages version.
    """
    import sys as _sys
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    _services_dir = os.path.dirname(_this_dir)
    _app_dir = os.path.dirname(_services_dir)

    _bad_prefixes = {os.path.normcase(os.path.abspath(p))
                     for p in (_this_dir, _services_dir, _app_dir)}

    _saved_path = list(_sys.path)
    _sys.path = [p for p in _sys.path
                 if os.path.normcase(os.path.abspath(p)) not in _bad_prefixes]

    _stale_keys = [k for k in _sys.modules if k == 'kafka' or k.startswith('kafka.')]
    _stale = {k: _sys.modules.pop(k) for k in _stale_keys}

    try:
        from kafka.producer import KafkaProducer
        from kafka.consumer.group import KafkaConsumer
        from kafka.errors import KafkaError
        return KafkaProducer, KafkaConsumer, KafkaError, True
    except Exception:
        _sys.modules.update(_stale)
        return None, None, None, False
    finally:
        _sys.path = _saved_path

KafkaProducer, KafkaConsumer, KafkaError, KAFKA_AVAILABLE = _import_kafka_python()

class KafkaEventType(str, Enum):
    """Event types for document processing, verification, and analytics."""
    DOCUMENT_UPLOADED = "document.uploaded"
    DOCUMENT_PROCESSED = "document.processed"
    DOCUMENT_VERIFIED = "document.verified"
    VERIFICATION_COMPLETED = "verification.completed"
    TRANSACTION_CREATED = "transaction.created"
    TRANSACTION_RECONCILED = "transaction.reconciled"
    RECOMMENDATION_GENERATED = "recommendation.generated"
    ANALYSIS_COMPLETED = "analysis.completed"
    NOTIFICATION_SENT = "notification.sent"
    REQUESTED = "requested"  # generic / legacy
    SCALING_REQUESTED = "scaling.requested"
    SCALING_COMPLETED = "scaling.completed"
    MONITORING_ALERT = "monitoring.alert"
    BACKUP_STARTED = "backup.started"
    BACKUP_COMPLETED = "backup.completed"
    DATA_IMPORT_COMPLETED = "data.import.completed"
    DATA_IMPORT_FAILED = "data.import.failed"
    SCRAPING_STARTED = "scraping.started"
    SCRAPING_COMPLETED = "scraping.completed"
    SCRAPING_FAILED = "scraping.failed"

class KafkaResourceType(str, Enum):
    """Resource types for events (documents, verifications, transactions, etc.)."""
    DOCUMENT = "document"
    VERIFICATION = "verification"
    TRANSACTION = "transaction"
    RECOMMENDATION = "recommendation"
    ANALYSIS = "analysis"
    NOTIFICATION = "notification"
    TENANT = "tenant"
    USER = "user"
    DATA_IMPORT = "data_import"
    GENERIC = "generic"

class KafkaEventStatus(str, Enum):
    REQUESTED = "requested"
    PENDING = "pending"
    READY = "ready"
    FAILED = "failed"
    COMPLETED = "completed"
    SENT = "sent"
    SCALING = "scaling"
    MONITORING = "monitoring"
    BACKING_UP = "backing_up"

class KafkaEvent(BaseModel):
    """Event payload for Kafka (document processing, verification, analytics)."""
    event_id: str
    resource_id: str
    resource_type: KafkaResourceType
    user_id: str
    organization_id: Optional[str] = None
    region: str = "default"
    availability_zone: Optional[str] = None
    event_type: KafkaEventType
    status: KafkaEventStatus
    timestamp: datetime
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = {}
    estimated_duration: Optional[int] = None  # in seconds
    cost_estimate: Optional[float] = None
    tags: Dict[str, str] = {}

class KafkaService:
    def __init__(self, redis_client: Any = None, postgres_client: Any = None):
        self.bootstrap_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', '3.80.77.66:9092')
        self.security_protocol = os.getenv('KAFKA_SECURITY_PROTOCOL', 'PLAINTEXT')
        self.sasl_mechanism = os.getenv('KAFKA_SASL_MECHANISM', 'PLAIN')
        self.sasl_username = os.getenv('KAFKA_SASL_USERNAME', '')
        self.sasl_password = os.getenv('KAFKA_SASL_PASSWORD', '')

        # Topic names for different event types (documents, verification, analytics)
        self.topic_infrastructure_events = 'infrastructure-events'
        self.topic_scaling_events = 'scaling-events'
        self.topic_monitoring_events = 'monitoring-events'
        self.topic_backup_events = 'backup-events'
        self.topic_cost_analytics = 'cost-analytics'
        self.topic_security_events = 'security-events'
        self.topic_compliance_events = 'compliance-events'
        self.topic_document_events = 'document-events'
        self.topic_scraping_tasks = 'scraping-tasks'

        self.producer = None
        self.consumer = None

        # Redis and Postgres clients for caching and audit logging
        self.redis_client = redis_client
        self.postgres_client = postgres_client

    async def cache_event_in_redis(self, event: KafkaEvent) -> bool:
        """
        Cache event in Redis for fast retrieval and deduplication.
        """
        if self.redis_client:
            key = f"kafka_event:{event.event_id}"
            value = event.json()
            await self.redis_client.set(key, value)
            return True
        return False

    async def log_event_to_postgres(self, event: KafkaEvent) -> bool:
        """
        Log event to Postgres for audit and analytics.
        """
        if self.postgres_client:
            # Example: Insert event into a table (pseudo-code)
            query = """
                INSERT INTO kafka_events (event_id, resource_id, resource_type, user_id, organization_id, region, event_type, status, timestamp, metadata, cost_estimate)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            values = (
                event.event_id, event.resource_id, event.resource_type.value, event.user_id, event.organization_id,
                event.region, event.event_type.value, event.status.value, event.timestamp, json.dumps(event.metadata), event.cost_estimate
            )
            await self.postgres_client.execute(query, values)
            return True
        return False

    async def publish_tenant_event(self, tenant_id: str, event: KafkaEvent, additional_topics: List[str] = None) -> bool:
        """
        Publish an event for a specific tenant, cache in Redis, and log to Postgres.
        """
        # Cache in Redis
        await self.cache_event_in_redis(event)
        # Log to Postgres
        await self.log_event_to_postgres(event)
        # Produce to Kafka
        return self.produce_event(event, additional_topics)

    async def batch_publish_events(self, tenant_id: str, events: List[KafkaEvent]) -> List[bool]:
        """
        Batch publish events for a tenant (documents, verification, analytics).
        """
        results = []
        for event in events:
            result = await self.publish_tenant_event(tenant_id, event)
            results.append(result)
        return results

    async def get_event_from_redis(self, event_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a cached event from Redis.
        """
        if self.redis_client:
            key = f"kafka_event:{event_id}"
            value = await self.redis_client.get(key)
            if value:
                return json.loads(value)
        return None

    async def get_events_for_tenant(self, tenant_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all events for a tenant from Postgres (event history and analytics).
        """
        if self.postgres_client:
            query = "SELECT * FROM kafka_events WHERE organization_id = %s ORDER BY timestamp DESC"
            rows = await self.postgres_client.fetch(query, (tenant_id,))
            return [dict(row) for row in rows]
        return []

    def create_producer(self):
        """Create Kafka producer for events"""
        if not KAFKA_AVAILABLE:
            raise ImportError("kafka-python not available")

        try:
            config = {
                'bootstrap_servers': self.bootstrap_servers.split(','),
                'value_serializer': lambda x: json.dumps(x, default=str).encode('utf-8'),
                'key_serializer': lambda x: x.encode('utf-8') if x else None,
                'acks': 'all',  # Wait for all replicas
                'retries': 5,
                'max_in_flight_requests_per_connection': 1,
                'enable_idempotence': True,
                'compression_type': 'gzip'  # Compress messages for better throughput
            }

            if self.security_protocol != 'PLAINTEXT':
                config.update({
                    'security_protocol': self.security_protocol,
                    'sasl_mechanism': self.sasl_mechanism,
                    'sasl_plain_username': self.sasl_username,
                    'sasl_plain_password': self.sasl_password
                })

            self.producer = KafkaProducer(**config)
            logger.info("Kafka producer created successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to create Kafka producer: {e}")
            return False

    def create_consumer(self, group_id: str, topics: List[str]):
        """Create Kafka consumer for events"""
        if not KAFKA_AVAILABLE:
            raise ImportError("kafka-python not available")

        try:
            config = {
                'bootstrap_servers': self.bootstrap_servers.split(','),
                'group_id': group_id,
                'value_deserializer': lambda x: json.loads(x.decode('utf-8')),
                'key_deserializer': lambda x: x.decode('utf-8') if x else None,
                'auto_offset_reset': 'earliest',
                'enable_auto_commit': True,
                'auto_commit_interval_ms': 1000,
                'max_poll_records': 100
            }

            if self.security_protocol != 'PLAINTEXT':
                config.update({
                    'security_protocol': self.security_protocol,
                    'sasl_mechanism': self.sasl_mechanism,
                    'sasl_plain_username': self.sasl_username,
                    'sasl_plain_password': self.sasl_password
                })

            self.consumer = KafkaConsumer(*topics, **config)
            logger.info(f"Kafka consumer created for topics: {topics}")
            return True
        except Exception as e:
            logger.error(f"Failed to create Kafka consumer: {e}")
            return False

    def produce_event(self, event: KafkaEvent, additional_topics: List[str] = None):
        """Produce event to Kafka topics"""
        try:
            if not self.producer:
                if not self.create_producer():
                    return False

            # Create message
            message = event.dict()
            key = f"{event.resource_id}_{event.event_type.value}"

            # Determine primary topic based on event type
            primary_topic = self._get_primary_topic(event.event_type)

            # Send to primary topic
            future = self.producer.send(
                primary_topic,
                key=key,
                value=message,
                headers=[
                    ('event_type', event.event_type.value.encode()),
                    ('resource_id', event.resource_id.encode()),
                    ('resource_type', event.resource_type.value.encode()),
                    ('user_id', event.user_id.encode()),
                    ('region', event.region.encode()),
                    ('timestamp', str(int(event.timestamp.timestamp())).encode())
                ]
            )

            # Send to cost analytics if cost estimate is available
            if event.cost_estimate is not None:
                cost_future = self.producer.send(
                    self.topic_cost_analytics,
                    key=key,
                    value=message
                )

            # Send to additional topics if specified
            if additional_topics:
                for topic in additional_topics:
                    self.producer.send(topic, key=key, value=message)

            # Wait for confirmation
            record_metadata = future.get(timeout=10)
            logger.info(f"Event produced to topic {record_metadata.topic} partition {record_metadata.partition}")

            return True
        except KafkaError as e:
            logger.error(f"Kafka error producing event: {e}")
            return False
        except Exception as e:
            logger.error(f"Error producing event: {e}")
            return False

    def _get_primary_topic(self, event_type: KafkaEventType) -> str:
        """Get primary topic based on event type"""
        if event_type.value.startswith('document'):
            return self.topic_document_events
        elif event_type.value.startswith('infrastructure'):
            return self.topic_infrastructure_events
        elif event_type.value.startswith('scaling'):
            return self.topic_scaling_events
        elif event_type.value.startswith('monitoring'):
            return self.topic_monitoring_events
        elif event_type.value.startswith('backup'):
            return self.topic_backup_events
        else:
            return self.topic_infrastructure_events

    def provision_infrastructure(self, event_data: Dict[str, Any]):
        """Handle provisioning request (legacy lambda); maps to generic event."""
        resource_type = event_data.get('resource_type', 'generic')
        try:
            rt = KafkaResourceType(resource_type)
        except ValueError:
            rt = KafkaResourceType.GENERIC
        event = KafkaEvent(
            event_id=str(uuid.uuid4()),
            resource_id=event_data.get('resource_id', f"res_{uuid.uuid4().hex[:8]}"),
            resource_type=rt,
            user_id=event_data.get('user_id'),
            organization_id=event_data.get('organization_id'),
            region=event_data.get('region', 'us-east-1'),
            availability_zone=event_data.get('availability_zone'),
            event_type=KafkaEventType.REQUESTED,
            status=KafkaEventStatus.REQUESTED,
            timestamp=datetime.utcnow(),
            correlation_id=event_data.get('correlation_id'),
            metadata=event_data.get('metadata', {}),
            estimated_duration=event_data.get('estimated_duration', 300),
            cost_estimate=event_data.get('cost_estimate'),
            tags=event_data.get('tags', {})
        )
        return self.produce_event(event)

    def scale_infrastructure(self, event_data: Dict[str, Any]):
        """Handle scaling request (legacy lambda)."""
        resource_type = event_data.get('resource_type', 'generic')
        try:
            rt = KafkaResourceType(resource_type)
        except ValueError:
            rt = KafkaResourceType.GENERIC
        event = KafkaEvent(
            event_id=str(uuid.uuid4()),
            resource_id=event_data.get('resource_id'),
            resource_type=rt,
            user_id=event_data.get('user_id'),
            organization_id=event_data.get('organization_id'),
            region=event_data.get('region', 'us-east-1'),
            availability_zone=event_data.get('availability_zone'),
            event_type=KafkaEventType.SCALING_REQUESTED,
            status=KafkaEventStatus.SCALING,
            timestamp=datetime.utcnow(),
            correlation_id=event_data.get('correlation_id'),
            metadata=event_data.get('metadata', {}),
            estimated_duration=event_data.get('estimated_duration', 180),
            cost_estimate=event_data.get('cost_estimate'),
            tags=event_data.get('tags', {})
        )
        return self.produce_event(event)

    def monitor_infrastructure(self, event_data: Dict[str, Any]):
        """Handle infrastructure monitoring event"""
        event = KafkaEvent(
            event_id=str(uuid.uuid4()),
            resource_id=event_data.get('resource_id'),
            resource_type=KafkaResourceType(event_data.get('resource_type', 'vm')),
            user_id=event_data.get('user_id'),
            organization_id=event_data.get('organization_id'),
            region=event_data.get('region', 'us-east-1'),
            availability_zone=event_data.get('availability_zone'),
            event_type=KafkaEventType.MONITORING_ALERT,
            status=KafkaEventStatus.MONITORING,
            timestamp=datetime.utcnow(),
            correlation_id=event_data.get('correlation_id'),
            metadata=event_data.get('metadata', {}),
            tags=event_data.get('tags', {})
        )
        return self.produce_event(event, [self.topic_monitoring_events])

    def backup_infrastructure(self, event_data: Dict[str, Any]):
        """Handle backup request (legacy lambda)."""
        resource_type = event_data.get('resource_type', 'generic')
        try:
            rt = KafkaResourceType(resource_type)
        except ValueError:
            rt = KafkaResourceType.GENERIC
        event = KafkaEvent(
            event_id=str(uuid.uuid4()),
            resource_id=event_data.get('resource_id'),
            resource_type=rt,
            user_id=event_data.get('user_id'),
            organization_id=event_data.get('organization_id'),
            region=event_data.get('region', 'us-east-1'),
            availability_zone=event_data.get('availability_zone'),
            event_type=KafkaEventType.BACKUP_STARTED,
            status=KafkaEventStatus.BACKING_UP,
            timestamp=datetime.utcnow(),
            correlation_id=event_data.get('correlation_id'),
            metadata=event_data.get('metadata', {}),
            estimated_duration=event_data.get('estimated_duration', 600),
            cost_estimate=event_data.get('cost_estimate'),
            tags=event_data.get('tags', {})
        )
        return self.produce_event(event)

    def health_check(self) -> Dict[str, Any]:
        """Non-destructive connectivity test. Re-checks kafka import at call time."""
        global KAFKA_AVAILABLE, KafkaProducer, KafkaConsumer, KafkaError
        if not KAFKA_AVAILABLE:
            KafkaProducer, KafkaConsumer, KafkaError, KAFKA_AVAILABLE = _import_kafka_python()
            if KAFKA_AVAILABLE:
                logger.info("kafka-python detected on health_check re-import")

        result = {
            "kafka_available": KAFKA_AVAILABLE,
            "bootstrap_servers": self.bootstrap_servers,
            "producer_active": self.producer is not None,
            "consumer_active": self.consumer is not None,
            "topics": {
                "infrastructure": self.topic_infrastructure_events,
                "scaling": self.topic_scaling_events,
                "monitoring": self.topic_monitoring_events,
                "backup": self.topic_backup_events,
                "document": self.topic_document_events,
                "scraping": self.topic_scraping_tasks,
            },
        }
        if KAFKA_AVAILABLE:
            try:
                if not self.producer:
                    self.create_producer()
                result["producer_connected"] = self.producer is not None
                result["status"] = "connected" if self.producer else "disconnected"
            except Exception as e:
                result["producer_connected"] = False
                result["status"] = "error"
                result["error"] = str(e)
        else:
            result["status"] = "unavailable"
        return result

    def close(self):
        """Close producer and consumer connections"""
        try:
            if self.producer:
                self.producer.close()
            if self.consumer:
                self.consumer.close()
            logger.info("Kafka connections closed")
        except Exception as e:
            logger.error(f"Error closing Kafka connections: {e}")

# Global service instance
kafka_service = KafkaService()  # global instance

def lambda_handler(event, context=None):
    """
    AWS Lambda handler for Kafka event operations (documents, verification, analytics).

    Event structure:
    {
        "action": "provision_infrastructure|scale_infrastructure|monitor_infrastructure|backup_infrastructure",
        "data": { ... }
    }
    """
    try:
        logger.info(f"Processing Kafka event: {json.dumps(event)}")

        action = event.get('action')
        data = event.get('data', {})

        if not action or not data:
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'error': 'Missing action or data in event'
                })
            }

        if not KAFKA_AVAILABLE:
            return {
                'statusCode': 500,
                'body': json.dumps({
                    'error': 'Kafka client not available. Install kafka-python package.'
                })
            }

        try:
            result = False
            if action == 'provision_infrastructure':
                result = kafka_service.provision_infrastructure(data)
            elif action == 'scale_infrastructure':
                result = kafka_service.scale_infrastructure(data)
            elif action == 'monitor_infrastructure':
                result = kafka_service.monitor_infrastructure(data)
            elif action == 'backup_infrastructure':
                result = kafka_service.backup_infrastructure(data)
            else:
                return {
                    'statusCode': 400,
                    'body': json.dumps({
                        'error': f'Unknown action: {action}'
                    })
                }

            if result:
                return {
                    'statusCode': 200,
                    'body': json.dumps({
                        'message': f'Event operation {action} produced to Kafka successfully',
                        'resource_id': data.get('resource_id'),
                        'event_id': data.get('event_id')
                    })
                }
            else:
                return {
                    'statusCode': 500,
                    'body': json.dumps({
                        'error': f'Failed to produce event operation {action} to Kafka'
                    })
                }

        finally:
            kafka_service.close()

    except Exception as e:
        logger.error(f"Error in lambda_handler: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            })
        }

# Consumer function for listening to events
def consume_events():
    """Consumer function to listen for events from Kafka (documents, verification, analytics)"""
    if not KAFKA_AVAILABLE:
        logger.error("Kafka client not available")
        return

    try:
        topics = [
            kafka_service.topic_infrastructure_events,
            kafka_service.topic_scaling_events,
            kafka_service.topic_monitoring_events,
            kafka_service.topic_backup_events,
            kafka_service.topic_document_events,
        ]

        if not kafka_service.create_consumer('kafka-processor-group', topics):
            logger.error("Failed to create Kafka consumer")
            return

        logger.info("Starting to consume events from Kafka...")

        for message in kafka_service.consumer:
            try:
                logger.info(f"Received event from topic {message.topic}: {message.value}")

                event_data = message.value
                resource_id = event_data.get('resource_id')
                event_type = event_data.get('event_type')

                logger.info(f"Processing resource {resource_id} with event type {event_type}")

            except Exception as e:
                logger.error(f"Error processing event from Kafka: {e}")

    except KeyboardInterrupt:
        logger.info("Stopping event consumer...")
    except Exception as e:
        logger.error(f"Error in event consumer: {e}")
    finally:
        kafka_service.close()

if __name__ == "__main__":
    test_event = {
        "action": "provision_infrastructure",
        "data": {
            "resource_id": "vm_kafka_test_123",
            "resource_type": "vm",
            "user_id": "user_test_456",
            "organization_id": "org_test_789",
            "region": "us-east-1",
            "availability_zone": "us-east-1a",
            "correlation_id": "corr_test_123",
            "metadata": {"instance_type": "t3.medium", "ami_id": "ami-12345"},
            "estimated_duration": 300,
            "cost_estimate": 50.0,
            "tags": {"Environment": "production", "Team": "backend"}
        }
    }
    result = lambda_handler(test_event)
    print(json.dumps(result, indent=2))
