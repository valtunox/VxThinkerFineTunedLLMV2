"""
Kafka Cloud Service
Handles high-throughput cloud operations and infrastructure events using Apache Kafka
"""

import json
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
import os
from pydantic import BaseModel
from enum import Enum
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from kafka import KafkaProducer, KafkaConsumer
    from kafka.errors import KafkaError
    KAFKA_AVAILABLE = True
except ImportError:
    logger.warning("kafka-python not installed. Install with: pip install kafka-python")
    KAFKA_AVAILABLE = False

class CloudEventType(str, Enum):
    INFRASTRUCTURE_REQUESTED = "infrastructure.requested"
    INFRASTRUCTURE_PROVISIONING = "infrastructure.provisioning"
    INFRASTRUCTURE_READY = "infrastructure.ready"
    INFRASTRUCTURE_FAILED = "infrastructure.failed"
    SCALING_REQUESTED = "scaling.requested"
    SCALING_IN_PROGRESS = "scaling.in_progress"
    SCALING_COMPLETED = "scaling.completed"
    MONITORING_ALERT = "monitoring.alert"
    BACKUP_STARTED = "backup.started"
    BACKUP_COMPLETED = "backup.completed"
    RESOURCE_DESTROYED = "resource.destroyed"

class CloudResourceType(str, Enum):
    VM = "vm"
    CONTAINER = "container"
    DATABASE = "database"
    LOAD_BALANCER = "load_balancer"
    STORAGE = "storage"
    NETWORK = "network"
    SECURITY_GROUP = "security_group"

class CloudEventStatus(str, Enum):
    REQUESTED = "requested"
    PROVISIONING = "provisioning"
    READY = "ready"
    FAILED = "failed"
    SCALING = "scaling"
    MONITORING = "monitoring"
    BACKING_UP = "backing_up"
    DESTROYING = "destroying"

class CloudEventKafka(BaseModel):
    event_id: str
    resource_id: str
    resource_type: CloudResourceType
    user_id: str
    organization_id: Optional[str] = None
    region: str
    availability_zone: Optional[str] = None
    event_type: CloudEventType
    status: CloudEventStatus
    timestamp: datetime
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = {}
    estimated_duration: Optional[int] = None  # in seconds
    cost_estimate: Optional[float] = None
    tags: Dict[str, str] = {}

class KafkaCloudService:
    def __init__(self, redis_client: Any = None, postgres_client: Any = None):
        self.bootstrap_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
        self.security_protocol = os.getenv('KAFKA_SECURITY_PROTOCOL', 'PLAINTEXT')
        self.sasl_mechanism = os.getenv('KAFKA_SASL_MECHANISM', 'PLAIN')
        self.sasl_username = os.getenv('KAFKA_SASL_USERNAME', '')
        self.sasl_password = os.getenv('KAFKA_SASL_PASSWORD', '')

        # Topic names for different cloud operations
        self.topic_infrastructure_events = 'infrastructure-events'
        self.topic_scaling_events = 'scaling-events'
        self.topic_monitoring_events = 'monitoring-events'
        self.topic_backup_events = 'backup-events'
        self.topic_cost_analytics = 'cost-analytics'
        self.topic_security_events = 'security-events'
        self.topic_compliance_events = 'compliance-events'

        self.producer = None
        self.consumer = None

        # Redis and Postgres clients for real-world cloud ops
        self.redis_client = redis_client
        self.postgres_client = postgres_client

    async def cache_event_in_redis(self, event: CloudEventKafka) -> bool:
        """
        Cache event in Redis for fast retrieval and deduplication.
        """
        if self.redis_client:
            key = f"cloud_event:{event.event_id}"
            value = event.json()
            await self.redis_client.set(key, value)
            return True
        return False

    async def log_event_to_postgres(self, event: CloudEventKafka) -> bool:
        """
        Log event to Postgres for audit and analytics.
        """
        if self.postgres_client:
            # Example: Insert event into a table (pseudo-code)
            query = """
                INSERT INTO cloud_events (event_id, resource_id, resource_type, user_id, organization_id, region, event_type, status, timestamp, metadata, cost_estimate)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            values = (
                event.event_id, event.resource_id, event.resource_type.value, event.user_id, event.organization_id,
                event.region, event.event_type.value, event.status.value, event.timestamp, json.dumps(event.metadata), event.cost_estimate
            )
            await self.postgres_client.execute(query, values)
            return True
        return False

    async def publish_tenant_event(self, tenant_id: str, event: CloudEventKafka, additional_topics: List[str] = None) -> bool:
        """
        Publish a cloud event for a specific tenant, cache in Redis, and log to Postgres.
        """
        # Cache in Redis
        await self.cache_event_in_redis(event)
        # Log to Postgres
        await self.log_event_to_postgres(event)
        # Produce to Kafka
        return self.produce_cloud_event(event, additional_topics)

    async def batch_publish_events(self, tenant_id: str, events: List[CloudEventKafka]) -> List[bool]:
        """
        Batch publish events for a tenant (cloud infra use case).
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
            key = f"cloud_event:{event_id}"
            value = await self.redis_client.get(key)
            if value:
                return json.loads(value)
        return None

    async def get_events_for_tenant(self, tenant_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all events for a tenant from Postgres (cloud ops analytics).
        """
        if self.postgres_client:
            query = "SELECT * FROM cloud_events WHERE organization_id = %s ORDER BY timestamp DESC"
            rows = await self.postgres_client.fetch(query, (tenant_id,))
            return [dict(row) for row in rows]
        return []

    def create_producer(self):
        """Create Kafka producer for cloud events"""
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
            logger.info("Kafka cloud producer created successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to create Kafka cloud producer: {e}")
            return False

    def create_consumer(self, group_id: str, topics: List[str]):
        """Create Kafka consumer for cloud events"""
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
            logger.info(f"Kafka cloud consumer created for topics: {topics}")
            return True
        except Exception as e:
            logger.error(f"Failed to create Kafka cloud consumer: {e}")
            return False

    def produce_cloud_event(self, event: CloudEventKafka, additional_topics: List[str] = None):
        """Produce cloud event to Kafka topics"""
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
            logger.info(f"Cloud event produced to topic {record_metadata.topic} partition {record_metadata.partition}")
            
            return True
        except KafkaError as e:
            logger.error(f"Kafka error producing cloud event: {e}")
            return False
        except Exception as e:
            logger.error(f"Error producing cloud event: {e}")
            return False

    def _get_primary_topic(self, event_type: CloudEventType) -> str:
        """Get primary topic based on event type"""
        if event_type.value.startswith('infrastructure'):
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
        """Handle infrastructure provisioning request"""
        event = CloudEventKafka(
            event_id=str(uuid.uuid4()),
            resource_id=event_data.get('resource_id', f"res_{uuid.uuid4().hex[:8]}"),
            resource_type=CloudResourceType(event_data.get('resource_type', 'vm')),
            user_id=event_data.get('user_id'),
            organization_id=event_data.get('organization_id'),
            region=event_data.get('region', 'us-east-1'),
            availability_zone=event_data.get('availability_zone'),
            event_type=CloudEventType.INFRASTRUCTURE_REQUESTED,
            status=CloudEventStatus.REQUESTED,
            timestamp=datetime.utcnow(),
            correlation_id=event_data.get('correlation_id'),
            metadata=event_data.get('metadata', {}),
            estimated_duration=event_data.get('estimated_duration', 300),
            cost_estimate=event_data.get('cost_estimate'),
            tags=event_data.get('tags', {})
        )
        return self.produce_cloud_event(event)

    def scale_infrastructure(self, event_data: Dict[str, Any]):
        """Handle infrastructure scaling request"""
        event = CloudEventKafka(
            event_id=str(uuid.uuid4()),
            resource_id=event_data.get('resource_id'),
            resource_type=CloudResourceType(event_data.get('resource_type', 'vm')),
            user_id=event_data.get('user_id'),
            organization_id=event_data.get('organization_id'),
            region=event_data.get('region', 'us-east-1'),
            availability_zone=event_data.get('availability_zone'),
            event_type=CloudEventType.SCALING_REQUESTED,
            status=CloudEventStatus.SCALING,
            timestamp=datetime.utcnow(),
            correlation_id=event_data.get('correlation_id'),
            metadata=event_data.get('metadata', {}),
            estimated_duration=event_data.get('estimated_duration', 180),
            cost_estimate=event_data.get('cost_estimate'),
            tags=event_data.get('tags', {})
        )
        return self.produce_cloud_event(event)

    def monitor_infrastructure(self, event_data: Dict[str, Any]):
        """Handle infrastructure monitoring event"""
        event = CloudEventKafka(
            event_id=str(uuid.uuid4()),
            resource_id=event_data.get('resource_id'),
            resource_type=CloudResourceType(event_data.get('resource_type', 'vm')),
            user_id=event_data.get('user_id'),
            organization_id=event_data.get('organization_id'),
            region=event_data.get('region', 'us-east-1'),
            availability_zone=event_data.get('availability_zone'),
            event_type=CloudEventType.MONITORING_ALERT,
            status=CloudEventStatus.MONITORING,
            timestamp=datetime.utcnow(),
            correlation_id=event_data.get('correlation_id'),
            metadata=event_data.get('metadata', {}),
            tags=event_data.get('tags', {})
        )
        return self.produce_cloud_event(event, [self.topic_monitoring_events])

    def backup_infrastructure(self, event_data: Dict[str, Any]):
        """Handle infrastructure backup request"""
        event = CloudEventKafka(
            event_id=str(uuid.uuid4()),
            resource_id=event_data.get('resource_id'),
            resource_type=CloudResourceType(event_data.get('resource_type', 'vm')),
            user_id=event_data.get('user_id'),
            organization_id=event_data.get('organization_id'),
            region=event_data.get('region', 'us-east-1'),
            availability_zone=event_data.get('availability_zone'),
            event_type=CloudEventType.BACKUP_STARTED,
            status=CloudEventStatus.BACKING_UP,
            timestamp=datetime.utcnow(),
            correlation_id=event_data.get('correlation_id'),
            metadata=event_data.get('metadata', {}),
            estimated_duration=event_data.get('estimated_duration', 600),
            cost_estimate=event_data.get('cost_estimate'),
            tags=event_data.get('tags', {})
        )
        return self.produce_cloud_event(event)

    def close(self):
        """Close producer and consumer connections"""
        try:
            if self.producer:
                self.producer.close()
            if self.consumer:
                self.consumer.close()
            logger.info("Kafka cloud connections closed")
        except Exception as e:
            logger.error(f"Error closing Kafka cloud connections: {e}")

# Global service instance
kafka_cloud_service = KafkaCloudService()

def lambda_handler(event, context=None):
    """
    AWS Lambda handler for Kafka cloud operations
    
    Event structure:
    {
        "action": "provision_infrastructure|scale_infrastructure|monitor_infrastructure|backup_infrastructure",
        "data": {
            "resource_id": "vm_123",
            "resource_type": "vm",
            "user_id": "user_456",
            "organization_id": "org_789",
            "region": "us-east-1",
            "availability_zone": "us-east-1a",
            "correlation_id": "corr_123",
            "metadata": {
                "instance_type": "t3.medium",
                "ami_id": "ami-12345",
                "security_groups": ["sg-123"]
            },
            "estimated_duration": 300,
            "cost_estimate": 50.0,
            "tags": {
                "Environment": "production",
                "Team": "backend"
            }
        }
    }
    """
    try:
        logger.info(f"Processing Kafka cloud event: {json.dumps(event)}")
        
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
            # Process based on action
            result = False
            if action == 'provision_infrastructure':
                result = kafka_cloud_service.provision_infrastructure(data)
            elif action == 'scale_infrastructure':
                result = kafka_cloud_service.scale_infrastructure(data)
            elif action == 'monitor_infrastructure':
                result = kafka_cloud_service.monitor_infrastructure(data)
            elif action == 'backup_infrastructure':
                result = kafka_cloud_service.backup_infrastructure(data)
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
                        'message': f'Cloud operation {action} produced to Kafka successfully',
                        'resource_id': data.get('resource_id'),
                        'event_id': data.get('event_id')
                    })
                }
            else:
                return {
                    'statusCode': 500,
                    'body': json.dumps({
                        'error': f'Failed to produce cloud operation {action} to Kafka'
                    })
                }
        
        finally:
            kafka_cloud_service.close()
    
    except Exception as e:
        logger.error(f"Error in lambda_handler: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            })
        }

# Consumer function for listening to cloud events
def consume_cloud_events():
    """Consumer function to listen for cloud events from Kafka"""
    if not KAFKA_AVAILABLE:
        logger.error("Kafka client not available")
        return
    
    try:
        # Create consumer for cloud events
        topics = [
            kafka_cloud_service.topic_infrastructure_events,
            kafka_cloud_service.topic_scaling_events,
            kafka_cloud_service.topic_monitoring_events,
            kafka_cloud_service.topic_backup_events
        ]
        
        if not kafka_cloud_service.create_consumer('cloud-processor-group', topics):
            logger.error("Failed to create Kafka consumer")
            return
        
        logger.info("Starting to consume cloud events from Kafka...")
        
        for message in kafka_cloud_service.consumer:
            try:
                logger.info(f"Received cloud event from topic {message.topic}: {message.value}")
                
                # Process the cloud event
                cloud_data = message.value
                resource_id = cloud_data.get('resource_id')
                event_type = cloud_data.get('event_type')
                
                # Implement your business logic here
                logger.info(f"Processing resource {resource_id} with event type {event_type}")
                
                # Example: Execute cloud operations, update database, send notifications
                
            except Exception as e:
                logger.error(f"Error processing cloud event from Kafka: {e}")
        
    except KeyboardInterrupt:
        logger.info("Stopping cloud event consumer...")
    except Exception as e:
        logger.error(f"Error in cloud event consumer: {e}")
    finally:
        kafka_cloud_service.close()

if __name__ == "__main__":
    # Test the service
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
            "metadata": {
                "instance_type": "t3.medium",
                "ami_id": "ami-12345",
                "security_groups": ["sg-123"]
            },
            "estimated_duration": 300,
            "cost_estimate": 50.0,
            "tags": {
                "Environment": "production",
                "Team": "backend"
            }
        }
    }
    
    result = lambda_handler(test_event)
    print(json.dumps(result, indent=2))
