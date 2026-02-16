"""
Prometheus Metrics Service

This module provides comprehensive metrics collection and exposure in Prometheus format.
It includes:

- HTTP request metrics (count, duration)
- Application-level metrics (requests, errors)
- Database query metrics (count, duration)
- Cache metrics (hits, misses)
- Message queue metrics
- Notification metrics
- System resource metrics (CPU, memory, connections)

The service gracefully handles cases where prometheus-client is not installed,
allowing the application to run without metrics collection.

All metrics are exposed in Prometheus format and can be scraped by Prometheus
or consumed via JSON endpoints.

Example:
    from services.monitoring.metrics import metrics_service
    
    # Record an HTTP request
    metrics_service.record_http_request("GET", "/api/users", 200, 0.15)
    
    # Record a database query
    metrics_service.record_db_query("SELECT", "users", 0.05)
    
    # Get metrics in Prometheus format
    prometheus_metrics = metrics_service.get_metrics()
"""

import time
from typing import Dict, Any
import os

# Try to import prometheus_client, handle gracefully if not available
try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Summary, 
        generate_latest, CONTENT_TYPE_LATEST,
        CollectorRegistry, REGISTRY
    )
    from prometheus_client.multiprocess import MultiProcessCollector
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Create dummy classes if prometheus_client is not available
    class Counter:
        def __init__(self, *args, **kwargs):
            pass
        def labels(self, **kwargs):
            return self
        def inc(self, *args, **kwargs):
            pass
    class Histogram:
        def __init__(self, *args, **kwargs):
            pass
        def labels(self, **kwargs):
            return self
        def observe(self, *args, **kwargs):
            pass
    class Gauge:
        def __init__(self, *args, **kwargs):
            pass
        def labels(self, **kwargs):
            return self
        def set(self, *args, **kwargs):
            pass
    class Summary:
        def __init__(self, *args, **kwargs):
            pass
        def labels(self, **kwargs):
            return self
        def observe(self, *args, **kwargs):
            pass
    class CollectorRegistry:
        pass
    def generate_latest(*args, **kwargs):
        return b"# Prometheus client not installed\n"
    
    from core.logger import logger
    logger.warning("prometheus-client not installed. Metrics will be disabled. Install with: pip install prometheus-client")

# Create a registry for metrics
if PROMETHEUS_AVAILABLE:
    registry = CollectorRegistry()
else:
    registry = None

# HTTP Request Metrics
http_requests_total = Counter(
    'http_requests_total',
    'Total number of HTTP requests',
    ['method', 'endpoint', 'status'],
    registry=registry
)

http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint'],
    registry=registry
)

# Application Metrics
app_requests_total = Counter(
    'app_requests_total',
    'Total number of application requests',
    ['service', 'operation'],
    registry=registry
)

app_errors_total = Counter(
    'app_errors_total',
    'Total number of application errors',
    ['service', 'error_type'],
    registry=registry
)

# Database Metrics
db_queries_total = Counter(
    'db_queries_total',
    'Total number of database queries',
    ['operation', 'table'],
    registry=registry
)

db_query_duration_seconds = Histogram(
    'db_query_duration_seconds',
    'Database query duration in seconds',
    ['operation', 'table'],
    registry=registry
)

# Cache/Redis Metrics
cache_hits_total = Counter(
    'cache_hits_total',
    'Total number of cache hits',
    ['cache_type'],
    registry=registry
)

cache_misses_total = Counter(
    'cache_misses_total',
    'Total number of cache misses',
    ['cache_type'],
    registry=registry
)

# Message Queue Metrics
queue_messages_total = Counter(
    'queue_messages_total',
    'Total number of messages processed',
    ['queue_name', 'status'],
    registry=registry
)

queue_message_duration_seconds = Histogram(
    'queue_message_duration_seconds',
    'Message processing duration in seconds',
    ['queue_name'],
    registry=registry
)

# Notification Metrics
notifications_sent_total = Counter(
    'notifications_sent_total',
    'Total number of notifications sent',
    ['notification_type', 'status'],
    registry=registry
)

# System Metrics (Gauges)
active_connections = Gauge(
    'active_connections',
    'Number of active connections',
    ['connection_type'],
    registry=registry
)

memory_usage_bytes = Gauge(
    'memory_usage_bytes',
    'Memory usage in bytes',
    ['component'],
    registry=registry
)

cpu_usage_percent = Gauge(
    'cpu_usage_percent',
    'CPU usage percentage',
    ['component'],
    registry=registry
)

# Summary for response sizes
response_size_bytes = Summary(
    'response_size_bytes',
    'Response size in bytes',
    ['endpoint'],
    registry=registry
)


class MetricsService:
    """
    Service for managing and exposing Prometheus metrics.
    
    This service provides methods to record various types of metrics
    and expose them in Prometheus format. It manages a metrics registry
    and supports both single-process and multi-process metric collection.
    
    Attributes:
        registry (CollectorRegistry): Prometheus metrics registry.
                                     None if prometheus-client is not available.
    
    Example:
        >>> service = MetricsService()
        >>> service.record_http_request("POST", "/api/users", 201, 0.25)
        >>> service.record_app_error("user_service", "ValidationError")
        >>> metrics = service.get_metrics()
    """
    
    def __init__(self):
        """
        Initialize the MetricsService.
        
        Sets up the metrics registry. If prometheus-client is not available,
        the registry will be None and metric recording will be no-ops.
        """
        self.registry = registry
    
    def record_http_request(self, method: str, endpoint: str, status: int, duration: float):
        """
        Record HTTP request metrics.
        
        Increments the HTTP request counter and records the request duration
        in the histogram. Both metrics are labeled with method and endpoint.
        
        Args:
            method (str): HTTP method (e.g., "GET", "POST", "PUT", "DELETE").
            endpoint (str): API endpoint path (e.g., "/api/users", "/health").
            status (int): HTTP status code (e.g., 200, 404, 500).
            duration (float): Request duration in seconds.
        
        Example:
            >>> service.record_http_request("GET", "/api/users", 200, 0.15)
            >>> service.record_http_request("POST", "/api/users", 201, 0.25)
        """
        http_requests_total.labels(method=method, endpoint=endpoint, status=status).inc()
        http_request_duration_seconds.labels(method=method, endpoint=endpoint).observe(duration)
    
    def record_app_request(self, service: str, operation: str):
        """
        Record an application-level request.
        
        Increments the application request counter for tracking
        internal service operations and business logic calls.
        
        Args:
            service (str): Service or module name (e.g., "user_service", "notification_service").
            operation (str): Operation name (e.g., "create_user", "send_email").
        
        Example:
            >>> service.record_app_request("user_service", "create_user")
            >>> service.record_app_request("notification_service", "send_email")
        """
        app_requests_total.labels(service=service, operation=operation).inc()
    
    def record_app_error(self, service: str, error_type: str):
        """
        Record an application error.
        
        Increments the application error counter for tracking
        errors by service and error type.
        
        Args:
            service (str): Service or module name where the error occurred
                         (e.g., "user_service", "database").
            error_type (str): Type of error (e.g., "ValidationError", "DatabaseError",
                            "TimeoutError").
        
        Example:
            >>> service.record_app_error("user_service", "ValidationError")
            >>> service.record_app_error("database", "ConnectionError")
        """
        app_errors_total.labels(service=service, error_type=error_type).inc()
    
    def record_db_query(self, operation: str, table: str, duration: float):
        """
        Record database query metrics.
        
        Increments the database query counter and records the query duration
        in the histogram. Both metrics are labeled with operation type and table name.
        
        Args:
            operation (str): Database operation type (e.g., "SELECT", "INSERT", "UPDATE",
                           "DELETE", "CREATE").
            table (str): Database table name (e.g., "users", "notifications").
            duration (float): Query execution duration in seconds.
        
        Example:
            >>> service.record_db_query("SELECT", "users", 0.05)
            >>> service.record_db_query("INSERT", "notifications", 0.12)
        """
        db_queries_total.labels(operation=operation, table=table).inc()
        db_query_duration_seconds.labels(operation=operation, table=table).observe(duration)
    
    def record_cache_hit(self, cache_type: str):
        """
        Record a cache hit.
        
        Increments the cache hits counter for tracking cache effectiveness.
        
        Args:
            cache_type (str): Type of cache (e.g., "redis", "memory", "file").
        
        Example:
            >>> service.record_cache_hit("redis")
            >>> service.record_cache_hit("memory")
        """
        cache_hits_total.labels(cache_type=cache_type).inc()
    
    def record_cache_miss(self, cache_type: str):
        """
        Record a cache miss.
        
        Increments the cache misses counter for tracking cache effectiveness.
        
        Args:
            cache_type (str): Type of cache (e.g., "redis", "memory", "file").
        
        Example:
            >>> service.record_cache_miss("redis")
            >>> service.record_cache_miss("memory")
        """
        cache_misses_total.labels(cache_type=cache_type).inc()
    
    def record_queue_message(self, queue_name: str, status: str, duration: float = None):
        """
        Record queue message processing metrics.
        
        Increments the queue message counter and optionally records
        the processing duration in the histogram.
        
        Args:
            queue_name (str): Name of the message queue (e.g., "email_queue", "notification_queue").
            status (str): Processing status (e.g., "success", "failed", "retry").
            duration (Optional[float]): Message processing duration in seconds.
                                      If None, only the counter is incremented.
        
        Example:
            >>> service.record_queue_message("email_queue", "success", 0.5)
            >>> service.record_queue_message("notification_queue", "failed", 1.2)
        """
        queue_messages_total.labels(queue_name=queue_name, status=status).inc()
        if duration is not None:
            queue_message_duration_seconds.labels(queue_name=queue_name).observe(duration)
    
    def record_notification(self, notification_type: str, status: str):
        """
        Record a notification being sent.
        
        Increments the notifications sent counter for tracking
        notification delivery by type and status.
        
        Args:
            notification_type (str): Type of notification (e.g., "email", "sms", "push",
                                   "webhook").
            status (str): Delivery status (e.g., "sent", "failed", "pending").
        
        Example:
            >>> service.record_notification("email", "sent")
            >>> service.record_notification("sms", "failed")
        """
        notifications_sent_total.labels(notification_type=notification_type, status=status).inc()
    
    def set_active_connections(self, connection_type: str, count: int):
        """
        Set the current number of active connections.
        
        Updates the gauge metric for active connections. This is a point-in-time
        value that can go up or down.
        
        Args:
            connection_type (str): Type of connection (e.g., "websocket", "database",
                                 "redis", "http").
            count (int): Current number of active connections.
        
        Example:
            >>> service.set_active_connections("websocket", 15)
            >>> service.set_active_connections("database", 5)
        """
        active_connections.labels(connection_type=connection_type).set(count)
    
    def set_memory_usage(self, component: str, bytes_used: int):
        """
        Set the current memory usage for a component.
        
        Updates the gauge metric for memory usage. This is a point-in-time
        value that can change over time.
        
        Args:
            component (str): Component name (e.g., "app", "cache", "database").
            bytes_used (int): Memory usage in bytes.
        
        Example:
            >>> # Set memory usage for application (100 MB)
            >>> service.set_memory_usage("app", 100 * 1024 * 1024)
            >>> service.set_memory_usage("cache", 50 * 1024 * 1024)
        """
        memory_usage_bytes.labels(component=component).set(bytes_used)
    
    def set_cpu_usage(self, component: str, percent: float):
        """
        Set the current CPU usage for a component.
        
        Updates the gauge metric for CPU usage. This is a point-in-time
        value that can change over time.
        
        Args:
            component (str): Component name (e.g., "app", "worker", "scheduler").
            percent (float): CPU usage as a percentage (0.0 to 100.0).
        
        Example:
            >>> service.set_cpu_usage("app", 25.5)
            >>> service.set_cpu_usage("worker", 45.2)
        """
        cpu_usage_percent.labels(component=component).set(percent)
    
    def record_response_size(self, endpoint: str, size: int):
        """
        Record the size of an HTTP response.
        
        Records the response size in the summary metric for tracking
        response payload sizes by endpoint.
        
        Args:
            endpoint (str): API endpoint path (e.g., "/api/users", "/api/data").
            size (int): Response size in bytes.
        
        Example:
            >>> service.record_response_size("/api/users", 2048)
            >>> service.record_response_size("/api/data", 10240)
        """
        response_size_bytes.labels(endpoint=endpoint).observe(size)
    
    def get_metrics(self) -> bytes:
        """
        Get all metrics in Prometheus text format.
        
        Generates metrics output in the Prometheus exposition format,
        which can be scraped by Prometheus or other monitoring tools.
        Supports both single-process and multi-process modes.
        
        Returns:
            bytes: Metrics in Prometheus text format. Returns a placeholder
                  message if prometheus-client is not installed.
        
        Example:
            >>> service = MetricsService()
            >>> metrics_bytes = service.get_metrics()
            >>> # Use in HTTP response
            >>> return Response(content=metrics_bytes, media_type="text/plain")
        """
        if not PROMETHEUS_AVAILABLE:
            return b"# Prometheus client not installed\n"
        
        if 'prometheus_multiproc_dir' in os.environ:
            # Multi-process mode
            collector = MultiProcessCollector(registry)
            return generate_latest(collector)
        else:
            return generate_latest(registry)
    
    def get_metrics_dict(self) -> Dict[str, Any]:
        """
        Get metrics as a dictionary for JSON API responses.
        
        Converts Prometheus text format metrics into a structured dictionary
        that can be easily consumed by JSON APIs or frontend applications.
        
        Returns:
            Dict[str, Any]: Dictionary containing:
                - metrics (Dict[str, str]): Dictionary of metric names to values
                - total_metrics (int): Total number of metrics
                - format (str): Format identifier ("prometheus")
        
        Example:
            >>> service = MetricsService()
            >>> metrics_dict = service.get_metrics_dict()
            >>> print(f"Total metrics: {metrics_dict['total_metrics']}")
            >>> for name, value in metrics_dict['metrics'].items():
            ...     print(f"{name}: {value}")
        """
        metrics_text = self.get_metrics().decode('utf-8')
        metrics_dict = {}
        
        for line in metrics_text.split('\n'):
            if line and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 2:
                    metric_name = parts[0]
                    value = parts[1]
                    metrics_dict[metric_name] = value
        
        return {
            "metrics": metrics_dict,
            "total_metrics": len(metrics_dict),
            "format": "prometheus"
        }


# Global metrics service instance
metrics_service = MetricsService()
