"""
Monitoring Service Package

This package provides comprehensive monitoring, metrics, logs, and performance
tracking capabilities for the application. It includes:

Modules:
    - metrics: Prometheus metrics collection and exposure
    - performance: Real-time performance monitoring (CPU, memory, response times)
    - logs: Log aggregation, querying, and analysis
    - monitoring_services_router: FastAPI router with all monitoring endpoints

Services:
    - metrics_service: Global instance for recording and exposing metrics
    - performance_service: Global instance for performance monitoring
    - logs_service: Global instance for log querying and analysis
    - monitoring_services_router: FastAPI router with all monitoring endpoints

Usage:
    ```python
    from services.monitoring import metrics_service, performance_service, logs_service, router
    
    # Record metrics
    metrics_service.record_http_request("GET", "/api/users", 200, 0.15)
    
    # Monitor performance
    performance_service.record_response_time("/api/users", 0.15)
    metrics = performance_service.get_current_metrics()
    
    # Query logs
    errors = logs_service.get_recent_errors(limit=50)
    
    # Include router in FastAPI app
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    ```

All services are initialized as global singletons and can be imported
directly for use throughout the application.
"""

from .metrics import metrics_service
from .performance import performance_service
from .logs import logs_service
from .monitoring_services_router import router

__all__ = [
    "metrics_service",
    "performance_service",
    "logs_service",
    "router"
]
