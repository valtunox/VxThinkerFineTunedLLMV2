"""
Monitoring Services Router

FastAPI router providing comprehensive monitoring endpoints for:
- Prometheus metrics exposure
- Performance metrics and statistics
- Log querying and analysis
- Health checks and service status

All endpoints are prefixed with "/monitoring" and organized by category:
- /monitoring/metrics/* - Prometheus metrics endpoints
- /monitoring/performance/* - Performance monitoring endpoints
- /monitoring/logs/* - Log query and analysis endpoints
- /monitoring/health - Health check endpoint

Example:
    from fastapi import FastAPI
    from services.monitoring.monitoring_services_router import router
    
    app = FastAPI()
    app.include_router(router)
    
    # Access endpoints:
    # GET /monitoring/metrics - Prometheus metrics
    # GET /monitoring/performance - Current performance metrics
    # GET /monitoring/logs - Query logs
"""

from fastapi import APIRouter, Query, HTTPException, Request, Response
from fastapi.responses import PlainTextResponse, FileResponse
from typing import Optional
from datetime import datetime, timedelta
import time

from .metrics import metrics_service
from .performance import performance_service
from .logs import logs_service
from .observability import observability_service
from core.logger import logger

router = APIRouter(prefix="/monitoring", tags=["Monitoring"])


# ==================== Metrics Endpoints ====================

@router.get("/metrics")
@router.get("/metrics/prometheus")
async def get_prometheus_metrics():
    """
    Get metrics in Prometheus exposition format.
    
    This endpoint exposes all collected metrics in the Prometheus text format,
    which can be scraped by Prometheus or other monitoring tools. The response
    uses the appropriate content type for Prometheus scraping.
    
    Returns:
        Response: Plain text response with Prometheus format metrics.
                 Content-Type: text/plain; version=0.0.4; charset=utf-8
    
    Raises:
        HTTPException: 500 if there's an error generating metrics.
    
    Example:
        ```bash
        curl http://localhost:8000/monitoring/metrics
        ```
        
        Prometheus scrape config:
        ```yaml
        scrape_configs:
          - job_name: 'app'
            scrape_interval: 15s
            metrics_path: '/monitoring/metrics'
            static_configs:
              - targets: ['localhost:8000']
        ```
    """
    try:
        metrics_data = metrics_service.get_metrics()
        return Response(
            content=metrics_data,
            media_type="text/plain; version=0.0.4; charset=utf-8"
        )
    except Exception as e:
        logger.error(f"Error getting Prometheus metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/json")
async def get_metrics_json():
    """
    Get metrics as JSON format.
    
    Returns all metrics in a structured JSON format suitable for API
    consumption, frontend dashboards, or programmatic access. This is
    more convenient than parsing Prometheus text format.
    
    Returns:
        Dict[str, Any]: JSON object containing:
            - metrics (Dict[str, str]): Dictionary of metric names to values
            - total_metrics (int): Total number of metrics
            - format (str): Format identifier ("prometheus")
    
    Raises:
        HTTPException: 500 if there's an error getting metrics.
    
    Example:
        ```bash
        curl http://localhost:8000/monitoring/metrics/json
        ```
        
        Response:
        ```json
        {
          "metrics": {
            "http_requests_total": "1234",
            "app_errors_total": "5"
          },
          "total_metrics": 2,
          "format": "prometheus"
        }
        ```
    """
    try:
        return metrics_service.get_metrics_dict()
    except Exception as e:
        logger.error(f"Error getting metrics JSON: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/metrics/test")
async def test_metrics():
    """
    Generate sample metrics for testing purposes.
    
    Creates sample metric data across various metric types to help
    test monitoring dashboards, alerts, and metric collection. Useful
    for development and testing scenarios.
    
    Returns:
        Dict[str, Any]: Success response with timestamp:
            - status (str): "success"
            - message (str): Confirmation message
            - timestamp (str): ISO format timestamp
    
    Raises:
        HTTPException: 500 if there's an error generating test metrics.
    
    Example:
        ```bash
        curl -X POST http://localhost:8000/monitoring/metrics/test
        ```
    """
    try:
        # Generate test metrics
        metrics_service.record_http_request("GET", "/test", 200, 0.1)
        metrics_service.record_app_request("test", "operation")
        metrics_service.record_db_query("SELECT", "users", 0.05)
        metrics_service.record_cache_hit("redis")
        metrics_service.record_notification("email", "sent")
        metrics_service.set_active_connections("websocket", 5)
        metrics_service.set_memory_usage("app", 1024 * 1024 * 100)  # 100MB
        metrics_service.set_cpu_usage("app", 25.5)
        
        return {
            "status": "success",
            "message": "Test metrics generated",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error generating test metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Performance Endpoints ====================

@router.get("/performance")
@router.get("/performance/current")
async def get_performance_metrics():
    """
    Get current comprehensive performance metrics.
    
    Returns real-time performance data including:
    - Process metrics (CPU, memory, threads, file descriptors)
    - System metrics (CPU, memory, disk usage)
    - Network I/O statistics
    - Application metrics (response times, error rates, requests/sec)
    - Performance alerts based on thresholds
    
    Returns:
        Dict[str, Any]: Comprehensive performance metrics dictionary.
                       See PerformanceService.get_current_metrics() for structure.
    
    Raises:
        HTTPException: 500 if there's an error getting performance metrics.
    
    Example:
        ```bash
        curl http://localhost:8000/monitoring/performance
        ```
    """
    try:
        return performance_service.get_current_metrics()
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance/response-times")
async def get_response_time_stats(
    endpoint: Optional[str] = Query(None, description="Filter by endpoint"),
    minutes: int = Query(5, description="Time period in minutes", ge=1, le=60)
):
    """
    Get response time statistics for a time period.
    
    Calculates statistical measures (min, max, average, percentiles) for
    response times. Can filter by specific endpoint or aggregate across all.
    
    Args:
        endpoint (Optional[str]): Filter by specific endpoint path.
                                 If not provided, aggregates across all endpoints.
        minutes (int): Time window in minutes (1-60). Defaults to 5.
    
    Returns:
        Dict[str, Any]: Response time statistics including:
            - endpoint (str): Endpoint name or "all"
            - period_minutes (int): Time period analyzed
            - count (int): Number of response time records
            - stats (Dict): min, max, avg, p50, p95, p99 percentiles
    
    Raises:
        HTTPException: 500 if there's an error getting statistics.
    
    Example:
        ```bash
        # Get stats for all endpoints (last 10 minutes)
        curl "http://localhost:8000/monitoring/performance/response-times?minutes=10"
        
        # Get stats for specific endpoint
        curl "http://localhost:8000/monitoring/performance/response-times?endpoint=/api/users&minutes=30"
        ```
    """
    try:
        return performance_service.get_response_time_stats(endpoint, minutes)
    except Exception as e:
        logger.error(f"Error getting response time stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance/errors")
async def get_error_stats(
    minutes: int = Query(5, description="Time period in minutes", ge=1, le=60)
):
    """
    Get error statistics for a time period.
    
    Analyzes errors within the specified time window and provides
    breakdown by error type and overall error rate.
    
    Args:
        minutes (int): Time window in minutes (1-60). Defaults to 5.
    
    Returns:
        Dict[str, Any]: Error statistics including:
            - period_minutes (int): Time period analyzed
            - total_errors (int): Total number of errors
            - error_breakdown (Dict[str, int]): Count by error type
            - error_rate (float): Error rate (errors / requests)
    
    Raises:
        HTTPException: 500 if there's an error getting error stats.
    
    Example:
        ```bash
        curl "http://localhost:8000/monitoring/performance/errors?minutes=60"
        ```
    """
    try:
        return performance_service.get_error_stats(minutes)
    except Exception as e:
        logger.error(f"Error getting error stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance/history")
async def get_performance_history(
    metric_type: str = Query("response_time", description="Type: response_time, errors, requests"),
    limit: int = Query(100, description="Number of records", ge=1, le=1000)
):
    """
    Get historical performance metrics.
    
    Retrieves the most recent records from the specified metric history.
    Useful for trend analysis and historical performance review.
    
    Args:
        metric_type (str): Type of metric to retrieve. Options:
                          - "response_time": Response time records
                          - "errors": Error records
                          - "requests": Request metric records
        limit (int): Maximum number of records to return (1-1000). Defaults to 100.
    
    Returns:
        Dict[str, Any]: Dictionary containing:
            - metric_type (str): The requested metric type
            - data (List[Dict]): List of historical metric records
    
    Raises:
        HTTPException: 500 if there's an error getting history.
    
    Example:
        ```bash
        # Get last 50 response times
        curl "http://localhost:8000/monitoring/performance/history?metric_type=response_time&limit=50"
        
        # Get last 20 errors
        curl "http://localhost:8000/monitoring/performance/history?metric_type=errors&limit=20"
        ```
    """
    try:
        return {
            "metric_type": metric_type,
            "data": performance_service.get_history(metric_type, limit)
        }
    except Exception as e:
        logger.error(f"Error getting performance history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/performance/test")
async def test_performance():
    """
    Generate sample performance data for testing.
    
    Creates sample response time and error records to help test
    performance monitoring dashboards and alerts. Useful for development
    and testing scenarios.
    
    Returns:
        Dict[str, Any]: Success response with timestamp:
            - status (str): "success"
            - message (str): Confirmation message
            - timestamp (str): ISO format timestamp
    
    Raises:
        HTTPException: 500 if there's an error generating test data.
    
    Example:
        ```bash
        curl -X POST http://localhost:8000/monitoring/performance/test
        ```
    """
    try:
        # Simulate some requests
        for i in range(10):
            endpoint = f"/test/endpoint/{i % 3}"
            duration = 0.1 + (i * 0.01)
            performance_service.record_response_time(endpoint, duration)
            time.sleep(0.01)
        
        # Simulate some errors
        performance_service.record_error("/test/endpoint/1", "TimeoutError")
        performance_service.record_error("/test/endpoint/2", "ValueError")
        
        return {
            "status": "success",
            "message": "Test performance data generated",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error generating test performance data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Logs Endpoints ====================

@router.get("/logs")
async def get_logs(
    level: Optional[str] = Query(None, description="Filter by log level (INFO, WARNING, ERROR)"),
    search: Optional[str] = Query(None, description="Search in log messages"),
    start_time: Optional[str] = Query(None, description="Start time (ISO format)"),
    end_time: Optional[str] = Query(None, description="End time (ISO format)"),
    limit: int = Query(100, description="Number of logs to return", ge=1, le=1000),
    offset: int = Query(0, description="Offset for pagination", ge=0)
):
    """
    Query logs with various filters and pagination.
    
    Retrieves log entries matching the specified filters. Supports filtering
    by log level, time range, and search terms. Results are paginated and
    sorted by timestamp (newest first).
    
    Args:
        level (Optional[str]): Filter by log level (INFO, WARNING, ERROR).
                              Case-insensitive.
        search (Optional[str]): Search term to match in log messages.
                               Case-insensitive.
        start_time (Optional[str]): Start time in ISO format (e.g., "2024-01-01T00:00:00").
        end_time (Optional[str]): End time in ISO format.
        limit (int): Maximum number of logs to return (1-1000). Defaults to 100.
        offset (int): Number of logs to skip for pagination. Defaults to 0.
    
    Returns:
        Dict[str, Any]: Dictionary containing:
            - logs (List[Dict]): List of log entries
            - total (int): Total number of matching logs
            - limit (int): Requested limit
            - offset (int): Requested offset
            - has_more (bool): Whether more logs are available
    
    Raises:
        HTTPException: 500 if there's an error querying logs.
    
    Example:
        ```bash
        # Get recent ERROR logs
        curl "http://localhost:8000/monitoring/logs?level=ERROR&limit=50"
        
        # Search logs
        curl "http://localhost:8000/monitoring/logs?search=database&limit=100"
        
        # Get logs in time range
        curl "http://localhost:8000/monitoring/logs?start_time=2024-01-01T00:00:00&end_time=2024-01-01T23:59:59"
        ```
    """
    try:
        start = datetime.fromisoformat(start_time) if start_time else None
        end = datetime.fromisoformat(end_time) if end_time else None
        
        return logs_service.get_logs(
            level=level,
            start_time=start,
            end_time=end,
            search=search,
            limit=limit,
            offset=offset
        )
    except Exception as e:
        logger.error(f"Error querying logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/logs/stats")
async def get_log_stats(
    hours: int = Query(24, description="Time period in hours", ge=1, le=168)
):
    """
    Get statistical information about logs for a time period.
    
    Calculates total log count, breakdown by log level, and hourly
    distribution for the specified number of hours from the current time.
    
    Args:
        hours (int): Number of hours to look back (1-168). Defaults to 24.
    
    Returns:
        Dict[str, Any]: Log statistics including:
            - period_hours (int): Time period analyzed
            - total_logs (int): Total number of logs
            - by_level (Dict[str, int]): Count by log level
            - by_hour (Dict[str, int]): Count by hour
            - start_time (str): ISO format start timestamp
            - end_time (str): ISO format end timestamp
    
    Raises:
        HTTPException: 500 if there's an error getting log stats.
    
    Example:
        ```bash
        # Get stats for last 48 hours
        curl "http://localhost:8000/monitoring/logs/stats?hours=48"
        ```
    """
    try:
        return logs_service.get_log_stats(hours)
    except Exception as e:
        logger.error(f"Error getting log stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/logs/errors")
async def get_recent_errors(
    limit: int = Query(50, description="Number of errors to return", ge=1, le=500)
):
    """
    Get the most recent error-level logs.
    
    Convenience endpoint that returns the most recent ERROR level logs,
    sorted by timestamp (newest first).
    
    Args:
        limit (int): Maximum number of error logs to return (1-500). Defaults to 50.
    
    Returns:
        Dict[str, Any]: Dictionary with same structure as /logs endpoint,
                       containing only ERROR level logs.
    
    Raises:
        HTTPException: 500 if there's an error getting errors.
    
    Example:
        ```bash
        curl "http://localhost:8000/monitoring/logs/errors?limit=100"
        ```
    """
    try:
        return logs_service.get_recent_errors(limit)
    except Exception as e:
        logger.error(f"Error getting recent errors: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/logs/warnings")
async def get_recent_warnings(
    limit: int = Query(50, description="Number of warnings to return", ge=1, le=500)
):
    """
    Get the most recent warning-level logs.
    
    Convenience endpoint that returns the most recent WARNING level logs,
    sorted by timestamp (newest first).
    
    Args:
        limit (int): Maximum number of warning logs to return (1-500). Defaults to 50.
    
    Returns:
        Dict[str, Any]: Dictionary with same structure as /logs endpoint,
                       containing only WARNING level logs.
    
    Raises:
        HTTPException: 500 if there's an error getting warnings.
    
    Example:
        ```bash
        curl "http://localhost:8000/monitoring/logs/warnings?limit=100"
        ```
    """
    try:
        return logs_service.get_recent_warnings(limit)
    except Exception as e:
        logger.error(f"Error getting recent warnings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/logs/search")
async def search_logs(
    q: str = Query(..., description="Search query"),
    limit: int = Query(100, description="Number of results", ge=1, le=1000)
):
    """
    Search logs by query string in log messages.
    
    Performs a case-insensitive search across all log messages and returns
    matching entries sorted by timestamp (newest first).
    
    Args:
        q (str): Search query string. Required.
        limit (int): Maximum number of results to return (1-1000). Defaults to 100.
    
    Returns:
        Dict[str, Any]: Dictionary with same structure as /logs endpoint,
                       containing only logs matching the search query.
    
    Raises:
        HTTPException: 500 if there's an error searching logs.
    
    Example:
        ```bash
        curl "http://localhost:8000/monitoring/logs/search?q=database%20connection&limit=50"
        ```
    """
    try:
        return logs_service.search_logs(q, limit)
    except Exception as e:
        logger.error(f"Error searching logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/logs/tail")
async def tail_logs(
    lines: int = Query(50, description="Number of lines to return", ge=1, le=1000)
):
    """
    Get the last N lines from the log file.
    
    Efficiently retrieves the most recent log entries by reading only
    the last N lines of the file. Useful for monitoring real-time
    log activity.
    
    Args:
        lines (int): Number of lines to retrieve from the end of the file (1-1000).
                    Defaults to 50.
    
    Returns:
        Dict[str, Any]: Dictionary containing:
            - logs (List[Dict]): List of parsed log entries
            - total (int): Number of logs returned
            - requested_lines (int): Number of lines requested
    
    Raises:
        HTTPException: 500 if there's an error tailing logs.
    
    Example:
        ```bash
        curl "http://localhost:8000/monitoring/logs/tail?lines=100"
        ```
    """
    try:
        return logs_service.tail_logs(lines)
    except Exception as e:
        logger.error(f"Error tailing logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/logs/info")
async def get_log_file_info():
    """
    Get metadata information about the log file.
    
    Retrieves file system information including file size, modification time,
    creation time, total line count, and file path.
    
    Returns:
        Dict[str, Any]: Dictionary containing:
            - exists (bool): Whether the log file exists
            - path (str): Full path to the log file
            - size_bytes (int): File size in bytes
            - size_mb (float): File size in megabytes
            - modified (str): Last modification time in ISO format
            - created (str): Creation time in ISO format
            - total_lines (int): Total number of lines in the file
    
    Raises:
        HTTPException: 500 if there's an error getting log file info.
    
    Example:
        ```bash
        curl http://localhost:8000/monitoring/logs/info
        ```
    """
    try:
        info = logs_service.get_log_file_info()
        # Add line count
        if info.get("exists"):
            info["total_lines"] = logs_service.get_log_file_lines_count()
        return info
    except Exception as e:
        logger.error(f"Error getting log file info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/logs/file")
async def get_log_file(
    limit_lines: Optional[int] = Query(None, description="Limit to last N lines (optional)", ge=1, le=10000),
    format: str = Query("text", description="Response format: text or json")
):
    """
    Get raw log file content.
    
    Returns the actual log file content as plain text or structured JSON.
    Useful for downloading logs or displaying raw log content in UIs.
    
    Args:
        limit_lines (Optional[int]): If specified, only return the last N lines (1-10000).
                                    If None, returns the entire file content.
        format (str): Response format. Options: "text" or "json". Defaults to "text".
    
    Returns:
        Union[PlainTextResponse, Dict[str, Any]]:
            - If format="text": PlainTextResponse with log content
            - If format="json": JSON object with file metadata and content
    
    Raises:
        HTTPException: 404 if log file doesn't exist.
        HTTPException: 500 if there's an error reading the file.
    
    Example:
        ```bash
        # Get last 1000 lines as text
        curl "http://localhost:8000/monitoring/logs/file?limit_lines=1000&format=text"
        
        # Get entire file as JSON
        curl "http://localhost:8000/monitoring/logs/file?format=json"
        ```
    """
    try:
        log_info = logs_service.get_log_file_info()
        if not log_info.get("exists"):
            raise HTTPException(status_code=404, detail="Log file not found")
        
        content = logs_service.get_log_file_content(limit_lines=limit_lines)
        total_lines = logs_service.get_log_file_lines_count()
        
        if format == "json":
            return {
                "file_path": log_info.get("path"),
                "file_size_bytes": log_info.get("size_bytes"),
                "total_lines": total_lines,
                "returned_lines": limit_lines if limit_lines else total_lines,
                "content": content,
                "content_length": len(content)
            }
        else:
            # Return as plain text
            return PlainTextResponse(
                content=content,
                media_type="text/plain; charset=utf-8",
                headers={
                    "X-Log-File-Path": log_info.get("path", ""),
                    "X-Total-Lines": str(total_lines),
                    "X-Returned-Lines": str(limit_lines if limit_lines else total_lines)
                }
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting log file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/logs/file/download")
async def download_log_file():
    """
    Download the log file as a file attachment.
    
    Returns the log file as a downloadable attachment with appropriate
    headers for file download. The filename includes a timestamp.
    
    Returns:
        FileResponse: File response with log file content and download headers.
                     Filename format: logs_YYYYMMDD_HHMMSS.txt
    
    Raises:
        HTTPException: 404 if log file doesn't exist.
        HTTPException: 500 if there's an error downloading the file.
    
    Example:
        ```bash
        curl -O http://localhost:8000/monitoring/logs/file/download
        ```
    """
    try:
        log_info = logs_service.get_log_file_info()
        if not log_info.get("exists"):
            raise HTTPException(status_code=404, detail="Log file not found")
        
        log_file_path = logs_service.log_file
        
        return FileResponse(
            path=str(log_file_path),
            filename=f"logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            media_type="text/plain",
            headers={
                "Content-Disposition": f"attachment; filename=logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading log file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Observability (OpenTelemetry) ====================

@router.get("/observability/status")
async def observability_status():
    try:
        return observability_service.status()
    except Exception as e:
        logger.error(f"Error getting observability status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Health & Status Endpoints ====================

@router.get("/health")
@router.get("/status")
async def monitoring_health():
    """
    Health check endpoint for monitoring service.
    
    Checks the availability and status of all monitoring services
    (metrics, performance, logs) and returns overall health status.
    
    Returns:
        Dict[str, Any]: Health status containing:
            - status (str): Overall status ("healthy" or "unhealthy")
            - timestamp (str): ISO format timestamp
            - services (Dict): Status of each service:
                - metrics (str): "available"
                - performance (str): "available" or "unavailable"
                - logs (str): "available" or "unavailable"
            - uptime_seconds (float): Service uptime
            - error (str): Error message if unhealthy
    
    Example:
        ```bash
        curl http://localhost:8000/monitoring/health
        ```
        
        Response:
        ```json
        {
          "status": "healthy",
          "timestamp": "2024-01-01T12:00:00",
          "services": {
            "metrics": "available",
            "performance": "available",
            "logs": "available"
          },
          "uptime_seconds": 3600.0
        }
        ```
    """
    try:
        perf_metrics = performance_service.get_current_metrics()
        log_info = logs_service.get_log_file_info()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "metrics": "available",
                "performance": "available" if perf_metrics.get("process") else "unavailable",
                "logs": "available" if log_info.get("exists") else "unavailable"
            },
            "uptime_seconds": perf_metrics.get("uptime_seconds", 0)
        }
    except Exception as e:
        logger.error(f"Error in monitoring health check: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@router.get("/")
async def monitoring_info():
    """
    Get monitoring service information and available endpoints.
    
    Returns comprehensive information about the monitoring service including
    version, description, and all available endpoints organized by category.
    Also includes information about external monitoring services.
    
    Returns:
        Dict[str, Any]: Service information containing:
            - service (str): Service name
            - version (str): Service version
            - description (str): Service description
            - endpoints (Dict): All available endpoints organized by category:
                - metrics: Metrics-related endpoints
                - performance: Performance monitoring endpoints
                - logs: Log query and analysis endpoints
                - health: Health check endpoints
            - external_services (Dict): External monitoring service URLs
    
    Example:
        ```bash
        curl http://localhost:8000/monitoring/
        ```
        
        This endpoint is useful for:
        - API documentation
        - Service discovery
        - Understanding available monitoring capabilities
    """
    return {
        "service": "Monitoring Service",
        "version": "1.0.0",
        "description": "Comprehensive monitoring, metrics, logs, and performance tracking",
        "endpoints": {
            "metrics": {
                "prometheus": "GET /monitoring/metrics - Prometheus format metrics",
                "json": "GET /monitoring/metrics/json - JSON format metrics",
                "test": "POST /monitoring/metrics/test - Generate test metrics"
            },
            "performance": {
                "current": "GET /monitoring/performance - Current performance metrics",
                "response_times": "GET /monitoring/performance/response-times - Response time stats",
                "errors": "GET /monitoring/performance/errors - Error statistics",
                "history": "GET /monitoring/performance/history - Historical data",
                "test": "POST /monitoring/performance/test - Generate test data"
            },
            "logs": {
                "query": "GET /monitoring/logs - Query logs with filters",
                "stats": "GET /monitoring/logs/stats - Log statistics",
                "errors": "GET /monitoring/logs/errors - Recent errors",
                "warnings": "GET /monitoring/logs/warnings - Recent warnings",
                "search": "GET /monitoring/logs/search?q=query - Search logs",
                "tail": "GET /monitoring/logs/tail - Last N lines",
                "info": "GET /monitoring/logs/info - Log file information",
                "file": "GET /monitoring/logs/file - Get raw log file content",
                "download": "GET /monitoring/logs/file/download - Download log file"
            },
            "health": {
                "status": "GET /monitoring/health - Health check",
                "info": "GET /monitoring/ - Service information"
            }
        },
        "external_services": {
            "prometheus": "http://localhost:9090",
            "grafana": "http://localhost:3001",
            "alertmanager": "http://localhost:9093",
            "node_exporter": "http://localhost:9100/metrics",
            "cadvisor": "http://localhost:8082"
        }
    }
