# Monitoring Service

Comprehensive monitoring, metrics, logs, and performance tracking for the LLM AI Lab application.

## Features

- **Prometheus Metrics**: Expose application metrics in Prometheus format
- **Performance Monitoring**: Track CPU, memory, response times, and system metrics
- **Log Aggregation**: Query and analyze application logs
- **Health Checks**: Monitor service health and status
- **Automatic Request Tracking**: Middleware automatically tracks all HTTP requests

## Dependencies

Install required packages:

```bash
pip install prometheus-client psutil
```

## Endpoints

All endpoints are prefixed with `/api/monitoring`

### Metrics Endpoints

- `GET /api/monitoring/metrics` - Prometheus format metrics (scraped by Prometheus)
- `GET /api/monitoring/metrics/json` - Metrics as JSON
- `POST /api/monitoring/metrics/test` - Generate test metrics

### Performance Endpoints

- `GET /api/monitoring/performance` - Current performance metrics
- `GET /api/monitoring/performance/response-times` - Response time statistics
- `GET /api/monitoring/performance/errors` - Error statistics
- `GET /api/monitoring/performance/history` - Historical performance data
- `POST /api/monitoring/performance/test` - Generate test performance data

### Logs Endpoints

- `GET /api/monitoring/logs` - Query logs with filters
- `GET /api/monitoring/logs/stats` - Log statistics
- `GET /api/monitoring/logs/errors` - Recent error logs
- `GET /api/monitoring/logs/warnings` - Recent warning logs
- `GET /api/monitoring/logs/search?q=query` - Search logs
- `GET /api/monitoring/logs/tail` - Last N lines of logs
- `GET /api/monitoring/logs/info` - Log file information
- `GET /api/monitoring/logs/file` - Get raw log file content (as text or JSON)
- `GET /api/monitoring/logs/file/download` - Download log file

### Health & Status

- `GET /api/monitoring/health` - Health check
- `GET /api/monitoring/` - Service information and available endpoints

## Testing Endpoints

### Test Metrics
```bash
curl -X POST http://localhost:8484/api/monitoring/metrics/test
```

### Test Performance
```bash
curl -X POST http://localhost:8484/api/monitoring/performance/test
```

### Get Current Performance
```bash
curl http://localhost:8484/api/monitoring/performance
```

### Get Prometheus Metrics
```bash
curl http://localhost:8484/api/monitoring/metrics
```

### Query Logs
```bash
curl "http://localhost:8484/api/monitoring/logs?level=ERROR&limit=10"
```

### Search Logs
```bash
curl "http://localhost:8484/api/monitoring/logs/search?q=error"
```

## Prometheus Integration

The metrics endpoint at `/api/monitoring/metrics` is designed to be scraped by Prometheus. Add this to your `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'llm-ai-lab'
    static_configs:
      - targets: ['localhost:8484']
    metrics_path: '/api/monitoring/metrics'
    scrape_interval: 15s
```

## Metrics Collected

### HTTP Metrics
- `http_requests_total` - Total HTTP requests by method, endpoint, status
- `http_request_duration_seconds` - HTTP request duration

### Application Metrics
- `app_requests_total` - Application requests by service, operation
- `app_errors_total` - Application errors by service, error type

### Database Metrics
- `db_queries_total` - Database queries by operation, table
- `db_query_duration_seconds` - Database query duration

### Cache Metrics
- `cache_hits_total` - Cache hits by cache type
- `cache_misses_total` - Cache misses by cache type

### Queue Metrics
- `queue_messages_total` - Queue messages by queue name, status
- `queue_message_duration_seconds` - Message processing duration

### Notification Metrics
- `notifications_sent_total` - Notifications sent by type, status

### System Metrics
- `active_connections` - Active connections by type
- `memory_usage_bytes` - Memory usage by component
- `cpu_usage_percent` - CPU usage by component
- `response_size_bytes` - Response size by endpoint

## Performance Monitoring

The performance service automatically collects:
- CPU usage (process and system)
- Memory usage (process and system)
- Response times (with percentiles: p50, p95, p99)
- Error rates
- Requests per second
- Network I/O statistics
- Disk usage

## Log Aggregation

The logs service provides:
- Filtering by log level (INFO, WARNING, ERROR)
- Time-based filtering
- Full-text search
- Statistics and breakdowns
- Tail functionality for recent logs

## Usage in Code

### Recording Metrics

```python
from services.monitoring.metrics import metrics_service

# Record HTTP request
metrics_service.record_http_request("GET", "/api/users", 200, 0.15)

# Record application request
metrics_service.record_app_request("users", "get_user")

# Record database query
metrics_service.record_db_query("SELECT", "users", 0.05)

# Record error
metrics_service.record_app_error("users", "NotFoundError")
```

### Recording Performance

```python
from services.monitoring.performance import performance_service

# Record response time
performance_service.record_response_time("/api/users", 0.15)

# Record error
performance_service.record_error("/api/users", "TimeoutError")
```

## External Services

The monitoring service integrates with:
- **Prometheus** (port 9090) - Metrics collection
- **Grafana** (port 3001) - Visualization
- **Alertmanager** (port 9093) - Alert handling
- **Node Exporter** (port 9100) - System metrics
- **cAdvisor** (port 8082) - Container metrics
