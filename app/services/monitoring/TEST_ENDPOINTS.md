# Monitoring Service - Test Endpoints Guide

All endpoints are available at `http://localhost:8000/api/monitoring`

## Quick Test Commands

### 1. Service Information
```bash
curl http://localhost:8000/api/monitoring/
```

### 2. Health Check
```bash
curl http://localhost:8000/api/monitoring/health
```

## Metrics Endpoints

### Get Prometheus Metrics (for Prometheus scraping)
```bash
curl http://localhost:8000/api/monitoring/metrics
```

### Get Metrics as JSON
```bash
curl http://localhost:8000/api/monitoring/metrics/json
```

### Generate Test Metrics
```bash
curl -X POST http://localhost:8000/api/monitoring/metrics/test
```

## Performance Endpoints

### Get Current Performance Metrics
```bash
curl http://localhost:8000/api/monitoring/performance
```

### Get Response Time Statistics
```bash
# All endpoints
curl http://localhost:8000/api/monitoring/performance/response-times

# Specific endpoint
curl "http://localhost:8000/api/monitoring/performance/response-times?endpoint=/api/users"

# Last 10 minutes
curl "http://localhost:8000/api/monitoring/performance/response-times?minutes=10"
```

### Get Error Statistics
```bash
curl http://localhost:8000/api/monitoring/performance/errors

# Last 30 minutes
curl "http://localhost:8000/api/monitoring/performance/errors?minutes=30"
```

### Get Performance History
```bash
# Response times
curl "http://localhost:8000/api/monitoring/performance/history?metric_type=response_time&limit=50"

# Errors
curl "http://localhost:8000/api/monitoring/performance/history?metric_type=errors&limit=100"

# Requests
curl "http://localhost:8000/api/monitoring/performance/history?metric_type=requests&limit=200"
```

### Generate Test Performance Data
```bash
curl -X POST http://localhost:8000/api/monitoring/performance/test
```

## Logs Endpoints

### Query Logs
```bash
# All logs (last 100)
curl http://localhost:8000/api/monitoring/logs

# Filter by level
curl "http://localhost:8000/api/monitoring/logs?level=ERROR&limit=50"

# Search in logs
curl "http://localhost:8000/api/monitoring/logs?search=error&limit=20"

# Time range
curl "http://localhost:8000/api/monitoring/logs?start_time=2024-01-01T00:00:00&end_time=2024-01-02T00:00:00"

# Combined filters
curl "http://localhost:8000/api/monitoring/logs?level=ERROR&search=timeout&limit=10&offset=0"
```

### Get Log Statistics
```bash
# Last 24 hours (default)
curl http://localhost:8000/api/monitoring/logs/stats

# Last 7 days
curl "http://localhost:8000/api/monitoring/logs/stats?hours=168"
```

### Get Recent Errors
```bash
curl http://localhost:8000/api/monitoring/logs/errors

# More errors
curl "http://localhost:8000/api/monitoring/logs/errors?limit=100"
```

### Get Recent Warnings
```bash
curl http://localhost:8000/api/monitoring/logs/warnings
```

### Search Logs
```bash
curl "http://localhost:8000/api/monitoring/logs/search?q=timeout"
```

### Tail Logs (Last N Lines)
```bash
# Last 50 lines (default)
curl http://localhost:8000/api/monitoring/logs/tail

# Last 100 lines
curl "http://localhost:8000/api/monitoring/logs/tail?lines=100"
```

### Get Log File Info
```bash
curl http://localhost:8000/api/monitoring/logs/info
```

### Get Raw Log File Content
```bash
# Get entire log file as text
curl http://localhost:8000/api/monitoring/logs/file

# Get last 100 lines as text
curl "http://localhost:8000/api/monitoring/logs/file?limit_lines=100"

# Get log file as JSON (with metadata)
curl "http://localhost:8000/api/monitoring/logs/file?format=json"

# Get last 50 lines as JSON
curl "http://localhost:8000/api/monitoring/logs/file?limit_lines=50&format=json"
```

### Download Log File
```bash
# Download the entire log file
curl -O http://localhost:8000/api/monitoring/logs/file/download

# Or with a specific filename
curl -o my_logs.txt http://localhost:8000/api/monitoring/logs/file/download
```

## Testing Workflow

1. **Start the application**
   ```bash
   python app/app.py
   ```

2. **Generate test data**
   ```bash
   curl -X POST http://localhost:8000/api/monitoring/metrics/test
   curl -X POST http://localhost:8000/api/monitoring/performance/test
   ```

3. **Check metrics**
   ```bash
   curl http://localhost:8000/api/monitoring/metrics/json
   ```

4. **Check performance**
   ```bash
   curl http://localhost:8000/api/monitoring/performance
   ```

5. **Check logs**
   ```bash
   curl http://localhost:8000/api/monitoring/logs?limit=10
   ```

6. **Verify Prometheus can scrape**
   ```bash
   curl http://localhost:8000/api/monitoring/metrics
   ```

## Integration with Prometheus

Add to your `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'llm-ai-lab'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/api/monitoring/metrics'
    scrape_interval: 15s
```

Then restart Prometheus and verify in Prometheus UI:
- Go to http://localhost:9090
- Check Status > Targets
- Query metrics like: `http_requests_total`

## Browser Testing

You can also test endpoints in your browser:
- http://localhost:8000/api/monitoring/
- http://localhost:8000/api/monitoring/health
- http://localhost:8000/api/monitoring/performance
- http://localhost:8000/api/monitoring/metrics/json
- http://localhost:8000/api/monitoring/logs/stats
- http://localhost:8000/api/monitoring/logs/file (view raw logs in browser)
- http://localhost:8000/api/monitoring/logs/file?format=json (view logs as JSON)
- http://localhost:8000/api/monitoring/logs/file/download (download log file)

## Expected Responses

### Health Check
```json
{
  "status": "healthy",
  "timestamp": "2024-01-28T...",
  "services": {
    "metrics": "available",
    "performance": "available",
    "logs": "available"
  },
  "uptime_seconds": 123.45
}
```

### Performance Metrics
```json
{
  "timestamp": "2024-01-28T...",
  "uptime_seconds": 123.45,
  "process": {
    "cpu_percent": 2.5,
    "memory_mb": 150.25,
    "memory_percent": 15.2,
    "status": "healthy"
  },
  "system": {
    "cpu_percent": 10.5,
    "memory_percent": 45.2
  },
  "application": {
    "avg_response_time_seconds": 0.125,
    "error_rate": 0.001,
    "requests_per_second": 5.2
  }
}
```
