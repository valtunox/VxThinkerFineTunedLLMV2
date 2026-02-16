"""
Performance Monitoring Service

This module provides comprehensive performance monitoring capabilities including:
- Real-time CPU and memory usage tracking
- Response time monitoring and statistics
- Error rate calculation
- System resource monitoring (CPU, memory, disk, network)
- Historical performance data collection
- Performance threshold alerts

The service can run in background mode to continuously collect metrics,
or provide on-demand performance snapshots. It uses psutil for system metrics
and gracefully handles cases where psutil is not installed.

Example:
    from services.monitoring.performance import performance_service
    
    # Start background monitoring
    performance_service.start_monitoring(interval=5)
    
    # Record a response time
    performance_service.record_response_time("/api/users", 0.15)
    
    # Get current metrics
    metrics = performance_service.get_current_metrics()
"""

import time
import threading
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import deque
import os

# Try to import psutil, handle gracefully if not available
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    # Create dummy psutil module
    class DummyPsutil:
        class Process:
            def __init__(self, *args, **kwargs):
                pass
            def cpu_percent(self, *args, **kwargs):
                return 0.0
            def memory_info(self):
                class MemoryInfo:
                    rss = 0
                return MemoryInfo()
            def memory_percent(self):
                return 0.0
            def num_threads(self):
                return 0
            def num_fds(self):
                return 0
        def Process(self, *args, **kwargs):
            return self.Process()
        def cpu_percent(self, *args, **kwargs):
            return 0.0
        def virtual_memory(self):
            class VirtualMemory:
                percent = 0.0
                available = 0
            return VirtualMemory()
        def disk_usage(self, *args, **kwargs):
            class DiskUsage:
                percent = 0.0
                free = 0
            return DiskUsage()
        def net_io_counters(self):
            class NetIO:
                bytes_sent = 0
                bytes_recv = 0
                packets_sent = 0
                packets_recv = 0
            return NetIO()
    psutil = DummyPsutil()
    
    from core.logger import logger
    logger.warning("psutil not installed. Performance monitoring will be limited. Install with: pip install psutil")

from core.logger import logger


class PerformanceService:
    """
    Service for monitoring application and system performance.
    
    This service tracks various performance metrics including:
    - Process and system CPU/memory usage
    - Response times and latency statistics
    - Error rates and counts
    - Network I/O statistics
    - Disk usage
    
    It maintains a rolling history of metrics and can run background
    monitoring to continuously collect system metrics.
    
    Attributes:
        history_size (int): Maximum number of historical records to keep
        response_times (deque): Rolling history of response times
        error_counts (deque): Rolling history of errors
        request_counts (deque): Rolling history of request metrics
        monitoring_active (bool): Whether background monitoring is running
        monitor_thread (Thread): Background monitoring thread
        start_time (float): Service start timestamp
        cpu_threshold (float): CPU usage alert threshold (percentage)
        memory_threshold (float): Memory usage alert threshold (percentage)
        response_time_threshold (float): Response time alert threshold (seconds)
    
    Example:
        >>> service = PerformanceService(history_size=500)
        >>> service.start_monitoring(interval=5)
        >>> service.record_response_time("/api/users", 0.15)
        >>> metrics = service.get_current_metrics()
    """
    
    def __init__(self, history_size: int = 1000):
        """
        Initialize the PerformanceService.
        
        Args:
            history_size (int): Maximum number of historical records to keep
                              in rolling buffers. Defaults to 1000.
        """
        self.history_size = history_size
        self.response_times = deque(maxlen=history_size)
        self.error_counts = deque(maxlen=history_size)
        self.request_counts = deque(maxlen=history_size)
        self.monitoring_active = False
        self.monitor_thread = None
        self.start_time = time.time()
        
        # Performance thresholds
        self.cpu_threshold = 80.0  # %
        self.memory_threshold = 80.0  # %
        self.response_time_threshold = 1.0  # seconds
    
    def start_monitoring(self, interval: int = 5):
        """
        Start background performance monitoring in a separate thread.
        
        Begins collecting system and process metrics at regular intervals.
        The monitoring runs in a daemon thread and will stop when the
        main process exits.
        
        Args:
            interval (int): Number of seconds between metric collection cycles.
                          Defaults to 5 seconds.
        
        Note:
            If monitoring is already active, this method does nothing.
            Use stop_monitoring() first if you need to change the interval.
        
        Example:
            >>> service = PerformanceService()
            >>> service.start_monitoring(interval=10)  # Collect every 10 seconds
        """
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        
        def monitor_loop():
            while self.monitoring_active:
                try:
                    self._collect_metrics()
                    time.sleep(interval)
                except Exception as e:
                    logger.error(f"Error in performance monitoring: {e}")
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """
        Stop background performance monitoring.
        
        Stops the monitoring thread and waits up to 5 seconds for it
        to finish gracefully.
        
        Example:
            >>> service = PerformanceService()
            >>> service.start_monitoring()
            >>> # ... do work ...
            >>> service.stop_monitoring()
        """
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Performance monitoring stopped")
    
    def _collect_metrics(self):
        """
        Collect current performance metrics and store in history.
        
        This is an internal method called by the background monitoring thread.
        It collects process and system metrics and stores them in the
        request_counts history buffer.
        
        Note:
            This method is called automatically by the monitoring thread.
            You typically don't need to call it directly.
        """
        try:
            # Get process metrics
            process = psutil.Process(os.getpid())
            
            # CPU usage
            cpu_percent = process.cpu_percent(interval=0.1)
            
            # Memory usage
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()
            
            # System-wide metrics
            system_cpu = psutil.cpu_percent(interval=0.1)
            system_memory = psutil.virtual_memory()
            
            # Store metrics (could be sent to metrics service)
            timestamp = time.time()
            self.request_counts.append({
                'timestamp': timestamp,
                'cpu': cpu_percent,
                'memory': memory_percent,
                'system_cpu': system_cpu,
                'system_memory': system_memory.percent
            })
        except Exception as e:
            logger.error(f"Error collecting performance metrics: {e}")
    
    def record_response_time(self, endpoint: str, duration: float):
        """
        Record response time for an API endpoint.
        
        Stores the response time in the rolling history for later
        statistical analysis and threshold checking.
        
        Args:
            endpoint (str): API endpoint path (e.g., "/api/users", "/health").
            duration (float): Response time in seconds.
        
        Example:
            >>> service = PerformanceService()
            >>> service.record_response_time("/api/users", 0.15)
            >>> service.record_response_time("/api/data", 0.45)
        """
        self.response_times.append({
            'timestamp': time.time(),
            'endpoint': endpoint,
            'duration': duration
        })
    
    def record_error(self, endpoint: str, error_type: str):
        """
        Record an error occurrence.
        
        Stores error information in the rolling history for later
        statistical analysis and error rate calculation.
        
        Args:
            endpoint (str): API endpoint path where the error occurred.
            error_type (str): Type of error (e.g., "TimeoutError", "ValueError",
                            "DatabaseError").
        
        Example:
            >>> service = PerformanceService()
            >>> service.record_error("/api/users", "TimeoutError")
            >>> service.record_error("/api/data", "DatabaseError")
        """
        self.error_counts.append({
            'timestamp': time.time(),
            'endpoint': endpoint,
            'error_type': error_type
        })
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive current performance metrics.
        
        Collects real-time metrics including:
        - Process metrics (CPU, memory, threads, file descriptors)
        - System metrics (CPU, memory, disk usage)
        - Network I/O statistics
        - Application metrics (avg response time, error rate, requests/sec)
        - Performance alerts based on thresholds
        
        Returns:
            Dict[str, Any]: Dictionary containing:
                - timestamp (str): ISO format timestamp
                - uptime_seconds (float): Service uptime
                - process (Dict): Process-level metrics
                - system (Dict): System-level metrics
                - network (Dict): Network I/O statistics
                - application (Dict): Application performance metrics
                - alerts (List[str]): List of threshold alerts
                - thresholds (Dict): Current threshold values
                - error (str): Error message if collection fails
        
        Example:
            >>> service = PerformanceService()
            >>> metrics = service.get_current_metrics()
            >>> print(f"CPU: {metrics['process']['cpu_percent']}%")
            >>> print(f"Memory: {metrics['process']['memory_mb']} MB")
            >>> if metrics['alerts']:
            ...     print(f"Alerts: {metrics['alerts']}")
        """
        try:
            process = psutil.Process(os.getpid())
            
            # Process metrics
            cpu_percent = process.cpu_percent(interval=0.1)
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()
            num_threads = process.num_threads()
            num_fds = process.num_fds() if hasattr(process, 'num_fds') else 0
            
            # System metrics
            system_cpu = psutil.cpu_percent(interval=0.1)
            system_memory = psutil.virtual_memory()
            system_disk = psutil.disk_usage('/')
            
            # Network stats
            network_io = psutil.net_io_counters()
            
            # Calculate averages from history
            avg_response_time = self._calculate_avg_response_time()
            error_rate = self._calculate_error_rate()
            requests_per_second = self._calculate_requests_per_second()
            
            # Check thresholds
            alerts = []
            if cpu_percent > self.cpu_threshold:
                alerts.append(f"High CPU usage: {cpu_percent:.2f}%")
            if memory_percent > self.memory_threshold:
                alerts.append(f"High memory usage: {memory_percent:.2f}%")
            if avg_response_time > self.response_time_threshold:
                alerts.append(f"High response time: {avg_response_time:.3f}s")
            
            uptime = time.time() - self.start_time
            
            return {
                "timestamp": datetime.now().isoformat(),
                "uptime_seconds": uptime,
                "process": {
                    "cpu_percent": round(cpu_percent, 2),
                    "memory_mb": round(memory_info.rss / 1024 / 1024, 2),
                    "memory_percent": round(memory_percent, 2),
                    "threads": num_threads,
                    "file_descriptors": num_fds,
                    "status": "healthy" if cpu_percent < self.cpu_threshold and memory_percent < self.memory_threshold else "warning"
                },
                "system": {
                    "cpu_percent": round(system_cpu, 2),
                    "memory_percent": round(system_memory.percent, 2),
                    "memory_available_mb": round(system_memory.available / 1024 / 1024, 2),
                    "disk_percent": round(system_disk.percent, 2),
                    "disk_free_gb": round(system_disk.free / 1024 / 1024 / 1024, 2)
                },
                "network": {
                    "bytes_sent": network_io.bytes_sent,
                    "bytes_recv": network_io.bytes_recv,
                    "packets_sent": network_io.packets_sent,
                    "packets_recv": network_io.packets_recv
                },
                "application": {
                    "avg_response_time_seconds": round(avg_response_time, 3),
                    "error_rate": round(error_rate, 4),
                    "requests_per_second": round(requests_per_second, 2),
                    "total_requests": len(self.response_times),
                    "total_errors": len(self.error_counts)
                },
                "alerts": alerts,
                "thresholds": {
                    "cpu_percent": self.cpu_threshold,
                    "memory_percent": self.memory_threshold,
                    "response_time_seconds": self.response_time_threshold
                }
            }
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_response_time_stats(self, endpoint: Optional[str] = None, minutes: int = 5) -> Dict[str, Any]:
        """
        Get response time statistics for a time period.
        
        Calculates statistical measures (min, max, avg, percentiles) for
        response times within the specified time window. Can filter by
        endpoint or aggregate across all endpoints.
        
        Args:
            endpoint (Optional[str]): Filter by specific endpoint. If None,
                                     aggregates across all endpoints.
            minutes (int): Time window in minutes to analyze. Defaults to 5.
        
        Returns:
            Dict[str, Any]: Dictionary containing:
                - endpoint (str): Endpoint name or "all"
                - period_minutes (int): Time period analyzed
                - count (int): Number of response time records
                - stats (Dict): Statistical measures (min, max, avg, p50, p95, p99)
        
        Example:
            >>> service = PerformanceService()
            >>> # Get stats for specific endpoint
            >>> stats = service.get_response_time_stats("/api/users", minutes=10)
            >>> print(f"Average: {stats['stats']['avg']}s")
            >>> print(f"P95: {stats['stats']['p95']}s")
            >>> # Get stats for all endpoints
            >>> all_stats = service.get_response_time_stats(minutes=30)
        """
        cutoff_time = time.time() - (minutes * 60)
        
        if endpoint:
            times = [r['duration'] for r in self.response_times 
                    if r['endpoint'] == endpoint and r['timestamp'] > cutoff_time]
        else:
            times = [r['duration'] for r in self.response_times 
                    if r['timestamp'] > cutoff_time]
        
        if not times:
            return {
                "endpoint": endpoint or "all",
                "period_minutes": minutes,
                "count": 0,
                "stats": {}
            }
        
        return {
            "endpoint": endpoint or "all",
            "period_minutes": minutes,
            "count": len(times),
            "stats": {
                "min": round(min(times), 3),
                "max": round(max(times), 3),
                "avg": round(sum(times) / len(times), 3),
                "p50": round(sorted(times)[len(times) // 2], 3),
                "p95": round(sorted(times)[int(len(times) * 0.95)], 3),
                "p99": round(sorted(times)[int(len(times) * 0.99)], 3)
            }
        }
    
    def get_error_stats(self, minutes: int = 5) -> Dict[str, Any]:
        """
        Get error statistics for a time period.
        
        Analyzes errors within the specified time window and provides
        breakdown by error type and overall error rate.
        
        Args:
            minutes (int): Time window in minutes to analyze. Defaults to 5.
        
        Returns:
            Dict[str, Any]: Dictionary containing:
                - period_minutes (int): Time period analyzed
                - total_errors (int): Total number of errors
                - error_breakdown (Dict[str, int]): Count by error type
                - error_rate (float): Error rate (errors / requests)
        
        Example:
            >>> service = PerformanceService()
            >>> error_stats = service.get_error_stats(minutes=60)
            >>> print(f"Total errors: {error_stats['total_errors']}")
            >>> print(f"Error rate: {error_stats['error_rate']:.2%}")
            >>> for error_type, count in error_stats['error_breakdown'].items():
            ...     print(f"{error_type}: {count}")
        """
        cutoff_time = time.time() - (minutes * 60)
        errors = [e for e in self.error_counts if e['timestamp'] > cutoff_time]
        
        error_counts = {}
        for error in errors:
            error_type = error['error_type']
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        return {
            "period_minutes": minutes,
            "total_errors": len(errors),
            "error_breakdown": error_counts,
            "error_rate": len(errors) / max(len(self.response_times), 1)
        }
    
    def _calculate_avg_response_time(self, minutes: int = 5) -> float:
        """
        Calculate average response time for a time period.
        
        Internal helper method that computes the mean response time
        from the response_times history within the specified window.
        
        Args:
            minutes (int): Time window in minutes. Defaults to 5.
        
        Returns:
            float: Average response time in seconds. Returns 0.0 if no data.
        """
        cutoff_time = time.time() - (minutes * 60)
        times = [r['duration'] for r in self.response_times if r['timestamp'] > cutoff_time]
        return sum(times) / len(times) if times else 0.0
    
    def _calculate_error_rate(self, minutes: int = 5) -> float:
        """
        Calculate error rate for a time period.
        
        Internal helper method that computes the ratio of errors to requests
        within the specified time window.
        
        Args:
            minutes (int): Time window in minutes. Defaults to 5.
        
        Returns:
            float: Error rate as a ratio (0.0 to 1.0). Returns 0.0 if no requests.
        """
        cutoff_time = time.time() - (minutes * 60)
        errors = len([e for e in self.error_counts if e['timestamp'] > cutoff_time])
        requests = len([r for r in self.response_times if r['timestamp'] > cutoff_time])
        return errors / max(requests, 1)
    
    def _calculate_requests_per_second(self, minutes: int = 5) -> float:
        """
        Calculate requests per second for a time period.
        
        Internal helper method that computes the average request rate
        from the response_times history within the specified window.
        
        Args:
            minutes (int): Time window in minutes. Defaults to 5.
        
        Returns:
            float: Requests per second. Returns 0.0 if no requests.
        """
        cutoff_time = time.time() - (minutes * 60)
        requests = len([r for r in self.response_times if r['timestamp'] > cutoff_time])
        period_seconds = minutes * 60
        return requests / max(period_seconds, 1)
    
    def get_history(self, metric_type: str = "response_time", limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get historical performance metrics.
        
        Retrieves the most recent records from the specified metric history.
        Timestamps are converted to ISO format for easy consumption.
        
        Args:
            metric_type (str): Type of metric to retrieve. Options:
                              - "response_time": Response time records
                              - "errors": Error records
                              - "requests": Request metric records
            limit (int): Maximum number of records to return. Defaults to 100.
        
        Returns:
            List[Dict[str, Any]]: List of metric records with ISO format timestamps.
                                 Returns empty list if metric_type is invalid.
        
        Example:
            >>> service = PerformanceService()
            >>> # Get last 50 response times
            >>> response_times = service.get_history("response_time", limit=50)
            >>> # Get last 20 errors
            >>> errors = service.get_history("errors", limit=20)
        """
        if metric_type == "response_time":
            data = list(self.response_times)[-limit:]
        elif metric_type == "errors":
            data = list(self.error_counts)[-limit:]
        elif metric_type == "requests":
            data = list(self.request_counts)[-limit:]
        else:
            return []
        
        return [
            {
                **item,
                'timestamp': datetime.fromtimestamp(item['timestamp']).isoformat()
            }
            for item in data
        ]


# Global performance service instance
performance_service = PerformanceService()
