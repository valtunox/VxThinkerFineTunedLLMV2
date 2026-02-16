"""
Log Aggregation and Query Service

This module provides comprehensive log management capabilities including:
- Querying and filtering logs by level, time range, and search terms
- Statistical analysis of log data
- Log file information and content retrieval
- Support for pagination and tail operations

The service parses log files in the standard format:
    YYYY-MM-DD HH:MM:SS,mmm LEVEL message

Example:
    logs_service = LogsService()
    recent_errors = logs_service.get_recent_errors(limit=50)
    stats = logs_service.get_log_stats(hours=24)
"""

import os
import re
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pathlib import Path

from core.logger import logger


class LogsService:
    """
    Service for log aggregation, querying, and analysis.
    
    This service provides methods to:
    - Query logs with various filters (level, time range, search)
    - Get statistical information about logs
    - Retrieve log file metadata and content
    - Support pagination for large log files
    
    Attributes:
        log_directory (Path): Directory containing log files
        log_file (Path): Path to the main log file
        log_pattern (re.Pattern): Compiled regex pattern for parsing log entries
    
    Example:
        >>> service = LogsService(log_directory="app/logs", log_file="logs.txt")
        >>> logs = service.get_logs(level="ERROR", limit=10)
        >>> print(f"Found {logs['total']} error logs")
    """
    
    def __init__(self, log_directory: str = "app/logs", log_file: str = "logs.txt"):
        """
        Initialize the LogsService.
        
        Args:
            log_directory (str): Directory path where log files are stored.
                                Defaults to "app/logs".
            log_file (str): Name of the log file to read. Defaults to "logs.txt".
        
        Note:
            The log file path will be constructed as: log_directory/log_file
        """
        self.log_directory = Path(log_directory)
        self.log_file = self.log_directory / log_file
        self.log_pattern = re.compile(
            r'(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) '
            r'(?P<level>\w+) '
            r'(?P<message>.*)'
        )
    
    def get_logs(
        self, 
        level: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        search: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Query logs with various filters and pagination support.
        
        This method reads the log file, parses entries, applies filters,
        and returns paginated results sorted by timestamp (newest first).
        
        Args:
            level (Optional[str]): Filter by log level (e.g., "INFO", "WARNING", "ERROR").
                                  Case-insensitive matching.
            start_time (Optional[datetime]): Only include logs after this timestamp.
            end_time (Optional[datetime]): Only include logs before this timestamp.
            search (Optional[str]): Search term to match in log messages. Case-insensitive.
            limit (int): Maximum number of logs to return. Defaults to 100.
            offset (int): Number of logs to skip for pagination. Defaults to 0.
        
        Returns:
            Dict[str, Any]: Dictionary containing:
                - logs (List[Dict]): List of log entries with timestamp, level, message, raw
                - total (int): Total number of matching logs
                - limit (int): Requested limit
                - offset (int): Requested offset
                - has_more (bool): Whether more logs are available
                - error (str): Error message if an error occurred
        
        Example:
            >>> from datetime import datetime, timedelta
            >>> service = LogsService()
            >>> one_hour_ago = datetime.now() - timedelta(hours=1)
            >>> logs = service.get_logs(
            ...     level="ERROR",
            ...     start_time=one_hour_ago,
            ...     search="database",
            ...     limit=50
            ... )
            >>> print(f"Found {logs['total']} matching logs")
        """
        if not self.log_file.exists():
            return {
                "logs": [],
                "total": 0,
                "limit": limit,
                "offset": offset
            }
        
        try:
            logs = []
            with open(self.log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    match = self.log_pattern.match(line.strip())
                    if match:
                        log_entry = match.groupdict()
                        log_entry['raw'] = line.strip()
                        
                        # Parse timestamp
                        try:
                            log_entry['timestamp'] = datetime.strptime(
                                log_entry['timestamp'], 
                                '%Y-%m-%d %H:%M:%S,%f'
                            )
                        except:
                            continue
                        
                        # Apply filters
                        if level and log_entry['level'].upper() != level.upper():
                            continue
                        
                        if start_time and log_entry['timestamp'] < start_time:
                            continue
                        
                        if end_time and log_entry['timestamp'] > end_time:
                            continue
                        
                        if search and search.lower() not in log_entry['message'].lower():
                            continue
                        
                        logs.append(log_entry)
            
            # Sort by timestamp (newest first)
            logs.sort(key=lambda x: x['timestamp'], reverse=True)
            
            # Apply pagination
            total = len(logs)
            paginated_logs = logs[offset:offset + limit]
            
            # Convert timestamps to ISO format
            for log in paginated_logs:
                log['timestamp'] = log['timestamp'].isoformat()
            
            return {
                "logs": paginated_logs,
                "total": total,
                "limit": limit,
                "offset": offset,
                "has_more": offset + limit < total
            }
        except Exception as e:
            logger.error(f"Error reading logs: {e}")
            return {
                "error": str(e),
                "logs": [],
                "total": 0
            }
    
    def get_log_stats(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get statistical information about logs for a specified time period.
        
        Calculates total log count, breakdown by log level, and hourly distribution
        for the specified number of hours from the current time.
        
        Args:
            hours (int): Number of hours to look back from current time.
                        Defaults to 24 hours.
        
        Returns:
            Dict[str, Any]: Dictionary containing:
                - period_hours (int): Time period analyzed
                - total_logs (int): Total number of logs in the period
                - by_level (Dict[str, int]): Count of logs grouped by level
                - by_hour (Dict[str, int]): Count of logs grouped by hour
                - start_time (str): ISO format start timestamp
                - end_time (str): ISO format end timestamp
                - error (str): Error message if an error occurred
        
        Example:
            >>> service = LogsService()
            >>> stats = service.get_log_stats(hours=48)
            >>> print(f"Total logs: {stats['total_logs']}")
            >>> print(f"Errors: {stats['by_level'].get('ERROR', 0)}")
        """
        if not self.log_file.exists():
            return {
                "total_logs": 0,
                "by_level": {},
                "by_hour": {}
            }
        
        try:
            start_time = datetime.now() - timedelta(hours=hours)
            level_counts = {}
            hour_counts = {}
            total = 0
            
            with open(self.log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    match = self.log_pattern.match(line.strip())
                    if match:
                        log_entry = match.groupdict()
                        try:
                            timestamp = datetime.strptime(
                                log_entry['timestamp'], 
                                '%Y-%m-%d %H:%M:%S,%f'
                            )
                        except:
                            continue
                        
                        if timestamp < start_time:
                            continue
                        
                        total += 1
                        level = log_entry['level'].upper()
                        level_counts[level] = level_counts.get(level, 0) + 1
                        
                        hour_key = timestamp.strftime('%Y-%m-%d %H:00')
                        hour_counts[hour_key] = hour_counts.get(hour_key, 0) + 1
            
            return {
                "period_hours": hours,
                "total_logs": total,
                "by_level": level_counts,
                "by_hour": hour_counts,
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting log stats: {e}")
            return {
                "error": str(e),
                "total_logs": 0
            }
    
    def get_recent_errors(self, limit: int = 50) -> Dict[str, Any]:
        """
        Get the most recent error-level logs.
        
        Convenience method that filters logs by ERROR level and returns
        the most recent entries sorted by timestamp (newest first).
        
        Args:
            limit (int): Maximum number of error logs to return. Defaults to 50.
        
        Returns:
            Dict[str, Any]: Dictionary with same structure as get_logs(),
                           containing only ERROR level logs.
        
        Example:
            >>> service = LogsService()
            >>> errors = service.get_recent_errors(limit=20)
            >>> for error in errors['logs']:
            ...     print(f"{error['timestamp']}: {error['message']}")
        """
        return self.get_logs(
            level="ERROR",
            limit=limit
        )
    
    def get_recent_warnings(self, limit: int = 50) -> Dict[str, Any]:
        """
        Get the most recent warning-level logs.
        
        Convenience method that filters logs by WARNING level and returns
        the most recent entries sorted by timestamp (newest first).
        
        Args:
            limit (int): Maximum number of warning logs to return. Defaults to 50.
        
        Returns:
            Dict[str, Any]: Dictionary with same structure as get_logs(),
                           containing only WARNING level logs.
        
        Example:
            >>> service = LogsService()
            >>> warnings = service.get_recent_warnings(limit=30)
            >>> print(f"Found {warnings['total']} recent warnings")
        """
        return self.get_logs(
            level="WARNING",
            limit=limit
        )
    
    def search_logs(self, query: str, limit: int = 100) -> Dict[str, Any]:
        """
        Search logs by query string in log messages.
        
        Performs a case-insensitive search across all log messages
        and returns matching entries sorted by timestamp (newest first).
        
        Args:
            query (str): Search term to find in log messages. Required.
            limit (int): Maximum number of matching logs to return. Defaults to 100.
        
        Returns:
            Dict[str, Any]: Dictionary with same structure as get_logs(),
                           containing only logs matching the search query.
        
        Example:
            >>> service = LogsService()
            >>> results = service.search_logs("database connection", limit=50)
            >>> print(f"Found {results['total']} logs containing 'database connection'")
        """
        return self.get_logs(
            search=query,
            limit=limit
        )
    
    def get_log_file_info(self) -> Dict[str, Any]:
        """
        Get metadata information about the log file.
        
        Retrieves file system information including file size, modification time,
        creation time, and file path.
        
        Returns:
            Dict[str, Any]: Dictionary containing:
                - exists (bool): Whether the log file exists
                - path (str): Full path to the log file
                - size_bytes (int): File size in bytes (if exists)
                - size_mb (float): File size in megabytes (if exists)
                - modified (str): Last modification time in ISO format (if exists)
                - created (str): Creation time in ISO format (if exists)
                - error (str): Error message if file exists but info cannot be read
        
        Example:
            >>> service = LogsService()
            >>> info = service.get_log_file_info()
            >>> if info['exists']:
            ...     print(f"Log file size: {info['size_mb']} MB")
            ...     print(f"Last modified: {info['modified']}")
        """
        if not self.log_file.exists():
            return {
                "exists": False,
                "path": str(self.log_file)
            }
        
        try:
            stat = self.log_file.stat()
            return {
                "exists": True,
                "path": str(self.log_file),
                "size_bytes": stat.st_size,
                "size_mb": round(stat.st_size / 1024 / 1024, 2),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat()
            }
        except Exception as e:
            return {
                "exists": True,
                "path": str(self.log_file),
                "error": str(e)
            }
    
    def tail_logs(self, lines: int = 50) -> Dict[str, Any]:
        """
        Get the last N lines from the log file.
        
        Efficiently retrieves the most recent log entries by reading
        only the last N lines of the file. Useful for monitoring
        real-time log activity.
        
        Args:
            lines (int): Number of lines to retrieve from the end of the file.
                        Defaults to 50.
        
        Returns:
            Dict[str, Any]: Dictionary containing:
                - logs (List[Dict]): List of parsed log entries
                - total (int): Number of logs returned
                - requested_lines (int): Number of lines requested
                - error (str): Error message if an error occurred
        
        Example:
            >>> service = LogsService()
            >>> recent = service.tail_logs(lines=100)
            >>> print(f"Retrieved {recent['total']} recent log entries")
        """
        if not self.log_file.exists():
            return {
                "logs": [],
                "total": 0
            }
        
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                all_lines = f.readlines()
                tail_lines = all_lines[-lines:]
            
            logs = []
            for line in tail_lines:
                match = self.log_pattern.match(line.strip())
                if match:
                    log_entry = match.groupdict()
                    log_entry['raw'] = line.strip()
                    try:
                        log_entry['timestamp'] = datetime.strptime(
                            log_entry['timestamp'], 
                            '%Y-%m-%d %H:%M:%S,%f'
                        ).isoformat()
                    except:
                        log_entry['timestamp'] = None
                    logs.append(log_entry)
            
            return {
                "logs": logs,
                "total": len(logs),
                "requested_lines": lines
            }
        except Exception as e:
            logger.error(f"Error tailing logs: {e}")
            return {
                "error": str(e),
                "logs": [],
                "total": 0
            }
    
    def get_log_file_content(self, limit_lines: Optional[int] = None) -> str:
        """
        Get raw log file content as plain text.
        
        Reads the entire log file or the last N lines and returns
        the content as a string. Useful for downloading or displaying
        raw log content.
        
        Args:
            limit_lines (Optional[int]): If specified, only return the last N lines.
                                        If None, returns the entire file content.
        
        Returns:
            str: Raw log file content as text. Returns empty string if file doesn't exist.
                 Returns error message string if reading fails.
        
        Example:
            >>> service = LogsService()
            >>> # Get last 1000 lines
            >>> content = service.get_log_file_content(limit_lines=1000)
            >>> # Get entire file
            >>> full_content = service.get_log_file_content()
        """
        if not self.log_file.exists():
            return ""
        
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                if limit_lines:
                    # Read only last N lines
                    all_lines = f.readlines()
                    lines = all_lines[-limit_lines:] if len(all_lines) > limit_lines else all_lines
                    return ''.join(lines)
                else:
                    # Read entire file
                    return f.read()
        except Exception as e:
            logger.error(f"Error reading log file content: {e}")
            return f"Error reading log file: {str(e)}"
    
    def get_log_file_lines_count(self) -> int:
        """
        Get the total number of lines in the log file.
        
        Efficiently counts all lines in the log file without loading
        the entire file into memory.
        
        Returns:
            int: Total number of lines in the log file. Returns 0 if file
                 doesn't exist or if an error occurs.
        
        Example:
            >>> service = LogsService()
            >>> line_count = service.get_log_file_lines_count()
            >>> print(f"Log file contains {line_count} lines")
        """
        if not self.log_file.exists():
            return 0
        
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                return sum(1 for _ in f)
        except Exception as e:
            logger.error(f"Error counting log file lines: {e}")
            return 0
    
    def get_log_file_bytes(self) -> bytes:
        """
        Get log file content as bytes for file download operations.
        
        Reads the entire log file in binary mode and returns the content
        as bytes. Useful for file download endpoints.
        
        Returns:
            bytes: Raw log file content as bytes. Returns empty bytes if file
                   doesn't exist. Returns error message encoded as UTF-8 if
                   reading fails.
        
        Example:
            >>> service = LogsService()
            >>> log_bytes = service.get_log_file_bytes()
            >>> # Use in file download response
            >>> response = Response(content=log_bytes, media_type="text/plain")
        """
        if not self.log_file.exists():
            return b""
        
        try:
            with open(self.log_file, 'rb') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading log file bytes: {e}")
            return f"Error reading log file: {str(e)}".encode('utf-8')


# Global logs service instance
logs_service = LogsService()
