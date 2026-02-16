"""
Rate Limiting Middleware (Optional)
Enable with: VALLM_RATE_LIMIT_ENABLED=true
Set limit with: VALLM_RATE_LIMIT_PER_MINUTE=60
"""
import os
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from collections import defaultdict
from time import time
from typing import Dict, Tuple

# Configuration
RATE_LIMIT_ENABLED = os.getenv("VALLM_RATE_LIMIT_ENABLED", "false").lower() == "true"
REQUESTS_PER_MINUTE = int(os.getenv("VALLM_RATE_LIMIT_PER_MINUTE", "60"))


class RateLimiter:
    """Simple in-memory rate limiter"""
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests: Dict[str, list] = defaultdict(list)
    
    def is_allowed(self, identifier: str) -> Tuple[bool, int]:
        """Check if request is allowed. Returns (allowed, remaining)"""
        now = time()
        minute_ago = now - 60
        
        # Clean old requests
        self.requests[identifier] = [
            req_time for req_time in self.requests[identifier]
            if req_time > minute_ago
        ]
        
        if len(self.requests[identifier]) >= self.requests_per_minute:
            return False, 0
        
        self.requests[identifier].append(now)
        remaining = self.requests_per_minute - len(self.requests[identifier])
        return True, remaining


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware (only active if enabled)"""
    
    def __init__(self, app, requests_per_minute: int = None):
        super().__init__(app)
        if requests_per_minute is None:
            requests_per_minute = REQUESTS_PER_MINUTE
        self.rate_limiter = RateLimiter(requests_per_minute) if RATE_LIMIT_ENABLED else None
    
    async def dispatch(self, request: Request, call_next):
        # Skip if rate limiting disabled
        if not RATE_LIMIT_ENABLED or not self.rate_limiter:
            return await call_next(request)
        
        # Get client identifier
        client_id = request.client.host if request.client else "unknown"
        
        allowed, remaining = self.rate_limiter.is_allowed(client_id)
        
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={"X-RateLimit-Remaining": "0"}
            )
        
        response = await call_next(request)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Limit"] = str(self.rate_limiter.requests_per_minute)
        
        return response
