"""
Custom Exception Classes for Better Error Handling
"""
from typing import Optional, Dict, Any


class VaLLMException(Exception):
    """Base exception for VaLLM"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class VectorStoreError(VaLLMException):
    """Vector store related errors"""
    pass


class ModelNotLoadedError(VaLLMException):
    """Model not loaded error"""
    pass


class InvalidQueryError(VaLLMException):
    """Invalid query error"""
    pass


class ServiceUnavailableError(VaLLMException):
    """Service unavailable error"""
    pass


class RateLimitExceededError(VaLLMException):
    """Rate limit exceeded error"""
    pass
