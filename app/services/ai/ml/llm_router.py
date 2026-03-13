"""
VaLLM Specialist Model - LLM Router Service.

Author: Joel Otepa Wembo
https://joelwembo.com

Intelligent routing between local models (Ollama) and commercial APIs (Claude, OpenAI).
"""

from typing import Dict, Any, List, Optional
import logging
import asyncio
import aiohttp
import json
from datetime import datetime
from enum import Enum

from core.settings import get_settings
from core.logging import performance_logger

settings = get_settings()
logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Available model types"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


class LLMRouter:
    """Intelligent LLM routing service"""
    
    def __init__(self):
        self.openai_api_key = settings.openai_api_key
        self.anthropic_api_key = settings.anthropic_api_key
        self.ollama_base_url = settings.ollama_base_url
        
        # Model configurations
        self.model_configs = {
            ModelType.OPENAI: {
                "base_url": "https://api.openai.com/v1",
                "model": settings.openai_model,
                "max_tokens": 4000,
                "temperature": 0.7
            },
            ModelType.ANTHROPIC: {
                "base_url": "https://api.anthropic.com/v1",
                "model": settings.anthropic_model,
                "max_tokens": 4000,
                "temperature": 0.7
            },
            ModelType.OLLAMA: {
                "base_url": self.ollama_base_url,
                "model": settings.ollama_model,
                "max_tokens": 4000,
                "temperature": 0.7
            }
        }
        
        # Routing rules
        self.routing_rules = {
            "general": ModelType.OPENAI,
            "analysis": ModelType.ANTHROPIC,
            "conversational": ModelType.OPENAI,
            "code": ModelType.OLLAMA,
            "summarization": ModelType.ANTHROPIC,
            "creative": ModelType.OPENAI
        }
        
        # Circuit breakers
        self.circuit_breakers = {
            model_type: {"failures": 0, "last_failure": None, "is_open": False}
            for model_type in ModelType
        }
        
        # Performance tracking
        self.request_count = 0
        self.success_count = 0
        self.error_count = 0
        self.total_response_time = 0.0
        
        self.is_initialized = False
    
    async def initialize(self) -> bool:
        """Initialize LLM router"""
        try:
            # Test connections to available services
            await self._test_connections()
            self.is_initialized = True
            logger.info("LLM Router initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize LLM Router: {e}")
            return False
    
    async def _test_connections(self):
        """Test connections to all available LLM services"""
        for model_type in ModelType:
            try:
                if await self._test_connection(model_type):
                    logger.info(f"{model_type.value} connection successful")
                else:
                    logger.warning(f"{model_type.value} connection failed")
            except Exception as e:
                logger.error(f"{model_type.value} connection error: {e}")
    
    async def _test_connection(self, model_type: ModelType) -> bool:
        """Test connection to a specific model service"""
        try:
            test_prompt = "Hello, this is a connection test."
            
            if model_type == ModelType.OLLAMA:
                return await self._test_ollama_connection()
            elif model_type == ModelType.OPENAI:
                return await self._test_openai_connection()
            elif model_type == ModelType.ANTHROPIC:
                return await self._test_anthropic_connection()
            
            return False
        except Exception as e:
            logger.error(f"Connection test failed for {model_type.value}: {e}")
            return False
    
    async def _test_ollama_connection(self) -> bool:
        """Test Ollama connection"""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.ollama_base_url}/api/tags"
                async with session.get(url, timeout=5) as response:
                    return response.status == 200
        except:
            return False
    
    async def _test_openai_connection(self) -> bool:
        """Test OpenAI connection"""
        if not self.openai_api_key:
            return False
        
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.model_configs[ModelType.OPENAI]['base_url']}/models"
                headers = {"Authorization": f"Bearer {self.openai_api_key}"}
                async with session.get(url, headers=headers, timeout=5) as response:
                    return response.status == 200
        except:
            return False
    
    async def _test_anthropic_connection(self) -> bool:
        """Test Anthropic connection"""
        if not self.anthropic_api_key:
            return False
        
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.model_configs[ModelType.ANTHROPIC]['base_url']}/messages"
                headers = {
                    "x-api-key": self.anthropic_api_key,
                    "Content-Type": "application/json"
                }
                # Just test if we can reach the endpoint
                async with session.get(url, headers=headers, timeout=5) as response:
                    return response.status in [200, 404]  # 404 is fine for GET on messages endpoint
        except:
            return False
    
    async def get_response(self, prompt: str, model_type: str = "general", 
                          temperature: float = 0.7, max_tokens: int = 1000,
                          **kwargs) -> str:
        """Get response from appropriate LLM"""
        try:
            start_time = datetime.utcnow()
            self.request_count += 1
            
            # Determine which model to use
            target_model = self._determine_model(model_type)
            
            # Check circuit breaker
            if self._is_circuit_breaker_open(target_model):
                # Try fallback model
                target_model = self._get_fallback_model(target_model)
                if self._is_circuit_breaker_open(target_model):
                    raise Exception("All models are currently unavailable")
            
            # Get response
            response = await self._call_model(target_model, prompt, temperature, max_tokens, **kwargs)
            
            # Update success metrics
            self.success_count += 1
            response_time = (datetime.utcnow() - start_time).total_seconds()
            self.total_response_time += response_time
            
            # Reset circuit breaker on success
            self._reset_circuit_breaker(target_model)
            
            # Log performance
            performance_logger.log_ai_model_inference(
                model_name=f"{target_model.value}_llm",
                input_tokens=len(prompt.split()),
                output_tokens=len(response.split()),
                duration_ms=response_time * 1000
            )
            
            return response
            
        except Exception as e:
            self.error_count += 1
            self._record_circuit_breaker_failure(target_model)
            logger.error(f"LLM request failed: {e}")
            raise
    
    def _determine_model(self, model_type: str) -> ModelType:
        """Determine which model to use based on type and availability"""
        # Check routing rules
        if model_type in self.routing_rules:
            preferred_model = self.routing_rules[model_type]
            
            # Check if preferred model is available
            if not self._is_circuit_breaker_open(preferred_model):
                return preferred_model
        
        # Fallback to available models
        for model in ModelType:
            if not self._is_circuit_breaker_open(model):
                return model
        
        # Default fallback
        return ModelType.OPENAI
    
    def _get_fallback_model(self, failed_model: ModelType) -> ModelType:
        """Get fallback model when primary model fails"""
        fallback_order = [ModelType.OPENAI, ModelType.ANTHROPIC, ModelType.OLLAMA]
        
        for model in fallback_order:
            if model != failed_model and not self._is_circuit_breaker_open(model):
                return model
        
        return failed_model  # No fallback available
    
    def _is_circuit_breaker_open(self, model_type: ModelType) -> bool:
        """Check if circuit breaker is open for a model"""
        breaker = self.circuit_breakers[model_type]
        
        if not breaker["is_open"]:
            return False
        
        # Check if enough time has passed to retry
        if breaker["last_failure"]:
            time_since_failure = (datetime.utcnow() - breaker["last_failure"]).total_seconds()
            if time_since_failure > 60:  # 1 minute timeout
                breaker["is_open"] = False
                breaker["failures"] = 0
                return False
        
        return True
    
    def _record_circuit_breaker_failure(self, model_type: ModelType):
        """Record failure for circuit breaker"""
        breaker = self.circuit_breakers[model_type]
        breaker["failures"] += 1
        breaker["last_failure"] = datetime.utcnow()
        
        # Open circuit breaker after 3 consecutive failures
        if breaker["failures"] >= 3:
            breaker["is_open"] = True
            logger.warning(f"Circuit breaker opened for {model_type.value}")
    
    def _reset_circuit_breaker(self, model_type: ModelType):
        """Reset circuit breaker on successful request"""
        breaker = self.circuit_breakers[model_type]
        breaker["failures"] = 0
        breaker["is_open"] = False
    
    async def _call_model(self, model_type: ModelType, prompt: str, 
                         temperature: float, max_tokens: int, **kwargs) -> str:
        """Call specific model API"""
        if model_type == ModelType.OPENAI:
            return await self._call_openai(prompt, temperature, max_tokens, **kwargs)
        elif model_type == ModelType.ANTHROPIC:
            return await self._call_anthropic(prompt, temperature, max_tokens, **kwargs)
        elif model_type == ModelType.OLLAMA:
            return await self._call_ollama(prompt, temperature, max_tokens, **kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    async def _call_openai(self, prompt: str, temperature: float, max_tokens: int, **kwargs) -> str:
        """Call OpenAI API"""
        if not self.openai_api_key:
            raise Exception("OpenAI API key not configured")
        
        config = self.model_configs[ModelType.OPENAI]
        
        payload = {
            "model": config["model"],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": min(max_tokens, config["max_tokens"])
        }
        
        async with aiohttp.ClientSession() as session:
            url = f"{config['base_url']}/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.openai_api_key}",
                "Content-Type": "application/json"
            }
            
            async with session.post(url, json=payload, headers=headers, timeout=30) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"OpenAI API error: {response.status} - {error_text}")
                
                data = await response.json()
                return data["choices"][0]["message"]["content"]
    
    async def _call_anthropic(self, prompt: str, temperature: float, max_tokens: int, **kwargs) -> str:
        """Call Anthropic API"""
        if not self.anthropic_api_key:
            raise Exception("Anthropic API key not configured")
        
        config = self.model_configs[ModelType.ANTHROPIC]
        
        payload = {
            "model": config["model"],
            "max_tokens": min(max_tokens, config["max_tokens"]),
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        async with aiohttp.ClientSession() as session:
            url = f"{config['base_url']}/messages"
            headers = {
                "x-api-key": self.anthropic_api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            }
            
            async with session.post(url, json=payload, headers=headers, timeout=30) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Anthropic API error: {response.status} - {error_text}")
                
                data = await response.json()
                return data["content"][0]["text"]
    
    async def _call_ollama(self, prompt: str, temperature: float, max_tokens: int, **kwargs) -> str:
        """Call Ollama API"""
        config = self.model_configs[ModelType.OLLAMA]
        
        payload = {
            "model": config["model"],
            "prompt": prompt,
            "temperature": temperature,
            "options": {
                "num_predict": min(max_tokens, config["max_tokens"])
            }
        }
        
        async with aiohttp.ClientSession() as session:
            url = f"{config['base_url']}/api/generate"
            
            async with session.post(url, json=payload, timeout=60) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Ollama API error: {response.status} - {error_text}")
                
                response_text = ""
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if line:
                        try:
                            data = json.loads(line)
                            if "response" in data:
                                response_text += data["response"]
                        except json.JSONDecodeError:
                            continue
                
                return response_text
    
    def get_stats(self) -> Dict[str, Any]:
        """Get LLM router statistics"""
        success_rate = (self.success_count / self.request_count * 100) if self.request_count > 0 else 0
        avg_response_time = (self.total_response_time / self.request_count) if self.request_count > 0 else 0
        
        return {
            "request_count": self.request_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "success_rate": success_rate,
            "average_response_time": avg_response_time,
            "circuit_breakers": {
                model.value: {
                    "failures": breaker["failures"],
                    "is_open": breaker["is_open"],
                    "last_failure": breaker["last_failure"].isoformat() if breaker["last_failure"] else None
                }
                for model, breaker in self.circuit_breakers.items()
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of LLM router"""
        available_models = []
        
        for model_type in ModelType:
            if not self._is_circuit_breaker_open(model_type):
                available_models.append(model_type.value)
        
        return {
            "status": "healthy" if available_models else "unhealthy",
            "available_models": available_models,
            "is_initialized": self.is_initialized,
            "stats": self.get_stats()
        }
    
    def reset_stats(self):
        """Reset router statistics"""
        self.request_count = 0
        self.success_count = 0
        self.error_count = 0
        self.total_response_time = 0.0
        logger.info("LLM Router stats reset")


# Global LLM router instance
llm_router = LLMRouter()
