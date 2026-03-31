"""
VaLLM Specialist Model - LLM Router Service.

Author: Joel Otepa Wembo
https://joelwembo.com

Intelligent routing between local models (Ollama) and commercial APIs (Claude, OpenAI).
"""

from typing import Dict, Any, List, Optional, Tuple
import logging
import asyncio
import aiohttp
import json
import re
import time
import threading
from datetime import datetime
from enum import Enum

import torch
from fastapi import APIRouter, Request

try:
    from app.core.settings import get_settings
    from app.core.logging import performance_logger
    from app.schemas.query import (
        DeveloperQueryRequest,
        DeveloperQueryResponse,
        QueryRequest,
        QueryResponse,
        QueryResult,
        TerminalQueryRequest,
        TerminalQueryResponse,
    )
    from app.services.ai.ml.direct_openai import direct_openai
    from app.services.ai.ml.embedding import embedding_service
    from app.services.ai.ml.specialist_profile import (
        SPECIALIST_DECLINE_MESSAGE,
        SPECIALIST_SYSTEM_PROMPT,
        assess_specialist_scope,
    )
    from app.services.ai.ml.web_search import web_search_service
except ImportError:
    from core.settings import get_settings
    from core.logging import performance_logger
    from schemas.query import (
        DeveloperQueryRequest,
        DeveloperQueryResponse,
        QueryRequest,
        QueryResponse,
        QueryResult,
        TerminalQueryRequest,
        TerminalQueryResponse,
    )
    from services.ai.ml.direct_openai import direct_openai
    from services.ai.ml.embedding import embedding_service
    from services.ai.ml.specialist_profile import (
        SPECIALIST_DECLINE_MESSAGE,
        SPECIALIST_SYSTEM_PROMPT,
        assess_specialist_scope,
    )
    from services.ai.ml.web_search import web_search_service

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

router = APIRouter(tags=["IT Specialist v1"])
router_v2 = APIRouter(tags=["IT Specialist v2"])
router_v3 = APIRouter(tags=["IT Specialist v3"])
_local_generation_lock = threading.Lock()


def _truncate_for_prompt(text: str, max_chars: int = 1200) -> str:
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


def _extract_code_blocks(text: str) -> List[str]:
    return [block.strip() for block in re.findall(r"```[^\n]*\n(.*?)```", text or "", flags=re.DOTALL) if block.strip()]


def _extract_shell_commands(text: str) -> List[str]:
    commands: List[str] = []
    for block in _extract_code_blocks(text):
        for line in block.splitlines():
            candidate = line.strip()
            if not candidate or candidate.startswith("#"):
                continue
            commands.append(candidate)
    return commands[:12]


def _score_value(item: Dict[str, Any]) -> float:
    try:
        return float(item.get("rerank_score", item.get("score", 0.0)) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _build_references(
    internal_results: List[Dict[str, Any]],
    web_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    references: List[Dict[str, Any]] = []

    for idx, item in enumerate(internal_results, start=1):
        metadata = item.get("metadata") or {}
        references.append(
            {
                "rank": idx,
                "provider": "internal",
                "title": metadata.get("file_name")
                or metadata.get("source_label")
                or metadata.get("dataset_file")
                or f"Internal result {idx}",
                "url": metadata.get("file_path") or metadata.get("dataset_file") or "",
                "snippet": _truncate_for_prompt(item.get("document") or item.get("content") or item.get("text") or ""),
                "score": _score_value(item),
            }
        )

    for item in web_results:
        references.append(
            {
                "rank": item.get("rank"),
                "provider": item.get("provider", "web"),
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("snippet", ""),
                "score": None,
            }
        )

    return references


def _to_query_results(internal_results: List[Dict[str, Any]]) -> List[QueryResult]:
    query_results: List[QueryResult] = []
    for item in internal_results:
        query_results.append(
            QueryResult(
                content=str(item.get("document") or item.get("content") or item.get("text") or ""),
                score=_score_value(item),
                metadata=item.get("metadata") or {},
            )
        )
    return query_results


def _build_context_sections(
    internal_results: List[Dict[str, Any]],
    web_results: List[Dict[str, Any]],
) -> Tuple[str, str]:
    internal_blocks = []
    for idx, item in enumerate(internal_results[:5], start=1):
        metadata = item.get("metadata") or {}
        title = metadata.get("file_name") or metadata.get("source_label") or f"Internal result {idx}"
        internal_blocks.append(
            f"[{idx}] {title}\n"
            f"Score: {_score_value(item):.4f}\n"
            f"Content: {_truncate_for_prompt(item.get('document') or item.get('content') or item.get('text') or '', 1400)}"
        )

    internal_context = "\n\n".join(internal_blocks) if internal_blocks else "No indexed internal context found."
    web_context = web_search_service.format_results_for_prompt(web_results)
    return internal_context, web_context


def _build_specialist_prompt(
    query: str,
    mode: str,
    internal_results: List[Dict[str, Any]],
    web_results: List[Dict[str, Any]],
    extra_context: Optional[Dict[str, Any]] = None,
) -> str:
    internal_context, web_context = _build_context_sections(internal_results, web_results)
    extra_context_text = json.dumps(extra_context or {}, ensure_ascii=True, default=str, indent=2)

    mode_instruction = {
        "general": "Provide a direct IT answer with practical steps and mention uncertainty when context is incomplete.",
        "developer": "Answer as a senior software and platform engineer. Include code or config only when it materially helps.",
        "terminal": "Answer with safe shell commands first, then explain what each command does and any risk or prerequisite.",
    }.get(mode, "Provide a direct IT answer with practical steps.")

    return (
        f"{SPECIALIST_SYSTEM_PROMPT}\n\n"
        f"Operating mode: {mode}\n"
        f"{mode_instruction}\n"
        "Prioritize internal indexed knowledge. Use live web context only to complement current facts or fill clear gaps.\n"
        "Do not answer outside the IT domain.\n\n"
        f"User request:\n{query}\n\n"
        f"Structured context:\n{extra_context_text}\n\n"
        f"Internal indexed knowledge:\n{internal_context}\n\n"
        f"Live web context:\n{web_context}\n\n"
        "Answer:"
    )


async def _search_internal_knowledge(query: str, top_k: int) -> List[Dict[str, Any]]:
    if embedding_service is None:
        return []

    try:
        if not getattr(embedding_service, "is_initialized", False):
            await embedding_service.initialize()
        return await embedding_service.search_faiss(query=query, top_k=top_k)
    except Exception as exc:
        logger.warning("Internal FAISS search failed: %s", exc)
        return []


async def _search_web(query: str, use_web_search: bool, max_results: int) -> List[Dict[str, Any]]:
    if not use_web_search or web_search_service is None or not web_search_service.is_available():
        return []

    try:
        return await web_search_service.search(query, max_results=max_results)
    except Exception as exc:
        logger.warning("Live web search failed: %s", exc)
        return []


def _strip_prompt_echo(prompt: str, generated_text: str) -> str:
    text = (generated_text or "").strip()
    if text.startswith(prompt):
        text = text[len(prompt):].strip()
    return text


async def _generate_with_local_model(http_request: Request, prompt: str, max_tokens: int = 700) -> str:
    model = getattr(http_request.app.state, "model", None)
    tokenizer = getattr(http_request.app.state, "tokenizer", None)
    if model is None or tokenizer is None:
        return ""

    def _run_generation() -> str:
        with _local_generation_lock:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")

            model_max_length = getattr(tokenizer, "model_max_length", 1024)
            if not isinstance(model_max_length, int) or model_max_length <= 0 or model_max_length > 4096:
                model_max_length = 1024

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=model_max_length,
            ).to(device)
            pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    temperature=0.2,
                    top_p=0.9,
                    repetition_penalty=1.05,
                    pad_token_id=pad_token_id,
                )

            generated = tokenizer.decode(output[0], skip_special_tokens=True)
            return _strip_prompt_echo(prompt, generated)

    try:
        return await asyncio.to_thread(_run_generation)
    except Exception as exc:
        logger.warning("Local model generation failed: %s", exc)
        return ""


async def _generate_specialist_answer(http_request: Request, prompt: str) -> str:
    local_answer = await _generate_with_local_model(http_request, prompt)
    if local_answer:
        return local_answer

    try:
        if getattr(settings, "openai_api_key", None):
            answer = await direct_openai.chat_completion(prompt, temperature=0.2, max_tokens=900)
            if answer:
                return answer
    except Exception as exc:
        logger.warning("Direct OpenAI generation failed: %s", exc)

    try:
        if getattr(settings, "anthropic_api_key", None):
            answer = await llm_router._call_anthropic(prompt, temperature=0.2, max_tokens=900)
            if answer:
                return answer
    except Exception as exc:
        logger.warning("Anthropic generation failed: %s", exc)

    try:
        answer = await llm_router._call_ollama(prompt, temperature=0.2, max_tokens=900)
        if answer:
            return answer
    except Exception as exc:
        logger.warning("Ollama generation failed: %s", exc)

    return ""


def _fallback_answer(
    query: str,
    internal_results: List[Dict[str, Any]],
    web_results: List[Dict[str, Any]],
) -> str:
    lines = [
        "A generative model was not available, so this is a structured answer assembled from retrieved IT context.",
        f"Request: {query}",
    ]

    if internal_results:
        lines.append("Internal knowledge:")
        for idx, item in enumerate(internal_results[:3], start=1):
            metadata = item.get("metadata") or {}
            title = metadata.get("file_name") or metadata.get("source_label") or f"Internal result {idx}"
            lines.append(f"{idx}. {title}: {_truncate_for_prompt(item.get('document') or item.get('content') or item.get('text') or '', 220)}")

    if web_results:
        lines.append("Web context:")
        for idx, item in enumerate(web_results[:3], start=1):
            lines.append(f"{idx}. {item.get('title', 'Web result')}: {_truncate_for_prompt(item.get('snippet', ''), 220)}")

    if not internal_results and not web_results:
        lines.append("No supporting IT context was retrieved. Refine the query with product, platform, error, or environment details.")

    return "\n".join(lines)


def _confidence_score(internal_results: List[Dict[str, Any]], web_results: List[Dict[str, Any]]) -> float:
    top_internal = _score_value(internal_results[0]) if internal_results else 0.0
    base = 0.35 + min(0.45, top_internal)
    if web_results:
        base += 0.1
    return round(min(base, 0.99), 2)


def _reasoning_summary(internal_results: List[Dict[str, Any]], web_results: List[Dict[str, Any]]) -> str:
    parts = []
    if internal_results:
        parts.append(f"Used {len(internal_results)} internal IT retrieval matches")
    if web_results:
        parts.append(f"added {len(web_results)} live web references")
    if not parts:
        parts.append("Answered without supporting retrieval context")
    return "; ".join(parts) + "."


@router.post("/query", response_model=QueryResponse)
async def query_specialist(payload: QueryRequest, http_request: Request) -> QueryResponse:
    started = time.perf_counter()
    assessment = assess_specialist_scope(payload.query)
    if not assessment.allowed:
        return QueryResponse(
            query=payload.query,
            answer=SPECIALIST_DECLINE_MESSAGE,
            reasoning=assessment.reason,
            confidence=1.0,
            model_version="it-specialist-v1",
            processing_time_ms=int((time.perf_counter() - started) * 1000),
        )

    internal_results = await _search_internal_knowledge(payload.query, payload.top_k)
    web_results = await _search_web(payload.query, payload.use_web_search, payload.web_search_max_results)
    prompt = _build_specialist_prompt(
        query=payload.query,
        mode="general",
        internal_results=internal_results,
        web_results=web_results,
        extra_context=payload.context,
    )
    answer = await _generate_specialist_answer(http_request, prompt)
    if not answer:
        answer = _fallback_answer(payload.query, internal_results, web_results)

    return QueryResponse(
        query=payload.query,
        answer=answer,
        results=_to_query_results(internal_results),
        references=_build_references(internal_results, web_results),
        reasoning=_reasoning_summary(internal_results, web_results) if payload.include_reasoning else None,
        confidence=_confidence_score(internal_results, web_results),
        model_version="it-specialist-v1",
        processing_time_ms=int((time.perf_counter() - started) * 1000),
    )


@router.post("/developer", response_model=DeveloperQueryResponse)
async def developer_specialist(payload: DeveloperQueryRequest, http_request: Request) -> DeveloperQueryResponse:
    started = time.perf_counter()
    assessment = assess_specialist_scope(payload.query)
    if not assessment.allowed:
        return DeveloperQueryResponse(
            query=payload.query,
            answer=SPECIALIST_DECLINE_MESSAGE,
            confidence=1.0,
            processing_time_ms=int((time.perf_counter() - started) * 1000),
        )

    extra_context = dict(payload.context or {})
    if payload.language:
        extra_context["language"] = payload.language
    if payload.framework:
        extra_context["framework"] = payload.framework

    internal_results = await _search_internal_knowledge(payload.query, 6)
    web_results = await _search_web(payload.query, payload.use_web_search, 5)
    prompt = _build_specialist_prompt(
        query=payload.query,
        mode="developer",
        internal_results=internal_results,
        web_results=web_results,
        extra_context=extra_context,
    )
    answer = await _generate_specialist_answer(http_request, prompt)
    if not answer:
        answer = _fallback_answer(payload.query, internal_results, web_results)

    return DeveloperQueryResponse(
        query=payload.query,
        answer=answer,
        code_snippets=_extract_code_blocks(answer),
        references=_build_references(internal_results, web_results),
        confidence=_confidence_score(internal_results, web_results),
        processing_time_ms=int((time.perf_counter() - started) * 1000),
    )


@router.post("/terminal", response_model=TerminalQueryResponse)
async def terminal_specialist(payload: TerminalQueryRequest, http_request: Request) -> TerminalQueryResponse:
    started = time.perf_counter()
    assessment = assess_specialist_scope(payload.query)
    if not assessment.allowed:
        return TerminalQueryResponse(
            query=payload.query,
            explanation=SPECIALIST_DECLINE_MESSAGE,
            warnings=["Out-of-scope request for an IT-only specialist model."],
            confidence=1.0,
            processing_time_ms=int((time.perf_counter() - started) * 1000),
        )

    extra_context = dict(payload.context or {})
    extra_context["shell"] = payload.shell
    extra_context["os_type"] = payload.os_type

    internal_results = await _search_internal_knowledge(payload.query, 6)
    web_results = await _search_web(payload.query, payload.use_web_search, 5)
    prompt = _build_specialist_prompt(
        query=payload.query,
        mode="terminal",
        internal_results=internal_results,
        web_results=web_results,
        extra_context=extra_context,
    )
    answer = await _generate_specialist_answer(http_request, prompt)
    if not answer:
        answer = _fallback_answer(payload.query, internal_results, web_results)

    commands = _extract_shell_commands(answer)
    warnings = []
    if not commands:
        warnings.append("No executable shell commands were detected in the generated answer.")

    return TerminalQueryResponse(
        query=payload.query,
        commands=commands,
        explanation=answer,
        warnings=warnings,
        confidence=_confidence_score(internal_results, web_results),
        processing_time_ms=int((time.perf_counter() - started) * 1000),
    )
