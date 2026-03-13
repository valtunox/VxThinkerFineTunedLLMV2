"""
VaLLM Specialist Model - Direct OpenAI API Service.

Author: Joel Otepa Wembo
https://joelwembo.com

Direct OpenAI API client that bypasses the LLM router for faster chat completions.
"""

import aiohttp
import logging
from typing import Dict, Any, Optional
from core.settings import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


class DirectOpenAIService:
    """Direct OpenAI API service for faster responses"""
    
    def __init__(self):
        self.api_key = settings.openai_api_key
        self.model = settings.openai_model
        self.base_url = "https://api.openai.com/v1"
        
    async def chat_completion(self, 
                             prompt: str, 
                             temperature: float = 0.7,
                             max_tokens: int = 1000) -> str:
        """Direct OpenAI chat completion"""
        if not self.api_key:
            raise Exception("OpenAI API key not configured")
        logger.info("OpenAI chat_completion: model=%s prompt_len=%s max_tokens=%s", self.model, len(prompt), max_tokens)

        # Some models (e.g. gpt-5, o1) only support temperature=1; others support any value.
        model_lower = (self.model or "").lower()
        use_temp_one = model_lower.startswith("gpt-5") or model_lower.startswith("o1") or "o1-" in model_lower
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": max_tokens,
        }
        if use_temp_one:
            payload["temperature"] = 1
        else:
            payload["temperature"] = temperature
        
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            async with session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=90)) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"OpenAI API error: {response.status} - {error_text}")
                
                data = await response.json()
                choices = data.get("choices") or []
                if not choices:
                    logger.warning("OpenAI chat_completion: empty choices in response")
                    return ""
                msg = choices[0].get("message") or {}
                content = msg.get("content")
                if content is None:
                    logger.warning("OpenAI chat_completion: content is null (finish_reason=%s)", choices[0].get("finish_reason"))
                    return ""
                text = (content if isinstance(content, str) else str(content)).strip()
                logger.info("OpenAI chat_completion: model=%s response_len=%s", self.model, len(text))
                return text


# Global instance
direct_openai = DirectOpenAIService()
