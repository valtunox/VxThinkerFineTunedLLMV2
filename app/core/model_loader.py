"""
Optimized Model Loader for Infinity AI Agents
Lazy loading and smart initialization for fast startup
"""

import os
import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class ModelCache:
    """Thread-safe model cache"""
    
    def __init__(self):
        self.cache = {}
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            return self.cache.get(key)
    
    def set(self, key: str, model: Any):
        with self.lock:
            self.cache[key] = model
    
    def clear(self):
        with self.lock:
            self.cache.clear()


class ModelLoader:
    """
    Optimized model loader for AI agents
    
    Architecture:
    - Cloud Agent: Uses tools (embeddings, NLP, FAISS) - loaded on demand
    - Other Agents: Use LLMs only (no embeddings) - no special loading needed
    """
    
    def __init__(self):
        self.model_cache = ModelCache()
        self.loaded_models = {}
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Only Cloud Agent needs special tools
        self.cloud_tools = ["embeddings_service", "nlp_processor", "vector_store"]
        
        # Environment detection
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.is_production = self.environment == "production"
        self.is_testing = self.environment in ["test", "testing", "development"]
        
        logger.info(f"🚀 Model Loader initialized (environment: {self.environment})")
    
    async def initialize_critical_services(self) -> Dict[str, bool]:
        """
        Initialize only critical services for fast startup
        Cloud Agent tools are loaded lazily when needed
        """
        logger.info("⚡ Fast startup: Skipping heavy model loading")
        logger.info("   - Cloud Agent tools will load on first use")
        logger.info("   - Other agents use commercial LLMs (no loading needed)")
        
        start_time = datetime.now()
        results = {"startup": "fast_mode"}
        
        # In testing/dev, don't load anything upfront
        if self.is_testing:
            logger.info("🧪 Testing mode: All models load on-demand")
            results["mode"] = "lazy_loading"
        
        load_time = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(f"✅ Startup completed in {load_time:.2f}ms")
        
        return results
    
    async def load_cloud_tools_on_demand(self) -> Dict[str, bool]:
        """
        Load Cloud Agent tools - DISABLED: Full agentic and generative AI
        Embeddings, NLP, and vector stores are disabled - using full agentic and generative AI only
        """
        logger.info("🔧 Cloud Agent tools loading requested - DISABLED: Using full agentic and generative AI")
        logger.info("   - Embeddings: Disabled")
        logger.info("   - NLP Processor: Disabled")
        logger.info("   - Vector Store: Disabled")
        logger.info("   - Using: Full agentic and generative AI (LLMs only)")
        
        # Disabled: Full agentic and generative AI - no embeddings/NLP/vector stores
        # Return empty results to indicate tools are not loaded (by design)
        results = {
            "embeddings_service": False,
            "nlp_processor": False,
            "vector_store": False,
            "note": "Tools disabled - using full agentic and generative AI"
        }
        
        return results
    
    async def ensure_cloud_tools_loaded(self) -> bool:
        """Ensure Cloud Agent tools are loaded - DISABLED: Full agentic and generative AI"""
        # Disabled: Full agentic and generative AI - no tools to load
        logger.info("ℹ️  Cloud tools disabled - using full agentic and generative AI")
        return True  # Always return True since tools are disabled
    
    def get_model(self, model_name: str):
        """Get a loaded model"""
        return self.loaded_models.get(model_name)
    
    def is_model_loaded(self, model_name: str) -> bool:
        """Check if a model is loaded"""
        return model_name in self.loaded_models
    
    def get_loading_stats(self) -> Dict[str, Any]:
        """Get model loading statistics"""
        return {
            "loaded_models": list(self.loaded_models.keys()),
            "cache_size": len(self.model_cache.cache),
            "cloud_tools": self.cloud_tools,
            "environment": self.environment,
            "is_testing": self.is_testing,
            "is_production": self.is_production,
            "lazy_loading": True
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of loaded models"""
        health_status = {}
        
        for model_name, model in self.loaded_models.items():
            try:
                if hasattr(model, 'health_check'):
                    health = await model.health_check()
                    health_status[model_name] = health
                else:
                    health_status[model_name] = {
                        "status": "healthy",
                        "is_initialized": getattr(model, 'is_initialized', True)
                    }
            except Exception as e:
                health_status[model_name] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
        
        return health_status


# Global model loader instance
_model_loader = None


def get_model_loader() -> ModelLoader:
    """Get or create the global model loader instance"""
    global _model_loader
    
    if _model_loader is None:
        _model_loader = ModelLoader()
    
    return _model_loader


# Convenience instance
model_loader = get_model_loader()

