"""
AI System - Core Settings
Enterprise-grade configuration management with environment variables and security
"""

from enum import Enum
from pydantic import field_validator, ConfigDict
from pydantic_settings import BaseSettings
from typing import Optional, List, Dict, Any
import os
from pathlib import Path
from urllib.parse import quote_plus

# --- 1. Enumerations ---
class ModelProvider(str, Enum):
    """Supported LLM providers - inherits from str for JSON/Env compatibility"""
    OLLAMA = "ollama"
    OPENAI = "openai"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"
    DEEPSEEK = "deepseek"
    QWEN = "qwen"
    HUGGINGFACE = "huggingface"

class EmbeddingProvider(str, Enum):
    """Supported embedding providers"""
    OLLAMA = "ollama"
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    SENTENCE_TRANSFORMERS = "sentence_transformers"

# --- 2. Existing Environment Loading Logic (Unchanged) ---
try:
    from dotenv import load_dotenv
    # Load .env file from project root
    env_path = Path(__file__).parent.parent.parent.parent / '.env'
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()
except ImportError:
    pass

# --- 3. Database URL Helper Logic (Unchanged) ---
_db_config_available = None
_db_config_func = None

def _load_db_config():
    """Lazy load db config to avoid circular imports"""
    global _db_config_available, _db_config_func
    if _db_config_available is None:
        try:
            try:
                from .db import get_db_config
                _db_config_func = get_db_config
                _db_config_available = True
            except ImportError:
                from app.core.db import get_db_config
                _db_config_func = get_db_config
                _db_config_available = True
        except (ImportError, ValueError):
            _db_config_available = False
            _db_config_func = None
    return _db_config_available, _db_config_func

def _get_database_url_from_db_config() -> Optional[str]:
    _db_available, _db_func = _load_db_config()
    if not _db_available or not _db_func:
        return None
    try:
        db_config = _db_func()
        user = quote_plus(db_config['user'])
        password = quote_plus(db_config['password'])
        host = db_config['host']
        port = db_config['port']
        database = db_config['database']
        return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{database}"
    except (ValueError, KeyError, Exception):
        return None

# --- 4. Main Settings Class ---
class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Application
    app_name: str = "AI AI Cloud Platform"
    app_version: str = "3.0.0"
    debug: bool = False
    environment: str = "development"
    
    # Provider Selection (New additions)
    model_provider: ModelProvider = ModelProvider.GEMINI
    embedding_provider: EmbeddingProvider = EmbeddingProvider.OPENAI
    
    # API Configuration
    api_v1_prefix: str = "/api/v1"
    cors_origins: List[str] = ["*"]
    trusted_hosts: List[str] = ["*"]
    
    # Database Configuration
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/Cloud"
    database_echo: bool = False
    database_pool_size: int = 20
    database_max_overflow: int = 30
    
    # Redis Configuration
    redis_url: str = "redis://localhost:6379/0"
    redis_password: Optional[str] = None
    redis_db: int = 0
    
    # Vector Database Configuration
    vector_db_type: str = "faiss"
    chroma_host: str = "localhost"
    chroma_port: int = 8001
    faiss_index_path: str = "./app/data/vectorstore"
    
    # AI Model Configuration
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    embedding_batch_size: int = 32
    embedding_device: str = "cpu"
    
    # LLM Configuration
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"
    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-sonnet-4-5"
    google_api_key: Optional[str] = None
    google_model: str = "models/gemini-2.5-flash"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama2"

    # Qwen Configuration (Voice + LLM)
    qwen_api_key: Optional[str] = None
    qwen_model: str = "qwen-max"
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # Voice Agent Configuration
    voice_mode: str = "omni"  # "omni" for Qwen3-Omni, "pipeline" for ASR->LLM->TTS
    voice_omni_model: str = "qwen3-omni"
    voice_asr_model: str = "qwen3-asr"
    voice_tts_model: str = "qwen3-tts"
    voice_llm_model: str = "qwen-turbo"
    voice_id: str = "alloy"
    voice_language: str = "en"
    voice_sample_rate: int = 24000

    # Voice Behavior
    voice_streaming: bool = True
    voice_enable_barge_in: bool = True
    voice_vad_threshold: float = 0.5
    voice_silence_duration_ms: int = 500
    voice_max_response_sentences: int = 3
    voice_enable_backchannels: bool = True

    # Local Voice Models (Fallback)
    local_asr_model: str = "faster-whisper-base"
    local_tts_model: str = "kokoro-82m"
    
    # Security, Monitoring & Logging (Unchanged)
    secret_key: str = "your-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 8
    rate_limit_per_minute: int = 100
    rate_limit_burst: int = 200
    prometheus_port: int = 9090
    grafana_port: int = 3000
    log_level: str = "INFO"
    log_format: str = "json"
    
    # Celery, File Storage, and Feature Flags (Unchanged)
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"
    celery_task_serializer: str = "json"
    celery_accept_content: List[str] = ["json"]
    celery_result_serializer: str = "json"
    celery_timezone: str = "UTC"
    upload_max_size: int = 50 * 1024 * 1024
    allowed_file_types: List[str] = [".pdf", ".doc", ".docx", ".txt"]
    storage_backend: str = "local"
    storage_path: str = "./uploads"
    max_concurrent_requests: int = 1000
    request_timeout: int = 30
    embedding_cache_ttl: int = 3600
    company_name: str = "NB Group"
    primary_color: str = "#DC2626"
    secondary_color: str = "#000000"
    enable_ai_matching: bool = True
    enable_fraud_detection: bool = True
    enable_analytics: bool = True
    enable_compliance_monitoring: bool = True
    enable_embeddings: bool = True
    enable_training: bool = True

    # --- Properties for resolving the "Active" model ---
    @property
    def active_api_key(self) -> Optional[str]:
        keys = {
            ModelProvider.OPENAI: self.openai_api_key,
            ModelProvider.ANTHROPIC: self.anthropic_api_key,
            ModelProvider.GEMINI: self.google_api_key,
            ModelProvider.HUGGINGFACE: os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN")
        }
        return keys.get(self.model_provider)

    @property
    def is_production(self) -> bool: return self.environment == "production"
    @property
    def is_staging(self) -> bool: return self.environment == "staging"
    @property
    def is_development(self) -> bool: return self.environment == "development"

    # --- Validators (Unchanged) ---
    @field_validator("database_url", mode="before")
    @classmethod
    def validate_database_url(cls, v):
        db_url_from_config = _get_database_url_from_db_config()
        if db_url_from_config:
            v = db_url_from_config
        if not v or v == "postgresql+asyncpg://postgres:postgres@localhost:5432/Cloud":
            env_url = os.environ.get("DATABASE_URL")
            if env_url: v = env_url
        if v and "postgresql" in v:
            v = v.replace("postgresql+psycopg2://", "postgresql+asyncpg://")
            if v.startswith("postgresql://"):
                v = v.replace("postgresql://", "postgresql+asyncpg://", 1)
        return v

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v):
        if not v: return "development"
        v_clean = str(v).strip().lower()
        if v_clean.startswith("production") or v_clean.startswith("prod"): return "production"
        if v_clean.startswith("staging") or v_clean.startswith("stag"): return "staging"
        return "development"

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        if v.upper() not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ValueError("Log level must be DEBUG, INFO, WARNING, ERROR, or CRITICAL")
        return v.upper()

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

# --- 5. Global Instance and Helper Functions ---
settings = Settings()

def get_settings() -> Settings: return settings
def get_database_url() -> str: return settings.database_url
def get_redis_url() -> str:
    if settings.redis_password:
        return f"redis://:{settings.redis_password}@{settings.redis_url.split('://')[1]}"
    return settings.redis_url

def get_ai_config() -> Dict[str, Any]:
    return {
        "model_provider": settings.model_provider,
        "embedding_provider": settings.embedding_provider,
        "active_api_key": settings.active_api_key,
        "embedding_model": settings.embedding_model,
        "openai_model": settings.openai_model,
        "anthropic_model": settings.anthropic_model,
        "google_model": settings.google_model,
        "ollama_base_url": settings.ollama_base_url,
        "qwen_model": settings.qwen_model,
        "qwen_base_url": settings.qwen_base_url,
    }


def get_voice_config() -> Dict[str, Any]:
    """Get voice agent configuration"""
    return {
        "mode": settings.voice_mode,
        "omni_model": settings.voice_omni_model,
        "asr_model": settings.voice_asr_model,
        "tts_model": settings.voice_tts_model,
        "llm_model": settings.voice_llm_model,
        "api_key": settings.qwen_api_key,
        "base_url": settings.qwen_base_url,
        "voice_id": settings.voice_id,
        "language": settings.voice_language,
        "sample_rate": settings.voice_sample_rate,
        "streaming": settings.voice_streaming,
        "enable_barge_in": settings.voice_enable_barge_in,
        "vad_threshold": settings.voice_vad_threshold,
        "silence_duration_ms": settings.voice_silence_duration_ms,
        "max_response_sentences": settings.voice_max_response_sentences,
        "enable_backchannels": settings.voice_enable_backchannels,
        "local_asr_model": settings.local_asr_model,
        "local_tts_model": settings.local_tts_model,
    }

# --- 6. Aliases for Backward Compatibility ---
# These ensure any existing agents still work with the new Settings structure
ModelConfig = Settings
CostModelConfig = Settings
SQLAgentConfig = Settings
ObservabilityModelConfig = Settings
CloudModelConfig = Settings
VoiceAgentConfig = Settings