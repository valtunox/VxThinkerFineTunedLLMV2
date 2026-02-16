"""
AI Model Registry
==================

Production-ready central registry for AI models across all providers.
Supports text, voice, and multimodal models with cloud-agnostic design.

Features:
- Multi-provider support (OpenAI, Anthropic, Gemini, Qwen, DeepSeek, Ollama, HuggingFace)
- Voice model registry (Qwen3-Omni, Whisper, Kokoro, etc.)
- Model capability detection
- Environment variable configuration
- Cost estimation per model
"""

import os
from typing import Dict, List, Optional, Any, Set
from enum import Enum
from dataclasses import dataclass, field


# ============================================================================
# MODEL CAPABILITY ENUMS
# ============================================================================

class ModelCapability(str, Enum):
    """Model capabilities for feature detection"""
    TEXT_GENERATION = "text_generation"
    CHAT = "chat"
    FUNCTION_CALLING = "function_calling"
    VISION = "vision"
    AUDIO_INPUT = "audio_input"
    AUDIO_OUTPUT = "audio_output"
    SPEECH_TO_SPEECH = "speech_to_speech"
    STREAMING = "streaming"
    EMBEDDINGS = "embeddings"
    CODE_GENERATION = "code_generation"
    REASONING = "reasoning"
    MULTIMODAL = "multimodal"


class ModelType(str, Enum):
    """Model type classification"""
    LLM = "llm"
    VOICE = "voice"
    EMBEDDING = "embedding"
    ASR = "asr"  # Automatic Speech Recognition
    TTS = "tts"  # Text to Speech
    OMNI = "omni"  # Speech-to-speech multimodal


# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

@dataclass
class ModelInfo:
    """Comprehensive model information"""
    id: str
    name: str
    provider: str
    model_type: ModelType
    capabilities: Set[ModelCapability] = field(default_factory=set)
    max_context_tokens: int = 4096
    max_output_tokens: int = 2048
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    latency_class: str = "standard"  # "low", "standard", "high"
    supports_streaming: bool = False
    supports_barge_in: bool = False
    api_base_url: Optional[str] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "provider": self.provider,
            "model_type": self.model_type.value,
            "capabilities": [c.value for c in self.capabilities],
            "max_context_tokens": self.max_context_tokens,
            "max_output_tokens": self.max_output_tokens,
            "cost_per_1k_input": self.cost_per_1k_input,
            "cost_per_1k_output": self.cost_per_1k_output,
            "latency_class": self.latency_class,
            "supports_streaming": self.supports_streaming,
            "supports_barge_in": self.supports_barge_in,
        }


# ============================================================================
# QWEN MODELS (Including Qwen3-Omni Voice Models)
# ============================================================================

QWEN_MODELS = [
    # Qwen3-Omni - Primary voice model (speech-to-speech)
    ModelInfo(
        id="qwen3-omni",
        name="Qwen3-Omni",
        provider="qwen",
        model_type=ModelType.OMNI,
        capabilities={
            ModelCapability.SPEECH_TO_SPEECH,
            ModelCapability.AUDIO_INPUT,
            ModelCapability.AUDIO_OUTPUT,
            ModelCapability.STREAMING,
            ModelCapability.CHAT,
            ModelCapability.MULTIMODAL,
        },
        max_context_tokens=32768,
        max_output_tokens=8192,
        cost_per_1k_input=0.002,
        cost_per_1k_output=0.006,
        latency_class="low",
        supports_streaming=True,
        supports_barge_in=True,
        api_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        notes="Primary streaming speech-to-speech model with barge-in support"
    ),
    # Qwen3-ASR - Speech recognition
    ModelInfo(
        id="qwen3-asr",
        name="Qwen3-ASR",
        provider="qwen",
        model_type=ModelType.ASR,
        capabilities={
            ModelCapability.AUDIO_INPUT,
            ModelCapability.STREAMING,
        },
        max_context_tokens=30000,  # ~30 seconds of audio
        cost_per_1k_input=0.001,
        latency_class="low",
        supports_streaming=True,
        api_base_url="https://dashscope.aliyuncs.com/api/v1/services/asr",
        notes="Real-time speech recognition"
    ),
    # Qwen3-TTS - Text to speech
    ModelInfo(
        id="qwen3-tts",
        name="Qwen3-TTS",
        provider="qwen",
        model_type=ModelType.TTS,
        capabilities={
            ModelCapability.AUDIO_OUTPUT,
            ModelCapability.STREAMING,
        },
        cost_per_1k_output=0.002,
        latency_class="low",
        supports_streaming=True,
        api_base_url="https://dashscope.aliyuncs.com/api/v1/services/tts",
        notes="Natural streaming text-to-speech"
    ),
    # Qwen Text Models
    ModelInfo(
        id="qwen-max",
        name="Qwen Max",
        provider="qwen",
        model_type=ModelType.LLM,
        capabilities={
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CHAT,
            ModelCapability.FUNCTION_CALLING,
            ModelCapability.CODE_GENERATION,
            ModelCapability.REASONING,
            ModelCapability.STREAMING,
        },
        max_context_tokens=32768,
        max_output_tokens=8192,
        cost_per_1k_input=0.004,
        cost_per_1k_output=0.012,
        supports_streaming=True,
        api_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    ),
    ModelInfo(
        id="qwen-plus",
        name="Qwen Plus",
        provider="qwen",
        model_type=ModelType.LLM,
        capabilities={
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CHAT,
            ModelCapability.FUNCTION_CALLING,
            ModelCapability.STREAMING,
        },
        max_context_tokens=131072,
        max_output_tokens=8192,
        cost_per_1k_input=0.002,
        cost_per_1k_output=0.006,
        supports_streaming=True,
        api_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    ),
    ModelInfo(
        id="qwen-turbo",
        name="Qwen Turbo",
        provider="qwen",
        model_type=ModelType.LLM,
        capabilities={
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CHAT,
            ModelCapability.STREAMING,
        },
        max_context_tokens=131072,
        max_output_tokens=8192,
        cost_per_1k_input=0.0005,
        cost_per_1k_output=0.002,
        latency_class="low",
        supports_streaming=True,
        api_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    ),
    # Qwen2.5 Local Models
    ModelInfo(
        id="Qwen/Qwen2.5-3B-Instruct",
        name="Qwen2.5 3B Instruct",
        provider="huggingface",
        model_type=ModelType.LLM,
        capabilities={
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CHAT,
            ModelCapability.STREAMING,
        },
        max_context_tokens=32768,
        max_output_tokens=2048,
        cost_per_1k_input=0.0,  # Local model
        cost_per_1k_output=0.0,
        supports_streaming=True,
        notes="Local HuggingFace model for offline use"
    ),
    ModelInfo(
        id="Qwen/Qwen2.5-7B-Instruct",
        name="Qwen2.5 7B Instruct",
        provider="huggingface",
        model_type=ModelType.LLM,
        capabilities={
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CHAT,
            ModelCapability.FUNCTION_CALLING,
            ModelCapability.STREAMING,
        },
        max_context_tokens=131072,
        max_output_tokens=8192,
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        supports_streaming=True,
        notes="Local HuggingFace model for offline use"
    ),
]

# ============================================================================
# OPENAI MODELS
# ============================================================================

OPENAI_MODELS = [
    ModelInfo(
        id="gpt-5",
        name="GPT-5",
        provider="openai",
        model_type=ModelType.LLM,
        capabilities={
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CHAT,
            ModelCapability.FUNCTION_CALLING,
            ModelCapability.VISION,
            ModelCapability.CODE_GENERATION,
            ModelCapability.REASONING,
            ModelCapability.STREAMING,
        },
        max_context_tokens=128000,
        max_output_tokens=16384,
        cost_per_1k_input=0.01,
        cost_per_1k_output=0.03,
        supports_streaming=True,
    ),
    ModelInfo(
        id="gpt-4o",
        name="GPT-4o",
        provider="openai",
        model_type=ModelType.LLM,
        capabilities={
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CHAT,
            ModelCapability.FUNCTION_CALLING,
            ModelCapability.VISION,
            ModelCapability.AUDIO_INPUT,
            ModelCapability.AUDIO_OUTPUT,
            ModelCapability.STREAMING,
            ModelCapability.MULTIMODAL,
        },
        max_context_tokens=128000,
        max_output_tokens=16384,
        cost_per_1k_input=0.005,
        cost_per_1k_output=0.015,
        supports_streaming=True,
        supports_barge_in=True,
    ),
    ModelInfo(
        id="gpt-4o-mini",
        name="GPT-4o Mini",
        provider="openai",
        model_type=ModelType.LLM,
        capabilities={
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CHAT,
            ModelCapability.FUNCTION_CALLING,
            ModelCapability.VISION,
            ModelCapability.STREAMING,
        },
        max_context_tokens=128000,
        max_output_tokens=16384,
        cost_per_1k_input=0.00015,
        cost_per_1k_output=0.0006,
        latency_class="low",
        supports_streaming=True,
    ),
    ModelInfo(
        id="gpt-4-turbo",
        name="GPT-4 Turbo",
        provider="openai",
        model_type=ModelType.LLM,
        capabilities={
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CHAT,
            ModelCapability.FUNCTION_CALLING,
            ModelCapability.VISION,
            ModelCapability.STREAMING,
        },
        max_context_tokens=128000,
        max_output_tokens=4096,
        cost_per_1k_input=0.01,
        cost_per_1k_output=0.03,
        supports_streaming=True,
    ),
    ModelInfo(
        id="gpt-3.5-turbo",
        name="GPT-3.5 Turbo",
        provider="openai",
        model_type=ModelType.LLM,
        capabilities={
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CHAT,
            ModelCapability.FUNCTION_CALLING,
            ModelCapability.STREAMING,
        },
        max_context_tokens=16385,
        max_output_tokens=4096,
        cost_per_1k_input=0.0005,
        cost_per_1k_output=0.0015,
        latency_class="low",
        supports_streaming=True,
    ),
    # OpenAI Whisper
    ModelInfo(
        id="whisper-1",
        name="Whisper",
        provider="openai",
        model_type=ModelType.ASR,
        capabilities={ModelCapability.AUDIO_INPUT},
        cost_per_1k_input=0.006,  # per minute
        notes="OpenAI hosted Whisper ASR"
    ),
    # OpenAI TTS
    ModelInfo(
        id="tts-1",
        name="TTS-1",
        provider="openai",
        model_type=ModelType.TTS,
        capabilities={ModelCapability.AUDIO_OUTPUT, ModelCapability.STREAMING},
        cost_per_1k_output=0.015,
        latency_class="low",
        supports_streaming=True,
    ),
    ModelInfo(
        id="tts-1-hd",
        name="TTS-1 HD",
        provider="openai",
        model_type=ModelType.TTS,
        capabilities={ModelCapability.AUDIO_OUTPUT, ModelCapability.STREAMING},
        cost_per_1k_output=0.030,
        supports_streaming=True,
    ),
]

# ============================================================================
# ANTHROPIC MODELS
# ============================================================================

ANTHROPIC_MODELS = [
    ModelInfo(
        id="claude-4.5",
        name="Claude 4.5",
        provider="anthropic",
        model_type=ModelType.LLM,
        capabilities={
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CHAT,
            ModelCapability.VISION,
            ModelCapability.CODE_GENERATION,
            ModelCapability.REASONING,
            ModelCapability.STREAMING,
        },
        max_context_tokens=200000,
        max_output_tokens=8192,
        cost_per_1k_input=0.003,
        cost_per_1k_output=0.015,
        supports_streaming=True,
        notes="Maps to claude-3-5-sonnet-20241022"
    ),
    ModelInfo(
        id="claude-3-5-sonnet-20241022",
        name="Claude 3.5 Sonnet",
        provider="anthropic",
        model_type=ModelType.LLM,
        capabilities={
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CHAT,
            ModelCapability.VISION,
            ModelCapability.CODE_GENERATION,
            ModelCapability.REASONING,
            ModelCapability.STREAMING,
        },
        max_context_tokens=200000,
        max_output_tokens=8192,
        cost_per_1k_input=0.003,
        cost_per_1k_output=0.015,
        supports_streaming=True,
    ),
    ModelInfo(
        id="claude-3-opus-20240229",
        name="Claude 3 Opus",
        provider="anthropic",
        model_type=ModelType.LLM,
        capabilities={
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CHAT,
            ModelCapability.VISION,
            ModelCapability.CODE_GENERATION,
            ModelCapability.REASONING,
            ModelCapability.STREAMING,
        },
        max_context_tokens=200000,
        max_output_tokens=4096,
        cost_per_1k_input=0.015,
        cost_per_1k_output=0.075,
        supports_streaming=True,
    ),
    ModelInfo(
        id="claude-3-haiku-20240307",
        name="Claude 3 Haiku",
        provider="anthropic",
        model_type=ModelType.LLM,
        capabilities={
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CHAT,
            ModelCapability.VISION,
            ModelCapability.STREAMING,
        },
        max_context_tokens=200000,
        max_output_tokens=4096,
        cost_per_1k_input=0.00025,
        cost_per_1k_output=0.00125,
        latency_class="low",
        supports_streaming=True,
    ),
]

# ============================================================================
# GEMINI MODELS
# ============================================================================

GEMINI_MODELS = [
    ModelInfo(
        id="gemini-2.0-flash",
        name="Gemini 2.0 Flash",
        provider="gemini",
        model_type=ModelType.LLM,
        capabilities={
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CHAT,
            ModelCapability.FUNCTION_CALLING,
            ModelCapability.VISION,
            ModelCapability.AUDIO_INPUT,
            ModelCapability.STREAMING,
            ModelCapability.MULTIMODAL,
        },
        max_context_tokens=1048576,
        max_output_tokens=8192,
        cost_per_1k_input=0.0,  # Free tier available
        cost_per_1k_output=0.0,
        latency_class="low",
        supports_streaming=True,
    ),
    ModelInfo(
        id="gemini-1.5-pro",
        name="Gemini 1.5 Pro",
        provider="gemini",
        model_type=ModelType.LLM,
        capabilities={
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CHAT,
            ModelCapability.FUNCTION_CALLING,
            ModelCapability.VISION,
            ModelCapability.AUDIO_INPUT,
            ModelCapability.STREAMING,
            ModelCapability.MULTIMODAL,
            ModelCapability.REASONING,
        },
        max_context_tokens=2097152,
        max_output_tokens=8192,
        cost_per_1k_input=0.00125,
        cost_per_1k_output=0.005,
        supports_streaming=True,
    ),
    ModelInfo(
        id="gemini-1.5-flash",
        name="Gemini 1.5 Flash",
        provider="gemini",
        model_type=ModelType.LLM,
        capabilities={
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CHAT,
            ModelCapability.FUNCTION_CALLING,
            ModelCapability.VISION,
            ModelCapability.STREAMING,
        },
        max_context_tokens=1048576,
        max_output_tokens=8192,
        cost_per_1k_input=0.000075,
        cost_per_1k_output=0.0003,
        latency_class="low",
        supports_streaming=True,
    ),
]

# ============================================================================
# DEEPSEEK MODELS
# ============================================================================

DEEPSEEK_MODELS = [
    ModelInfo(
        id="deepseek-chat",
        name="DeepSeek Chat",
        provider="deepseek",
        model_type=ModelType.LLM,
        capabilities={
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CHAT,
            ModelCapability.FUNCTION_CALLING,
            ModelCapability.REASONING,
            ModelCapability.STREAMING,
        },
        max_context_tokens=64000,
        max_output_tokens=8192,
        cost_per_1k_input=0.00014,
        cost_per_1k_output=0.00028,
        supports_streaming=True,
        api_base_url="https://api.deepseek.com/v1"
    ),
    ModelInfo(
        id="deepseek-coder",
        name="DeepSeek Coder",
        provider="deepseek",
        model_type=ModelType.LLM,
        capabilities={
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CHAT,
            ModelCapability.CODE_GENERATION,
            ModelCapability.STREAMING,
        },
        max_context_tokens=64000,
        max_output_tokens=8192,
        cost_per_1k_input=0.00014,
        cost_per_1k_output=0.00028,
        supports_streaming=True,
        api_base_url="https://api.deepseek.com/v1"
    ),
]

# ============================================================================
# OLLAMA MODELS (Local)
# ============================================================================

OLLAMA_MODELS = [
    ModelInfo(
        id="ollama",
        name="Ollama (Local)",
        provider="ollama",
        model_type=ModelType.LLM,
        capabilities={
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CHAT,
            ModelCapability.STREAMING,
        },
        max_context_tokens=8192,
        max_output_tokens=4096,
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        supports_streaming=True,
        api_base_url="http://localhost:11434"
    ),
    ModelInfo(
        id="llama3.1",
        name="Llama 3.1",
        provider="ollama",
        model_type=ModelType.LLM,
        capabilities={
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CHAT,
            ModelCapability.STREAMING,
        },
        max_context_tokens=131072,
        max_output_tokens=8192,
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        supports_streaming=True,
        api_base_url="http://localhost:11434"
    ),
]

# ============================================================================
# VOICE-SPECIFIC MODELS (Local)
# ============================================================================

VOICE_MODELS = [
    # Faster Whisper (Local ASR)
    ModelInfo(
        id="faster-whisper-base",
        name="Faster Whisper Base",
        provider="local",
        model_type=ModelType.ASR,
        capabilities={ModelCapability.AUDIO_INPUT},
        cost_per_1k_input=0.0,
        latency_class="low",
        notes="Local Whisper model for offline ASR"
    ),
    ModelInfo(
        id="faster-whisper-small",
        name="Faster Whisper Small",
        provider="local",
        model_type=ModelType.ASR,
        capabilities={ModelCapability.AUDIO_INPUT},
        cost_per_1k_input=0.0,
        notes="Local Whisper model with better accuracy"
    ),
    ModelInfo(
        id="faster-whisper-medium",
        name="Faster Whisper Medium",
        provider="local",
        model_type=ModelType.ASR,
        capabilities={ModelCapability.AUDIO_INPUT},
        cost_per_1k_input=0.0,
        notes="Local Whisper model with high accuracy"
    ),
    ModelInfo(
        id="faster-whisper-large-v3",
        name="Faster Whisper Large V3",
        provider="local",
        model_type=ModelType.ASR,
        capabilities={ModelCapability.AUDIO_INPUT},
        cost_per_1k_input=0.0,
        notes="Highest accuracy local Whisper"
    ),
    # Kokoro TTS (Local)
    ModelInfo(
        id="kokoro-82m",
        name="Kokoro 82M",
        provider="local",
        model_type=ModelType.TTS,
        capabilities={ModelCapability.AUDIO_OUTPUT},
        cost_per_1k_output=0.0,
        latency_class="low",
        notes="Lightweight high-quality local TTS"
    ),
]

# ============================================================================
# HUGGINGFACE MODELS
# ============================================================================

HUGGINGFACE_MODELS = [
    ModelInfo(
        id="huggingface",
        name="Hugging Face",
        provider="huggingface",
        model_type=ModelType.LLM,
        capabilities={
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CHAT,
        },
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        notes="Generic HuggingFace model placeholder"
    ),
]


# ============================================================================
# CONSOLIDATED REGISTRY
# ============================================================================

# All models combined
ALL_MODELS: List[ModelInfo] = (
    QWEN_MODELS +
    OPENAI_MODELS +
    ANTHROPIC_MODELS +
    GEMINI_MODELS +
    DEEPSEEK_MODELS +
    OLLAMA_MODELS +
    VOICE_MODELS +
    HUGGINGFACE_MODELS
)

# Index by ID for fast lookup
MODEL_REGISTRY: Dict[str, ModelInfo] = {model.id: model for model in ALL_MODELS}

# Legacy format for backward compatibility
SUPPORTED_MODELS = [{"id": m.id, "name": m.name, "provider": m.provider} for m in ALL_MODELS]
MODEL_PROVIDER_MAP = {m.id: m.provider for m in ALL_MODELS}

# Default models per provider
DEFAULT_MODELS = {
    "openai": os.getenv("OPENAI_MODEL", "gpt-4o"),
    "anthropic": os.getenv("ANTHROPIC_MODEL", "claude-4.5"),
    "gemini": os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"),
    "deepseek": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
    "qwen": os.getenv("QWEN_MODEL", "qwen-max"),
    "qwen_voice": os.getenv("QWEN_VOICE_MODEL", "qwen3-omni"),
    "ollama": os.getenv("OLLAMA_MODEL", "llama3.1"),
    "local_asr": os.getenv("LOCAL_ASR_MODEL", "faster-whisper-base"),
    "local_tts": os.getenv("LOCAL_TTS_MODEL", "kokoro-82m"),
}


# ============================================================================
# REGISTRY FUNCTIONS
# ============================================================================

def get_model(model_id: str) -> Optional[ModelInfo]:
    """Get model information by ID"""
    return MODEL_REGISTRY.get(model_id)


def get_default_model(provider: str) -> str:
    """Get default model for a provider"""
    provider = provider.lower().strip()
    return DEFAULT_MODELS.get(provider, "gpt-4o-mini")


def get_models_by_provider(provider: Optional[str] = None) -> Dict[str, List[ModelInfo]]:
    """Get models grouped by provider, optionally filtered"""
    result = {}
    for model in ALL_MODELS:
        if provider and model.provider != provider:
            continue
        if model.provider not in result:
            result[model.provider] = []
        result[model.provider].append(model)
    return result


def get_models_by_type(model_type: ModelType) -> List[ModelInfo]:
    """Get all models of a specific type"""
    return [m for m in ALL_MODELS if m.model_type == model_type]


def get_models_by_capability(capability: ModelCapability) -> List[ModelInfo]:
    """Get all models with a specific capability"""
    return [m for m in ALL_MODELS if capability in m.capabilities]


def get_voice_models() -> Dict[str, List[ModelInfo]]:
    """Get all voice-related models (ASR, TTS, Omni)"""
    return {
        "omni": get_models_by_type(ModelType.OMNI),
        "asr": get_models_by_type(ModelType.ASR),
        "tts": get_models_by_type(ModelType.TTS),
    }


def get_streaming_voice_models() -> List[ModelInfo]:
    """Get voice models that support streaming"""
    return [
        m for m in ALL_MODELS
        if m.supports_streaming and (
            ModelCapability.AUDIO_INPUT in m.capabilities or
            ModelCapability.AUDIO_OUTPUT in m.capabilities or
            ModelCapability.SPEECH_TO_SPEECH in m.capabilities
        )
    ]


def get_barge_in_capable_models() -> List[ModelInfo]:
    """Get models that support interruption (barge-in)"""
    return [m for m in ALL_MODELS if m.supports_barge_in]


def get_low_latency_models() -> List[ModelInfo]:
    """Get models optimized for low latency"""
    return [m for m in ALL_MODELS if m.latency_class == "low"]


def get_provider_for_model(model_id: str) -> Optional[str]:
    """Get provider for a specific model"""
    model = MODEL_REGISTRY.get(model_id)
    return model.provider if model else None


def is_model_supported(model_id: str) -> bool:
    """Check if a model is supported"""
    return model_id in MODEL_REGISTRY


def get_model_info(model_id: str) -> Optional[Dict[str, Any]]:
    """Get detailed model information as dictionary"""
    model = MODEL_REGISTRY.get(model_id)
    return model.to_dict() if model else None


def list_all_models() -> List[str]:
    """Get list of all model IDs"""
    return list(MODEL_REGISTRY.keys())


def validate_provider_model(provider: str, model: str) -> bool:
    """Validate that a model belongs to a provider"""
    expected_provider = get_provider_for_model(model)
    return expected_provider is not None and expected_provider.lower() == provider.lower()


def estimate_cost(model_id: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost for a model request"""
    model = MODEL_REGISTRY.get(model_id)
    if not model:
        return 0.0
    return (
        (input_tokens / 1000) * model.cost_per_1k_input +
        (output_tokens / 1000) * model.cost_per_1k_output
    )


# ============================================================================
# API KEY MANAGEMENT
# ============================================================================

def get_api_key(provider: str) -> Optional[str]:
    """Get API key for a provider from environment"""
    provider = provider.lower().strip()

    key_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        "deepseek": "DEEPSEEK_API_KEY",
        "qwen": ["QWEN_API_KEY", "DASHSCOPE_API_KEY"],
        "huggingface": ["HUGGINGFACE_API_KEY", "HF_TOKEN"],
    }

    env_vars = key_map.get(provider)
    if not env_vars:
        return None

    if isinstance(env_vars, str):
        return os.getenv(env_vars)

    for var in env_vars:
        key = os.getenv(var)
        if key:
            return key
    return None


def get_provider_config(provider: str) -> Dict[str, Any]:
    """Get complete configuration for a provider"""
    provider = provider.lower().strip()

    models = get_models_by_provider(provider).get(provider, [])

    return {
        "provider": provider,
        "api_key": get_api_key(provider),
        "default_model": get_default_model(provider),
        "available_models": [m.id for m in models],
        "model_details": [m.to_dict() for m in models],
    }


def check_api_keys_available() -> Dict[str, bool]:
    """Check which API keys are configured"""
    providers = ["openai", "anthropic", "gemini", "deepseek", "qwen", "huggingface"]
    status = {}

    for provider in providers:
        api_key = get_api_key(provider)
        status[provider] = bool(api_key and api_key.strip() and not api_key.startswith("your_"))

    # Ollama and local models are always available
    status["ollama"] = True
    status["local"] = True

    return status


# ============================================================================
# MODEL NAME MAPPING (for compatibility)
# ============================================================================

MODEL_NAME_MAPPING = {
    # Claude aliases
    "claude-4.5": "claude-3-5-sonnet-20241022",
    "claude-4": "claude-3-5-sonnet-20241022",
    "claude-sonnet": "claude-3-5-sonnet-20241022",
    "claude-opus": "claude-3-opus-20240229",
    "claude-haiku": "claude-3-haiku-20240307",

    # GPT aliases
    "gpt-5": "gpt-4o",  # Map to available model
    "gpt4": "gpt-4-turbo",
    "gpt4o": "gpt-4o",
    "gpt4o-mini": "gpt-4o-mini",

    # Gemini aliases
    "gemini": "gemini-2.0-flash",
    "gemini-pro": "gemini-1.5-pro",
    "gemini-flash": "gemini-2.0-flash",

    # Qwen aliases
    "qwen": "qwen-max",
    "qwen-voice": "qwen3-omni",
    "qwen-asr": "qwen3-asr",
    "qwen-tts": "qwen3-tts",

    # DeepSeek aliases
    "deepseek": "deepseek-chat",

    # Whisper aliases
    "whisper": "faster-whisper-base",
    "whisper-base": "faster-whisper-base",
    "whisper-small": "faster-whisper-small",
    "whisper-medium": "faster-whisper-medium",
    "whisper-large": "faster-whisper-large-v3",
}


def resolve_model_name(model_name: str) -> str:
    """Resolve model aliases to actual model IDs"""
    return MODEL_NAME_MAPPING.get(model_name.lower(), model_name)


# ============================================================================
# LEGACY COMPATIBILITY FUNCTIONS
# ============================================================================

def get_openai_models() -> List[Dict[str, str]]:
    """Get OpenAI models (legacy)"""
    return [{"id": m.id, "name": m.name, "provider": m.provider} for m in OPENAI_MODELS]

def get_anthropic_models() -> List[Dict[str, str]]:
    """Get Anthropic models (legacy)"""
    return [{"id": m.id, "name": m.name, "provider": m.provider} for m in ANTHROPIC_MODELS]

def get_gemini_models() -> List[Dict[str, str]]:
    """Get Gemini models (legacy)"""
    return [{"id": m.id, "name": m.name, "provider": m.provider} for m in GEMINI_MODELS]

def get_deepseek_models() -> List[Dict[str, str]]:
    """Get DeepSeek models (legacy)"""
    return [{"id": m.id, "name": m.name, "provider": m.provider} for m in DEEPSEEK_MODELS]

def get_qwen_models() -> List[Dict[str, str]]:
    """Get Qwen models (legacy)"""
    return [{"id": m.id, "name": m.name, "provider": m.provider} for m in QWEN_MODELS]


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Classes
    "ModelInfo",
    "ModelCapability",
    "ModelType",

    # Model lists
    "ALL_MODELS",
    "MODEL_REGISTRY",
    "SUPPORTED_MODELS",
    "MODEL_PROVIDER_MAP",
    "DEFAULT_MODELS",
    "QWEN_MODELS",
    "OPENAI_MODELS",
    "ANTHROPIC_MODELS",
    "GEMINI_MODELS",
    "DEEPSEEK_MODELS",
    "OLLAMA_MODELS",
    "VOICE_MODELS",

    # Core functions
    "get_model",
    "get_default_model",
    "get_models_by_provider",
    "get_models_by_type",
    "get_models_by_capability",
    "get_provider_for_model",
    "is_model_supported",
    "get_model_info",
    "list_all_models",
    "validate_provider_model",
    "estimate_cost",
    "resolve_model_name",

    # Voice-specific functions
    "get_voice_models",
    "get_streaming_voice_models",
    "get_barge_in_capable_models",
    "get_low_latency_models",

    # API key functions
    "get_api_key",
    "get_provider_config",
    "check_api_keys_available",

    # Legacy functions
    "get_openai_models",
    "get_anthropic_models",
    "get_gemini_models",
    "get_deepseek_models",
    "get_qwen_models",
]
