"""
Cloud Model Manager for Multi-LLM and Voice Support
====================================================

Production-ready model manager supporting:
- Multi-LLM providers (OpenAI, Anthropic, Gemini, Qwen, DeepSeek, Ollama, HuggingFace)
- Voice models (Qwen3-Omni, ASR, TTS)
- Streaming support with barge-in
- Cloud-agnostic design for any use case
"""

import os
import logging
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, AsyncIterator, Union, List, Callable
from dataclasses import dataclass, field
from enum import Enum

# Import LangChain components with error handling
try:
    from langchain_community.llms import Ollama
except ImportError:
    Ollama = None
    logging.warning("langchain_community not available, Ollama model will not work")

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None
    logging.warning("langchain_openai not available")

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None
    logging.warning("langchain_google_genai not available")

try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    ChatAnthropic = None
    logging.warning("langchain_anthropic not available")

try:
    from langchain_community.llms import HuggingFacePipeline
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
except ImportError:
    HuggingFacePipeline = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    pipeline = None
    logging.warning("langchain_community or transformers not available, HuggingFace models will not work")

from .settings import Settings, ModelProvider

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION DATACLASSES
# ============================================================================

@dataclass
class ModelConfig:
    """Configuration for any model (LLM, Voice, etc.)"""
    provider: ModelProvider
    model_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: int = 120
    streaming: bool = False
    extra_params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_settings(cls, settings: Settings, provider: Optional[ModelProvider] = None) -> "ModelConfig":
        """Create config from application settings"""
        prov = provider or settings.model_provider

        model_map = {
            ModelProvider.OPENAI: (settings.openai_model, settings.openai_api_key, None),
            ModelProvider.ANTHROPIC: (settings.anthropic_model, settings.anthropic_api_key, None),
            ModelProvider.GEMINI: (settings.google_model, settings.google_api_key, None),
            ModelProvider.DEEPSEEK: ("deepseek-chat", os.getenv("DEEPSEEK_API_KEY"), "https://api.deepseek.com/v1"),
            ModelProvider.QWEN: ("qwen-max", os.getenv("QWEN_API_KEY"), "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            ModelProvider.OLLAMA: (settings.ollama_model, None, settings.ollama_base_url),
            ModelProvider.HUGGINGFACE: ("Qwen/Qwen2.5-3B-Instruct", os.getenv("HF_TOKEN"), None),
        }

        model_name, api_key, base_url = model_map.get(prov, ("gpt-4o-mini", None, None))

        return cls(
            provider=prov,
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
        )


@dataclass
class VoiceConfig:
    """Configuration for voice models (ASR, TTS, Omni)"""
    mode: str = "omni"  # "omni", "pipeline"
    omni_model: str = "qwen3-omni"
    asr_model: str = "qwen3-asr"
    tts_model: str = "qwen3-tts"
    llm_model: str = "qwen-turbo"
    api_key: Optional[str] = None
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # Voice behavior
    voice_id: str = "alloy"
    language: str = "en"
    sample_rate: int = 24000

    # Streaming and interruption
    streaming: bool = True
    enable_barge_in: bool = True
    vad_threshold: float = 0.5  # Voice Activity Detection threshold
    silence_duration_ms: int = 500  # Silence before end of utterance
    max_response_sentences: int = 3  # Keep responses short

    # Backchannels
    enable_backchannels: bool = True
    backchannel_phrases: List[str] = field(default_factory=lambda: ["mm-hm", "okay", "got it", "right"])

    @classmethod
    def from_env(cls) -> "VoiceConfig":
        """Create config from environment variables"""
        return cls(
            mode=os.getenv("VOICE_MODE", "omni"),
            omni_model=os.getenv("QWEN_OMNI_MODEL", "qwen3-omni"),
            asr_model=os.getenv("QWEN_ASR_MODEL", "qwen3-asr"),
            tts_model=os.getenv("QWEN_TTS_MODEL", "qwen3-tts"),
            llm_model=os.getenv("QWEN_LLM_MODEL", "qwen-turbo"),
            api_key=os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY"),
            voice_id=os.getenv("VOICE_ID", "alloy"),
            streaming=os.getenv("VOICE_STREAMING", "true").lower() == "true",
            enable_barge_in=os.getenv("VOICE_BARGE_IN", "true").lower() == "true",
        )


# Aliases for backward compatibility
CloudModelConfig = ModelConfig
CodingModelConfig = ModelConfig
SQLModelConfig = ModelConfig
ObservabilityModelConfig = ModelConfig
CostModelConfig = ModelConfig


# ============================================================================
# BASE MODEL MANAGER
# ============================================================================

class BaseModelManager(ABC):
    """Abstract base class for model managers"""

    @abstractmethod
    def get_model(self) -> Any:
        """Get the model instance"""
        pass

    @abstractmethod
    async def ainvoke(self, input: Any, **kwargs) -> Any:
        """Async invocation"""
        pass

    @abstractmethod
    def invoke(self, input: Any, **kwargs) -> Any:
        """Sync invocation"""
        pass


# ============================================================================
# CLOUD MODEL MANAGER (LLM)
# ============================================================================

class CloudModelManager(BaseModelManager):
    """Manager for LLM models across all providers"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self._model = None
        self._model_info = None

    def get_model(self) -> Any:
        """Get the configured model instance"""
        if self._model is None:
            self._model = self._create_model()
        return self._model

    def _create_model(self) -> Any:
        """Create model instance based on configuration"""
        try:
            logger.info(f"Creating model: {self.config.provider.value} - {self.config.model_name}")
            logger.info(f"API key available: {bool(self.config.api_key)}")

            creators = {
                ModelProvider.OLLAMA: self._create_ollama_model,
                ModelProvider.OPENAI: self._create_openai_model,
                ModelProvider.GEMINI: self._create_gemini_model,
                ModelProvider.ANTHROPIC: self._create_anthropic_model,
                ModelProvider.AZURE_OPENAI: self._create_azure_openai_model,
                ModelProvider.DEEPSEEK: self._create_deepseek_model,
                ModelProvider.QWEN: self._create_qwen_model,
                ModelProvider.HUGGINGFACE: self._create_huggingface_model,
            }

            creator = creators.get(self.config.provider)
            if not creator:
                raise ValueError(f"Unsupported provider: {self.config.provider}")

            return creator()

        except Exception as e:
            logger.error(f"Failed to create model {self.config.provider.value}: {e}")
            return self._create_fallback_model()

    def _create_ollama_model(self) -> Any:
        """Create Ollama model"""
        if Ollama is None:
            raise ImportError("langchain_community not available")
        return Ollama(
            model=self.config.model_name,
            base_url=self.config.base_url or "http://localhost:11434",
            temperature=self.config.temperature,
            timeout=self.config.timeout,
            **self.config.extra_params
        )

    def _create_openai_model(self) -> Any:
        """Create OpenAI model"""
        if ChatOpenAI is None:
            raise ImportError("langchain_openai not available")
        return ChatOpenAI(
            model=self.config.model_name,
            api_key=self.config.api_key,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            timeout=self.config.timeout,
            streaming=self.config.streaming,
            **self.config.extra_params
        )

    def _create_gemini_model(self) -> Any:
        """Create Google Gemini model"""
        if ChatGoogleGenerativeAI is None:
            raise ImportError("langchain_google_genai not available")

        if not self.config.api_key:
            raise ValueError("Google API key is required for Gemini model")

        # Clear service account credentials to force API key usage
        old_creds = os.environ.pop('GOOGLE_APPLICATION_CREDENTIALS', None)

        try:
            model = ChatGoogleGenerativeAI(
                model=self.config.model_name,
                google_api_key=self.config.api_key,
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_tokens,
                **self.config.extra_params
            )
            logger.info(f"Successfully created Gemini model: {self.config.model_name}")
            return model
        finally:
            if old_creds:
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = old_creds

    def _create_anthropic_model(self) -> Any:
        """Create Anthropic model"""
        if ChatAnthropic is None:
            raise ImportError("langchain_anthropic not available")

        # Map model names
        model_name = self.config.model_name
        if model_name in ["claude-4.5", "claude-4"]:
            model_name = "claude-3-5-sonnet-20241022"

        return ChatAnthropic(
            model=model_name,
            anthropic_api_key=self.config.api_key,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            timeout=self.config.timeout,
            **self.config.extra_params
        )

    def _create_azure_openai_model(self) -> Any:
        """Create Azure OpenAI model"""
        if ChatOpenAI is None:
            raise ImportError("langchain_openai not available")
        return ChatOpenAI(
            model=self.config.model_name,
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            timeout=self.config.timeout,
            **self.config.extra_params
        )

    def _create_deepseek_model(self) -> Any:
        """Create DeepSeek model using OpenAI-compatible API"""
        if ChatOpenAI is None:
            raise ImportError("langchain_openai not available")
        base_url = self.config.base_url or "https://api.deepseek.com/v1"
        return ChatOpenAI(
            model=self.config.model_name,
            api_key=self.config.api_key,
            base_url=base_url,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            timeout=self.config.timeout,
            **self.config.extra_params
        )

    def _create_qwen_model(self) -> Any:
        """Create Qwen model using OpenAI-compatible API"""
        if ChatOpenAI is None:
            raise ImportError("langchain_openai not available")
        base_url = self.config.base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        return ChatOpenAI(
            model=self.config.model_name,
            api_key=self.config.api_key,
            base_url=base_url,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            timeout=self.config.timeout,
            **self.config.extra_params
        )

    def _create_huggingface_model(self) -> Any:
        """Create HuggingFace local model pipeline"""
        if HuggingFacePipeline is None:
            raise ImportError("langchain_community or transformers not available")

        try:
            model_id = self.config.model_name
            quantization_config = None

            if self.config.extra_params.get('load_in_4bit'):
                from transformers import BitsAndBytesConfig
                import torch
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16
                )
            elif self.config.extra_params.get('load_in_8bit'):
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)

            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )

            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                repetition_penalty=1.1,
                return_full_text=False
            )

            return HuggingFacePipeline(pipeline=pipe)

        except Exception as e:
            logger.error(f"Failed to create HuggingFace pipeline: {e}")
            raise

    def _create_fallback_model(self) -> Any:
        """Create a fallback model when others fail"""
        logger.warning("Using fallback model")

        try:
            from langchain_core.runnables import Runnable
            base_class = Runnable
        except ImportError:
            base_class = object

        class FallbackModel(base_class):
            def invoke(self, input: Any, config: Optional[Dict] = None, **kwargs) -> str:
                if hasattr(input, 'to_string'):
                    query = input.to_string()
                elif isinstance(input, str):
                    query = input
                elif hasattr(input, 'content'):
                    query = input.content
                elif isinstance(input, dict) and 'input' in input:
                    query = input['input']
                else:
                    query = str(input)

                return f"[Fallback] I received your message about: {query[:100]}... Please configure a proper AI model."

            async def ainvoke(self, input: Any, config: Optional[Dict] = None, **kwargs) -> str:
                return self.invoke(input, config, **kwargs)

        return FallbackModel()

    def invoke(self, input: Any, **kwargs) -> Any:
        """Synchronous invocation"""
        model = self.get_model()
        return model.invoke(input, **kwargs)

    async def ainvoke(self, input: Any, **kwargs) -> Any:
        """Asynchronous invocation"""
        model = self.get_model()
        if hasattr(model, 'ainvoke'):
            return await model.ainvoke(input, **kwargs)
        # Fallback to sync in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: model.invoke(input, **kwargs))

    async def astream(self, input: Any, **kwargs) -> AsyncIterator[str]:
        """Async streaming invocation"""
        model = self.get_model()
        if hasattr(model, 'astream'):
            async for chunk in model.astream(input, **kwargs):
                if hasattr(chunk, 'content'):
                    yield chunk.content
                else:
                    yield str(chunk)
        else:
            result = await self.ainvoke(input, **kwargs)
            yield str(result)

    def switch_model(self, new_config: ModelConfig) -> None:
        """Switch to a different model configuration"""
        self.config = new_config
        self._model = None
        self._model_info = None

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        if self._model_info is None:
            self._model_info = {
                "provider": self.config.provider.value,
                "model_name": self.config.model_name,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "streaming": self.config.streaming,
            }
        return self._model_info

    def test_connection(self) -> bool:
        """Test if the model is accessible"""
        try:
            model = self.get_model()
            result = model.invoke("Test connection")
            return bool(result)
        except Exception as e:
            logger.error(f"Model connection test failed: {e}")
            return False

    def get_supported_features(self) -> Dict[str, bool]:
        """Get features supported by the current model"""
        features = {
            "streaming": False,
            "async": True,
            "function_calling": False,
            "system_prompts": True,
            "chat_history": True,
            "vision": False,
            "audio": False,
        }

        if self.config.provider in [ModelProvider.OPENAI, ModelProvider.AZURE_OPENAI]:
            features.update({"streaming": True, "function_calling": True, "vision": True})
        elif self.config.provider == ModelProvider.GEMINI:
            features.update({"streaming": True, "function_calling": True, "vision": True, "audio": True})
        elif self.config.provider == ModelProvider.ANTHROPIC:
            features.update({"streaming": True, "vision": True})
        elif self.config.provider == ModelProvider.QWEN:
            features.update({"streaming": True, "function_calling": True, "audio": True})

        return features


# ============================================================================
# VOICE MODEL MANAGER
# ============================================================================

class VoiceModelManager(BaseModelManager):
    """
    Manager for voice models supporting:
    - Qwen3-Omni (streaming speech-to-speech)
    - Pipeline mode (ASR -> LLM -> TTS)
    - Barge-in (interruption) support
    - Backchannels
    """

    def __init__(self, config: VoiceConfig):
        self.config = config
        self._omni_client = None
        self._asr_client = None
        self._tts_client = None
        self._llm_manager = None
        self._is_speaking = False
        self._interrupted = False
        self._on_interrupt_callback: Optional[Callable] = None

    def get_model(self) -> Any:
        """Get the primary voice model"""
        if self.config.mode == "omni":
            return self._get_omni_client()
        else:
            return self._get_llm_manager()

    def _get_omni_client(self) -> Any:
        """Get Qwen3-Omni client for speech-to-speech"""
        if self._omni_client is None:
            self._omni_client = self._create_omni_client()
        return self._omni_client

    def _get_llm_manager(self) -> CloudModelManager:
        """Get LLM manager for pipeline mode"""
        if self._llm_manager is None:
            config = ModelConfig(
                provider=ModelProvider.QWEN,
                model_name=self.config.llm_model,
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                streaming=True,
            )
            self._llm_manager = CloudModelManager(config)
        return self._llm_manager

    def _create_omni_client(self) -> Any:
        """Create Qwen3-Omni streaming client"""
        try:
            import httpx

            class Qwen3OmniClient:
                def __init__(self, api_key: str, base_url: str, config: VoiceConfig):
                    self.api_key = api_key
                    self.base_url = base_url.rstrip('/')
                    self.config = config
                    self.http_client = httpx.AsyncClient(timeout=60.0)

                async def speech_to_speech_stream(
                    self,
                    audio_stream: AsyncIterator[bytes],
                    system_prompt: str = "",
                    on_text: Optional[Callable[[str], None]] = None,
                    on_audio: Optional[Callable[[bytes], None]] = None,
                ) -> AsyncIterator[Union[str, bytes]]:
                    """
                    Streaming speech-to-speech with Qwen3-Omni.
                    Yields text transcriptions and audio chunks.
                    """
                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    }

                    # Collect audio chunks for processing
                    audio_chunks = []
                    async for chunk in audio_stream:
                        audio_chunks.append(chunk)

                    import base64
                    audio_data = base64.b64encode(b"".join(audio_chunks)).decode()

                    payload = {
                        "model": self.config.omni_model,
                        "messages": [
                            {"role": "system", "content": system_prompt or "You are a helpful voice assistant. Keep responses brief."},
                            {"role": "user", "content": [
                                {"type": "audio", "audio_data": audio_data}
                            ]}
                        ],
                        "stream": True,
                        "response_format": {"type": "audio"},
                    }

                    async with self.http_client.stream(
                        "POST",
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=payload,
                    ) as response:
                        async for line in response.aiter_lines():
                            if line.startswith("data: "):
                                data = line[6:]
                                if data == "[DONE]":
                                    break
                                try:
                                    import json
                                    chunk_data = json.loads(data)
                                    delta = chunk_data.get("choices", [{}])[0].get("delta", {})

                                    # Text content
                                    if "content" in delta:
                                        text = delta["content"]
                                        if on_text:
                                            on_text(text)
                                        yield ("text", text)

                                    # Audio content
                                    if "audio" in delta:
                                        audio_b64 = delta["audio"]
                                        audio_bytes = base64.b64decode(audio_b64)
                                        if on_audio:
                                            on_audio(audio_bytes)
                                        yield ("audio", audio_bytes)

                                except Exception as e:
                                    logger.warning(f"Error parsing stream chunk: {e}")

                async def close(self):
                    await self.http_client.aclose()

            return Qwen3OmniClient(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                config=self.config,
            )

        except Exception as e:
            logger.error(f"Failed to create Qwen3-Omni client: {e}")
            return None

    async def process_audio_stream(
        self,
        audio_stream: AsyncIterator[bytes],
        system_prompt: str = "",
    ) -> AsyncIterator[Union[str, bytes]]:
        """
        Process streaming audio input and return streaming audio/text output.
        Supports barge-in (interruption).
        """
        self._is_speaking = False
        self._interrupted = False

        if self.config.mode == "omni":
            client = self._get_omni_client()
            if client:
                async for item in client.speech_to_speech_stream(
                    audio_stream,
                    system_prompt=system_prompt,
                ):
                    if self._interrupted:
                        logger.info("Response interrupted by barge-in")
                        break
                    yield item
        else:
            # Pipeline mode: ASR -> LLM -> TTS
            async for item in self._pipeline_process(audio_stream, system_prompt):
                if self._interrupted:
                    break
                yield item

    async def _pipeline_process(
        self,
        audio_stream: AsyncIterator[bytes],
        system_prompt: str,
    ) -> AsyncIterator[Union[str, bytes]]:
        """Process audio using ASR -> LLM -> TTS pipeline"""
        # Collect audio
        audio_chunks = []
        async for chunk in audio_stream:
            audio_chunks.append(chunk)

        audio_data = b"".join(audio_chunks)

        # ASR: Convert speech to text
        text_input = await self._asr_transcribe(audio_data)
        if not text_input:
            return

        yield ("transcription", text_input)

        # LLM: Generate response
        llm = self._get_llm_manager()
        response_text = ""

        async for chunk in llm.astream(f"{system_prompt}\n\nUser: {text_input}\nAssistant:"):
            if self._interrupted:
                break
            response_text += chunk
            yield ("text", chunk)

        # TTS: Convert response to speech
        if response_text and not self._interrupted:
            async for audio_chunk in self._tts_synthesize_stream(response_text):
                if self._interrupted:
                    break
                yield ("audio", audio_chunk)

    async def _asr_transcribe(self, audio_data: bytes) -> str:
        """Transcribe audio using Qwen3-ASR or local Whisper"""
        try:
            # Try local Whisper first
            from app.services.ai.agents.voice.utils.stt import STTService
            stt = STTService(model_size="base")
            return stt.transcribe(audio_data)
        except Exception as e:
            logger.warning(f"Local ASR failed, trying Qwen3-ASR: {e}")

            # Fallback to Qwen3-ASR API
            try:
                import httpx
                import base64

                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self.config.base_url}/services/asr",
                        headers={"Authorization": f"Bearer {self.config.api_key}"},
                        json={
                            "model": self.config.asr_model,
                            "audio": base64.b64encode(audio_data).decode(),
                        }
                    )
                    result = response.json()
                    return result.get("text", "")
            except Exception as api_error:
                logger.error(f"ASR failed: {api_error}")
                return ""

    async def _tts_synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        """Stream TTS synthesis"""
        try:
            # Try local Kokoro TTS first
            from app.services.ai.agents.voice.utils.tts import TTSService
            tts = TTSService()
            audio_bytes = tts.synthesize(text)
            if audio_bytes:
                yield audio_bytes
                return
        except Exception as e:
            logger.warning(f"Local TTS failed, trying Qwen3-TTS: {e}")

        # Fallback to Qwen3-TTS API
        try:
            import httpx

            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST",
                    f"{self.config.base_url}/services/tts",
                    headers={"Authorization": f"Bearer {self.config.api_key}"},
                    json={
                        "model": self.config.tts_model,
                        "text": text,
                        "voice": self.config.voice_id,
                        "stream": True,
                    }
                ) as response:
                    async for chunk in response.aiter_bytes():
                        yield chunk
        except Exception as e:
            logger.error(f"TTS streaming failed: {e}")

    def interrupt(self) -> None:
        """Interrupt current speech output (barge-in)"""
        if self.config.enable_barge_in:
            self._interrupted = True
            logger.info("Barge-in triggered - interrupting response")
            if self._on_interrupt_callback:
                self._on_interrupt_callback()

    def reset_interrupt(self) -> None:
        """Reset interrupt state for new utterance"""
        self._interrupted = False

    def set_interrupt_callback(self, callback: Callable) -> None:
        """Set callback for when interrupt occurs"""
        self._on_interrupt_callback = callback

    def get_backchannel(self) -> str:
        """Get a random backchannel phrase"""
        if self.config.enable_backchannels:
            import random
            return random.choice(self.config.backchannel_phrases)
        return ""

    def invoke(self, input: Any, **kwargs) -> Any:
        """Synchronous invocation (for text-only mode)"""
        if self.config.mode == "omni":
            raise NotImplementedError("Omni mode requires async streaming")
        return self._get_llm_manager().invoke(input, **kwargs)

    async def ainvoke(self, input: Any, **kwargs) -> Any:
        """Async invocation for text input"""
        llm = self._get_llm_manager()
        return await llm.ainvoke(input, **kwargs)


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

# Aliases for backward compatibility
ModelManager = CloudModelManager
ObservabilityModelManager = CloudModelManager
CostModelManager = CloudModelManager
SQLModelManager = CloudModelManager


def get_model_manager(model_name: str = 'gemini-2.0-flash') -> CloudModelManager:
    """
    Factory function to get model manager with specified model.
    Used by agents for convenience.
    """
    from .model_registry import resolve_model_name, get_provider_for_model, MODEL_NAME_MAPPING

    # Resolve aliases
    actual_model_name = resolve_model_name(model_name)

    # Determine provider
    provider = None
    model_lower = actual_model_name.lower()

    if 'gpt' in model_lower:
        provider = ModelProvider.OPENAI
    elif 'claude' in model_lower:
        provider = ModelProvider.ANTHROPIC
    elif 'gemini' in model_lower:
        provider = ModelProvider.GEMINI
    elif 'deepseek' in model_lower:
        provider = ModelProvider.DEEPSEEK
    elif 'qwen' in model_lower:
        provider = ModelProvider.QWEN
    elif 'llama' in model_lower or 'mistral' in model_lower:
        provider = ModelProvider.OLLAMA
    elif '/' in model_lower:  # HuggingFace format
        provider = ModelProvider.HUGGINGFACE
    else:
        provider = ModelProvider.GEMINI  # Default

    # Get API key
    api_key_map = {
        ModelProvider.OPENAI: os.getenv("OPENAI_API_KEY"),
        ModelProvider.ANTHROPIC: os.getenv("ANTHROPIC_API_KEY"),
        ModelProvider.GEMINI: os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
        ModelProvider.DEEPSEEK: os.getenv("DEEPSEEK_API_KEY"),
        ModelProvider.QWEN: os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY"),
        ModelProvider.HUGGINGFACE: os.getenv("HF_TOKEN"),
    }

    config = ModelConfig(
        provider=provider,
        model_name=actual_model_name,
        api_key=api_key_map.get(provider),
        temperature=0.7,
        max_tokens=4096,
    )

    return CloudModelManager(config)


def get_voice_model_manager(mode: str = "omni") -> VoiceModelManager:
    """Factory function to get voice model manager"""
    config = VoiceConfig.from_env()
    config.mode = mode
    return VoiceModelManager(config)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Config classes
    "ModelConfig",
    "VoiceConfig",
    "CloudModelConfig",
    "CodingModelConfig",
    "SQLModelConfig",
    "ObservabilityModelConfig",
    "CostModelConfig",

    # Manager classes
    "BaseModelManager",
    "CloudModelManager",
    "VoiceModelManager",
    "ModelManager",

    # Factory functions
    "get_model_manager",
    "get_voice_model_manager",

    # Aliases
    "ObservabilityModelManager",
    "CostModelManager",
    "SQLModelManager",
]
