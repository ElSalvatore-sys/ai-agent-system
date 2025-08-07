"""
Model Platform Integrations
Comprehensive integrations for various model platforms and providers
"""

import asyncio
import aiohttp
import json
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import subprocess
import docker
from huggingface_hub import HfApi, hf_hub_download
import requests

from app.core.integration_framework import (
    BaseIntegration, IntegrationConfig, IntegrationType, IntegrationStatus
)
from app.core.logger import LoggerMixin


class OllamaIntegration(BaseIntegration):
    """Integration with Ollama for local model management"""
    
    def __init__(self, config: IntegrationConfig):
        super().__init__(config)
        self.base_url = config.settings.get("base_url", "http://localhost:11434")
        self.timeout = config.settings.get("timeout", 30)
        self.session: Optional[aiohttp.ClientSession] = None
        self._discovered_models: List[Dict[str, Any]] = []
    
    async def initialize(self) -> bool:
        """Initialize Ollama integration"""
        try:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
            
            # Check if Ollama is running
            await self._check_ollama_status()
            
            # Discover available models
            await self._discover_models()
            
            self.logger.info("Ollama integration initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Ollama integration: {e}")
            return False
    
    async def health_check(self) -> bool:
        """Check Ollama service health"""
        try:
            await self._check_ollama_status()
            return True
        except Exception as e:
            self.logger.error(f"Ollama health check failed: {e}")
            return False
    
    async def shutdown(self) -> None:
        """Shutdown Ollama integration"""
        if self.session:
            await self.session.close()
    
    async def _check_ollama_status(self) -> bool:
        """Check if Ollama service is running"""
        if not self.session:
            raise RuntimeError("Session not initialized")
        
        async with self.session.get(f"{self.base_url}/api/tags") as response:
            if response.status == 200:
                return True
            else:
                raise RuntimeError(f"Ollama service not available: {response.status}")
    
    async def _discover_models(self) -> List[Dict[str, Any]]:
        """Discover available Ollama models"""
        try:
            if not self.session:
                raise RuntimeError("Session not initialized")
            
            async with self.session.get(f"{self.base_url}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    self._discovered_models = data.get("models", [])
                    
                    self.logger.info(f"Discovered {len(self._discovered_models)} Ollama models")
                    return self._discovered_models
                else:
                    raise RuntimeError(f"Failed to discover models: {response.status}")
                    
        except Exception as e:
            self.logger.error(f"Error discovering Ollama models: {e}")
            return []
    
    async def pull_model(self, model_name: str) -> Dict[str, Any]:
        """Pull a model from Ollama"""
        try:
            if not self.session:
                raise RuntimeError("Session not initialized")
            
            pull_data = {"name": model_name}
            
            async with self.session.post(
                f"{self.base_url}/api/pull",
                json=pull_data
            ) as response:
                if response.status == 200:
                    # Stream the pull progress
                    async for line in response.content:
                        if line:
                            status_data = json.loads(line.decode())
                            self.logger.info(f"Pull progress: {status_data}")
                    
                    # Refresh discovered models
                    await self._discover_models()
                    
                    return {"status": "success", "model": model_name}
                else:
                    error_data = await response.json()
                    raise RuntimeError(f"Failed to pull model {model_name}: {error_data}")
                    
        except Exception as e:
            self.logger.error(f"Error pulling model {model_name}: {e}")
            return {"status": "error", "error": str(e)}
    
    async def generate(self, model_name: str, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate text using Ollama model"""
        try:
            if not self.session:
                raise RuntimeError("Session not initialized")
            
            generation_data = {
                "model": model_name,
                "prompt": prompt,
                "stream": kwargs.get("stream", False),
                **kwargs
            }
            
            async with self.session.post(
                f"{self.base_url}/api/generate",
                json=generation_data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return {"status": "success", "response": result}
                else:
                    error_data = await response.json()
                    raise RuntimeError(f"Generation failed: {error_data}")
                    
        except Exception as e:
            self.logger.error(f"Error generating text: {e}")
            return {"status": "error", "error": str(e)}
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models"""
        return self._discovered_models


class HuggingFaceHubIntegration(BaseIntegration):
    """Integration with Hugging Face Hub for model discovery and download"""
    
    def __init__(self, config: IntegrationConfig):
        super().__init__(config)
        self.api_token = config.settings.get("api_token", os.getenv("HUGGINGFACE_TOKEN"))
        self.cache_dir = Path(config.settings.get("cache_dir", "~/.cache/huggingface")).expanduser()
        self.api: Optional[HfApi] = None
        self._model_cache: Dict[str, Any] = {}
    
    async def initialize(self) -> bool:
        """Initialize Hugging Face Hub integration"""
        try:
            self.api = HfApi(token=self.api_token)
            
            # Test connection
            await self._test_connection()
            
            # Create cache directory
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
            self.logger.info("Hugging Face Hub integration initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize HF Hub integration: {e}")
            return False
    
    async def health_check(self) -> bool:
        """Check HF Hub connection health"""
        try:
            await self._test_connection()
            return True
        except Exception as e:
            self.logger.error(f"HF Hub health check failed: {e}")
            return False
    
    async def shutdown(self) -> None:
        """Shutdown HF Hub integration"""
        pass  # No cleanup needed
    
    async def _test_connection(self) -> bool:
        """Test connection to Hugging Face Hub"""
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            whoami = await loop.run_in_executor(None, self.api.whoami)
            return True
        except Exception as e:
            raise RuntimeError(f"HF Hub connection failed: {e}")
    
    async def search_models(
        self,
        query: str = "",
        filter_dict: Optional[Dict[str, Any]] = None,
        sort: str = "downloads",
        direction: int = -1,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Search for models on Hugging Face Hub"""
        try:
            loop = asyncio.get_event_loop()
            
            models = await loop.run_in_executor(
                None,
                lambda: list(self.api.list_models(
                    search=query,
                    filter=filter_dict,
                    sort=sort,
                    direction=direction,
                    limit=limit
                ))
            )
            
            model_list = []
            for model in models:
                model_info = {
                    "id": model.modelId,
                    "author": model.author,
                    "downloads": getattr(model, 'downloads', 0),
                    "likes": getattr(model, 'likes', 0),
                    "tags": getattr(model, 'tags', []),
                    "library_name": getattr(model, 'library_name', None),
                    "created_at": getattr(model, 'created_at', None),
                    "last_modified": getattr(model, 'lastModified', None)
                }
                model_list.append(model_info)
            
            return model_list
            
        except Exception as e:
            self.logger.error(f"Error searching HF models: {e}")
            return []
    
    async def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get detailed information about a model"""
        try:
            # Check cache first
            if model_id in self._model_cache:
                cache_time = self._model_cache[model_id].get("_cached_at", datetime.min)
                if datetime.utcnow() - cache_time < timedelta(hours=1):
                    return self._model_cache[model_id]
            
            loop = asyncio.get_event_loop()
            
            model_info = await loop.run_in_executor(
                None,
                self.api.model_info,
                model_id
            )
            
            result = {
                "id": model_info.modelId,
                "author": model_info.author,
                "downloads": getattr(model_info, 'downloads', 0),
                "likes": getattr(model_info, 'likes', 0),
                "tags": getattr(model_info, 'tags', []),
                "library_name": getattr(model_info, 'library_name', None),
                "pipeline_tag": getattr(model_info, 'pipeline_tag', None),
                "created_at": getattr(model_info, 'created_at', None),
                "last_modified": getattr(model_info, 'lastModified', None),
                "siblings": [{"filename": s.rfilename, "size": getattr(s, 'size', 0)} 
                           for s in getattr(model_info, 'siblings', [])],
                "_cached_at": datetime.utcnow()
            }
            
            # Cache the result
            self._model_cache[model_id] = result
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting model info for {model_id}: {e}")
            return {}
    
    async def download_model(
        self,
        model_id: str,
        filename: Optional[str] = None,
        local_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Download a model from Hugging Face Hub"""
        try:
            loop = asyncio.get_event_loop()
            
            download_path = await loop.run_in_executor(
                None,
                hf_hub_download,
                model_id,
                filename,
                local_dir or str(self.cache_dir),
                self.api_token
            )
            
            return {
                "status": "success",
                "model_id": model_id,
                "download_path": download_path
            }
            
        except Exception as e:
            self.logger.error(f"Error downloading model {model_id}: {e}")
            return {"status": "error", "error": str(e)}


class OpenAICompatibleIntegration(BaseIntegration):
    """Integration with OpenAI-compatible local servers (LocalAI, text-generation-webui)"""
    
    def __init__(self, config: IntegrationConfig):
        super().__init__(config)
        self.base_url = config.settings.get("base_url", "http://localhost:8000")
        self.api_key = config.settings.get("api_key", "not-needed")
        self.timeout = config.settings.get("timeout", 30)
        self.session: Optional[aiohttp.ClientSession] = None
        self._available_models: List[str] = []
    
    async def initialize(self) -> bool:
        """Initialize OpenAI-compatible integration"""
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            self.session = aiohttp.ClientSession(
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
            
            # Check if server is running
            await self._check_server_status()
            
            # Discover available models
            await self._discover_models()
            
            self.logger.info("OpenAI-compatible integration initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI-compatible integration: {e}")
            return False
    
    async def health_check(self) -> bool:
        """Check server health"""
        try:
            await self._check_server_status()
            return True
        except Exception as e:
            self.logger.error(f"OpenAI-compatible health check failed: {e}")
            return False
    
    async def shutdown(self) -> None:
        """Shutdown integration"""
        if self.session:
            await self.session.close()
    
    async def _check_server_status(self) -> bool:
        """Check if server is running"""
        if not self.session:
            raise RuntimeError("Session not initialized")
        
        async with self.session.get(f"{self.base_url}/v1/models") as response:
            if response.status == 200:
                return True
            else:
                raise RuntimeError(f"Server not available: {response.status}")
    
    async def _discover_models(self) -> List[str]:
        """Discover available models"""
        try:
            if not self.session:
                raise RuntimeError("Session not initialized")
            
            async with self.session.get(f"{self.base_url}/v1/models") as response:
                if response.status == 200:
                    data = await response.json()
                    models = data.get("data", [])
                    self._available_models = [model.get("id", "") for model in models]
                    
                    self.logger.info(f"Discovered {len(self._available_models)} models")
                    return self._available_models
                else:
                    raise RuntimeError(f"Failed to discover models: {response.status}")
                    
        except Exception as e:
            self.logger.error(f"Error discovering models: {e}")
            return []
    
    async def generate(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate text using OpenAI-compatible API"""
        try:
            if not self.session:
                raise RuntimeError("Session not initialized")
            
            request_data = {
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                **kwargs
            }
            
            async with self.session.post(
                f"{self.base_url}/v1/completions",
                json=request_data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return {"status": "success", "response": result}
                else:
                    error_data = await response.json()
                    raise RuntimeError(f"Generation failed: {error_data}")
                    
        except Exception as e:
            self.logger.error(f"Error generating text: {e}")
            return {"status": "error", "error": str(e)}
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        return self._available_models


class CustomModelIntegration(BaseIntegration):
    """Integration for custom model formats and runtimes"""
    
    def __init__(self, config: IntegrationConfig):
        super().__init__(config)
        self.models_dir = Path(config.settings.get("models_dir", "./custom_models"))
        self.supported_formats = config.settings.get("supported_formats", [
            ".gguf", ".ggml", ".bin", ".safetensors", ".pt", ".pth"
        ])
        self.runtime_configs = config.settings.get("runtime_configs", {})
        self._registered_models: Dict[str, Dict[str, Any]] = {}
    
    async def initialize(self) -> bool:
        """Initialize custom model integration"""
        try:
            # Create models directory
            self.models_dir.mkdir(parents=True, exist_ok=True)
            
            # Scan for existing models
            await self._scan_models()
            
            self.logger.info("Custom model integration initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize custom model integration: {e}")
            return False
    
    async def health_check(self) -> bool:
        """Check custom model integration health"""
        try:
            # Check if models directory is accessible
            return self.models_dir.exists() and self.models_dir.is_dir()
        except Exception as e:
            self.logger.error(f"Custom model health check failed: {e}")
            return False
    
    async def shutdown(self) -> None:
        """Shutdown custom model integration"""
        pass  # No cleanup needed
    
    async def _scan_models(self) -> None:
        """Scan models directory for custom models"""
        try:
            for model_file in self.models_dir.rglob("*"):
                if model_file.is_file() and model_file.suffix in self.supported_formats:
                    model_info = {
                        "path": str(model_file),
                        "format": model_file.suffix,
                        "size": model_file.stat().st_size,
                        "modified": datetime.fromtimestamp(model_file.stat().st_mtime),
                        "runtime_config": self.runtime_configs.get(model_file.suffix, {})
                    }
                    self._registered_models[model_file.stem] = model_info
            
            self.logger.info(f"Scanned {len(self._registered_models)} custom models")
            
        except Exception as e:
            self.logger.error(f"Error scanning models: {e}")
    
    async def register_model(
        self,
        name: str,
        model_path: str,
        runtime_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Register a custom model"""
        try:
            model_file = Path(model_path)
            
            if not model_file.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            if model_file.suffix not in self.supported_formats:
                raise ValueError(f"Unsupported model format: {model_file.suffix}")
            
            model_info = {
                "path": str(model_file),
                "format": model_file.suffix,
                "size": model_file.stat().st_size,
                "modified": datetime.fromtimestamp(model_file.stat().st_mtime),
                "runtime_config": runtime_config or self.runtime_configs.get(model_file.suffix, {})
            }
            
            self._registered_models[name] = model_info
            
            self.logger.info(f"Registered custom model: {name}")
            return {"status": "success", "model": name, "info": model_info}
            
        except Exception as e:
            self.logger.error(f"Error registering model {name}: {e}")
            return {"status": "error", "error": str(e)}
    
    async def load_model(self, name: str) -> Dict[str, Any]:
        """Load a custom model (placeholder implementation)"""
        try:
            if name not in self._registered_models:
                raise ValueError(f"Model not found: {name}")
            
            model_info = self._registered_models[name]
            
            # This would implement actual model loading based on format
            # For now, return success
            return {
                "status": "success",
                "model": name,
                "loaded": True,
                "info": model_info
            }
            
        except Exception as e:
            self.logger.error(f"Error loading model {name}: {e}")
            return {"status": "error", "error": str(e)}
    
    def get_registered_models(self) -> Dict[str, Dict[str, Any]]:
        """Get all registered custom models"""
        return self._registered_models


# Default configurations for integrations
DEFAULT_OLLAMAINTEGRATION_CONFIG = IntegrationConfig(
    name="ollama",
    type=IntegrationType.MODEL_PLATFORM,
    enabled=True,
    priority=90,
    settings={
        "base_url": "http://localhost:11434",
        "timeout": 30
    },
    health_check_interval=60
)

DEFAULT_HUGGINGFACEHUBINTEGRATION_CONFIG = IntegrationConfig(
    name="huggingface_hub",
    type=IntegrationType.MODEL_PLATFORM,
    enabled=True,
    priority=80,
    settings={
        "cache_dir": "~/.cache/huggingface"
    },
    health_check_interval=300  # Check less frequently
)

DEFAULT_OPENAICOMPATIBLEINTEGRATION_CONFIG = IntegrationConfig(
    name="openai_compatible",
    type=IntegrationType.MODEL_PLATFORM,
    enabled=True,
    priority=85,
    settings={
        "base_url": "http://localhost:8000",
        "api_key": "not-needed",
        "timeout": 30
    },
    health_check_interval=60
)

DEFAULT_CUSTOMMODELINTEGRATION_CONFIG = IntegrationConfig(
    name="custom_models",
    type=IntegrationType.MODEL_PLATFORM,
    enabled=True,
    priority=70,
    settings={
        "models_dir": "./custom_models",
        "supported_formats": [".gguf", ".ggml", ".bin", ".safetensors", ".pt", ".pth"],
        "runtime_configs": {
            ".gguf": {"backend": "llama_cpp"},
            ".ggml": {"backend": "llama_cpp"},
            ".bin": {"backend": "transformers"},
            ".safetensors": {"backend": "transformers"}
        }
    },
    health_check_interval=120
)