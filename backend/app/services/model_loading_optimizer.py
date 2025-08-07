"""Model Loading Optimization System.

Enterprise-grade model loading optimization with lazy loading, quantization,
dynamic batching, and warm-up strategies for maximum performance.
"""
from __future__ import annotations

import asyncio
import time
import threading
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import json
import hashlib

from app.core.logger import LoggerMixin
from app.core.config import settings
from app.database.models import ModelProvider


class QuantizationType(str, Enum):
    """Quantization types for model optimization"""
    INT4 = "int4"
    INT8 = "int8"
    FP16 = "fp16"
    DYNAMIC = "dynamic"
    NONE = "none"


class LoadingStrategy(str, Enum):
    """Model loading strategies"""
    LAZY = "lazy"
    EAGER = "eager"
    PRELOAD = "preload"
    ON_DEMAND = "on_demand"


@dataclass
class ModelConfig:
    """Model configuration for optimization"""
    provider: ModelProvider
    model_name: str
    quantization: QuantizationType
    max_batch_size: int
    context_length: int
    memory_requirement_mb: int
    gpu_layers: int
    cpu_threads: int
    loading_strategy: LoadingStrategy
    warm_up_prompts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelInstance:
    """Model instance with optimization metadata"""
    config: ModelConfig
    instance_id: str
    status: str  # loading, ready, unloading, error
    created_at: datetime
    last_used: datetime
    request_count: int
    total_tokens: int
    memory_usage_mb: float
    gpu_memory_mb: float
    warmup_completed: bool
    batch_processor: Optional[Any] = None
    model_handle: Optional[Any] = None
    loading_lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass
class BatchRequest:
    """Batch request for optimization"""
    request_id: str
    prompt: str
    system_prompt: Optional[str]
    temperature: float
    max_tokens: int
    created_at: datetime
    priority: int = 0
    callback: Optional[Callable] = None


class ModelLoadingOptimizer(LoggerMixin):
    """Enterprise model loading optimization system"""
    
    def __init__(self):
        super().__init__()
        self.models: Dict[str, ModelInstance] = {}
        self.model_configs: Dict[str, ModelConfig] = {}
        self.batch_queues: Dict[str, List[BatchRequest]] = {}
        self.loading_semaphore = asyncio.Semaphore(2)  # Limit concurrent loads
        
        # Optimization settings
        self.lazy_unload_timeout = 600  # 10 minutes
        self.batch_timeout = 0.1  # 100ms for batch collection
        self.max_memory_usage = 0.8  # 80% of available memory
        self.quantization_cache_dir = Path("/var/cache/llm-quantized")
        self.warmup_cache_dir = Path("/var/cache/llm-warmup")
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._batch_processor_task: Optional[asyncio.Task] = None
        self._memory_monitor_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        # Statistics
        self.stats = {
            "models_loaded": 0,
            "models_unloaded": 0,
            "quantization_hits": 0,
            "batch_requests_processed": 0,
            "memory_optimizations": 0,
            "warmup_cache_hits": 0
        }
    
    async def initialize(self):
        """Initialize the optimization system"""
        try:
            # Create cache directories
            self.quantization_cache_dir.mkdir(parents=True, exist_ok=True)
            self.warmup_cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Load model configurations
            await self._load_model_configs()
            
            # Start background tasks
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            self._batch_processor_task = asyncio.create_task(self._batch_processor_loop())
            self._memory_monitor_task = asyncio.create_task(self._memory_monitor_loop())
            
            self.logger.info("Model loading optimizer initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize model loading optimizer: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown the optimization system"""
        self.logger.info("Shutting down model loading optimizer")
        self._shutdown_event.set()
        
        # Cancel background tasks
        for task in [self._cleanup_task, self._batch_processor_task, self._memory_monitor_task]:
            if task:
                task.cancel()
        
        # Unload all models
        for model_key in list(self.models.keys()):
            await self._unload_model(model_key)
    
    async def load_model(
        self, 
        provider: ModelProvider, 
        model_name: str,
        config_override: Optional[Dict[str, Any]] = None
    ) -> str:
        """Load a model with optimization"""
        model_key = f"{provider.value}:{model_name}"
        
        # Check if already loaded
        if model_key in self.models and self.models[model_key].status == "ready":
            instance = self.models[model_key]
            instance.last_used = datetime.utcnow()
            return instance.instance_id
        
        async with self.loading_semaphore:
            return await self._load_model_optimized(provider, model_name, config_override)
    
    async def _load_model_optimized(
        self, 
        provider: ModelProvider, 
        model_name: str,
        config_override: Optional[Dict[str, Any]] = None
    ) -> str:
        """Load model with all optimizations applied"""
        model_key = f"{provider.value}:{model_name}"
        
        try:
            # Get model configuration
            config = await self._get_model_config(provider, model_name, config_override)
            
            # Create model instance
            instance_id = f"{model_key}:{int(time.time())}"
            instance = ModelInstance(
                config=config,
                instance_id=instance_id,
                status="loading",
                created_at=datetime.utcnow(),
                last_used=datetime.utcnow(),
                request_count=0,
                total_tokens=0,
                memory_usage_mb=0.0,
                gpu_memory_mb=0.0,
                warmup_completed=False
            )
            
            self.models[model_key] = instance
            
            # Check for cached quantized model
            quantized_path = await self._get_quantized_model_path(config)
            if not quantized_path.exists() and config.quantization != QuantizationType.NONE:
                await self._create_quantized_model(config, quantized_path)
            
            # Load model based on provider
            if provider == ModelProvider.LOCAL_OLLAMA:
                await self._load_ollama_model(instance, quantized_path)
            elif provider == ModelProvider.LOCAL_HF:
                await self._load_hf_model(instance, quantized_path)
            else:
                raise ValueError(f"Unsupported provider: {provider}")
            
            # Initialize batch processor
            if config.max_batch_size > 1:
                await self._initialize_batch_processor(instance)
            
            # Perform warm-up
            await self._warmup_model(instance)
            
            instance.status = "ready"
            self.stats["models_loaded"] += 1
            
            self.logger.info(f"Model {model_key} loaded successfully with optimizations")
            return instance_id
            
        except Exception as e:
            self.logger.error(f"Failed to load model {model_key}: {e}")
            if model_key in self.models:
                self.models[model_key].status = "error"
            raise
    
    async def _get_model_config(
        self, 
        provider: ModelProvider, 
        model_name: str,
        config_override: Optional[Dict[str, Any]] = None
    ) -> ModelConfig:
        """Get optimized model configuration"""
        model_key = f"{provider.value}:{model_name}"
        
        # Use cached config if available
        if model_key in self.model_configs:
            config = self.model_configs[model_key]
        else:
            # Create default config
            config = ModelConfig(
                provider=provider,
                model_name=model_name,
                quantization=QuantizationType.INT8,
                max_batch_size=4,
                context_length=4096,
                memory_requirement_mb=2048,
                gpu_layers=32,
                cpu_threads=4,
                loading_strategy=LoadingStrategy.LAZY,
                warm_up_prompts=["Hello", "How are you?", "What is AI?"]
            )
        
        # Apply overrides
        if config_override:
            for key, value in config_override.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        return config
    
    async def _create_quantized_model(self, config: ModelConfig, output_path: Path):
        """Create quantized version of model"""
        self.logger.info(f"Creating quantized model: {config.model_name} ({config.quantization})")
        
        try:
            if config.provider == ModelProvider.LOCAL_HF:
                await self._quantize_hf_model(config, output_path)
            elif config.provider == ModelProvider.LOCAL_OLLAMA:
                await self._quantize_ollama_model(config, output_path)
            
            self.stats["quantization_hits"] += 1
            
        except Exception as e:
            self.logger.error(f"Failed to create quantized model: {e}")
            raise
    
    async def _quantize_hf_model(self, config: ModelConfig, output_path: Path):
        """Quantize HuggingFace model"""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from optimum.onnxruntime import ORTModelForCausalLM
            
            model_name = config.model_name
            
            if config.quantization == QuantizationType.INT4:
                # Load model with 4-bit quantization
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                
            elif config.quantization == QuantizationType.INT8:
                # Load model with 8-bit quantization
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    load_in_8bit=True
                )
                
            elif config.quantization == QuantizationType.FP16:
                # Load model with FP16
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            
            # Save quantized model
            output_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(output_path))
            
            # Save tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.save_pretrained(str(output_path))
            
        except Exception as e:
            self.logger.error(f"HF model quantization failed: {e}")
            raise
    
    async def _quantize_ollama_model(self, config: ModelConfig, output_path: Path):
        """Quantize Ollama model"""
        try:
            # For Ollama, we create a quantized Modelfile
            modelfile_content = f"""
FROM {config.model_name}

# Quantization parameters
PARAMETER num_gpu {config.gpu_layers}
PARAMETER num_thread {config.cpu_threads}
"""
            
            if config.quantization == QuantizationType.INT4:
                modelfile_content += "PARAMETER quantization Q4_0\n"
            elif config.quantization == QuantizationType.INT8:
                modelfile_content += "PARAMETER quantization Q8_0\n"
            
            output_path.mkdir(parents=True, exist_ok=True)
            (output_path / "Modelfile").write_text(modelfile_content)
            
        except Exception as e:
            self.logger.error(f"Ollama model quantization failed: {e}")
            raise
    
    async def _load_ollama_model(self, instance: ModelInstance, quantized_path: Path):
        """Load Ollama model with optimizations"""
        try:
            import httpx
            
            config = instance.config
            
            # Create optimized model from Modelfile if quantized
            if quantized_path.exists() and (quantized_path / "Modelfile").exists():
                model_name = f"{config.model_name}-optimized"
                
                # Create model from Modelfile
                async with httpx.AsyncClient() as client:
                    modelfile = (quantized_path / "Modelfile").read_text()
                    
                    create_response = await client.post(
                        f"{settings.OLLAMA_HOST}/api/create",
                        json={
                            "name": model_name,
                            "modelfile": modelfile
                        }
                    )
                    create_response.raise_for_status()
                
                instance.model_handle = model_name
            else:
                instance.model_handle = config.model_name
            
            # Test model availability
            async with httpx.AsyncClient() as client:
                test_response = await client.post(
                    f"{settings.OLLAMA_HOST}/api/generate",
                    json={
                        "model": instance.model_handle,
                        "prompt": "test",
                        "stream": False
                    }
                )
                test_response.raise_for_status()
            
        except Exception as e:
            self.logger.error(f"Failed to load Ollama model: {e}")
            raise
    
    async def _load_hf_model(self, instance: ModelInstance, quantized_path: Path):
        """Load HuggingFace model with optimizations"""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            
            config = instance.config
            
            # Load from quantized path if available
            if quantized_path.exists():
                model_path = str(quantized_path)
                self.logger.info(f"Loading quantized model from {model_path}")
            else:
                model_path = config.model_name
            
            # Load model and tokenizer
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Create pipeline for easier inference
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device_map="auto"
            )
            
            instance.model_handle = pipe
            
        except Exception as e:
            self.logger.error(f"Failed to load HF model: {e}")
            raise
    
    async def _initialize_batch_processor(self, instance: ModelInstance):
        """Initialize batch processing for model"""
        model_key = f"{instance.config.provider.value}:{instance.config.model_name}"
        self.batch_queues[model_key] = []
        
        self.logger.info(f"Batch processor initialized for {model_key}")
    
    async def _warmup_model(self, instance: ModelInstance):
        """Perform model warm-up"""
        try:
            config = instance.config
            model_key = f"{config.provider.value}:{config.model_name}"
            
            # Check for cached warmup results
            warmup_cache_path = self.warmup_cache_dir / f"{model_key}.json"
            if warmup_cache_path.exists():
                self.stats["warmup_cache_hits"] += 1
                instance.warmup_completed = True
                return
            
            self.logger.info(f"Warming up model: {model_key}")
            
            warmup_results = []
            
            for prompt in config.warm_up_prompts:
                start_time = time.time()
                
                if config.provider == ModelProvider.LOCAL_OLLAMA:
                    await self._warmup_ollama(instance, prompt)
                elif config.provider == ModelProvider.LOCAL_HF:
                    await self._warmup_hf(instance, prompt)
                
                warmup_time = time.time() - start_time
                warmup_results.append({
                    "prompt": prompt,
                    "time": warmup_time
                })
            
            # Cache warmup results
            warmup_cache_path.write_text(json.dumps({
                "model_key": model_key,
                "warmup_results": warmup_results,
                "timestamp": datetime.utcnow().isoformat()
            }))
            
            instance.warmup_completed = True
            self.logger.info(f"Model {model_key} warmup completed")
            
        except Exception as e:
            self.logger.error(f"Model warmup failed: {e}")
            # Don't fail loading if warmup fails
            instance.warmup_completed = False
    
    async def _warmup_ollama(self, instance: ModelInstance, prompt: str):
        """Warmup Ollama model"""
        import httpx
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.OLLAMA_HOST}/api/generate",
                json={
                    "model": instance.model_handle,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 10
                    }
                }
            )
            response.raise_for_status()
    
    async def _warmup_hf(self, instance: ModelInstance, prompt: str):
        """Warmup HuggingFace model"""
        pipe = instance.model_handle
        
        # Generate a short response for warmup
        pipe(prompt, max_length=20, do_sample=False)
    
    async def _get_quantized_model_path(self, config: ModelConfig) -> Path:
        """Get path for quantized model cache"""
        model_hash = hashlib.md5(
            f"{config.provider.value}:{config.model_name}:{config.quantization}".encode()
        ).hexdigest()
        
        return self.quantization_cache_dir / f"{model_hash}"
    
    async def process_batch_request(
        self, 
        provider: ModelProvider, 
        model_name: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        priority: int = 0
    ) -> str:
        """Process request with batching optimization"""
        model_key = f"{provider.value}:{model_name}"
        
        # Ensure model is loaded
        await self.load_model(provider, model_name)
        
        instance = self.models[model_key]
        config = instance.config
        
        # If batching is not enabled, process immediately
        if config.max_batch_size <= 1:
            return await self._process_single_request(instance, prompt, system_prompt, temperature, max_tokens)
        
        # Add to batch queue
        request_id = f"{model_key}:{int(time.time() * 1000)}"
        batch_request = BatchRequest(
            request_id=request_id,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            created_at=datetime.utcnow(),
            priority=priority
        )
        
        if model_key not in self.batch_queues:
            self.batch_queues[model_key] = []
        
        self.batch_queues[model_key].append(batch_request)
        
        # Wait for batch processing (simplified for now)
        # In production, this would use proper async coordination
        await asyncio.sleep(self.batch_timeout)
        
        return await self._process_single_request(instance, prompt, system_prompt, temperature, max_tokens)
    
    async def _process_single_request(
        self, 
        instance: ModelInstance,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> str:
        """Process single request"""
        try:
            config = instance.config
            
            if config.provider == ModelProvider.LOCAL_OLLAMA:
                return await self._process_ollama_request(instance, prompt, system_prompt, temperature, max_tokens)
            elif config.provider == ModelProvider.LOCAL_HF:
                return await self._process_hf_request(instance, prompt, system_prompt, temperature, max_tokens)
            else:
                raise ValueError(f"Unsupported provider: {config.provider}")
                
        except Exception as e:
            self.logger.error(f"Request processing failed: {e}")
            raise
    
    async def _process_ollama_request(
        self, 
        instance: ModelInstance,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int
    ) -> str:
        """Process Ollama request"""
        import httpx
        
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.OLLAMA_HOST}/api/generate",
                json={
                    "model": instance.model_handle,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                }
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "")
    
    async def _process_hf_request(
        self, 
        instance: ModelInstance,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int
    ) -> str:
        """Process HuggingFace request"""
        pipe = instance.model_handle
        
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        
        result = pipe(
            full_prompt,
            max_length=max_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=pipe.tokenizer.eos_token_id
        )
        
        return result[0]["generated_text"]
    
    async def _batch_processor_loop(self):
        """Background task for processing batches"""
        while not self._shutdown_event.is_set():
            try:
                for model_key, queue in self.batch_queues.items():
                    if len(queue) > 0:
                        await self._process_batch_queue(model_key, queue)
                
                await asyncio.sleep(self.batch_timeout)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Batch processor error: {e}")
                await asyncio.sleep(1)
    
    async def _process_batch_queue(self, model_key: str, queue: List[BatchRequest]):
        """Process batch queue for a model"""
        if not queue:
            return
        
        instance = self.models.get(model_key)
        if not instance or instance.status != "ready":
            return
        
        config = instance.config
        
        # Process up to max_batch_size requests
        batch_size = min(len(queue), config.max_batch_size)
        batch = queue[:batch_size]
        
        try:
            # Process batch (simplified implementation)
            for request in batch:
                await self._process_single_request(
                    instance,
                    request.prompt,
                    request.system_prompt,
                    request.temperature,
                    request.max_tokens
                )
                
                if request.callback:
                    request.callback()
            
            # Remove processed requests
            queue[:batch_size] = []
            
            self.stats["batch_requests_processed"] += batch_size
            
        except Exception as e:
            self.logger.error(f"Batch processing failed for {model_key}: {e}")
    
    async def _cleanup_loop(self):
        """Background cleanup of unused models"""
        while not self._shutdown_event.is_set():
            try:
                current_time = datetime.utcnow()
                
                for model_key, instance in list(self.models.items()):
                    time_since_use = (current_time - instance.last_used).total_seconds()
                    
                    if (time_since_use > self.lazy_unload_timeout and 
                        instance.status == "ready" and
                        instance.config.loading_strategy == LoadingStrategy.LAZY):
                        
                        await self._unload_model(model_key)
                
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(60)
    
    async def _memory_monitor_loop(self):
        """Monitor memory usage and optimize"""
        while not self._shutdown_event.is_set():
            try:
                await self._check_memory_usage()
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Memory monitor error: {e}")
                await asyncio.sleep(30)
    
    async def _check_memory_usage(self):
        """Check and optimize memory usage"""
        try:
            import psutil
            
            # Get memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent / 100
            
            if memory_usage > self.max_memory_usage:
                self.logger.warning(f"High memory usage: {memory_usage:.2%}")
                
                # Find least recently used models to unload
                models_by_usage = sorted(
                    self.models.items(),
                    key=lambda x: x[1].last_used
                )
                
                for model_key, instance in models_by_usage:
                    if memory_usage <= self.max_memory_usage * 0.9:
                        break
                    
                    if instance.config.loading_strategy == LoadingStrategy.LAZY:
                        await self._unload_model(model_key)
                        self.stats["memory_optimizations"] += 1
                        
                        # Re-check memory
                        memory = psutil.virtual_memory()
                        memory_usage = memory.percent / 100
                        
        except Exception as e:
            self.logger.error(f"Memory check failed: {e}")
    
    async def _unload_model(self, model_key: str):
        """Unload a model from memory"""
        try:
            if model_key not in self.models:
                return
            
            instance = self.models[model_key]
            instance.status = "unloading"
            
            # Cleanup model handle
            if instance.model_handle:
                instance.model_handle = None
            
            # Remove from tracking
            del self.models[model_key]
            
            # Cleanup batch queue
            if model_key in self.batch_queues:
                del self.batch_queues[model_key]
            
            self.stats["models_unloaded"] += 1
            self.logger.info(f"Model {model_key} unloaded")
            
        except Exception as e:
            self.logger.error(f"Failed to unload model {model_key}: {e}")
    
    async def _load_model_configs(self):
        """Load model configurations from file or database"""
        try:
            # This would typically load from a configuration file or database
            # For now, we'll use default configurations
            
            default_configs = {
                "local_ollama:llama2": ModelConfig(
                    provider=ModelProvider.LOCAL_OLLAMA,
                    model_name="llama2",
                    quantization=QuantizationType.INT8,
                    max_batch_size=4,
                    context_length=4096,
                    memory_requirement_mb=2048,
                    gpu_layers=32,
                    cpu_threads=4,
                    loading_strategy=LoadingStrategy.LAZY
                ),
                "local_ollama:codellama": ModelConfig(
                    provider=ModelProvider.LOCAL_OLLAMA,
                    model_name="codellama",
                    quantization=QuantizationType.INT4,
                    max_batch_size=2,
                    context_length=8192,
                    memory_requirement_mb=4096,
                    gpu_layers=48,
                    cpu_threads=6,
                    loading_strategy=LoadingStrategy.PRELOAD
                )
            }
            
            self.model_configs.update(default_configs)
            
        except Exception as e:
            self.logger.error(f"Failed to load model configs: {e}")
    
    async def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        return {
            "stats": self.stats,
            "loaded_models": len(self.models),
            "ready_models": len([m for m in self.models.values() if m.status == "ready"]),
            "batch_queues": {k: len(v) for k, v in self.batch_queues.items()},
            "cache_info": {
                "quantization_cache_size": len(list(self.quantization_cache_dir.glob("*"))),
                "warmup_cache_size": len(list(self.warmup_cache_dir.glob("*")))
            }
        }


# Global instance
_optimizer: Optional[ModelLoadingOptimizer] = None


async def get_model_loading_optimizer() -> ModelLoadingOptimizer:
    """Get the global model loading optimizer instance"""
    global _optimizer
    if _optimizer is None:
        _optimizer = ModelLoadingOptimizer()
        await _optimizer.initialize()
    return _optimizer