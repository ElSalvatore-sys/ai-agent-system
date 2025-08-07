import asyncio
import time
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass
from enum import Enum
import openai
import anthropic
import google.generativeai as genai
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logger import LoggerMixin
from app.database.models import ModelProvider, AIModel, UsageLog
from app.services.cost_optimizer import CostOptimizer

class ModelCapability(str, Enum):
    TEXT_GENERATION = "text_generation"
    CODE_GENERATION = "code_generation"
    ANALYSIS = "analysis"
    CREATIVE_WRITING = "creative_writing"
    REASONING = "reasoning"
    FUNCTION_CALLING = "function_calling"
    VISION = "vision"
    STREAMING = "streaming"

@dataclass
class ModelRequest:
    """Request object for AI model calls"""
    prompt: str
    system_prompt: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2048
    stream: bool = False
    model_preference: Optional[str] = None
    provider_preference: Optional[ModelProvider] = None
    capabilities_required: List[ModelCapability] = None
    user_id: Optional[int] = None
    conversation_id: Optional[int] = None

@dataclass
class ModelResponse:
    """Response object from AI model calls"""
    content: str
    model_used: str
    provider: ModelProvider
    tokens_used: int
    cost: float
    response_time: float
    finish_reason: Optional[str] = None
    metadata: Dict[str, Any] = None

class ModelClient:
    """Base class for AI model clients"""
    
    def __init__(self, provider: ModelProvider):
        self.provider = provider
        self.is_available = False
        self.last_error = None
    
    async def initialize(self) -> bool:
        """Initialize the client"""
        raise NotImplementedError
    
    async def generate(self, request: ModelRequest) -> ModelResponse:
        """Generate response"""
        raise NotImplementedError
    
    async def stream_generate(self, request: ModelRequest) -> AsyncGenerator[str, None]:
        """Stream response generation"""
        raise NotImplementedError
    
    async def health_check(self) -> Dict[str, Any]:
        """Check if the model is healthy"""
        raise NotImplementedError

class OpenAIClient(ModelClient, LoggerMixin):
    """OpenAI API client"""
    
    def __init__(self):
        super().__init__(ModelProvider.OPENAI)
        self.client = None
        self.models = {
            "gpt-4": {"input_cost": 0.03, "output_cost": 0.06, "context": 8192},
            "gpt-4-turbo": {"input_cost": 0.01, "output_cost": 0.03, "context": 128000},
            "gpt-3.5-turbo": {"input_cost": 0.0015, "output_cost": 0.002, "context": 16385},
            "o3-mini": {"input_cost": 0.002, "output_cost": 0.008, "context": 128000},
        }
    
    async def initialize(self) -> bool:
        """Initialize OpenAI client"""
        try:
            if not settings.OPENAI_API_KEY:
                self.logger.warning("OpenAI API key not configured")
                return False
            
            self.client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
            
            # Test the connection
            await self.health_check()
            self.is_available = True
            self.logger.info("OpenAI client initialized successfully")
            return True
            
        except Exception as e:
            self.last_error = str(e)
            self.logger.error(f"Failed to initialize OpenAI client: {e}")
            return False
    
    async def generate(self, request: ModelRequest) -> ModelResponse:
        """Generate response using OpenAI"""
        if not self.is_available:
            raise RuntimeError("OpenAI client is not available")
        
        start_time = time.time()
        model = request.model_preference or settings.DEFAULT_MODEL_NAME
        
        try:
            messages = []
            if request.system_prompt:
                messages.append({"role": "system", "content": request.system_prompt})
            messages.append({"role": "user", "content": request.prompt})
            
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=False
            )
            
            response_time = time.time() - start_time
            tokens_used = response.usage.total_tokens
            cost = self._calculate_cost(model, response.usage.prompt_tokens, response.usage.completion_tokens)
            
            return ModelResponse(
                content=response.choices[0].message.content,
                model_used=model,
                provider=self.provider,
                tokens_used=tokens_used,
                cost=cost,
                response_time=response_time,
                finish_reason=response.choices[0].finish_reason,
                metadata={"usage": response.usage.model_dump()}
            )
            
        except Exception as e:
            self.logger.error(f"OpenAI generation failed: {e}")
            raise
    
    async def stream_generate(self, request: ModelRequest) -> AsyncGenerator[str, None]:
        """Stream response generation"""
        if not self.is_available:
            raise RuntimeError("OpenAI client is not available")
        
        model = request.model_preference or settings.DEFAULT_MODEL_NAME
        
        try:
            messages = []
            if request.system_prompt:
                messages.append({"role": "system", "content": request.system_prompt})
            messages.append({"role": "user", "content": request.prompt})
            
            stream = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            self.logger.error(f"OpenAI streaming failed: {e}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Check OpenAI API health"""
        try:
            response = await self.client.models.list()
            return {"status": "healthy", "models_available": len(response.data)}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for OpenAI model usage"""
        model_info = self.models.get(model, self.models["gpt-3.5-turbo"])
        input_cost = (input_tokens / 1000) * model_info["input_cost"]
        output_cost = (output_tokens / 1000) * model_info["output_cost"]
        return input_cost + output_cost

class AnthropicClient(ModelClient, LoggerMixin):
    """Anthropic Claude API client"""
    
    def __init__(self):
        super().__init__(ModelProvider.ANTHROPIC)
        self.client = None
        self.models = {
            "claude-3-5-sonnet-20241022": {"input_cost": 0.003, "output_cost": 0.015, "context": 200000},
            "claude-3-haiku-20240307": {"input_cost": 0.00025, "output_cost": 0.00125, "context": 200000},
            "claude-3-opus-20240229": {"input_cost": 0.015, "output_cost": 0.075, "context": 200000},
        }
    
    async def initialize(self) -> bool:
        """Initialize Anthropic client"""
        try:
            if not settings.ANTHROPIC_API_KEY:
                self.logger.warning("Anthropic API key not configured")
                return False
            
            self.client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
            self.is_available = True
            self.logger.info("Anthropic client initialized successfully")
            return True
            
        except Exception as e:
            self.last_error = str(e)
            self.logger.error(f"Failed to initialize Anthropic client: {e}")
            return False
    
    async def generate(self, request: ModelRequest) -> ModelResponse:
        """Generate response using Anthropic Claude"""
        if not self.is_available:
            raise RuntimeError("Anthropic client is not available")
        
        start_time = time.time()
        model = request.model_preference or "claude-3-5-sonnet-20241022"
        
        try:
            system_prompt = request.system_prompt or "You are a helpful AI assistant."
            
            response = await self.client.messages.create(
                model=model,
                system=system_prompt,
                messages=[{"role": "user", "content": request.prompt}],
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
            
            response_time = time.time() - start_time
            tokens_used = response.usage.input_tokens + response.usage.output_tokens
            cost = self._calculate_cost(model, response.usage.input_tokens, response.usage.output_tokens)
            
            return ModelResponse(
                content=response.content[0].text,
                model_used=model,
                provider=self.provider,
                tokens_used=tokens_used,
                cost=cost,
                response_time=response_time,
                finish_reason=response.stop_reason,
                metadata={"usage": response.usage.model_dump()}
            )
            
        except Exception as e:
            self.logger.error(f"Anthropic generation failed: {e}")
            raise
    
    async def stream_generate(self, request: ModelRequest) -> AsyncGenerator[str, None]:
        """Stream response generation"""
        if not self.is_available:
            raise RuntimeError("Anthropic client is not available")
        
        model = request.model_preference or "claude-3-5-sonnet-20241022"
        system_prompt = request.system_prompt or "You are a helpful AI assistant."
        
        try:
            async with self.client.messages.stream(
                model=model,
                system=system_prompt,
                messages=[{"role": "user", "content": request.prompt}],
                temperature=request.temperature,
                max_tokens=request.max_tokens
            ) as stream:
                async for text in stream.text_stream:
                    yield text
                    
        except Exception as e:
            self.logger.error(f"Anthropic streaming failed: {e}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Anthropic API health"""
        try:
            # Simple test request
            response = await self.client.messages.create(
                model="claude-3-haiku-20240307",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            return {"status": "healthy"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for Anthropic model usage"""
        model_info = self.models.get(model, self.models["claude-3-haiku-20240307"])
        input_cost = (input_tokens / 1000) * model_info["input_cost"]
        output_cost = (output_tokens / 1000) * model_info["output_cost"]
        return input_cost + output_cost

class GoogleClient(ModelClient, LoggerMixin):
    """Google Gemini API client"""
    
    def __init__(self):
        super().__init__(ModelProvider.GOOGLE)
        self.models = {
            "gemini-1.5-pro": {"input_cost": 0.001, "output_cost": 0.003, "context": 1000000},
            "gemini-1.5-flash": {"input_cost": 0.00005, "output_cost": 0.00015, "context": 1000000},
        }
    
    async def initialize(self) -> bool:
        """Initialize Google client"""
        try:
            if not settings.GOOGLE_API_KEY:
                self.logger.warning("Google API key not configured")
                return False
            
            genai.configure(api_key=settings.GOOGLE_API_KEY)
            self.is_available = True
            self.logger.info("Google client initialized successfully")
            return True
            
        except Exception as e:
            self.last_error = str(e)
            self.logger.error(f"Failed to initialize Google client: {e}")
            return False
    
    async def generate(self, request: ModelRequest) -> ModelResponse:
        """Generate response using Google Gemini"""
        if not self.is_available:
            raise RuntimeError("Google client is not available")
        
        start_time = time.time()
        model_name = request.model_preference or "gemini-1.5-flash"
        
        try:
            model = genai.GenerativeModel(model_name)
            
            prompt = request.prompt
            if request.system_prompt:
                prompt = f"System: {request.system_prompt}\n\nUser: {prompt}"
            
            response = await model.generate_content_async(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=request.temperature,
                    max_output_tokens=request.max_tokens,
                )
            )
            
            response_time = time.time() - start_time
            tokens_used = response.usage_metadata.total_token_count if response.usage_metadata else 0
            cost = self._calculate_cost(model_name, tokens_used, 0)  # Google doesn't separate input/output tokens
            
            return ModelResponse(
                content=response.text,
                model_used=model_name,
                provider=self.provider,
                tokens_used=tokens_used,
                cost=cost,
                response_time=response_time,
                finish_reason=response.candidates[0].finish_reason.name if response.candidates else None,
                metadata={"usage": response.usage_metadata._pb if response.usage_metadata else {}}
            )
            
        except Exception as e:
            self.logger.error(f"Google generation failed: {e}")
            raise
    
    async def stream_generate(self, request: ModelRequest) -> AsyncGenerator[str, None]:
        """Stream response generation"""
        if not self.is_available:
            raise RuntimeError("Google client is not available")
        
        model_name = request.model_preference or "gemini-1.5-flash"
        
        try:
            model = genai.GenerativeModel(model_name)
            
            prompt = request.prompt
            if request.system_prompt:
                prompt = f"System: {request.system_prompt}\n\nUser: {prompt}"
            
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=request.temperature,
                    max_output_tokens=request.max_tokens,
                ),
                stream=True
            )
            
            for chunk in response:
                if chunk.text:
                    yield chunk.text
                    
        except Exception as e:
            self.logger.error(f"Google streaming failed: {e}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Google API health"""
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = await model.generate_content_async("Hello")
            return {"status": "healthy"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    def _calculate_cost(self, model: str, total_tokens: int, output_tokens: int) -> float:
        """Calculate cost for Google model usage"""
        model_info = self.models.get(model, self.models["gemini-1.5-flash"])
        # Google pricing is often per character or request, simplified here
        return (total_tokens / 1000) * model_info["input_cost"]

class AIOrchestrator(LoggerMixin):
    """Main orchestrator for AI model routing and optimization"""
    
    def __init__(self):
        self.clients: Dict[ModelProvider, ModelClient] = {}
        self.cost_optimizer = CostOptimizer()
        self.initialized = False
    
    async def initialize(self):
        """Initialize all AI model clients"""
        self.logger.info("Initializing AI Orchestrator...")
        
        # Initialize clients
        from app.models.local_llm_clients import OllamaClient, HFLocalClient
        clients = [
            OpenAIClient(),
            AnthropicClient(),
            GoogleClient(),
            OllamaClient(),
            HFLocalClient(),
        ]
        
        initialization_tasks = [client.initialize() for client in clients]
        results = await asyncio.gather(*initialization_tasks, return_exceptions=True)
        
        for client, result in zip(clients, results):
            if isinstance(result, Exception):
                self.logger.error(f"Failed to initialize {client.provider}: {result}")
            elif result:
                self.clients[client.provider] = client
                self.logger.info(f"Successfully initialized {client.provider}")
        
        if not self.clients:
            self.logger.warning("No AI model clients were successfully initialized - API features will be disabled")
            # Don't raise error, just continue without AI clients
        
        self.initialized = True
        self.logger.info(f"AI Orchestrator initialized with {len(self.clients)} providers")
    
    async def generate(self, request: ModelRequest) -> ModelResponse:
        """Generate response using the optimal model"""
        if not self.initialized:
            raise RuntimeError("AI Orchestrator not initialized")
        
        # Select optimal client
        client = await self._select_optimal_client(request)
        
        # Generate response
        response = await client.generate(request)
        
        # Log usage for cost tracking
        await self._log_usage(request, response)
        
        return response
    
    async def stream_generate(self, request: ModelRequest) -> AsyncGenerator[str, None]:
        """Stream response generation using optimal model"""
        if not self.initialized:
            raise RuntimeError("AI Orchestrator not initialized")
        
        # Select optimal client
        client = await self._select_optimal_client(request)
        
        # Stream response
        async for chunk in client.stream_generate(request):
            yield chunk
    
    async def _select_optimal_client(self, request: ModelRequest) -> ModelClient:
        """Select the optimal client based on request requirements"""
        
        # If specific provider requested, use it
        if request.provider_preference and request.provider_preference in self.clients:
            return self.clients[request.provider_preference]
        
        # If specific model requested, find provider
        if request.model_preference:
            for provider, client in self.clients.items():
                if hasattr(client, 'models') and request.model_preference in client.models:
                    return client
        
        # Use cost optimizer to select best option
        available_providers = list(self.clients.keys())
        optimal_provider = await self.cost_optimizer.select_optimal_provider(
            available_providers, request
        )
        
        return self.clients.get(optimal_provider, next(iter(self.clients.values())))
    
    async def _log_usage(self, request: ModelRequest, response: ModelResponse):
        """Log usage for cost tracking and analytics"""
        if not settings.COST_TRACKING_ENABLED:
            return
        
        # This would typically save to database
        # For now, just log the usage
        self.logger.info(
            f"Usage: {response.provider.value} {response.model_used} - "
            f"Tokens: {response.tokens_used}, Cost: ${response.cost:.4f}, "
            f"Time: {response.response_time:.2f}s"
        )
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of all AI model clients"""
        health_status = {"status": "healthy", "providers": {}}
        
        for provider, client in self.clients.items():
            try:
                client_health = await client.health_check()
                health_status["providers"][provider.value] = client_health
            except Exception as e:
                health_status["providers"][provider.value] = {
                    "status": "unhealthy", 
                    "error": str(e)
                }
                health_status["status"] = "degraded"
        
        if not any(
            p.get("status") == "healthy" 
            for p in health_status["providers"].values()
        ):
            health_status["status"] = "unhealthy"
        
        return health_status
    
    def get_available_providers(self) -> List[str]:
        """Get list of available providers"""
        return [provider.value for provider in self.clients.keys()]
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """Get available models by provider"""
        models = {}
        for provider, client in self.clients.items():
            if hasattr(client, 'models'):
                models[provider.value] = list(client.models.keys())
        return models