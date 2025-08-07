"""Unified API Gateway for Local and Cloud LLM Models.

Provides a unified interface with intelligent routing, load balancing,
caching, and failover capabilities for both local and cloud models.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, AsyncGenerator
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum

from app.core.logger import LoggerMixin
from app.database.models import ModelProvider
from app.models.advanced_ai_orchestrator import RoutingStrategy, RoutingDecision
from app.services.redis_cache import get_cache_manager, get_response_cache


logger = logging.getLogger(__name__)


class LoadBalanceStrategy(str, Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    RESPONSE_TIME = "response_time"
    HEALTH_BASED = "health_based"


@dataclass
class ModelEndpoint:
    """Model endpoint configuration"""
    id: str
    provider: ModelProvider
    model_name: str
    endpoint_url: str
    health_check_url: str
    weight: float
    max_connections: int
    current_connections: int
    avg_response_time: float
    health_score: float
    last_health_check: datetime
    is_active: bool
    error_count: int
    success_count: int
    total_requests: int


@dataclass
class GatewayRequest:
    """Unified request format"""
    prompt: str
    system_prompt: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2048
    stream: bool = False
    model_preference: Optional[str] = None
    provider_preference: Optional[ModelProvider] = None
    user_id: Optional[int] = None
    conversation_id: Optional[int] = None
    routing_strategy: RoutingStrategy = RoutingStrategy.BALANCED
    use_cache: bool = True
    timeout: float = 30.0
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class GatewayResponse:
    """Unified response format"""
    content: str
    model_used: str
    provider: ModelProvider
    endpoint_id: str
    tokens_used: int
    cost: float
    response_time: float
    cached: bool = False
    attempts: int = 1
    fallback_used: bool = False
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class GatewayStats:
    """Gateway statistics"""
    total_requests: int
    successful_requests: int
    failed_requests: int
    cached_responses: int
    avg_response_time: float
    total_tokens: int
    total_cost: float
    active_endpoints: int
    cache_hit_rate: float
    error_rate: float


class LoadBalancer(LoggerMixin):
    """Load balancer for model endpoints"""
    
    def __init__(self, strategy: LoadBalanceStrategy = LoadBalanceStrategy.HEALTH_BASED):
        super().__init__()
        self.strategy = strategy
        self.endpoints: Dict[str, ModelEndpoint] = {}
        self.round_robin_index = 0
    
    def add_endpoint(self, endpoint: ModelEndpoint):
        """Add a model endpoint"""
        self.endpoints[endpoint.id] = endpoint
        self.logger.info(f"Added endpoint: {endpoint.id} ({endpoint.provider.value}:{endpoint.model_name})")
    
    def remove_endpoint(self, endpoint_id: str):
        """Remove a model endpoint"""
        if endpoint_id in self.endpoints:
            del self.endpoints[endpoint_id]
            self.logger.info(f"Removed endpoint: {endpoint_id}")
    
    async def select_endpoint(
        self, 
        provider_preference: Optional[ModelProvider] = None,
        model_preference: Optional[str] = None
    ) -> Optional[ModelEndpoint]:
        """Select the best endpoint based on strategy"""
        # Filter endpoints based on preferences
        available_endpoints = [
            ep for ep in self.endpoints.values()
            if ep.is_active and ep.health_score > 0.5
        ]
        
        if provider_preference:
            available_endpoints = [
                ep for ep in available_endpoints
                if ep.provider == provider_preference
            ]
        
        if model_preference:
            available_endpoints = [
                ep for ep in available_endpoints
                if ep.model_name == model_preference
            ]
        
        if not available_endpoints:
            return None
        
        # Apply load balancing strategy
        if self.strategy == LoadBalanceStrategy.ROUND_ROBIN:
            return self._round_robin_select(available_endpoints)
        elif self.strategy == LoadBalanceStrategy.LEAST_CONNECTIONS:
            return self._least_connections_select(available_endpoints)
        elif self.strategy == LoadBalanceStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_select(available_endpoints)
        elif self.strategy == LoadBalanceStrategy.RESPONSE_TIME:
            return self._response_time_select(available_endpoints)
        elif self.strategy == LoadBalanceStrategy.HEALTH_BASED:
            return self._health_based_select(available_endpoints)
        else:
            return available_endpoints[0]
    
    def _round_robin_select(self, endpoints: List[ModelEndpoint]) -> ModelEndpoint:
        """Round robin selection"""
        endpoint = endpoints[self.round_robin_index % len(endpoints)]
        self.round_robin_index += 1
        return endpoint
    
    def _least_connections_select(self, endpoints: List[ModelEndpoint]) -> ModelEndpoint:
        """Select endpoint with least connections"""
        return min(endpoints, key=lambda ep: ep.current_connections)
    
    def _weighted_round_robin_select(self, endpoints: List[ModelEndpoint]) -> ModelEndpoint:
        """Weighted round robin selection"""
        total_weight = sum(ep.weight for ep in endpoints)
        if total_weight == 0:
            return endpoints[0]
        
        # Simple weighted selection
        weights = [ep.weight / total_weight for ep in endpoints]
        import random
        return random.choices(endpoints, weights=weights)[0]
    
    def _response_time_select(self, endpoints: List[ModelEndpoint]) -> ModelEndpoint:
        """Select endpoint with best response time"""
        return min(endpoints, key=lambda ep: ep.avg_response_time)
    
    def _health_based_select(self, endpoints: List[ModelEndpoint]) -> ModelEndpoint:
        """Select endpoint based on health score and load"""
        def score_endpoint(ep: ModelEndpoint) -> float:
            # Combine health score, connection load, and response time
            load_factor = 1.0 - (ep.current_connections / max(ep.max_connections, 1))
            response_factor = 1.0 / (1.0 + ep.avg_response_time)
            return ep.health_score * load_factor * response_factor
        
        return max(endpoints, key=score_endpoint)
    
    async def update_endpoint_stats(
        self, 
        endpoint_id: str, 
        response_time: float, 
        success: bool
    ):
        """Update endpoint statistics"""
        if endpoint_id in self.endpoints:
            endpoint = self.endpoints[endpoint_id]
            
            # Update response time (moving average)
            alpha = 0.1  # Smoothing factor
            endpoint.avg_response_time = (
                alpha * response_time + (1 - alpha) * endpoint.avg_response_time
            )
            
            # Update counters
            endpoint.total_requests += 1
            if success:
                endpoint.success_count += 1
            else:
                endpoint.error_count += 1
            
            # Update health score based on recent performance
            error_rate = endpoint.error_count / endpoint.total_requests
            endpoint.health_score = max(0.0, min(1.0, 1.0 - error_rate))


class APIGateway(LoggerMixin):
    """Unified API Gateway for LLM models"""
    
    def __init__(self):
        super().__init__()
        self.load_balancer = LoadBalancer()
        self.cache_manager = None
        self.response_cache = None
        
        # Gateway configuration
        self.max_retries = 3
        self.retry_delay = 1.0
        self.circuit_breaker_threshold = 0.5
        self.health_check_interval = 60
        
        # Statistics
        self.stats = GatewayStats(
            total_requests=0,
            successful_requests=0,
            failed_requests=0,
            cached_responses=0,
            avg_response_time=0.0,
            total_tokens=0,
            total_cost=0.0,
            active_endpoints=0,
            cache_hit_rate=0.0,
            error_rate=0.0
        )
        
        # Background tasks
        self._health_check_task = None
        self._stats_update_task = None
        self._shutdown_event = asyncio.Event()
    
    async def initialize(self):
        """Initialize the API gateway"""
        try:
            # Initialize cache
            self.cache_manager = await get_cache_manager()
            self.response_cache = await get_response_cache()
            
            # Discover and register endpoints
            await self._discover_endpoints()
            
            # Start background tasks
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            self._stats_update_task = asyncio.create_task(self._stats_update_loop())
            
            self.logger.info("API Gateway initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize API Gateway: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown the API gateway"""
        self.logger.info("Shutting down API Gateway")
        self._shutdown_event.set()
        
        if self._health_check_task:
            self._health_check_task.cancel()
        if self._stats_update_task:
            self._stats_update_task.cancel()
    
    async def process_request(self, request: GatewayRequest) -> GatewayResponse:
        """Process a unified request"""
        start_time = time.time()
        self.stats.total_requests += 1
        
        try:
            # Check cache first
            if request.use_cache:
                cached_response = await self._check_cache(request)
                if cached_response:
                    self.stats.cached_responses += 1
                    return cached_response
            
            # Route and execute request
            response = await self._execute_request(request)
            
            # Cache response if successful
            if request.use_cache and response.content:
                await self._cache_response(request, response)
            
            # Update statistics
            response_time = time.time() - start_time
            self.stats.successful_requests += 1
            self.stats.total_tokens += response.tokens_used
            self.stats.total_cost += response.cost
            self._update_avg_response_time(response_time)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Request processing failed: {e}")
            self.stats.failed_requests += 1
            raise
    
    async def stream_request(self, request: GatewayRequest) -> AsyncGenerator[str, None]:
        """Process a streaming request"""
        try:
            # Select endpoint
            endpoint = await self.load_balancer.select_endpoint(
                request.provider_preference,
                request.model_preference
            )
            
            if not endpoint:
                raise Exception("No available endpoints for streaming request")
            
            # Execute streaming request
            async for chunk in self._execute_streaming_request(request, endpoint):
                yield chunk
                
        except Exception as e:
            self.logger.error(f"Streaming request failed: {e}")
            raise
    
    async def _check_cache(self, request: GatewayRequest) -> Optional[GatewayResponse]:
        """Check if response is cached"""
        try:
            prompt_hash = self.response_cache.hash_prompt(
                request.prompt,
                request.system_prompt or "",
                request.temperature
            )
            
            model_key = f"{request.provider_preference}:{request.model_preference}"
            cached_data = await self.response_cache.get_cached_response(model_key, prompt_hash)
            
            if cached_data:
                return GatewayResponse(
                    content=cached_data["response"],
                    model_used=request.model_preference or "cached",
                    provider=request.provider_preference or ModelProvider.OPENAI,
                    endpoint_id="cache",
                    tokens_used=cached_data["tokens_used"],
                    cost=cached_data["cost"],
                    response_time=cached_data["response_time"],
                    cached=True,
                    attempts=1,
                    fallback_used=False,
                    metadata={"cached_at": cached_data["timestamp"]}
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Cache check failed: {e}")
            return None
    
    async def _cache_response(self, request: GatewayRequest, response: GatewayResponse):
        """Cache a response"""
        try:
            prompt_hash = self.response_cache.hash_prompt(
                request.prompt,
                request.system_prompt or "",
                request.temperature
            )
            
            model_key = f"{response.provider.value}:{response.model_used}"
            
            await self.response_cache.cache_response(
                model_key,
                prompt_hash,
                response.content,
                response.tokens_used,
                response.cost,
                response.response_time
            )
            
        except Exception as e:
            self.logger.error(f"Response caching failed: {e}")
    
    async def _execute_request(self, request: GatewayRequest) -> GatewayResponse:
        """Execute a request with retries and fallback"""
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                # Select endpoint
                endpoint = await self.load_balancer.select_endpoint(
                    request.provider_preference,
                    request.model_preference
                )
                
                if not endpoint:
                    raise Exception("No available endpoints")
                
                # Track connection
                endpoint.current_connections += 1
                
                try:
                    # Execute request based on provider
                    start_time = time.time()
                    
                    if endpoint.provider == ModelProvider.LOCAL_OLLAMA:
                        response = await self._execute_ollama_request(request, endpoint)
                    elif endpoint.provider == ModelProvider.LOCAL_HF:
                        response = await self._execute_hf_request(request, endpoint)
                    else:
                        response = await self._execute_cloud_request(request, endpoint)
                    
                    response_time = time.time() - start_time
                    
                    # Update endpoint stats
                    await self.load_balancer.update_endpoint_stats(
                        endpoint.id, response_time, True
                    )
                    
                    response.response_time = response_time
                    response.attempts = attempt + 1
                    response.fallback_used = attempt > 0
                    response.endpoint_id = endpoint.id
                    
                    return response
                    
                finally:
                    endpoint.current_connections -= 1
                    
            except Exception as e:
                last_exception = e
                self.logger.warning(f"Request attempt {attempt + 1} failed: {e}")
                
                # Update endpoint stats
                if 'endpoint' in locals():
                    await self.load_balancer.update_endpoint_stats(
                        endpoint.id, 0.0, False
                    )
                
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
        
        raise last_exception or Exception("All retry attempts failed")
    
    async def _execute_ollama_request(
        self, 
        request: GatewayRequest, 
        endpoint: ModelEndpoint
    ) -> GatewayResponse:
        """Execute request to Ollama endpoint"""
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{endpoint.endpoint_url}/api/generate",
                json={
                    "model": endpoint.model_name,
                    "prompt": request.prompt,
                    "system": request.system_prompt,
                    "options": {
                        "temperature": request.temperature,
                        "num_predict": request.max_tokens
                    },
                    "stream": False
                },
                timeout=aiohttp.ClientTimeout(total=request.timeout)
            ) as response:
                response.raise_for_status()
                data = await response.json()
                
                return GatewayResponse(
                    content=data.get("response", ""),
                    model_used=endpoint.model_name,
                    provider=endpoint.provider,
                    endpoint_id=endpoint.id,
                    tokens_used=data.get("eval_count", 0),
                    cost=0.0,  # Local models are free
                    response_time=0.0  # Will be set by caller
                )
    
    async def _execute_hf_request(
        self, 
        request: GatewayRequest, 
        endpoint: ModelEndpoint
    ) -> GatewayResponse:
        """Execute request to HuggingFace endpoint"""
        # This would connect to a HuggingFace model API
        # For now, return a placeholder response
        return GatewayResponse(
            content="HuggingFace response placeholder",
            model_used=endpoint.model_name,
            provider=endpoint.provider,
            endpoint_id=endpoint.id,
            tokens_used=50,
            cost=0.0,
            response_time=0.0
        )
    
    async def _execute_cloud_request(
        self, 
        request: GatewayRequest, 
        endpoint: ModelEndpoint
    ) -> GatewayResponse:
        """Execute request to cloud model endpoint"""
        # This would use the existing AI orchestrator for cloud models
        from app.services.advanced_orchestrator_service import get_advanced_orchestrator
        
        orchestrator = await get_advanced_orchestrator()
        
        # Convert to orchestrator format
        result = await orchestrator.execute_request(
            request.prompt,
            request.user_id or 1,
            system_prompt=request.system_prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        return GatewayResponse(
            content=result.get("content", ""),
            model_used=result.get("model_used", endpoint.model_name),
            provider=endpoint.provider,
            endpoint_id=endpoint.id,
            tokens_used=result.get("tokens_used", 0),
            cost=result.get("cost", 0.0),
            response_time=0.0
        )
    
    async def _execute_streaming_request(
        self, 
        request: GatewayRequest, 
        endpoint: ModelEndpoint
    ) -> AsyncGenerator[str, None]:
        """Execute streaming request"""
        if endpoint.provider == ModelProvider.LOCAL_OLLAMA:
            async for chunk in self._stream_ollama_request(request, endpoint):
                yield chunk
        else:
            # For other providers, fall back to non-streaming
            response = await self._execute_request(request)
            yield response.content
    
    async def _stream_ollama_request(
        self, 
        request: GatewayRequest, 
        endpoint: ModelEndpoint
    ) -> AsyncGenerator[str, None]:
        """Stream from Ollama endpoint"""
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{endpoint.endpoint_url}/api/generate",
                json={
                    "model": endpoint.model_name,
                    "prompt": request.prompt,
                    "system": request.system_prompt,
                    "options": {
                        "temperature": request.temperature,
                        "num_predict": request.max_tokens
                    },
                    "stream": True
                },
                timeout=aiohttp.ClientTimeout(total=request.timeout)
            ) as response:
                response.raise_for_status()
                
                async for line in response.content:
                    if line:
                        try:
                            import json
                            data = json.loads(line.decode())
                            if "response" in data:
                                yield data["response"]
                        except json.JSONDecodeError:
                            continue
    
    async def _discover_endpoints(self):
        """Discover available model endpoints"""
        try:
            # Import here to avoid circular imports
            from app.services.model_discovery import get_lifecycle_manager
            from app.services.container_orchestrator import get_container_orchestrator
            from app.core.config import settings
            
            manager = await get_lifecycle_manager()
            orchestrator = await get_container_orchestrator()
            
            # Add Ollama endpoint
            ollama_endpoint = ModelEndpoint(
                id="ollama_local",
                provider=ModelProvider.LOCAL_OLLAMA,
                model_name="llama2:7b",  # Default model
                endpoint_url=settings.OLLAMA_HOST,
                health_check_url=f"{settings.OLLAMA_HOST}/api/tags",
                weight=1.0,
                max_connections=10,
                current_connections=0,
                avg_response_time=2.0,
                health_score=1.0,
                last_health_check=datetime.utcnow(),
                is_active=True,
                error_count=0,
                success_count=0,
                total_requests=0
            )
            self.load_balancer.add_endpoint(ollama_endpoint)
            
            # Add HuggingFace endpoint
            hf_endpoint = ModelEndpoint(
                id="hf_local",
                provider=ModelProvider.LOCAL_HF,
                model_name="microsoft/DialoGPT-small",
                endpoint_url=settings.HF_LOCAL_HOST,
                health_check_url=f"{settings.HF_LOCAL_HOST}/health",
                weight=1.0,
                max_connections=5,
                current_connections=0,
                avg_response_time=3.0,
                health_score=1.0,
                last_health_check=datetime.utcnow(),
                is_active=True,
                error_count=0,
                success_count=0,
                total_requests=0
            )
            self.load_balancer.add_endpoint(hf_endpoint)
            
            # Add cloud model endpoints (OpenAI, Anthropic, etc.)
            cloud_providers = [
                (ModelProvider.OPENAI, "gpt-4"),
                (ModelProvider.ANTHROPIC, "claude-3-sonnet"),
                (ModelProvider.GOOGLE, "gemini-pro")
            ]
            
            for provider, model in cloud_providers:
                endpoint = ModelEndpoint(
                    id=f"{provider.value}_cloud",
                    provider=provider,
                    model_name=model,
                    endpoint_url="https://api.cloud.provider",  # Placeholder
                    health_check_url="https://api.cloud.provider/health",
                    weight=1.0,
                    max_connections=100,
                    current_connections=0,
                    avg_response_time=1.5,
                    health_score=1.0,
                    last_health_check=datetime.utcnow(),
                    is_active=True,
                    error_count=0,
                    success_count=0,
                    total_requests=0
                )
                self.load_balancer.add_endpoint(endpoint)
            
            self.stats.active_endpoints = len(self.load_balancer.endpoints)
            self.logger.info(f"Discovered {self.stats.active_endpoints} endpoints")
            
        except Exception as e:
            self.logger.error(f"Endpoint discovery failed: {e}")
    
    async def _health_check_loop(self):
        """Background health check loop"""
        while not self._shutdown_event.is_set():
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(60)
    
    async def _perform_health_checks(self):
        """Perform health checks on all endpoints"""
        for endpoint in self.load_balancer.endpoints.values():
            try:
                # Simple health check - ping the endpoint
                import aiohttp
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        endpoint.health_check_url,
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as response:
                        if response.status == 200:
                            endpoint.health_score = min(1.0, endpoint.health_score + 0.1)
                            endpoint.is_active = True
                        else:
                            endpoint.health_score = max(0.0, endpoint.health_score - 0.2)
                            endpoint.is_active = endpoint.health_score > self.circuit_breaker_threshold
                        
                        endpoint.last_health_check = datetime.utcnow()
                        
            except Exception as e:
                self.logger.warning(f"Health check failed for {endpoint.id}: {e}")
                endpoint.health_score = max(0.0, endpoint.health_score - 0.3)
                endpoint.is_active = endpoint.health_score > self.circuit_breaker_threshold
                endpoint.last_health_check = datetime.utcnow()
    
    async def _stats_update_loop(self):
        """Background statistics update loop"""
        while not self._shutdown_event.is_set():
            try:
                await self._update_stats()
                await asyncio.sleep(30)  # Update every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Stats update loop error: {e}")
                await asyncio.sleep(60)
    
    async def _update_stats(self):
        """Update gateway statistics"""
        # Update active endpoints count
        self.stats.active_endpoints = len([
            ep for ep in self.load_balancer.endpoints.values()
            if ep.is_active
        ])
        
        # Update cache hit rate
        if self.stats.total_requests > 0:
            self.stats.cache_hit_rate = (
                self.stats.cached_responses / self.stats.total_requests * 100
            )
        
        # Update error rate
        if self.stats.total_requests > 0:
            self.stats.error_rate = (
                self.stats.failed_requests / self.stats.total_requests * 100
            )
    
    def _update_avg_response_time(self, response_time: float):
        """Update average response time"""
        alpha = 0.1  # Smoothing factor
        self.stats.avg_response_time = (
            alpha * response_time + (1 - alpha) * self.stats.avg_response_time
        )
    
    async def get_gateway_stats(self) -> GatewayStats:
        """Get current gateway statistics"""
        return self.stats
    
    async def get_endpoint_stats(self) -> List[ModelEndpoint]:
        """Get statistics for all endpoints"""
        return list(self.load_balancer.endpoints.values())


# Global instance
_api_gateway: Optional[APIGateway] = None


async def get_api_gateway() -> APIGateway:
    """Get the global API gateway instance"""
    global _api_gateway
    if _api_gateway is None:
        _api_gateway = APIGateway()
        await _api_gateway.initialize()
    return _api_gateway