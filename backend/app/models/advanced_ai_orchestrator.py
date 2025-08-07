import asyncio
import time
import json
import hashlib
import re
from typing import Dict, List, Optional, Any, AsyncGenerator, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import openai
import anthropic
import google.generativeai as genai
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, func, and_

from app.core.config import settings
from app.core.logger import LoggerMixin
from app.database.models import ModelProvider, AIModel, UsageLog, User, Conversation
from app.database.database import async_session_factory
from app.services.cost_optimizer import CostOptimizer
from app.utils.cache import CacheManager


class TaskComplexity(str, Enum):
    SIMPLE = "simple"           # Basic queries, translations, simple Q&A
    MODERATE = "moderate"       # Analysis, summaries, explanations  
    COMPLEX = "complex"         # Code generation, complex reasoning
    EXPERT = "expert"           # Advanced problem solving, research


class RoutingStrategy(str, Enum):
    COST_OPTIMAL = "cost_optimal"           # Cheapest suitable model
    PERFORMANCE_OPTIMAL = "performance_optimal"  # Fastest suitable model
    QUALITY_OPTIMAL = "quality_optimal"     # Best quality regardless of cost
    BALANCED = "balanced"                   # Balance all factors
    USER_PREFERENCE = "user_preference"     # Based on user history


@dataclass
class TaskAnalysis:
    """Analysis results for a given task"""
    complexity: TaskComplexity
    estimated_tokens: int
    requires_code: bool
    requires_reasoning: bool
    requires_multimodal: bool
    language_detection: Optional[str]
    domain: Optional[str]
    confidence_score: float


@dataclass
class ModelCapabilities:
    """Model capabilities and characteristics"""
    provider: ModelProvider
    model_name: str
    max_tokens: int
    context_window: int
    cost_per_1k_input: float
    cost_per_1k_output: float
    avg_response_time: float
    success_rate: float
    quality_score: float
    supports_code: bool
    supports_reasoning: bool
    supports_multimodal: bool
    supports_streaming: bool
    is_local: bool


@dataclass
class RoutingDecision:
    """Decision made by the orchestrator"""
    chosen_provider: ModelProvider
    chosen_model: str
    reasoning: str
    estimated_cost: float
    estimated_time: float
    fallback_models: List[Tuple[ModelProvider, str]]
    confidence: float


@dataclass
class UserPreferences:
    """User's model preferences and patterns"""
    user_id: int
    preferred_models: Dict[TaskComplexity, List[str]]
    cost_sensitivity: float  # 0-1, higher = more cost sensitive
    speed_preference: float  # 0-1, higher = prefers faster models
    quality_preference: float  # 0-1, higher = prefers quality
    monthly_budget: float
    current_spend: float
    avg_tokens_per_request: float
    peak_usage_hours: List[int]


@dataclass
class BatchRequest:
    """Batch processing request"""
    requests: List[Dict[str, Any]]
    batch_id: str
    user_id: int
    priority: int
    created_at: datetime


class AdvancedAIOrchestrator(LoggerMixin):
    """Advanced AI Model Orchestrator with intelligent routing"""
    
    def __init__(self, cache_manager: Optional[CacheManager] = None):
        super().__init__()
        self.cache = cache_manager or CacheManager(settings.REDIS_URL)
        self.cost_optimizer = CostOptimizer()
        self.model_clients = {}
        self.model_capabilities = {}
        self.performance_metrics = {}
        self.user_preferences_cache = {}
        self.batch_queue = {}
        self.active_streams = {}
        self.fallback_chains = {}
        self.initialized = False
        
        # Initialize model capabilities
        self._initialize_model_capabilities()
        
        # Task complexity patterns
        self.complexity_patterns = {
            TaskComplexity.SIMPLE: [
                r'\b(translate|what is|define|hello|hi|thanks)\b',
                r'\b(simple|basic|quick)\b',
                r'\?\s*$',  # Single questions
            ],
            TaskComplexity.MODERATE: [
                r'\b(explain|analyze|summarize|compare|describe)\b',
                r'\b(how does|why does|what are the)\b',
                r'\b(advantages|disadvantages|pros|cons)\b',
            ],
            TaskComplexity.COMPLEX: [
                r'\b(code|program|script|function|algorithm)\b',
                r'\b(debug|optimize|refactor|implement)\b',
                r'\b(design|architecture|system)\b',
            ],
            TaskComplexity.EXPERT: [
                r'\b(research|thesis|academic|scientific)\b',
                r'\b(complex|advanced|sophisticated)\b',
                r'\b(mathematical|statistical|analysis)\b',
            ]
        }

    def _initialize_model_capabilities(self):
        """Initialize model capabilities database"""
        self.model_capabilities = {
            "openai": {
                "o3-mini": ModelCapabilities(
                    provider=ModelProvider.OPENAI,
                    model_name="o3-mini",
                    max_tokens=65536,
                    context_window=128000,
                    cost_per_1k_input=0.002,
                    cost_per_1k_output=0.008,
                    avg_response_time=2.5,
                    success_rate=0.98,
                    quality_score=0.95,
                    supports_code=True,
                    supports_reasoning=True,
                    supports_multimodal=False,
                    supports_streaming=True,
                    is_local=False
                ),
                "gpt-4": ModelCapabilities(
                    provider=ModelProvider.OPENAI,
                    model_name="gpt-4",
                    max_tokens=8192,
                    context_window=8192,
                    cost_per_1k_input=0.03,
                    cost_per_1k_output=0.06,
                    avg_response_time=3.0,
                    success_rate=0.96,
                    quality_score=0.92,
                    supports_code=True,
                    supports_reasoning=True,
                    supports_multimodal=False,
                    supports_streaming=True,
                    is_local=False
                ),
                "gpt-3.5-turbo": ModelCapabilities(
                    provider=ModelProvider.OPENAI,
                    model_name="gpt-3.5-turbo",
                    max_tokens=4096,
                    context_window=16385,
                    cost_per_1k_input=0.0015,
                    cost_per_1k_output=0.002,
                    avg_response_time=1.5,
                    success_rate=0.94,
                    quality_score=0.85,
                    supports_code=True,
                    supports_reasoning=False,
                    supports_multimodal=False,
                    supports_streaming=True,
                    is_local=False
                )
            },
            "anthropic": {
                "claude-3-5-sonnet-20241022": ModelCapabilities(
                    provider=ModelProvider.ANTHROPIC,
                    model_name="claude-3-5-sonnet-20241022",
                    max_tokens=8192,
                    context_window=200000,
                    cost_per_1k_input=0.003,
                    cost_per_1k_output=0.015,
                    avg_response_time=2.8,
                    success_rate=0.97,
                    quality_score=0.94,
                    supports_code=True,
                    supports_reasoning=True,
                    supports_multimodal=False,
                    supports_streaming=True,
                    is_local=False
                ),
                "claude-3-haiku-20240307": ModelCapabilities(
                    provider=ModelProvider.ANTHROPIC,
                    model_name="claude-3-haiku-20240307",
                    max_tokens=4096,
                    context_window=200000,
                    cost_per_1k_input=0.00025,
                    cost_per_1k_output=0.00125,
                    avg_response_time=1.2,
                    success_rate=0.95,
                    quality_score=0.88,
                    supports_code=True,
                    supports_reasoning=False,
                    supports_multimodal=False,
                    supports_streaming=True,
                    is_local=False
                )
            },
            "google": {
                "gemini-1.5-pro": ModelCapabilities(
                    provider=ModelProvider.GOOGLE,
                    model_name="gemini-1.5-pro",
                    max_tokens=8192,
                    context_window=1000000,
                    cost_per_1k_input=0.001,
                    cost_per_1k_output=0.003,
                    avg_response_time=2.2,
                    success_rate=0.94,
                    quality_score=0.89,
                    supports_code=True,
                    supports_reasoning=True,
                    supports_multimodal=True,
                    supports_streaming=True,
                    is_local=False
                ),
                "gemini-1.5-flash": ModelCapabilities(
                    provider=ModelProvider.GOOGLE,
                    model_name="gemini-1.5-flash",
                    max_tokens=8192,
                    context_window=1000000,
                    cost_per_1k_input=0.00005,
                    cost_per_1k_output=0.00015,
                    avg_response_time=0.8,
                    success_rate=0.92,
                    quality_score=0.82,
                    supports_code=False,
                    supports_reasoning=False,
                    supports_multimodal=True,
                    supports_streaming=True,
                    is_local=False
                )
            }
        }

    async def initialize(self):
        """Initialize the orchestrator and all model clients"""
        self.logger.info("Initializing Advanced AI Orchestrator...")
        
        try:
            # Initialize cache
            await self.cache.connect()
            
            # Initialize model clients (reuse from existing orchestrator)
            from app.models.ai_orchestrator import OpenAIClient, AnthropicClient, GoogleClient
            
            clients = [
                OpenAIClient(),
                AnthropicClient(),
                GoogleClient()
            ]
            
            initialization_tasks = [client.initialize() for client in clients]
            results = await asyncio.gather(*initialization_tasks, return_exceptions=True)
            
            for client, result in zip(clients, results):
                if isinstance(result, Exception):
                    self.logger.error(f"Failed to initialize {client.provider}: {result}")
                elif result:
                    self.model_clients[client.provider.value] = client
                    self.logger.info(f"Successfully initialized {client.provider}")
            
            if not self.model_clients:
                self.logger.warning("No AI model clients were successfully initialized - API features will be disabled")
            # Don't raise error, just continue without AI clients
            
            # Load performance metrics from cache
            await self._load_performance_metrics()
            
            # Initialize fallback chains
            self._initialize_fallback_chains()
            
            self.initialized = True
            self.logger.info(f"Advanced AI Orchestrator initialized with {len(self.model_clients)} providers")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Advanced AI Orchestrator: {e}")
            raise

    async def route_request(
        self, 
        task: str, 
        user_id: int, 
        budget_limit: Optional[float] = None,
        strategy: RoutingStrategy = RoutingStrategy.BALANCED,
        context: Optional[Dict] = None,
        stream: bool = False
    ) -> RoutingDecision:
        """
        Intelligently route a request to the best AI model
        
        Args:
            task: The task/prompt to process
            user_id: User identifier  
            budget_limit: Maximum cost allowed for this request
            strategy: Routing strategy to use
            context: Additional context (conversation history, etc.)
            stream: Whether streaming is required
            
        Returns:
            RoutingDecision with chosen model and reasoning
        """
        if not self.initialized:
            await self.initialize()
        
        self.logger.info(f"Routing request for user {user_id} with strategy {strategy}")
        
        try:
            # Analyze the task
            task_analysis = await self._analyze_task_complexity(task, context)
            
            # Get user preferences
            user_prefs = await self._get_user_preferences(user_id)
            
            # Check budget constraints
            available_budget = await self._calculate_available_budget(user_id, budget_limit)
            
            # Filter suitable models
            suitable_models = await self._filter_suitable_models(
                task_analysis, user_prefs, available_budget, stream
            )
            
            if not suitable_models:
                raise ValueError("No suitable models available for this request")
            
            # Select optimal model based on strategy
            decision = await self._select_optimal_model(
                suitable_models, task_analysis, user_prefs, strategy
            )
            
            # Cache the decision
            await self._cache_routing_decision(user_id, task, decision)
            
            self.logger.info(f"Selected {decision.chosen_provider.value}:{decision.chosen_model} - {decision.reasoning}")
            
            return decision
            
        except Exception as e:
            self.logger.error(f"Failed to route request: {e}")
            # Fallback to default model
            return RoutingDecision(
                chosen_provider=ModelProvider.OPENAI,
                chosen_model="gpt-3.5-turbo",
                reasoning="Fallback due to routing error",
                estimated_cost=0.01,
                estimated_time=2.0,
                fallback_models=[],
                confidence=0.5
            )

    async def execute_request(
        self,
        task: str,
        user_id: int,
        routing_decision: Optional[RoutingDecision] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a request using the optimal model with fallback handling
        """
        if not routing_decision:
            routing_decision = await self.route_request(task, user_id, **kwargs)
        
        start_time = time.time()
        attempts = 0
        max_attempts = 3
        
        # Primary attempt
        models_to_try = [(routing_decision.chosen_provider, routing_decision.chosen_model)]
        models_to_try.extend(routing_decision.fallback_models)
        
        for provider, model_name in models_to_try:
            attempts += 1
            
            try:
                self.logger.info(f"Attempt {attempts}: Using {provider.value}:{model_name}")
                
                # Get model client
                client = self.model_clients.get(provider.value)
                if not client:
                    continue
                
                # Prepare request
                from app.models.ai_orchestrator import ModelRequest
                request = ModelRequest(
                    prompt=task,
                    user_id=user_id,
                    model_preference=model_name,
                    provider_preference=provider,
                    **kwargs
                )
                
                # Execute request
                response = await client.generate(request)
                
                # Track performance
                execution_time = time.time() - start_time
                await self._track_model_performance(
                    provider, model_name, True, execution_time, 
                    response.tokens_used, response.cost
                )
                
                # Update user preferences based on success
                await self._update_user_preferences(
                    user_id, provider, model_name, True, response.cost, execution_time
                )
                
                return {
                    "response": response,
                    "model_used": f"{provider.value}:{model_name}",
                    "attempts": attempts,
                    "execution_time": execution_time,
                    "success": True
                }
                
            except Exception as e:
                self.logger.warning(f"Model {provider.value}:{model_name} failed: {e}")
                
                # Track failure
                execution_time = time.time() - start_time
                await self._track_model_performance(
                    provider, model_name, False, execution_time, 0, 0
                )
                
                # Continue to next model if available
                if attempts < max_attempts and attempts < len(models_to_try):
                    continue
                else:
                    # All models failed
                    await self._update_user_preferences(
                        user_id, provider, model_name, False, 0, execution_time
                    )
                    
                    raise Exception(f"All models failed. Last error: {e}")
        
        raise Exception("No suitable models available")

    async def stream_request(
        self,
        task: str,
        user_id: int,
        routing_decision: Optional[RoutingDecision] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Execute a streaming request with fallback handling
        """
        if not routing_decision:
            routing_decision = await self.route_request(task, user_id, stream=True, **kwargs)
        
        stream_id = hashlib.md5(f"{user_id}:{task}:{time.time()}".encode()).hexdigest()
        self.active_streams[stream_id] = {
            "user_id": user_id,
            "start_time": time.time(),
            "model": f"{routing_decision.chosen_provider.value}:{routing_decision.chosen_model}"
        }
        
        try:
            # Get model client
            client = self.model_clients.get(routing_decision.chosen_provider.value)
            if not client:
                raise ValueError(f"Client not available: {routing_decision.chosen_provider.value}")
            
            # Prepare request
            from app.models.ai_orchestrator import ModelRequest
            request = ModelRequest(
                prompt=task,
                user_id=user_id,
                model_preference=routing_decision.chosen_model,
                provider_preference=routing_decision.chosen_provider,
                stream=True,
                **kwargs
            )
            
            # Stream response
            total_content = ""
            chunk_count = 0
            
            async for chunk in client.stream_generate(request):
                total_content += chunk
                chunk_count += 1
                yield chunk
            
            # Track streaming performance
            execution_time = time.time() - self.active_streams[stream_id]["start_time"]
            estimated_tokens = len(total_content.split()) * 1.3  # Rough estimation
            
            await self._track_model_performance(
                routing_decision.chosen_provider,
                routing_decision.chosen_model,
                True,
                execution_time,
                int(estimated_tokens),
                0  # Cost calculated separately
            )
            
        except Exception as e:
            self.logger.error(f"Streaming failed: {e}")
            
            # Try fallback if available
            if routing_decision.fallback_models:
                provider, model = routing_decision.fallback_models[0]
                self.logger.info(f"Switching to fallback: {provider.value}:{model}")
                
                fallback_decision = RoutingDecision(
                    chosen_provider=provider,
                    chosen_model=model,
                    reasoning="Fallback due to streaming failure",
                    estimated_cost=routing_decision.estimated_cost,
                    estimated_time=routing_decision.estimated_time,
                    fallback_models=[],
                    confidence=0.5
                )
                
                async for chunk in self.stream_request(task, user_id, fallback_decision, **kwargs):
                    yield chunk
            else:
                raise
        
        finally:
            # Cleanup
            if stream_id in self.active_streams:
                del self.active_streams[stream_id]

    async def batch_process(
        self,
        requests: List[Dict[str, Any]],
        user_id: int,
        batch_id: Optional[str] = None,
        priority: int = 0
    ) -> str:
        """
        Process multiple requests efficiently in batch
        """
        if not batch_id:
            batch_id = hashlib.md5(f"{user_id}:{time.time()}".encode()).hexdigest()
        
        batch_request = BatchRequest(
            requests=requests,
            batch_id=batch_id,
            user_id=user_id,
            priority=priority,
            created_at=datetime.utcnow()
        )
        
        self.batch_queue[batch_id] = batch_request
        
        # Process batch asynchronously
        asyncio.create_task(self._process_batch(batch_request))
        
        return batch_id

    async def _analyze_task_complexity(self, task: str, context: Optional[Dict] = None) -> TaskAnalysis:
        """Analyze task complexity using pattern matching and heuristics"""
        
        task_lower = task.lower()
        word_count = len(task.split())
        
        # Pattern matching for complexity
        complexity_scores = {complexity: 0 for complexity in TaskComplexity}
        
        for complexity, patterns in self.complexity_patterns.items():
            for pattern in patterns:
                if re.search(pattern, task_lower):
                    complexity_scores[complexity] += 1
        
        # Heuristic scoring
        if word_count > 100:
            complexity_scores[TaskComplexity.COMPLEX] += 1
        elif word_count < 20:
            complexity_scores[TaskComplexity.SIMPLE] += 1
        
        # Check for code-related keywords
        code_keywords = ["code", "function", "class", "algorithm", "debug", "implement"]
        requires_code = any(keyword in task_lower for keyword in code_keywords)
        if requires_code:
            complexity_scores[TaskComplexity.COMPLEX] += 2
        
        # Check for reasoning keywords
        reasoning_keywords = ["analyze", "compare", "evaluate", "reasoning", "logic"]
        requires_reasoning = any(keyword in task_lower for keyword in reasoning_keywords)
        if requires_reasoning:
            complexity_scores[TaskComplexity.MODERATE] += 1
        
        # Determine final complexity
        max_score = max(complexity_scores.values())
        if max_score == 0:
            complexity = TaskComplexity.SIMPLE
        else:
            complexity = max(complexity_scores, key=complexity_scores.get)
        
        # Estimate tokens (rough)
        estimated_tokens = min(word_count * 1.3 + 500, 4000)  # Input + expected output
        
        return TaskAnalysis(
            complexity=complexity,
            estimated_tokens=int(estimated_tokens),
            requires_code=requires_code,
            requires_reasoning=requires_reasoning,
            requires_multimodal=False,  # Would need more sophisticated detection
            language_detection=None,    # Could add language detection
            domain=None,               # Could add domain classification
            confidence_score=min(max_score / 3, 1.0)
        )

    async def _get_user_preferences(self, user_id: int) -> UserPreferences:
        """Get or create user preferences"""
        
        # Check cache first
        cache_key = f"user_prefs:{user_id}"
        cached_prefs = await self.cache.get(cache_key)
        if cached_prefs:
            return UserPreferences(**cached_prefs)
        
        # Load from database
        try:
            async with async_session_factory() as db:
                # Get user data
                user_query = select(User).where(User.id == user_id)
                result = await db.execute(user_query)
                user = result.scalar_one_or_none()
                
                if not user:
                    raise ValueError(f"User {user_id} not found")
                
                # Calculate current monthly spend
                current_month_start = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                usage_query = select(func.sum(UsageLog.cost)).where(
                    and_(
                        UsageLog.user_id == user_id,
                        UsageLog.created_at >= current_month_start
                    )
                )
                result = await db.execute(usage_query)
                current_spend = result.scalar() or 0.0
                
                # Create default preferences
                prefs = UserPreferences(
                    user_id=user_id,
                    preferred_models={
                        TaskComplexity.SIMPLE: ["gpt-3.5-turbo", "claude-3-haiku-20240307"],
                        TaskComplexity.MODERATE: ["gpt-4", "claude-3-5-sonnet-20241022"],
                        TaskComplexity.COMPLEX: ["o3-mini", "claude-3-5-sonnet-20241022"],
                        TaskComplexity.EXPERT: ["o3-mini", "gpt-4"]
                    },
                    cost_sensitivity=0.7,  # Default values
                    speed_preference=0.5,
                    quality_preference=0.8,
                    monthly_budget=user.api_quota_limit or 100.0,
                    current_spend=current_spend,
                    avg_tokens_per_request=1000,
                    peak_usage_hours=[9, 10, 11, 14, 15, 16]  # Default business hours
                )
                
                # Cache preferences
                await self.cache.set(cache_key, asdict(prefs), ttl=3600)
                
                return prefs
                
        except Exception as e:
            self.logger.error(f"Failed to load user preferences: {e}")
            # Return default preferences
            return UserPreferences(
                user_id=user_id,
                preferred_models={
                    TaskComplexity.SIMPLE: ["gpt-3.5-turbo"],
                    TaskComplexity.MODERATE: ["gpt-4"],
                    TaskComplexity.COMPLEX: ["gpt-4"],
                    TaskComplexity.EXPERT: ["gpt-4"]
                },
                cost_sensitivity=0.7,
                speed_preference=0.5,
                quality_preference=0.8,
                monthly_budget=100.0,
                current_spend=0.0,
                avg_tokens_per_request=1000,
                peak_usage_hours=[9, 10, 11, 14, 15, 16]
            )

    async def _calculate_available_budget(self, user_id: int, budget_limit: Optional[float] = None) -> float:
        """Calculate available budget for the request"""
        
        user_prefs = await self._get_user_preferences(user_id)
        
        # User's remaining monthly budget
        monthly_remaining = user_prefs.monthly_budget - user_prefs.current_spend
        
        # Request-specific limit
        request_limit = budget_limit or monthly_remaining
        
        # Return the minimum
        return min(monthly_remaining, request_limit, 50.0)  # Cap at $50 per request

    async def _filter_suitable_models(
        self,
        task_analysis: TaskAnalysis,
        user_prefs: UserPreferences,
        available_budget: float,
        stream: bool = False
    ) -> List[ModelCapabilities]:
        """Filter models that are suitable for the task"""
        
        suitable_models = []
        
        for provider_models in self.model_capabilities.values():
            for model_name, capabilities in provider_models.items():
                # Check if provider is available
                if capabilities.provider.value not in self.model_clients:
                    continue
                
                # Check streaming requirement
                if stream and not capabilities.supports_streaming:
                    continue
                
                # Check task requirements
                if task_analysis.requires_code and not capabilities.supports_code:
                    continue
                
                if task_analysis.requires_reasoning and not capabilities.supports_reasoning:
                    continue
                
                if task_analysis.requires_multimodal and not capabilities.supports_multimodal:
                    continue
                
                # Check token limits
                if task_analysis.estimated_tokens > capabilities.max_tokens:
                    continue
                
                # Estimate cost
                estimated_cost = self._estimate_request_cost(capabilities, task_analysis.estimated_tokens)
                
                # Check budget
                if estimated_cost > available_budget:
                    continue
                
                suitable_models.append(capabilities)
        
        return suitable_models

    def _estimate_request_cost(self, capabilities: ModelCapabilities, estimated_tokens: int) -> float:
        """Estimate the cost for a request"""
        
        # Rough split: 30% input, 70% output
        input_tokens = int(estimated_tokens * 0.3)
        output_tokens = int(estimated_tokens * 0.7)
        
        input_cost = (input_tokens / 1000) * capabilities.cost_per_1k_input
        output_cost = (output_tokens / 1000) * capabilities.cost_per_1k_output
        
        return input_cost + output_cost

    async def _select_optimal_model(
        self,
        suitable_models: List[ModelCapabilities],
        task_analysis: TaskAnalysis,
        user_prefs: UserPreferences,
        strategy: RoutingStrategy
    ) -> RoutingDecision:
        """Select the optimal model based on strategy"""
        
        if not suitable_models:
            raise ValueError("No suitable models available")
        
        scored_models = []
        
        for model in suitable_models:
            score = 0.0
            reasoning_parts = []
            
            # Base scoring factors
            estimated_cost = self._estimate_request_cost(model, task_analysis.estimated_tokens)
            
            if strategy == RoutingStrategy.COST_OPTIMAL:
                # Prioritize cost (lower is better)
                score = 1.0 / (estimated_cost + 0.001)
                reasoning_parts.append(f"cost: ${estimated_cost:.4f}")
                
            elif strategy == RoutingStrategy.PERFORMANCE_OPTIMAL:
                # Prioritize speed (lower response time is better)
                score = 1.0 / (model.avg_response_time + 0.1)
                reasoning_parts.append(f"speed: {model.avg_response_time:.1f}s")
                
            elif strategy == RoutingStrategy.QUALITY_OPTIMAL:
                # Prioritize quality
                score = model.quality_score * model.success_rate
                reasoning_parts.append(f"quality: {model.quality_score:.2f}")
                
            elif strategy == RoutingStrategy.USER_PREFERENCE:
                # Check if model is in user's preferred models for this complexity
                preferred = user_prefs.preferred_models.get(task_analysis.complexity, [])
                if model.model_name in preferred:
                    score = 10.0  # High score for preferred models
                    reasoning_parts.append("user preferred")
                else:
                    score = model.quality_score * 0.5
                
            else:  # BALANCED
                # Balance all factors
                cost_score = 1.0 / (estimated_cost + 0.001)
                speed_score = 1.0 / (model.avg_response_time + 0.1)
                quality_score = model.quality_score * model.success_rate
                
                # Weight based on user preferences
                score = (
                    cost_score * user_prefs.cost_sensitivity +
                    speed_score * user_prefs.speed_preference + 
                    quality_score * user_prefs.quality_preference
                ) / 3.0
                
                reasoning_parts.extend([
                    f"cost: ${estimated_cost:.4f}",
                    f"speed: {model.avg_response_time:.1f}s", 
                    f"quality: {model.quality_score:.2f}"
                ])
            
            scored_models.append((score, model, reasoning_parts))
        
        # Sort by score (highest first)
        scored_models.sort(key=lambda x: x[0], reverse=True)
        
        # Select top model
        best_score, best_model, reasoning_parts = scored_models[0]
        
        # Prepare fallback models
        fallback_models = [
            (model.provider, model.model_name) 
            for _, model, _ in scored_models[1:3]  # Top 2 fallbacks
        ]
        
        reasoning = f"{strategy.value} strategy: {', '.join(reasoning_parts)}"
        estimated_cost = self._estimate_request_cost(best_model, task_analysis.estimated_tokens)
        
        return RoutingDecision(
            chosen_provider=best_model.provider,
            chosen_model=best_model.model_name,
            reasoning=reasoning,
            estimated_cost=estimated_cost,
            estimated_time=best_model.avg_response_time,
            fallback_models=fallback_models,
            confidence=min(best_score / 10.0, 1.0)
        )

    async def _track_model_performance(
        self,
        provider: ModelProvider,
        model_name: str,
        success: bool,
        response_time: float,
        tokens_used: int,
        cost: float
    ):
        """Track model performance metrics"""
        
        cache_key = f"model_perf:{provider.value}:{model_name}"
        
        try:
            # Get current metrics
            current_metrics = await self.cache.get(cache_key) or {
                "total_requests": 0,
                "successful_requests": 0,
                "total_response_time": 0.0,
                "total_tokens": 0,
                "total_cost": 0.0,
                "last_updated": time.time()
            }
            
            # Update metrics
            current_metrics["total_requests"] += 1
            if success:
                current_metrics["successful_requests"] += 1
            current_metrics["total_response_time"] += response_time
            current_metrics["total_tokens"] += tokens_used
            current_metrics["total_cost"] += cost
            current_metrics["last_updated"] = time.time()
            
            # Calculate derived metrics
            if current_metrics["total_requests"] > 0:
                current_metrics["success_rate"] = current_metrics["successful_requests"] / current_metrics["total_requests"]
                current_metrics["avg_response_time"] = current_metrics["total_response_time"] / current_metrics["total_requests"]
            
            # Cache updated metrics
            await self.cache.set(cache_key, current_metrics, ttl=86400)  # 24 hours
            
        except Exception as e:
            self.logger.error(f"Failed to track model performance: {e}")

    async def _update_user_preferences(
        self,
        user_id: int,
        provider: ModelProvider,
        model_name: str,
        success: bool,
        cost: float,
        response_time: float
    ):
        """Update user preferences based on experience"""
        
        try:
            user_prefs = await self._get_user_preferences(user_id)
            
            # Update spending
            user_prefs.current_spend += cost
            
            # Update preferred models if successful
            if success:
                # Add to preferred models if not already there
                for complexity in TaskComplexity:
                    if model_name not in user_prefs.preferred_models[complexity]:
                        # Add if this model performed well
                        if response_time < 3.0 and cost < 0.1:  # Fast and cheap
                            user_prefs.preferred_models[complexity].append(model_name)
                            # Keep only top 3 preferred models per complexity
                            user_prefs.preferred_models[complexity] = user_prefs.preferred_models[complexity][:3]
            
            # Cache updated preferences
            cache_key = f"user_prefs:{user_id}"
            await self.cache.set(cache_key, asdict(user_prefs), ttl=3600)
            
        except Exception as e:
            self.logger.error(f"Failed to update user preferences: {e}")

    async def _load_performance_metrics(self):
        """Load historical performance metrics"""
        try:
            # This would typically load from database
            # For now, we'll use cached metrics
            self.logger.info("Performance metrics loaded from cache")
        except Exception as e:
            self.logger.error(f"Failed to load performance metrics: {e}")

    def _initialize_fallback_chains(self):
        """Initialize fallback chains for different scenarios"""
        self.fallback_chains = {
            TaskComplexity.SIMPLE: [
                ("openai", "gpt-3.5-turbo"),
                ("google", "gemini-1.5-flash"),
                ("anthropic", "claude-3-haiku-20240307")
            ],
            TaskComplexity.MODERATE: [
                ("openai", "gpt-4"),
                ("anthropic", "claude-3-5-sonnet-20241022"),
                ("google", "gemini-1.5-pro")
            ],
            TaskComplexity.COMPLEX: [
                ("openai", "o3-mini"),
                ("anthropic", "claude-3-5-sonnet-20241022"),
                ("openai", "gpt-4")
            ],
            TaskComplexity.EXPERT: [
                ("openai", "o3-mini"),
                ("openai", "gpt-4"),
                ("anthropic", "claude-3-5-sonnet-20241022")
            ]
        }

    async def _cache_routing_decision(self, user_id: int, task: str, decision: RoutingDecision):
        """Cache routing decision for potential reuse"""
        task_hash = hashlib.md5(task.encode()).hexdigest()
        cache_key = f"routing_decision:{user_id}:{task_hash}"
        
        try:
            await self.cache.set(cache_key, asdict(decision), ttl=1800)  # 30 minutes
        except Exception as e:
            self.logger.error(f"Failed to cache routing decision: {e}")

    async def _process_batch(self, batch_request: BatchRequest):
        """Process a batch of requests"""
        self.logger.info(f"Processing batch {batch_request.batch_id} with {len(batch_request.requests)} requests")
        
        try:
            # Process requests concurrently
            tasks = []
            for request_data in batch_request.requests:
                task = asyncio.create_task(
                    self.execute_request(
                        request_data.get("task", ""),
                        batch_request.user_id,
                        **request_data
                    )
                )
                tasks.append(task)
            
            # Wait for all to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Cache results
            cache_key = f"batch_results:{batch_request.batch_id}"
            await self.cache.set(cache_key, results, ttl=3600)
            
            self.logger.info(f"Batch {batch_request.batch_id} completed")
            
        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
        finally:
            # Clean up
            if batch_request.batch_id in self.batch_queue:
                del self.batch_queue[batch_request.batch_id]

    async def get_analytics(self) -> Dict[str, Any]:
        """Get orchestrator analytics and performance data"""
        
        analytics = {
            "models": {},
            "users": {},
            "performance": {},
            "costs": {},
            "active_streams": len(self.active_streams),
            "queued_batches": len(self.batch_queue)
        }
        
        try:
            # Model performance analytics
            for provider_models in self.model_capabilities.values():
                for model_name, capabilities in provider_models.items():
                    cache_key = f"model_perf:{capabilities.provider.value}:{model_name}"
                    metrics = await self.cache.get(cache_key)
                    
                    if metrics:
                        analytics["models"][f"{capabilities.provider.value}:{model_name}"] = metrics
            
            # Overall performance
            total_requests = sum(m.get("total_requests", 0) for m in analytics["models"].values())
            total_cost = sum(m.get("total_cost", 0) for m in analytics["models"].values())
            
            analytics["performance"] = {
                "total_requests": total_requests,
                "total_cost": total_cost,
                "avg_cost_per_request": total_cost / max(total_requests, 1)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get analytics: {e}")
        
        return analytics

    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check"""
        
        health = {
            "status": "healthy",
            "initialized": self.initialized,
            "model_clients": {},
            "cache_connected": False,
            "active_streams": len(self.active_streams),
            "batch_queue_size": len(self.batch_queue)
        }
        
        try:
            # Check cache connection
            health["cache_connected"] = await self.cache.ping()
            
            # Check model clients
            for provider, client in self.model_clients.items():
                try:
                    client_health = await client.health_check()
                    health["model_clients"][provider] = client_health
                except Exception as e:
                    health["model_clients"][provider] = {"status": "unhealthy", "error": str(e)}
                    health["status"] = "degraded"
            
            # Check if any models are healthy
            healthy_models = sum(
                1 for status in health["model_clients"].values() 
                if status.get("status") == "healthy"
            )
            
            if healthy_models == 0:
                health["status"] = "unhealthy"
            elif healthy_models < len(self.model_clients):
                health["status"] = "degraded"
                
        except Exception as e:
            health["status"] = "unhealthy"
            health["error"] = str(e)
        
        return health

    async def cleanup(self):
        """Cleanup resources"""
        try:
            # Close cache connections
            if self.cache:
                await self.cache.close()
            
            # Clear active streams
            self.active_streams.clear()
            
            # Clear batch queue
            self.batch_queue.clear()
            
            self.logger.info("Advanced AI Orchestrator cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")