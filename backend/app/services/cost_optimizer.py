import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json

from app.core.config import settings
from app.core.logger import LoggerMixin
from app.database.models import ModelProvider

class OptimizationStrategy(str, Enum):
    COST_FIRST = "cost_first"           # Minimize cost
    PERFORMANCE_FIRST = "performance_first"  # Minimize latency
    BALANCED = "balanced"               # Balance cost and performance
    QUALITY_FIRST = "quality_first"     # Best quality regardless of cost

@dataclass
class ModelMetrics:
    """Metrics for a specific model"""
    provider: ModelProvider
    model_name: str
    avg_cost_per_token: float
    avg_response_time: float
    success_rate: float
    quality_score: float  # 0-1, based on user feedback
    tokens_per_second: float
    context_window: int
    capabilities: List[str]

@dataclass
class UsagePattern:
    """User's usage pattern analysis"""
    avg_tokens_per_request: float
    peak_usage_hours: List[int]
    preferred_capabilities: List[str]
    cost_sensitivity: float  # 0-1, higher = more cost sensitive
    latency_sensitivity: float  # 0-1, higher = more latency sensitive

class CostOptimizer(LoggerMixin):
    """Intelligent cost optimization and model selection service"""
    
    def __init__(self):
        self.model_metrics: Dict[str, ModelMetrics] = {}
        self.user_patterns: Dict[int, UsagePattern] = {}
        self.hourly_costs: Dict[int, float] = {}  # Cost by hour for rate limiting
        self.monthly_usage = 0.0
        self.optimization_strategy = OptimizationStrategy.BALANCED
        
        # Model pricing and performance data
        self._initialize_model_data()
    
    def _initialize_model_data(self):
        """Initialize model metrics with baseline data"""
        
        # OpenAI Models
        self.model_metrics["openai_gpt-4"] = ModelMetrics(
            provider=ModelProvider.OPENAI,
            model_name="gpt-4",
            avg_cost_per_token=0.00003,  # $0.03/1K tokens average
            avg_response_time=2.5,
            success_rate=0.99,
            quality_score=0.95,
            tokens_per_second=20,
            context_window=8192,
            capabilities=["text", "reasoning", "code", "analysis"]
        )
        
        self.model_metrics["openai_gpt-4-turbo"] = ModelMetrics(
            provider=ModelProvider.OPENAI,
            model_name="gpt-4-turbo",
            avg_cost_per_token=0.00002,  # $0.02/1K tokens average
            avg_response_time=1.8,
            success_rate=0.99,
            quality_score=0.93,
            tokens_per_second=30,
            context_window=128000,
            capabilities=["text", "reasoning", "code", "analysis", "vision"]
        )
        
        self.model_metrics["openai_gpt-3.5-turbo"] = ModelMetrics(
            provider=ModelProvider.OPENAI,
            model_name="gpt-3.5-turbo",
            avg_cost_per_token=0.000002,  # $0.002/1K tokens average
            avg_response_time=1.2,
            success_rate=0.98,
            quality_score=0.85,
            tokens_per_second=50,
            context_window=16385,
            capabilities=["text", "reasoning", "code"]
        )
        
        self.model_metrics["openai_o3-mini"] = ModelMetrics(
            provider=ModelProvider.OPENAI,
            model_name="o3-mini",
            avg_cost_per_token=0.000005,  # $0.005/1K tokens average
            avg_response_time=3.0,
            success_rate=0.99,
            quality_score=0.98,
            tokens_per_second=15,
            context_window=128000,
            capabilities=["text", "reasoning", "complex_reasoning", "analysis"]
        )
        
        # Anthropic Models
        self.model_metrics["anthropic_claude-3-opus"] = ModelMetrics(
            provider=ModelProvider.ANTHROPIC,
            model_name="claude-3-opus-20240229",
            avg_cost_per_token=0.000045,  # $0.045/1K tokens average
            avg_response_time=2.8,
            success_rate=0.99,
            quality_score=0.97,
            tokens_per_second=18,
            context_window=200000,
            capabilities=["text", "reasoning", "code", "analysis", "creative"]
        )
        
        self.model_metrics["anthropic_claude-3-sonnet"] = ModelMetrics(
            provider=ModelProvider.ANTHROPIC,
            model_name="claude-3-5-sonnet-20241022",
            avg_cost_per_token=0.000009,  # $0.009/1K tokens average
            avg_response_time=1.9,
            success_rate=0.99,
            quality_score=0.92,
            tokens_per_second=25,
            context_window=200000,
            capabilities=["text", "reasoning", "code", "analysis"]
        )
        
        self.model_metrics["anthropic_claude-3-haiku"] = ModelMetrics(
            provider=ModelProvider.ANTHROPIC,
            model_name="claude-3-haiku-20240307",
            avg_cost_per_token=0.000001,  # $0.001/1K tokens average
            avg_response_time=0.8,
            success_rate=0.98,
            quality_score=0.88,
            tokens_per_second=60,
            context_window=200000,
            capabilities=["text", "reasoning", "code"]
        )
        
        # Google Models
        self.model_metrics["google_gemini-1.5-pro"] = ModelMetrics(
            provider=ModelProvider.GOOGLE,
            model_name="gemini-1.5-pro",
            avg_cost_per_token=0.000002,  # $0.002/1K tokens average
            avg_response_time=2.2,
            success_rate=0.97,
            quality_score=0.90,
            tokens_per_second=22,
            context_window=1000000,
            capabilities=["text", "reasoning", "code", "vision", "multimodal"]
        )
        
        self.model_metrics["google_gemini-1.5-flash"] = ModelMetrics(
            provider=ModelProvider.GOOGLE,
            model_name="gemini-1.5-flash",
            avg_cost_per_token=0.0000001,  # $0.0001/1K tokens average
            avg_response_time=0.6,
            success_rate=0.96,
            quality_score=0.82,
            tokens_per_second=80,
            context_window=1000000,
            capabilities=["text", "reasoning", "code", "fast_response"]
        )
    
    async def select_optimal_provider(
        self, 
        available_providers: List[ModelProvider], 
        request
    ) -> ModelProvider:
        """Select the optimal provider based on current optimization strategy"""
        
        # Check monthly cost limits
        if await self._check_cost_limits():
            self.logger.warning("Monthly cost limit reached, selecting cheapest option")
            return await self._select_cheapest_provider(available_providers, request)
        
        # Get user pattern if available
        user_pattern = self.user_patterns.get(request.user_id) if request.user_id else None
        
        # Score each available provider
        provider_scores = {}
        for provider in available_providers:
            score = await self._calculate_provider_score(
                provider, request, user_pattern
            )
            provider_scores[provider] = score
        
        # Select provider with highest score
        optimal_provider = max(provider_scores.items(), key=lambda x: x[1])[0]
        
        self.logger.info(
            f"Selected provider: {optimal_provider.value} "
            f"(scores: {[(p.value, f'{s:.3f}') for p, s in provider_scores.items()]})"
        )
        
        return optimal_provider
    
    async def _calculate_provider_score(
        self, 
        provider: ModelProvider, 
        request, 
        user_pattern: Optional[UsagePattern]
    ) -> float:
        """Calculate a score for a provider based on multiple factors"""
        
        # Find best model for this provider
        provider_models = [
            m for m in self.model_metrics.values() 
            if m.provider == provider
        ]
        
        if not provider_models:
            return 0.0
        
        # Select best model for this provider based on capabilities
        best_model = self._select_best_model_for_capabilities(
            provider_models, 
            request.capabilities_required or []
        )
        
        if not best_model:
            return 0.0
        
        # Calculate score based on optimization strategy
        score = 0.0
        
        if self.optimization_strategy == OptimizationStrategy.COST_FIRST:
            # Prioritize low cost
            cost_score = 1.0 - min(best_model.avg_cost_per_token / 0.00005, 1.0)
            score = cost_score * 0.7 + best_model.success_rate * 0.3
            
        elif self.optimization_strategy == OptimizationStrategy.PERFORMANCE_FIRST:
            # Prioritize low latency
            latency_score = 1.0 - min(best_model.avg_response_time / 5.0, 1.0)
            throughput_score = min(best_model.tokens_per_second / 100.0, 1.0)
            score = latency_score * 0.5 + throughput_score * 0.3 + best_model.success_rate * 0.2
            
        elif self.optimization_strategy == OptimizationStrategy.QUALITY_FIRST:
            # Prioritize quality
            score = best_model.quality_score * 0.6 + best_model.success_rate * 0.4
            
        else:  # BALANCED
            # Balance all factors
            cost_score = 1.0 - min(best_model.avg_cost_per_token / 0.00005, 1.0)
            latency_score = 1.0 - min(best_model.avg_response_time / 5.0, 1.0)
            
            score = (
                cost_score * 0.25 +
                latency_score * 0.25 +
                best_model.quality_score * 0.25 +
                best_model.success_rate * 0.25
            )
        
        # Apply user pattern adjustments
        if user_pattern:
            if user_pattern.cost_sensitivity > 0.7:
                cost_factor = 1.0 - min(best_model.avg_cost_per_token / 0.00005, 1.0)
                score = score * 0.7 + cost_factor * 0.3
            
            if user_pattern.latency_sensitivity > 0.7:
                latency_factor = 1.0 - min(best_model.avg_response_time / 5.0, 1.0)
                score = score * 0.7 + latency_factor * 0.3
        
        return score
    
    def _select_best_model_for_capabilities(
        self, 
        models: List[ModelMetrics], 
        required_capabilities: List[str]
    ) -> Optional[ModelMetrics]:
        """Select the best model that supports required capabilities"""
        
        # Filter models that support all required capabilities
        compatible_models = []
        for model in models:
            if all(cap in model.capabilities for cap in required_capabilities):
                compatible_models.append(model)
        
        if not compatible_models:
            # If no model supports all capabilities, use the most capable one
            compatible_models = models
        
        # Return model with best quality score among compatible ones
        return max(compatible_models, key=lambda m: m.quality_score)
    
    async def _select_cheapest_provider(
        self, 
        available_providers: List[ModelProvider], 
        request
    ) -> ModelProvider:
        """Select the cheapest available provider"""
        
        cheapest_provider = None
        lowest_cost = float('inf')
        
        for provider in available_providers:
            provider_models = [
                m for m in self.model_metrics.values() 
                if m.provider == provider
            ]
            
            if provider_models:
                cheapest_model = min(provider_models, key=lambda m: m.avg_cost_per_token)
                if cheapest_model.avg_cost_per_token < lowest_cost:
                    lowest_cost = cheapest_model.avg_cost_per_token
                    cheapest_provider = provider
        
        return cheapest_provider or available_providers[0]
    
    async def _check_cost_limits(self) -> bool:
        """Check if we're approaching cost limits"""
        return self.monthly_usage >= settings.MONTHLY_COST_LIMIT * 0.9
    
    async def track_usage(
        self, 
        user_id: Optional[int], 
        provider: ModelProvider, 
        model_name: str,
        tokens_used: int, 
        cost: float, 
        response_time: float,
        success: bool
    ):
        """Track usage for optimization"""
        
        # Update monthly usage
        self.monthly_usage += cost
        
        # Update hourly costs for rate limiting
        current_hour = datetime.now().hour
        self.hourly_costs[current_hour] = self.hourly_costs.get(current_hour, 0) + cost
        
        # Update model metrics
        model_key = f"{provider.value}_{model_name}"
        if model_key in self.model_metrics:
            metrics = self.model_metrics[model_key]
            
            # Update running averages (simple exponential moving average)
            alpha = 0.1  # Learning rate
            
            if success:
                metrics.avg_response_time = (
                    metrics.avg_response_time * (1 - alpha) + 
                    response_time * alpha
                )
                metrics.tokens_per_second = (
                    metrics.tokens_per_second * (1 - alpha) + 
                    (tokens_used / max(response_time, 0.1)) * alpha
                )
            
            # Update success rate
            metrics.success_rate = (
                metrics.success_rate * 0.95 + 
                (1.0 if success else 0.0) * 0.05
            )
        
        # Update user patterns
        if user_id:
            await self._update_user_pattern(user_id, tokens_used, cost, response_time)
        
        self.logger.debug(
            f"Tracked usage: {provider.value} {model_name} - "
            f"${cost:.4f}, {tokens_used} tokens, {response_time:.2f}s"
        )
    
    async def _update_user_pattern(
        self, 
        user_id: int, 
        tokens_used: int, 
        cost: float, 
        response_time: float
    ):
        """Update user's usage pattern"""
        
        if user_id not in self.user_patterns:
            self.user_patterns[user_id] = UsagePattern(
                avg_tokens_per_request=tokens_used,
                peak_usage_hours=[datetime.now().hour],
                preferred_capabilities=[],
                cost_sensitivity=0.5,
                latency_sensitivity=0.5
            )
        else:
            pattern = self.user_patterns[user_id]
            alpha = 0.1
            
            # Update average tokens per request
            pattern.avg_tokens_per_request = (
                pattern.avg_tokens_per_request * (1 - alpha) + 
                tokens_used * alpha
            )
            
            # Update peak usage hours
            current_hour = datetime.now().hour
            if current_hour not in pattern.peak_usage_hours:
                pattern.peak_usage_hours.append(current_hour)
            
            # Infer cost sensitivity from usage patterns
            if cost > 0.01:  # High cost request
                if response_time > 2.0:  # User waited for expensive model
                    pattern.cost_sensitivity = max(0.0, pattern.cost_sensitivity - 0.1)
                else:
                    pattern.cost_sensitivity = min(1.0, pattern.cost_sensitivity + 0.1)
            
            # Keep only recent peak hours (last 7 days worth)
            if len(pattern.peak_usage_hours) > 24 * 7:
                pattern.peak_usage_hours = pattern.peak_usage_hours[-24*7:]
    
    async def get_cost_analytics(self) -> Dict:
        """Get cost analytics and insights"""
        
        total_requests = sum(1 for _ in self.model_metrics.values())
        
        # Calculate provider distribution
        provider_costs = {}
        for metrics in self.model_metrics.values():
            provider = metrics.provider.value
            provider_costs[provider] = provider_costs.get(provider, 0) + metrics.avg_cost_per_token
        
        # Get hourly cost distribution
        hourly_distribution = dict(sorted(self.hourly_costs.items()))
        
        return {
            "monthly_usage": self.monthly_usage,
            "monthly_limit": settings.MONTHLY_COST_LIMIT,
            "usage_percentage": (self.monthly_usage / settings.MONTHLY_COST_LIMIT) * 100,
            "provider_distribution": provider_costs,
            "hourly_costs": hourly_distribution,
            "optimization_strategy": self.optimization_strategy.value,
            "total_requests": total_requests,
            "cost_per_request": self.monthly_usage / max(total_requests, 1)
        }
    
    async def optimize_for_user(self, user_id: int, strategy: OptimizationStrategy):
        """Set optimization strategy for a specific user"""
        # This could be user-specific in a more advanced implementation
        self.optimization_strategy = strategy
        self.logger.info(f"Updated optimization strategy to {strategy.value}")
    
    async def predict_monthly_cost(self) -> Dict:
        """Predict monthly cost based on current usage patterns"""
        
        # Simple prediction based on current daily average
        days_elapsed = datetime.now().day
        daily_average = self.monthly_usage / max(days_elapsed, 1)
        predicted_monthly = daily_average * 30
        
        return {
            "current_usage": self.monthly_usage,
            "daily_average": daily_average,
            "predicted_monthly": predicted_monthly,
            "predicted_vs_limit": (predicted_monthly / settings.MONTHLY_COST_LIMIT) * 100,
            "days_remaining": 30 - days_elapsed,
            "recommended_daily_limit": (settings.MONTHLY_COST_LIMIT - self.monthly_usage) / max(30 - days_elapsed, 1)
        }
    
    def get_model_recommendations(self, capabilities: List[str]) -> List[Dict]:
        """Get model recommendations for specific capabilities"""
        
        recommendations = []
        
        for model in self.model_metrics.values():
            if all(cap in model.capabilities for cap in capabilities):
                recommendations.append({
                    "provider": model.provider.value,
                    "model": model.model_name,
                    "cost_per_token": model.avg_cost_per_token,
                    "avg_response_time": model.avg_response_time,
                    "quality_score": model.quality_score,
                    "success_rate": model.success_rate,
                    "capabilities": model.capabilities
                })
        
        # Sort by balanced score
        for rec in recommendations:
            cost_score = 1.0 - min(rec["cost_per_token"] / 0.00005, 1.0)
            latency_score = 1.0 - min(rec["avg_response_time"] / 5.0, 1.0)
            rec["score"] = (
                cost_score * 0.25 +
                latency_score * 0.25 +
                rec["quality_score"] * 0.25 +
                rec["success_rate"] * 0.25
            )
        
        return sorted(recommendations, key=lambda x: x["score"], reverse=True)