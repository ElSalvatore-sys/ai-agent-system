import asyncio
from typing import Dict, List, Optional, Any
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.advanced_ai_orchestrator import (
    AdvancedAIOrchestrator, 
    RoutingStrategy, 
    TaskComplexity,
    RoutingDecision,
    TaskAnalysis,
    UserPreferences
)
from app.core.config import settings
from app.core.logger import LoggerMixin
from app.utils.cache import CacheManager
from app.database.database import get_db


class AdvancedOrchestratorService(LoggerMixin):
    """Service layer for the Advanced AI Orchestrator"""
    
    _instance: Optional[AdvancedAIOrchestrator] = None
    _cache_manager: Optional[CacheManager] = None
    
    @classmethod
    async def get_instance(cls) -> AdvancedAIOrchestrator:
        """Get singleton instance of the Advanced AI Orchestrator"""
        if cls._instance is None:
            if cls._cache_manager is None:
                cls._cache_manager = CacheManager(settings.REDIS_URL)
                await cls._cache_manager.connect()
            
            cls._instance = AdvancedAIOrchestrator(cls._cache_manager)
            await cls._instance.initialize()
        
        return cls._instance
    
    @classmethod
    async def cleanup(cls):
        """Cleanup singleton instance"""
        if cls._instance:
            await cls._instance.cleanup()
            cls._instance = None
        
        if cls._cache_manager:
            await cls._cache_manager.close()
            cls._cache_manager = None


class ChatOrchestratorService(LoggerMixin):
    """High-level service for chat-related orchestration"""
    
    def __init__(self):
        super().__init__()
        self.orchestrator: Optional[AdvancedAIOrchestrator] = None
    
    async def _get_orchestrator(self) -> AdvancedAIOrchestrator:
        """Get orchestrator instance"""
        if self.orchestrator is None:
            self.orchestrator = await AdvancedOrchestratorService.get_instance()
        return self.orchestrator
    
    async def route_chat_message(
        self,
        message: str,
        user_id: int,
        conversation_context: Optional[Dict] = None,
        budget_limit: Optional[float] = None,
        strategy: RoutingStrategy = RoutingStrategy.BALANCED,
        stream: bool = False
    ) -> RoutingDecision:
        """
        Route a chat message to the optimal AI model
        
        Args:
            message: The user's message
            user_id: User identifier
            conversation_context: Previous conversation context
            budget_limit: Maximum cost for this request
            strategy: Routing strategy to use
            stream: Whether streaming is required
        
        Returns:
            RoutingDecision with chosen model and reasoning
        """
        orchestrator = await self._get_orchestrator()
        
        # Add conversation context to the routing decision
        context = conversation_context or {}
        if conversation_context and "messages" in conversation_context:
            # Analyze conversation history for better routing
            recent_messages = conversation_context["messages"][-5:]  # Last 5 messages
            context["recent_messages"] = recent_messages
            context["conversation_length"] = len(conversation_context["messages"])
        
        try:
            decision = await orchestrator.route_request(
                task=message,
                user_id=user_id,
                budget_limit=budget_limit,
                strategy=strategy,
                context=context,
                stream=stream
            )
            
            self.logger.info(f"Routed message for user {user_id}: {decision.chosen_provider.value}:{decision.chosen_model}")
            return decision
            
        except Exception as e:
            self.logger.error(f"Failed to route chat message: {e}")
            raise
    
    async def execute_chat_request(
        self,
        message: str,
        user_id: int,
        conversation_context: Optional[Dict] = None,
        routing_decision: Optional[RoutingDecision] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a chat request with optimal model selection and fallback
        
        Args:
            message: The user's message
            user_id: User identifier
            conversation_context: Previous conversation context
            routing_decision: Pre-computed routing decision (optional)
            **kwargs: Additional parameters
        
        Returns:
            Execution result with response and metadata
        """
        orchestrator = await self._get_orchestrator()
        
        try:
            # Route if not already done
            if not routing_decision:
                routing_decision = await self.route_chat_message(
                    message, user_id, conversation_context, **kwargs
                )
            
            # Execute the request
            result = await orchestrator.execute_request(
                task=message,
                user_id=user_id,
                routing_decision=routing_decision,
                **kwargs
            )
            
            # Add routing information to result
            result["routing_decision"] = {
                "chosen_model": f"{routing_decision.chosen_provider.value}:{routing_decision.chosen_model}",
                "reasoning": routing_decision.reasoning,
                "estimated_cost": routing_decision.estimated_cost,
                "confidence": routing_decision.confidence
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to execute chat request: {e}")
            raise
    
    async def stream_chat_response(
        self,
        message: str,
        user_id: int,
        conversation_context: Optional[Dict] = None,
        routing_decision: Optional[RoutingDecision] = None,
        **kwargs
    ):
        """
        Stream a chat response using optimal model selection
        
        Args:
            message: The user's message
            user_id: User identifier
            conversation_context: Previous conversation context
            routing_decision: Pre-computed routing decision (optional)
            **kwargs: Additional parameters
        
        Yields:
            Streaming response chunks
        """
        orchestrator = await self._get_orchestrator()
        
        try:
            # Route if not already done
            if not routing_decision:
                routing_decision = await self.route_chat_message(
                    message, user_id, conversation_context, stream=True, **kwargs
                )
            
            # Stream the response
            async for chunk in orchestrator.stream_request(
                task=message,
                user_id=user_id,
                routing_decision=routing_decision,
                **kwargs
            ):
                yield chunk
                
        except Exception as e:
            self.logger.error(f"Failed to stream chat response: {e}")
            raise
    
    async def analyze_user_patterns(self, user_id: int) -> Dict[str, Any]:
        """
        Analyze user's usage patterns and preferences
        
        Args:
            user_id: User identifier
            
        Returns:
            User pattern analysis
        """
        orchestrator = await self._get_orchestrator()
        
        try:
            # Get user preferences
            user_prefs = await orchestrator._get_user_preferences(user_id)
            
            # Get analytics for this user
            analytics = await orchestrator.get_analytics()
            
            # Calculate user-specific metrics
            user_metrics = {
                "monthly_budget": user_prefs.monthly_budget,
                "current_spend": user_prefs.current_spend,
                "budget_utilization": (user_prefs.current_spend / user_prefs.monthly_budget) * 100,
                "cost_sensitivity": user_prefs.cost_sensitivity,
                "speed_preference": user_prefs.speed_preference,
                "quality_preference": user_prefs.quality_preference,
                "preferred_models": user_prefs.preferred_models,
                "avg_tokens_per_request": user_prefs.avg_tokens_per_request,
                "peak_usage_hours": user_prefs.peak_usage_hours
            }
            
            return {
                "user_id": user_id,
                "metrics": user_metrics,
                "recommendations": await self._generate_user_recommendations(user_prefs)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to analyze user patterns: {e}")
            raise
    
    async def _generate_user_recommendations(self, user_prefs: UserPreferences) -> List[Dict[str, str]]:
        """Generate recommendations for the user based on their patterns"""
        
        recommendations = []
        
        # Budget recommendations
        if user_prefs.current_spend / user_prefs.monthly_budget > 0.8:
            recommendations.append({
                "type": "budget_warning",
                "title": "Budget Alert",
                "message": "You've used 80% of your monthly budget. Consider using cost-optimal routing.",
                "action": "switch_to_cost_optimal"
            })
        
        # Model efficiency recommendations
        if user_prefs.cost_sensitivity > 0.8:
            recommendations.append({
                "type": "cost_optimization",
                "title": "Cost Optimization",
                "message": "Based on your cost sensitivity, consider using Gemini Flash for simple tasks.",
                "action": "update_model_preferences"
            })
        
        # Usage pattern recommendations
        if user_prefs.avg_tokens_per_request > 2000:
            recommendations.append({
                "type": "efficiency",
                "title": "Token Efficiency",
                "message": "Your requests tend to be long. Consider breaking them into smaller chunks.",
                "action": "optimize_prompts"
            })
        
        return recommendations
    
    async def get_model_performance_report(self) -> Dict[str, Any]:
        """
        Get comprehensive model performance report
        
        Returns:
            Performance report with model statistics
        """
        orchestrator = await self._get_orchestrator()
        
        try:
            analytics = await orchestrator.get_analytics()
            
            # Process analytics into a report
            report = {
                "summary": {
                    "total_requests": analytics["performance"]["total_requests"],
                    "total_cost": analytics["performance"]["total_cost"],
                    "avg_cost_per_request": analytics["performance"]["avg_cost_per_request"],
                    "active_streams": analytics["active_streams"],
                    "queued_batches": analytics["queued_batches"]
                },
                "models": {},
                "recommendations": []
            }
            
            # Process model-specific data
            for model_key, metrics in analytics["models"].items():
                if metrics["total_requests"] > 0:
                    report["models"][model_key] = {
                        "total_requests": metrics["total_requests"],
                        "success_rate": metrics.get("success_rate", 0),
                        "avg_response_time": metrics.get("avg_response_time", 0),
                        "total_cost": metrics["total_cost"],
                        "avg_cost_per_request": metrics["total_cost"] / metrics["total_requests"],
                        "total_tokens": metrics["total_tokens"],
                        "cost_per_token": metrics["total_cost"] / max(metrics["total_tokens"], 1)
                    }
            
            # Generate recommendations
            if report["models"]:
                # Find most cost-effective model
                cost_effective = min(
                    report["models"].items(),
                    key=lambda x: x[1]["cost_per_token"]
                )
                
                # Find fastest model
                fastest = min(
                    report["models"].items(),
                    key=lambda x: x[1]["avg_response_time"]
                )
                
                report["recommendations"] = [
                    {
                        "type": "cost_effective",
                        "model": cost_effective[0],
                        "metric": f"${cost_effective[1]['cost_per_token']:.6f} per token"
                    },
                    {
                        "type": "fastest",
                        "model": fastest[0],
                        "metric": f"{fastest[1]['avg_response_time']:.2f}s avg response time"
                    }
                ]
            
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to get model performance report: {e}")
            raise


class BatchOrchestratorService(LoggerMixin):
    """Service for batch processing using the orchestrator"""
    
    def __init__(self):
        super().__init__()
        self.orchestrator: Optional[AdvancedAIOrchestrator] = None
    
    async def _get_orchestrator(self) -> AdvancedAIOrchestrator:
        """Get orchestrator instance"""
        if self.orchestrator is None:
            self.orchestrator = await AdvancedOrchestratorService.get_instance()
        return self.orchestrator
    
    async def submit_batch(
        self,
        requests: List[Dict[str, Any]],
        user_id: int,
        priority: int = 0
    ) -> str:
        """
        Submit a batch of requests for processing
        
        Args:
            requests: List of request dictionaries
            user_id: User identifier
            priority: Processing priority (higher = more urgent)
            
        Returns:
            Batch ID for tracking
        """
        orchestrator = await self._get_orchestrator()
        
        try:
            batch_id = await orchestrator.batch_process(
                requests=requests,
                user_id=user_id,
                priority=priority
            )
            
            self.logger.info(f"Submitted batch {batch_id} with {len(requests)} requests for user {user_id}")
            return batch_id
            
        except Exception as e:
            self.logger.error(f"Failed to submit batch: {e}")
            raise
    
    async def get_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """
        Get status of a batch processing job
        
        Args:
            batch_id: Batch identifier
            
        Returns:
            Batch status information
        """
        orchestrator = await self._get_orchestrator()
        
        try:
            # Check if batch is still in queue
            if batch_id in orchestrator.batch_queue:
                batch_request = orchestrator.batch_queue[batch_id]
                return {
                    "batch_id": batch_id,
                    "status": "queued",
                    "total_requests": len(batch_request.requests),
                    "created_at": batch_request.created_at.isoformat(),
                    "priority": batch_request.priority
                }
            
            # Check cache for results
            cache_key = f"batch_results:{batch_id}"
            results = await orchestrator.cache.get(cache_key)
            
            if results:
                successful = sum(1 for r in results if not isinstance(r, Exception))
                failed = len(results) - successful
                
                return {
                    "batch_id": batch_id,
                    "status": "completed",
                    "total_requests": len(results),
                    "successful": successful,
                    "failed": failed,
                    "results": results
                }
            
            return {
                "batch_id": batch_id,
                "status": "not_found"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get batch status: {e}")
            return {
                "batch_id": batch_id,
                "status": "error",
                "error": str(e)
            }


# Dependency functions for FastAPI
async def get_chat_orchestrator() -> ChatOrchestratorService:
    """Dependency to get chat orchestrator service"""
    return ChatOrchestratorService()


async def get_batch_orchestrator() -> BatchOrchestratorService:
    """Dependency to get batch orchestrator service"""
    return BatchOrchestratorService()


async def get_advanced_orchestrator() -> AdvancedAIOrchestrator:
    """Dependency to get advanced orchestrator instance"""
    return await AdvancedOrchestratorService.get_instance()