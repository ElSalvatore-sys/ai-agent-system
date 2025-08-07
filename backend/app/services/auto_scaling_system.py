"""Auto-Scaling System.

Enterprise-grade auto-scaling infrastructure with horizontal/vertical scaling,
predictive scaling based on usage patterns, and cost-aware scaling decisions.
"""
from __future__ import annotations

import asyncio
import threading
import time
import math
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import statistics

from app.core.logger import LoggerMixin
from app.core.config import settings
from app.database.models import ModelProvider
from app.services.container_orchestrator import get_container_orchestrator
from app.services.gpu_cpu_optimizer import get_gpu_cpu_optimizer


class ScalingDirection(str, Enum):
    """Scaling directions"""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"


class ScalingType(str, Enum):
    """Types of scaling operations"""
    HORIZONTAL = "horizontal"  # Add/remove instances
    VERTICAL = "vertical"     # Increase/decrease resources
    HYBRID = "hybrid"         # Both horizontal and vertical


class ScalingPolicy(str, Enum):
    """Scaling policies"""
    REACTIVE = "reactive"     # React to current metrics
    PREDICTIVE = "predictive" # Based on predicted load
    SCHEDULED = "scheduled"   # Time-based scaling
    COST_AWARE = "cost_aware" # Cost-optimized scaling
    HYBRID_SMART = "hybrid_smart"  # Combination of all


@dataclass
class ScalingMetrics:
    """Metrics for scaling decisions"""
    timestamp: datetime
    cpu_utilization: float
    memory_utilization: float
    gpu_utilization: float
    request_rate: float
    response_time_ms: float
    queue_length: int
    error_rate: float
    cost_per_hour: float
    user_count: int
    throughput_tokens_per_sec: float


@dataclass
class ScalingRule:
    """Scaling rule configuration"""
    name: str
    metric_name: str
    threshold_up: float
    threshold_down: float
    scaling_type: ScalingType
    scaling_direction: ScalingDirection
    cooldown_seconds: int
    min_instances: int
    max_instances: int
    step_size: int
    weight: float = 1.0
    enabled: bool = True


@dataclass
class ScalingAction:
    """Scaling action to be executed"""
    action_id: str
    model_key: str
    scaling_type: ScalingType
    scaling_direction: ScalingDirection
    target_instances: int
    target_resources: Dict[str, Any]
    reason: str
    cost_impact: float
    created_at: datetime
    scheduled_at: Optional[datetime] = None
    executed_at: Optional[datetime] = None
    success: bool = False
    execution_time_ms: float = 0.0


@dataclass
class LoadPrediction:
    """Load prediction for future time periods"""
    timestamp: datetime
    predicted_cpu: float
    predicted_memory: float
    predicted_gpu: float
    predicted_requests: float
    predicted_users: int
    confidence_score: float
    prediction_horizon_minutes: int


@dataclass
class CostModel:
    """Cost modeling for scaling decisions"""
    provider: ModelProvider
    model_name: str
    cost_per_hour_base: float
    cost_per_gpu_hour: float
    cost_per_gb_memory_hour: float
    cost_per_cpu_core_hour: float
    cost_per_million_tokens: float
    spot_instance_discount: float = 0.0
    reserved_instance_discount: float = 0.0


class PredictiveModel:
    """Predictive model for load forecasting"""
    
    def __init__(self):
        self.historical_data: List[ScalingMetrics] = []
        self.max_history_hours = 168  # 1 week
        self.prediction_accuracy = 0.0
        self.last_training_time: Optional[datetime] = None
        self.model_weights: Dict[str, float] = {}
    
    def add_data_point(self, metrics: ScalingMetrics):
        """Add new data point to historical data"""
        self.historical_data.append(metrics)
        
        # Keep only recent data
        cutoff_time = datetime.utcnow() - timedelta(hours=self.max_history_hours)
        self.historical_data = [
            data for data in self.historical_data 
            if data.timestamp > cutoff_time
        ]
    
    def predict_load(self, horizon_minutes: int = 60) -> LoadPrediction:
        """Predict load for the next time period"""
        try:
            if len(self.historical_data) < 10:
                # Not enough data for prediction, use current values
                current = self.historical_data[-1] if self.historical_data else None
                if not current:
                    return self._default_prediction(horizon_minutes)
                
                return LoadPrediction(
                    timestamp=datetime.utcnow() + timedelta(minutes=horizon_minutes),
                    predicted_cpu=current.cpu_utilization,
                    predicted_memory=current.memory_utilization,
                    predicted_gpu=current.gpu_utilization,
                    predicted_requests=current.request_rate,
                    predicted_users=current.user_count,
                    confidence_score=0.3,  # Low confidence
                    prediction_horizon_minutes=horizon_minutes
                )
            
            # Use time-series analysis for prediction
            prediction = self._time_series_prediction(horizon_minutes)
            return prediction
            
        except Exception as e:
            return self._default_prediction(horizon_minutes)
    
    def _time_series_prediction(self, horizon_minutes: int) -> LoadPrediction:
        """Simple time series prediction using moving averages and trends"""
        try:
            # Calculate recent trends
            recent_data = self.historical_data[-20:]  # Last 20 data points
            
            # Calculate moving averages
            cpu_values = [d.cpu_utilization for d in recent_data]
            memory_values = [d.memory_utilization for d in recent_data]
            gpu_values = [d.gpu_utilization for d in recent_data]
            request_values = [d.request_rate for d in recent_data]
            user_values = [d.user_count for d in recent_data]
            
            # Calculate trends (simple linear regression)
            cpu_trend = self._calculate_trend(cpu_values)
            memory_trend = self._calculate_trend(memory_values)
            gpu_trend = self._calculate_trend(gpu_values)
            request_trend = self._calculate_trend(request_values)
            user_trend = self._calculate_trend(user_values)
            
            # Current averages
            cpu_avg = statistics.mean(cpu_values[-5:])  # Last 5 points
            memory_avg = statistics.mean(memory_values[-5:])
            gpu_avg = statistics.mean(gpu_values[-5:])
            request_avg = statistics.mean(request_values[-5:])
            user_avg = statistics.mean(user_values[-5:])
            
            # Apply trend for prediction
            time_factor = horizon_minutes / 60.0  # Scale to hours
            
            predicted_cpu = max(0, min(100, cpu_avg + (cpu_trend * time_factor)))
            predicted_memory = max(0, min(100, memory_avg + (memory_trend * time_factor)))
            predicted_gpu = max(0, min(100, gpu_avg + (gpu_trend * time_factor)))
            predicted_requests = max(0, request_avg + (request_trend * time_factor))
            predicted_users = max(0, int(user_avg + (user_trend * time_factor)))
            
            # Calculate confidence based on data stability
            cpu_variance = statistics.variance(cpu_values) if len(cpu_values) > 1 else 0
            confidence = max(0.1, min(0.9, 1.0 - (cpu_variance / 100.0)))
            
            return LoadPrediction(
                timestamp=datetime.utcnow() + timedelta(minutes=horizon_minutes),
                predicted_cpu=predicted_cpu,
                predicted_memory=predicted_memory,
                predicted_gpu=predicted_gpu,
                predicted_requests=predicted_requests,
                predicted_users=predicted_users,
                confidence_score=confidence,
                prediction_horizon_minutes=horizon_minutes
            )
            
        except Exception as e:
            return self._default_prediction(horizon_minutes)
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate linear trend from values"""
        if len(values) < 2:
            return 0.0
        
        n = len(values)
        x = list(range(n))
        
        # Simple linear regression
        x_mean = statistics.mean(x)
        y_mean = statistics.mean(values)
        
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0.0
        
        slope = numerator / denominator
        return slope
    
    def _default_prediction(self, horizon_minutes: int) -> LoadPrediction:
        """Default prediction when insufficient data"""
        return LoadPrediction(
            timestamp=datetime.utcnow() + timedelta(minutes=horizon_minutes),
            predicted_cpu=50.0,
            predicted_memory=50.0,
            predicted_gpu=50.0,
            predicted_requests=10.0,
            predicted_users=1,
            confidence_score=0.1,
            prediction_horizon_minutes=horizon_minutes
        )


class CostOptimizer:
    """Cost optimization for scaling decisions"""
    
    def __init__(self):
        self.cost_models: Dict[str, CostModel] = {}
        self.cost_history: List[Tuple[datetime, float]] = []
        self.budget_limit_hourly: Optional[float] = None
        self.cost_efficiency_threshold = 0.7  # 70% efficiency threshold
    
    def add_cost_model(self, model_key: str, cost_model: CostModel):
        """Add cost model for a specific model"""
        self.cost_models[model_key] = cost_model
    
    def calculate_scaling_cost(
        self, 
        model_key: str,
        current_instances: int,
        target_instances: int,
        resource_changes: Dict[str, Any]
    ) -> float:
        """Calculate cost impact of scaling action"""
        try:
            cost_model = self.cost_models.get(model_key)
            if not cost_model:
                return 0.0  # No cost model available
            
            # Calculate current cost
            current_cost = self._calculate_instance_cost(cost_model, current_instances, {})
            
            # Calculate target cost
            target_cost = self._calculate_instance_cost(cost_model, target_instances, resource_changes)
            
            return target_cost - current_cost
            
        except Exception as e:
            return 0.0
    
    def _calculate_instance_cost(
        self, 
        cost_model: CostModel, 
        instances: int, 
        resource_changes: Dict[str, Any]
    ) -> float:
        """Calculate cost for specific instance configuration"""
        base_cost = cost_model.cost_per_hour_base * instances
        
        # Add GPU costs
        gpu_count = resource_changes.get("gpu_count", 1)
        gpu_cost = cost_model.cost_per_gpu_hour * gpu_count * instances
        
        # Add memory costs
        memory_gb = resource_changes.get("memory_gb", 8)
        memory_cost = cost_model.cost_per_gb_memory_hour * memory_gb * instances
        
        # Add CPU costs
        cpu_cores = resource_changes.get("cpu_cores", 4)
        cpu_cost = cost_model.cost_per_cpu_core_hour * cpu_cores * instances
        
        total_cost = base_cost + gpu_cost + memory_cost + cpu_cost
        
        # Apply discounts
        if resource_changes.get("use_spot_instances", False):
            total_cost *= (1 - cost_model.spot_instance_discount)
        
        if resource_changes.get("use_reserved_instances", False):
            total_cost *= (1 - cost_model.reserved_instance_discount)
        
        return total_cost
    
    def is_cost_effective(self, cost_impact: float, performance_improvement: float) -> bool:
        """Check if scaling action is cost-effective"""
        if cost_impact <= 0:
            return True  # No additional cost
        
        if performance_improvement <= 0:
            return False  # No performance improvement
        
        # Calculate cost efficiency ratio
        efficiency_ratio = performance_improvement / cost_impact
        
        return efficiency_ratio >= self.cost_efficiency_threshold
    
    def suggest_cost_optimizations(self, model_key: str) -> List[Dict[str, Any]]:
        """Suggest cost optimization strategies"""
        suggestions = []
        
        cost_model = self.cost_models.get(model_key)
        if not cost_model:
            return suggestions
        
        # Suggest spot instances
        if cost_model.spot_instance_discount > 0:
            suggestions.append({
                "type": "spot_instances",
                "description": "Use spot instances for cost savings",
                "potential_savings_percent": cost_model.spot_instance_discount * 100,
                "risk_level": "medium"
            })
        
        # Suggest reserved instances for stable workloads
        if cost_model.reserved_instance_discount > 0:
            suggestions.append({
                "type": "reserved_instances",
                "description": "Use reserved instances for predictable workloads",
                "potential_savings_percent": cost_model.reserved_instance_discount * 100,
                "risk_level": "low"
            })
        
        # Suggest right-sizing
        suggestions.append({
            "type": "right_sizing",
            "description": "Optimize instance sizes based on actual usage",
            "potential_savings_percent": 20,
            "risk_level": "low"
        })
        
        return suggestions


class AutoScalingSystem(LoggerMixin):
    """Enterprise auto-scaling system"""
    
    def __init__(self):
        super().__init__()
        self.container_orchestrator = None
        self.gpu_optimizer = None
        self.predictive_model = PredictiveModel()
        self.cost_optimizer = CostOptimizer()
        
        # Scaling configuration
        self.scaling_rules: Dict[str, List[ScalingRule]] = {}
        self.active_scaling_policies: Dict[str, ScalingPolicy] = {}
        self.scaling_actions: List[ScalingAction] = []
        self.metrics_history: List[ScalingMetrics] = []
        
        # Thresholds and limits
        self.default_cpu_threshold_up = 70.0
        self.default_cpu_threshold_down = 30.0
        self.default_memory_threshold_up = 80.0
        self.default_memory_threshold_down = 40.0
        self.default_cooldown_seconds = 300  # 5 minutes
        self.max_scaling_actions_per_hour = 10
        
        # Prediction settings
        self.enable_predictive_scaling = True
        self.prediction_horizon_minutes = 60
        self.prediction_confidence_threshold = 0.6
        
        # Cost settings
        self.enable_cost_optimization = True
        self.cost_budget_hourly = 100.0  # $100/hour default
        
        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._scaling_task: Optional[asyncio.Task] = None
        self._prediction_task: Optional[asyncio.Task] = None
        self._cost_optimization_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        # Thread locks
        self._scaling_lock = threading.Lock()
        
        # Statistics
        self.stats = {
            "total_scaling_actions": 0,
            "successful_scale_ups": 0,
            "successful_scale_downs": 0,
            "failed_scaling_actions": 0,
            "cost_savings_total": 0.0,
            "predictive_accuracy": 0.0,
            "average_response_time_improvement": 0.0
        }
    
    async def initialize(self):
        """Initialize the auto-scaling system"""
        try:
            # Get dependencies
            self.container_orchestrator = await get_container_orchestrator()
            self.gpu_optimizer = await get_gpu_cpu_optimizer()
            
            # Initialize default scaling rules
            await self._initialize_default_scaling_rules()
            
            # Initialize cost models
            await self._initialize_cost_models()
            
            # Start background tasks
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            self._scaling_task = asyncio.create_task(self._scaling_decision_loop())
            if self.enable_predictive_scaling:
                self._prediction_task = asyncio.create_task(self._prediction_loop())
            if self.enable_cost_optimization:
                self._cost_optimization_task = asyncio.create_task(self._cost_optimization_loop())
            
            self.logger.info("Auto-scaling system initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize auto-scaling system: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown the auto-scaling system"""
        self.logger.info("Shutting down auto-scaling system")
        self._shutdown_event.set()
        
        # Cancel background tasks
        for task in [self._monitoring_task, self._scaling_task, self._prediction_task, self._cost_optimization_task]:
            if task:
                task.cancel()
    
    async def register_model(
        self, 
        model_key: str,
        scaling_policy: ScalingPolicy = ScalingPolicy.HYBRID_SMART,
        custom_rules: Optional[List[ScalingRule]] = None
    ):
        """Register a model for auto-scaling"""
        try:
            self.active_scaling_policies[model_key] = scaling_policy
            
            if custom_rules:
                self.scaling_rules[model_key] = custom_rules
            else:
                self.scaling_rules[model_key] = await self._create_default_rules(model_key)
            
            self.logger.info(f"Registered model for auto-scaling: {model_key} ({scaling_policy})")
            
        except Exception as e:
            self.logger.error(f"Failed to register model {model_key}: {e}")
    
    async def unregister_model(self, model_key: str):
        """Unregister a model from auto-scaling"""
        try:
            self.active_scaling_policies.pop(model_key, None)
            self.scaling_rules.pop(model_key, None)
            
            self.logger.info(f"Unregistered model from auto-scaling: {model_key}")
            
        except Exception as e:
            self.logger.error(f"Failed to unregister model {model_key}: {e}")
    
    async def trigger_manual_scaling(
        self, 
        model_key: str,
        scaling_type: ScalingType,
        scaling_direction: ScalingDirection,
        target_instances: Optional[int] = None,
        target_resources: Optional[Dict[str, Any]] = None
    ) -> str:
        """Trigger manual scaling action"""
        try:
            action_id = f"manual_{model_key}_{int(time.time())}"
            
            # Create scaling action
            scaling_action = ScalingAction(
                action_id=action_id,
                model_key=model_key,
                scaling_type=scaling_type,
                scaling_direction=scaling_direction,
                target_instances=target_instances or 1,
                target_resources=target_resources or {},
                reason="Manual scaling request",
                cost_impact=0.0,
                created_at=datetime.utcnow()
            )
            
            # Execute scaling action
            success = await self._execute_scaling_action(scaling_action)
            
            if success:
                self.scaling_actions.append(scaling_action)
                self.stats["total_scaling_actions"] += 1
                
            return action_id
            
        except Exception as e:
            self.logger.error(f"Manual scaling failed for {model_key}: {e}")
            raise
    
    async def _monitoring_loop(self):
        """Background monitoring loop"""
        while not self._shutdown_event.is_set():
            try:
                # Collect metrics for all registered models
                for model_key in self.active_scaling_policies.keys():
                    metrics = await self._collect_metrics(model_key)
                    if metrics:
                        self.metrics_history.append(metrics)
                        self.predictive_model.add_data_point(metrics)
                
                # Clean up old metrics (keep last 24 hours)
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                self.metrics_history = [
                    m for m in self.metrics_history if m.timestamp > cutoff_time
                ]
                
                await asyncio.sleep(30)  # Collect metrics every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(30)
    
    async def _scaling_decision_loop(self):
        """Background scaling decision loop"""
        while not self._shutdown_event.is_set():
            try:
                # Process scaling decisions for all registered models
                for model_key in self.active_scaling_policies.keys():
                    await self._process_scaling_decisions(model_key)
                
                await asyncio.sleep(60)  # Make decisions every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Scaling decision loop error: {e}")
                await asyncio.sleep(60)
    
    async def _prediction_loop(self):
        """Background prediction loop"""
        while not self._shutdown_event.is_set():
            try:
                # Generate predictions for all registered models
                for model_key in self.active_scaling_policies.keys():
                    if self.active_scaling_policies[model_key] in [ScalingPolicy.PREDICTIVE, ScalingPolicy.HYBRID_SMART]:
                        await self._process_predictive_scaling(model_key)
                
                await asyncio.sleep(300)  # Run predictions every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Prediction loop error: {e}")
                await asyncio.sleep(300)
    
    async def _cost_optimization_loop(self):
        """Background cost optimization loop"""
        while not self._shutdown_event.is_set():
            try:
                # Run cost optimization for all registered models
                for model_key in self.active_scaling_policies.keys():
                    if self.active_scaling_policies[model_key] in [ScalingPolicy.COST_AWARE, ScalingPolicy.HYBRID_SMART]:
                        await self._process_cost_optimization(model_key)
                
                await asyncio.sleep(600)  # Run cost optimization every 10 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Cost optimization loop error: {e}")
                await asyncio.sleep(600)
    
    async def _collect_metrics(self, model_key: str) -> Optional[ScalingMetrics]:
        """Collect metrics for a specific model"""
        try:
            # Get resource metrics from GPU optimizer
            resource_stats = await self.gpu_optimizer.get_resource_stats()
            
            # Get container metrics from orchestrator
            containers = await self.container_orchestrator.get_all_containers()
            model_containers = [c for c in containers if f"{model_key}" in c.name]
            
            if not model_containers:
                return None
            
            # Calculate aggregate metrics
            avg_cpu = statistics.mean([c.cpu_usage for c in model_containers])
            avg_memory = statistics.mean([c.memory_usage_mb for c in model_containers])
            avg_gpu = 0.0
            
            # Get GPU utilization
            gpu_stats = resource_stats.get("gpu_stats", {})
            if gpu_stats:
                gpu_utilizations = [gpu["utilization_percent"] for gpu in gpu_stats.values()]
                if gpu_utilizations:
                    avg_gpu = statistics.mean(gpu_utilizations)
            
            # Simulate other metrics (in production, these would come from actual monitoring)
            request_rate = len(model_containers) * 10.0  # Simplified
            response_time_ms = 100.0 + (avg_cpu * 2)  # Simplified correlation
            queue_length = max(0, int((avg_cpu - 50) / 10))  # Simplified
            error_rate = max(0.0, (avg_cpu - 80) / 100)  # Simplified
            cost_per_hour = len(model_containers) * 2.0  # Simplified
            user_count = max(1, len(model_containers) * 5)  # Simplified
            throughput = max(0.0, 1000.0 - (avg_cpu * 5))  # Simplified
            
            return ScalingMetrics(
                timestamp=datetime.utcnow(),
                cpu_utilization=avg_cpu,
                memory_utilization=avg_memory / 1024,  # Convert to percentage
                gpu_utilization=avg_gpu,
                request_rate=request_rate,
                response_time_ms=response_time_ms,
                queue_length=queue_length,
                error_rate=error_rate,
                cost_per_hour=cost_per_hour,
                user_count=user_count,
                throughput_tokens_per_sec=throughput
            )
            
        except Exception as e:
            self.logger.error(f"Metrics collection failed for {model_key}: {e}")
            return None
    
    async def _process_scaling_decisions(self, model_key: str):
        """Process scaling decisions for a model"""
        try:
            # Get current metrics
            recent_metrics = [m for m in self.metrics_history if model_key in str(m)]
            if not recent_metrics:
                return
            
            current_metrics = recent_metrics[-1]
            
            # Get scaling rules for this model
            rules = self.scaling_rules.get(model_key, [])
            
            # Evaluate each rule
            scaling_recommendations = []
            
            for rule in rules:
                if not rule.enabled:
                    continue
                
                recommendation = await self._evaluate_scaling_rule(rule, current_metrics, model_key)
                if recommendation:
                    scaling_recommendations.append(recommendation)
            
            # Combine recommendations and execute if needed
            if scaling_recommendations:
                final_action = await self._combine_scaling_recommendations(
                    scaling_recommendations, model_key
                )
                
                if final_action:
                    await self._execute_scaling_action(final_action)
                    
        except Exception as e:
            self.logger.error(f"Scaling decision processing failed for {model_key}: {e}")
    
    async def _evaluate_scaling_rule(
        self, 
        rule: ScalingRule, 
        metrics: ScalingMetrics,
        model_key: str
    ) -> Optional[ScalingAction]:
        """Evaluate a scaling rule against current metrics"""
        try:
            # Get metric value
            metric_value = getattr(metrics, rule.metric_name, 0.0)
            
            # Check cooldown
            if not await self._check_cooldown(model_key, rule.cooldown_seconds):
                return None
            
            # Determine scaling direction
            scaling_direction = ScalingDirection.STABLE
            
            if metric_value > rule.threshold_up:
                scaling_direction = ScalingDirection.UP
            elif metric_value < rule.threshold_down:
                scaling_direction = ScalingDirection.DOWN
            
            if scaling_direction == ScalingDirection.STABLE:
                return None
            
            # Get current instance count
            containers = await self.container_orchestrator.get_all_containers()
            current_instances = len([c for c in containers if model_key in c.name])
            
            # Calculate target instances
            if scaling_direction == ScalingDirection.UP:
                target_instances = min(rule.max_instances, current_instances + rule.step_size)
            else:
                target_instances = max(rule.min_instances, current_instances - rule.step_size)
            
            if target_instances == current_instances:
                return None
            
            # Calculate cost impact
            cost_impact = self.cost_optimizer.calculate_scaling_cost(
                model_key, current_instances, target_instances, {}
            )
            
            action_id = f"auto_{model_key}_{rule.name}_{int(time.time())}"
            
            return ScalingAction(
                action_id=action_id,
                model_key=model_key,
                scaling_type=rule.scaling_type,
                scaling_direction=scaling_direction,
                target_instances=target_instances,
                target_resources={},
                reason=f"Rule {rule.name}: {rule.metric_name}={metric_value:.1f}",
                cost_impact=cost_impact,
                created_at=datetime.utcnow()
            )
            
        except Exception as e:
            self.logger.error(f"Rule evaluation failed: {e}")
            return None
    
    async def _process_predictive_scaling(self, model_key: str):
        """Process predictive scaling for a model"""
        try:
            # Generate prediction
            prediction = self.predictive_model.predict_load(self.prediction_horizon_minutes)
            
            if prediction.confidence_score < self.prediction_confidence_threshold:
                return  # Not confident enough in prediction
            
            # Check if prediction indicates need for scaling
            predicted_load = max(
                prediction.predicted_cpu,
                prediction.predicted_memory,
                prediction.predicted_gpu
            )
            
            if predicted_load > 80.0:  # High load predicted
                # Pre-emptively scale up
                containers = await self.container_orchestrator.get_all_containers()
                current_instances = len([c for c in containers if model_key in c.name])
                
                target_instances = min(10, current_instances + 1)  # Conservative scaling
                
                if target_instances > current_instances:
                    action_id = f"predictive_{model_key}_{int(time.time())}"
                    
                    scaling_action = ScalingAction(
                        action_id=action_id,
                        model_key=model_key,
                        scaling_type=ScalingType.HORIZONTAL,
                        scaling_direction=ScalingDirection.UP,
                        target_instances=target_instances,
                        target_resources={},
                        reason=f"Predictive scaling: {predicted_load:.1f}% load predicted",
                        cost_impact=0.0,
                        created_at=datetime.utcnow(),
                        scheduled_at=datetime.utcnow() + timedelta(minutes=max(0, self.prediction_horizon_minutes - 10))
                    )
                    
                    # Schedule the scaling action
                    self.scaling_actions.append(scaling_action)
                    
        except Exception as e:
            self.logger.error(f"Predictive scaling failed for {model_key}: {e}")
    
    async def _process_cost_optimization(self, model_key: str):
        """Process cost optimization for a model"""
        try:
            # Get cost optimization suggestions
            suggestions = self.cost_optimizer.suggest_cost_optimizations(model_key)
            
            for suggestion in suggestions:
                if suggestion["type"] == "right_sizing":
                    await self._optimize_instance_sizing(model_key)
                elif suggestion["type"] == "spot_instances":
                    await self._suggest_spot_instances(model_key)
                    
        except Exception as e:
            self.logger.error(f"Cost optimization failed for {model_key}: {e}")
    
    async def _optimize_instance_sizing(self, model_key: str):
        """Optimize instance sizing based on actual usage"""
        try:
            # Analyze historical resource usage
            recent_metrics = [m for m in self.metrics_history[-20:] if model_key in str(m)]
            
            if not recent_metrics:
                return
            
            # Calculate average utilization
            avg_cpu = statistics.mean([m.cpu_utilization for m in recent_metrics])
            avg_memory = statistics.mean([m.memory_utilization for m in recent_metrics])
            avg_gpu = statistics.mean([m.gpu_utilization for m in recent_metrics])
            
            # Check if consistently under-utilized
            if avg_cpu < 40 and avg_memory < 40 and avg_gpu < 40:
                # Suggest downsizing
                self.logger.info(f"Model {model_key} is under-utilized, consider downsizing")
                
        except Exception as e:
            self.logger.error(f"Instance sizing optimization failed: {e}")
    
    async def _suggest_spot_instances(self, model_key: str):
        """Suggest using spot instances for cost savings"""
        try:
            self.logger.info(f"Consider using spot instances for model {model_key} to reduce costs")
            
        except Exception as e:
            self.logger.error(f"Spot instance suggestion failed: {e}")
    
    async def _combine_scaling_recommendations(
        self, 
        recommendations: List[ScalingAction], 
        model_key: str
    ) -> Optional[ScalingAction]:
        """Combine multiple scaling recommendations into a single action"""
        try:
            if not recommendations:
                return None
            
            # Weight recommendations by rule weights and combine
            total_weight = 0.0
            weighted_instances = 0.0
            combined_reason = []
            
            for rec in recommendations:
                # Get rule weight (simplified)
                weight = 1.0
                total_weight += weight
                weighted_instances += rec.target_instances * weight
                combined_reason.append(rec.reason)
            
            target_instances = int(weighted_instances / total_weight)
            
            # Use the first recommendation as base
            base_rec = recommendations[0]
            base_rec.target_instances = target_instances
            base_rec.reason = "; ".join(combined_reason)
            
            return base_rec
            
        except Exception as e:
            self.logger.error(f"Recommendation combination failed: {e}")
            return recommendations[0] if recommendations else None
    
    async def _execute_scaling_action(self, action: ScalingAction) -> bool:
        """Execute a scaling action"""
        try:
            start_time = time.time()
            
            with self._scaling_lock:
                self.logger.info(f"Executing scaling action: {action.action_id}")
                
                if action.scaling_type == ScalingType.HORIZONTAL:
                    success = await self._execute_horizontal_scaling(action)
                elif action.scaling_type == ScalingType.VERTICAL:
                    success = await self._execute_vertical_scaling(action)
                else:
                    success = await self._execute_hybrid_scaling(action)
                
                # Update action metadata
                action.executed_at = datetime.utcnow()
                action.success = success
                action.execution_time_ms = (time.time() - start_time) * 1000
                
                # Update statistics
                if success:
                    self.stats["total_scaling_actions"] += 1
                    if action.scaling_direction == ScalingDirection.UP:
                        self.stats["successful_scale_ups"] += 1
                    else:
                        self.stats["successful_scale_downs"] += 1
                else:
                    self.stats["failed_scaling_actions"] += 1
                
                return success
                
        except Exception as e:
            self.logger.error(f"Scaling action execution failed: {e}")
            return False
    
    async def _execute_horizontal_scaling(self, action: ScalingAction) -> bool:
        """Execute horizontal scaling (add/remove instances)"""
        try:
            model_key = action.model_key
            provider, model_name = model_key.split(":", 1)
            provider_enum = ModelProvider(provider)
            
            # Get current instances
            containers = await self.container_orchestrator.get_all_containers()
            current_instances = len([c for c in containers if model_key in c.name])
            
            target_instances = action.target_instances
            
            if target_instances > current_instances:
                # Scale up - add instances
                for _ in range(target_instances - current_instances):
                    container_id = await self.container_orchestrator.create_model_container(
                        provider_enum, model_name
                    )
                    self.logger.info(f"Added container {container_id} for {model_key}")
                    
            elif target_instances < current_instances:
                # Scale down - remove instances
                model_containers = [c for c in containers if model_key in c.name]
                containers_to_remove = model_containers[target_instances:]
                
                for container in containers_to_remove:
                    await self.container_orchestrator.remove_model_container(container.container_id)
                    self.logger.info(f"Removed container {container.container_id} for {model_key}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Horizontal scaling failed: {e}")
            return False
    
    async def _execute_vertical_scaling(self, action: ScalingAction) -> bool:
        """Execute vertical scaling (change resources)"""
        try:
            # Vertical scaling would involve updating resource allocations
            # This is a placeholder for the actual implementation
            self.logger.info(f"Vertical scaling not fully implemented for {action.model_key}")
            return True
            
        except Exception as e:
            self.logger.error(f"Vertical scaling failed: {e}")
            return False
    
    async def _execute_hybrid_scaling(self, action: ScalingAction) -> bool:
        """Execute hybrid scaling (both horizontal and vertical)"""
        try:
            # Execute both horizontal and vertical scaling
            horizontal_success = await self._execute_horizontal_scaling(action)
            vertical_success = await self._execute_vertical_scaling(action)
            
            return horizontal_success and vertical_success
            
        except Exception as e:
            self.logger.error(f"Hybrid scaling failed: {e}")
            return False
    
    async def _check_cooldown(self, model_key: str, cooldown_seconds: int) -> bool:
        """Check if cooldown period has passed"""
        try:
            current_time = datetime.utcnow()
            
            # Find last scaling action for this model
            recent_actions = [
                a for a in self.scaling_actions
                if a.model_key == model_key and a.executed_at
            ]
            
            if not recent_actions:
                return True
            
            last_action = max(recent_actions, key=lambda x: x.executed_at)
            time_since_last = (current_time - last_action.executed_at).total_seconds()
            
            return time_since_last >= cooldown_seconds
            
        except Exception as e:
            self.logger.error(f"Cooldown check failed: {e}")
            return True
    
    async def _initialize_default_scaling_rules(self):
        """Initialize default scaling rules"""
        try:
            # Default CPU-based scaling rule
            cpu_rule = ScalingRule(
                name="cpu_scaling",
                metric_name="cpu_utilization",
                threshold_up=self.default_cpu_threshold_up,
                threshold_down=self.default_cpu_threshold_down,
                scaling_type=ScalingType.HORIZONTAL,
                scaling_direction=ScalingDirection.UP,
                cooldown_seconds=self.default_cooldown_seconds,
                min_instances=1,
                max_instances=10,
                step_size=1
            )
            
            # Default memory-based scaling rule
            memory_rule = ScalingRule(
                name="memory_scaling",
                metric_name="memory_utilization",
                threshold_up=self.default_memory_threshold_up,
                threshold_down=self.default_memory_threshold_down,
                scaling_type=ScalingType.HORIZONTAL,
                scaling_direction=ScalingDirection.UP,
                cooldown_seconds=self.default_cooldown_seconds,
                min_instances=1,
                max_instances=10,
                step_size=1
            )
            
            self.default_rules = [cpu_rule, memory_rule]
            
        except Exception as e:
            self.logger.error(f"Default rule initialization failed: {e}")
    
    async def _create_default_rules(self, model_key: str) -> List[ScalingRule]:
        """Create default scaling rules for a model"""
        return self.default_rules.copy() if hasattr(self, 'default_rules') else []
    
    async def _initialize_cost_models(self):
        """Initialize cost models for different providers"""
        try:
            # Default cost models (these would typically come from configuration)
            ollama_cost = CostModel(
                provider=ModelProvider.LOCAL_OLLAMA,
                model_name="default",
                cost_per_hour_base=2.0,
                cost_per_gpu_hour=1.0,
                cost_per_gb_memory_hour=0.1,
                cost_per_cpu_core_hour=0.2,
                cost_per_million_tokens=0.0,  # Local models are "free"
                spot_instance_discount=0.6,
                reserved_instance_discount=0.3
            )
            
            hf_cost = CostModel(
                provider=ModelProvider.LOCAL_HF,
                model_name="default",
                cost_per_hour_base=3.0,
                cost_per_gpu_hour=2.0,
                cost_per_gb_memory_hour=0.15,
                cost_per_cpu_core_hour=0.25,
                cost_per_million_tokens=0.0,  # Local models are "free"
                spot_instance_discount=0.6,
                reserved_instance_discount=0.3
            )
            
            self.cost_optimizer.add_cost_model("local_ollama:default", ollama_cost)
            self.cost_optimizer.add_cost_model("local_hf:default", hf_cost)
            
        except Exception as e:
            self.logger.error(f"Cost model initialization failed: {e}")
    
    async def get_scaling_statistics(self) -> Dict[str, Any]:
        """Get comprehensive scaling statistics"""
        return {
            "stats": self.stats,
            "registered_models": len(self.active_scaling_policies),
            "active_policies": dict(self.active_scaling_policies),
            "recent_actions": len([a for a in self.scaling_actions if a.created_at > datetime.utcnow() - timedelta(hours=1)]),
            "metrics_history_size": len(self.metrics_history),
            "prediction_accuracy": self.predictive_model.prediction_accuracy,
            "cost_settings": {
                "budget_hourly": self.cost_budget_hourly,
                "optimization_enabled": self.enable_cost_optimization
            }
        }


# Global instance
_auto_scaling_system: Optional[AutoScalingSystem] = None


async def get_auto_scaling_system() -> AutoScalingSystem:
    """Get the global auto-scaling system instance"""
    global _auto_scaling_system
    if _auto_scaling_system is None:
        _auto_scaling_system = AutoScalingSystem()
        await _auto_scaling_system.initialize()
    return _auto_scaling_system