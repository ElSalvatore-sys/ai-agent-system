"""Monitoring and Alerting System.

Enterprise-grade monitoring, alerting, and optimization strategies with
Prometheus metrics, Grafana dashboards, intelligent alerts, and automated optimization.
"""
from __future__ import annotations

import asyncio
import threading
import time
import json
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import statistics
import math

from app.core.logger import LoggerMixin
from app.core.config import settings


class MetricType(str, Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertSeverity(str, Enum):
    """Alert severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlertStatus(str, Enum):
    """Alert status"""
    ACTIVE = "active"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"
    SUPPRESSED = "suppressed"


class OptimizationType(str, Enum):
    """Types of optimizations"""
    PERFORMANCE = "performance"
    COST = "cost"
    RESOURCE = "resource"
    QUALITY = "quality"
    SECURITY = "security"


@dataclass
class Metric:
    """Metric definition"""
    name: str
    metric_type: MetricType
    description: str
    labels: Dict[str, str] = field(default_factory=dict)
    value: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tags: List[str] = field(default_factory=list)


@dataclass
class AlertRule:
    """Alert rule configuration"""
    rule_id: str
    name: str
    description: str
    metric_name: str
    condition: str  # e.g., "> 90", "< 10", "== 0"
    threshold: float
    severity: AlertSeverity
    duration_seconds: int = 60
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    notification_channels: List[str] = field(default_factory=list)


@dataclass
class Alert:
    """Active alert"""
    alert_id: str
    rule_id: str
    name: str
    description: str
    severity: AlertSeverity
    status: AlertStatus
    metric_value: float
    threshold: float
    fired_at: datetime
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)


@dataclass
class DashboardPanel:
    """Dashboard panel configuration"""
    panel_id: str
    title: str
    panel_type: str  # graph, stat, table, etc.
    metrics: List[str]
    time_range: str = "1h"
    refresh_interval: str = "30s"
    thresholds: List[Dict[str, Any]] = field(default_factory=list)
    visualization_options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Dashboard:
    """Dashboard configuration"""
    dashboard_id: str
    title: str
    description: str
    panels: List[DashboardPanel]
    tags: List[str] = field(default_factory=list)
    auto_refresh: bool = True
    time_range: str = "24h"


@dataclass
class OptimizationRecommendation:
    """Optimization recommendation"""
    recommendation_id: str
    optimization_type: OptimizationType
    title: str
    description: str
    impact_estimate: str
    confidence_score: float
    implementation_effort: str
    cost_impact: Optional[float] = None
    performance_impact: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    applied: bool = False
    applied_at: Optional[datetime] = None


class PrometheusExporter:
    """Prometheus metrics exporter"""
    
    def __init__(self, port: int = 9090):
        self.port = port
        self.metrics: Dict[str, Metric] = {}
        self.metrics_lock = threading.Lock()
        
    def register_metric(self, metric: Metric):
        """Register a metric for export"""
        with self.metrics_lock:
            self.metrics[metric.name] = metric
    
    def update_metric(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Update metric value"""
        with self.metrics_lock:
            if name in self.metrics:
                metric = self.metrics[name]
                metric.value = value
                metric.timestamp = datetime.utcnow()
                if labels:
                    metric.labels.update(labels)
    
    def get_metrics_exposition(self) -> str:
        """Get metrics in Prometheus exposition format"""
        with self.metrics_lock:
            exposition = []
            
            for metric in self.metrics.values():
                # Add HELP comment
                exposition.append(f"# HELP {metric.name} {metric.description}")
                
                # Add TYPE comment
                exposition.append(f"# TYPE {metric.name} {metric.metric_type}")
                
                # Add metric line
                labels_str = ""
                if metric.labels:
                    labels_list = [f'{k}="{v}"' for k, v in metric.labels.items()]
                    labels_str = "{" + ",".join(labels_list) + "}"
                
                exposition.append(f"{metric.name}{labels_str} {metric.value}")
                exposition.append("")
            
            return "\n".join(exposition)
    
    async def start_server(self):
        """Start HTTP server for metrics exposition"""
        try:
            from aiohttp import web
            
            async def metrics_handler(request):
                return web.Response(
                    text=self.get_metrics_exposition(),
                    content_type="text/plain"
                )
            
            app = web.Application()
            app.router.add_get("/metrics", metrics_handler)
            
            runner = web.AppRunner(app)
            await runner.setup()
            
            site = web.TCPSite(runner, "0.0.0.0", self.port)
            await site.start()
            
        except Exception as e:
            raise RuntimeError(f"Failed to start metrics server: {e}")


class AlertManager:
    """Alert management system"""
    
    def __init__(self):
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.notification_channels: Dict[str, Dict[str, Any]] = {}
        self.alert_lock = threading.Lock()
        
    def add_alert_rule(self, rule: AlertRule):
        """Add alert rule"""
        self.alert_rules[rule.rule_id] = rule
    
    def remove_alert_rule(self, rule_id: str):
        """Remove alert rule"""
        self.alert_rules.pop(rule_id, None)
    
    def add_notification_channel(self, channel_id: str, config: Dict[str, Any]):
        """Add notification channel"""
        self.notification_channels[channel_id] = config
    
    async def evaluate_rules(self, metrics: Dict[str, Metric]):
        """Evaluate alert rules against metrics"""
        with self.alert_lock:
            for rule in self.alert_rules.values():
                if not rule.enabled:
                    continue
                
                metric = metrics.get(rule.metric_name)
                if not metric:
                    continue
                
                # Evaluate condition
                alert_triggered = self._evaluate_condition(metric.value, rule.condition, rule.threshold)
                
                alert_id = f"{rule.rule_id}_{rule.metric_name}"
                existing_alert = self.active_alerts.get(alert_id)
                
                if alert_triggered and not existing_alert:
                    # Fire new alert
                    await self._fire_alert(rule, metric, alert_id)
                elif not alert_triggered and existing_alert:
                    # Resolve existing alert
                    await self._resolve_alert(existing_alert)
    
    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Evaluate alert condition"""
        try:
            if condition.startswith(">"):
                if "=" in condition:
                    return value >= threshold
                else:
                    return value > threshold
            elif condition.startswith("<"):
                if "=" in condition:
                    return value <= threshold
                else:
                    return value < threshold
            elif condition.startswith("=="):
                return value == threshold
            elif condition.startswith("!="):
                return value != threshold
            else:
                return False
        except:
            return False
    
    async def _fire_alert(self, rule: AlertRule, metric: Metric, alert_id: str):
        """Fire a new alert"""
        try:
            alert = Alert(
                alert_id=alert_id,
                rule_id=rule.rule_id,
                name=rule.name,
                description=rule.description,
                severity=rule.severity,
                status=AlertStatus.ACTIVE,
                metric_value=metric.value,
                threshold=rule.threshold,
                fired_at=datetime.utcnow(),
                labels=rule.labels.copy(),
                annotations=rule.annotations.copy()
            )
            
            self.active_alerts[alert_id] = alert
            
            # Send notifications
            for channel_id in rule.notification_channels:
                await self._send_notification(alert, channel_id)
            
        except Exception as e:
            pass  # Log error in production
    
    async def _resolve_alert(self, alert: Alert):
        """Resolve an active alert"""
        try:
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.utcnow()
            
            # Move to history
            self.alert_history.append(alert)
            del self.active_alerts[alert.alert_id]
            
            # Send resolution notification
            for channel_id in self.alert_rules[alert.rule_id].notification_channels:
                await self._send_resolution_notification(alert, channel_id)
                
        except Exception as e:
            pass  # Log error in production
    
    async def _send_notification(self, alert: Alert, channel_id: str):
        """Send alert notification"""
        try:
            channel_config = self.notification_channels.get(channel_id)
            if not channel_config:
                return
            
            channel_type = channel_config.get("type")
            
            if channel_type == "slack":
                await self._send_slack_notification(alert, channel_config)
            elif channel_type == "email":
                await self._send_email_notification(alert, channel_config)
            elif channel_type == "webhook":
                await self._send_webhook_notification(alert, channel_config)
                
        except Exception as e:
            pass  # Log error in production
    
    async def _send_resolution_notification(self, alert: Alert, channel_id: str):
        """Send alert resolution notification"""
        # Similar to _send_notification but for resolution
        pass
    
    async def _send_slack_notification(self, alert: Alert, config: Dict[str, Any]):
        """Send Slack notification"""
        try:
            import aiohttp
            
            webhook_url = config.get("webhook_url")
            if not webhook_url:
                return
            
            color = {
                AlertSeverity.CRITICAL: "danger",
                AlertSeverity.HIGH: "warning",
                AlertSeverity.MEDIUM: "warning",
                AlertSeverity.LOW: "good",
                AlertSeverity.INFO: "good"
            }.get(alert.severity, "warning")
            
            message = {
                "attachments": [{
                    "color": color,
                    "title": f"🚨 {alert.name}",
                    "text": alert.description,
                    "fields": [
                        {"title": "Severity", "value": alert.severity, "short": True},
                        {"title": "Metric Value", "value": str(alert.metric_value), "short": True},
                        {"title": "Threshold", "value": str(alert.threshold), "short": True},
                        {"title": "Time", "value": alert.fired_at.isoformat(), "short": True}
                    ]
                }]
            }
            
            async with aiohttp.ClientSession() as session:
                await session.post(webhook_url, json=message)
                
        except Exception as e:
            pass
    
    async def _send_email_notification(self, alert: Alert, config: Dict[str, Any]):
        """Send email notification"""
        # Email notification implementation
        pass
    
    async def _send_webhook_notification(self, alert: Alert, config: Dict[str, Any]):
        """Send webhook notification"""
        try:
            import aiohttp
            
            webhook_url = config.get("url")
            if not webhook_url:
                return
            
            payload = {
                "alert_id": alert.alert_id,
                "name": alert.name,
                "description": alert.description,
                "severity": alert.severity,
                "metric_value": alert.metric_value,
                "threshold": alert.threshold,
                "fired_at": alert.fired_at.isoformat(),
                "labels": alert.labels,
                "annotations": alert.annotations
            }
            
            async with aiohttp.ClientSession() as session:
                await session.post(webhook_url, json=payload)
                
        except Exception as e:
            pass


class DashboardGenerator:
    """Dashboard generation for Grafana"""
    
    def __init__(self):
        self.dashboards: Dict[str, Dashboard] = {}
        
    def create_llm_dashboard(self) -> Dashboard:
        """Create comprehensive LLM monitoring dashboard"""
        panels = [
            DashboardPanel(
                panel_id="system_overview",
                title="System Overview",
                panel_type="stat",
                metrics=["llm_active_models", "llm_total_requests", "llm_error_rate"],
                thresholds=[
                    {"value": 90, "color": "red"},
                    {"value": 70, "color": "yellow"}
                ]
            ),
            DashboardPanel(
                panel_id="response_times",
                title="Response Times",
                panel_type="graph",
                metrics=["llm_response_time_histogram"],
                visualization_options={
                    "stacking": {"mode": "none"},
                    "legend": {"displayMode": "table", "placement": "bottom"}
                }
            ),
            DashboardPanel(
                panel_id="gpu_utilization",
                title="GPU Utilization",
                panel_type="graph",
                metrics=["gpu_utilization_percent", "gpu_memory_usage_percent"],
                thresholds=[
                    {"value": 90, "color": "red"},
                    {"value": 80, "color": "yellow"}
                ]
            ),
            DashboardPanel(
                panel_id="cost_tracking",
                title="Cost Tracking",
                panel_type="graph",
                metrics=["llm_cost_hourly", "llm_cost_total"],
                visualization_options={"unit": "currencyUSD"}
            ),
            DashboardPanel(
                panel_id="model_performance",
                title="Model Performance",
                panel_type="table",
                metrics=["llm_model_tokens_per_second", "llm_model_accuracy"],
                visualization_options={"displayMode": "table"}
            )
        ]
        
        dashboard = Dashboard(
            dashboard_id="llm_monitoring",
            title="LLM System Monitoring",
            description="Comprehensive monitoring for LLM infrastructure",
            panels=panels,
            tags=["llm", "monitoring", "performance"]
        )
        
        self.dashboards[dashboard.dashboard_id] = dashboard
        return dashboard
    
    def generate_grafana_json(self, dashboard: Dashboard) -> Dict[str, Any]:
        """Generate Grafana dashboard JSON"""
        return {
            "dashboard": {
                "id": None,
                "title": dashboard.title,
                "description": dashboard.description,
                "tags": dashboard.tags,
                "timezone": "browser",
                "panels": [self._generate_panel_json(panel) for panel in dashboard.panels],
                "time": {"from": f"now-{dashboard.time_range}", "to": "now"},
                "refresh": "30s",
                "version": 1
            },
            "overwrite": True
        }
    
    def _generate_panel_json(self, panel: DashboardPanel) -> Dict[str, Any]:
        """Generate Grafana panel JSON"""
        base_panel = {
            "id": hash(panel.panel_id) % 1000000,
            "title": panel.title,
            "type": panel.panel_type,
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
            "targets": [
                {
                    "expr": metric,
                    "format": "time_series",
                    "intervalFactor": 1,
                    "refId": f"query_{i}"
                }
                for i, metric in enumerate(panel.metrics)
            ]
        }
        
        # Add panel-specific options
        if panel.thresholds:
            base_panel["fieldConfig"] = {
                "defaults": {
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [{"color": "green", "value": None}] + 
                                [{"color": t["color"], "value": t["value"]} for t in panel.thresholds]
                    }
                }
            }
        
        if panel.visualization_options:
            base_panel["options"] = panel.visualization_options
        
        return base_panel


class OptimizationEngine:
    """Intelligent optimization engine"""
    
    def __init__(self):
        self.recommendations: List[OptimizationRecommendation] = []
        self.optimization_history: List[OptimizationRecommendation] = []
        self.optimization_rules: Dict[str, Callable] = {}
        
    def register_optimization_rule(self, name: str, rule_func: Callable):
        """Register optimization rule"""
        self.optimization_rules[name] = rule_func
    
    async def analyze_and_recommend(self, metrics: Dict[str, Metric]) -> List[OptimizationRecommendation]:
        """Analyze metrics and generate optimization recommendations"""
        recommendations = []
        
        try:
            # Performance optimizations
            perf_recommendations = await self._analyze_performance(metrics)
            recommendations.extend(perf_recommendations)
            
            # Cost optimizations
            cost_recommendations = await self._analyze_costs(metrics)
            recommendations.extend(cost_recommendations)
            
            # Resource optimizations
            resource_recommendations = await self._analyze_resources(metrics)
            recommendations.extend(resource_recommendations)
            
            # Quality optimizations
            quality_recommendations = await self._analyze_quality(metrics)
            recommendations.extend(quality_recommendations)
            
            # Store recommendations
            self.recommendations.extend(recommendations)
            
            return recommendations
            
        except Exception as e:
            return []
    
    async def _analyze_performance(self, metrics: Dict[str, Metric]) -> List[OptimizationRecommendation]:
        """Analyze performance metrics"""
        recommendations = []
        
        # Check response time
        response_time_metric = metrics.get("llm_response_time_avg")
        if response_time_metric and response_time_metric.value > 5000:  # 5 seconds
            recommendations.append(OptimizationRecommendation(
                recommendation_id=f"perf_response_time_{int(time.time())}",
                optimization_type=OptimizationType.PERFORMANCE,
                title="High Response Time Detected",
                description=f"Average response time is {response_time_metric.value:.0f}ms. Consider enabling caching, optimizing model quantization, or scaling horizontally.",
                impact_estimate="20-50% response time improvement",
                confidence_score=0.8,
                implementation_effort="Medium",
                performance_impact=30.0
            ))
        
        # Check GPU utilization
        gpu_util_metric = metrics.get("gpu_utilization_percent")
        if gpu_util_metric and gpu_util_metric.value > 90:
            recommendations.append(OptimizationRecommendation(
                recommendation_id=f"perf_gpu_util_{int(time.time())}",
                optimization_type=OptimizationType.PERFORMANCE,
                title="High GPU Utilization",
                description=f"GPU utilization is {gpu_util_metric.value:.1f}%. Consider adding more GPU instances or implementing model sharding.",
                impact_estimate="Reduce GPU bottleneck by 40-60%",
                confidence_score=0.9,
                implementation_effort="High",
                performance_impact=50.0
            ))
        
        # Check memory usage
        memory_metric = metrics.get("memory_usage_percent")
        if memory_metric and memory_metric.value > 85:
            recommendations.append(OptimizationRecommendation(
                recommendation_id=f"perf_memory_{int(time.time())}",
                optimization_type=OptimizationType.PERFORMANCE,
                title="High Memory Usage",
                description=f"Memory usage is {memory_metric.value:.1f}%. Consider enabling model quantization or increasing memory allocation.",
                impact_estimate="15-30% memory efficiency improvement",
                confidence_score=0.7,
                implementation_effort="Medium",
                performance_impact=25.0
            ))
        
        return recommendations
    
    async def _analyze_costs(self, metrics: Dict[str, Metric]) -> List[OptimizationRecommendation]:
        """Analyze cost metrics"""
        recommendations = []
        
        # Check hourly costs
        cost_metric = metrics.get("llm_cost_hourly")
        if cost_metric and cost_metric.value > 50:  # $50/hour
            recommendations.append(OptimizationRecommendation(
                recommendation_id=f"cost_hourly_{int(time.time())}",
                optimization_type=OptimizationType.COST,
                title="High Hourly Costs",
                description=f"Hourly cost is ${cost_metric.value:.2f}. Consider using spot instances, smaller models, or optimizing usage patterns.",
                impact_estimate="20-40% cost reduction",
                confidence_score=0.6,
                implementation_effort="Medium",
                cost_impact=-cost_metric.value * 0.3
            ))
        
        # Check request efficiency
        requests_metric = metrics.get("llm_total_requests")
        cost_per_request = cost_metric.value / max(1, requests_metric.value) if requests_metric else 0
        
        if cost_per_request > 0.1:  # $0.10 per request
            recommendations.append(OptimizationRecommendation(
                recommendation_id=f"cost_efficiency_{int(time.time())}",
                optimization_type=OptimizationType.COST,
                title="High Cost Per Request",
                description=f"Cost per request is ${cost_per_request:.3f}. Consider implementing response caching or using more efficient models.",
                impact_estimate="30-50% cost per request reduction",
                confidence_score=0.7,
                implementation_effort="Low",
                cost_impact=-cost_metric.value * 0.4
            ))
        
        return recommendations
    
    async def _analyze_resources(self, metrics: Dict[str, Metric]) -> List[OptimizationRecommendation]:
        """Analyze resource utilization"""
        recommendations = []
        
        # Check CPU utilization
        cpu_metric = metrics.get("cpu_utilization_percent")
        if cpu_metric and cpu_metric.value < 30:  # Under-utilized
            recommendations.append(OptimizationRecommendation(
                recommendation_id=f"resource_cpu_{int(time.time())}",
                optimization_type=OptimizationType.RESOURCE,
                title="Low CPU Utilization",
                description=f"CPU utilization is only {cpu_metric.value:.1f}%. Consider consolidating workloads or downsizing instances.",
                impact_estimate="15-25% resource efficiency improvement",
                confidence_score=0.6,
                implementation_effort="Low",
                cost_impact=-10.0
            ))
        
        # Check container count vs load
        active_models_metric = metrics.get("llm_active_models")
        if active_models_metric and active_models_metric.value > 5:
            recommendations.append(OptimizationRecommendation(
                recommendation_id=f"resource_scaling_{int(time.time())}",
                optimization_type=OptimizationType.RESOURCE,
                title="Multiple Model Instances",
                description=f"{active_models_metric.value:.0f} model instances are active. Consider implementing auto-scaling or model consolidation.",
                impact_estimate="20-35% resource optimization",
                confidence_score=0.5,
                implementation_effort="High",
                cost_impact=-20.0
            ))
        
        return recommendations
    
    async def _analyze_quality(self, metrics: Dict[str, Metric]) -> List[OptimizationRecommendation]:
        """Analyze quality metrics"""
        recommendations = []
        
        # Check error rate
        error_rate_metric = metrics.get("llm_error_rate")
        if error_rate_metric and error_rate_metric.value > 5:  # 5% error rate
            recommendations.append(OptimizationRecommendation(
                recommendation_id=f"quality_errors_{int(time.time())}",
                optimization_type=OptimizationType.QUALITY,
                title="High Error Rate",
                description=f"Error rate is {error_rate_metric.value:.1f}%. Consider implementing better input validation, retry logic, or model health checks.",
                impact_estimate="50-80% error rate reduction",
                confidence_score=0.8,
                implementation_effort="Medium"
            ))
        
        # Check model accuracy (if available)
        accuracy_metric = metrics.get("llm_model_accuracy")
        if accuracy_metric and accuracy_metric.value < 85:  # 85% accuracy
            recommendations.append(OptimizationRecommendation(
                recommendation_id=f"quality_accuracy_{int(time.time())}",
                optimization_type=OptimizationType.QUALITY,
                title="Low Model Accuracy",
                description=f"Model accuracy is {accuracy_metric.value:.1f}%. Consider model fine-tuning, better prompts, or using larger models.",
                impact_estimate="10-20% accuracy improvement",
                confidence_score=0.6,
                implementation_effort="High"
            ))
        
        return recommendations
    
    async def apply_recommendation(self, recommendation_id: str) -> bool:
        """Apply an optimization recommendation"""
        try:
            recommendation = next(
                (r for r in self.recommendations if r.recommendation_id == recommendation_id),
                None
            )
            
            if not recommendation or recommendation.applied:
                return False
            
            # Apply the optimization based on type
            success = await self._apply_optimization(recommendation)
            
            if success:
                recommendation.applied = True
                recommendation.applied_at = datetime.utcnow()
                self.optimization_history.append(recommendation)
                
                # Remove from active recommendations
                self.recommendations = [
                    r for r in self.recommendations 
                    if r.recommendation_id != recommendation_id
                ]
            
            return success
            
        except Exception as e:
            return False
    
    async def _apply_optimization(self, recommendation: OptimizationRecommendation) -> bool:
        """Apply specific optimization"""
        try:
            # This would implement actual optimization logic
            # For now, simulate application
            await asyncio.sleep(1)
            return True
            
        except Exception as e:
            return False


class MonitoringAlertingSystem(LoggerMixin):
    """Enterprise monitoring and alerting system"""
    
    def __init__(self):
        super().__init__()
        self.prometheus_exporter = PrometheusExporter()
        self.alert_manager = AlertManager()
        self.dashboard_generator = DashboardGenerator()
        self.optimization_engine = OptimizationEngine()
        
        # Metrics storage
        self.metrics: Dict[str, Metric] = {}
        self.metrics_history: List[Tuple[datetime, Dict[str, float]]] = []
        
        # Configuration
        self.metrics_retention_hours = 168  # 1 week
        self.optimization_interval_minutes = 60
        self.dashboard_refresh_interval = 30
        
        # Background tasks
        self._metrics_collection_task: Optional[asyncio.Task] = None
        self._alert_evaluation_task: Optional[asyncio.Task] = None
        self._optimization_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        # Statistics
        self.stats = {
            "total_metrics_collected": 0,
            "active_alerts": 0,
            "alerts_fired_total": 0,
            "optimizations_applied": 0,
            "system_uptime_seconds": 0
        }
        
        self._start_time = time.time()
    
    async def initialize(self):
        """Initialize the monitoring and alerting system"""
        try:
            # Initialize components
            await self._initialize_default_metrics()
            await self._initialize_default_alerts()
            await self._initialize_notification_channels()
            await self._create_default_dashboards()
            
            # Start Prometheus exporter
            await self.prometheus_exporter.start_server()
            
            # Start background tasks
            self._metrics_collection_task = asyncio.create_task(self._metrics_collection_loop())
            self._alert_evaluation_task = asyncio.create_task(self._alert_evaluation_loop())
            self._optimization_task = asyncio.create_task(self._optimization_loop())
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            self.logger.info("Monitoring and alerting system initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize monitoring system: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown the monitoring system"""
        self.logger.info("Shutting down monitoring and alerting system")
        self._shutdown_event.set()
        
        # Cancel background tasks
        for task in [self._metrics_collection_task, self._alert_evaluation_task, 
                    self._optimization_task, self._cleanup_task]:
            if task:
                task.cancel()
    
    async def collect_system_metrics(self) -> Dict[str, float]:
        """Collect system-wide metrics"""
        try:
            import psutil
            
            # System metrics
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network I/O
            net_io = psutil.net_io_counters()
            
            # Process metrics
            process = psutil.Process()
            process_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            return {
                "system_cpu_percent": cpu_percent,
                "system_memory_percent": memory.percent,
                "system_memory_available_mb": memory.available / 1024 / 1024,
                "system_disk_usage_percent": (disk.used / disk.total) * 100,
                "system_network_bytes_sent": net_io.bytes_sent,
                "system_network_bytes_recv": net_io.bytes_recv,
                "process_memory_mb": process_memory,
                "process_cpu_percent": process.cpu_percent()
            }
            
        except Exception as e:
            self.logger.error(f"System metrics collection failed: {e}")
            return {}
    
    async def collect_llm_metrics(self) -> Dict[str, float]:
        """Collect LLM-specific metrics"""
        try:
            # These would come from actual LLM services
            # For now, simulate metrics
            
            import random
            
            base_time = time.time()
            
            return {
                "llm_active_models": random.randint(1, 5),
                "llm_total_requests": random.randint(100, 1000),
                "llm_requests_per_second": random.uniform(5, 50),
                "llm_response_time_avg": random.uniform(500, 3000),
                "llm_response_time_p95": random.uniform(1000, 5000),
                "llm_error_rate": random.uniform(0, 10),
                "llm_tokens_per_second": random.uniform(50, 500),
                "llm_cost_hourly": random.uniform(10, 100),
                "llm_cost_total": random.uniform(100, 10000),
                "llm_cache_hit_rate": random.uniform(60, 95),
                "llm_model_accuracy": random.uniform(80, 95),
                "gpu_utilization_percent": random.uniform(30, 95),
                "gpu_memory_usage_percent": random.uniform(40, 90),
                "gpu_temperature": random.uniform(60, 85)
            }
            
        except Exception as e:
            self.logger.error(f"LLM metrics collection failed: {e}")
            return {}
    
    async def update_metrics(self, metric_values: Dict[str, float]):
        """Update metrics with new values"""
        try:
            timestamp = datetime.utcnow()
            
            for name, value in metric_values.items():
                # Update metric
                if name in self.metrics:
                    metric = self.metrics[name]
                    metric.value = value
                    metric.timestamp = timestamp
                
                # Update Prometheus exporter
                self.prometheus_exporter.update_metric(name, value)
            
            # Store in history
            self.metrics_history.append((timestamp, metric_values.copy()))
            
            # Update statistics
            self.stats["total_metrics_collected"] += len(metric_values)
            self.stats["system_uptime_seconds"] = int(time.time() - self._start_time)
            
        except Exception as e:
            self.logger.error(f"Metrics update failed: {e}")
    
    async def get_metric_value(self, metric_name: str) -> Optional[float]:
        """Get current value of a metric"""
        metric = self.metrics.get(metric_name)
        return metric.value if metric else None
    
    async def get_metric_history(
        self, 
        metric_name: str, 
        hours: int = 24
    ) -> List[Tuple[datetime, float]]:
        """Get metric history"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            history = []
            for timestamp, values in self.metrics_history:
                if timestamp >= cutoff_time and metric_name in values:
                    history.append((timestamp, values[metric_name]))
            
            return history
            
        except Exception as e:
            self.logger.error(f"Metric history retrieval failed: {e}")
            return []
    
    async def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        return list(self.alert_manager.active_alerts.values())
    
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert"""
        try:
            alert = self.alert_manager.active_alerts.get(alert_id)
            if alert:
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_at = datetime.utcnow()
                alert.acknowledged_by = acknowledged_by
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Alert acknowledgment failed: {e}")
            return False
    
    async def get_optimization_recommendations(self) -> List[OptimizationRecommendation]:
        """Get current optimization recommendations"""
        return self.optimization_engine.recommendations
    
    async def apply_optimization(self, recommendation_id: str) -> bool:
        """Apply an optimization recommendation"""
        try:
            success = await self.optimization_engine.apply_recommendation(recommendation_id)
            if success:
                self.stats["optimizations_applied"] += 1
            return success
            
        except Exception as e:
            self.logger.error(f"Optimization application failed: {e}")
            return False
    
    async def get_dashboard_data(self, dashboard_id: str) -> Optional[Dict[str, Any]]:
        """Get dashboard data"""
        try:
            dashboard = self.dashboard_generator.dashboards.get(dashboard_id)
            if not dashboard:
                return None
            
            dashboard_data = {
                "dashboard": dashboard.__dict__,
                "panels": []
            }
            
            for panel in dashboard.panels:
                panel_data = {
                    "panel": panel.__dict__,
                    "data": []
                }
                
                # Get data for each metric in the panel
                for metric_name in panel.metrics:
                    metric_history = await self.get_metric_history(metric_name, 24)
                    panel_data["data"].append({
                        "metric": metric_name,
                        "history": [{"time": t.isoformat(), "value": v} for t, v in metric_history]
                    })
                
                dashboard_data["panels"].append(panel_data)
            
            return dashboard_data
            
        except Exception as e:
            self.logger.error(f"Dashboard data retrieval failed: {e}")
            return None
    
    async def export_grafana_dashboard(self, dashboard_id: str) -> Optional[Dict[str, Any]]:
        """Export dashboard as Grafana JSON"""
        try:
            dashboard = self.dashboard_generator.dashboards.get(dashboard_id)
            if not dashboard:
                return None
            
            return self.dashboard_generator.generate_grafana_json(dashboard)
            
        except Exception as e:
            self.logger.error(f"Grafana export failed: {e}")
            return None
    
    async def _initialize_default_metrics(self):
        """Initialize default metrics"""
        try:
            default_metrics = [
                # System metrics
                Metric("system_cpu_percent", MetricType.GAUGE, "System CPU utilization percentage"),
                Metric("system_memory_percent", MetricType.GAUGE, "System memory utilization percentage"),
                Metric("system_disk_usage_percent", MetricType.GAUGE, "System disk usage percentage"),
                
                # LLM metrics
                Metric("llm_active_models", MetricType.GAUGE, "Number of active LLM models"),
                Metric("llm_total_requests", MetricType.COUNTER, "Total number of LLM requests"),
                Metric("llm_requests_per_second", MetricType.GAUGE, "LLM requests per second"),
                Metric("llm_response_time_avg", MetricType.GAUGE, "Average LLM response time in ms"),
                Metric("llm_error_rate", MetricType.GAUGE, "LLM error rate percentage"),
                Metric("llm_cost_hourly", MetricType.GAUGE, "Hourly LLM cost in USD"),
                
                # GPU metrics
                Metric("gpu_utilization_percent", MetricType.GAUGE, "GPU utilization percentage"),
                Metric("gpu_memory_usage_percent", MetricType.GAUGE, "GPU memory usage percentage"),
                Metric("gpu_temperature", MetricType.GAUGE, "GPU temperature in Celsius")
            ]
            
            for metric in default_metrics:
                self.metrics[metric.name] = metric
                self.prometheus_exporter.register_metric(metric)
                
        except Exception as e:
            self.logger.error(f"Default metrics initialization failed: {e}")
    
    async def _initialize_default_alerts(self):
        """Initialize default alert rules"""
        try:
            default_rules = [
                AlertRule(
                    rule_id="high_cpu_usage",
                    name="High CPU Usage",
                    description="System CPU usage is above 90%",
                    metric_name="system_cpu_percent",
                    condition=">",
                    threshold=90.0,
                    severity=AlertSeverity.HIGH,
                    notification_channels=["default"]
                ),
                AlertRule(
                    rule_id="high_memory_usage",
                    name="High Memory Usage",
                    description="System memory usage is above 90%",
                    metric_name="system_memory_percent",
                    condition=">",
                    threshold=90.0,
                    severity=AlertSeverity.HIGH,
                    notification_channels=["default"]
                ),
                AlertRule(
                    rule_id="high_error_rate",
                    name="High LLM Error Rate",
                    description="LLM error rate is above 10%",
                    metric_name="llm_error_rate",
                    condition=">",
                    threshold=10.0,
                    severity=AlertSeverity.CRITICAL,
                    notification_channels=["default"]
                ),
                AlertRule(
                    rule_id="slow_response_time",
                    name="Slow LLM Response Time",
                    description="Average LLM response time is above 5 seconds",
                    metric_name="llm_response_time_avg",
                    condition=">",
                    threshold=5000.0,
                    severity=AlertSeverity.MEDIUM,
                    notification_channels=["default"]
                ),
                AlertRule(
                    rule_id="high_gpu_temperature",
                    name="High GPU Temperature",
                    description="GPU temperature is above 85°C",
                    metric_name="gpu_temperature",
                    condition=">",
                    threshold=85.0,
                    severity=AlertSeverity.HIGH,
                    notification_channels=["default"]
                )
            ]
            
            for rule in default_rules:
                self.alert_manager.add_alert_rule(rule)
                
        except Exception as e:
            self.logger.error(f"Default alerts initialization failed: {e}")
    
    async def _initialize_notification_channels(self):
        """Initialize notification channels"""
        try:
            # Default webhook channel
            self.alert_manager.add_notification_channel("default", {
                "type": "webhook",
                "url": "http://localhost:8080/api/v1/alerts/webhook",
                "timeout": 10
            })
            
            # Slack channel (if configured)
            slack_webhook = settings.get("SLACK_WEBHOOK_URL")
            if slack_webhook:
                self.alert_manager.add_notification_channel("slack", {
                    "type": "slack",
                    "webhook_url": slack_webhook
                })
                
        except Exception as e:
            self.logger.error(f"Notification channels initialization failed: {e}")
    
    async def _create_default_dashboards(self):
        """Create default monitoring dashboards"""
        try:
            # Create main LLM dashboard
            self.dashboard_generator.create_llm_dashboard()
            
        except Exception as e:
            self.logger.error(f"Default dashboards creation failed: {e}")
    
    async def _metrics_collection_loop(self):
        """Background metrics collection loop"""
        while not self._shutdown_event.is_set():
            try:
                # Collect system metrics
                system_metrics = await self.collect_system_metrics()
                
                # Collect LLM metrics
                llm_metrics = await self.collect_llm_metrics()
                
                # Combine and update
                all_metrics = {**system_metrics, **llm_metrics}
                await self.update_metrics(all_metrics)
                
                await asyncio.sleep(30)  # Collect every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(30)
    
    async def _alert_evaluation_loop(self):
        """Background alert evaluation loop"""
        while not self._shutdown_event.is_set():
            try:
                # Evaluate alert rules
                await self.alert_manager.evaluate_rules(self.metrics)
                
                # Update statistics
                self.stats["active_alerts"] = len(self.alert_manager.active_alerts)
                self.stats["alerts_fired_total"] = len(self.alert_manager.alert_history)
                
                await asyncio.sleep(60)  # Evaluate every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Alert evaluation error: {e}")
                await asyncio.sleep(60)
    
    async def _optimization_loop(self):
        """Background optimization loop"""
        while not self._shutdown_event.is_set():
            try:
                # Generate optimization recommendations
                recommendations = await self.optimization_engine.analyze_and_recommend(self.metrics)
                
                if recommendations:
                    self.logger.info(f"Generated {len(recommendations)} optimization recommendations")
                
                await asyncio.sleep(self.optimization_interval_minutes * 60)  # Run hourly
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Optimization loop error: {e}")
                await asyncio.sleep(self.optimization_interval_minutes * 60)
    
    async def _cleanup_loop(self):
        """Background cleanup loop"""
        while not self._shutdown_event.is_set():
            try:
                # Clean up old metrics history
                cutoff_time = datetime.utcnow() - timedelta(hours=self.metrics_retention_hours)
                
                self.metrics_history = [
                    (timestamp, values) for timestamp, values in self.metrics_history
                    if timestamp > cutoff_time
                ]
                
                # Clean up old alert history
                self.alert_manager.alert_history = [
                    alert for alert in self.alert_manager.alert_history
                    if alert.fired_at > cutoff_time
                ]
                
                await asyncio.sleep(3600)  # Run every hour
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(3600)
    
    async def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        return {
            "stats": self.stats,
            "metrics_count": len(self.metrics),
            "active_alerts": len(self.alert_manager.active_alerts),
            "alert_rules": len(self.alert_manager.alert_rules),
            "optimization_recommendations": len(self.optimization_engine.recommendations),
            "dashboards": len(self.dashboard_generator.dashboards),
            "metrics_history_size": len(self.metrics_history)
        }


# Global instance
_monitoring_system: Optional[MonitoringAlertingSystem] = None


async def get_monitoring_alerting_system() -> MonitoringAlertingSystem:
    """Get the global monitoring and alerting system instance"""
    global _monitoring_system
    if _monitoring_system is None:
        _monitoring_system = MonitoringAlertingSystem()
        await _monitoring_system.initialize()
    return _monitoring_system