"""
Monitoring and Observability Integrations
Comprehensive monitoring, metrics, tracing, and alerting integrations
"""

import asyncio
import aiohttp
import json
import os
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import subprocess
from urllib.parse import urljoin

from opentelemetry import trace, metrics
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from prometheus_client import CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST

from app.core.integration_framework import (
    BaseIntegration, IntegrationConfig, IntegrationType, IntegrationStatus
)
from app.core.logger import LoggerMixin


class PrometheusIntegration(BaseIntegration):
    """Integration with Prometheus for metrics collection and export"""
    
    def __init__(self, config: IntegrationConfig):
        super().__init__(config)
        self.port = config.settings.get("port", 9090)
        self.host = config.settings.get("host", "localhost")
        self.metrics_path = config.settings.get("metrics_path", "/metrics")
        self.job_name = config.settings.get("job_name", "llm-platform")
        self.push_gateway_url = config.settings.get("push_gateway_url")
        self.scrape_interval = config.settings.get("scrape_interval", "15s")
        
        self.registry = CollectorRegistry()
        self.custom_metrics: Dict[str, Any] = {}
        self._metrics_server_task: Optional[asyncio.Task] = None
    
    async def initialize(self) -> bool:
        """Initialize Prometheus integration"""
        try:
            # Setup Prometheus metrics reader
            self._setup_prometheus_reader()
            
            # Create custom metrics
            await self._create_custom_metrics()
            
            # Generate Prometheus configuration
            await self._generate_prometheus_config()
            
            # Start metrics server if needed
            if self.config.settings.get("start_metrics_server", True):
                await self._start_metrics_server()
            
            self.logger.info("Prometheus integration initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Prometheus integration: {e}")
            return False
    
    async def health_check(self) -> bool:
        """Check Prometheus health"""
        try:
            # Try to access metrics endpoint
            async with aiohttp.ClientSession() as session:
                url = f"http://{self.host}:{self.port}{self.metrics_path}"
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    return response.status == 200
        except Exception as e:
            self.logger.error(f"Prometheus health check failed: {e}")
            return False
    
    async def shutdown(self) -> None:
        """Shutdown Prometheus integration"""
        if self._metrics_server_task:
            self._metrics_server_task.cancel()
    
    def _setup_prometheus_reader(self) -> None:
        """Setup Prometheus metrics reader"""
        try:
            resource = Resource.create({
                "service.name": self.job_name,
                "service.version": "1.0.0"
            })
            
            # Create Prometheus metric reader
            prometheus_reader = PrometheusMetricReader()
            
            # Initialize meter provider
            meter_provider = MeterProvider(
                resource=resource,
                metric_readers=[prometheus_reader]
            )
            
            metrics.set_meter_provider(meter_provider)
            
        except Exception as e:
            self.logger.error(f"Failed to setup Prometheus reader: {e}")
            raise
    
    async def _create_custom_metrics(self) -> None:
        """Create custom metrics for the LLM platform"""
        try:
            meter = metrics.get_meter(__name__)
            
            # Request metrics
            self.custom_metrics["request_count"] = meter.create_counter(
                name="llm_requests_total",
                description="Total number of LLM requests",
                unit="1"
            )
            
            self.custom_metrics["request_duration"] = meter.create_histogram(
                name="llm_request_duration_seconds",
                description="Duration of LLM requests in seconds",
                unit="s"
            )
            
            # Model metrics
            self.custom_metrics["model_load_count"] = meter.create_counter(
                name="llm_model_loads_total",
                description="Total number of model loads",
                unit="1"
            )
            
            self.custom_metrics["active_models"] = meter.create_up_down_counter(
                name="llm_active_models",
                description="Number of currently active models",
                unit="1"
            )
            
            # Resource metrics
            self.custom_metrics["gpu_memory_usage"] = meter.create_gauge(
                name="llm_gpu_memory_usage_bytes",
                description="GPU memory usage in bytes",
                unit="By"
            )
            
            self.custom_metrics["cpu_usage"] = meter.create_gauge(
                name="llm_cpu_usage_percent",
                description="CPU usage percentage",
                unit="%"
            )
            
            # Error metrics
            self.custom_metrics["error_count"] = meter.create_counter(
                name="llm_errors_total",
                description="Total number of errors",
                unit="1"
            )
            
            # Cost metrics
            self.custom_metrics["total_cost"] = meter.create_counter(
                name="llm_cost_total",
                description="Total cost in USD",
                unit="USD"
            )
            
            self.logger.info("Created custom Prometheus metrics")
            
        except Exception as e:
            self.logger.error(f"Failed to create custom metrics: {e}")
            raise
    
    async def _generate_prometheus_config(self) -> None:
        """Generate Prometheus configuration file"""
        try:
            config = {
                "global": {
                    "scrape_interval": self.scrape_interval,
                    "evaluation_interval": "15s"
                },
                "scrape_configs": [
                    {
                        "job_name": self.job_name,
                        "static_configs": [
                            {
                                "targets": [f"{self.host}:{self.port}"]
                            }
                        ],
                        "metrics_path": self.metrics_path,
                        "scrape_interval": self.scrape_interval
                    }
                ],
                "rule_files": [
                    "llm_platform_rules.yml"
                ]
            }
            
            config_path = Path("monitoring/prometheus.yml")
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
            
            # Generate alerting rules
            await self._generate_alerting_rules()
            
            self.logger.info(f"Generated Prometheus config: {config_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate Prometheus config: {e}")
    
    async def _generate_alerting_rules(self) -> None:
        """Generate Prometheus alerting rules"""
        try:
            rules = {
                "groups": [
                    {
                        "name": "llm_platform_alerts",
                        "rules": [
                            {
                                "alert": "HighErrorRate",
                                "expr": "rate(llm_errors_total[5m]) > 0.1",
                                "for": "2m",
                                "labels": {
                                    "severity": "warning"
                                },
                                "annotations": {
                                    "summary": "High error rate detected",
                                    "description": "Error rate is {{ $value }} errors per second"
                                }
                            },
                            {
                                "alert": "HighGPUMemoryUsage",
                                "expr": "llm_gpu_memory_usage_bytes > 0.9 * 8589934592",  # 90% of 8GB
                                "for": "5m",
                                "labels": {
                                    "severity": "warning"
                                },
                                "annotations": {
                                    "summary": "High GPU memory usage",
                                    "description": "GPU memory usage is {{ $value }} bytes"
                                }
                            },
                            {
                                "alert": "ModelLoadFailure",
                                "expr": "increase(llm_model_loads_total{status=\"failed\"}[5m]) > 0",
                                "for": "1m",
                                "labels": {
                                    "severity": "critical"
                                },
                                "annotations": {
                                    "summary": "Model load failure detected",
                                    "description": "{{ $value }} model load failures in the last 5 minutes"
                                }
                            },
                            {
                                "alert": "HighRequestLatency",
                                "expr": "histogram_quantile(0.95, rate(llm_request_duration_seconds_bucket[5m])) > 10",
                                "for": "5m",
                                "labels": {
                                    "severity": "warning"
                                },
                                "annotations": {
                                    "summary": "High request latency",
                                    "description": "95th percentile latency is {{ $value }} seconds"
                                }
                            }
                        ]
                    }
                ]
            }
            
            rules_path = Path("monitoring/llm_platform_rules.yml")
            with open(rules_path, "w") as f:
                yaml.dump(rules, f, default_flow_style=False)
            
            self.logger.info(f"Generated alerting rules: {rules_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate alerting rules: {e}")
    
    async def _start_metrics_server(self) -> None:
        """Start simple metrics server"""
        # This would start a simple HTTP server to serve metrics
        # For production, metrics would be served by the main FastAPI app
        self.logger.info("Metrics server would be started here")
    
    def record_metric(self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a custom metric"""
        try:
            if metric_name in self.custom_metrics:
                metric = self.custom_metrics[metric_name]
                if hasattr(metric, 'add'):
                    metric.add(value, labels or {})
                elif hasattr(metric, 'set'):
                    metric.set(value, labels or {})
                    
        except Exception as e:
            self.logger.error(f"Failed to record metric {metric_name}: {e}")


class GrafanaIntegration(BaseIntegration):
    """Integration with Grafana for dashboards and visualization"""
    
    def __init__(self, config: IntegrationConfig):
        super().__init__(config)
        self.base_url = config.settings.get("base_url", "http://localhost:3000")
        self.api_key = config.settings.get("api_key", os.getenv("GRAFANA_API_KEY"))
        self.username = config.settings.get("username", "admin")
        self.password = config.settings.get("password", "admin")
        self.org_id = config.settings.get("org_id", 1)
        
        self.session: Optional[aiohttp.ClientSession] = None
        self._dashboard_configs: List[Dict[str, Any]] = []
    
    async def initialize(self) -> bool:
        """Initialize Grafana integration"""
        try:
            # Setup HTTP session with authentication
            auth = aiohttp.BasicAuth(self.username, self.password) if not self.api_key else None
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            
            self.session = aiohttp.ClientSession(
                headers=headers,
                auth=auth,
                timeout=aiohttp.ClientTimeout(total=30)
            )
            
            # Test connection
            await self._test_connection()
            
            # Setup data sources
            await self._setup_data_sources()
            
            # Create dashboards
            await self._create_dashboards()
            
            self.logger.info("Grafana integration initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Grafana integration: {e}")
            return False
    
    async def health_check(self) -> bool:
        """Check Grafana health"""
        try:
            await self._test_connection()
            return True
        except Exception as e:
            self.logger.error(f"Grafana health check failed: {e}")
            return False
    
    async def shutdown(self) -> None:
        """Shutdown Grafana integration"""
        if self.session:
            await self.session.close()
    
    async def _test_connection(self) -> bool:
        """Test connection to Grafana"""
        if not self.session:
            raise RuntimeError("Session not initialized")
        
        async with self.session.get(f"{self.base_url}/api/health") as response:
            if response.status == 200:
                return True
            else:
                raise RuntimeError(f"Grafana not available: {response.status}")
    
    async def _setup_data_sources(self) -> None:
        """Setup Prometheus data source in Grafana"""
        try:
            datasource_config = {
                "name": "Prometheus-LLM",
                "type": "prometheus",
                "url": "http://localhost:9090",
                "access": "proxy",
                "isDefault": True,
                "jsonData": {
                    "httpMethod": "POST",
                    "manageAlerts": True,
                    "prometheusType": "Prometheus",
                    "prometheusVersion": "2.40.0"
                }
            }
            
            if not self.session:
                raise RuntimeError("Session not initialized")
            
            async with self.session.post(
                f"{self.base_url}/api/datasources",
                json=datasource_config
            ) as response:
                if response.status in [200, 409]:  # 409 = already exists
                    self.logger.info("Prometheus data source configured")
                else:
                    error_text = await response.text()
                    self.logger.warning(f"Failed to create data source: {error_text}")
                    
        except Exception as e:
            self.logger.error(f"Failed to setup data sources: {e}")
    
    async def _create_dashboards(self) -> None:
        """Create Grafana dashboards for LLM platform"""
        try:
            # LLM Platform Overview Dashboard
            overview_dashboard = await self._create_overview_dashboard()
            await self._import_dashboard(overview_dashboard, "LLM Platform Overview")
            
            # Model Performance Dashboard
            model_dashboard = await self._create_model_dashboard()
            await self._import_dashboard(model_dashboard, "Model Performance")
            
            # Resource Monitoring Dashboard
            resource_dashboard = await self._create_resource_dashboard()
            await self._import_dashboard(resource_dashboard, "Resource Monitoring")
            
            # Cost Analysis Dashboard
            cost_dashboard = await self._create_cost_dashboard()
            await self._import_dashboard(cost_dashboard, "Cost Analysis")
            
        except Exception as e:
            self.logger.error(f"Failed to create dashboards: {e}")
    
    async def _create_overview_dashboard(self) -> Dict[str, Any]:
        """Create overview dashboard configuration"""
        return {
            "dashboard": {
                "title": "LLM Platform Overview",
                "tags": ["llm", "overview"],
                "timezone": "utc",
                "refresh": "30s",
                "time": {
                    "from": "now-1h",
                    "to": "now"
                },
                "panels": [
                    {
                        "id": 1,
                        "title": "Request Rate",
                        "type": "stat",
                        "targets": [
                            {
                                "expr": "rate(llm_requests_total[5m])",
                                "legendFormat": "Requests/sec"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 6, "x": 0, "y": 0}
                    },
                    {
                        "id": 2,
                        "title": "Error Rate",
                        "type": "stat",
                        "targets": [
                            {
                                "expr": "rate(llm_errors_total[5m])",
                                "legendFormat": "Errors/sec"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 6, "x": 6, "y": 0}
                    },
                    {
                        "id": 3,
                        "title": "Active Models",
                        "type": "stat",
                        "targets": [
                            {
                                "expr": "llm_active_models",
                                "legendFormat": "Active Models"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 6, "x": 12, "y": 0}
                    },
                    {
                        "id": 4,
                        "title": "Total Cost",
                        "type": "stat",
                        "targets": [
                            {
                                "expr": "llm_cost_total",
                                "legendFormat": "Total Cost ($)"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 6, "x": 18, "y": 0}
                    },
                    {
                        "id": 5,
                        "title": "Request Duration",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "histogram_quantile(0.50, rate(llm_request_duration_seconds_bucket[5m]))",
                                "legendFormat": "50th percentile"
                            },
                            {
                                "expr": "histogram_quantile(0.95, rate(llm_request_duration_seconds_bucket[5m]))",
                                "legendFormat": "95th percentile"
                            },
                            {
                                "expr": "histogram_quantile(0.99, rate(llm_request_duration_seconds_bucket[5m]))",
                                "legendFormat": "99th percentile"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 24, "x": 0, "y": 8}
                    }
                ]
            }
        }
    
    async def _create_model_dashboard(self) -> Dict[str, Any]:
        """Create model performance dashboard"""
        return {
            "dashboard": {
                "title": "Model Performance",
                "tags": ["llm", "models"],
                "timezone": "utc",
                "refresh": "30s",
                "panels": [
                    {
                        "id": 1,
                        "title": "Model Load Success Rate",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "rate(llm_model_loads_total{status=\"success\"}[5m])",
                                "legendFormat": "Successful loads/sec"
                            },
                            {
                                "expr": "rate(llm_model_loads_total{status=\"failed\"}[5m])",
                                "legendFormat": "Failed loads/sec"
                            }
                        ]
                    }
                ]
            }
        }
    
    async def _create_resource_dashboard(self) -> Dict[str, Any]:
        """Create resource monitoring dashboard"""
        return {
            "dashboard": {
                "title": "Resource Monitoring",
                "tags": ["llm", "resources"],
                "timezone": "utc",
                "refresh": "30s",
                "panels": [
                    {
                        "id": 1,
                        "title": "GPU Memory Usage",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "llm_gpu_memory_usage_bytes",
                                "legendFormat": "GPU Memory (bytes)"
                            }
                        ]
                    },
                    {
                        "id": 2,
                        "title": "CPU Usage",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "llm_cpu_usage_percent",
                                "legendFormat": "CPU Usage (%)"
                            }
                        ]
                    }
                ]
            }
        }
    
    async def _create_cost_dashboard(self) -> Dict[str, Any]:
        """Create cost analysis dashboard"""
        return {
            "dashboard": {
                "title": "Cost Analysis",
                "tags": ["llm", "cost"],
                "timezone": "utc",
                "refresh": "5m",
                "panels": [
                    {
                        "id": 1,
                        "title": "Cost Over Time",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "increase(llm_cost_total[1h])",
                                "legendFormat": "Hourly Cost ($)"
                            }
                        ]
                    }
                ]
            }
        }
    
    async def _import_dashboard(self, dashboard_config: Dict[str, Any], title: str) -> None:
        """Import dashboard into Grafana"""
        try:
            if not self.session:
                raise RuntimeError("Session not initialized")
            
            import_payload = {
                "dashboard": dashboard_config["dashboard"],
                "overwrite": True,
                "inputs": []
            }
            
            async with self.session.post(
                f"{self.base_url}/api/dashboards/db",
                json=import_payload
            ) as response:
                if response.status == 200:
                    self.logger.info(f"Imported dashboard: {title}")
                else:
                    error_text = await response.text()
                    self.logger.warning(f"Failed to import dashboard {title}: {error_text}")
                    
        except Exception as e:
            self.logger.error(f"Failed to import dashboard {title}: {e}")


class OpenTelemetryIntegration(BaseIntegration):
    """Integration with OpenTelemetry for distributed tracing and observability"""
    
    def __init__(self, config: IntegrationConfig):
        super().__init__(config)
        self.service_name = config.settings.get("service_name", "llm-platform")
        self.service_version = config.settings.get("service_version", "1.0.0")
        self.otlp_endpoint = config.settings.get("otlp_endpoint", "http://localhost:4317")
        self.jaeger_endpoint = config.settings.get("jaeger_endpoint", "http://localhost:14268/api/traces")
        self.enable_auto_instrumentation = config.settings.get("enable_auto_instrumentation", True)
        
        self.tracer_provider: Optional[TracerProvider] = None
        self.tracer: Optional[trace.Tracer] = None
    
    async def initialize(self) -> bool:
        """Initialize OpenTelemetry integration"""
        try:
            # Setup tracer provider
            self._setup_tracer_provider()
            
            # Setup auto-instrumentation
            if self.enable_auto_instrumentation:
                self._setup_auto_instrumentation()
            
            self.logger.info("OpenTelemetry integration initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenTelemetry integration: {e}")
            return False
    
    async def health_check(self) -> bool:
        """Check OpenTelemetry health"""
        try:
            # Create a test span to verify tracing is working
            if self.tracer:
                with self.tracer.start_as_current_span("health_check"):
                    return True
            return False
        except Exception as e:
            self.logger.error(f"OpenTelemetry health check failed: {e}")
            return False
    
    async def shutdown(self) -> None:
        """Shutdown OpenTelemetry integration"""
        if self.tracer_provider:
            self.tracer_provider.shutdown()
    
    def _setup_tracer_provider(self) -> None:
        """Setup OpenTelemetry tracer provider"""
        try:
            resource = Resource.create({
                "service.name": self.service_name,
                "service.version": self.service_version
            })
            
            # Create tracer provider
            self.tracer_provider = TracerProvider(resource=resource)
            
            # Setup OTLP exporter
            otlp_exporter = OTLPSpanExporter(
                endpoint=self.otlp_endpoint,
                insecure=True
            )
            
            # Add span processor
            span_processor = BatchSpanProcessor(otlp_exporter)
            self.tracer_provider.add_span_processor(span_processor)
            
            # Set global tracer provider
            trace.set_tracer_provider(self.tracer_provider)
            
            # Get tracer
            self.tracer = trace.get_tracer(__name__)
            
            self.logger.info("OpenTelemetry tracer provider configured")
            
        except Exception as e:
            self.logger.error(f"Failed to setup tracer provider: {e}")
            raise
    
    def _setup_auto_instrumentation(self) -> None:
        """Setup auto-instrumentation for common libraries"""
        try:
            # Instrument FastAPI
            FastAPIInstrumentor().instrument()
            
            # Instrument aiohttp client
            AioHttpClientInstrumentor().instrument()
            
            # Instrument SQLAlchemy
            SQLAlchemyInstrumentor().instrument()
            
            self.logger.info("Auto-instrumentation configured")
            
        except Exception as e:
            self.logger.error(f"Failed to setup auto-instrumentation: {e}")
    
    def create_span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Create a new span"""
        if self.tracer:
            span = self.tracer.start_span(name)
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)
            return span
        return None


class AlertManagerIntegration(BaseIntegration):
    """Integration with Prometheus AlertManager for alerting"""
    
    def __init__(self, config: IntegrationConfig):
        super().__init__(config)
        self.base_url = config.settings.get("base_url", "http://localhost:9093")
        self.webhook_url = config.settings.get("webhook_url")
        self.slack_api_url = config.settings.get("slack_api_url")
        self.email_config = config.settings.get("email_config", {})
        
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def initialize(self) -> bool:
        """Initialize AlertManager integration"""
        try:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
            
            # Test connection
            await self._test_connection()
            
            # Generate AlertManager configuration
            await self._generate_alertmanager_config()
            
            self.logger.info("AlertManager integration initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize AlertManager integration: {e}")
            return False
    
    async def health_check(self) -> bool:
        """Check AlertManager health"""
        try:
            await self._test_connection()
            return True
        except Exception as e:
            self.logger.error(f"AlertManager health check failed: {e}")
            return False
    
    async def shutdown(self) -> None:
        """Shutdown AlertManager integration"""
        if self.session:
            await self.session.close()
    
    async def _test_connection(self) -> bool:
        """Test connection to AlertManager"""
        if not self.session:
            raise RuntimeError("Session not initialized")
        
        async with self.session.get(f"{self.base_url}/api/v1/status") as response:
            if response.status == 200:
                return True
            else:
                raise RuntimeError(f"AlertManager not available: {response.status}")
    
    async def _generate_alertmanager_config(self) -> None:
        """Generate AlertManager configuration"""
        try:
            config = {
                "global": {
                    "smtp_smarthost": self.email_config.get("smtp_host", "localhost:587"),
                    "smtp_from": self.email_config.get("from_email", "alerts@llm-platform.com")
                },
                "route": {
                    "group_by": ["alertname"],
                    "group_wait": "10s",
                    "group_interval": "10s",
                    "repeat_interval": "1h",
                    "receiver": "web.hook"
                },
                "receivers": [
                    {
                        "name": "web.hook",
                        "webhook_configs": [
                            {
                                "url": self.webhook_url or "http://localhost:5001/webhook",
                                "send_resolved": True
                            }
                        ]
                    }
                ]
            }
            
            # Add Slack configuration if available
            if self.slack_api_url:
                config["receivers"].append({
                    "name": "slack",
                    "slack_configs": [
                        {
                            "api_url": self.slack_api_url,
                            "channel": "#alerts",
                            "title": "LLM Platform Alert",
                            "text": "{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}"
                        }
                    ]
                })
            
            # Add email configuration if available
            if self.email_config.get("to_email"):
                config["receivers"].append({
                    "name": "email",
                    "email_configs": [
                        {
                            "to": self.email_config["to_email"],
                            "subject": "LLM Platform Alert: {{ .GroupLabels.alertname }}",
                            "body": "{{ range .Alerts }}{{ .Annotations.description }}{{ end }}"
                        }
                    ]
                })
            
            config_path = Path("monitoring/alertmanager.yml")
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
            
            self.logger.info(f"Generated AlertManager config: {config_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate AlertManager config: {e}")
    
    async def send_alert(self, alert_name: str, summary: str, description: str, severity: str = "warning") -> None:
        """Send a custom alert"""
        try:
            if not self.session:
                raise RuntimeError("Session not initialized")
            
            alert_data = [
                {
                    "labels": {
                        "alertname": alert_name,
                        "severity": severity,
                        "service": self.config.name
                    },
                    "annotations": {
                        "summary": summary,
                        "description": description
                    },
                    "startsAt": datetime.utcnow().isoformat() + "Z"
                }
            ]
            
            async with self.session.post(
                f"{self.base_url}/api/v1/alerts",
                json=alert_data
            ) as response:
                if response.status == 200:
                    self.logger.info(f"Alert sent: {alert_name}")
                else:
                    error_text = await response.text()
                    self.logger.error(f"Failed to send alert: {error_text}")
                    
        except Exception as e:
            self.logger.error(f"Failed to send alert {alert_name}: {e}")


# Default configurations for monitoring integrations
DEFAULT_PROMETHEUSINTEGRATION_CONFIG = IntegrationConfig(
    name="prometheus",
    type=IntegrationType.MONITORING,
    enabled=True,
    priority=95,
    settings={
        "port": 9090,
        "host": "localhost",
        "metrics_path": "/metrics",
        "job_name": "llm-platform",
        "scrape_interval": "15s",
        "start_metrics_server": False  # Served by main app
    },
    health_check_interval=60
)

DEFAULT_GRAFANAINTEGRATION_CONFIG = IntegrationConfig(
    name="grafana",
    type=IntegrationType.MONITORING,
    enabled=True,
    priority=85,
    dependencies=["prometheus"],
    settings={
        "base_url": "http://localhost:3000",
        "username": "admin",
        "password": "admin"
    },
    health_check_interval=120
)

DEFAULT_OPENTELEMETRYINTEGRATION_CONFIG = IntegrationConfig(
    name="opentelemetry",
    type=IntegrationType.MONITORING,
    enabled=True,
    priority=90,
    settings={
        "service_name": "llm-platform",
        "service_version": "1.0.0",
        "otlp_endpoint": "http://localhost:4317",
        "enable_auto_instrumentation": True
    },
    health_check_interval=60
)

DEFAULT_ALERTMANAGERINTEGRATION_CONFIG = IntegrationConfig(
    name="alertmanager",
    type=IntegrationType.MONITORING,
    enabled=True,
    priority=80,
    dependencies=["prometheus"],
    settings={
        "base_url": "http://localhost:9093",
        "email_config": {
            "smtp_host": "localhost:587",
            "from_email": "alerts@llm-platform.com"
        }
    },
    health_check_interval=120
)