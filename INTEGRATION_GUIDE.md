# LLM Platform Integration Guide

A comprehensive guide to the extensive integration layer built for the Local LLM System, providing seamless connectivity with various platforms, tools, and services.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Model Platform Integrations](#model-platform-integrations)
4. [Monitoring & Observability](#monitoring--observability)
5. [API Compatibility](#api-compatibility)
6. [CI/CD Pipeline](#cicd-pipeline)
7. [Development Tools](#development-tools)
8. [External Services](#external-services)
9. [Configuration](#configuration)
10. [Best Practices](#best-practices)
11. [Troubleshooting](#troubleshooting)

## Overview

The LLM Platform Integration Layer provides a unified, extensible framework for connecting with various external systems, platforms, and services. Built with enterprise-grade reliability, monitoring, and security in mind.

### Key Features

- **Plugin Architecture**: Extensible framework for adding new integrations
- **Health Monitoring**: Continuous health checks and status monitoring
- **Event-Driven**: Real-time event emission and handling
- **Dependency Management**: Automatic startup ordering based on dependencies
- **Graceful Degradation**: System continues operating when services are unavailable
- **Comprehensive Logging**: Detailed logging and error tracking

## Architecture

```mermaid
graph TB
    subgraph "Integration Framework Core"
        IM[Integration Manager]
        IR[Integration Registry]
        BF[Base Integration Framework]
    end
    
    subgraph "Model Platform Integrations"
        OL[Ollama Integration]
        HF[HuggingFace Hub]
        OAI[OpenAI Compatible]
        CM[Custom Models]
    end
    
    subgraph "Monitoring & Observability"
        PR[Prometheus]
        GR[Grafana]
        OT[OpenTelemetry]
        AM[AlertManager]
    end
    
    subgraph "API Compatibility"
        OA[OpenAI API]
        GQL[GraphQL]
        GRPC[gRPC]
        WS[WebSocket]
    end
    
    subgraph "External Services"
        DB[Databases]
        Q[Message Queues]
        S[Object Storage]
        AU[Authentication]
    end
    
    IM --> IR
    IR --> BF
    BF --> OL
    BF --> HF
    BF --> OAI
    BF --> CM
    BF --> PR
    BF --> GR
    BF --> OT
    BF --> AM
    BF --> OA
    BF --> GQL
    BF --> GRPC
    BF --> WS
    BF --> DB
    BF --> Q
    BF --> S
    BF --> AU
```

## Model Platform Integrations

### 1. Ollama Integration

Provides seamless integration with Ollama for local model management.

#### Features
- Automatic model discovery
- Model pulling and management
- Real-time model status monitoring
- Text generation with streaming support

#### Configuration
```python
from app.integrations.model_platforms import OllamaIntegration, DEFAULT_OLLAMAINTEGRATION_CONFIG

# Default configuration
config = DEFAULT_OLLAMAINTEGRATION_CONFIG
config.settings.update({
    "base_url": "http://localhost:11434",
    "timeout": 30
})
```

#### Usage
```python
# Get integration manager
manager = await get_integration_manager()
ollama = manager.get_integration("ollama")

# Pull a model
result = await ollama.pull_model("llama2:7b")

# Generate text
response = await ollama.generate("llama2:7b", "Hello, world!")
```

### 2. HuggingFace Hub Integration

Connect with HuggingFace Hub for model discovery and downloading.

#### Features
- Model search and discovery
- Model information retrieval
- Model downloading with caching
- Authentication support

#### Configuration
```python
config = DEFAULT_HUGGINGFACEHUBINTEGRATION_CONFIG
config.settings.update({
    "api_token": os.getenv("HUGGINGFACE_TOKEN"),
    "cache_dir": "~/.cache/huggingface"
})
```

#### Usage
```python
hf_hub = manager.get_integration("huggingface_hub")

# Search for models
models = await hf_hub.search_models("llama", limit=10)

# Get model information
info = await hf_hub.get_model_info("microsoft/DialoGPT-medium")

# Download model
result = await hf_hub.download_model("microsoft/DialoGPT-medium")
```

### 3. OpenAI Compatible Integration

Support for OpenAI-compatible local servers (LocalAI, text-generation-webui).

#### Configuration
```python
config = DEFAULT_OPENAICOMPATIBLEINTEGRATION_CONFIG
config.settings.update({
    "base_url": "http://localhost:8000",
    "api_key": "not-needed",
    "timeout": 30
})
```

### 4. Custom Model Integration

Support for custom model formats and runtimes.

#### Supported Formats
- `.gguf` - GGUF format (llama.cpp)
- `.ggml` - GGML format (llama.cpp)
- `.bin` - Binary format (transformers)
- `.safetensors` - SafeTensors format
- `.pt`, `.pth` - PyTorch formats

## Monitoring & Observability

### 1. Prometheus Integration

Comprehensive metrics collection and export.

#### Custom Metrics
- `llm_requests_total` - Total number of LLM requests
- `llm_request_duration_seconds` - Request duration histogram
- `llm_model_loads_total` - Model load counter
- `llm_active_models` - Number of active models
- `llm_gpu_memory_usage_bytes` - GPU memory usage
- `llm_cpu_usage_percent` - CPU usage percentage
- `llm_errors_total` - Error counter
- `llm_cost_total` - Total cost tracker

#### Configuration
```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'llm-platform'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 15s

rule_files:
  - "llm_platform_rules.yml"
```

### 2. Grafana Integration

Pre-built dashboards for visualization.

#### Available Dashboards
- **LLM Platform Overview**: High-level system metrics
- **Model Performance**: Model-specific performance metrics
- **Resource Monitoring**: CPU, GPU, memory usage
- **Cost Analysis**: Cost tracking and optimization

#### Automatic Setup
```python
grafana = manager.get_integration("grafana")
# Dashboards are automatically created and imported
```

### 3. OpenTelemetry Integration

Distributed tracing and observability.

#### Features
- Automatic instrumentation for FastAPI, aiohttp, SQLAlchemy
- OTLP export support
- Custom span creation
- Service topology mapping

#### Usage
```python
otel = manager.get_integration("opentelemetry")

# Create custom span
with otel.create_span("model_inference", {"model": "llama2-7b"}):
    result = await model.generate(prompt)
```

### 4. AlertManager Integration

Intelligent alerting system.

#### Pre-configured Alerts
- High error rate (>10% in 5 minutes)
- High GPU memory usage (>90%)
- Model load failures
- High request latency (95th percentile >10s)

## API Compatibility

### 1. OpenAI API Compatibility

Full OpenAI API compatibility for seamless integration with existing tools.

#### Supported Endpoints
- `/v1/chat/completions` - Chat completions
- `/v1/completions` - Text completions
- `/v1/models` - List available models
- `/v1/embeddings` - Text embeddings

#### Example Usage
```bash
# Chat completion
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama2-7b",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "stream": true
  }'
```

### 2. GraphQL Integration

Advanced query capabilities with real-time subscriptions.

#### Available Operations
- **Queries**: Models, generation requests, metrics
- **Mutations**: Generate text, load models
- **Subscriptions**: Model status updates, metrics stream

#### Example Query
```graphql
query GetModels {
  models {
    id
    name
    provider
    status
    capabilities
  }
}

mutation GenerateText($model: String!, $prompt: String!) {
  generateText(model: $model, prompt: $prompt) {
    id
    text
    tokens
    cost
  }
}

subscription ModelUpdates {
  modelStatusUpdates {
    id
    status
  }
}
```

### 3. gRPC Integration

High-performance inter-service communication.

#### Features
- Protocol buffer definitions
- Service reflection
- Load balancing support
- Authentication integration

### 4. WebSocket Integration

Real-time communication for chat, model status, and metrics.

#### Endpoint Types
- `/ws/chat` - Real-time chat interface
- `/ws/models` - Model status updates
- `/ws/metrics` - Live metrics streaming

#### Example Usage
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/chat');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Response:', data.message);
};

ws.send(JSON.stringify({
  message: "Hello, AI!",
  model: "llama2-7b"
}));
```

## CI/CD Pipeline

Comprehensive CI/CD pipeline with GitHub Actions.

### Pipeline Stages

1. **Code Quality & Security**
   - Black code formatting
   - isort import sorting
   - Pylint linting
   - MyPy type checking
   - Bandit security scanning
   - Safety dependency scanning

2. **Testing**
   - Backend unit tests with coverage
   - Frontend tests with coverage
   - Integration tests
   - Performance tests (K6)

3. **Security Scanning**
   - Trivy vulnerability scanning
   - Container security scanning
   - SARIF report generation

4. **Build & Deploy**
   - Multi-stage Docker builds
   - Container registry push
   - Staging deployment
   - Production deployment
   - Post-deployment verification

### Configuration

Pipeline automatically triggers on:
- Push to `main` or `develop` branches
- Pull requests to `main`
- Release creation

### Required Secrets
```yaml
KUBE_CONFIG_STAGING: # Base64 encoded kubeconfig for staging
KUBE_CONFIG_PRODUCTION: # Base64 encoded kubeconfig for production
SLACK_WEBHOOK_URL: # Slack webhook for notifications
GRAFANA_API_KEY: # Grafana API key (optional)
HUGGINGFACE_TOKEN: # HuggingFace API token (optional)
```

## Development Tools

### VS Code Integration

Provides development environment integration.

#### Features
- Model management commands
- Debugging support
- Integrated terminal
- Extension recommendations

### Docker Integration

Container orchestration and management.

#### Available Compose Files
- `docker-compose.yml` - Basic development
- `docker-compose.local.yml` - Local development with GPU
- `docker-compose.prod.yml` - Production deployment

### Kubernetes Integration

Production-grade orchestration.

#### Resources
- Deployments
- Services
- ConfigMaps
- Secrets
- Ingress
- HPA (Horizontal Pod Autoscaler)

### Helm Integration

Package management for Kubernetes.

#### Chart Features
- Configurable values
- Multi-environment support
- Resource management
- Dependency handling

## Configuration

### Environment Variables

```bash
# Core Configuration
LLM_PLATFORM_ENV=production
LOG_LEVEL=INFO
DATABASE_URL=postgresql://user:pass@localhost:5432/db
REDIS_URL=redis://localhost:6379/0

# Model Platform Integrations
OLLAMA_BASE_URL=http://localhost:11434
HUGGINGFACE_TOKEN=your_token_here

# Monitoring
PROMETHEUS_ENABLED=true
GRAFANA_ENABLED=true
OPENTELEMETRY_ENABLED=true

# API Compatibility
OPENAI_API_ENABLED=true
GRAPHQL_ENABLED=true
GRPC_ENABLED=true
WEBSOCKET_ENABLED=true
```

### Integration Configuration

Each integration can be configured through the `IntegrationConfig` class:

```python
from app.core.integration_framework import IntegrationConfig, IntegrationType

config = IntegrationConfig(
    name="my_integration",
    type=IntegrationType.MODEL_PLATFORM,
    enabled=True,
    priority=100,
    dependencies=["other_integration"],
    settings={
        "custom_setting": "value"
    },
    health_check_interval=60,
    retry_config={
        "max_retries": 3,
        "backoff_factor": 2,
        "max_backoff": 300
    }
)
```

## Best Practices

### 1. Integration Development

- **Inherit from BaseIntegration**: Use the provided base class
- **Implement Required Methods**: `initialize()`, `health_check()`, `shutdown()`
- **Handle Errors Gracefully**: Don't crash the system on integration failures
- **Use Async/Await**: Ensure non-blocking operations
- **Emit Events**: Use the event system for notifications

### 2. Configuration Management

- **Environment-Specific Configs**: Use different configs for dev/staging/prod
- **Secret Management**: Use proper secret management for sensitive data
- **Validation**: Validate configuration on startup
- **Documentation**: Document all configuration options

### 3. Monitoring & Logging

- **Structured Logging**: Use consistent log formats
- **Metric Naming**: Follow Prometheus naming conventions
- **Alert Thresholds**: Set meaningful alert thresholds
- **Dashboard Design**: Create clear, actionable dashboards

### 4. Testing

- **Unit Tests**: Test integration logic in isolation
- **Integration Tests**: Test actual service interactions
- **Mocking**: Mock external services in tests
- **Performance Tests**: Load test critical paths

## Troubleshooting

### Common Issues

#### 1. Integration Fails to Start

**Symptoms**: Integration status shows "error" or "inactive"

**Solutions**:
- Check logs for specific error messages
- Verify configuration settings
- Ensure dependencies are available
- Check network connectivity

```bash
# Check integration status
curl http://localhost:8000/integrations/health

# Check specific integration
curl http://localhost:8000/integrations/ollama/status
```

#### 2. Health Checks Failing

**Symptoms**: Intermittent health check failures

**Solutions**:
- Increase health check timeout
- Check service availability
- Review network latency
- Adjust health check interval

#### 3. High Memory Usage

**Symptoms**: System running out of memory

**Solutions**:
- Monitor model loading patterns
- Implement model unloading
- Adjust batch sizes
- Scale horizontally

#### 4. API Compatibility Issues

**Symptoms**: OpenAI API requests failing

**Solutions**:
- Check API version compatibility
- Verify request format
- Review model availability
- Check authentication

### Debugging Tools

#### 1. Integration Status Endpoint

```bash
curl http://localhost:8000/integrations/status
```

#### 2. Integration Metrics

```bash
curl http://localhost:8000/metrics | grep integration_
```

#### 3. Health Check Endpoint

```bash
curl http://localhost:8000/integrations/health
```

#### 4. Log Analysis

```bash
# Container logs
docker logs llm-platform-backend

# Kubernetes logs
kubectl logs -f deployment/llm-platform -n production
```

### Performance Optimization

#### 1. Model Loading

- Use model quantization
- Implement lazy loading
- Cache frequently used models
- Optimize memory usage

#### 2. API Response Times

- Enable response caching
- Use connection pooling
- Implement request batching
- Optimize database queries

#### 3. Resource Utilization

- Monitor GPU utilization
- Balance CPU/GPU workloads
- Implement auto-scaling
- Use resource limits

## Support and Contributing

### Getting Help

1. **Documentation**: Check this guide and API documentation
2. **Issues**: Submit issues on GitHub
3. **Discussions**: Use GitHub Discussions for questions
4. **Community**: Join our Discord server

### Contributing

1. **Fork the Repository**: Create your own fork
2. **Create Feature Branch**: Use descriptive branch names
3. **Write Tests**: Include comprehensive tests
4. **Update Documentation**: Update relevant documentation
5. **Submit Pull Request**: Follow the PR template

### Development Setup

```bash
# Clone repository
git clone https://github.com/your-org/llm-platform.git
cd llm-platform

# Install dependencies
pip install -r backend/requirements.txt
npm install --prefix frontend

# Start development environment
docker-compose -f docker-compose.local.yml up -d

# Run tests
pytest backend/tests/
npm test --prefix frontend
```

---

For more detailed information, refer to the specific integration documentation in the `backend/app/integrations/` directory.