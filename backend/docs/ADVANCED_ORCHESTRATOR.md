# Advanced AI Model Orchestrator

The Advanced AI Model Orchestrator is a sophisticated routing and optimization system that intelligently selects the best AI model for each request based on task complexity, cost optimization, performance monitoring, user preferences, and automatic fallback handling.

## 🎯 Key Features

### Intelligent Routing Logic
- **Task Complexity Analysis**: Automatically analyzes prompts to determine complexity (simple, moderate, complex, expert)
- **Cost Optimization**: Routes to cheaper models for simple tasks while maintaining quality
- **Performance Monitoring**: Tracks success rates and response times to optimize future routing
- **User Preferences**: Learns and adapts to individual user preferences and usage patterns
- **Fallback System**: Automatically retries with different models if the first choice fails

### Supported Models
- **OpenAI o3-mini**: Complex reasoning, high cost, excellent for expert-level tasks
- **Claude Sonnet 4**: Code generation, medium cost, great for development tasks  
- **Gemini Pro**: Multimodal support, variable cost, good for diverse content
- **GPT-4**: Balanced performance, reliable for most tasks
- **GPT-3.5 Turbo**: Fast and cost-effective for simple tasks

### Advanced Features
- **Streaming Response Support**: Real-time response streaming with intelligent model selection
- **Batch Processing**: Efficient parallel processing of multiple requests
- **Context Preservation**: Maintains conversation context across model switches
- **Budget Management**: Per-user and per-request budget limits with monitoring
- **Comprehensive Analytics**: Detailed performance metrics and cost tracking

## 🚀 Quick Start

### Basic Usage

```python
from app.services.advanced_orchestrator_service import ChatOrchestratorService
from app.models.advanced_ai_orchestrator import RoutingStrategy

# Initialize the service
chat_service = ChatOrchestratorService()

# Simple request with automatic routing
result = await chat_service.execute_chat_request(
    message="Explain machine learning",
    user_id=123,
    strategy=RoutingStrategy.BALANCED
)

print(f"Response: {result['response'].content}")
print(f"Model used: {result['model_used']}")
print(f"Cost: ${result['response'].cost:.4f}")
```

### Streaming Response

```python
# Stream response with real-time chunks
async for chunk in chat_service.stream_chat_response(
    message="Write a story about AI",
    user_id=123,
    strategy=RoutingStrategy.COST_OPTIMAL
):
    print(chunk, end="", flush=True)
```

### Batch Processing

```python
from app.services.advanced_orchestrator_service import BatchOrchestratorService

batch_service = BatchOrchestratorService()

# Submit multiple requests for parallel processing
batch_id = await batch_service.submit_batch(
    requests=[
        {"task": "Summarize this article: ..."},
        {"task": "Translate to Spanish: ..."},
        {"task": "Generate code for: ..."}
    ],
    user_id=123,
    priority=5
)

# Check batch status
status = await batch_service.get_batch_status(batch_id)
print(f"Status: {status['status']}")
```

## 🎮 API Endpoints

### Chat Operations

#### POST `/api/v1/orchestrator/chat/route`
Route a message to the optimal model without executing it.

```json
{
  "message": "Explain quantum computing",
  "budget_limit": 0.50,
  "strategy": "balanced",
  "stream": false
}
```

**Response:**
```json
{
  "chosen_provider": "openai",
  "chosen_model": "gpt-4",
  "reasoning": "balanced strategy: cost: $0.0234, speed: 3.0s, quality: 0.92",
  "estimated_cost": 0.0234,
  "estimated_time": 3.0,
  "fallback_models": [["anthropic", "claude-3-5-sonnet-20241022"]],
  "confidence": 0.89
}
```

#### POST `/api/v1/orchestrator/chat/execute`
Execute a chat request with optimal model selection and fallback.

```json
{
  "message": "Write a Python function for binary search",
  "strategy": "quality_optimal",
  "system_prompt": "You are a coding expert",
  "temperature": 0.2,
  "max_tokens": 1000
}
```

#### POST `/api/v1/orchestrator/chat/stream`
Stream a chat response with intelligent model selection.

### Analytics and Monitoring

#### GET `/api/v1/orchestrator/analytics/user`
Get user-specific analytics and recommendations.

**Response:**
```json
{
  "user_id": 123,
  "metrics": {
    "monthly_budget": 100.0,
    "current_spend": 15.67,
    "budget_utilization": 15.67,
    "cost_sensitivity": 0.7,
    "preferred_models": {
      "simple": ["gpt-3.5-turbo"],
      "complex": ["o3-mini", "claude-3-5-sonnet-20241022"]
    }
  },
  "recommendations": [
    {
      "type": "cost_optimization",
      "title": "Use Gemini Flash",
      "message": "Consider using Gemini Flash for simple tasks to reduce costs"
    }
  ]
}
```

#### GET `/api/v1/orchestrator/analytics/models`
Get comprehensive model performance analytics.

#### GET `/api/v1/orchestrator/health`
Check orchestrator and model client health.

### Batch Operations

#### POST `/api/v1/orchestrator/batch/submit`
Submit batch requests for parallel processing.

#### GET `/api/v1/orchestrator/batch/{batch_id}/status`
Get batch processing status and results.

### Configuration and Utilities

#### GET `/api/v1/orchestrator/models/capabilities`
Get detailed information about available models.

#### POST `/api/v1/orchestrator/preferences/update`
Update user preferences for model selection.

```json
{
  "cost_sensitivity": 0.8,
  "speed_preference": 0.6,
  "quality_preference": 0.9,
  "monthly_budget": 200.0
}
```

## 🔧 Routing Strategies

### Cost Optimal
- **Goal**: Minimize cost while maintaining quality
- **Use Cases**: Budget-conscious usage, simple tasks, high-volume processing
- **Selection**: Cheapest suitable model

### Performance Optimal  
- **Goal**: Minimize response time
- **Use Cases**: Real-time applications, user-facing chat, time-sensitive tasks
- **Selection**: Fastest suitable model

### Quality Optimal
- **Goal**: Maximum quality regardless of cost
- **Use Cases**: Critical tasks, research, creative work
- **Selection**: Highest quality model

### Balanced (Default)
- **Goal**: Balance cost, speed, and quality
- **Use Cases**: General usage, mixed workloads
- **Selection**: Weighted scoring based on user preferences

### User Preference
- **Goal**: Use historically preferred models
- **Use Cases**: Personalized experience, consistency
- **Selection**: Models user has had success with

## 🧠 Task Complexity Analysis

The orchestrator automatically analyzes tasks and classifies them:

### Simple Tasks
- Basic queries, translations, simple Q&A
- **Models**: GPT-3.5 Turbo, Gemini Flash
- **Example**: "What is the weather like?"

### Moderate Tasks
- Analysis, summaries, explanations
- **Models**: GPT-4, Claude Haiku
- **Example**: "Explain the advantages of renewable energy"

### Complex Tasks
- Code generation, complex reasoning
- **Models**: o3-mini, Claude Sonnet
- **Example**: "Write a machine learning algorithm"

### Expert Tasks
- Advanced problem solving, research
- **Models**: o3-mini, GPT-4
- **Example**: "Analyze quantum computing's impact on cryptography"

## 💰 Cost Management

### Budget Controls
- **User Monthly Budgets**: Configurable spending limits per user
- **Request-Level Limits**: Maximum cost per individual request
- **Real-time Monitoring**: Track spending against budgets
- **Automatic Alerts**: Notifications when approaching limits

### Cost Optimization Features
- **Model Selection**: Route simple tasks to cheaper models
- **Token Estimation**: Predict costs before execution
- **Usage Analytics**: Detailed cost breakdowns and trends
- **Recommendations**: Suggestions for cost reduction

## 📊 Performance Monitoring

### Metrics Tracked
- **Success Rates**: Model reliability and failure rates
- **Response Times**: Average latency per model
- **Cost per Token**: Efficiency measurements
- **User Satisfaction**: Implicit feedback from usage patterns

### Analytics Dashboard
- **Model Performance**: Comparative analysis across providers
- **User Patterns**: Individual usage and preference trends
- **System Health**: Overall orchestrator status
- **Cost Analytics**: Spending patterns and optimization opportunities

## 🔄 Fallback and Retry Logic

### Automatic Fallback
1. **Primary Model Fails**: Automatically tries the next best option
2. **Multiple Attempts**: Up to 3 attempts with different models
3. **Smart Ordering**: Fallback models ranked by suitability
4. **Context Preservation**: Maintains request context across attempts

### Failure Handling
- **Graceful Degradation**: Always provides a response when possible
- **Error Tracking**: Logs failures for analysis and improvement
- **User Feedback**: Transparent reporting of retry attempts
- **Learning**: Updates model reliability scores based on outcomes

## 🏥 Health Monitoring

### System Health Checks
- **Model Availability**: Real-time status of AI providers
- **Cache Connectivity**: Redis connection health
- **Performance Metrics**: Response time and success rate monitoring
- **Resource Usage**: Active streams and queue sizes

### Alerts and Notifications
- **Model Outages**: Automatic detection and alerts
- **Performance Degradation**: Threshold-based warnings
- **Cost Anomalies**: Unusual spending pattern detection
- **System Errors**: Comprehensive error tracking and reporting

## 🔧 Configuration

### Environment Variables

```bash
# AI Model API Keys
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
GOOGLE_API_KEY=your-google-key

# Cost Tracking
COST_TRACKING_ENABLED=true
MONTHLY_COST_LIMIT=1000.0

# Rate Limiting
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60

# Redis Cache
REDIS_URL=redis://localhost:6379
REDIS_PASSWORD=optional-password

# Default Model Settings
DEFAULT_MODEL_PROVIDER=openai
DEFAULT_MODEL_NAME=gpt-4
DEFAULT_TEMPERATURE=0.7
DEFAULT_MAX_TOKENS=2048
```

### Model Configuration

Models can be configured in the orchestrator initialization:

```python
# Custom model capabilities
model_capabilities = {
    "custom_model": ModelCapabilities(
        provider=ModelProvider.CUSTOM,
        model_name="custom-model-v1",
        max_tokens=4096,
        cost_per_1k_input=0.001,
        cost_per_1k_output=0.003,
        supports_code=True,
        supports_reasoning=True,
        supports_streaming=True
    )
}
```

## 🧪 Testing and Development

### Running the Demo

```bash
# Navigate to the backend directory
cd backend

# Run the comprehensive demo
python examples/advanced_orchestrator_demo.py
```

### Unit Tests

```bash
# Run orchestrator tests
pytest tests/test_advanced_orchestrator.py -v

# Run with coverage
pytest tests/test_advanced_orchestrator.py --cov=app.models.advanced_ai_orchestrator
```

### Development Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# Start Redis (required for caching)
redis-server

# Run the application
python main.py
```

## 📈 Usage Examples

### Example 1: Simple Chat Integration

```python
async def handle_user_message(message: str, user_id: int):
    """Handle a user message with optimal routing"""
    
    chat_service = ChatOrchestratorService()
    
    # Route and execute with budget limit
    result = await chat_service.execute_chat_request(
        message=message,
        user_id=user_id,
        budget_limit=1.00,  # Max $1 per request
        strategy=RoutingStrategy.BALANCED
    )
    
    return {
        "response": result["response"].content,
        "model": result["model_used"],
        "cost": result["response"].cost,
        "reasoning": result["routing_decision"]["reasoning"]
    }
```

### Example 2: Code Generation Workflow

```python
async def generate_code(description: str, user_id: int):
    """Generate code with quality-focused routing"""
    
    chat_service = ChatOrchestratorService()
    
    # Use quality-optimal strategy for code generation
    result = await chat_service.execute_chat_request(
        message=f"Write Python code: {description}",
        user_id=user_id,
        strategy=RoutingStrategy.QUALITY_OPTIMAL,
        system_prompt="You are an expert Python developer. Write clean, well-documented code.",
        temperature=0.2  # Lower temperature for deterministic code
    )
    
    return result["response"].content
```

### Example 3: Cost-Conscious Bulk Processing

```python
async def process_documents(documents: List[str], user_id: int):
    """Process multiple documents with cost optimization"""
    
    batch_service = BatchOrchestratorService()
    
    # Prepare batch requests with cost-optimal strategy
    requests = [
        {
            "task": f"Summarize this document: {doc}",
            "strategy": "cost_optimal",
            "max_tokens": 500
        }
        for doc in documents
    ]
    
    # Submit batch
    batch_id = await batch_service.submit_batch(
        requests=requests,
        user_id=user_id,
        priority=1
    )
    
    # Poll for completion
    while True:
        status = await batch_service.get_batch_status(batch_id)
        if status["status"] == "completed":
            return status["results"]
        elif status["status"] == "failed":
            raise Exception("Batch processing failed")
        
        await asyncio.sleep(5)  # Wait 5 seconds before checking again
```

### Example 4: Real-time Streaming Chat

```python
async def stream_chat_response(message: str, user_id: int):
    """Stream a chat response with performance optimization"""
    
    chat_service = ChatOrchestratorService()
    
    print("AI: ", end="", flush=True)
    
    async for chunk in chat_service.stream_chat_response(
        message=message,
        user_id=user_id,
        strategy=RoutingStrategy.PERFORMANCE_OPTIMAL
    ):
        print(chunk, end="", flush=True)
    
    print()  # New line after streaming
```

## 🔮 Advanced Customization

### Custom Routing Logic

```python
class CustomOrchestrator(AdvancedAIOrchestrator):
    """Custom orchestrator with specialized routing"""
    
    async def custom_route_for_domain(self, task: str, domain: str, user_id: int):
        """Custom routing based on domain expertise"""
        
        # Domain-specific model preferences
        domain_models = {
            "medical": ["claude-3-5-sonnet-20241022"],  # Better for medical content
            "code": ["o3-mini", "claude-3-5-sonnet-20241022"],  # Code specialists
            "creative": ["gpt-4", "claude-3-5-sonnet-20241022"],  # Creative tasks
            "analytical": ["o3-mini", "gpt-4"]  # Data analysis
        }
        
        # Get analysis
        analysis = await self._analyze_task_complexity(task)
        
        # Override model selection based on domain
        if domain in domain_models:
            preferred_models = domain_models[domain]
            # Custom scoring logic here...
        
        return await self.route_request(task, user_id)
```

### Custom Performance Metrics

```python
async def track_custom_metrics(orchestrator: AdvancedAIOrchestrator):
    """Track custom performance metrics"""
    
    # Add custom metric tracking
    await orchestrator.cache.set(
        "custom_metric:user_satisfaction",
        {"score": 4.5, "timestamp": time.time()},
        ttl=3600
    )
    
    # Custom analytics
    analytics = await orchestrator.get_analytics()
    custom_analytics = {
        **analytics,
        "custom_metrics": {
            "user_satisfaction": 4.5,
            "cost_efficiency": calculate_cost_efficiency(analytics),
            "response_quality": assess_quality_scores(analytics)
        }
    }
    
    return custom_analytics
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Model Not Available
```
Error: Client not available: openai
```
**Solution**: Check API keys and model client initialization
- Verify API keys in environment variables
- Check network connectivity
- Ensure models are properly initialized in startup

#### 2. Redis Connection Failed
```
Error: Failed to connect to Redis
```
**Solution**: Verify Redis configuration
- Check Redis server is running
- Verify REDIS_URL environment variable
- Check Redis authentication if required

#### 3. Budget Exceeded
```
Error: No suitable models available for this request
```
**Solution**: Check budget limits and model costs
- Verify user monthly budget isn't exceeded
- Check request-level budget limits
- Consider using cost-optimal routing strategy

#### 4. Task Complexity Analysis Failed
```
Error: Failed to analyze task complexity
```
**Solution**: Review task content and patterns
- Check for extremely long or malformed prompts
- Verify regex patterns in complexity analysis
- Review error logs for specific issues

### Debug Mode

Enable detailed logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable orchestrator debug logging
orchestrator.logger.setLevel(logging.DEBUG)
```

### Performance Optimization

1. **Cache Configuration**: Optimize Redis settings for your workload
2. **Batch Size**: Adjust batch processing limits based on system capacity
3. **Model Selection**: Fine-tune model capabilities and cost settings
4. **Rate Limiting**: Configure appropriate rate limits for your usage patterns

## 📚 Additional Resources

- [API Documentation](./API_REFERENCE.md)
- [Model Comparison Guide](./MODEL_COMPARISON.md)
- [Cost Optimization Best Practices](./COST_OPTIMIZATION.md)
- [Performance Tuning Guide](./PERFORMANCE_TUNING.md)

## 🤝 Contributing

Contributions are welcome! Please see our [Contributing Guide](../CONTRIBUTING.md) for details on:
- Setting up the development environment
- Running tests
- Submitting pull requests
- Code style guidelines

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

---

*For support or questions, please open an issue in the GitHub repository or contact the development team.*