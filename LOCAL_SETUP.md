# Local LLM Development Setup

A simplified Docker Compose setup for local development with Ollama integration, avoiding Kubernetes complexity while maintaining all core functionality.

## 🚀 Quick Start

### Prerequisites
- Docker Desktop with WSL2 support
- At least 8GB RAM (16GB+ recommended for larger models)
- Optional: NVIDIA GPU with Docker GPU support for acceleration

### 1. Start the System

**CPU-only mode:**
```bash
scripts\start-local.bat
```

**GPU-accelerated mode (if NVIDIA GPU available):**
```bash
scripts\start-gpu.bat
```

### 2. Access Services

| Service | URL | Description |
|---------|-----|-------------|
| Frontend | http://localhost:7000 | React web interface |
| Backend API | http://localhost:8000 | FastAPI server |
| API Documentation | http://localhost:8000/docs | Interactive API docs |
| Monitoring | http://localhost:9090 | Prometheus metrics |
| Task Monitor | http://localhost:5555 | Celery task queue (admin/admin123) |
| Database | localhost:5432 | PostgreSQL (postgres/postgres123) |
| Redis | localhost:6379 | Cache and message broker |
| Ollama | localhost:11434 | Local LLM service |

### 3. Test LLM Integration

```bash
# Check if Ollama is running
curl http://localhost:11434/api/version

# List available models
curl http://localhost:11434/api/tags

# Test generation via backend
curl -X POST http://localhost:8000/api/v1/local-llm/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello, how are you?",
    "provider": "local_ollama",
    "model": "llama2:7b"
  }'
```

## 🛠️ Configuration

### Environment Variables
The system uses sensible defaults, but you can customize by editing `docker-compose.local.yml`:

- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis connection string  
- `OLLAMA_HOST`: Ollama service endpoint
- `MONTHLY_COST_LIMIT`: Cost tracking limit ($100 default)

### Adding More Models
```bash
# Enter Ollama container
docker exec -it llm-ollama bash

# Pull additional models
ollama pull mistral:7b
ollama pull codellama:13b
ollama pull neural-chat:7b
```

### GPU Configuration
For NVIDIA GPU support:
1. Install NVIDIA Container Toolkit
2. Use `start-gpu.bat` instead of `start-local.bat`
3. Models will automatically use GPU acceleration

## 📊 Monitoring

### Prometheus Metrics
- **Backend metrics**: http://localhost:9090
- **Query examples**:
  - `rate(http_requests_total[5m])` - Request rate
  - `histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))` - Response time P95

### Task Monitoring
- **Flower dashboard**: http://localhost:5555
- Monitor background tasks, queue lengths, worker status
- Basic auth: admin/admin123

## 🔧 Development Workflow

### Making Changes
1. Edit backend code in `./backend/` - hot reload enabled
2. Edit frontend code in `./frontend/` - hot reload enabled
3. Changes reflect immediately without rebuilding

### Debugging
```bash
# View all logs
docker-compose -f docker-compose.local.yml logs -f

# View specific service logs
docker-compose -f docker-compose.local.yml logs -f backend
docker-compose -f docker-compose.local.yml logs -f ollama

# Check service health
docker-compose -f docker-compose.local.yml ps
```

### Database Operations
```bash
# Connect to PostgreSQL
docker exec -it llm-postgres psql -U postgres -d llm_optimization

# Run migrations
docker-compose -f docker-compose.local.yml exec backend alembic upgrade head
```

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend       │    │   Ollama        │
│   React/Vite    │────│   FastAPI       │────│   LLM Service   │
│   Port: 7000    │    │   Port: 8000    │    │   Port: 11434   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌─────────────────┐              │
         │              │   PostgreSQL    │              │
         └──────────────│   Database      │──────────────┘
                        │   Port: 5432    │
                        └─────────────────┘
                                 │
                        ┌─────────────────┐
                        │   Redis Cache   │
                        │   Port: 6379    │
                        └─────────────────┘
                                 │
                        ┌─────────────────┐
                        │   Celery        │
                        │   Workers       │
                        └─────────────────┘
```

## 🚀 Scaling to Production

When ready to deploy to cloud:

### Option 1: Simple Cloud Deployment
- **Railway/Render**: Deploy backend as container
- **Vercel/Netlify**: Deploy frontend 
- **Upstash Redis**: Managed Redis (free tier)
- **Supabase**: Managed PostgreSQL (free tier)

### Option 2: Container Platform
- **Google Cloud Run**: Serverless containers
- **AWS ECS**: Container orchestration
- **Azure Container Instances**: Simple container hosting

### Option 3: Full Kubernetes (when needed)
- Use existing `k8s/` manifests
- Deploy to GKE, EKS, or AKS
- Add autoscaling and load balancing

## 🛡️ Security Notes

### Development Security
- Default passwords used (change in production)
- No TLS/SSL (add reverse proxy for production)
- Basic authentication on monitoring

### Production Considerations
- Use environment-specific secrets
- Enable TLS everywhere
- Implement proper authentication
- Set up monitoring and alerting
- Configure backup strategies

## 📝 Troubleshooting

### Common Issues

**Ollama not starting:**
```bash
# Check logs
docker-compose -f docker-compose.local.yml logs ollama

# Restart service
docker-compose -f docker-compose.local.yml restart ollama
```

**Out of memory:**
```bash
# Check resource usage
docker stats

# Scale down or use smaller models
docker exec llm-ollama ollama pull llama2:7b  # instead of 13b
```

**Database connection errors:**
```bash
# Reset database
docker-compose -f docker-compose.local.yml down -v
docker-compose -f docker-compose.local.yml up -d
```

**Port conflicts:**
Edit `docker-compose.local.yml` to change port mappings if needed.

## 🛑 Stopping Services

```bash
# Stop all services
scripts\stop-local.bat

# Or manually
docker-compose -f docker-compose.local.yml down

# Remove all data (careful!)
docker-compose -f docker-compose.local.yml down -v
```

## 💡 Tips

1. **Model Selection**: Start with 7B models for development, scale to 13B+ for production
2. **Resource Management**: Monitor GPU/CPU usage in Docker Desktop
3. **Caching**: Redis caches frequently used responses - check hit rates
4. **Development**: Use hot reload for faster iteration
5. **Production**: Plan for persistent storage and backup strategies

This setup gives you full local LLM capabilities without Kubernetes complexity, while maintaining a clear path to production scaling.