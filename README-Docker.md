# 🐳 AI Agent System - Docker Setup & Integration Guide

A comprehensive Docker-based setup for the AI Agent System with full integration testing, monitoring, and troubleshooting capabilities.

## 🚀 Quick Start

```bash
# 1. Clone and navigate to the project
cd ai-agent-system

# 2. Copy environment file and configure
cp .env.example .env
# Edit .env with your API keys and settings

# 3. Start the entire system
./startup-verification.sh

# 4. Access the system
open http://localhost:3000  # Frontend
open http://localhost:8000/docs  # Backend API
```

## 📋 System Components

### Core Services
- **Frontend** (React TypeScript) - Port 3000
- **Backend** (FastAPI) - Port 8000  
- **PostgreSQL Database** - Port 5432
- **Redis Cache** - Port 6379
- **Ollama Local LLM Server** - Port 11434

### Monitoring & Management
- **Prometheus** (Metrics) - Port 9090
- **Grafana** (Dashboard) - Port 3001
- **Redis Commander** - Port 8081 (dev only)
- **PgAdmin** - Port 5050 (dev only)

## 🔧 Prerequisites

### Required Software
```bash
# Docker & Docker Compose
docker --version  # >= 20.10
docker-compose --version  # >= 2.0

# Optional but recommended
python3 --version  # >= 3.8 (for integration tests)
curl --version     # for health checks
```

### System Requirements
- **Memory**: 8GB RAM minimum, 16GB recommended
- **CPU**: 4 cores minimum (for Ollama)
- **Disk**: 20GB free space minimum
- **OS**: Linux, macOS, Windows (with WSL2)

## 📁 Project Structure

```
ai-agent-system/
├── docker-compose.yml           # Main services configuration
├── docker-compose.override.yml  # Development overrides
├── .env.example                 # Environment template
├── health-check.sh              # Service health verification
├── integration-test.py          # Comprehensive test suite
├── startup-verification.sh      # Complete startup automation
├── README-Docker.md            # This guide
├── frontend/                   # React TypeScript app
├── backend/                    # FastAPI application
└── monitoring/                 # Prometheus configuration
```

## ⚙️ Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# Required - Database
POSTGRES_PASSWORD=your-secure-password

# Required - Security  
SECRET_KEY=your-super-secret-key-min-32-chars

# Optional - AI API Keys
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
GOOGLE_API_KEY=your-google-key

# Optional - Monitoring
GRAFANA_PASSWORD=admin
```

### Docker Compose Profiles

```bash
# Development (includes dev tools)
docker-compose up -d

# Production (minimal services)
docker-compose -f docker-compose.yml up -d

# GPU-enabled Ollama
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

## 🚦 Startup Procedures

### Method 1: Automated Startup (Recommended)

```bash
# Complete automated startup with verification
./startup-verification.sh

# Options
./startup-verification.sh --help
./startup-verification.sh --skip-tests      # Skip integration tests
./startup-verification.sh --debug           # Verbose output
```

### Method 2: Manual Startup

```bash
# 1. Start core services first
docker-compose up -d postgres redis

# 2. Wait for databases to be ready
./health-check.sh

# 3. Start application services
docker-compose up -d backend frontend

# 4. Start monitoring services
docker-compose up -d prometheus grafana

# 5. Verify all services
./health-check.sh
```

### Method 3: Service-by-Service

```bash
# Start individual services
docker-compose up -d postgres
docker-compose up -d redis
docker-compose up -d ollama
docker-compose up -d backend
docker-compose up -d frontend
```

## 🔍 Health Checks & Monitoring

### Automated Health Checks

```bash
# Run comprehensive health check
./health-check.sh

# Options
./health-check.sh --quiet      # Errors only
./health-check.sh --verbose    # Debug info
```

### Manual Health Verification

```bash
# Check service status
docker-compose ps

# Check specific service logs
docker-compose logs backend
docker-compose logs frontend

# Check container health
docker inspect ai-agent-system-backend-1 | grep Health -A 10
```

### Health Check Endpoints

| Service | Health Endpoint | Expected Response |
|---------|----------------|-------------------|
| Backend | `http://localhost:8000/health` | `{"status": "healthy"}` |
| Frontend | `http://localhost:3000/` | HTML page |
| Ollama | `http://localhost:11434/api/tags` | Model list JSON |
| Prometheus | `http://localhost:9090/-/healthy` | OK |
| Grafana | `http://localhost:3001/api/health` | `{"database": "ok"}` |

## 🧪 Integration Testing

### Automated Integration Tests

```bash
# Run comprehensive integration test suite
python3 integration-test.py

# Install test dependencies if needed
pip3 install requests redis psycopg2-binary websockets docker openai anthropic google-generativeai
```

### Test Categories

1. **Docker Compose Health Checks**
   - Service container status
   - Health check endpoints
   - Service dependencies

2. **Database Connection Tests**
   - PostgreSQL connectivity
   - Redis operations
   - Database schema validation

3. **AI Model Integration Tests**
   - OpenAI o3 API connectivity
   - Anthropic Claude Sonnet 4 integration
   - Google Gemini Pro integration  
   - Ollama local model tests
   - Hugging Face model integration

4. **Frontend-Backend Communication**
   - REST API endpoints
   - WebSocket connections
   - File upload functionality
   - Authentication flow

5. **System Integration Tests**
   - End-to-end chat functionality
   - Cost tracking accuracy
   - Real-time analytics

6. **Performance & Monitoring**
   - Response time measurements
   - Resource usage monitoring
   - Prometheus metrics collection

### Test Results

```bash
# View test results
cat integration-test-results.json

# View detailed logs
cat integration-test.log
```

## 🔧 Troubleshooting

### Common Issues

#### Port Conflicts
```bash
# Check what's using ports
lsof -i :3000
lsof -i :8000
lsof -i :5432

# Stop conflicting services or change ports in docker-compose.yml
```

#### Service Won't Start
```bash
# Check service logs
docker-compose logs [service_name]

# Restart specific service
docker-compose restart [service_name]

# Rebuild service
docker-compose up -d --build [service_name]
```

#### Database Connection Issues
```bash
# Check PostgreSQL is ready
docker-compose exec postgres pg_isready -U postgres

# Check Redis connectivity
docker-compose exec redis redis-cli ping

# Reset database (WARNING: destroys data)
docker-compose down -v
docker-compose up -d postgres
```

#### Ollama Model Issues
```bash
# List installed models
curl http://localhost:11434/api/tags

# Install a model
docker-compose exec ollama ollama pull llama2

# Check Ollama logs
docker-compose logs ollama
```

#### Memory/Performance Issues
```bash
# Check resource usage
docker stats

# Adjust resource limits in docker-compose.yml
# Reduce Ollama memory if needed:
#   deploy:
#     resources:
#       limits:
#         memory: 4G  # Reduce from 8G
```

### Debugging Commands

```bash
# Full system status
docker-compose ps
docker stats --no-stream

# Service logs (last 50 lines)
docker-compose logs --tail=50 [service_name]

# Follow logs in real-time
docker-compose logs -f [service_name]

# Execute commands in containers
docker-compose exec backend bash
docker-compose exec postgres psql -U postgres -d ai_agent_system

# Network debugging
docker network ls
docker network inspect ai-agent-system_ai-agent-network
```

### Reset Procedures

#### Soft Reset (Keep Data)
```bash
docker-compose restart
```

#### Hard Reset (Lose Data) 
```bash
docker-compose down -v --remove-orphans
docker-compose up -d
```

#### Complete Cleanup
```bash
# Stop and remove everything
docker-compose down -v --remove-orphans

# Remove images
docker-compose down --rmi all

# Prune system (careful!)
docker system prune -a --volumes
```

## 📊 Monitoring & Analytics

### Prometheus Metrics
- **URL**: http://localhost:9090
- **Default queries**:
  ```promql
  up                           # Service availability
  container_memory_usage_bytes # Memory usage
  container_cpu_usage_seconds  # CPU usage
  ```

### Grafana Dashboard
- **URL**: http://localhost:3001
- **Login**: admin / admin (or GRAFANA_PASSWORD)
- **Dashboards**: Auto-imported from prometheus data

### Application Metrics
- **Backend API**: http://localhost:8000/metrics
- **System health**: http://localhost:8000/health
- **API docs**: http://localhost:8000/docs

## 🌍 Production Deployment

### Production Configuration

```bash
# Use production docker-compose
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Set production environment
ENVIRONMENT=production
DEBUG=false
```

### Security Checklist

- [ ] Change default passwords
- [ ] Use strong SECRET_KEY  
- [ ] Configure CORS_ORIGINS
- [ ] Set up SSL certificates
- [ ] Enable firewall rules
- [ ] Configure backup procedures
- [ ] Set up log rotation
- [ ] Monitor resource usage

### Performance Optimization

```yaml
# docker-compose.yml optimizations
services:
  backend:
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
      replicas: 2  # Scale backend
    
  postgres:
    command: >
      postgres
      -c shared_preload_libraries=pg_stat_statements
      -c max_connections=100
      -c shared_buffers=256MB
```

## 🔄 Development Workflow

### Development Mode

```bash
# Start with development overrides
docker-compose up -d

# Enable hot reloading
# Frontend: Vite hot reload enabled
# Backend: uvicorn --reload enabled

# Access development tools
open http://localhost:8081  # Redis Commander
open http://localhost:5050  # PgAdmin
```

### Making Changes

```bash
# Rebuild after code changes
docker-compose up -d --build [service_name]

# View real-time logs
docker-compose logs -f backend frontend

# Run tests
python3 integration-test.py
```

### Database Migrations

```bash
# Run migrations
docker-compose exec backend alembic upgrade head

# Create new migration
docker-compose exec backend alembic revision --autogenerate -m "description"
```

## 📚 API Documentation

### Backend API
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

### Key Endpoints
```
GET  /health              # System health check
POST /api/chat           # Chat with AI models
GET  /api/models         # List available models
POST /api/upload         # File upload
GET  /api/analytics      # Usage analytics
WebSocket /ws            # Real-time communication
```

## ⚡ Performance Tips

### Resource Optimization

```bash
# Reduce Ollama memory for smaller systems
# In docker-compose.yml:
ollama:
  deploy:
    resources:
      limits:
        memory: 4G  # Instead of 8G

# Limit database connections
postgres:
  command: postgres -c max_connections=50

# Use Redis for caching
redis:
  command: redis-server --maxmemory 256mb --maxmemory-policy allkeys-lru
```

### Scaling Services

```bash
# Scale backend replicas
docker-compose up -d --scale backend=3

# Use load balancer (nginx example)
nginx:
  image: nginx:alpine
  ports: ["80:80"]
  depends_on: [backend]
```

## 🆘 Support & Contributing

### Getting Help

1. **Check logs**: `./health-check.sh --verbose`
2. **Run diagnostics**: `python3 integration-test.py`
3. **View documentation**: http://localhost:8000/docs
4. **Check issues**: Review startup-verification.log

### Reporting Issues

Include in your issue report:
- OS and Docker versions
- Log files (startup-verification.log, health-check.log)
- Integration test results
- Service configuration (docker-compose.yml)

### Contributing

1. Fork the repository
2. Create feature branch
3. Test with integration suite
4. Submit pull request with test results

---

## 🎯 Success Indicators

When everything is working correctly, you should see:

✅ All services running (`docker-compose ps`)  
✅ All health checks passing (`./health-check.sh`)  
✅ Integration tests passing (`python3 integration-test.py`)  
✅ Frontend accessible at http://localhost:3000  
✅ Backend API docs at http://localhost:8000/docs  
✅ Monitoring dashboard at http://localhost:3001  
✅ AI models responding via chat API  
✅ Cost tracking displaying usage metrics  
✅ Real-time WebSocket connections working  

## 🔄 Quick Commands Reference

```bash
# Complete system startup
./startup-verification.sh

# Health check all services
./health-check.sh

# Run integration tests
python3 integration-test.py

# View service logs
docker-compose logs -f [service_name]

# Restart specific service
docker-compose restart [service_name]

# Complete system reset
docker-compose down -v && docker-compose up -d

# Scale backend for production
docker-compose up -d --scale backend=3

# Check resource usage
docker stats --no-stream
```

## 🚀 One-Command Deployment

For the ultimate quick start experience:

```bash
# Single command to get everything running
git clone <repository> && cd ai-agent-system && cp .env.example .env && ./startup-verification.sh
```

**Happy coding! 🚀**