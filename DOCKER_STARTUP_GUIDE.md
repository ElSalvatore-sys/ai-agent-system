# 🐳 AI Agent System - Docker Startup Guide

## Overview

This guide explains how to run your advanced AI Agent System using Docker Compose, which includes the frontend, backend, database, cache, and monitoring services.

## 🚀 Quick Start

### Prerequisites
- Docker Desktop installed and running
- At least 4GB of available RAM
- Git

### 1. Clone and Navigate
```bash
git clone <your-repo-url>
cd ai-agent-system
```

### 2. Set Environment Variables (Optional)
Create a `.env` file in the root directory:
```env
# Database
POSTGRES_DB=ai_agent_system
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_secure_password

# Redis
REDIS_PASSWORD=your_redis_password

# API Keys (optional for basic functionality)
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key

# Security
SECRET_KEY=your_secret_key

# Monitoring
GRAFANA_PASSWORD=admin
```

### 3. Start the System

#### Option A: Using the Rebuild Script (Recommended)
```bash
# Windows
docker-rebuild.bat

# Linux/Mac
chmod +x docker-rebuild.sh
./docker-rebuild.sh
```

#### Option B: Manual Commands
```bash
# Stop any existing containers
docker-compose down

# Rebuild containers (fixes dependency issues)
docker-compose build --no-cache

# Start all services
docker-compose up -d

# Check status
docker-compose ps
```

## 📊 System Architecture

Your Docker setup includes:

### Core Services
- **Frontend** (React + TypeScript) - Port 8080
- **Backend** (FastAPI + Python) - Port 8000
- **PostgreSQL** (Database) - Port 5432
- **Redis** (Cache) - Port 6379

### Monitoring Services
- **Prometheus** (Metrics) - Port 9090
- **Grafana** (Dashboards) - Port 3001
- **Redis Commander** (Redis UI) - Port 8081
- **pgAdmin** (Database UI) - Port 5050

### Optional Services
- **Ollama** (Local LLM) - Port 11434 (commented out)

## 🌐 Access URLs

Once the system is running, you can access:

### Main Application
- **Frontend**: http://localhost:8080
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

### Advanced Features
- **Enhanced Dashboard**: http://localhost:8080/
- **Multi-Model Chat**: http://localhost:8080/enhanced-chat
- **Monitoring**: http://localhost:8080/monitoring
- **Component Showcase**: http://localhost:8080/showcase
- **Settings**: http://localhost:8080/settings

### Monitoring & Admin
- **Grafana**: http://localhost:3001 (admin/admin)
- **Prometheus**: http://localhost:9090
- **Redis Commander**: http://localhost:8081
- **pgAdmin**: http://localhost:5050

## 🔧 Common Commands

### Start/Stop Services
```bash
# Start all services
docker-compose up -d

# Stop all services
docker-compose down

# Restart specific service
docker-compose restart frontend

# View logs
docker-compose logs -f frontend
docker-compose logs -f backend
```

### Troubleshooting
```bash
# Check service status
docker-compose ps

# Rebuild specific service
docker-compose build --no-cache frontend

# Clean up Docker cache
docker system prune -f

# View resource usage
docker stats
```

### Development
```bash
# Access container shell
docker-compose exec frontend sh
docker-compose exec backend bash

# Run commands inside container
docker-compose exec frontend npm install new-package
docker-compose exec backend python -m pip install new-package
```

## 🛠️ Troubleshooting

### Common Issues

#### 1. Port Already in Use
```bash
# Check what's using the port
netstat -ano | findstr :8080

# Kill the process or change ports in docker-compose.yml
```

#### 2. Memory Issues
```bash
# Check Docker memory allocation
docker stats

# Increase Docker Desktop memory limit
# Docker Desktop > Settings > Resources > Memory
```

#### 3. Dependency Issues (like recharts)
```bash
# Rebuild frontend container
docker-compose build --no-cache frontend
docker-compose up -d frontend
```

#### 4. Database Connection Issues
```bash
# Check database logs
docker-compose logs postgres

# Reset database (WARNING: loses data)
docker-compose down -v
docker-compose up -d
```

### Performance Optimization

#### Resource Limits
The docker-compose.yml includes resource limits:
- Frontend: 1GB RAM, 0.5 CPU
- Backend: 2GB RAM, 1.0 CPU
- PostgreSQL: 1GB RAM, 0.5 CPU
- Redis: 512MB RAM, 0.25 CPU

#### Scaling
```bash
# Scale backend for more performance
docker-compose up -d --scale backend=2
```

## 🔒 Security Considerations

### Production Deployment
1. **Change default passwords** in `.env` file
2. **Use HTTPS** with reverse proxy
3. **Restrict network access** to monitoring ports
4. **Enable authentication** for admin interfaces
5. **Regular security updates**

### Environment Variables
```bash
# Required for production
SECRET_KEY=your_very_secure_secret_key
POSTGRES_PASSWORD=your_secure_db_password
REDIS_PASSWORD=your_secure_redis_password
```

## 📈 Monitoring & Analytics

### Built-in Monitoring
- **Prometheus**: Collects metrics from all services
- **Grafana**: Visualizes metrics and creates dashboards
- **Health Checks**: Automatic service health monitoring

### Custom Dashboards
Access Grafana at http://localhost:3001 to create custom dashboards for:
- Application performance
- Database metrics
- System resources
- Business metrics

## 🚀 Advanced Features

### Local LLM Integration
To enable Ollama (local AI models):
1. Uncomment the ollama service in docker-compose.yml
2. Restart: `docker-compose up -d`
3. Access: http://localhost:11434

### Custom Configurations
Modify `docker-compose.yml` to:
- Add new services
- Change resource limits
- Modify environment variables
- Add volume mounts

## 📝 Logs and Debugging

### View Logs
```bash
# All services
docker-compose logs

# Specific service
docker-compose logs frontend

# Follow logs in real-time
docker-compose logs -f backend

# Last 100 lines
docker-compose logs --tail=100 frontend
```

### Debug Mode
```bash
# Enable debug logging
docker-compose up -d --env-file .env.debug
```

## 🔄 Updates and Maintenance

### Updating the System
```bash
# Pull latest changes
git pull

# Rebuild containers
docker-compose build --no-cache

# Restart services
docker-compose up -d
```

### Backup and Restore
```bash
# Backup database
docker-compose exec postgres pg_dump -U postgres ai_agent_system > backup.sql

# Restore database
docker-compose exec -T postgres psql -U postgres ai_agent_system < backup.sql
```

## 🆘 Support

### Getting Help
1. Check the logs: `docker-compose logs`
2. Verify service status: `docker-compose ps`
3. Check resource usage: `docker stats`
4. Review this documentation
5. Check the main README.md

### Emergency Commands
```bash
# Stop everything
docker-compose down

# Remove all data (WARNING: destructive)
docker-compose down -v
docker system prune -a

# Restart Docker Desktop
# Then run: docker-compose up -d
```

---

**🚀 Your advanced AI Agent System is now running with Docker!**

For more information, see the main `ADVANCED_SYSTEM_README.md` file. 