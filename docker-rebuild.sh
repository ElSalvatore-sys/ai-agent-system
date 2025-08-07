#!/bin/bash

echo "========================================"
echo "   AI Agent System - Docker Rebuild"
echo "========================================"
echo

echo "🔄 Stopping existing containers..."
docker-compose down

echo "🧹 Cleaning up Docker cache..."
docker system prune -f

echo "🔨 Rebuilding frontend container with all dependencies..."
docker-compose build --no-cache frontend

echo "🔨 Rebuilding backend container..."
docker-compose build --no-cache backend

echo "🚀 Starting all services..."
docker-compose up -d

echo
echo "⏳ Waiting for services to start..."
sleep 30

echo
echo "========================================"
echo "   System Status Check"
echo "========================================"

echo "Checking Backend..."
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Backend: RUNNING"
else
    echo "❌ Backend: NOT RESPONDING"
fi

echo "Checking Frontend..."
if curl -f http://localhost:8080 > /dev/null 2>&1; then
    echo "✅ Frontend: RUNNING"
else
    echo "❌ Frontend: NOT RESPONDING"
fi

echo "Checking PostgreSQL..."
if docker-compose exec -T postgres pg_isready -U postgres > /dev/null 2>&1; then
    echo "✅ PostgreSQL: RUNNING"
else
    echo "❌ PostgreSQL: NOT RESPONDING"
fi

echo "Checking Redis..."
if docker-compose exec -T redis redis-cli ping > /dev/null 2>&1; then
    echo "✅ Redis: RUNNING"
else
    echo "❌ Redis: NOT RESPONDING"
fi

echo
echo "========================================"
echo "   Access URLs"
echo "========================================"
echo "🌐 Frontend Application: http://localhost:8080"
echo "📊 Backend API: http://localhost:8000"
echo "📚 API Documentation: http://localhost:8000/docs"
echo "📈 Prometheus: http://localhost:9090"
echo "📊 Grafana: http://localhost:3001"
echo

echo "========================================"
echo "   Advanced Features Available"
echo "========================================"
echo "🎯 Enhanced Dashboard: http://localhost:8080/"
echo "💬 Multi-Model Chat: http://localhost:8080/enhanced-chat"
echo "📈 Monitoring: http://localhost:8080/monitoring"
echo "🎨 Component Showcase: http://localhost:8080/showcase"
echo "⚙️ Settings: http://localhost:8080/settings"
echo

echo "🚀 Your advanced AI Agent System is ready!"
echo
echo "💡 To view logs:"
echo "   docker-compose logs -f frontend"
echo "   docker-compose logs -f backend"
echo
echo "💡 To stop:"
echo "   docker-compose down" 