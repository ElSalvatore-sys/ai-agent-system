#!/bin/bash

# =======================================================
# FRONTEND FIX SCRIPT
# =======================================================
# Quick script to restart frontend with proper configuration
# =======================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔧 FIXING FRONTEND CONFIGURATION...${NC}"
echo "======================================================="

echo -e "${YELLOW}Step 1:${NC} Stopping frontend container..."
docker-compose stop frontend

echo -e "${YELLOW}Step 2:${NC} Rebuilding frontend with new configuration..."
docker-compose build --no-cache frontend

echo -e "${YELLOW}Step 3:${NC} Starting frontend with correct port mapping..."
docker-compose up -d frontend

echo -e "${YELLOW}Step 4:${NC} Waiting for frontend to start..."
sleep 10

echo -e "${YELLOW}Step 5:${NC} Checking frontend status..."
docker-compose ps frontend

echo -e "${YELLOW}Step 6:${NC} Testing connectivity..."
if curl -s -o /dev/null -w "%{http_code}" --connect-timeout 10 "http://localhost:3000" >/dev/null 2>&1; then
    echo -e "${GREEN}✅ Frontend is accessible at http://localhost:3000${NC}"
else
    echo -e "${RED}❌ Frontend is not yet accessible${NC}"
    echo -e "${BLUE}Checking logs for issues...${NC}"
    docker-compose logs --tail=20 frontend
fi

echo ""
echo "======================================================="
echo -e "${BLUE}Frontend fix completed!${NC}"
echo -e "${GREEN}🌐 Access URLs:${NC}"
echo "   Frontend:     http://localhost:3000"
echo "   Backend API:  http://localhost:8000/docs"
echo ""
echo -e "${BLUE}📋 Troubleshooting:${NC}"
echo "   View logs:    docker-compose logs -f frontend"
echo "   Diagnose:     ./diagnose-frontend.sh"
echo "   Restart:      docker-compose restart frontend"
echo "======================================================="