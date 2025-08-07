#!/bin/bash

echo "🔍 FRONTEND RENDERING DIAGNOSTIC"
echo "=================================="

echo ""
echo "1. Testing root HTML structure..."
curl -s http://localhost:3000 | grep -A5 -B5 "root"

echo ""
echo "2. Testing for JavaScript errors in console..."
curl -s http://localhost:3000 | grep -i "error\|Error\|ERROR"

echo ""
echo "3. Checking if React is loading..."
curl -s http://localhost:3000 | grep -o "React" | head -5

echo ""
echo "4. Testing Vite HMR connection..."
curl -s -I http://localhost:3000/@vite/client

echo ""
echo "5. Checking if main.tsx is accessible..."
curl -s -I "http://localhost:3000/src/main.tsx"

echo ""
echo "6. Testing theme context loading..."
curl -s "http://localhost:3000/src/context/providers/ThemeContext.tsx" | head -10

echo ""
echo "7. Checking App.tsx loading..."
curl -s -I "http://localhost:3000/src/App.tsx"

echo ""
echo "8. Testing if CSS variables are loaded..."
curl -s http://localhost:3000 | grep -o ":root" | head -3

echo ""
echo "9. Container logs (recent)..."
docker-compose logs --tail=10 frontend

echo ""
echo "=================================="
echo "🔍 Diagnostic complete"