# AI Agent System

A modern AI agent system built with React TypeScript frontend and FastAPI Python backend.

## Project Structure

```
ai-agent-system/
├── frontend/          # React TypeScript frontend
├── backend/           # FastAPI Python backend
├── docker/           # Docker configuration
├── scripts/          # Development and deployment scripts
├── docker-compose.yml
└── README.md
```

## Quick Start

### Local Development (Recommended)
```bash
# Install dependencies
npm install
npm run install

# Start both frontend and backend
npm run dev

# Access the application
# Frontend: http://localhost:7000
# Backend API: http://localhost:8000/docs
```

### Individual Services
```bash
# Frontend only (port 7000)
npm run dev:frontend

# Backend only (port 8000)  
npm run dev:backend

# Build for production
npm run build

# Preview production build
npm run preview
```

### Docker Development
```bash
npm run dev:docker
```

### Production
```bash
npm run prod
```

## Features

- **Modern React 18** with TypeScript and Vite
- **FastAPI Python backend** with async support  
- **TailwindCSS** with dark theme and glassmorphism design
- **React Query** for server state management
- **Socket.IO** for real-time communication
- **Docker** containerization for deployment
- **Hot reloading** in development
- **Production-ready** configuration