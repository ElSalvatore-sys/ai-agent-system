# AI Agent System Backend

A comprehensive FastAPI backend with multi-AI model support, real-time chat, cost optimization, and advanced security features.

## Features

### Core AI Integration
- **Multi-Model Support**: OpenAI (GPT-4, GPT-4-Turbo, GPT-3.5-Turbo, O3-Mini), Anthropic (Claude 3 variants), Google (Gemini 1.5)
- **Intelligent Model Routing**: Automatic selection of optimal models based on cost, performance, and capabilities
- **Cost Optimization**: Real-time cost tracking and optimization with monthly budget limits
- **Streaming Responses**: WebSocket and Server-Sent Events support for real-time chat

### Database & Caching
- **Async SQLAlchemy**: High-performance async database operations with PostgreSQL/SQLite support
- **Redis Caching**: Multi-layer caching for conversations, models, users, and rate limiting
- **Database Models**: Comprehensive models for users, conversations, messages, agents, tasks, and usage tracking

### Security & Authentication
- **JWT Authentication**: Secure token-based authentication with refresh tokens
- **Role-Based Access Control**: Admin and user roles with permission management
- **Rate Limiting**: Sliding window rate limiting with Redis backend
- **Security Middleware**: Request logging, cost tracking, and security headers

### Real-Time Features
- **WebSocket Chat**: Real-time chat with streaming AI responses
- **Live Analytics**: Real-time system metrics and usage tracking
- **Background Tasks**: Async task processing and scheduling

### Admin & Analytics
- **Admin Dashboard**: Comprehensive analytics and system management
- **Usage Analytics**: Detailed cost and performance analytics
- **User Management**: User administration and activity monitoring
- **System Health**: Health checks and system monitoring

## Architecture

```
backend/
├── main.py                 # FastAPI application entry point
├── app/
│   ├── api/               # API routes
│   │   ├── routes.py      # Route registration
│   │   └── endpoints/     # Individual route modules
│   │       ├── auth.py    # Authentication endpoints
│   │       ├── chat.py    # Chat and WebSocket endpoints
│   │       ├── admin.py   # Admin analytics endpoints
│   │       ├── agents.py  # Agent management
│   │       ├── tasks.py   # Task management
│   │       └── system.py  # System info and health
│   ├── core/              # Core configuration
│   │   ├── config.py      # Application settings
│   │   └── logger.py      # Structured logging setup
│   ├── database/          # Database layer
│   │   ├── database.py    # Database connection and session
│   │   └── models.py      # SQLAlchemy models
│   ├── middleware/        # Custom middleware
│   │   ├── auth.py        # JWT authentication middleware
│   │   ├── rate_limit.py  # Rate limiting middleware
│   │   ├── cost_tracking.py # Cost tracking middleware
│   │   └── logging_middleware.py # Request logging
│   ├── models/            # AI model integration
│   │   └── ai_orchestrator.py # Multi-model orchestration
│   ├── services/          # Business logic services
│   │   └── cost_optimizer.py # Cost optimization service
│   └── utils/             # Utility modules
│       └── cache.py       # Redis caching utilities
└── requirements.txt       # Python dependencies
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Configuration

Create a `.env` file:

```env
# Environment
ENVIRONMENT=development
DEBUG=true

# Database
DATABASE_URL=sqlite+aiosqlite:///./ai_agents.db

# Security
SECRET_KEY=your-super-secret-key-change-in-production
ACCESS_TOKEN_EXPIRE_MINUTES=30

# AI Model API Keys
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
GOOGLE_API_KEY=your-google-api-key

# Redis
REDIS_URL=redis://localhost:6379

# Rate Limiting
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60

# Cost Limits
MONTHLY_COST_LIMIT=1000.0
```

### 3. Run the Application

```bash
# Development with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Or using Python directly
python main.py
```

### 4. Access the API

- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **System Info**: http://localhost:8000/api/v1/system/info

## API Endpoints

### Authentication
- `POST /api/v1/auth/register` - User registration
- `POST /api/v1/auth/login` - User login
- `POST /api/v1/auth/refresh` - Refresh access token
- `GET /api/v1/auth/profile` - Get user profile

### Chat
- `POST /api/v1/chat/send` - Send chat message
- `POST /api/v1/chat/stream` - Stream chat response
- `WS /api/v1/chat/ws` - WebSocket chat
- `GET /api/v1/chat/conversations` - Get conversations
- `GET /api/v1/chat/conversations/{id}/messages` - Get messages

### Agents
- `GET /api/v1/agents/` - List agents
- `POST /api/v1/agents/` - Create agent
- `GET /api/v1/agents/{id}` - Get agent details
- `PUT /api/v1/agents/{id}` - Update agent
- `DELETE /api/v1/agents/{id}` - Delete agent

### Tasks
- `GET /api/v1/tasks/` - List tasks
- `POST /api/v1/tasks/` - Create task
- `GET /api/v1/tasks/{id}` - Get task details
- `PUT /api/v1/tasks/{id}` - Update task
- `DELETE /api/v1/tasks/{id}` - Delete task

### Admin (Requires admin role)
- `GET /api/v1/admin/overview` - System overview
- `GET /api/v1/admin/users` - User management
- `GET /api/v1/admin/usage-analytics` - Usage analytics
- `GET /api/v1/admin/cost-optimization` - Cost insights

## Database Models

### Core Models
- **User**: User accounts with roles and preferences
- **Conversation**: Chat conversations with AI configuration
- **Message**: Individual messages with metadata and costs
- **Agent**: AI agents with specific configurations
- **Task**: Background tasks and scheduling

### Analytics Models
- **UsageLog**: Detailed API usage tracking
- **SystemMetrics**: System performance metrics
- **AIModel**: Available AI models and pricing

## AI Model Integration

### Supported Providers
- **OpenAI**: GPT-4, GPT-4-Turbo, GPT-3.5-Turbo, O3-Mini
- **Anthropic**: Claude 3 Opus, Sonnet, Haiku
- **Google**: Gemini 1.5 Pro, Flash

### Model Selection
The AI Orchestrator automatically selects optimal models based on:
- **Cost efficiency**: Minimize costs while maintaining quality
- **Performance requirements**: Balance speed vs. capability
- **User preferences**: Respect user model preferences
- **Capability matching**: Select models that support required features

### Cost Optimization
- Real-time cost tracking per request
- Monthly budget limits and alerts
- Model performance analytics
- Usage pattern analysis
- Cost prediction and recommendations

## Security Features

### Authentication & Authorization
- JWT tokens with refresh capability
- Role-based access control (User, Admin)
- Secure password hashing with bcrypt
- Token blacklisting support

### Rate Limiting
- Sliding window algorithm
- Per-user and per-IP limiting
- Configurable limits and windows
- Redis-backed state management

### Data Protection
- Request/response logging with sensitive data filtering
- Secure headers and CORS configuration
- Input validation and sanitization
- SQL injection protection

## Monitoring & Observability

### Structured Logging
- JSON-structured logs with context
- Multiple log levels and outputs
- Request tracing with correlation IDs
- Performance and error tracking

### Health Checks
- Database connectivity
- Redis availability
- AI model health
- System resource monitoring

### Metrics
- Request rates and response times
- AI model usage and costs
- User activity patterns
- System resource utilization

## Development

### Code Structure
- Clean architecture with separation of concerns
- Async/await throughout for high performance
- Type hints and Pydantic validation
- Comprehensive error handling

### Testing
```bash
# Run tests
pytest

# Run with coverage
pytest --cov=app tests/
```

### Database Migrations
```bash
# Generate migration
alembic revision --autogenerate -m "Description"

# Apply migrations
alembic upgrade head
```

## Deployment

### Docker
```bash
# Build image
docker build -t ai-agent-backend .

# Run container
docker run -p 8000:8000 ai-agent-backend
```

### Environment Variables
See `.env.example` for all configuration options.

### Production Considerations
- Use PostgreSQL for production database
- Configure Redis cluster for high availability
- Set up proper logging aggregation
- Configure monitoring and alerting
- Use HTTPS with proper certificates
- Set up database backups
- Configure auto-scaling policies

## Contributing

1. Follow the existing code style and patterns
2. Add comprehensive tests for new features
3. Update documentation for API changes
4. Use semantic commit messages
5. Ensure all security checks pass

## License

This project is proprietary software. All rights reserved.