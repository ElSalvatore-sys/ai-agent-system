-- Initialize AI Agent System Database
-- This script creates the database and initial schema

-- Create database (if it doesn't exist - PostgreSQL Docker image handles this via POSTGRES_DB)
CREATE DATABASE IF NOT EXISTS ai_agent_system;

-- Connect to the database
\c ai_agent_system;

-- Create users table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) DEFAULT 'user',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE,
    monthly_budget DECIMAL(10,2) DEFAULT 100.00,
    current_spend DECIMAL(10,2) DEFAULT 0.00,
    preferred_model VARCHAR(50) DEFAULT 'gpt-3.5-turbo',
    timezone VARCHAR(50) DEFAULT 'UTC'
);

-- Create conversations table
CREATE TABLE IF NOT EXISTS conversations (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    model VARCHAR(50) NOT NULL DEFAULT 'gpt-3.5-turbo',
    system_prompt TEXT,
    temperature DECIMAL(3,2) DEFAULT 0.7,
    max_tokens INTEGER DEFAULT 1000,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE,
    total_cost DECIMAL(10,4) DEFAULT 0.00,
    message_count INTEGER DEFAULT 0
);

-- Create messages table
CREATE TABLE IF NOT EXISTS messages (
    id SERIAL PRIMARY KEY,
    conversation_id INTEGER REFERENCES conversations(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    tokens_used INTEGER DEFAULT 0,
    cost DECIMAL(10,4) DEFAULT 0.00,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- Create ai_models table
CREATE TABLE IF NOT EXISTS ai_models (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    provider VARCHAR(50) NOT NULL,
    input_cost_per_token DECIMAL(12,8) NOT NULL DEFAULT 0.00000000,
    output_cost_per_token DECIMAL(12,8) NOT NULL DEFAULT 0.00000000,
    max_tokens INTEGER DEFAULT 4000,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    description TEXT,
    capabilities JSONB DEFAULT '[]'
);

-- Create cost_tracking table
CREATE TABLE IF NOT EXISTS cost_tracking (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    conversation_id INTEGER REFERENCES conversations(id) ON DELETE CASCADE,
    model VARCHAR(50) NOT NULL,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    total_cost DECIMAL(10,4) NOT NULL,
    date DATE DEFAULT CURRENT_DATE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- Create agents table
CREATE TABLE IF NOT EXISTS agents (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    type VARCHAR(50) DEFAULT 'chat',
    model VARCHAR(50) DEFAULT 'gpt-3.5-turbo',
    system_prompt TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    config JSONB DEFAULT '{}',
    total_runs INTEGER DEFAULT 0,
    success_rate DECIMAL(5,2) DEFAULT 100.00
);

-- Create tasks table
CREATE TABLE IF NOT EXISTS tasks (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    agent_id INTEGER REFERENCES agents(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    status VARCHAR(20) DEFAULT 'pending',
    priority INTEGER DEFAULT 3,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    scheduled_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    result JSONB DEFAULT '{}'
);

-- Create system_metrics table
CREATE TABLE IF NOT EXISTS system_metrics (
    id SERIAL PRIMARY KEY,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(15,4) NOT NULL,
    metric_type VARCHAR(50) DEFAULT 'gauge',
    tags JSONB DEFAULT '{}',
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create usage_logs table
CREATE TABLE IF NOT EXISTS usage_logs (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    action VARCHAR(100) NOT NULL,
    resource VARCHAR(100),
    details JSONB DEFAULT '{}',
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    session_id VARCHAR(255)
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id);
CREATE INDEX IF NOT EXISTS idx_conversations_created_at ON conversations(created_at);
CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at);
CREATE INDEX IF NOT EXISTS idx_cost_tracking_user_id ON cost_tracking(user_id);
CREATE INDEX IF NOT EXISTS idx_cost_tracking_date ON cost_tracking(date);
CREATE INDEX IF NOT EXISTS idx_agents_user_id ON agents(user_id);
CREATE INDEX IF NOT EXISTS idx_tasks_user_id ON tasks(user_id);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_system_metrics_name_timestamp ON system_metrics(metric_name, timestamp);
CREATE INDEX IF NOT EXISTS idx_usage_logs_user_id_created_at ON usage_logs(user_id, created_at);

-- Insert default AI models
INSERT INTO ai_models (name, provider, input_cost_per_token, output_cost_per_token, max_tokens, description) VALUES
('gpt-3.5-turbo', 'openai', 0.0000015, 0.000002, 4000, 'Fast and efficient model for most tasks'),
('gpt-4', 'openai', 0.00003, 0.00006, 8000, 'Most capable GPT-4 model'),
('claude-3.5-sonnet', 'anthropic', 0.000003, 0.000015, 200000, 'Anthropic''s most intelligent model'),
('claude-3-haiku', 'anthropic', 0.00000025, 0.00000125, 200000, 'Anthropic''s fastest model'),
('gemini-pro', 'google', 0.0000005, 0.0000015, 30720, 'Google''s advanced multimodal model'),
('llama-2-70b', 'meta', 0.00000065, 0.00000275, 4096, 'Meta''s open-source large language model')
ON CONFLICT (name) DO NOTHING;

-- Insert default admin user (password: 'admin123' - change in production!)
INSERT INTO users (username, email, password_hash, role, monthly_budget) VALUES
('admin', 'admin@aiagent.local', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewKyNiRU6wt.EZaW', 'admin', 1000.00)
ON CONFLICT (username) DO NOTHING;

-- Create a sample conversation for demo purposes
INSERT INTO conversations (user_id, title, model, system_prompt) 
SELECT id, 'Welcome to AI Agent System', 'gpt-3.5-turbo', 'You are a helpful AI assistant in the AI Agent System.' 
FROM users WHERE username = 'admin' 
ON CONFLICT DO NOTHING;

-- Insert sample message
INSERT INTO messages (conversation_id, role, content, tokens_used, cost)
SELECT c.id, 'assistant', 'Welcome to the AI Agent System! I''m here to help you with your AI-powered tasks. You can ask me questions, request code generation, data analysis, and much more.', 25, 0.00005
FROM conversations c 
JOIN users u ON c.user_id = u.id 
WHERE u.username = 'admin' AND c.title = 'Welcome to AI Agent System'
ON CONFLICT DO NOTHING;

-- Create sample agent
INSERT INTO agents (user_id, name, description, type, model, system_prompt, total_runs, success_rate)
SELECT id, 'Code Assistant', 'Helps with coding tasks, debugging, and code reviews', 'chat', 'claude-3.5-sonnet', 'You are an expert software developer assistant.', 247, 96.8
FROM users WHERE username = 'admin'
ON CONFLICT DO NOTHING;

-- Initialize system metrics
INSERT INTO system_metrics (metric_name, metric_value, metric_type) VALUES
('total_users', 1, 'gauge'),
('total_conversations', 1, 'gauge'),
('total_messages', 1, 'gauge'),
('total_agents', 1, 'gauge'),
('system_uptime', 0, 'gauge'),
('api_requests_total', 0, 'counter'),
('active_sessions', 0, 'gauge')
ON CONFLICT DO NOTHING;

-- Grant permissions (if using specific database user)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO ai_agent_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO ai_agent_user;

-- Create trigger to update 'updated_at' column automatically
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply the trigger to relevant tables
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_conversations_updated_at BEFORE UPDATE ON conversations FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_ai_models_updated_at BEFORE UPDATE ON ai_models FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_agents_updated_at BEFORE UPDATE ON agents FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_tasks_updated_at BEFORE UPDATE ON tasks FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Success message
DO $$
BEGIN
    RAISE NOTICE 'AI Agent System database initialized successfully!';
    RAISE NOTICE 'Default admin user created: username="admin", password="admin123"';
    RAISE NOTICE 'Please change the default password in production!';
END $$;