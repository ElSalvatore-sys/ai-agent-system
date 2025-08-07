-- AI Agent System Database Initialization
-- This script is automatically executed by PostgreSQL when the container starts

-- The database is already created by POSTGRES_DB environment variable
-- So we just need to create the schema and initial data

BEGIN;

-- Create users table
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) DEFAULT 'user',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE,
    monthly_budget DECIMAL(10,2) DEFAULT 100.00,
    current_spend DECIMAL(10,2) DEFAULT 0.00
);

-- Create conversations table
CREATE TABLE conversations (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    model VARCHAR(50) NOT NULL DEFAULT 'gpt-3.5-turbo',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE,
    total_cost DECIMAL(10,4) DEFAULT 0.00
);

-- Create messages table
CREATE TABLE messages (
    id SERIAL PRIMARY KEY,
    conversation_id INTEGER REFERENCES conversations(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    tokens_used INTEGER DEFAULT 0,
    cost DECIMAL(10,4) DEFAULT 0.00,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create agents table
CREATE TABLE agents (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    type VARCHAR(50) DEFAULT 'chat',
    model VARCHAR(50) DEFAULT 'gpt-3.5-turbo',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    total_runs INTEGER DEFAULT 0,
    success_rate DECIMAL(5,2) DEFAULT 100.00
);

-- Create basic indexes
CREATE INDEX idx_conversations_user_id ON conversations(user_id);
CREATE INDEX idx_messages_conversation_id ON messages(conversation_id);
CREATE INDEX idx_agents_user_id ON agents(user_id);

-- Insert default admin user (password: 'admin123')
INSERT INTO users (username, email, password_hash, role, monthly_budget) VALUES
('admin', 'admin@aiagent.local', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewKyNiRU6wt.EZaW', 'admin', 1000.00);

-- Insert demo data
INSERT INTO conversations (user_id, title, model) VALUES
(1, 'Welcome to AI Agent System', 'gpt-3.5-turbo');

INSERT INTO messages (conversation_id, role, content, tokens_used, cost) VALUES
(1, 'assistant', 'Welcome to the AI Agent System! How can I help you today?', 15, 0.00003);

INSERT INTO agents (user_id, name, description, type, model, total_runs, success_rate) VALUES
(1, 'Code Assistant', 'Helps with coding tasks and debugging', 'chat', 'claude-3.5-sonnet', 247, 96.8),
(1, 'Data Analyzer', 'Analyzes datasets and generates insights', 'analysis', 'gpt-4', 89, 94.4),
(1, 'Task Scheduler', 'Manages automated tasks', 'task', 'gemini-pro', 156, 91.7);

COMMIT;

-- Success notification
SELECT 'AI Agent System database initialized successfully!' AS status;