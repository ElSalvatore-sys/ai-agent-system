"""
Comprehensive Database Schema Migration
Creates all tables for the AI Agent System with proper relationships and indexes

Revision ID: 001_comprehensive_schema
Revises: 
Create Date: 2024-01-01 12:00:00.000000
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '001_comprehensive_schema'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    """Create comprehensive database schema"""
    
    # Create enums
    user_role_enum = sa.Enum('admin', 'user', 'guest', name='userrole')
    message_role_enum = sa.Enum('user', 'assistant', 'system', name='messagerole')
    conversation_status_enum = sa.Enum('active', 'archived', 'deleted', name='conversationstatus')
    model_provider_enum = sa.Enum('openai', 'anthropic', 'google', name='modelprovider')
    task_status_enum = sa.Enum('pending', 'running', 'completed', 'failed', 'cancelled', name='taskstatus')
    request_status_enum = sa.Enum('pending', 'processing', 'completed', 'failed', 'cancelled', name='requeststatus')
    content_type_enum = sa.Enum('code', 'document', 'image', 'text', 'json', 'xml', 'html', 'csv', 'markdown', name='contenttype')
    budget_period_enum = sa.Enum('hourly', 'daily', 'weekly', 'monthly', 'yearly', name='budgetperiod')
    alert_type_enum = sa.Enum('budget_warning', 'budget_exceeded', 'performance_issue', 'system_error', 'security_alert', name='alerttype')
    
    # Users table
    op.create_table('users',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('uuid', sa.String(length=36), nullable=True),
        sa.Column('username', sa.String(length=50), nullable=False),
        sa.Column('email', sa.String(length=100), nullable=False),
        sa.Column('hashed_password', sa.String(length=255), nullable=False),
        sa.Column('full_name', sa.String(length=100), nullable=True),
        sa.Column('role', user_role_enum, nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('is_verified', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_login', sa.DateTime(timezone=True), nullable=True),
        sa.Column('preferences', sa.JSON(), nullable=True),
        sa.Column('api_quota_limit', sa.Integer(), nullable=True),
        sa.Column('api_quota_used', sa.Integer(), nullable=True),
        sa.Column('monthly_budget_limit', sa.Float(), nullable=True),
        sa.Column('current_month_spend', sa.Float(), nullable=True),
        sa.Column('budget_alerts_enabled', sa.Boolean(), nullable=True),
        sa.Column('budget_warning_threshold', sa.Float(), nullable=True),
        sa.Column('preferred_model_provider', model_provider_enum, nullable=True),
        sa.Column('preferred_model', sa.String(length=50), nullable=True),
        sa.Column('default_temperature', sa.Float(), nullable=True),
        sa.Column('timezone', sa.String(length=50), nullable=True),
        sa.Column('failed_login_attempts', sa.Integer(), nullable=True),
        sa.Column('locked_until', sa.DateTime(timezone=True), nullable=True),
        sa.Column('password_changed_at', sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint('monthly_budget_limit >= 0', name='check_positive_budget'),
        sa.CheckConstraint('budget_warning_threshold >= 0 AND budget_warning_threshold <= 1', name='check_valid_warning_threshold'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_users_email_active', 'users', ['email', 'is_active'], unique=False)
    op.create_index('ix_users_created_at', 'users', ['created_at'], unique=False)
    op.create_index(op.f('ix_users_id'), 'users', ['id'], unique=False)
    op.create_index(op.f('ix_users_uuid'), 'users', ['uuid'], unique=True)
    op.create_index(op.f('ix_users_username'), 'users', ['username'], unique=True)
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)

    # Conversations table
    op.create_table('conversations',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('uuid', sa.String(length=36), nullable=True),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('title', sa.String(length=200), nullable=False),
        sa.Column('status', conversation_status_enum, nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('model_provider', model_provider_enum, nullable=True),
        sa.Column('model_name', sa.String(length=50), nullable=True),
        sa.Column('temperature', sa.Float(), nullable=True),
        sa.Column('max_tokens', sa.Integer(), nullable=True),
        sa.Column('system_prompt', sa.Text(), nullable=True),
        sa.Column('total_messages', sa.Integer(), nullable=True),
        sa.Column('total_tokens', sa.BigInteger(), nullable=True),
        sa.Column('total_cost', sa.Float(), nullable=True),
        sa.Column('average_response_time', sa.Float(), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.Column('tags', sa.JSON(), nullable=True),
        sa.CheckConstraint('total_cost >= 0', name='check_positive_cost'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_conversations_user_status', 'conversations', ['user_id', 'status'], unique=False)
    op.create_index('ix_conversations_updated', 'conversations', ['updated_at'], unique=False)
    op.create_index('ix_conversations_cost', 'conversations', ['total_cost'], unique=False)
    op.create_index(op.f('ix_conversations_id'), 'conversations', ['id'], unique=False)
    op.create_index(op.f('ix_conversations_uuid'), 'conversations', ['uuid'], unique=True)

    # Messages table
    op.create_table('messages',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('uuid', sa.String(length=36), nullable=True),
        sa.Column('conversation_id', sa.Integer(), nullable=False),
        sa.Column('role', message_role_enum, nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('model_used', sa.String(length=50), nullable=True),
        sa.Column('tokens_used', sa.Integer(), nullable=True),
        sa.Column('cost', sa.Float(), nullable=True),
        sa.Column('response_time', sa.Float(), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(['conversation_id'], ['conversations.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_messages_conversation_created', 'messages', ['conversation_id', 'created_at'], unique=False)
    op.create_index(op.f('ix_messages_id'), 'messages', ['id'], unique=False)
    op.create_index(op.f('ix_messages_uuid'), 'messages', ['uuid'], unique=True)

    # AI Requests table
    op.create_table('ai_requests',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('uuid', sa.String(length=36), nullable=True),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('conversation_id', sa.Integer(), nullable=True),
        sa.Column('message_id', sa.Integer(), nullable=True),
        sa.Column('request_id', sa.String(length=100), nullable=True),
        sa.Column('model_provider', model_provider_enum, nullable=False),
        sa.Column('model_name', sa.String(length=100), nullable=False),
        sa.Column('model_version', sa.String(length=50), nullable=True),
        sa.Column('temperature', sa.Float(), nullable=True),
        sa.Column('max_tokens', sa.Integer(), nullable=True),
        sa.Column('top_p', sa.Float(), nullable=True),
        sa.Column('frequency_penalty', sa.Float(), nullable=True),
        sa.Column('presence_penalty', sa.Float(), nullable=True),
        sa.Column('system_prompt', sa.Text(), nullable=True),
        sa.Column('input_text', sa.Text(), nullable=True),
        sa.Column('output_text', sa.Text(), nullable=True),
        sa.Column('input_tokens', sa.Integer(), nullable=True),
        sa.Column('output_tokens', sa.Integer(), nullable=True),
        sa.Column('total_tokens', sa.Integer(), nullable=True),
        sa.Column('request_start_time', sa.DateTime(timezone=True), nullable=False),
        sa.Column('request_end_time', sa.DateTime(timezone=True), nullable=True),
        sa.Column('response_time_ms', sa.Integer(), nullable=True),
        sa.Column('time_to_first_token_ms', sa.Integer(), nullable=True),
        sa.Column('tokens_per_second', sa.Float(), nullable=True),
        sa.Column('input_cost_usd', sa.Float(), nullable=True),
        sa.Column('output_cost_usd', sa.Float(), nullable=True),
        sa.Column('total_cost_usd', sa.Float(), nullable=True),
        sa.Column('cost_per_token', sa.Float(), nullable=True),
        sa.Column('status', request_status_enum, nullable=False),
        sa.Column('success', sa.Boolean(), nullable=True),
        sa.Column('error_code', sa.String(length=50), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('retry_count', sa.Integer(), nullable=True),
        sa.Column('confidence_score', sa.Float(), nullable=True),
        sa.Column('safety_score', sa.Float(), nullable=True),
        sa.Column('relevance_score', sa.Float(), nullable=True),
        sa.Column('user_rating', sa.Integer(), nullable=True),
        sa.Column('user_feedback', sa.Text(), nullable=True),
        sa.Column('api_endpoint', sa.String(length=200), nullable=True),
        sa.Column('user_agent', sa.String(length=500), nullable=True),
        sa.Column('ip_address', sa.String(length=45), nullable=True),
        sa.Column('request_headers', sa.JSON(), nullable=True),
        sa.Column('response_headers', sa.JSON(), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.Column('tags', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint('total_cost_usd >= 0', name='check_positive_ai_request_cost'),
        sa.CheckConstraint('response_time_ms >= 0', name='check_positive_response_time'),
        sa.ForeignKeyConstraint(['conversation_id'], ['conversations.id'], ),
        sa.ForeignKeyConstraint(['message_id'], ['messages.id'], ),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_ai_requests_user_created', 'ai_requests', ['user_id', 'created_at'], unique=False)
    op.create_index('ix_ai_requests_model_created', 'ai_requests', ['model_provider', 'model_name', 'created_at'], unique=False)
    op.create_index('ix_ai_requests_status_created', 'ai_requests', ['status', 'created_at'], unique=False)
    op.create_index('ix_ai_requests_cost', 'ai_requests', ['total_cost_usd'], unique=False)
    op.create_index('ix_ai_requests_performance', 'ai_requests', ['response_time_ms'], unique=False)
    op.create_index(op.f('ix_ai_requests_id'), 'ai_requests', ['id'], unique=False)
    op.create_index(op.f('ix_ai_requests_uuid'), 'ai_requests', ['uuid'], unique=True)
    op.create_index(op.f('ix_ai_requests_request_id'), 'ai_requests', ['request_id'], unique=True)

    # Generated Content table
    op.create_table('generated_content',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('uuid', sa.String(length=36), nullable=True),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('ai_request_id', sa.Integer(), nullable=True),
        sa.Column('conversation_id', sa.Integer(), nullable=True),
        sa.Column('title', sa.String(length=200), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('content_type', content_type_enum, nullable=False),
        sa.Column('file_extension', sa.String(length=10), nullable=True),
        sa.Column('mime_type', sa.String(length=100), nullable=True),
        sa.Column('content_text', sa.Text(), nullable=True),
        sa.Column('content_binary', sa.Text(), nullable=True),
        sa.Column('file_path', sa.String(length=500), nullable=True),
        sa.Column('file_size_bytes', sa.BigInteger(), nullable=True),
        sa.Column('language', sa.String(length=50), nullable=True),
        sa.Column('framework', sa.String(length=100), nullable=True),
        sa.Column('version', sa.String(length=50), nullable=True),
        sa.Column('syntax_valid', sa.Boolean(), nullable=True),
        sa.Column('executable', sa.Boolean(), nullable=True),
        sa.Column('compilation_status', sa.String(length=50), nullable=True),
        sa.Column('test_results', sa.JSON(), nullable=True),
        sa.Column('execution_time_ms', sa.Integer(), nullable=True),
        sa.Column('memory_usage_mb', sa.Float(), nullable=True),
        sa.Column('cpu_usage_percent', sa.Float(), nullable=True),
        sa.Column('is_public', sa.Boolean(), nullable=True),
        sa.Column('is_template', sa.Boolean(), nullable=True),
        sa.Column('download_count', sa.Integer(), nullable=True),
        sa.Column('view_count', sa.Integer(), nullable=True),
        sa.Column('parent_content_id', sa.Integer(), nullable=True),
        sa.Column('version_number', sa.String(length=20), nullable=True),
        sa.Column('is_latest_version', sa.Boolean(), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.Column('tags', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint('file_size_bytes >= 0', name='check_positive_file_size'),
        sa.ForeignKeyConstraint(['ai_request_id'], ['ai_requests.id'], ),
        sa.ForeignKeyConstraint(['conversation_id'], ['conversations.id'], ),
        sa.ForeignKeyConstraint(['parent_content_id'], ['generated_content.id'], ),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_generated_content_user_created', 'generated_content', ['user_id', 'created_at'], unique=False)
    op.create_index('ix_generated_content_type_created', 'generated_content', ['content_type', 'created_at'], unique=False)
    op.create_index('ix_generated_content_public', 'generated_content', ['is_public', 'created_at'], unique=False)
    op.create_index('ix_generated_content_template', 'generated_content', ['is_template', 'created_at'], unique=False)
    op.create_index(op.f('ix_generated_content_id'), 'generated_content', ['id'], unique=False)
    op.create_index(op.f('ix_generated_content_uuid'), 'generated_content', ['uuid'], unique=True)

    # Cost Tracking table
    op.create_table('cost_tracking',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('uuid', sa.String(length=36), nullable=True),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('ai_request_id', sa.Integer(), nullable=True),
        sa.Column('period_start', sa.DateTime(timezone=True), nullable=False),
        sa.Column('period_end', sa.DateTime(timezone=True), nullable=False),
        sa.Column('period_type', budget_period_enum, nullable=False),
        sa.Column('model_provider', model_provider_enum, nullable=False),
        sa.Column('model_name', sa.String(length=100), nullable=False),
        sa.Column('total_requests', sa.Integer(), nullable=True),
        sa.Column('successful_requests', sa.Integer(), nullable=True),
        sa.Column('failed_requests', sa.Integer(), nullable=True),
        sa.Column('total_input_tokens', sa.BigInteger(), nullable=True),
        sa.Column('total_output_tokens', sa.BigInteger(), nullable=True),
        sa.Column('total_tokens', sa.BigInteger(), nullable=True),
        sa.Column('input_cost', sa.Float(), nullable=True),
        sa.Column('output_cost', sa.Float(), nullable=True),
        sa.Column('total_cost', sa.Float(), nullable=True),
        sa.Column('average_cost_per_request', sa.Float(), nullable=True),
        sa.Column('average_cost_per_token', sa.Float(), nullable=True),
        sa.Column('average_response_time_ms', sa.Float(), nullable=True),
        sa.Column('median_response_time_ms', sa.Float(), nullable=True),
        sa.Column('p95_response_time_ms', sa.Float(), nullable=True),
        sa.Column('average_tokens_per_second', sa.Float(), nullable=True),
        sa.Column('average_user_rating', sa.Float(), nullable=True),
        sa.Column('success_rate', sa.Float(), nullable=True),
        sa.Column('retry_rate', sa.Float(), nullable=True),
        sa.Column('budget_limit', sa.Float(), nullable=True),
        sa.Column('budget_remaining', sa.Float(), nullable=True),
        sa.Column('budget_utilization_percent', sa.Float(), nullable=True),
        sa.Column('is_over_budget', sa.Boolean(), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint('total_cost >= 0', name='check_positive_total_cost'),
        sa.CheckConstraint('budget_utilization_percent >= 0', name='check_positive_budget_utilization'),
        sa.ForeignKeyConstraint(['ai_request_id'], ['ai_requests.id'], ),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('user_id', 'model_provider', 'model_name', 'period_start', 'period_type', name='uq_cost_tracking_user_model_period')
    )
    op.create_index('ix_cost_tracking_user_period', 'cost_tracking', ['user_id', 'period_start', 'period_end'], unique=False)
    op.create_index('ix_cost_tracking_model_period', 'cost_tracking', ['model_provider', 'model_name', 'period_start'], unique=False)
    op.create_index('ix_cost_tracking_cost', 'cost_tracking', ['total_cost'], unique=False)
    op.create_index(op.f('ix_cost_tracking_id'), 'cost_tracking', ['id'], unique=False)
    op.create_index(op.f('ix_cost_tracking_uuid'), 'cost_tracking', ['uuid'], unique=True)

    # User Budgets table
    op.create_table('user_budgets',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('uuid', sa.String(length=36), nullable=True),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('budget_name', sa.String(length=100), nullable=False),
        sa.Column('budget_amount', sa.Float(), nullable=False),
        sa.Column('budget_period', budget_period_enum, nullable=False),
        sa.Column('period_start', sa.DateTime(timezone=True), nullable=False),
        sa.Column('period_end', sa.DateTime(timezone=True), nullable=False),
        sa.Column('current_spend', sa.Float(), nullable=True),
        sa.Column('remaining_budget', sa.Float(), nullable=True),
        sa.Column('utilization_percent', sa.Float(), nullable=True),
        sa.Column('warning_threshold_percent', sa.Float(), nullable=True),
        sa.Column('critical_threshold_percent', sa.Float(), nullable=True),
        sa.Column('alerts_enabled', sa.Boolean(), nullable=True),
        sa.Column('warning_alert_sent', sa.Boolean(), nullable=True),
        sa.Column('critical_alert_sent', sa.Boolean(), nullable=True),
        sa.Column('budget_exceeded', sa.Boolean(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.Column('auto_renew', sa.Boolean(), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint('budget_amount > 0', name='check_positive_budget_amount'),
        sa.CheckConstraint('warning_threshold_percent > 0 AND warning_threshold_percent <= 100', name='check_valid_warning_threshold'),
        sa.CheckConstraint('critical_threshold_percent > 0 AND critical_threshold_percent <= 100', name='check_valid_critical_threshold'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_user_budgets_user_period', 'user_budgets', ['user_id', 'period_start'], unique=False)
    op.create_index('ix_user_budgets_active', 'user_budgets', ['is_active', 'period_end'], unique=False)
    op.create_index(op.f('ix_user_budgets_id'), 'user_budgets', ['id'], unique=False)
    op.create_index(op.f('ix_user_budgets_uuid'), 'user_budgets', ['uuid'], unique=True)

    # System Alerts table
    op.create_table('system_alerts',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('uuid', sa.String(length=36), nullable=True),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.Column('alert_type', alert_type_enum, nullable=False),
        sa.Column('severity', sa.String(length=20), nullable=True),
        sa.Column('title', sa.String(length=200), nullable=False),
        sa.Column('message', sa.Text(), nullable=False),
        sa.Column('source_table', sa.String(length=50), nullable=True),
        sa.Column('source_id', sa.Integer(), nullable=True),
        sa.Column('is_read', sa.Boolean(), nullable=True),
        sa.Column('is_dismissed', sa.Boolean(), nullable=True),
        sa.Column('is_resolved', sa.Boolean(), nullable=True),
        sa.Column('action_required', sa.Boolean(), nullable=True),
        sa.Column('action_url', sa.String(length=500), nullable=True),
        sa.Column('action_button_text', sa.String(length=50), nullable=True),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('resolved_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_system_alerts_user_created', 'system_alerts', ['user_id', 'created_at'], unique=False)
    op.create_index('ix_system_alerts_type_created', 'system_alerts', ['alert_type', 'created_at'], unique=False)
    op.create_index('ix_system_alerts_severity', 'system_alerts', ['severity', 'created_at'], unique=False)
    op.create_index('ix_system_alerts_status', 'system_alerts', ['is_read', 'is_dismissed', 'created_at'], unique=False)
    op.create_index(op.f('ix_system_alerts_id'), 'system_alerts', ['id'], unique=False)
    op.create_index(op.f('ix_system_alerts_uuid'), 'system_alerts', ['uuid'], unique=True)

    # Performance Metrics table
    op.create_table('performance_metrics',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('metric_name', sa.String(length=100), nullable=False),
        sa.Column('metric_category', sa.String(length=50), nullable=False),
        sa.Column('metric_type', sa.String(length=20), nullable=False),
        sa.Column('value', sa.Float(), nullable=False),
        sa.Column('previous_value', sa.Float(), nullable=True),
        sa.Column('change_percent', sa.Float(), nullable=True),
        sa.Column('min_value', sa.Float(), nullable=True),
        sa.Column('max_value', sa.Float(), nullable=True),
        sa.Column('avg_value', sa.Float(), nullable=True),
        sa.Column('median_value', sa.Float(), nullable=True),
        sa.Column('p95_value', sa.Float(), nullable=True),
        sa.Column('p99_value', sa.Float(), nullable=True),
        sa.Column('labels', sa.JSON(), nullable=True),
        sa.Column('dimensions', sa.JSON(), nullable=True),
        sa.Column('time_window_seconds', sa.Integer(), nullable=True),
        sa.Column('sample_count', sa.Integer(), nullable=True),
        sa.Column('unit', sa.String(length=20), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_performance_metrics_name_created', 'performance_metrics', ['metric_name', 'created_at'], unique=False)
    op.create_index('ix_performance_metrics_category_created', 'performance_metrics', ['metric_category', 'created_at'], unique=False)
    op.create_index('ix_performance_metrics_value', 'performance_metrics', ['value'], unique=False)
    op.create_index(op.f('ix_performance_metrics_id'), 'performance_metrics', ['id'], unique=False)

    # Audit Logs table
    op.create_table('audit_logs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('uuid', sa.String(length=36), nullable=True),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.Column('action', sa.String(length=100), nullable=False),
        sa.Column('resource_type', sa.String(length=50), nullable=False),
        sa.Column('resource_id', sa.String(length=50), nullable=True),
        sa.Column('ip_address', sa.String(length=45), nullable=True),
        sa.Column('user_agent', sa.String(length=500), nullable=True),
        sa.Column('request_method', sa.String(length=10), nullable=True),
        sa.Column('request_path', sa.String(length=500), nullable=True),
        sa.Column('request_headers', sa.JSON(), nullable=True),
        sa.Column('old_values', sa.JSON(), nullable=True),
        sa.Column('new_values', sa.JSON(), nullable=True),
        sa.Column('success', sa.Boolean(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_audit_logs_user_created', 'audit_logs', ['user_id', 'created_at'], unique=False)
    op.create_index('ix_audit_logs_action_created', 'audit_logs', ['action', 'created_at'], unique=False)
    op.create_index('ix_audit_logs_resource', 'audit_logs', ['resource_type', 'resource_id', 'created_at'], unique=False)
    op.create_index('ix_audit_logs_ip', 'audit_logs', ['ip_address', 'created_at'], unique=False)
    op.create_index(op.f('ix_audit_logs_id'), 'audit_logs', ['id'], unique=False)
    op.create_index(op.f('ix_audit_logs_uuid'), 'audit_logs', ['uuid'], unique=True)

    # Keep existing tables (unchanged)
    op.create_table('usage_logs',
        sa.Column('id', sa.Integer(), nullable=False, primary_key=True),
        sa.Column('user_id', sa.Integer(), sa.ForeignKey('users.id'), nullable=False),
        sa.Column('model_provider', model_provider_enum, nullable=False),
        sa.Column('model_name', sa.String(length=50), nullable=False),
        sa.Column('endpoint', sa.String(length=100), nullable=False),
        sa.Column('input_tokens', sa.Integer(), default=0),
        sa.Column('output_tokens', sa.Integer(), default=0),
        sa.Column('total_tokens', sa.Integer(), default=0),
        sa.Column('cost', sa.Float(), default=0.0),
        sa.Column('response_time', sa.Float()),
        sa.Column('request_id', sa.String(length=100)),
        sa.Column('conversation_id', sa.Integer(), sa.ForeignKey('conversations.id')),
        sa.Column('message_id', sa.Integer(), sa.ForeignKey('messages.id')),
        sa.Column('success', sa.Boolean(), default=True),
        sa.Column('error_message', sa.Text()),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'))
    )
    op.create_index('ix_usage_logs_user_created', 'usage_logs', ['user_id', 'created_at'])
    op.create_index('ix_usage_logs_model_created', 'usage_logs', ['model_provider', 'created_at'])
    op.create_index(op.f('ix_usage_logs_id'), 'usage_logs', ['id'])

    op.create_table('ai_models',
        sa.Column('id', sa.Integer(), nullable=False, primary_key=True),
        sa.Column('provider', model_provider_enum, nullable=False),
        sa.Column('name', sa.String(length=50), nullable=False),
        sa.Column('display_name', sa.String(length=100), nullable=False),
        sa.Column('description', sa.Text()),
        sa.Column('supports_streaming', sa.Boolean(), default=False),
        sa.Column('supports_functions', sa.Boolean(), default=False),
        sa.Column('supports_vision', sa.Boolean(), default=False),
        sa.Column('max_tokens', sa.Integer(), default=4096),
        sa.Column('context_window', sa.Integer(), default=4096),
        sa.Column('input_cost', sa.Float(), default=0.0),
        sa.Column('output_cost', sa.Float(), default=0.0),
        sa.Column('is_active', sa.Boolean(), default=True),
        sa.Column('priority', sa.Integer(), default=0),
        sa.Column('rate_limit_rpm', sa.Integer(), default=60),
        sa.Column('metadata', sa.JSON(), default={}),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=sa.text('now()'))
    )
    op.create_index('ix_ai_models_provider_active', 'ai_models', ['provider', 'is_active'])
    op.create_index(op.f('ix_ai_models_id'), 'ai_models', ['id'])

    op.create_table('agents',
        sa.Column('id', sa.Integer(), nullable=False, primary_key=True),
        sa.Column('uuid', sa.String(length=36), unique=True, index=True, default=sa.text('gen_random_uuid()')),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('description', sa.Text()),
        sa.Column('system_prompt', sa.Text()),
        sa.Column('model_provider', model_provider_enum, default='openai'),
        sa.Column('model_name', sa.String(length=50), default='gpt-4'),
        sa.Column('temperature', sa.Float(), default=0.7),
        sa.Column('max_tokens', sa.Integer(), default=2048),
        sa.Column('is_active', sa.Boolean(), default=True),
        sa.Column('is_public', sa.Boolean(), default=False),
        sa.Column('capabilities', sa.JSON(), default=[]),
        sa.Column('tags', sa.JSON(), default=[]),
        sa.Column('metadata', sa.JSON(), default={}),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=sa.text('now()'))
    )
    op.create_index(op.f('ix_agents_id'), 'agents', ['id'])
    
    op.create_table('tasks',
        sa.Column('id', sa.Integer(), nullable=False, primary_key=True),
        sa.Column('uuid', sa.String(length=36), unique=True, index=True, default=sa.text('gen_random_uuid()')),
        sa.Column('agent_id', sa.Integer(), sa.ForeignKey('agents.id')),
        sa.Column('user_id', sa.Integer(), sa.ForeignKey('users.id')),
        sa.Column('title', sa.String(length=200), nullable=False),
        sa.Column('description', sa.Text()),
        sa.Column('status', task_status_enum, default='pending', nullable=False),
        sa.Column('priority', sa.Integer(), default=0),
        sa.Column('input_data', sa.JSON(), default={}),
        sa.Column('output_data', sa.JSON(), default={}),
        sa.Column('error_message', sa.Text()),
        sa.Column('started_at', sa.DateTime(timezone=True)),
        sa.Column('completed_at', sa.DateTime(timezone=True)),
        sa.Column('estimated_duration', sa.Integer()),
        sa.Column('actual_duration', sa.Integer()),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=sa.text('now()'))
    )
    op.create_index('ix_tasks_status_created', 'tasks', ['status', 'created_at'])
    op.create_index('ix_tasks_user_status', 'tasks', ['user_id', 'status'])
    op.create_index(op.f('ix_tasks_id'), 'tasks', ['id'])
    
    op.create_table('system_metrics',
        sa.Column('id', sa.Integer(), nullable=False, primary_key=True),
        sa.Column('metric_name', sa.String(length=100), nullable=False),
        sa.Column('metric_value', sa.Float(), nullable=False),
        sa.Column('metric_type', sa.String(length=50), nullable=False),
        sa.Column('labels', sa.JSON(), default={}),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'))
    )
    op.create_index('ix_system_metrics_name_created', 'system_metrics', ['metric_name', 'created_at'])
    op.create_index(op.f('ix_system_metrics_id'), 'system_metrics', ['id'])

def downgrade():
    """Drop all tables and enums"""
    
    # Drop tables in reverse order
    op.drop_table('system_metrics')
    op.drop_table('tasks')
    op.drop_table('agents')
    op.drop_table('ai_models')
    op.drop_table('usage_logs')
    op.drop_table('audit_logs')
    op.drop_table('performance_metrics')
    op.drop_table('system_alerts')
    op.drop_table('user_budgets')
    op.drop_table('cost_tracking')
    op.drop_table('generated_content')
    op.drop_table('ai_requests')
    op.drop_table('messages')
    op.drop_table('conversations')
    op.drop_table('users')
    
    # Drop enums
    op.execute('DROP TYPE IF EXISTS alerttype')
    op.execute('DROP TYPE IF EXISTS budgetperiod')
    op.execute('DROP TYPE IF EXISTS contenttype')
    op.execute('DROP TYPE IF EXISTS requeststatus')
    op.execute('DROP TYPE IF EXISTS taskstatus')
    op.execute('DROP TYPE IF EXISTS modelprovider')
    op.execute('DROP TYPE IF EXISTS conversationstatus')
    op.execute('DROP TYPE IF EXISTS messagerole')
    op.execute('DROP TYPE IF EXISTS userrole')