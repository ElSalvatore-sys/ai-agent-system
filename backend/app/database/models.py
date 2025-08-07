from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, Float, JSON, ForeignKey, Enum as SQLEnum, Index, UniqueConstraint, CheckConstraint, BigInteger
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any
import uuid

from app.database.database import Base

class UserRole(str, Enum):
    ADMIN = "admin"
    USER = "user"
    GUEST = "guest"

class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class ConversationStatus(str, Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"

class ModelProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    LOCAL_OLLAMA = "local_ollama"
    LOCAL_HF = "local_hf"

class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class RequestStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ContentType(str, Enum):
    CODE = "code"
    DOCUMENT = "document"
    IMAGE = "image"
    TEXT = "text"
    JSON = "json"
    XML = "xml"
    HTML = "html"
    CSV = "csv"
    MARKDOWN = "markdown"

class BudgetPeriod(str, Enum):
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"

class AlertType(str, Enum):
    BUDGET_WARNING = "budget_warning"
    BUDGET_EXCEEDED = "budget_exceeded"
    PERFORMANCE_ISSUE = "performance_issue"
    SYSTEM_ERROR = "system_error"
    SECURITY_ALERT = "security_alert"

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(String(36), unique=True, index=True, default=lambda: str(uuid.uuid4()))
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(100))
    role = Column(SQLEnum(UserRole), default=UserRole.USER, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_login = Column(DateTime(timezone=True))
    
    # Preferences and Settings
    preferences = Column(JSON, default=dict)
    api_quota_limit = Column(Integer, default=1000)  # Monthly API calls
    api_quota_used = Column(Integer, default=0)
    
    # Budget Management
    monthly_budget_limit = Column(Float, default=100.0)  # USD
    current_month_spend = Column(Float, default=0.0)
    budget_alerts_enabled = Column(Boolean, default=True)
    budget_warning_threshold = Column(Float, default=0.8)  # 80%
    
    # User Preferences
    preferred_model_provider = Column(SQLEnum(ModelProvider), default=ModelProvider.OPENAI)
    preferred_model = Column(String(50), default="gpt-4")
    default_temperature = Column(Float, default=0.7)
    timezone = Column(String(50), default="UTC")
    
    # Security
    failed_login_attempts = Column(Integer, default=0)
    locked_until = Column(DateTime(timezone=True))
    password_changed_at = Column(DateTime(timezone=True))
    
    # Relationships
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")
    usage_logs = relationship("UsageLog", back_populates="user", cascade="all, delete-orphan")
    ai_requests = relationship("AIRequest", back_populates="user", cascade="all, delete-orphan")
    generated_content = relationship("GeneratedContent", back_populates="user", cascade="all, delete-orphan")
    cost_tracking = relationship("CostTracking", back_populates="user", cascade="all, delete-orphan")
    user_budgets = relationship("UserBudget", back_populates="user", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('ix_users_email_active', 'email', 'is_active'),
        Index('ix_users_created_at', 'created_at'),
        CheckConstraint('monthly_budget_limit >= 0', name='check_positive_budget'),
        CheckConstraint('budget_warning_threshold >= 0 AND budget_warning_threshold <= 1', name='check_valid_warning_threshold'),
    )

class Conversation(Base):
    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(String(36), unique=True, index=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(String(200), nullable=False)
    status = Column(SQLEnum(ConversationStatus), default=ConversationStatus.ACTIVE, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # AI Configuration
    model_provider = Column(SQLEnum(ModelProvider), default=ModelProvider.OPENAI)
    model_name = Column(String(50), default="gpt-4")
    temperature = Column(Float, default=0.7)
    max_tokens = Column(Integer, default=2048)
    system_prompt = Column(Text)
    
    # Analytics
    total_messages = Column(Integer, default=0)
    total_tokens = Column(BigInteger, default=0)
    total_cost = Column(Float, default=0.0)
    average_response_time = Column(Float, default=0.0)
    
    # Metadata
    meta_data = Column(JSON, default=dict)
    tags = Column(JSON, default=list)  # Array of strings
    
    # Relationships
    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
    ai_requests = relationship("AIRequest", back_populates="conversation")
    
    __table_args__ = (
        Index('ix_conversations_user_status', 'user_id', 'status'),
        Index('ix_conversations_updated', 'updated_at'),
        Index('ix_conversations_cost', 'total_cost'),
        CheckConstraint('total_cost >= 0', name='check_positive_cost'),
    )

class Message(Base):
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(String(36), unique=True, index=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False)
    role = Column(SQLEnum(MessageRole), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # AI metadata
    model_used = Column(String(50))
    tokens_used = Column(Integer, default=0)
    cost = Column(Float, default=0.0)
    response_time = Column(Float)  # in seconds
    
    # Additional metadata
    meta_data = Column(JSON, default=dict)
    
    # Relationships
    conversation = relationship("Conversation", back_populates="messages")
    
    __table_args__ = (
        Index('ix_messages_conversation_created', 'conversation_id', 'created_at'),
    )

class UsageLog(Base):
    __tablename__ = "usage_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    model_provider = Column(SQLEnum(ModelProvider), nullable=False)
    model_name = Column(String(50), nullable=False)
    endpoint = Column(String(100), nullable=False)
    
    # Usage metrics
    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    cost = Column(Float, default=0.0)
    response_time = Column(Float)  # in seconds
    
    # Request details
    request_id = Column(String(100))
    conversation_id = Column(Integer, ForeignKey("conversations.id"))
    message_id = Column(Integer, ForeignKey("messages.id"))
    
    # Status
    success = Column(Boolean, default=True)
    error_message = Column(Text)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="usage_logs")
    
    __table_args__ = (
        Index('ix_usage_logs_user_created', 'user_id', 'created_at'),
        Index('ix_usage_logs_model_created', 'model_provider', 'created_at'),
    )

class AIModel(Base):
    __tablename__ = "ai_models"
    
    id = Column(Integer, primary_key=True, index=True)
    provider = Column(SQLEnum(ModelProvider), nullable=False)
    name = Column(String(50), nullable=False)
    display_name = Column(String(100), nullable=False)
    description = Column(Text)
    
    # Host information
    host_type = Column(String(20), default="cloud")  # cloud | local | edge
    device_id = Column(String(64))
    availability = Column(Boolean, default=True)
    
    # Capabilities
    supports_streaming = Column(Boolean, default=False)
    supports_functions = Column(Boolean, default=False)
    supports_vision = Column(Boolean, default=False)
    
    # Limits
    max_tokens = Column(Integer, default=4096)
    context_window = Column(Integer, default=4096)
    
    # Pricing (per 1K tokens)
    input_cost = Column(Float, default=0.0)
    output_cost = Column(Float, default=0.0)
    
    # Configuration
    is_active = Column(Boolean, default=True)
    priority = Column(Integer, default=0)  # Higher priority = preferred
    rate_limit_rpm = Column(Integer, default=60)  # Requests per minute
    
    # Metadata
    meta_data = Column(JSON, default=dict)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    __table_args__ = (
        Index('ix_ai_models_provider_active', 'provider', 'is_active'),
    )

class Agent(Base):
    __tablename__ = "agents"
    
    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(String(36), unique=True, index=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(100), nullable=False)
    description = Column(Text)
    
    # Configuration
    system_prompt = Column(Text)
    model_provider = Column(SQLEnum(ModelProvider), default=ModelProvider.OPENAI)
    model_name = Column(String(50), default="gpt-4")
    temperature = Column(Float, default=0.7)
    max_tokens = Column(Integer, default=2048)
    
    # Status
    is_active = Column(Boolean, default=True)
    is_public = Column(Boolean, default=False)
    
    # Metadata
    capabilities = Column(JSON, default=list)  # List of capabilities
    tags = Column(JSON, default=list)  # List of tags
    meta_data = Column(JSON, default=dict)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    tasks = relationship("Task", back_populates="agent", cascade="all, delete-orphan")

class Task(Base):
    __tablename__ = "tasks"
    
    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(String(36), unique=True, index=True, default=lambda: str(uuid.uuid4()))
    agent_id = Column(Integer, ForeignKey("agents.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    
    # Task details
    title = Column(String(200), nullable=False)
    description = Column(Text)
    status = Column(SQLEnum(TaskStatus), default=TaskStatus.PENDING, nullable=False)
    priority = Column(Integer, default=0)
    
    # Execution
    input_data = Column(JSON, default=dict)
    output_data = Column(JSON, default=dict)
    error_message = Column(Text)
    
    # Timing
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    estimated_duration = Column(Integer)  # in seconds
    actual_duration = Column(Integer)  # in seconds
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    agent = relationship("Agent", back_populates="tasks")
    
    __table_args__ = (
        Index('ix_tasks_status_created', 'status', 'created_at'),
        Index('ix_tasks_user_status', 'user_id', 'status'),
    )

class SystemMetrics(Base):
    __tablename__ = "system_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float, nullable=False)
    metric_type = Column(String(50), nullable=False)  # counter, gauge, histogram
    
    # Dimensions
    labels = Column(JSON, default=dict)  # Key-value pairs for metric dimensions
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (
        Index('ix_system_metrics_name_created', 'metric_name', 'created_at'),
    )

# New Comprehensive Models

class AIRequest(Base):
    """
    Detailed tracking of all AI model requests with performance metrics
    """
    __tablename__ = "ai_requests"
    
    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(String(36), unique=True, index=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    conversation_id = Column(Integer, ForeignKey("conversations.id"))
    message_id = Column(Integer, ForeignKey("messages.id"))
    
    # Request Details
    request_id = Column(String(100), unique=True, index=True)
    model_provider = Column(SQLEnum(ModelProvider), nullable=False)
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50))
    
    # Request Configuration
    temperature = Column(Float)
    max_tokens = Column(Integer)
    top_p = Column(Float)
    frequency_penalty = Column(Float)
    presence_penalty = Column(Float)
    system_prompt = Column(Text)
    
    # Input/Output
    input_text = Column(Text)
    output_text = Column(Text)
    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    
    # Performance Metrics
    request_start_time = Column(DateTime(timezone=True), nullable=False)
    request_end_time = Column(DateTime(timezone=True))
    response_time_ms = Column(Integer)  # milliseconds
    time_to_first_token_ms = Column(Integer)  # For streaming
    tokens_per_second = Column(Float)
    
    # Cost and Billing
    input_cost_usd = Column(Float, default=0.0)
    output_cost_usd = Column(Float, default=0.0)
    total_cost_usd = Column(Float, default=0.0)
    cost_per_token = Column(Float, default=0.0)
    
    # Status and Quality
    status = Column(SQLEnum(RequestStatus), default=RequestStatus.PENDING, nullable=False)
    success = Column(Boolean, default=False)
    error_code = Column(String(50))
    error_message = Column(Text)
    retry_count = Column(Integer, default=0)
    
    # Quality Metrics
    confidence_score = Column(Float)  # Model confidence (if available)
    safety_score = Column(Float)     # Content safety score
    relevance_score = Column(Float)  # Response relevance
    user_rating = Column(Integer)    # User feedback (1-5)
    user_feedback = Column(Text)     # User feedback text
    
    # Technical Details
    api_endpoint = Column(String(200))
    user_agent = Column(String(500))
    ip_address = Column(String(45))  # IPv6 compatible
    request_headers = Column(JSON, default=dict)
    response_headers = Column(JSON, default=dict)
    
    # Metadata
    meta_data = Column(JSON, default=dict)
    tags = Column(JSON, default=list)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="ai_requests")
    conversation = relationship("Conversation", back_populates="ai_requests")
    message = relationship("Message")
    generated_content = relationship("GeneratedContent", back_populates="ai_request")
    
    __table_args__ = (
        Index('ix_ai_requests_user_created', 'user_id', 'created_at'),
        Index('ix_ai_requests_model_created', 'model_provider', 'model_name', 'created_at'),
        Index('ix_ai_requests_status_created', 'status', 'created_at'),
        Index('ix_ai_requests_cost', 'total_cost_usd'),
        Index('ix_ai_requests_performance', 'response_time_ms'),
        CheckConstraint('total_cost_usd >= 0', name='check_positive_ai_request_cost'),
        CheckConstraint('response_time_ms >= 0', name='check_positive_response_time'),
    )

class GeneratedContent(Base):
    """
    Track all content generated by AI (code, documents, media)
    """
    __tablename__ = "generated_content"
    
    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(String(36), unique=True, index=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    ai_request_id = Column(Integer, ForeignKey("ai_requests.id"))
    conversation_id = Column(Integer, ForeignKey("conversations.id"))
    
    # Content Details
    title = Column(String(200))
    description = Column(Text)
    content_type = Column(SQLEnum(ContentType), nullable=False)
    file_extension = Column(String(10))
    mime_type = Column(String(100))
    
    # Content Storage
    content_text = Column(Text)  # For text-based content
    content_binary = Column(Text)  # Base64 encoded binary content
    file_path = Column(String(500))  # Path to stored file
    file_size_bytes = Column(BigInteger)
    
    # Content Metadata
    language = Column(String(50))  # Programming language or document language
    framework = Column(String(100))  # For code: framework/library used
    version = Column(String(50))     # Version of generated content
    
    # Quality and Validation
    syntax_valid = Column(Boolean)   # For code: syntax validation
    executable = Column(Boolean)     # For code: can be executed
    compilation_status = Column(String(50))  # For compiled languages
    test_results = Column(JSON, default=dict)  # Test execution results
    
    # Usage and Performance
    execution_time_ms = Column(Integer)  # For code execution
    memory_usage_mb = Column(Float)      # Memory usage during execution
    cpu_usage_percent = Column(Float)    # CPU usage during execution
    
    # Sharing and Access
    is_public = Column(Boolean, default=False)
    is_template = Column(Boolean, default=False)
    download_count = Column(Integer, default=0)
    view_count = Column(Integer, default=0)
    
    # Version Control
    parent_content_id = Column(Integer, ForeignKey("generated_content.id"))
    version_number = Column(String(20), default="1.0")
    is_latest_version = Column(Boolean, default=True)
    
    # Metadata
    meta_data = Column(JSON, default=dict)
    tags = Column(JSON, default=list)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="generated_content")
    ai_request = relationship("AIRequest", back_populates="generated_content")
    conversation = relationship("Conversation")
    parent_content = relationship("GeneratedContent", remote_side=[id])
    child_versions = relationship("GeneratedContent", remote_side=[parent_content_id])
    
    __table_args__ = (
        Index('ix_generated_content_user_created', 'user_id', 'created_at'),
        Index('ix_generated_content_type_created', 'content_type', 'created_at'),
        Index('ix_generated_content_public', 'is_public', 'created_at'),
        Index('ix_generated_content_template', 'is_template', 'created_at'),
        CheckConstraint('file_size_bytes >= 0', name='check_positive_file_size'),
    )

class CostTracking(Base):
    """
    Detailed cost tracking and analytics
    """
    __tablename__ = "cost_tracking"
    
    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(String(36), unique=True, index=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    ai_request_id = Column(Integer, ForeignKey("ai_requests.id"))
    
    # Time Period
    period_start = Column(DateTime(timezone=True), nullable=False)
    period_end = Column(DateTime(timezone=True), nullable=False)
    period_type = Column(SQLEnum(BudgetPeriod), default=BudgetPeriod.DAILY, nullable=False)
    
    # Cost Breakdown
    model_provider = Column(SQLEnum(ModelProvider), nullable=False)
    model_name = Column(String(100), nullable=False)
    
    # Usage Metrics
    total_requests = Column(Integer, default=0)
    successful_requests = Column(Integer, default=0)
    failed_requests = Column(Integer, default=0)
    
    # Token Usage
    total_input_tokens = Column(BigInteger, default=0)
    total_output_tokens = Column(BigInteger, default=0)
    total_tokens = Column(BigInteger, default=0)
    
    # Cost Details (USD)
    input_cost = Column(Float, default=0.0)
    output_cost = Column(Float, default=0.0)
    total_cost = Column(Float, default=0.0)
    average_cost_per_request = Column(Float, default=0.0)
    average_cost_per_token = Column(Float, default=0.0)
    
    # Performance Metrics
    average_response_time_ms = Column(Float, default=0.0)
    median_response_time_ms = Column(Float, default=0.0)
    p95_response_time_ms = Column(Float, default=0.0)
    average_tokens_per_second = Column(Float, default=0.0)
    
    # Quality Metrics
    average_user_rating = Column(Float)
    success_rate = Column(Float, default=0.0)
    retry_rate = Column(Float, default=0.0)
    
    # Budget Tracking
    budget_limit = Column(Float)
    budget_remaining = Column(Float)
    budget_utilization_percent = Column(Float, default=0.0)
    is_over_budget = Column(Boolean, default=False)
    
    # Metadata
    meta_data = Column(JSON, default=dict)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="cost_tracking")
    ai_request = relationship("AIRequest")
    
    __table_args__ = (
        Index('ix_cost_tracking_user_period', 'user_id', 'period_start', 'period_end'),
        Index('ix_cost_tracking_model_period', 'model_provider', 'model_name', 'period_start'),
        Index('ix_cost_tracking_cost', 'total_cost'),
        UniqueConstraint('user_id', 'model_provider', 'model_name', 'period_start', 'period_type', 
                        name='uq_cost_tracking_user_model_period'),
        CheckConstraint('total_cost >= 0', name='check_positive_total_cost'),
        CheckConstraint('budget_utilization_percent >= 0', name='check_positive_budget_utilization'),
    )

class UserBudget(Base):
    """
    User budget management and alerts
    """
    __tablename__ = "user_budgets"
    
    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(String(36), unique=True, index=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Budget Configuration
    budget_name = Column(String(100), nullable=False)
    budget_amount = Column(Float, nullable=False)
    budget_period = Column(SQLEnum(BudgetPeriod), default=BudgetPeriod.MONTHLY, nullable=False)
    
    # Period Tracking
    period_start = Column(DateTime(timezone=True), nullable=False)
    period_end = Column(DateTime(timezone=True), nullable=False)
    
    # Usage Tracking
    current_spend = Column(Float, default=0.0)
    remaining_budget = Column(Float, default=0.0)
    utilization_percent = Column(Float, default=0.0)
    
    # Alert Configuration
    warning_threshold_percent = Column(Float, default=80.0)  # 80%
    critical_threshold_percent = Column(Float, default=95.0)  # 95%
    alerts_enabled = Column(Boolean, default=True)
    
    # Alert Status
    warning_alert_sent = Column(Boolean, default=False)
    critical_alert_sent = Column(Boolean, default=False)
    budget_exceeded = Column(Boolean, default=False)
    
    # Status
    is_active = Column(Boolean, default=True)
    auto_renew = Column(Boolean, default=True)
    
    # Metadata
    meta_data = Column(JSON, default=dict)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="user_budgets")
    
    __table_args__ = (
        Index('ix_user_budgets_user_period', 'user_id', 'period_start'),
        Index('ix_user_budgets_active', 'is_active', 'period_end'),
        CheckConstraint('budget_amount > 0', name='check_positive_budget_amount'),
        CheckConstraint('warning_threshold_percent > 0 AND warning_threshold_percent <= 100', 
                       name='check_valid_warning_threshold'),
        CheckConstraint('critical_threshold_percent > 0 AND critical_threshold_percent <= 100', 
                       name='check_valid_critical_threshold'),
    )

class SystemAlert(Base):
    """
    System alerts and notifications
    """
    __tablename__ = "system_alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(String(36), unique=True, index=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(Integer, ForeignKey("users.id"))  # Nullable for system-wide alerts
    
    # Alert Details
    alert_type = Column(SQLEnum(AlertType), nullable=False)
    severity = Column(String(20), default="medium")  # low, medium, high, critical
    title = Column(String(200), nullable=False)
    message = Column(Text, nullable=False)
    
    # Context
    source_table = Column(String(50))  # Table that triggered the alert
    source_id = Column(Integer)        # ID of the record that triggered the alert
    
    # Status
    is_read = Column(Boolean, default=False)
    is_dismissed = Column(Boolean, default=False)
    is_resolved = Column(Boolean, default=False)
    
    # Actions
    action_required = Column(Boolean, default=False)
    action_url = Column(String(500))
    action_button_text = Column(String(50))
    
    # Timing
    expires_at = Column(DateTime(timezone=True))
    resolved_at = Column(DateTime(timezone=True))
    
    # Metadata
    meta_data = Column(JSON, default=dict)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User")
    
    __table_args__ = (
        Index('ix_system_alerts_user_created', 'user_id', 'created_at'),
        Index('ix_system_alerts_type_created', 'alert_type', 'created_at'),
        Index('ix_system_alerts_severity', 'severity', 'created_at'),
        Index('ix_system_alerts_status', 'is_read', 'is_dismissed', 'created_at'),
    )

class PerformanceMetrics(Base):
    """
    Enhanced system performance monitoring
    """
    __tablename__ = "performance_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Metric Identification
    metric_name = Column(String(100), nullable=False)
    metric_category = Column(String(50), nullable=False)  # system, api, model, user
    metric_type = Column(String(20), nullable=False)      # counter, gauge, histogram, timer
    
    # Metric Values
    value = Column(Float, nullable=False)
    previous_value = Column(Float)
    change_percent = Column(Float)
    
    # Statistical Data
    min_value = Column(Float)
    max_value = Column(Float)
    avg_value = Column(Float)
    median_value = Column(Float)
    p95_value = Column(Float)
    p99_value = Column(Float)
    
    # Dimensions and Labels
    labels = Column(JSON, default=dict)
    dimensions = Column(JSON, default=dict)
    
    # Time Window
    time_window_seconds = Column(Integer, default=60)  # Aggregation window
    sample_count = Column(Integer, default=1)
    
    # Metadata
    unit = Column(String(20))  # ms, requests/sec, MB, etc.
    description = Column(Text)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (
        Index('ix_performance_metrics_name_created', 'metric_name', 'created_at'),
        Index('ix_performance_metrics_category_created', 'metric_category', 'created_at'),
        Index('ix_performance_metrics_value', 'value'),
    )

class AuditLog(Base):
    """
    Comprehensive audit logging for security and compliance
    """
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(String(36), unique=True, index=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(Integer, ForeignKey("users.id"))
    
    # Action Details
    action = Column(String(100), nullable=False)  # create, update, delete, login, etc.
    resource_type = Column(String(50), nullable=False)  # user, conversation, ai_request, etc.
    resource_id = Column(String(50))
    
    # Request Details
    ip_address = Column(String(45))
    user_agent = Column(String(500))
    request_method = Column(String(10))
    request_path = Column(String(500))
    request_headers = Column(JSON, default=dict)
    
    # Changes
    old_values = Column(JSON, default=dict)
    new_values = Column(JSON, default=dict)
    
    # Status
    success = Column(Boolean, default=True)
    error_message = Column(Text)
    
    # Metadata
    meta_data = Column(JSON, default=dict)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User")
    
    __table_args__ = (
        Index('ix_audit_logs_user_created', 'user_id', 'created_at'),
        Index('ix_audit_logs_action_created', 'action', 'created_at'),
        Index('ix_audit_logs_resource', 'resource_type', 'resource_id', 'created_at'),
        Index('ix_audit_logs_ip', 'ip_address', 'created_at'),
    )