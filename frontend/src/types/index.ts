// Core application types
export interface User {
  id: string;
  name: string;
  email: string;
  avatar?: string;
  isOnline: boolean;
  lastSeen?: Date;
}

export type ProcessingType = 'local' | 'cloud';

export interface Message {
  id: string;
  content: string;
  senderId: string;
  timestamp: Date;
  type: 'text' | 'ai' | 'system';
  isEdited?: boolean;
  cost?: number;
  modelId?: string;
  processingType?: ProcessingType;
}

export interface ChatSession {
  id: string;
  title: string;
  userId: string;
  messages: Message[];
  createdAt: Date;
  updatedAt: Date;
  isActive: boolean;
  modelId?: string;
}

// AI Model related types
export type ModelProvider = 'Ollama' | 'Hugging Face' | 'OpenAI' | 'Claude' | 'Gemini';
export type ModelStatus = 'online' | 'offline' | 'loading' | 'error';
export type ModelCapability = 'coding' | 'reasoning' | 'multimodal' | 'chat';
export type InstallationStatus = 'installed' | 'not-installed' | 'installing' | 'uninstalling' | 'error';
export type QuantizationLevel = '4-bit' | '8-bit' | '16-bit' | 'full';

export interface ModelPerformance {
  speed: number; // tokens/sec
  quality: number; // 1-5 rating or a score
  cost: number; // cost per 1M tokens
}

export interface AIModel {
  id: string;
  name: string;
  description: string;
  provider: ModelProvider;
  hostType: 'cloud' | 'local';
  status: ModelStatus;
  performance?: ModelPerformance;
  capabilities: ModelCapability[];
  installationStatus?: InstallationStatus; // Only for local models
  isAvailable: boolean; // can be derived from status === 'online'
  maxTokens: number;
}


export interface AIResponse {
  content: string;
  tokens: number;
  model: string;
  timestamp: Date;
  confidence?: number;
}

export interface Analytics {
  totalSessions: number;
  totalMessages: number;
  averageResponseTime: number;
  modelUsage: Record<string, number>;
  dailyStats: DailyStats[];
  userEngagement: UserEngagement;
}

export interface DailyStats {
  date: string;
  sessions: number;
  messages: number;
  responseTime: number;
}

export interface UserEngagement {
  activeUsers: number;
  avgSessionDuration: number;
  retentionRate: number;
  satisfactionScore: number;
}

export interface Settings {
  theme: 'light' | 'dark' | 'system';
  language: string;
  notifications: NotificationSettings;
  ai: AISettings;
  privacy: PrivacySettings;
}

export interface NotificationSettings {
  email: boolean;
  push: boolean;
  sound: boolean;
  messagePreview: boolean;
}

export interface AISettings {
  defaultModel: string;
  temperature: number;
  maxTokens: number;
  streamResponse: boolean;
  saveHistory: boolean;
  preferLocalModels: boolean;
}

export interface PrivacySettings {
  dataCollection: boolean;
  analytics: boolean;
  shareUsageData: boolean;
}

// WebSocket event types
export interface WebSocketEvents {
  'user:connected': User;
  'user:disconnected': { userId: string };
  'message:new': Message;
  'message:typing': { userId: string; isTyping: boolean };
  'session:created': ChatSession;
  'session:updated': ChatSession;
  'ai:response': AIResponse;
  'ai:thinking': { sessionId: string };
  'system:notification': SystemNotification;
  'local_model:status': AIModel;
  'local_model:resource_usage': LocalResourceUsage;
}


export interface SystemNotification {
  id: string;
  type: 'info' | 'warning' | 'error' | 'success';
  title: string;
  message: string;
  timestamp: Date;
  actions?: NotificationAction[];
}

export interface NotificationAction {
  label: string;
  action: string;
  variant?: 'primary' | 'secondary' | 'destructive';
}

// API response types
export interface ApiResponse<T = any> {
  data?: T;
  message?: string;
  error?: string;
  success: boolean;
  timestamp: Date;
}

export interface PaginatedResponse<T> extends ApiResponse<T[]> {
  pagination: {
    page: number;
    limit: number;
    total: number;
    totalPages: number;
  };
}

// Component prop types
export interface BaseComponentProps {
  className?: string;
  children?: React.ReactNode;
}

export interface ButtonProps extends BaseComponentProps {
  variant?: 'primary' | 'secondary' | 'outline' | 'ghost' | 'destructive';
  size?: 'sm' | 'md' | 'lg';
  disabled?: boolean;
  loading?: boolean;
  onClick?: () => void;
  type?: 'button' | 'submit' | 'reset';
}

export interface InputProps extends BaseComponentProps {
  type?: string;
  placeholder?: string;
  value?: string;
  onChange?: (value: string) => void;
  disabled?: boolean;
  error?: string;
  label?: string;
}

export interface ModalProps extends BaseComponentProps {
  isOpen: boolean;
  onClose: () => void;
  title?: string;
  size?: 'sm' | 'md' | 'lg' | 'xl';
}

// Hook return types
export interface UseAIReturn {
  sendMessage: (message: string, model?: string) => Promise<AIResponse>;
  sendParallelMessages: (params: {
    sessionId: string;
    message: string;
    modelIds: string[];
    options?: { temperature?: number; maxTokens?: number };
  }) => Promise<any>;
  isLoading: boolean;
  error: string | null;
  currentModel: AIModel | null;
  availableModels: AIModel[];
  setModel: (modelId: string) => void;
  clearError: () => void;
}

export interface UseWebSocketReturn {
  isConnected: boolean;
  error: string | null;
  emit: <K extends keyof WebSocketEvents>(event: K, data: WebSocketEvents[K]) => void;
  on: <K extends keyof WebSocketEvents>(event: K, callback: (data: WebSocketEvents[K]) => void) => void;
  off: <K extends keyof WebSocketEvents>(event: K, callback?: (data: WebSocketEvents[K]) => void) => void;
  connect: () => void;
  disconnect: () => void;
}

// Utility types
export type Theme = 'light' | 'dark' | 'system';
export type LoadingState = 'idle' | 'loading' | 'success' | 'error';
export type SortOrder = 'asc' | 'desc';

export interface SortConfig {
  key: string;
  order: SortOrder;
}

export interface FilterConfig {
  [key: string]: any;
}

export interface SearchConfig {
  query: string;
  fields: string[];
}

// New types for Local LLM UI

// Local Model Management
export interface LocalResourceUsage {
  cpu: number; // percentage
  gpu: number; // percentage
  memory: number; // in GB
  totalMemory: number; // in GB
}

export interface LocalModelSettings {
  temperature: number;
  topP: number;
  contextLength: number;
  quantization?: QuantizationLevel;
}

// Performance Comparison
export interface ModelComparisonResult {
  id: string;
  prompt: string;
  responses: ComparisonResponse[];
}

export interface ComparisonResponse {
    modelId: string;
    modelName: string;
    content: string;
    latency: number; // in ms
    qualityRating?: number; // 1-5
    cost?: number;
}

// Analytics Dashboard
export interface CostSavingsData {
    period: string; // e.g., 'Last 30 Days'
    cloudCost: number;
    localCost: number;
    savings: number;
    roi: number; // percentage
}

export interface ModelPerformanceData {
    modelId: string;
    modelName:string;
    avgLatency: number;
    avgQuality: number;
    totalUsage: number; // tokens or requests
}

export interface ResourceUtilizationDataPoint {
    timestamp: number; // Unix timestamp
    cpu: number;
    gpu: number;
    memory: number;
}