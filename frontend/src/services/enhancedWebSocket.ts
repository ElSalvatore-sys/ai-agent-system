import { io, Socket } from 'socket.io-client';
import type { WebSocketEvents } from '@/types';

interface QueuedMessage {
  id: string;
  event: string;
  data: any;
  timestamp: number;
  priority: 'high' | 'medium' | 'low';
  retryCount: number;
  maxRetries: number;
}

interface ConnectionMetrics {
  connectTime: number;
  disconnectTime: number;
  totalConnections: number;
  totalDisconnections: number;
  avgLatency: number;
  lastLatency: number;
  qualityScore: number;
}

interface SmartReconnectConfig {
  baseDelay: number;
  maxDelay: number;
  backoffFactor: number;
  jitter: boolean;
  adaptiveTimeout: boolean;
}

export class EnhancedWebSocketService {
  private socket: Socket | null = null;
  private messageQueue: QueuedMessage[] = [];
  private listeners: Map<string, Set<Function>> = new Map();
  private reconnectConfig: SmartReconnectConfig = {
    baseDelay: 1000,
    maxDelay: 30000,
    backoffFactor: 1.5,
    jitter: true,
    adaptiveTimeout: true
  };
  
  private connectionMetrics: ConnectionMetrics = {
    connectTime: 0,
    disconnectTime: 0,
    totalConnections: 0,
    totalDisconnections: 0,
    avgLatency: 0,
    lastLatency: 0,
    qualityScore: 100
  };

  private reconnectAttempts = 0;
  private maxReconnectAttempts = 10;
  private currentReconnectDelay = this.reconnectConfig.baseDelay;
  private heartbeatInterval: NodeJS.Timeout | null = null;
  private latencyCheckInterval: NodeJS.Timeout | null = null;
  private qualityMonitorInterval: NodeJS.Timeout | null = null;
  
  // State synchronization
  private syncedState: Map<string, any> = new Map();
  private pendingUpdates: Map<string, any> = new Map();

  constructor() {
    this.initializeQualityMonitoring();
    this.connect();
  }

  connect(): void {
    if (this.socket?.connected) {
      return;
    }

    const token = localStorage.getItem('authToken');
    const url = import.meta.env.VITE_WS_URL || 'ws://localhost:8000';

    // Calculate adaptive timeout based on connection history
    const timeout = this.calculateAdaptiveTimeout();

    this.socket = io(url, {
      auth: { token },
      transports: ['websocket', 'polling'],
      timeout,
      forceNew: false,
      reconnection: false, // We handle reconnection manually
      query: {
        clientId: this.generateClientId(),
        capabilities: JSON.stringify(['real-time-sync', 'message-queue', 'smart-reconnect'])
      }
    });

    this.setupEventListeners();
    this.setupHeartbeat();
    this.setupLatencyMonitoring();
  }

  disconnect(): void {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }

    this.clearTimers();
    this.listeners.clear();
    this.reconnectAttempts = 0;
    this.currentReconnectDelay = this.reconnectConfig.baseDelay;
  }

  private setupEventListeners(): void {
    if (!this.socket) return;

    this.socket.on('connect', () => {
      console.log('✅ Enhanced WebSocket connected');
      this.connectionMetrics.connectTime = Date.now();
      this.connectionMetrics.totalConnections++;
      this.reconnectAttempts = 0;
      this.currentReconnectDelay = this.reconnectConfig.baseDelay;
      
      // Process queued messages
      this.processMessageQueue();
      
      // Sync pending state updates
      this.syncPendingUpdates();
      
      this.emit('connection:established', { 
        timestamp: new Date(),
        metrics: this.connectionMetrics
      });
    });

    this.socket.on('disconnect', (reason) => {
      console.log('❌ Enhanced WebSocket disconnected:', reason);
      this.connectionMetrics.disconnectTime = Date.now();
      this.connectionMetrics.totalDisconnections++;
      
      this.emit('connection:lost', { 
        reason, 
        timestamp: new Date(),
        willReconnect: reason !== 'io server disconnect'
      });

      if (reason !== 'io server disconnect') {
        this.handleSmartReconnect();
      }
    });

    this.socket.on('connect_error', (error) => {
      console.error('WebSocket connection error:', error);
      this.updateQualityScore(-10);
      this.emit('connection:error', { 
        error: error.message, 
        timestamp: new Date(),
        qualityScore: this.connectionMetrics.qualityScore
      });
      this.handleSmartReconnect();
    });

    // Enhanced event handlers
    this.socket.on('pong', (latency) => {
      this.connectionMetrics.lastLatency = latency;
      this.updateAverageLatency(latency);
      this.updateQualityScore(latency < 100 ? 5 : latency < 500 ? 0 : -5);
    });

    // State synchronization events
    this.socket.on('state:sync', (data) => {
      this.handleStateSync(data);
    });

    this.socket.on('state:update', (data) => {
      this.handleStateUpdate(data);
    });

    // Message acknowledgment
    this.socket.on('message:ack', (data) => {
      this.handleMessageAck(data.messageId);
    });

    // Setup custom event listeners
    this.setupCustomEventListeners();
  }

  private setupCustomEventListeners(): void {
    if (!this.socket) return;

    // Enhanced user events with state sync
    this.socket.on('user:connected', (data) => {
      this.updateSyncedState('users', data);
      this.emit('user:connected', data);
    });

    this.socket.on('user:disconnected', (data) => {
      this.removeSyncedState('users', data.userId);
      this.emit('user:disconnected', data);
    });

    // Enhanced message events with optimistic updates
    this.socket.on('message:new', (data) => {
      this.updateSyncedState('messages', data);
      this.emit('message:new', data);
    });

    this.socket.on('message:typing', (data) => {
      this.updateSyncedState('typing', data);
      this.emit('message:typing', data);
    });

    // Session events with context preservation
    this.socket.on('session:created', (data) => {
      this.updateSyncedState('sessions', data);
      this.emit('session:created', data);
    });

    this.socket.on('session:updated', (data) => {
      this.updateSyncedState('sessions', data);
      this.emit('session:updated', data);
    });

    // AI events with streaming support
    this.socket.on('ai:response', (data) => {
      this.emit('ai:response', data);
    });

    this.socket.on('ai:thinking', (data) => {
      this.updateSyncedState('aiStatus', data);
      this.emit('ai:thinking', data);
    });

    this.socket.on('ai:stream', (data) => {
      this.emit('ai:stream', data);
    });

    // System events
    this.socket.on('system:notification', (data) => {
      this.emit('system:notification', data);
    });
  }

  private handleSmartReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('❌ Max reconnection attempts reached');
      this.emit('connection:failed', { 
        maxAttempts: this.maxReconnectAttempts,
        totalFailures: this.reconnectAttempts,
        timestamp: new Date()
      });
      return;
    }

    this.reconnectAttempts++;
    
    // Calculate smart delay with exponential backoff and jitter
    let delay = Math.min(
      this.reconnectConfig.baseDelay * Math.pow(this.reconnectConfig.backoffFactor, this.reconnectAttempts - 1),
      this.reconnectConfig.maxDelay
    );

    // Add jitter to prevent thundering herd
    if (this.reconnectConfig.jitter) {
      delay += Math.random() * delay * 0.3;
    }

    // Adapt delay based on connection quality
    if (this.connectionMetrics.qualityScore < 50) {
      delay *= 1.5; // Slower reconnection for poor quality
    }

    this.currentReconnectDelay = delay;

    console.log(`🔄 Attempting smart reconnection ${this.reconnectAttempts}/${this.maxReconnectAttempts} in ${Math.round(delay)}ms`);
    
    setTimeout(() => {
      if (!this.socket?.connected) {
        this.connect();
      }
    }, delay);
  }

  private setupHeartbeat(): void {
    this.heartbeatInterval = setInterval(() => {
      if (this.socket?.connected) {
        this.socket.emit('ping', Date.now());
      }
    }, 30000); // Every 30 seconds
  }

  private setupLatencyMonitoring(): void {
    this.latencyCheckInterval = setInterval(() => {
      if (this.socket?.connected) {
        const start = Date.now();
        this.socket.emit('latency-check', start, (response: any) => {
          const latency = Date.now() - start;
          this.connectionMetrics.lastLatency = latency;
          this.updateAverageLatency(latency);
        });
      }
    }, 60000); // Every minute
  }

  private initializeQualityMonitoring(): void {
    this.qualityMonitorInterval = setInterval(() => {
      this.calculateQualityScore();
    }, 30000); // Every 30 seconds
  }

  // Smart message queuing with priority
  emit<K extends keyof WebSocketEvents>(event: K, data: WebSocketEvents[K], options?: {
    priority?: 'high' | 'medium' | 'low';
    maxRetries?: number;
    requireAck?: boolean;
  }): void {
    const { priority = 'medium', maxRetries = 3, requireAck = false } = options || {};
    
    const messageId = this.generateMessageId();
    const queuedMessage: QueuedMessage = {
      id: messageId,
      event: String(event),
      data,
      timestamp: Date.now(),
      priority,
      retryCount: 0,
      maxRetries
    };

    if (this.socket?.connected) {
      this.sendMessage(queuedMessage, requireAck);
    } else {
      // Queue message for later delivery
      this.addToQueue(queuedMessage);
    }
  }

  private sendMessage(message: QueuedMessage, requireAck: boolean = false): void {
    if (!this.socket?.connected) {
      this.addToQueue(message);
      return;
    }

    if (requireAck) {
      this.socket.emit(message.event, { 
        ...message.data, 
        _messageId: message.id 
      });
    } else {
      this.socket.emit(message.event, message.data);
    }
  }

  private addToQueue(message: QueuedMessage): void {
    // Remove existing message with same ID
    this.messageQueue = this.messageQueue.filter(m => m.id !== message.id);
    
    // Add to queue and sort by priority and timestamp
    this.messageQueue.push(message);
    this.messageQueue.sort((a, b) => {
      const priorityOrder = { high: 3, medium: 2, low: 1 };
      const priorityDiff = priorityOrder[b.priority] - priorityOrder[a.priority];
      return priorityDiff !== 0 ? priorityDiff : a.timestamp - b.timestamp;
    });

    // Limit queue size
    if (this.messageQueue.length > 100) {
      this.messageQueue = this.messageQueue.slice(0, 100);
    }
  }

  private processMessageQueue(): void {
    const toProcess = [...this.messageQueue];
    this.messageQueue = [];

    toProcess.forEach(message => {
      if (message.retryCount < message.maxRetries) {
        message.retryCount++;
        this.sendMessage(message);
      } else {
        console.warn(`Message ${message.id} dropped after ${message.maxRetries} retries`);
      }
    });
  }

  private handleMessageAck(messageId: string): void {
    this.messageQueue = this.messageQueue.filter(m => m.id !== messageId);
  }

  // State synchronization
  private updateSyncedState(key: string, data: any): void {
    this.syncedState.set(key, data);
    this.emit('state:changed' as any, { key, data });
  }

  private removeSyncedState(key: string, id?: string): void {
    if (id) {
      const current = this.syncedState.get(key);
      if (Array.isArray(current)) {
        const updated = current.filter(item => item.id !== id);
        this.syncedState.set(key, updated);
      }
    } else {
      this.syncedState.delete(key);
    }
    this.emit('state:changed' as any, { key, removed: true });
  }

  private handleStateSync(data: { key: string; value: any }): void {
    this.updateSyncedState(data.key, data.value);
  }

  private handleStateUpdate(data: { key: string; updates: any }): void {
    const current = this.syncedState.get(data.key);
    const updated = { ...current, ...data.updates };
    this.updateSyncedState(data.key, updated);
  }

  private syncPendingUpdates(): void {
    this.pendingUpdates.forEach((value, key) => {
      this.socket?.emit('state:update', { key, value });
    });
    this.pendingUpdates.clear();
  }

  // Public state access
  getSyncedState(key: string): any {
    return this.syncedState.get(key);
  }

  updateLocalState(key: string, value: any): void {
    if (this.socket?.connected) {
      this.socket.emit('state:update', { key, value });
    } else {
      this.pendingUpdates.set(key, value);
    }
    this.updateSyncedState(key, value);
  }

  // Event management
  on<K extends keyof WebSocketEvents>(
    event: K,
    callback: (data: WebSocketEvents[K]) => void
  ): void {
    if (!this.listeners.has(String(event))) {
      this.listeners.set(String(event), new Set());
    }
    this.listeners.get(String(event))!.add(callback);
  }

  off<K extends keyof WebSocketEvents>(
    event: K,
    callback?: (data: WebSocketEvents[K]) => void
  ): void {
    const eventListeners = this.listeners.get(String(event));
    if (eventListeners) {
      if (callback) {
        eventListeners.delete(callback);
      } else {
        eventListeners.clear();
      }
    }
  }

  private emit(event: string, data: any): void {
    const eventListeners = this.listeners.get(event);
    if (eventListeners) {
      eventListeners.forEach(callback => {
        try {
          callback(data);
        } catch (error) {
          console.error(`Error in event listener for ${event}:`, error);
        }
      });
    }
  }

  // Quality monitoring
  private updateAverageLatency(latency: number): void {
    this.connectionMetrics.avgLatency = 
      (this.connectionMetrics.avgLatency * 0.8) + (latency * 0.2);
  }

  private updateQualityScore(delta: number): void {
    this.connectionMetrics.qualityScore = Math.max(0, 
      Math.min(100, this.connectionMetrics.qualityScore + delta)
    );
  }

  private calculateQualityScore(): void {
    let score = 100;
    
    // Factor in latency
    if (this.connectionMetrics.avgLatency > 500) score -= 30;
    else if (this.connectionMetrics.avgLatency > 200) score -= 15;
    
    // Factor in connection stability
    const disconnectRate = this.connectionMetrics.totalDisconnections / 
      Math.max(1, this.connectionMetrics.totalConnections);
    score -= disconnectRate * 40;
    
    // Factor in reconnection attempts
    score -= Math.min(20, this.reconnectAttempts * 2);
    
    this.connectionMetrics.qualityScore = Math.max(0, Math.min(100, score));
  }

  private calculateAdaptiveTimeout(): number {
    if (!this.reconnectConfig.adaptiveTimeout) {
      return 20000; // Default
    }
    
    // Base timeout on quality score and average latency
    let timeout = 20000;
    
    if (this.connectionMetrics.qualityScore < 50) {
      timeout += 10000; // Add 10s for poor quality
    }
    
    if (this.connectionMetrics.avgLatency > 1000) {
      timeout += 15000; // Add 15s for high latency
    }
    
    return Math.min(60000, timeout); // Max 60s
  }

  private clearTimers(): void {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }
    if (this.latencyCheckInterval) {
      clearInterval(this.latencyCheckInterval);
      this.latencyCheckInterval = null;
    }
    if (this.qualityMonitorInterval) {
      clearInterval(this.qualityMonitorInterval);
      this.qualityMonitorInterval = null;
    }
  }

  // Utility methods
  private generateClientId(): string {
    return `client_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  private generateMessageId(): string {
    return `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  // Public API
  isConnected(): boolean {
    return this.socket?.connected || false;
  }

  getConnectionMetrics(): ConnectionMetrics {
    return { ...this.connectionMetrics };
  }

  getQueueSize(): number {
    return this.messageQueue.length;
  }

  getConnectionStatus(): {
    connected: boolean;
    reconnectAttempts: number;
    queueSize: number;
    metrics: ConnectionMetrics;
  } {
    return {
      connected: this.isConnected(),
      reconnectAttempts: this.reconnectAttempts,
      queueSize: this.getQueueSize(),
      metrics: this.getConnectionMetrics()
    };
  }

  // Configuration
  updateReconnectConfig(config: Partial<SmartReconnectConfig>): void {
    this.reconnectConfig = { ...this.reconnectConfig, ...config };
  }

  setMaxReconnectAttempts(attempts: number): void {
    this.maxReconnectAttempts = attempts;
  }
}

// Export singleton instance
export const enhancedWebSocketService = new EnhancedWebSocketService();