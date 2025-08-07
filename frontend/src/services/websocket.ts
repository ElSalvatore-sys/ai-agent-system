import { io, Socket } from 'socket.io-client';
import type { WebSocketEvents } from '@/types';

class WebSocketService {
  private socket: Socket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectInterval = 1000;
  private listeners: Map<string, Set<Function>> = new Map();

  constructor() {
    this.connect();
  }

  connect(): void {
    if (this.socket?.connected) {
      return;
    }

    const token = localStorage.getItem('authToken');
    const url = import.meta.env.VITE_WS_URL || 'ws://localhost:8000';

    this.socket = io(url, {
      auth: {
        token,
      },
      transports: ['websocket', 'polling'],
      timeout: 20000,
      forceNew: false,
      reconnection: true,
      reconnectionAttempts: this.maxReconnectAttempts,
      reconnectionDelay: this.reconnectInterval,
    });

    this.setupEventListeners();
  }

  disconnect(): void {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
    this.listeners.clear();
    this.reconnectAttempts = 0;
  }

  private setupEventListeners(): void {
    if (!this.socket) return;

    this.socket.on('connect', () => {
      console.log('✅ WebSocket connected');
      this.reconnectAttempts = 0;
      this.emit('connection:established', { timestamp: new Date() });
    });

    this.socket.on('disconnect', (reason) => {
      console.log('❌ WebSocket disconnected:', reason);
      this.emit('connection:lost', { reason, timestamp: new Date() });

      if (reason === 'io server disconnect') {
        // Server initiated disconnect, don't reconnect automatically
        return;
      }

      this.handleReconnect();
    });

    this.socket.on('connect_error', (error) => {
      console.error('WebSocket connection error:', error);
      this.emit('connection:error', { error: error.message, timestamp: new Date() });
      this.handleReconnect();
    });

    this.socket.on('reconnect', (attemptNumber) => {
      console.log(`🔄 WebSocket reconnected after ${attemptNumber} attempts`);
      this.reconnectAttempts = 0;
    });

    this.socket.on('reconnect_error', (error) => {
      console.error('WebSocket reconnection error:', error);
    });

    this.socket.on('reconnect_failed', () => {
      console.error('❌ WebSocket reconnection failed after maximum attempts');
      this.emit('connection:failed', { 
        maxAttempts: this.maxReconnectAttempts, 
        timestamp: new Date() 
      });
    });

    // Handle auth errors
    this.socket.on('auth:error', (data) => {
      console.error('WebSocket authentication error:', data);
      localStorage.removeItem('authToken');
      window.location.href = '/login';
    });

    // Set up listeners for custom events
    this.setupCustomEventListeners();
  }

  private setupCustomEventListeners(): void {
    if (!this.socket) return;

    // User events
    this.socket.on('user:connected', (data) => this.emit('user:connected', data));
    this.socket.on('user:disconnected', (data) => this.emit('user:disconnected', data));

    // Message events
    this.socket.on('message:new', (data) => this.emit('message:new', data));
    this.socket.on('message:typing', (data) => this.emit('message:typing', data));

    // Session events
    this.socket.on('session:created', (data) => this.emit('session:created', data));
    this.socket.on('session:updated', (data) => this.emit('session:updated', data));

    // AI events
    this.socket.on('ai:response', (data) => this.emit('ai:response', data));
    this.socket.on('ai:thinking', (data) => this.emit('ai:thinking', data));

    // System events
    this.socket.on('system:notification', (data) => this.emit('system:notification', data));
  }

  private handleReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('Max reconnection attempts reached');
      return;
    }

    this.reconnectAttempts++;
    const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);

    console.log(`🔄 Attempting to reconnect in ${delay}ms (attempt ${this.reconnectAttempts})`);

    setTimeout(() => {
      if (!this.socket?.connected) {
        this.connect();
      }
    }, delay);
  }

  // Public API
  isConnected(): boolean {
    return this.socket?.connected ?? false;
  }

  emit<K extends keyof WebSocketEvents>(event: K, data: WebSocketEvents[K]): void {
    if (!this.socket?.connected) {
      console.warn(`Cannot emit ${String(event)}: WebSocket not connected`);
      return;
    }

    this.socket.emit(event, data);
  }

  on<K extends keyof WebSocketEvents>(
    event: K,
    callback: (data: WebSocketEvents[K]) => void
  ): void {
    if (!this.listeners.has(String(event))) {
      this.listeners.set(String(event), new Set());
    }

    this.listeners.get(String(event))!.add(callback);

    // Also listen on the actual socket
    if (this.socket) {
      this.socket.on(String(event), callback);
    }
  }

  off<K extends keyof WebSocketEvents>(
    event: K,
    callback?: (data: WebSocketEvents[K]) => void
  ): void {
    const eventListeners = this.listeners.get(String(event));
    
    if (callback && eventListeners) {
      eventListeners.delete(callback);
      if (eventListeners.size === 0) {
        this.listeners.delete(String(event));
      }
    } else {
      this.listeners.delete(String(event));
    }

    // Also remove from actual socket
    if (this.socket) {
      if (callback) {
        this.socket.off(String(event), callback);
      } else {
        this.socket.removeAllListeners(String(event));
      }
    }
  }

  // Convenience methods for common events
  onUserConnected(callback: (user: WebSocketEvents['user:connected']) => void): void {
    this.on('user:connected', callback);
  }

  onUserDisconnected(callback: (data: WebSocketEvents['user:disconnected']) => void): void {
    this.on('user:disconnected', callback);
  }

  onNewMessage(callback: (message: WebSocketEvents['message:new']) => void): void {
    this.on('message:new', callback);
  }

  onTyping(callback: (data: WebSocketEvents['message:typing']) => void): void {
    this.on('message:typing', callback);
  }

  onAIResponse(callback: (response: WebSocketEvents['ai:response']) => void): void {
    this.on('ai:response', callback);
  }

  onSystemNotification(callback: (notification: WebSocketEvents['system:notification']) => void): void {
    this.on('system:notification', callback);
  }

  // Emit convenience methods
  sendTyping(isTyping: boolean): void {
    this.emit('message:typing', { userId: 'current-user', isTyping });
  }

  joinSession(sessionId: string): void {
    if (this.socket) {
      this.socket.emit('session:join', { sessionId });
    }
  }

  leaveSession(sessionId: string): void {
    if (this.socket) {
      this.socket.emit('session:leave', { sessionId });
    }
  }

  // Get connection status
  getConnectionStatus(): {
    connected: boolean;
    reconnectAttempts: number;
    lastError?: string;
  } {
    return {
      connected: this.isConnected(),
      reconnectAttempts: this.reconnectAttempts,
    };
  }
}

export const webSocketService = new WebSocketService();
export default webSocketService;