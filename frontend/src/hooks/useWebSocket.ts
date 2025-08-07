import { useState, useEffect, useCallback, useRef } from 'react';
import { webSocketService } from '@/services/websocket';
import type { UseWebSocketReturn, WebSocketEvents } from '@/types';

export function useWebSocket(): UseWebSocketReturn {
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const listenersRef = useRef<Map<string, Set<Function>>>(new Map());

  useEffect(() => {
    // Set up connection status listeners
    const handleConnect = () => {
      setIsConnected(true);
      setError(null);
    };

    const handleDisconnect = () => {
      setIsConnected(false);
    };

    const handleError = (data: { error: string }) => {
      setError(data.error);
    };

    const handleConnectionFailed = () => {
      setError('Failed to connect to server');
    };

    // Listen to connection events
    webSocketService.on('connection:established', handleConnect);
    webSocketService.on('connection:lost', handleDisconnect);
    webSocketService.on('connection:error', handleError);
    webSocketService.on('connection:failed', handleConnectionFailed);

    // Set initial connection state
    setIsConnected(webSocketService.isConnected());

    return () => {
      webSocketService.off('connection:established', handleConnect);
      webSocketService.off('connection:lost', handleDisconnect);
      webSocketService.off('connection:error', handleError);
      webSocketService.off('connection:failed', handleConnectionFailed);
    };
  }, []);

  const emit = useCallback(<K extends keyof WebSocketEvents>(
    event: K,
    data: WebSocketEvents[K]
  ) => {
    if (!isConnected) {
      console.warn(`Cannot emit ${String(event)}: WebSocket not connected`);
      return;
    }
    webSocketService.emit(event, data);
  }, [isConnected]);

  const on = useCallback(<K extends keyof WebSocketEvents>(
    event: K,
    callback: (data: WebSocketEvents[K]) => void
  ) => {
    // Store the listener locally
    const eventName = String(event);
    if (!listenersRef.current.has(eventName)) {
      listenersRef.current.set(eventName, new Set());
    }
    listenersRef.current.get(eventName)!.add(callback);

    // Register with the service
    webSocketService.on(event, callback);
  }, []);

  const off = useCallback(<K extends keyof WebSocketEvents>(
    event: K,
    callback?: (data: WebSocketEvents[K]) => void
  ) => {
    const eventName = String(event);
    const listeners = listenersRef.current.get(eventName);

    if (callback && listeners) {
      listeners.delete(callback);
      if (listeners.size === 0) {
        listenersRef.current.delete(eventName);
      }
    } else {
      listenersRef.current.delete(eventName);
    }

    webSocketService.off(event, callback);
  }, []);

  const connect = useCallback(() => {
    webSocketService.connect();
  }, []);

  const disconnect = useCallback(() => {
    // Clean up all local listeners
    listenersRef.current.clear();
    webSocketService.disconnect();
    setIsConnected(false);
  }, []);

  return {
    isConnected,
    error,
    emit,
    on,
    off,
    connect,
    disconnect,
  };
}

// Specialized hook for chat functionality
export function useChatWebSocket(sessionId?: string) {
  const { isConnected, emit, on, off } = useWebSocket();
  const [typingUsers, setTypingUsers] = useState<Set<string>>(new Set());
  const [onlineUsers, setOnlineUsers] = useState<Set<string>>(new Set());

  useEffect(() => {
    if (!sessionId) return;

    // Join the session room
    if (isConnected) {
      webSocketService.joinSession(sessionId);
    }

    return () => {
      if (sessionId) {
        webSocketService.leaveSession(sessionId);
      }
    };
  }, [sessionId, isConnected]);

  useEffect(() => {
    const handleUserConnected = (user: WebSocketEvents['user:connected']) => {
      setOnlineUsers(prev => new Set([...prev, user.id]));
    };

    const handleUserDisconnected = (data: WebSocketEvents['user:disconnected']) => {
      setOnlineUsers(prev => {
        const next = new Set(prev);
        next.delete(data.userId);
        return next;
      });
      setTypingUsers(prev => {
        const next = new Set(prev);
        next.delete(data.userId);
        return next;
      });
    };

    const handleTyping = (data: WebSocketEvents['message:typing']) => {
      setTypingUsers(prev => {
        const next = new Set(prev);
        if (data.isTyping) {
          next.add(data.userId);
        } else {
          next.delete(data.userId);
        }
        return next;
      });
    };

    on('user:connected', handleUserConnected);
    on('user:disconnected', handleUserDisconnected);
    on('message:typing', handleTyping);

    return () => {
      off('user:connected', handleUserConnected);
      off('user:disconnected', handleUserDisconnected);
      off('message:typing', handleTyping);
    };
  }, [on, off]);

  const sendTyping = useCallback((isTyping: boolean) => {
    webSocketService.sendTyping(isTyping);
  }, []);

  const sendMessage = useCallback((content: string) => {
    if (!sessionId) return;
    
    emit('message:new', {
      id: Math.random().toString(36).substring(2),
      content,
      senderId: 'current-user', // Get from auth context
      timestamp: new Date(),
      type: 'text',
    });
  }, [sessionId, emit]);

  return {
    isConnected,
    typingUsers: Array.from(typingUsers),
    onlineUsers: Array.from(onlineUsers),
    sendTyping,
    sendMessage,
    on,
    off,
  };
}

// Hook for real-time notifications
export function useNotifications() {
  const { on, off } = useWebSocket();
  const [notifications, setNotifications] = useState<WebSocketEvents['system:notification'][]>([]);

  useEffect(() => {
    const handleNotification = (notification: WebSocketEvents['system:notification']) => {
      setNotifications(prev => [notification, ...prev].slice(0, 50)); // Keep last 50 notifications
    };

    on('system:notification', handleNotification);

    return () => {
      off('system:notification', handleNotification);
    };
  }, [on, off]);

  const clearNotification = useCallback((id: string) => {
    setNotifications(prev => prev.filter(n => n.id !== id));
  }, []);

  const clearAllNotifications = useCallback(() => {
    setNotifications([]);
  }, []);

  return {
    notifications,
    clearNotification,
    clearAllNotifications,
  };
}

// Hook for connection monitoring
export function useConnectionStatus() {
  const [status, setStatus] = useState(webSocketService.getConnectionStatus());

  useEffect(() => {
    const updateStatus = () => {
      setStatus(webSocketService.getConnectionStatus());
    };

    const interval = setInterval(updateStatus, 1000);

    return () => clearInterval(interval);
  }, []);

  return status;
}