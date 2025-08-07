import { useEffect, useState } from 'react';
import { useWebSocket } from '@/hooks/useWebSocket';
import { WebSocketEvents, Message } from '@/types';

export interface PresenceData {
  userId: string;
  name?: string;
  isTyping?: boolean;
  cursor?: { x: number; y: number };
  lastSeen: Date;
}

export interface UseCollaborationReturn {
  presenceMap: Record<string, PresenceData>;
  typingUsers: string[];
  sendTyping: (isTyping: boolean) => void;
  broadcastCursor: (pos: { x: number; y: number }) => void;
}

export function useCollaboration(sessionId: string | undefined): UseCollaborationReturn {
  const { emit, on, off, isConnected } = useWebSocket();
  const [presenceMap, setPresenceMap] = useState<Record<string, PresenceData>>({});
  const [typingUsers, setTypingUsers] = useState<string[]>([]);

  /* --------------------------- JOIN / LEAVE ROOM -------------------------- */
  useEffect(() => {
    if (!sessionId || !isConnected) return;
    (emit as any)('session:join', { sessionId });
    return () => {
      (emit as any)('session:leave', { sessionId });
    };
  }, [sessionId, isConnected]);

  /* ----------------------------- USER CONNECT ----------------------------- */
  useEffect(() => {
    const handleUserConnected = (user: WebSocketEvents['user:connected']) => {
      setPresenceMap((prev) => ({ ...prev, [user.id]: { ...user, lastSeen: new Date() } }));
    };
    const handleUserDisconnected = (data: WebSocketEvents['user:disconnected']) => {
      setPresenceMap((prev) => {
        const copy = { ...prev };
        delete copy[data.userId];
        return copy;
      });
    };
    on('user:connected', handleUserConnected);
    on('user:disconnected', handleUserDisconnected);
    return () => {
      off('user:connected', handleUserConnected);
      off('user:disconnected', handleUserDisconnected);
    };
  }, [on, off]);

  /* ------------------------------ TYPING ---------------------------------- */
  useEffect(() => {
    const handleTyping = (data: WebSocketEvents['message:typing']) => {
      const { userId, isTyping } = data;
      setTypingUsers((prev) => {
        if (isTyping && !prev.includes(userId)) return [...prev, userId];
        if (!isTyping) return prev.filter((id) => id !== userId);
        return prev;
      });
    };
    on('message:typing', handleTyping);
    return () => off('message:typing', handleTyping);
  }, [on, off]);

  const sendTyping = (isTyping: boolean) => (emit as any)('message:typing', { sessionId, isTyping });

  /* ----------------------------- CURSOR ----------------------------------- */
  useEffect(() => {
    const handleCursor = (data: { userId: string; x: number; y: number }) => {
      setPresenceMap((prev) => ({
        ...prev,
        [data.userId]: {
          ...(prev[data.userId] || { userId: data.userId, lastSeen: new Date() }),
          cursor: { x: data.x, y: data.y },
        },
      }));
    };
    // eslint-disable-next-line @typescript-eslint/ban-ts-comment
    // @ts-expect-error - custom event not yet in shared types
    on('user:cursor', handleCursor);
    // eslint-disable-next-line @typescript-eslint/ban-ts-comment
    // @ts-expect-error - custom event not yet in shared types
    return () => off('user:cursor', handleCursor);
  }, [on, off]);

  const broadcastCursor = (pos: { x: number; y: number }) => (emit as any)('user:cursor', pos);

  return { presenceMap, typingUsers, sendTyping, broadcastCursor };
}
