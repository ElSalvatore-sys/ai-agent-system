import React, { createContext, useContext, useReducer, useEffect, ReactNode } from 'react';
import { ChatSession, Message } from '@/types';
import { generateId } from '@/utils/helpers';

interface ConversationsState {
  sessions: Record<string, ChatSession>;
  activeSessionId: string | null;
}

type ConversationsAction =
  | { type: 'CREATE_SESSION'; payload: Partial<ChatSession> }
  | { type: 'SET_ACTIVE_SESSION'; payload: string }
  | { type: 'ADD_MESSAGE'; payload: { sessionId: string; message: Message } }
  | { type: 'UPDATE_SESSION_MODEL'; payload: { sessionId: string; modelId: string } }
  | { type: 'LOAD_SESSIONS'; payload: ChatSession[] };

const ConversationsContext = createContext<{
  state: ConversationsState;
  dispatch: React.Dispatch<ConversationsAction>;
} | undefined>(undefined);

const conversationsReducer = (state: ConversationsState, action: ConversationsAction): ConversationsState => {
  switch (action.type) {
    case 'CREATE_SESSION':
      const newSession: ChatSession = {
        id: generateId(),
        title: 'New Conversation',
        messages: [],
        createdAt: new Date(),
        updatedAt: new Date(),
        isActive: true,
        userId: 'local-user',
        ...action.payload,
      };
      return {
        ...state,
        sessions: { ...state.sessions, [newSession.id]: newSession },
        activeSessionId: newSession.id,
      };
    case 'SET_ACTIVE_SESSION':
      return { ...state, activeSessionId: action.payload };
    case 'ADD_MESSAGE':
      const { sessionId, message } = action.payload;
      const session = state.sessions[sessionId];
      if (session) {
        const updatedSession = {
          ...session,
          messages: [...session.messages, message],
          updatedAt: new Date(),
        };
        return {
          ...state,
          sessions: { ...state.sessions, [sessionId]: updatedSession },
        };
      }
      return state;
    case 'UPDATE_SESSION_MODEL': {
      const { sessionId, modelId } = action.payload;
      const session = state.sessions[sessionId];
      if (session) {
        const updatedSession = { ...session, modelId };
        return {
          ...state,
          sessions: { ...state.sessions, [sessionId]: updatedSession },
        };
      }
      return state;
    }
    case 'LOAD_SESSIONS':
      const sessions = action.payload.reduce((acc, session) => {
        acc[session.id] = session;
        return acc;
      }, {} as Record<string, ChatSession>);
      return {
        ...state,
        sessions,
        activeSessionId: state.activeSessionId || action.payload[0]?.id || null,
      };
    default:
      return state;
  }
};

export const ConversationsProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [state, dispatch] = useReducer(conversationsReducer, {
    sessions: {},
    activeSessionId: null,
  });

  useEffect(() => {
    const savedState = localStorage.getItem('conversationsState');
    if (savedState) {
      const parsedState = JSON.parse(savedState);
      dispatch({ type: 'LOAD_SESSIONS', payload: Object.values(parsedState.sessions) });
    }
  }, []);

  useEffect(() => {
    localStorage.setItem('conversationsState', JSON.stringify(state));
  }, [state]);

  return (
    <ConversationsContext.Provider value={{ state, dispatch }}>
      {children}
    </ConversationsContext.Provider>
  );
};

export const useConversations = () => {
  const context = useContext(ConversationsContext);
  if (!context) {
    throw new Error('useConversations must be used within a ConversationsProvider');
  }
  return context;
};