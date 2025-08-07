import React, { useState, useEffect, useRef } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Bot, Users } from 'lucide-react';
import { Card, Loading, Button } from '@/components/ui';
import { ChatMessage, ChatInput } from '@/components/features';
import { useAI, useChatWebSocket } from '@/hooks';
import { apiService } from '@/services/api';
import { generateId } from '@/utils/helpers';
import type { ChatSession, Message } from '@/types';

export const Chat: React.FC = () => {
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isAIThinking, setIsAIThinking] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const queryClient = useQueryClient();

  // Hooks
  const { sendMessage: sendAIMessage, isLoading: aiLoading, currentModel } = useAI();
  const { 
    isConnected, 
    typingUsers, 
    onlineUsers, 
    sendTyping, 
    on, 
    off 
  } = useChatWebSocket(currentSessionId || undefined);

  // Fetch chat sessions
  const { data: sessionsData, isLoading: sessionsLoading } = useQuery({
    queryKey: ['chat-sessions'],
    queryFn: () => apiService.getChatSessions(),
  });

  // Fetch messages for current session
  const { data: messagesData, isLoading: messagesLoading } = useQuery({
    queryKey: ['messages', currentSessionId],
    queryFn: () => currentSessionId ? apiService.getMessages(currentSessionId) : null,
    enabled: !!currentSessionId,
  });

  // Create new chat session mutation
  const createSessionMutation = useMutation({
    mutationFn: (title: string) => apiService.createChatSession(title),
    onSuccess: (response) => {
      const newSession = response.data;
      if (newSession) {
        setCurrentSessionId(newSession.id);
        queryClient.invalidateQueries({ queryKey: ['chat-sessions'] });
      }
    },
  });

  // Send message mutation
  const sendMessageMutation = useMutation({
    mutationFn: (data: { sessionId: string; content: string }) =>
      apiService.sendMessage(data.sessionId, data.content),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['messages', currentSessionId] });
    },
  });

  // Set up real-time message listeners
  useEffect(() => {
    const handleNewMessage = (message: Message) => {
      setMessages(prev => [...prev, message]);
      queryClient.invalidateQueries({ queryKey: ['messages', currentSessionId] });
    };

    const handleAIThinking = (data: { sessionId: string }) => {
      if (data.sessionId === currentSessionId) {
        setIsAIThinking(true);
      }
    };

    const handleAIResponse = () => {
      setIsAIThinking(false);
    };

    on('message:new', handleNewMessage);
    on('ai:thinking', handleAIThinking);
    on('ai:response', handleAIResponse);

    return () => {
      off('message:new', handleNewMessage);
      off('ai:thinking', handleAIThinking);
      off('ai:response', handleAIResponse);
    };
  }, [currentSessionId, on, off, queryClient]);

  // Update messages when data changes
  useEffect(() => {
    if (messagesData?.data) {
      setMessages(messagesData.data);
    }
  }, [messagesData]);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isAIThinking]);

  // Auto-select first session or create one
  useEffect(() => {
    if (sessionsData?.data && sessionsData.data.length > 0 && !currentSessionId) {
      setCurrentSessionId(sessionsData.data[0].id);
    }
  }, [sessionsData, currentSessionId]);

  const handleNewChat = async () => {
    const title = `Chat ${new Date().toLocaleString()}`;
    createSessionMutation.mutate(title);
  };

  const handleSendMessage = async (content: string) => {
    if (!currentSessionId) {
      // Create new session if none exists
      await handleNewChat();
      return;
    }

    // Create optimistic user message
    const userMessage: Message = {
      id: generateId(),
      content,
      senderId: 'current-user',
      timestamp: new Date(),
      type: 'text',
    };

    // Add user message immediately
    setMessages(prev => [...prev, userMessage]);

    try {
      // Send message to API
      await sendMessageMutation.mutateAsync({
        sessionId: currentSessionId,
        content,
      });

      // Send to AI
      setIsAIThinking(true);
      await sendAIMessage(content, currentModel?.id, currentSessionId);
      
    } catch (error) {
      console.error('Failed to send message:', error);
      // Remove optimistic message on error
      setMessages(prev => prev.filter(msg => msg.id !== userMessage.id));
    }
  };

  const handleTyping = (isTyping: boolean) => {
    sendTyping(isTyping);
  };

  const sessions = sessionsData?.data || [];

  if (sessionsLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loading size="lg" text="Loading chats..." />
      </div>
    );
  }

  return (
    <div className="flex h-[calc(100vh-8rem)]">
      {/* Sessions Sidebar */}
      <div className="w-80 border-r border-gray-200 bg-white">
        <div className="p-4 border-b border-gray-200">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold text-gray-900">Chats</h2>
            <Button size="sm" onClick={handleNewChat} disabled={createSessionMutation.isPending}>
              New Chat
            </Button>
          </div>
        </div>
        
        <div className="overflow-y-auto h-full p-4 space-y-2">
          {sessions.map((session: ChatSession) => (
            <button
              key={session.id}
              onClick={() => setCurrentSessionId(session.id)}
              className={`
                w-full text-left p-3 rounded-lg transition-colors
                ${session.id === currentSessionId 
                  ? 'bg-blue-50 border border-blue-200' 
                  : 'hover:bg-gray-50 border border-transparent'
                }
              `}
            >
              <div className="flex items-center gap-2 mb-1">
                <Bot className="h-4 w-4 text-gray-400" />
                <span className="font-medium text-gray-900 truncate">
                  {session.title}
                </span>
              </div>
              <p className="text-sm text-gray-500">
                {session.messages.length} messages
              </p>
            </button>
          ))}
          
          {sessions.length === 0 && (
            <div className="text-center py-8 text-gray-500">
              <Bot className="h-12 w-12 mx-auto mb-4 text-gray-300" />
              <p>No chats yet</p>
              <p className="text-sm">Start a new conversation</p>
            </div>
          )}
        </div>
      </div>

      {/* Chat Area */}
      <div className="flex-1 flex flex-col">
        {currentSessionId ? (
          <>
            {/* Chat Header */}
            <div className="bg-white border-b border-gray-200 p-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <h3 className="text-lg font-semibold text-gray-900">
                    {sessions.find(s => s.id === currentSessionId)?.title || 'Chat'}
                  </h3>
                  {currentModel && (
                    <span className="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded-full">
                      {currentModel.name}
                    </span>
                  )}
                </div>
                
                <div className="flex items-center gap-4 text-sm text-gray-500">
                  {isConnected && (
                    <div className="flex items-center gap-1">
                      <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                      Connected
                    </div>
                  )}
                  
                  {onlineUsers.length > 0 && (
                    <div className="flex items-center gap-1">
                      <Users className="h-4 w-4" />
                      {onlineUsers.length} online
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
              {messagesLoading ? (
                <div className="flex items-center justify-center h-32">
                  <Loading text="Loading messages..." />
                </div>
              ) : (
                <>
                  {messages.map((message) => (
                    <ChatMessage
                      key={message.id}
                      message={message}
                      isOwn={message.senderId === 'current-user'}
                    />
                  ))}
                  
                  {isAIThinking && (
                    <div className="flex gap-3">
                      <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
                        <Bot className="h-4 w-4 text-blue-600" />
                      </div>
                      <div className="bg-gray-100 rounded-2xl rounded-bl-md px-4 py-2">
                        <div className="flex space-x-1">
                          <div className="w-2 h-2 bg-gray-400 rounded-full animate-pulse"></div>
                          <div className="w-2 h-2 bg-gray-400 rounded-full animate-pulse" style={{ animationDelay: '0.2s' }}></div>
                          <div className="w-2 h-2 bg-gray-400 rounded-full animate-pulse" style={{ animationDelay: '0.4s' }}></div>
                        </div>
                      </div>
                    </div>
                  )}
                  
                  {typingUsers.length > 0 && (
                    <div className="text-sm text-gray-500 italic">
                      {typingUsers.join(', ')} {typingUsers.length === 1 ? 'is' : 'are'} typing...
                    </div>
                  )}
                  
                  <div ref={messagesEndRef} />
                </>
              )}
            </div>

            {/* Chat Input */}
            <ChatInput
              onSendMessage={handleSendMessage}
              onTyping={handleTyping}
              disabled={aiLoading || sendMessageMutation.isPending}
              placeholder={`Message ${currentModel?.name || 'AI'}...`}
            />
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center">
            <Card className="p-8 text-center max-w-md">
              <Bot className="h-16 w-16 mx-auto mb-4 text-gray-300" />
              <h3 className="text-lg font-semibold text-gray-900 mb-2">
                Welcome to AI Chat
              </h3>
              <p className="text-gray-600 mb-4">
                Start a new conversation or select an existing chat from the sidebar.
              </p>
              <Button onClick={handleNewChat} disabled={createSessionMutation.isPending}>
                Start New Chat
              </Button>
            </Card>
          </div>
        )}
      </div>
    </div>
  );
};