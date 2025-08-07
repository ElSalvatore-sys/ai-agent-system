import { useState, useEffect, useCallback } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiService } from '@/services/api';
import { webSocketService } from '@/services/websocket';
import type { AIModel, AIResponse, UseAIReturn } from '@/types';

export function useAI(): UseAIReturn {
  const [currentModel, setCurrentModel] = useState<AIModel | null>(null);
  const [error, setError] = useState<string | null>(null);
  const queryClient = useQueryClient();

  // Fetch available AI models
  const { data: modelsResponse, isLoading: modelsLoading } = useQuery({
    queryKey: ['ai-models'],
    queryFn: () => apiService.getAIModels(),
    staleTime: 5 * 60 * 1000, // 5 minutes
    retry: 2,
  });

  const availableModels = modelsResponse?.data || [];

  // Set default model when models are loaded
  useEffect(() => {
    if (availableModels.length > 0 && !currentModel) {
      const defaultModel = availableModels.find((model: AIModel) => model.name.includes('gpt'))
        || availableModels[0];
      setCurrentModel(defaultModel);
    }
  }, [availableModels, currentModel]);

  // Send message mutation
  const sendMessageMutation = useMutation({
    mutationFn: async (params: {
      sessionId: string;
      message: string;
      modelId?: string;
      history?: Message[];
      options?: { temperature?: number; maxTokens?: number };
    }) => {
      const response = await apiService.sendAIMessage(
        params.sessionId,
        params.message,
        params.modelId || currentModel?.id,
        params.history,
        params.options
      );
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['chat-sessions'] });
      queryClient.invalidateQueries({ queryKey: ['messages'] });
      setError(null);
    },
    onError: (error: any) => {
      const errorMessage = error.response?.data?.message || error.message || 'Failed to send message';
      setError(errorMessage);
      console.error('AI message error:', error);
    },
  });

  const sendParallelMessages = useMutation({
    mutationFn: async (params: {
      sessionId: string;
      message: string;
      modelIds: string[];
      options?: { temperature?: number; maxTokens?: number };
    }) => {
      const promises = params.modelIds.map(modelId =>
        apiService.sendAIMessage(
          params.sessionId,
          params.message,
          modelId,
          params.options
        )
      );
      const responses = await Promise.all(promises);
      return responses.map(res => res.data);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['chat-sessions'] });
      queryClient.invalidateQueries({ queryKey: ['messages'] });
      setError(null);
    },
    onError: (error: any) => {
      const errorMessage = error.response?.data?.message || error.message || 'Failed to send parallel messages';
      setError(errorMessage);
      console.error('AI parallel message error:', error);
    },
  });

  // WebSocket listeners for real-time AI responses
  useEffect(() => {
    const handleAIResponse = (response: AIResponse) => {
      // Update queries with new response
      queryClient.invalidateQueries({ queryKey: ['messages'] });
    };

    const handleAIThinking = (data: { sessionId: string }) => {
      // You can use this to show typing indicators
      console.log('AI is thinking for session:', data.sessionId);
    };

    webSocketService.onAIResponse(handleAIResponse);
    webSocketService.on('ai:thinking', handleAIThinking);

    return () => {
      webSocketService.off('ai:response', handleAIResponse);
      webSocketService.off('ai:thinking', handleAIThinking);
    };
  }, [queryClient]);

  // Send message function
  const sendMessage = useCallback(async (
    message: string,
    modelId?: string,
    sessionId?: string,
    history?: Message[],
    options?: { temperature?: number; maxTokens?: number }
  ): Promise<AIResponse> => {
    if (!message.trim()) {
      throw new Error('Message cannot be empty');
    }

    if (!currentModel && !modelId) {
      throw new Error('No AI model selected');
    }

    // Use current active session or create a new one
    const activeSessionId = sessionId || 'current-session'; // You might want to get this from context/state

    try {
      const result = await sendMessageMutation.mutateAsync({
        sessionId: activeSessionId,
        message: message.trim(),
        modelId: modelId || currentModel?.id,
        history: history,
        options,
      });

      // Convert to AIResponse format
      const aiResponse: AIResponse = {
        content: result.content,
        tokens: 0, // You might want to extract this from the response
        model: modelId || currentModel?.name || 'unknown',
        timestamp: new Date(result.timestamp),
        confidence: undefined, // Add if your API provides this
      };

      return aiResponse;
    } catch (error: any) {
      const errorMessage = error.response?.data?.message || error.message || 'Failed to send message';
      setError(errorMessage);
      throw new Error(errorMessage);
    }
  }, [currentModel, sendMessageMutation]);

  // Set model function
  const setModel = useCallback((modelId: string) => {
    const model = availableModels.find((m: AIModel) => m.id === modelId);
    if (model) {
      setCurrentModel(model);
      setError(null);
    } else {
      setError(`Model with ID ${modelId} not found`);
    }
  }, [availableModels]);

  // Clear error function
  const clearError = useCallback(() => {
    setError(null);
  }, []);

  return {
    sendMessage,
    sendParallelMessages: sendParallelMessages.mutateAsync,
    isLoading: sendMessageMutation.isPending || sendParallelMessages.isPending || modelsLoading,
    error,
    currentModel,
    availableModels,
    setModel,
    clearError,
  };
}

// Additional hook for streaming AI responses
export function useAIStream() {
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamContent, setStreamContent] = useState('');
  const [error, setError] = useState<string | null>(null);

  const startStream = useCallback(async (
    message: string,
    onChunk: (chunk: string) => void,
    onComplete: (fullResponse: string) => void,
    onError: (error: string) => void
  ) => {
    setIsStreaming(true);
    setStreamContent('');
    setError(null);

    try {
      // This would implement Server-Sent Events or WebSocket streaming
      // For now, we'll simulate streaming
      const response = await apiService.sendAIMessage('current-session', message);
      const fullContent = response.data?.content || '';
      
      // Simulate streaming by sending chunks
      let currentContent = '';
      const words = fullContent.split(' ');
      
      for (let i = 0; i < words.length; i++) {
        await new Promise(resolve => setTimeout(resolve, 50)); // Simulate delay
        currentContent += (i > 0 ? ' ' : '') + words[i];
        setStreamContent(currentContent);
        onChunk(currentContent);
      }
      
      onComplete(fullContent);
    } catch (error: any) {
      const errorMessage = error.response?.data?.message || error.message || 'Streaming failed';
      setError(errorMessage);
      onError(errorMessage);
    } finally {
      setIsStreaming(false);
    }
  }, []);

  const stopStream = useCallback(() => {
    setIsStreaming(false);
  }, []);

  return {
    isStreaming,
    streamContent,
    error,
    startStream,
    stopStream,
  };
}

// Hook for AI model usage analytics
export function useAIAnalytics() {
  const { data: usageData, isLoading } = useQuery({
    queryKey: ['ai-usage'],
    queryFn: () => apiService.getModelUsage(),
    staleTime: 10 * 60 * 1000, // 10 minutes
    refetchInterval: 5 * 60 * 1000, // Refetch every 5 minutes
  });

  return {
    usage: usageData?.data || {},
    isLoading,
  };
}