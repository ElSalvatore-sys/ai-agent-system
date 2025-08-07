import axios, { AxiosInstance, AxiosResponse } from 'axios';
import type {
  ApiResponse,
  PaginatedResponse,
  User,
  ChatSession,
  Message,
  AIModel,
  Analytics,
  Settings,
} from '@/types';

class ApiService {
  private api: AxiosInstance;

  constructor() {
    // Use Vite environment variables instead of process.env
    const baseURL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';
    
    console.log('🔧 API Service initialized:', {
      baseURL,
      mode: import.meta.env.MODE,
      dev: import.meta.env.DEV
    });

    this.api = axios.create({
      baseURL,
      timeout: 10000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    this.setupInterceptors();
  }

  private setupInterceptors(): void {
    // Request interceptor for auth token
    this.api.interceptors.request.use(
      (config) => {
        const token = localStorage.getItem('authToken');
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor for error handling
    this.api.interceptors.response.use(
      (response: AxiosResponse) => response,
      (error) => {
        if (error.response?.status === 401) {
          localStorage.removeItem('authToken');
          window.location.href = '/login';
        }
        return Promise.reject(error);
      }
    );
  }

  // Auth endpoints
  async login(email: string, password: string): Promise<ApiResponse<{ user: User; token: string }>> {
    const response = await this.api.post('/auth/login', { email, password });
    return response.data;
  }

  async register(userData: { name: string; email: string; password: string }): Promise<ApiResponse<User>> {
    const response = await this.api.post('/auth/register', userData);
    return response.data;
  }

  async logout(): Promise<ApiResponse> {
    const response = await this.api.post('/auth/logout');
    localStorage.removeItem('authToken');
    return response.data;
  }

  async refreshToken(): Promise<ApiResponse<{ token: string }>> {
    const response = await this.api.post('/auth/refresh');
    return response.data;
  }

  // User endpoints
  async getCurrentUser(): Promise<ApiResponse<User>> {
    const response = await this.api.get('/users/me');
    return response.data;
  }

  async updateProfile(userData: Partial<User>): Promise<ApiResponse<User>> {
    const response = await this.api.put('/users/me', userData);
    return response.data;
  }

  async getUsers(): Promise<ApiResponse<User[]>> {
    const response = await this.api.get('/users');
    return response.data;
  }

  // Chat session endpoints
  async getChatSessions(page = 1, limit = 20): Promise<PaginatedResponse<ChatSession>> {
    const response = await this.api.get('/chat/sessions', {
      params: { page, limit },
    });
    return response.data;
  }

  async getChatSession(sessionId: string): Promise<ApiResponse<ChatSession>> {
    const response = await this.api.get(`/chat/sessions/${sessionId}`);
    return response.data;
  }

  async createChatSession(title: string): Promise<ApiResponse<ChatSession>> {
    const response = await this.api.post('/chat/sessions', { title });
    return response.data;
  }

  async updateChatSession(sessionId: string, data: Partial<ChatSession>): Promise<ApiResponse<ChatSession>> {
    const response = await this.api.put(`/chat/sessions/${sessionId}`, data);
    return response.data;
  }

  async deleteChatSession(sessionId: string): Promise<ApiResponse> {
    const response = await this.api.delete(`/chat/sessions/${sessionId}`);
    return response.data;
  }

  // Message endpoints
  async getMessages(sessionId: string, page = 1, limit = 50): Promise<PaginatedResponse<Message>> {
    const response = await this.api.get(`/chat/sessions/${sessionId}/messages`, {
      params: { page, limit },
    });
    return response.data;
  }

  async sendMessage(sessionId: string, content: string): Promise<ApiResponse<Message>> {
    const response = await this.api.post(`/chat/sessions/${sessionId}/messages`, {
      content,
    });
    return response.data;
  }

  async updateMessage(messageId: string, content: string): Promise<ApiResponse<Message>> {
    const response = await this.api.put(`/chat/messages/${messageId}`, {
      content,
    });
    return response.data;
  }

  async deleteMessage(messageId: string): Promise<ApiResponse> {
    const response = await this.api.delete(`/chat/messages/${messageId}`);
    return response.data;
  }

  // AI model endpoints
  async getAIModels(): Promise<ApiResponse<AIModel[]>> {
    const response = await this.api.get('/ai/models');
    return response.data;
  }

  async sendAIMessage(
    sessionId: string,
    message: string,
    modelId?: string,
    history?: Message[],
    options?: { temperature?: number; maxTokens?: number }
  ): Promise<ApiResponse<Message>> {
    const response = await this.api.post('/ai/chat', {
      sessionId,
      message,
      modelId,
      history,
      ...options,
    });
    return response.data;
  }

  async getModelUsage(): Promise<ApiResponse<Record<string, number>>> {
    const response = await this.api.get('/ai/usage');
    return response.data;
  }

  // Analytics endpoints
  async getAnalytics(dateRange?: { from: Date; to: Date }): Promise<ApiResponse<Analytics>> {
    const params = dateRange
      ? {
          from: dateRange.from.toISOString(),
          to: dateRange.to.toISOString(),
        }
      : {};

    const response = await this.api.get('/analytics', { params });
    return response.data;
  }

  async getDashboardStats(): Promise<ApiResponse<{
    totalSessions: number;
    totalMessages: number;
    activeUsers: number;
    systemHealth: number;
  }>> {
    const response = await this.api.get('/analytics/dashboard');
    return response.data;
  }

  // Settings endpoints
  async getSettings(): Promise<ApiResponse<Settings>> {
    const response = await this.api.get('/settings');
    return response.data;
  }

  async updateSettings(settings: Partial<Settings>): Promise<ApiResponse<Settings>> {
    const response = await this.api.put('/settings', settings);
    return response.data;
  }

  // File upload
  async uploadFile(file: File, type: 'avatar' | 'attachment'): Promise<ApiResponse<{ url: string }>> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('type', type);

    const response = await this.api.post('/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  }

  // Search
  async search(query: string, type: 'messages' | 'sessions' | 'all' = 'all'): Promise<ApiResponse<{
    messages: Message[];
    sessions: ChatSession[];
  }>> {
    const response = await this.api.get('/search', {
      params: { query, type },
    });
    return response.data;
  }

  // Health check
  async healthCheck(): Promise<ApiResponse<{ status: string; timestamp: Date }>> {
    const response = await this.api.get('/health');
    return response.data;
  }
}

export const apiService = new ApiService();
export default apiService;