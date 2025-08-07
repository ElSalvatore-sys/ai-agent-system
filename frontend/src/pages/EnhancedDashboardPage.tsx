import React, { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { ErrorBoundary, useSuccessToast, useErrorToast } from '@/components/ui';
import { EnhancedDashboard } from '@/components/features';
import { apiService } from '@/services/api';

interface CostData {
  date: string;
  cost: number;
  tokens: number;
  model: string;
}

interface Conversation {
  id: string;
  title: string;
  lastMessage: string;
  timestamp: Date;
  messageCount: number;
  cost: number;
}

interface SystemStatus {
  api: 'operational' | 'degraded' | 'down';
  database: 'operational' | 'degraded' | 'down';
  ai_models: 'operational' | 'degraded' | 'down';
  websocket: 'operational' | 'degraded' | 'down';
}

// Mock data - replace with actual API calls
const generateMockCostData = (): CostData[] => {
  const data: CostData[] = [];
  const now = new Date();
  
  for (let i = 6; i >= 0; i--) {
    const date = new Date(now);
    date.setDate(date.getDate() - i);
    
    data.push({
      date: date.toISOString().split('T')[0],
      cost: Math.random() * 5 + 1, // $1-6 per day
      tokens: Math.floor(Math.random() * 50000 + 10000), // 10k-60k tokens
      model: ['o3', 'claude-3.5', 'gemini-pro'][Math.floor(Math.random() * 3)],
    });
  }
  
  return data;
};

const generateMockConversations = (): Conversation[] => {
  const titles = [
    'React Component Help',
    'Database Design Discussion',
    'Code Review Session',
    'API Development',
    'UI/UX Improvements',
    'Performance Optimization',
    'Bug Fixing Session',
    'Feature Planning',
  ];
  
  const lastMessages = [
    'Thanks for the help with the TypeScript types!',
    'The database schema looks good now.',
    'Could you review this implementation?',
    'Let\'s discuss the API endpoints.',
    'The new design looks great!',
    'Performance has improved significantly.',
    'Bug has been fixed successfully.',
    'Planning the next sprint features.',
  ];
  
  return titles.map((title, index) => ({
    id: `conv-${index}`,
    title,
    lastMessage: lastMessages[index],
    timestamp: new Date(Date.now() - Math.random() * 7 * 24 * 60 * 60 * 1000),
    messageCount: Math.floor(Math.random() * 50 + 5),
    cost: Math.random() * 2 + 0.1,
  }));
};

const mockSystemStatus: SystemStatus = {
  api: 'operational',
  database: 'operational',
  ai_models: 'operational',
  websocket: 'operational',
};

export const EnhancedDashboardPage: React.FC = () => {
  const [costData, setCostData] = useState<CostData[]>(generateMockCostData());
  const [conversations, setConversations] = useState<Conversation[]>(generateMockConversations());
  const [systemStatus, setSystemStatus] = useState<SystemStatus>(mockSystemStatus);
  const [isRefreshing, setIsRefreshing] = useState(false);
  
  const successToast = useSuccessToast();
  const errorToast = useErrorToast();

  // Fetch dashboard stats from API
  const { data: statsData, isLoading: statsLoading, refetch } = useQuery({
    queryKey: ['dashboard-stats'],
    queryFn: () => apiService.getDashboardStats(),
    refetchInterval: 30000, // Refetch every 30 seconds
  });

  const handleRefreshData = async () => {
    setIsRefreshing(true);
    try {
      // Simulate data refresh
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Generate new mock data
      setCostData(generateMockCostData());
      setConversations(generateMockConversations());
      
      // Refetch real data
      await refetch();
      
      successToast('Dashboard data refreshed');
    } catch (error) {
      errorToast('Failed to refresh data');
    } finally {
      setIsRefreshing(false);
    }
  };

  const handleQuickAction = async (action: 'generate-code' | 'create-pdf' | 'deploy-app') => {
    try {
      switch (action) {
        case 'generate-code':
          // Navigate to chat with code generation prompt
          window.location.href = '/enhanced-chat?action=generate-code';
          break;
        case 'create-pdf':
          // Navigate to chat with PDF creation prompt
          window.location.href = '/enhanced-chat?action=create-pdf';
          break;
        case 'deploy-app':
          // Navigate to chat with deployment prompt
          window.location.href = '/enhanced-chat?action=deploy-app';
          break;
      }
      
      successToast(`Starting ${action.replace('-', ' ')}`);
    } catch (error) {
      errorToast(`Failed to start ${action.replace('-', ' ')}`);
    }
  };

  // Simulate periodic system status updates
  useEffect(() => {
    const interval = setInterval(() => {
      // Randomly change system status to simulate real monitoring
      const services: (keyof SystemStatus)[] = ['api', 'database', 'ai_models', 'websocket'];
      const statuses: ('operational' | 'degraded' | 'down')[] = ['operational', 'degraded', 'down'];
      
      setSystemStatus(prev => {
        const newStatus = { ...prev };
        
        // 90% chance each service stays operational
        services.forEach(service => {
          const rand = Math.random();
          if (rand > 0.9) {
            newStatus[service] = statuses[Math.floor(Math.random() * statuses.length)];
          } else {
            newStatus[service] = 'operational';
          }
        });
        
        return newStatus;
      });
    }, 10000); // Update every 10 seconds

    return () => clearInterval(interval);
  }, []);

  return (
    <ErrorBoundary>
      <div className="min-h-screen bg-gray-50">
        <div className="container mx-auto px-4 py-6 max-w-7xl">
          <EnhancedDashboard
            costData={costData}
            recentConversations={conversations}
            systemStatus={systemStatus}
            onQuickAction={handleQuickAction}
            onRefreshData={handleRefreshData}
            isLoading={isRefreshing || statsLoading}
          />
        </div>
      </div>
    </ErrorBoundary>
  );
};