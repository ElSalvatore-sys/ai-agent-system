import React, { useState, useEffect } from 'react';
import { 
  TrendingUp, 
  DollarSign, 
  MessageSquare, 
  Code, 
  FileText, 
  Zap,
  Server,
  Activity,
  Play,
  AlertCircle,
  CheckCircle,
  RefreshCw
} from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent, Button } from '@/components/ui';
import { StatsCard } from '@/components/features';
import { cn, formatNumber, formatDate } from '@/utils/helpers';

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

interface EnhancedDashboardProps {
  costData: CostData[];
  recentConversations: Conversation[];
  systemStatus: SystemStatus;
  onQuickAction: (action: 'generate-code' | 'create-pdf' | 'deploy-app') => void;
  onRefreshData?: () => void;
  isLoading?: boolean;
  className?: string;
}

const quickActions = [
  {
    id: 'generate-code' as const,
    title: 'Generate Code',
    description: 'Create applications or scripts',
    icon: Code,
    color: 'bg-accent-google/80 hover:bg-accent-google',
  },
  {
    id: 'create-pdf' as const,
    title: 'Create PDF',
    description: 'Generate documents or reports',
    icon: FileText,
    color: 'bg-accent-openai/80 hover:bg-accent-openai',
  },
  {
    id: 'deploy-app' as const,
    title: 'Deploy App',
    description: 'Deploy to cloud platforms',
    icon: Zap,
    color: 'bg-accent-anthropic/80 hover:bg-accent-anthropic',
  },
];

const getStatusColor = (status: string) => {
  switch (status) {
    case 'operational':
      return 'text-success';
    case 'degraded':
      return 'text-warning';
    case 'down':
      return 'text-error';
    default:
      return 'text-muted-foreground';
  }
};

const getStatusIcon = (status: string) => {
  switch (status) {
    case 'operational':
      return CheckCircle;
    case 'degraded':
      return AlertCircle;
    case 'down':
      return AlertCircle;
    default:
      return Server;
  }
};

export const EnhancedDashboard: React.FC<EnhancedDashboardProps> = ({
  costData,
  recentConversations,
  systemStatus,
  onQuickAction,
  onRefreshData,
  isLoading = false,
  className,
}) => {
  const [selectedTimeRange, setSelectedTimeRange] = useState<'24h' | '7d' | '30d'>('7d');
  const [autoRefresh, setAutoRefresh] = useState(true);

  useEffect(() => {
    if (!autoRefresh || !onRefreshData) return;
    const interval = setInterval(() => onRefreshData(), 30000);
    return () => clearInterval(interval);
  }, [autoRefresh, onRefreshData]);

  const totalCost = costData.reduce((sum, item) => sum + item.cost, 0);
  const totalTokens = costData.reduce((sum, item) => sum + item.tokens, 0);
  const costTrend = costData.length > 1 ? ((costData[costData.length - 1].cost - costData[0].cost) / costData[0].cost) * 100 : 0;
  const chartData = costData.slice(-7).map(item => ({
    day: new Date(item.date).toLocaleDateString('en-US', { weekday: 'short' }),
    cost: item.cost,
    tokens: item.tokens / 1000,
  }));

  return (
    <div className={cn('space-y-8 animate-fadeIn', className)}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-4xl font-bold text-foreground">Dashboard</h1>
          <p className="mt-2 text-muted-foreground">
            Monitor your AI usage, costs, and system performance.
          </p>
        </div>
        
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1 p-1 bg-secondary/80 rounded-lg">
            {(['24h', '7d', '30d'] as const).map(range => (
              <button
                key={range}
                onClick={() => setSelectedTimeRange(range)}
                className={cn(
                  'px-3 py-1 text-sm font-medium rounded-md transition-colors',
                  selectedTimeRange === range ? 'bg-background text-foreground shadow-sm' : 'text-muted-foreground hover:text-foreground'
                )}
              >
                {range}
              </button>
            ))}
          </div>
          <Button
            variant={autoRefresh ? 'secondary' : 'outline'}
            size="sm"
            onClick={() => setAutoRefresh(!autoRefresh)}
          >
            <RefreshCw className={cn('h-4 w-4 mr-2', isLoading && 'animate-spin')} />
            Auto
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={onRefreshData}
            disabled={isLoading}
          >
            <RefreshCw className={cn('h-4 w-4 mr-2', isLoading && 'animate-spin')} />
            Refresh
          </Button>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatsCard title="Total Cost" value={`$${totalCost.toFixed(2)}`} icon={<DollarSign />} trend={{ value: costTrend }} color="blue" />
        <StatsCard title="Tokens Used" value={formatNumber(totalTokens)} icon={<Activity />} trend={{ value: 12, isPositive: true }} color="green" />
        <StatsCard title="Conversations" value={recentConversations.length} icon={<MessageSquare />} trend={{ value: 8, isPositive: true }} color="purple" />
        <StatsCard title="Avg Cost/Token" value={`$${((totalTokens > 0 ? totalCost / totalTokens : 0) * 1000).toFixed(3)}`} subtitle="per 1K" icon={<TrendingUp />} trend={{ value: -5 }} color="yellow" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <Card>
            <CardHeader><CardTitle>Cost Tracking</CardTitle></CardHeader>
            <CardContent>
              <div className="h-64 flex items-end justify-between space-x-2">
                {chartData.map((item, index) => {
                  const maxCost = Math.max(...chartData.map(d => d.cost));
                  const height = (item.cost / maxCost) * 100;
                  return (
                    <div key={index} className="flex flex-col items-center flex-1 group">
                      <div className="w-full bg-accent-google/50 rounded-t-sm mb-2 min-h-[4px] transition-all duration-300 group-hover:bg-accent-google" style={{ height: `${height}%` }} title={`$${item.cost.toFixed(2)}`} />
                      <div className="text-xs text-muted-foreground text-center">
                        <div>{item.day}</div>
                        <div className="font-medium">${item.cost.toFixed(2)}</div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </CardContent>
          </Card>
        </div>

        <div>
          <Card>
            <CardHeader><CardTitle>Quick Actions</CardTitle></CardHeader>
            <CardContent className="space-y-3">
              {quickActions.map(action => (
                <button key={action.id} onClick={() => onQuickAction(action.id)} className={cn('w-full p-4 rounded-lg text-left text-white', action.color)}>
                  <div className="flex items-start gap-3">
                    <action.icon className="h-6 w-6 flex-shrink-0 mt-0.5" />
                    <div>
                      <h3 className="font-medium">{action.title}</h3>
                      <p className="text-sm opacity-90 mt-1">{action.description}</p>
                    </div>
                  </div>
                </button>
              ))}
            </CardContent>
          </Card>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader><CardTitle>Recent Conversations</CardTitle></CardHeader>
          <CardContent>
            <div className="space-y-3">
              {recentConversations.slice(0, 5).map(conv => (
                <div key={conv.id} className="flex items-start justify-between p-3 bg-secondary/50 rounded-lg hover:bg-secondary transition-colors">
                  <div className="flex-1 min-w-0">
                    <h4 className="text-sm font-medium truncate">{conv.title}</h4>
                    <p className="text-sm text-muted-foreground truncate mt-1">{conv.lastMessage}</p>
                    <div className="flex items-center gap-3 mt-2 text-xs text-muted-foreground">
                      <span>{formatDate(conv.timestamp, { month: 'short', day: 'numeric' })}</span>
                      <span>{conv.messageCount} messages</span>
                      <span>${conv.cost.toFixed(3)}</span>
                    </div>
                  </div>
                  <Button variant="ghost" size="sm" className="ml-2"><Play className="h-4 w-4" /></Button>
                </div>
              ))}
              {recentConversations.length === 0 && <div className="text-center py-6 text-muted-foreground">No recent conversations</div>}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader><CardTitle>System Status</CardTitle></CardHeader>
          <CardContent>
            <div className="space-y-4">
              {Object.entries(systemStatus).map(([service, status]) => {
                const StatusIcon = getStatusIcon(status);
                return (
                  <div key={service} className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <StatusIcon className={cn('h-5 w-5', getStatusColor(status))} />
                      <span className="text-sm font-medium capitalize">{service.replace('_', ' ')}</span>
                    </div>
                    <span className={cn('px-2 py-1 text-xs font-medium rounded-full capitalize', getStatusColor(status), 'bg-opacity-20')}>{status}</span>
                  </div>
                );
              })}
              <div className="pt-4 border-t border-card-border">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">Overall Health</span>
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-success rounded-full"></div>
                    <span className="text-sm text-success font-medium">All Systems Operational</span>
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};
