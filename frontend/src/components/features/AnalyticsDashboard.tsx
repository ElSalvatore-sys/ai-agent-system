import React, { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Calendar, Download, TrendingUp, MessageSquare, Users, Clock, Activity, Bot, Server, AlertCircle, CheckCircle, RefreshCw } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent, Button, Loading } from '@/components/ui';
import { StatsCard } from '@/components/features';
import { apiService } from '@/services/api';
import { useAIAnalytics } from '@/hooks/useAI';
import { formatNumber, formatDate } from '@/utils/helpers';
import { cn } from '@/lib/utils';

interface AnalyticsDashboardProps {
  className?: string;
}

export const AnalyticsDashboard: React.FC<AnalyticsDashboardProps> = ({ className }) => {
  const [dateRange, setDateRange] = useState<{ from: Date; to: Date }>({
    from: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000),
    to: new Date(),
  });

  const { data: analyticsData, isLoading: analyticsLoading } = useQuery({
    queryKey: ['analytics', dateRange],
    queryFn: () => apiService.getAnalytics(dateRange),
    staleTime: 5 * 60 * 1000,
  });

  const { usage: aiUsage, isLoading: usageLoading } = useAIAnalytics();
  const analytics = analyticsData?.data;

  if (analyticsLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loading size="lg" text="Loading analytics..." />
      </div>
    );
  }

  const handleExportData = () => {
    console.log('Export data not implemented yet');
  };

  return (
    <div className={cn('space-y-8 animate-fadeIn', className)}>
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-4xl font-bold text-foreground">Analytics Dashboard</h1>
          <p className="mt-2 text-muted-foreground">
            Monitor your AI usage, costs, and system performance.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1 p-1 bg-secondary/80 rounded-lg">
            <div className="flex items-center gap-2 text-sm text-gray-600">
              <Calendar className="h-4 w-4" />
              <span>
                {formatDate(dateRange.from, { month: 'short', day: 'numeric' })} - {' '}
                {formatDate(dateRange.to, { month: 'short', day: 'numeric' })}
              </span>
            </div>
          </div>
          <Button 
            variant="outline" 
            size="sm" 
            onClick={handleExportData}
            className="flex items-center gap-2"
          >
            <Download className="h-4 w-4" />
            Export
          </Button>
        </div>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatsCard
          title="Total Sessions"
          value={analytics?.totalSessions || 0}
          icon={<MessageSquare />}
          color="blue"
        />
        <StatsCard
          title="Total Messages"
          value={analytics?.totalMessages || 0}
          icon={<TrendingUp />}
          color="green"
        />
        <StatsCard
          title="Avg Response Time"
          value={`${analytics?.averageResponseTime || 0}ms`}
          icon={<Clock />}
          color="purple"
        />
        <StatsCard
          title="User Satisfaction"
          value={`${analytics?.userEngagement?.satisfactionScore || 0}%`}
          icon={<Users />}
          color="yellow"
        />
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5" />
              Daily Activity
            </CardTitle>
          </CardHeader>
          <CardContent>
            {analytics?.dailyStats ? (
              <div className="space-y-4">
                {analytics.dailyStats.slice(-7).map((stat) => (
                  <div key={stat.date} className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className="text-sm font-medium text-gray-900">
                        {new Date(stat.date).toLocaleDateString('en-US', { weekday: 'short' })}
                      </div>
                      <div className="text-sm text-gray-600">
                        {stat.date}
                      </div>
                    </div>
                    <div className="flex items-center gap-4">
                      <div className="text-right">
                        <div className="text-sm font-medium text-gray-900">
                          {formatNumber(stat.sessions)}
                        </div>
                        <div className="text-xs text-gray-500">sessions</div>
                      </div>
                      <div className="text-right">
                        <div className="text-sm font-medium text-gray-900">
                          {formatNumber(stat.messages)}
                        </div>
                        <div className="text-xs text-gray-500">messages</div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-8 text-gray-500">
                <Activity className="h-12 w-12 mx-auto mb-4 text-gray-300" />
                <p>No activity data available</p>
              </div>
            )}
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Bot className="h-5 w-5" />
              AI Model Usage
            </CardTitle>
          </CardHeader>
          <CardContent>
            {!usageLoading && Object.keys(aiUsage).length > 0 ? (
              <div className="space-y-4">
                {Object.entries(aiUsage).map(([model, count]) => {
                  const total = Object.values(aiUsage).reduce((sum, val) => sum + (val as number), 0);
                  const percentage = total > 0 ? Math.round(((count as number) / total) * 100) : 0;
                  
                  return (
                    <div key={model} className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-medium text-gray-900">
                          {model}
                        </span>
                        <div className="text-right">
                          <span className="text-sm font-medium text-gray-900">
                            {formatNumber(count as number)}
                          </span>
                          <span className="text-xs text-gray-500 ml-2">
                            ({percentage}%)
                          </span>
                        </div>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div
                          className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                          style={{ width: `${percentage}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="text-center py-8 text-gray-500">
                {usageLoading ? (
                  <Loading size="sm" text="Loading usage data..." />
                ) : (
                  <>
                    <Bot className="h-12 w-12 mx-auto mb-4 text-gray-300" />
                    <p>No model usage data available</p>
                  </>
                )}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
};