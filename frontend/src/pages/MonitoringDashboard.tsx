
import React from 'react';
import { ResourceUsageChart, CostSavingsChart, ModelPerformanceTable } from '@/components/features';

export const MonitoringDashboard: React.FC = () => {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Monitoring Dashboard</h1>
        <p className="mt-2 text-gray-600">
          Track your local AI model performance, resource usage, and cost savings.
        </p>
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <ResourceUsageChart />
        <CostSavingsChart />
      </div>
      <div>
        <ModelPerformanceTable />
      </div>
    </div>
  );
};
