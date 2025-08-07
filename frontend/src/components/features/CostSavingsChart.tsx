
import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';

interface CostData {
  month: string;
  local: number;
  cloud: number;
}

const data: CostData[] = [
  { month: 'Jan', local: 0, cloud: 120 },
  { month: 'Feb', local: 0, cloud: 150 },
  { month: 'Mar', local: 0, cloud: 130 },
  { month: 'Apr', local: 0, cloud: 160 },
  { month: 'May', local: 0, cloud: 180 },
  { month: 'Jun', local: 0, cloud: 200 },
];

export const CostSavingsChart: React.FC = () => {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Cost Savings (Local vs. Cloud)</CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="month" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="local" fill="#8884d8" name="Local" />
            <Bar dataKey="cloud" fill="#82ca9d" name="Cloud" />
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
};
