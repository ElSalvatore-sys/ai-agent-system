
import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui';
import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';

interface ResourceData {
  time: string;
  cpu: number;
  gpu: number;
  ram: number;
}

const data: ResourceData[] = [
  { time: '10:00', cpu: 45, gpu: 60, ram: 55 },
  { time: '10:05', cpu: 50, gpu: 65, ram: 58 },
  { time: '10:10', cpu: 55, gpu: 70, ram: 62 },
  { time: '10:15', cpu: 60, gpu: 75, ram: 65 },
  { time: '10:20', cpu: 58, gpu: 72, ram: 63 },
  { time: '10:25', cpu: 62, gpu: 78, ram: 68 },
  { time: '10:30', cpu: 65, gpu: 80, ram: 70 },
];

export const ResourceUsageChart: React.FC = () => {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Resource Usage</CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={300}>
          <AreaChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="time" />
            <YAxis />
            <Tooltip />
            <Area type="monotone" dataKey="cpu" stackId="1" stroke="#8884d8" fill="#8884d8" />
            <Area type="monotone" dataKey="gpu" stackId="1" stroke="#82ca9d" fill="#82ca9d" />
            <Area type="monotone" dataKey="ram" stackId="1" stroke="#ffc658" fill="#ffc658" />
          </AreaChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
};
