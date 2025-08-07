
import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui';

interface ModelPerformanceData {
  model: string;
  requests: number;
  avgLatency: number;
  avgTokensPerSecond: number;
  errorRate: number;
}

const data: ModelPerformanceData[] = [
  { model: 'Llama 3 8B', requests: 1250, avgLatency: 55, avgTokensPerSecond: 120, errorRate: 0.5 },
  { model: 'Mistral 7B', requests: 980, avgLatency: 65, avgTokensPerSecond: 110, errorRate: 0.8 },
  { model: 'GPT-4 Omni', requests: 320, avgLatency: 210, avgTokensPerSecond: 80, errorRate: 0.2 },
  { model: 'Phi-3 Mini', requests: 2500, avgLatency: 45, avgTokensPerSecond: 150, errorRate: 1.2 },
];

export const ModelPerformanceTable: React.FC = () => {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Model Performance</CardTitle>
      </CardHeader>
      <CardContent>
        <table className="w-full">
          <thead>
            <tr className="border-b">
              <th className="text-left p-2">Model</th>
              <th className="text-right p-2">Requests</th>
              <th className="text-right p-2">Avg. Latency (ms)</th>
              <th className="text-right p-2">Avg. Tokens/s</th>
              <th className="text-right p-2">Error Rate (%)</th>
            </tr>
          </thead>
          <tbody>
            {data.map((row) => (
              <tr key={row.model} className="border-b">
                <td className="text-left p-2">{row.model}</td>
                <td className="text-right p-2">{row.requests}</td>
                <td className="text-right p-2">{row.avgLatency}</td>
                <td className="text-right p-2">{row.avgTokensPerSecond}</td>
                <td className="text-right p-2">{row.errorRate}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </CardContent>
    </Card>
  );
};
