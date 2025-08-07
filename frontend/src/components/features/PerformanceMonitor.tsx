import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/Card';
import { BarChart, DollarSign, Zap, Star, AlertTriangle } from 'lucide-react';
import { AIModel, ModelPerformanceData } from '../../types';
import { useTheme } from '../../context/providers/ThemeContext';

// Mock Data - In a real app, this data would be fetched from your analytics backend or WebSocket stream.
const mockPerformanceData: (ModelPerformanceData & { model: Pick<AIModel, 'provider' | 'hostType'> })[] = [
  {
    modelId: 'ollama-llama2',
    modelName: 'Llama 2',
    model: { provider: 'Ollama', hostType: 'local' },
    avgLatency: 80, // ms
    avgQuality: 4.2,
    totalUsage: 1500000, // tokens
  },
  {
    modelId: 'huggingface-mistral',
    modelName: 'Mistral-7B',
    model: { provider: 'Hugging Face', hostType: 'local' },
    avgLatency: 65,
    avgQuality: 4.5,
    totalUsage: 2200000,
  },
  {
    modelId: 'openai-gpt4',
    modelName: 'GPT-4',
    model: { provider: 'OpenAI', hostType: 'cloud' },
    avgLatency: 550,
    avgQuality: 4.9,
    totalUsage: 8500000,
  },
  {
    modelId: 'claude-opus',
    modelName: 'Claude 3 Opus',
    model: { provider: 'Claude', hostType: 'cloud' },
    avgLatency: 620,
    avgQuality: 4.8,
    totalUsage: 6300000,
  },
  {
    modelId: 'gemini-1.5-pro',
    modelName: 'Gemini 1.5 Pro',
    model: { provider: 'Gemini', hostType: 'cloud' },
    avgLatency: 500,
    avgQuality: 4.7,
    totalUsage: 9100000,
  },
];


const PerformanceTableRow: React.FC<{ data: ModelPerformanceData & { model: Pick<AIModel, 'provider' | 'hostType'> } }> = ({ data }) => {
    const { theme } = useTheme();

    // Simple color scaling for latency: green < 150ms, yellow < 500ms, red >= 500ms
    const getLatencyColor = (latency: number) => {
        if (latency < 150) return 'text-green-500';
        if (latency < 500) return 'text-yellow-500';
        return 'text-red-500';
    };

    // Simple color scaling for quality: green > 4.5, yellow > 4.0, red <= 4.0
    const getQualityColor = (quality: number) => {
        if (quality >= 4.5) return 'text-green-500';
        if (quality > 4.0) return 'text-yellow-500';
        return 'text-red-500';
    };

    return (
        <tr className={`border-b ${theme === 'dark' ? 'border-gray-700' : 'border-gray-200'} hover:bg-gray-50 dark:hover:bg-gray-800`}>
            <td className="p-4">
                <div className="font-bold">{data.modelName}</div>
                <div className="text-xs text-gray-500 dark:text-gray-400">{data.model.provider} - {data.model.hostType}</div>
            </td>
            <td className={`p-4 font-mono font-semibold ${getLatencyColor(data.avgLatency)}`}>
                {data.avgLatency} ms
            </td>
            <td className={`p-4 font-mono font-semibold ${getQualityColor(data.avgQuality)}`}>
                <div className="flex items-center">
                   <Star size={16} className="mr-1" /> {data.avgQuality.toFixed(1)} / 5.0
                </div>
            </td>
            <td className="p-4 font-mono">
                {(data.totalUsage / 1000000).toFixed(2)}M
            </td>
        </tr>
    );
};


export const PerformanceMonitor: React.FC = () => {
    const { theme } = useTheme();

    return (
        <div className={`p-4 ${theme === 'dark' ? 'bg-gray-900 text-gray-100' : 'bg-gray-50 text-gray-900'}`}>
            <Card className={theme === 'dark' ? 'bg-gray-800' : 'bg-white'}>
                <CardHeader>
                    <CardTitle className="flex items-center text-xl">
                        <BarChart className="mr-2"/>
                        Model Performance Monitor
                    </CardTitle>
                    <p className="text-sm text-gray-500 dark:text-gray-400">
                        Comparative analysis of key performance indicators across all models.
                    </p>
                </CardHeader>
                <CardContent>
                    <div className="overflow-x-auto">
                        <table className="w-full text-left">
                            <thead className={`text-xs uppercase ${theme === 'dark' ? 'bg-gray-700 text-gray-300' : 'bg-gray-100 text-gray-600'}`}>
                                <tr>
                                    <th className="p-4">Model</th>
                                    <th className="p-4 flex items-center"><Zap size={14} className="mr-1"/> Avg Latency</th>
                                    <th className="p-4 flex items-center"><Star size={14} className="mr-1"/> Avg Quality</th>
                                    <th className="p-4 flex items-center"><DollarSign size={14} className="mr-1"/> Total Tokens</th>
                                </tr>
                            </thead>
                            <tbody>
                                {mockPerformanceData.map(data => (
                                    <PerformanceTableRow key={data.modelId} data={data} />
                                ))}
                            </tbody>
                        </table>
                    </div>
                     <div className="mt-4 text-xs text-gray-500 dark:text-gray-400 p-3 bg-gray-100 dark:bg-gray-900 rounded-lg flex items-center">
                        <AlertTriangle size={16} className="mr-2 text-yellow-500" />
                        <div>
                            <strong>Latency</strong> is average time to first token. <strong>Quality</strong> is based on user ratings and benchmarks. <strong>Local model</strong> usage is estimated. All data is for the last 7 days.
                        </div>
                    </div>
                </CardContent>
            </Card>
        </div>
    );
};

export default PerformanceMonitor;
