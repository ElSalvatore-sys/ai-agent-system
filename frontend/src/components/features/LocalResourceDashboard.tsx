import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/Card';
import { Cpu, MemoryStick, Zap } from 'lucide-react';
import { LocalResourceUsage } from '../../types';
import { useTheme } from '../../hooks/useTheme';

// Mock Data - In a real app, this data would be from a WebSocket stream.
const mockResourceUsage: LocalResourceUsage = {
  cpu: 45, // percentage
  gpu: 76, // percentage
  memory: 10.2, // in GB
  totalMemory: 16, // in GB
};

interface GaugeProps {
  label: string;
  value: number;
  maxValue: number;
  unit: string;
  icon: React.ReactNode;
  colorClass: string;
}

const Gauge: React.FC<GaugeProps> = ({ label, value, maxValue, unit, icon, colorClass }) => {
  const percentage = (value / maxValue) * 100;
  const circumference = 2 * Math.PI * 40; // a circle with radius 40
  const strokeDashoffset = circumference - (percentage / 100) * circumference;

  return (
    <div className="flex flex-col items-center justify-center">
      <div className="relative">
        <svg className="w-32 h-32 transform -rotate-90">
          <circle
            cx="64"
            cy="64"
            r="40"
            stroke="currentColor"
            strokeWidth="8"
            fill="transparent"
            className="text-gray-200 dark:text-gray-700"
          />
          <circle
            cx="64"
            cy="64"
            r="40"
            stroke="currentColor"
            strokeWidth="8"
            fill="transparent"
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            className={`transition-all duration-500 ease-in-out ${colorClass}`}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
            <div className="text-2xl font-bold">{value.toFixed(1)}{unit}</div>
        </div>
      </div>
       <div className="mt-2 text-center">
        <div className="flex items-center justify-center font-semibold">
           {icon}
           <span className="ml-2">{label}</span>
        </div>
      </div>
    </div>
  );
};


export const LocalResourceDashboard: React.FC = () => {
    const { theme } = useTheme();

    // In a real app, you would use a WebSocket hook to get live data.
    // const { resourceUsage } = useSystemMonitor();
    const resourceUsage = mockResourceUsage;
    const memoryPercentage = (resourceUsage.memory / resourceUsage.totalMemory) * 100;

    return (
        <div className={`p-4 ${theme === 'dark' ? 'bg-gray-900 text-gray-100' : 'bg-gray-50 text-gray-900'}`}>
             <Card className={theme === 'dark' ? 'bg-gray-800' : 'bg-white'}>
                <CardHeader>
                    <CardTitle className="text-xl">Local Resource Dashboard</CardTitle>
                    <p className="text-sm text-gray-500 dark:text-gray-400">
                        Real-time hardware utilization for local model inference.
                    </p>
                </CardHeader>
                <CardContent>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-8 text-center">
                        <Gauge 
                            label="GPU Utilization"
                            value={resourceUsage.gpu}
                            maxValue={100}
                            unit="%"
                            icon={<Zap size={20} />}
                            colorClass="text-green-500"
                        />
                         <Gauge 
                            label="CPU Utilization"
                            value={resourceUsage.cpu}
                            maxValue={100}
                            unit="%"
                            icon={<Cpu size={20} />}
                            colorClass="text-blue-500"
                        />
                         <Gauge 
                            label="Memory Usage"
                            value={resourceUsage.memory}
                            maxValue={resourceUsage.totalMemory}
                            unit="GB"
                            icon={<MemoryStick size={20} />}
                            colorClass="text-purple-500"
                        />
                    </div>
                    <div className="mt-8 text-center text-sm text-gray-500 dark:text-gray-400">
                        <p>Total Memory: {resourceUsage.totalMemory} GB</p>
                        <p className="mt-2">
                           <span className={`font-bold ${memoryPercentage > 90 ? 'text-red-500' : memoryPercentage > 75 ? 'text-yellow-500' : 'text-green-500'}`}>
                                Memory usage is at {memoryPercentage.toFixed(0)}%.
                           </span>
                           {memoryPercentage > 90 && " Consider unloading models to free up resources."}
                        </p>
                    </div>
                </CardContent>
            </Card>
        </div>
    );
};

export default LocalResourceDashboard;
