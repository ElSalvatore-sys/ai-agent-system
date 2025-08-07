
import React from 'react';
import { motion } from 'framer-motion';
import { Zap, Cpu, MemoryStick, Timer } from 'lucide-react';
import { cn } from '@/utils/helpers';

interface PerformanceMetrics {
  latency: number;
  throughput: number;
  cpuUsage: number;
  memoryUsage: number;
}

interface PerformanceIndicatorProps {
  metrics: PerformanceMetrics;
  className?: string;
}

const Gauge: React.FC<{ value: number; label: string; icon: React.ReactNode }> = ({ value, label, icon }) => {
  const circumference = 2 * Math.PI * 28;
  const strokeDashoffset = circumference - (value / 100) * circumference;

  return (
    <div className="flex flex-col items-center justify-center">
      <div className="relative h-20 w-20">
        <svg className="h-full w-full" viewBox="0 0 64 64">
          <circle
            className="text-gray-200/50"
            strokeWidth="4"
            stroke="currentColor"
            fill="transparent"
            r="28"
            cx="32"
            cy="32"
          />
          <motion.circle
            className="text-blue-500"
            strokeWidth="4"
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            strokeLinecap="round"
            stroke="currentColor"
            fill="transparent"
            r="28"
            cx="32"
            cy="32"
            initial={{ strokeDashoffset: circumference }}
            animate={{ strokeDashoffset }}
            transition={{ duration: 0.8, ease: "easeOut" }}
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          {icon}
        </div>
      </div>
      <p className="mt-2 text-xs text-gray-600 font-medium">{label}</p>
      <p className="text-sm font-semibold text-gray-800">{value}%</p>
    </div>
  );
};

export const PerformanceIndicators: React.FC<PerformanceIndicatorProps> = ({ metrics, className }) => {
  return (
    <div
      className={cn(
        'p-4 rounded-lg border bg-white/30 backdrop-blur-md shadow-lg',
        'border-gray-200/50',
        className
      )}
    >
      <h3 className="text-md font-bold text-gray-800 mb-4">Performance</h3>
      <div className="grid grid-cols-2 gap-4">
        <Gauge value={metrics.cpuUsage} label="CPU" icon={<Cpu className="h-6 w-6 text-gray-500" />} />
        <Gauge value={metrics.memoryUsage} label="Memory" icon={<MemoryStick className="h-6 w-6 text-gray-500" />} />
      </div>
      <div className="mt-4 pt-4 border-t border-gray-200/50 space-y-2 text-sm">
        <div className="flex justify-between items-center">
          <span className="flex items-center text-gray-600"><Timer className="h-4 w-4 mr-2"/> Latency</span>
          <span className="font-semibold text-gray-800">{metrics.latency}ms</span>
        </div>
        <div className="flex justify-between items-center">
          <span className="flex items-center text-gray-600"><Zap className="h-4 w-4 mr-2"/> Throughput</span>
          <span className="font-semibold text-gray-800">{metrics.throughput} t/s</span>
        </div>
      </div>
    </div>
  );
};
