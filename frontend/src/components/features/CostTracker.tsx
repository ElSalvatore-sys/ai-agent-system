import React from 'react';
import { DollarSign } from 'lucide-react';
import { cn } from '@/utils/helpers';

interface CostTrackerProps {
  /** Total cost in USD (optional). Defaults to 0 when not provided. */
  cost?: number;
  className?: string;
}

export const CostTracker: React.FC<CostTrackerProps> = ({ cost, className }) => {
  return (
    <div className={cn('flex items-center text-sm text-gray-500', className)}>
      <DollarSign className="h-4 w-4 mr-1" />
      <span>Cost: ${(cost ?? 0).toFixed(4)}</span>
    </div>
  );
};
