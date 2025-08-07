import React from 'react';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';
import { Card, CardContent } from '@/components/ui';
import { cn, formatNumber } from '@/utils/helpers';

interface StatsCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon?: React.ReactNode;
  trend?: {
    value: number;
    label?: string;
    isPositive?: boolean;
  };
  color?: 'blue' | 'green' | 'red' | 'yellow' | 'purple' | 'gray';
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

export const StatsCard: React.FC<StatsCardProps> = ({
  title,
  value,
  subtitle,
  icon,
  trend,
  color = 'blue',
  size = 'md',
  className,
}) => {
  const colors = {
    blue: 'text-accent-google',
    green: 'text-success',
    red: 'text-error',
    yellow: 'text-warning',
    purple: 'text-accent-anthropic',
    gray: 'text-muted-foreground',
  };

  const sizes = {
    sm: { container: 'p-4', value: 'text-xl', title: 'text-sm' },
    md: { container: 'p-6', value: 'text-3xl', title: 'text-sm' },
    lg: { container: 'p-8', value: 'text-4xl', title: 'text-base' },
  };

  const formatValue = (val: string | number) => typeof val === 'number' ? formatNumber(val) : val;

  const getTrendIcon = () => {
    if (!trend) return null;
    if (trend.value > 0) return <TrendingUp className="h-4 w-4" />;
    if (trend.value < 0) return <TrendingDown className="h-4 w-4" />;
    return <Minus className="h-4 w-4" />;
  };

  const getTrendColor = () => {
    if (!trend) return '';
    if (trend.isPositive !== undefined) {
      return trend.isPositive ? 'text-success' : 'text-error';
    }
    if (trend.value > 0) return 'text-success';
    if (trend.value < 0) return 'text-error';
    return 'text-muted-foreground';
  };

  return (
    <Card className={cn('overflow-hidden', className)}>
      <CardContent className={cn(sizes[size].container, 'relative')}>
        <div className="flex items-start justify-between">
          <div>
            <p className={cn('font-medium text-muted-foreground', sizes[size].title)}>
              {title}
            </p>
            <p className={cn('font-bold text-foreground', sizes[size].value)}>
              {formatValue(value)}
            </p>
            {subtitle && (
              <p className="text-xs text-muted-foreground mt-1">
                {subtitle}
              </p>
            )}
          </div>
          {icon && (
            <div className={cn('p-2 rounded-lg bg-secondary/80', colors[color])}>
              {icon}
            </div>
          )}
        </div>
        {trend && (
          <div className="mt-4 flex items-center gap-1 text-sm">
            <div className={getTrendColor()}>{getTrendIcon()}</div>
            <span className={cn('font-medium', getTrendColor())}>
              {Math.abs(trend.value).toFixed(1)}%
            </span>
            {trend.label && (
              <span className="text-muted-foreground ml-1">{trend.label}</span>
            )}
          </div>
        )}
        <div className={cn('absolute -bottom-8 -right-8 w-24 h-24 rounded-full opacity-10', colors[color], 'bg-current')} />
      </CardContent>
    </Card>
  );
};
