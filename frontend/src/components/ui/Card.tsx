import React from 'react';
import { cn } from '@/utils/helpers';
import type { BaseComponentProps } from '@/types';

interface CardProps extends BaseComponentProps {
  padding?: 'none' | 'sm' | 'md' | 'lg';
  interactive?: boolean;
}

export const Card: React.FC<CardProps> = ({
  padding = 'md',
  interactive = false,
  className,
  children,
}) => {
  const paddings = {
    none: '',
    sm: 'p-3',
    md: 'p-4',
    lg: 'p-6',
  };

  return (
    <div
      className={cn(
        'glass-card',
        paddings[padding],
        interactive && 'hover:border-primary/50 cursor-pointer',
        className
      )}
    >
      {children}
    </div>
  );
};

export const CardHeader: React.FC<BaseComponentProps> = ({ className, children }) => (
  <div className={cn('mb-4', className)}>
    {children}
  </div>
);

export const CardTitle: React.FC<BaseComponentProps> = ({ className, children }) => (
  <h3 className={cn('text-lg font-semibold text-foreground', className)}>
    {children}
  </h3>
);

export const CardDescription: React.FC<BaseComponentProps> = ({ className, children }) => (
  <p className={cn('text-sm text-muted-foreground mt-1', className)}>
    {children}
  </p>
);

export const CardContent: React.FC<BaseComponentProps> = ({ className, children }) => (
  <div className={cn(className)}>
    {children}
  </div>
);

export const CardFooter: React.FC<BaseComponentProps> = ({ className, children }) => (
  <div className={cn('mt-4 pt-4 border-t border-white/10', className)}>
    {children}
  </div>
);
