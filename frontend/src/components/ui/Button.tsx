import React from 'react';
import { Loader2 } from 'lucide-react';
import { cn } from '@/utils/helpers';
import type { ButtonProps } from '@/types';

export const Button: React.FC<ButtonProps> = ({
  variant = 'primary',
  size = 'md',
  disabled = false,
  loading = false,
  onClick,
  type = 'button',
  className,
  children,
  ...props
}) => {
  const baseClasses = [
    'inline-flex items-center justify-center rounded-lg font-semibold transition-all duration-200 ease-in-out',
    'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:ring-offset-background',
    'disabled:pointer-events-none disabled:opacity-50',
  ];

  const variants = {
    primary: [
      'bg-primary text-primary-foreground shadow-md hover:bg-opacity-90',
      'focus-visible:ring-primary',
    ],
    secondary: [
      'bg-secondary text-secondary-foreground hover:bg-opacity-80',
      'focus-visible:ring-secondary',
    ],
    outline: [
      'border border-input bg-transparent hover:bg-accent hover:text-accent-foreground',
      'focus-visible:ring-ring',
    ],
    ghost: [
      'hover:bg-accent hover:text-accent-foreground',
      'focus-visible:ring-ring',
    ],
    destructive: [
      'bg-destructive text-destructive-foreground hover:bg-opacity-90',
      'focus-visible:ring-destructive',
    ],
    glass: [
      'bg-white/10 text-white backdrop-blur-lg border border-white/20',
      'hover:bg-white/20 hover:border-white/30',
      'focus-visible:ring-white',
    ],
  };

  const sizes = {
    sm: 'h-9 px-3 text-sm',
    md: 'h-10 px-4 text-sm',
    lg: 'h-12 px-6 text-base',
  };

  return (
    <button
      type={type}
      className={cn(
        baseClasses,
        variants[variant],
        sizes[size],
        'transform-gpu active:scale-[0.98]',
        className
      )}
      disabled={disabled || loading}
      onClick={onClick}
      {...props}
    >
      {loading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
      {children}
    </button>
  );
};
