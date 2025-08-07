import React, { ReactNode } from 'react';
import { Notifications } from '@/components/features/Notifications';
import { Sidebar } from '@/components/features';
import { useTheme } from '@/context/providers/ThemeContext';
import { cn } from '@/utils/helpers';

interface DashboardLayoutProps {
  children: ReactNode;
  sidebar?: ReactNode;
  rightPanel?: ReactNode;
}

export const DashboardLayout: React.FC<DashboardLayoutProps> = ({ children, rightPanel }) => {
  const { resolvedTheme } = useTheme();

  return (
    <div className={cn('flex h-screen font-sans', resolvedTheme)}>
      <Sidebar />
      <main className="flex-1 flex flex-col overflow-hidden">
        <div className="flex-1 flex overflow-hidden">
          <div className="flex-1 overflow-y-auto">
            {children}
          </div>
          {rightPanel && (
            <div className="w-80 border-l overflow-y-auto">
              {rightPanel}
            </div>
          )}
        </div>
      </main>
      <Notifications />
    </div>
  );
};