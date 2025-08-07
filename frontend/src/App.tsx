import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { ErrorBoundary } from '@/components/ui';
import { useTheme } from '@/context/providers/ThemeContext';
import { cn } from '@/utils/helpers';
import { DashboardLayout } from '@/components/layout/DashboardLayout';
import { ComponentShowcase } from '@/components/demo/ComponentShowcase';
import { 
  Dashboard, 
  Chat, 
  Analytics, 
  Settings, 
  EnhancedChat, 
  EnhancedDashboardPage,
  MultiModelPage,
  Agents
} from '@/pages';
import { MonitoringDashboard } from '@/pages/MonitoringDashboard';

function App() {
  console.log('🏠 App component mounting...');
  const { resolvedTheme } = useTheme();
  
  return (
    <ErrorBoundary>
      <div className={cn('min-h-screen font-sans transition-colors', resolvedTheme)}>
        <Routes>
            {/* Redirect root to enhanced dashboard */}
            <Route path="/" element={<Navigate to="/dashboard" replace />} />
            
            {/* Main Dashboard Routes with Layout */}
            <Route path="/dashboard" element={
              <DashboardLayout>
                <EnhancedDashboardPage />
              </DashboardLayout>
            } />
            
            <Route path="/enhanced-chat" element={
              <DashboardLayout>
                <EnhancedChat />
              </DashboardLayout>
            } />
            
            <Route path="/chat" element={
              <DashboardLayout>
                <Chat />
              </DashboardLayout>
            } />
            
            <Route path="/agents" element={
              <DashboardLayout>
                <Agents />
              </DashboardLayout>
            } />
            
            <Route path="/analytics" element={
              <DashboardLayout>
                <Analytics />
              </DashboardLayout>
            } />
            
            <Route path="/monitoring" element={
              <DashboardLayout>
                <MonitoringDashboard />
              </DashboardLayout>
            } />
            
            <Route path="/multi-model" element={
              <DashboardLayout>
                <MultiModelPage />
              </DashboardLayout>
            } />
            
            <Route path="/settings" element={
              <DashboardLayout>
                <Settings />
              </DashboardLayout>
            } />
            
            {/* Component Showcase for Development */}
            <Route path="/showcase" element={
              <DashboardLayout>
                <ComponentShowcase />
              </DashboardLayout>
            } />
            
            {/* Fallback to dashboard for unknown routes */}
            <Route path="*" element={<Navigate to="/dashboard" replace />} />
          </Routes>
        </div>
      </ErrorBoundary>
    );
}

export default App;