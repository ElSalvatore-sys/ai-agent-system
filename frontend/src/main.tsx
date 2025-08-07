import React from 'react';
import ReactDOM from 'react-dom/client';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';
import { BrowserRouter } from 'react-router-dom';
import { ThemeProvider } from '@/context/providers/ThemeContext';
import { NotificationsProvider } from '@/context/providers/NotificationsContext';
import { ToastProvider } from '@/components/ui';
import App from './App';
import './index.css';

console.log('🚀 Starting AI Agent System...');
console.log('📍 Main.tsx loaded successfully');
console.log('🏠 Root element:', document.getElementById('root'));

// Create a client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60 * 5, // 5 minutes
      retry: (failureCount, error: any) => {
        // Don't retry on 4xx errors (client errors)
        if (error?.response?.status >= 400 && error?.response?.status < 500) {
          return false;
        }
        return failureCount < 3;
      },
    },
    mutations: {
      retry: 1,
    },
  },
});

console.log('🎯 About to create React root...');
const rootElement = document.getElementById('root');
console.log('🎯 Root element found:', !!rootElement);

if (rootElement) {
  console.log('✅ Creating ReactDOM root...');
  const root = ReactDOM.createRoot(rootElement);
  
  console.log('🚀 Rendering React app...');
  root.render(
    <React.StrictMode>
      <QueryClientProvider client={queryClient}>
        <BrowserRouter>
          <ThemeProvider>
            <NotificationsProvider>
              <ToastProvider>
                <App />
                {/* Only show devtools in development */}
                {import.meta.env.DEV && <ReactQueryDevtools initialIsOpen={false} />}
              </ToastProvider>
            </NotificationsProvider>
          </ThemeProvider>
        </BrowserRouter>
      </QueryClientProvider>
    </React.StrictMode>
  );
  console.log('✅ React app rendered successfully!');
} else {
  console.error('❌ Root element not found!');
}