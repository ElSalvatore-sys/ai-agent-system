import { useState, useEffect, useCallback, useRef } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { useLocation, useNavigate } from 'react-router-dom';
import { debounce } from '@/utils/helpers';

interface ContextState {
  route: string;
  scrollPosition: { [key: string]: number };
  formData: { [key: string]: any };
  filters: { [key: string]: any };
  viewState: { [key: string]: any };
  timestamp: number;
}

interface SmartContextOptions {
  persistenceKey?: string;
  autoSave?: boolean;
  saveInterval?: number;
  maxHistory?: number;
}

export function useSmartContext(options: SmartContextOptions = {}) {
  const {
    persistenceKey = 'smart-context',
    autoSave = true,
    saveInterval = 1000,
    maxHistory = 10
  } = options;

  const queryClient = useQueryClient();
  const location = useLocation();
  const navigate = useNavigate();
  
  const [contextStack, setContextStack] = useState<ContextState[]>([]);
  const [currentContext, setCurrentContext] = useState<ContextState | null>(null);
  const saveTimeoutRef = useRef<NodeJS.Timeout>();

  // Load context from localStorage on mount
  useEffect(() => {
    const savedContext = localStorage.getItem(persistenceKey);
    if (savedContext) {
      try {
        const parsed = JSON.parse(savedContext);
        setContextStack(parsed.stack || []);
        setCurrentContext(parsed.current || null);
      } catch (error) {
        console.warn('Failed to load context from localStorage:', error);
      }
    }
  }, [persistenceKey]);

  // Debounced save function
  const debouncedSave = useCallback(
    debounce((stack: ContextState[], current: ContextState | null) => {
      localStorage.setItem(persistenceKey, JSON.stringify({
        stack: stack.slice(-maxHistory),
        current,
        timestamp: Date.now()
      }));
    }, saveInterval),
    [persistenceKey, maxHistory, saveInterval]
  );

  // Auto-save when context changes
  useEffect(() => {
    if (autoSave) {
      debouncedSave(contextStack, currentContext);
    }
  }, [contextStack, currentContext, autoSave, debouncedSave]);

  // Save context for current route
  const saveContext = useCallback((partialState: Partial<Omit<ContextState, 'route' | 'timestamp'>>) => {
    const newContext: ContextState = {
      route: location.pathname,
      scrollPosition: {},
      formData: {},
      filters: {},
      viewState: {},
      ...partialState,
      timestamp: Date.now()
    };

    setCurrentContext(newContext);
    
    // Add to stack, removing any existing context for the same route
    setContextStack(prev => {
      const filtered = prev.filter(ctx => ctx.route !== location.pathname);
      return [...filtered, newContext].slice(-maxHistory);
    });
  }, [location.pathname, maxHistory]);

  // Restore context for a route
  const restoreContext = useCallback((route?: string): ContextState | null => {
    const targetRoute = route || location.pathname;
    const context = contextStack.find(ctx => ctx.route === targetRoute);
    
    if (context) {
      setCurrentContext(context);
      return context;
    }
    
    return null;
  }, [contextStack, location.pathname]);

  // Clear context for a specific route or all
  const clearContext = useCallback((route?: string) => {
    if (route) {
      setContextStack(prev => prev.filter(ctx => ctx.route !== route));
      if (currentContext?.route === route) {
        setCurrentContext(null);
      }
    } else {
      setContextStack([]);
      setCurrentContext(null);
      localStorage.removeItem(persistenceKey);
    }
  }, [currentContext, persistenceKey]);

  // Smart navigation with context preservation
  const smartNavigate = useCallback((to: string, options?: { 
    preserveQuery?: boolean;
    prefetch?: string[];
    state?: any;
  }) => {
    const { preserveQuery = false, prefetch = [], state } = options || {};
    
    // Save current context before navigation
    saveContext({
      scrollPosition: {
        [location.pathname]: window.scrollY
      },
      viewState: state || {}
    });

    // Prefetch data for target route
    prefetch.forEach(queryKey => {
      queryClient.prefetchQuery({ queryKey: [queryKey] });
    });

    // Navigate with optional query preservation
    const targetUrl = preserveQuery && location.search 
      ? `${to}${location.search}` 
      : to;
    
    navigate(targetUrl, { state });
  }, [location, navigate, saveContext, queryClient]);

  // Auto-restore scroll position
  useEffect(() => {
    const context = restoreContext();
    if (context?.scrollPosition?.[location.pathname]) {
      // Restore scroll position after DOM updates
      setTimeout(() => {
        window.scrollTo(0, context.scrollPosition[location.pathname]);
      }, 100);
    }
  }, [location.pathname]);

  // Save scroll position before unmount
  useEffect(() => {
    const handleBeforeUnload = () => {
      saveContext({
        scrollPosition: {
          [location.pathname]: window.scrollY
        }
      });
    };

    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => window.removeEventListener('beforeunload', handleBeforeUnload);
  }, [location.pathname, saveContext]);

  return {
    // Current context
    currentContext,
    contextStack,
    
    // Context management
    saveContext,
    restoreContext,
    clearContext,
    
    // Smart navigation
    smartNavigate,
    
    // Utility functions
    isContextAvailable: (route?: string) => {
      const targetRoute = route || location.pathname;
      return contextStack.some(ctx => ctx.route === targetRoute);
    },
    
    getContextAge: (route?: string) => {
      const targetRoute = route || location.pathname;
      const context = contextStack.find(ctx => ctx.route === targetRoute);
      return context ? Date.now() - context.timestamp : null;
    }
  };
}