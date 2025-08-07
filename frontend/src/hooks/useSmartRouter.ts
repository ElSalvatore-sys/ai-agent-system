import { useCallback, useEffect, useRef, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { useQueryClient } from '@tanstack/react-query';
import { useSmartContext } from './useSmartContext';
import { usePredictiveUI } from './usePredictiveUI';

interface RouteConfig {
  path: string;
  mode: 'chat' | 'builder' | 'analytics' | 'dashboard' | 'settings';
  preloadQueries?: string[];
  prefetchRoutes?: string[];
  dependencies?: string[];
  cacheStrategy?: 'aggressive' | 'conservative' | 'adaptive';
  priority?: 'high' | 'medium' | 'low';
}

interface NavigationMetrics {
  totalNavigations: number;
  averageLoadTime: number;
  cacheHitRate: number;
  routeFrequency: Map<string, number>;
  lastVisited: Map<string, number>;
  userFlow: string[];
}

interface SmartRouteState {
  currentMode: string;
  isTransitioning: boolean;
  prefetchedRoutes: Set<string>;
  loadingStates: Map<string, boolean>;
  metrics: NavigationMetrics;
  routeConfigs: Map<string, RouteConfig>;
}

export function useSmartRouter() {
  const location = useLocation();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { smartNavigate: contextNavigate, saveContext, restoreContext } = useSmartContext();
  const { trackAction, predictions, userProfile } = usePredictiveUI();

  const [state, setState] = useState<SmartRouteState>({
    currentMode: determineMode(location.pathname),
    isTransitioning: false,
    prefetchedRoutes: new Set(),
    loadingStates: new Map(),
    metrics: {
      totalNavigations: 0,
      averageLoadTime: 0,
      cacheHitRate: 0,
      routeFrequency: new Map(),
      lastVisited: new Map(),
      userFlow: []
    },
    routeConfigs: new Map()
  });

  const navigationStartTime = useRef<number>(0);
  const prefetchQueue = useRef<Set<string>>(new Set());
  const loadTimeCache = useRef<number[]>([]);

  // Initialize route configurations
  useEffect(() => {
    const configs: RouteConfig[] = [
      {
        path: '/',
        mode: 'dashboard',
        preloadQueries: ['system-health', 'recent-activity', 'usage-stats'],
        prefetchRoutes: ['/enhanced-chat', '/analytics'],
        cacheStrategy: 'aggressive',
        priority: 'high'
      },
      {
        path: '/enhanced-chat',
        mode: 'chat',
        preloadQueries: ['chat-sessions', 'ai-models', 'user-preferences'],
        prefetchRoutes: ['/analytics', '/settings'],
        dependencies: ['messages'],
        cacheStrategy: 'adaptive',
        priority: 'high'
      },
      {
        path: '/chat',
        mode: 'chat',
        preloadQueries: ['chat-sessions', 'messages'],
        prefetchRoutes: ['/enhanced-chat'],
        cacheStrategy: 'conservative',
        priority: 'medium'
      },
      {
        path: '/analytics',
        mode: 'analytics',
        preloadQueries: ['cost-data', 'performance-metrics', 'user-analytics'],
        prefetchRoutes: ['/dashboard'],
        cacheStrategy: 'aggressive',
        priority: 'medium'
      },
      {
        path: '/multi-model',
        mode: 'analytics',
        preloadQueries: ['ai-models', 'model-comparison-data'],
        dependencies: ['chat-sessions'],
        cacheStrategy: 'adaptive',
        priority: 'medium'
      },
      {
        path: '/settings',
        mode: 'settings',
        preloadQueries: ['user-preferences', 'system-config'],
        cacheStrategy: 'conservative',
        priority: 'low'
      }
    ];

    const configMap = new Map(configs.map(config => [config.path, config]));
    setState(prev => ({ ...prev, routeConfigs: configMap }));
  }, []);

  // Track navigation metrics
  useEffect(() => {
    const currentPath = location.pathname;
    const now = Date.now();
    
    // Calculate load time if we were transitioning
    if (navigationStartTime.current > 0) {
      const loadTime = now - navigationStartTime.current;
      loadTimeCache.current.push(loadTime);
      
      // Keep only last 50 load times
      if (loadTimeCache.current.length > 50) {
        loadTimeCache.current = loadTimeCache.current.slice(-50);
      }
      
      // Update average load time
      const avgLoadTime = loadTimeCache.current.reduce((sum, time) => sum + time, 0) / loadTimeCache.current.length;
      
      setState(prev => ({
        ...prev,
        isTransitioning: false,
        metrics: {
          ...prev.metrics,
          totalNavigations: prev.metrics.totalNavigations + 1,
          averageLoadTime: avgLoadTime
        }
      }));
      
      navigationStartTime.current = 0;
    }

    // Update route frequency
    setState(prev => {
      const newFrequency = new Map(prev.metrics.routeFrequency);
      newFrequency.set(currentPath, (newFrequency.get(currentPath) || 0) + 1);
      
      const newLastVisited = new Map(prev.metrics.lastVisited);
      newLastVisited.set(currentPath, now);
      
      const newUserFlow = [...prev.metrics.userFlow, currentPath].slice(-10); // Keep last 10
      
      return {
        ...prev,
        currentMode: determineMode(currentPath),
        metrics: {
          ...prev.metrics,
          routeFrequency: newFrequency,
          lastVisited: newLastVisited,
          userFlow: newUserFlow
        }
      };
    });

    // Track action for predictive UI
    trackAction({
      type: 'navigation',
      target: currentPath
    });
  }, [location.pathname, trackAction]);

  // Smart prefetching based on predictions and user patterns
  useEffect(() => {
    const prefetchBasedOnPredictions = () => {
      const predictedRoute = predictions.nextAction;
      if (predictedRoute && predictions.probability > 60) {
        const [actionType, target] = predictedRoute.split(':');
        if (actionType === 'navigation') {
          prefetchRoute(target);
        }
      }
    };

    const prefetchBasedOnFrequency = () => {
      const currentConfig = state.routeConfigs.get(location.pathname);
      if (currentConfig?.prefetchRoutes) {
        currentConfig.prefetchRoutes.forEach(route => {
          const frequency = state.metrics.routeFrequency.get(route) || 0;
          const lastVisited = state.metrics.lastVisited.get(route) || 0;
          const timeSinceVisit = Date.now() - lastVisited;
          
          // Prefetch if frequently visited and not visited recently
          if (frequency > 3 && timeSinceVisit > 60000) { // 1 minute
            prefetchRoute(route);
          }
        });
      }
    };

    const prefetchBasedOnUserProfile = () => {
      if (userProfile.expertise === 'expert') {
        // Expert users get more aggressive prefetching
        const favoriteRoutes = userProfile.usage.favoriteFeatures
          .map(feature => getRouteForFeature(feature))
          .filter(Boolean);
        
        favoriteRoutes.forEach(route => prefetchRoute(route));
      }
    };

    // Run prefetching strategies
    prefetchBasedOnPredictions();
    prefetchBasedOnFrequency();
    prefetchBasedOnUserProfile();
  }, [location.pathname, predictions, userProfile, state.metrics, state.routeConfigs]);

  // Preload data for current route
  useEffect(() => {
    const currentConfig = state.routeConfigs.get(location.pathname);
    if (currentConfig?.preloadQueries) {
      preloadQueries(currentConfig.preloadQueries, currentConfig.cacheStrategy);
    }
  }, [location.pathname, state.routeConfigs]);

  // Smart navigation function
  const smartNavigate = useCallback((to: string, options?: {
    mode?: 'chat' | 'builder' | 'analytics' | 'dashboard' | 'settings';
    preload?: boolean;
    context?: any;
    force?: boolean;
  }) => {
    const { mode, preload = true, context, force = false } = options || {};
    
    navigationStartTime.current = Date.now();
    
    setState(prev => ({ ...prev, isTransitioning: true }));
    
    // Save current context
    saveContext({
      viewState: context || {},
      formData: extractFormData(),
      filters: extractFilters()
    });

    // Preload data for target route if needed
    if (preload) {
      const targetConfig = state.routeConfigs.get(to);
      if (targetConfig?.preloadQueries) {
        preloadQueries(targetConfig.preloadQueries, targetConfig.cacheStrategy);
      }
    }

    // Navigate with mode switching logic
    if (mode && mode !== state.currentMode) {
      performModeSwitch(state.currentMode, mode, to);
    }

    // Track navigation action
    trackAction({
      type: 'navigation',
      target: to,
      context: { from: location.pathname, mode }
    });

    navigate(to);
  }, [location.pathname, state.currentMode, state.routeConfigs, saveContext, trackAction, navigate]);

  // Prefetch route data
  const prefetchRoute = useCallback((route: string) => {
    if (state.prefetchedRoutes.has(route)) return;
    
    const config = state.routeConfigs.get(route);
    if (!config) return;

    // Add to prefetch queue
    prefetchQueue.current.add(route);
    
    // Prefetch queries
    if (config.preloadQueries) {
      config.preloadQueries.forEach(queryKey => {
        queryClient.prefetchQuery({
          queryKey: [queryKey],
          staleTime: getCacheTime(config.cacheStrategy)
        });
      });
    }

    setState(prev => ({
      ...prev,
      prefetchedRoutes: new Set([...prev.prefetchedRoutes, route])
    }));
  }, [state.prefetchedRoutes, state.routeConfigs, queryClient]);

  // Preload queries for current route
  const preloadQueries = useCallback((queries: string[], strategy: string = 'conservative') => {
    const staleTime = getCacheTime(strategy);
    
    queries.forEach(queryKey => {
      // Check if already cached
      const existingData = queryClient.getQueryData([queryKey]);
      const queryState = queryClient.getQueryState([queryKey]);
      
      if (!existingData || (queryState && queryState.dataUpdatedAt < Date.now() - staleTime)) {
        queryClient.prefetchQuery({
          queryKey: [queryKey],
          staleTime
        });
      } else {
        // Cache hit
        setState(prev => ({
          ...prev,
          metrics: {
            ...prev.metrics,
            cacheHitRate: (prev.metrics.cacheHitRate * 0.9) + (0.1 * 100)
          }
        }));
      }
    });
  }, [queryClient]);

  // Mode switching logic
  const performModeSwitch = useCallback((fromMode: string, toMode: string, route: string) => {
    // Clear mode-specific state
    if (fromMode === 'chat' && toMode !== 'chat') {
      // Clear typing indicators, temporary chat state
      queryClient.invalidateQueries({ queryKey: ['typing-users'] });
    }
    
    if (fromMode === 'analytics' && toMode !== 'analytics') {
      // Clear temporary filters, selections
      queryClient.removeQueries({ 
        queryKey: ['temp-filters'],
        exact: false 
      });
    }

    // Pre-warm new mode
    const modeQueries = getModeQueries(toMode);
    preloadQueries(modeQueries, 'adaptive');
  }, [queryClient, preloadQueries]);

  // Route optimization
  const optimizeRouting = useCallback(() => {
    // Analyze user patterns and optimize cache strategies
    const routeAnalysis = Array.from(state.metrics.routeFrequency.entries())
      .map(([route, frequency]) => ({
        route,
        frequency,
        lastVisited: state.metrics.lastVisited.get(route) || 0,
        config: state.routeConfigs.get(route)
      }))
      .sort((a, b) => b.frequency - a.frequency);

    // Update cache strategies based on usage
    const updatedConfigs = new Map(state.routeConfigs);
    
    routeAnalysis.forEach(({ route, frequency, config }) => {
      if (!config) return;
      
      let newStrategy: 'aggressive' | 'conservative' | 'adaptive' = config.cacheStrategy || 'conservative';
      
      if (frequency > 10) {
        newStrategy = 'aggressive';
      } else if (frequency > 5) {
        newStrategy = 'adaptive';
      }
      
      if (newStrategy !== config.cacheStrategy) {
        updatedConfigs.set(route, { ...config, cacheStrategy: newStrategy });
      }
    });

    setState(prev => ({ ...prev, routeConfigs: updatedConfigs }));
  }, [state.metrics, state.routeConfigs]);

  // Auto-optimization on interval
  useEffect(() => {
    const interval = setInterval(optimizeRouting, 5 * 60 * 1000); // Every 5 minutes
    return () => clearInterval(interval);
  }, [optimizeRouting]);

  // Utility functions
  function determineMode(pathname: string): string {
    if (pathname.includes('chat')) return 'chat';
    if (pathname.includes('analytics') || pathname.includes('multi-model')) return 'analytics';
    if (pathname.includes('settings')) return 'settings';
    if (pathname === '/showcase') return 'builder';
    return 'dashboard';
  }

  function getCacheTime(strategy: string = 'conservative'): number {
    switch (strategy) {
      case 'aggressive': return 15 * 60 * 1000; // 15 minutes
      case 'adaptive': return 10 * 60 * 1000;   // 10 minutes
      case 'conservative': return 5 * 60 * 1000; // 5 minutes
      default: return 5 * 60 * 1000;
    }
  }

  function getModeQueries(mode: string): string[] {
    const modeQueryMap: Record<string, string[]> = {
      chat: ['chat-sessions', 'ai-models', 'messages'],
      analytics: ['cost-data', 'performance-metrics', 'usage-stats'],
      dashboard: ['system-health', 'recent-activity', 'notifications'],
      settings: ['user-preferences', 'system-config'],
      builder: ['templates', 'components', 'examples']
    };
    
    return modeQueryMap[mode] || [];
  }

  function getRouteForFeature(feature: string): string {
    const featureRouteMap: Record<string, string> = {
      'chat': '/enhanced-chat',
      'analytics': '/analytics',
      'dashboard': '/',
      'settings': '/settings',
      'multi-model': '/multi-model'
    };
    
    return featureRouteMap[feature] || '';
  }

  function extractFormData(): Record<string, any> {
    const forms = document.querySelectorAll('form');
    const data: Record<string, any> = {};
    
    forms.forEach((form, index) => {
      const formData = new FormData(form);
      const formObject: Record<string, any> = {};
      
      formData.forEach((value, key) => {
        formObject[key] = value;
      });
      
      if (Object.keys(formObject).length > 0) {
        data[`form_${index}`] = formObject;
      }
    });
    
    return data;
  }

  function extractFilters(): Record<string, any> {
    // Extract filter states from URL params and component state
    const urlParams = new URLSearchParams(location.search);
    const filters: Record<string, any> = {};
    
    urlParams.forEach((value, key) => {
      if (key.startsWith('filter_')) {
        filters[key] = value;
      }
    });
    
    return filters;
  }

  // Public API
  return {
    // Current state
    currentMode: state.currentMode,
    isTransitioning: state.isTransitioning,
    metrics: state.metrics,
    
    // Navigation
    smartNavigate,
    prefetchRoute,
    
    // Mode management
    switchMode: (mode: string) => {
      const currentRoute = location.pathname;
      const targetRoute = getModeDefaultRoute(mode);
      if (targetRoute !== currentRoute) {
        smartNavigate(targetRoute, { mode: mode as any });
      }
    },
    
    // Optimization
    optimizeRouting,
    clearPrefetchCache: () => {
      setState(prev => ({ ...prev, prefetchedRoutes: new Set() }));
      prefetchQueue.current.clear();
    },
    
    // Analytics
    getRouteAnalytics: () => ({
      totalNavigations: state.metrics.totalNavigations,
      averageLoadTime: state.metrics.averageLoadTime,
      cacheHitRate: state.metrics.cacheHitRate,
      topRoutes: Array.from(state.metrics.routeFrequency.entries())
        .sort(([,a], [,b]) => b - a)
        .slice(0, 5),
      userFlow: state.metrics.userFlow
    }),
    
    // Configuration
    updateRouteConfig: (path: string, config: Partial<RouteConfig>) => {
      setState(prev => {
        const updatedConfigs = new Map(prev.routeConfigs);
        const existing = updatedConfigs.get(path);
        if (existing) {
          updatedConfigs.set(path, { ...existing, ...config });
        }
        return { ...prev, routeConfigs: updatedConfigs };
      });
    }
  };

  function getModeDefaultRoute(mode: string): string {
    const modeRoutes: Record<string, string> = {
      chat: '/enhanced-chat',
      analytics: '/analytics',
      dashboard: '/',
      settings: '/settings',
      builder: '/showcase'
    };
    
    return modeRoutes[mode] || '/';
  }
}