import { useEffect, useRef, useCallback, useState } from 'react';
import { useLocation } from 'react-router-dom';

interface PerformanceMetric {
  name: string;
  value: number;
  timestamp: number;
  route: string;
  metadata?: Record<string, any>;
}

interface RenderMetrics {
  componentName: string;
  renderCount: number;
  averageRenderTime: number;
  lastRenderTime: number;
  propsChanges: number;
  stateChanges: number;
}

interface PerformanceReport {
  pageLoadTime: number;
  interactionMetrics: {
    firstInput: number;
    cumulativeLayoutShift: number;
    largestContentfulPaint: number;
  };
  renderMetrics: RenderMetrics[];
  memoryUsage: {
    used: number;
    total: number;
    percentage: number;
  };
  networkMetrics: {
    slowRequests: number;
    averageRequestTime: number;
    failedRequests: number;
  };
}

export function usePerformanceMonitor(componentName?: string) {
  const location = useLocation();
  const [metrics, setMetrics] = useState<PerformanceMetric[]>([]);
  const [report, setReport] = useState<PerformanceReport | null>(null);
  
  const metricsRef = useRef<PerformanceMetric[]>([]);
  const renderCountRef = useRef(0);
  const renderTimesRef = useRef<number[]>([]);
  const lastRenderStartRef = useRef<number>(0);
  const observerRef = useRef<PerformanceObserver | null>(null);

  // Record a performance metric
  const recordMetric = useCallback((name: string, value: number, metadata?: Record<string, any>) => {
    const metric: PerformanceMetric = {
      name,
      value,
      timestamp: Date.now(),
      route: location.pathname,
      metadata
    };

    metricsRef.current.push(metric);
    setMetrics(prev => [...prev, metric]);

    // Keep only last 100 metrics
    if (metricsRef.current.length > 100) {
      metricsRef.current = metricsRef.current.slice(-100);
    }
  }, [location.pathname]);

  // Start render timing
  const startRender = useCallback(() => {
    lastRenderStartRef.current = performance.now();
  }, []);

  // End render timing
  const endRender = useCallback(() => {
    if (lastRenderStartRef.current > 0) {
      const renderTime = performance.now() - lastRenderStartRef.current;
      renderTimesRef.current.push(renderTime);
      renderCountRef.current++;

      // Keep only last 50 render times
      if (renderTimesRef.current.length > 50) {
        renderTimesRef.current = renderTimesRef.current.slice(-50);
      }

      if (componentName) {
        recordMetric(`${componentName}.renderTime`, renderTime);
      }

      lastRenderStartRef.current = 0;
    }
  }, [componentName, recordMetric]);

  // Measure component performance
  const measureComponent = useCallback(<T extends Record<string, any>>(
    name: string,
    props: T,
    prevProps?: T
  ) => {
    startRender();

    // Count prop changes
    let propChanges = 0;
    if (prevProps) {
      Object.keys(props).forEach(key => {
        if (props[key] !== prevProps[key]) {
          propChanges++;
        }
      });
    }

    if (propChanges > 0) {
      recordMetric(`${name}.propChanges`, propChanges);
    }

    // Return cleanup function
    return () => endRender();
  }, [startRender, endRender, recordMetric]);

  // Web Vitals monitoring
  useEffect(() => {
    if (!('PerformanceObserver' in window)) return;

    try {
      // Monitor Core Web Vitals
      const observer = new PerformanceObserver((list) => {
        list.getEntries().forEach((entry) => {
          switch (entry.entryType) {
            case 'largest-contentful-paint':
              recordMetric('LCP', entry.startTime);
              break;
            case 'first-input':
              recordMetric('FID', (entry as any).processingStart - entry.startTime);
              break;
            case 'layout-shift':
              if (!(entry as any).hadRecentInput) {
                recordMetric('CLS', (entry as any).value);
              }
              break;
            case 'navigation':
              const navEntry = entry as PerformanceNavigationTiming;
              recordMetric('pageLoadTime', navEntry.loadEventEnd - navEntry.navigationStart);
              recordMetric('domContentLoaded', navEntry.domContentLoadedEventEnd - navEntry.navigationStart);
              recordMetric('timeToInteractive', navEntry.loadEventEnd - navEntry.navigationStart);
              break;
            case 'resource':
              const resourceEntry = entry as PerformanceResourceTiming;
              if (resourceEntry.duration > 1000) { // Slow requests > 1s
                recordMetric('slowRequest', resourceEntry.duration, {
                  url: resourceEntry.name,
                  type: resourceEntry.initiatorType
                });
              }
              break;
          }
        });
      });

      // Observe different performance metrics
      try {
        observer.observe({ entryTypes: ['largest-contentful-paint'] });
      } catch (e) {
        // LCP not supported
      }
      
      try {
        observer.observe({ entryTypes: ['first-input'] });
      } catch (e) {
        // FID not supported
      }
      
      try {
        observer.observe({ entryTypes: ['layout-shift'] });
      } catch (e) {
        // CLS not supported
      }
      
      try {
        observer.observe({ entryTypes: ['navigation'] });
      } catch (e) {
        // Navigation timing not supported
      }
      
      try {
        observer.observe({ entryTypes: ['resource'] });
      } catch (e) {
        // Resource timing not supported
      }

      observerRef.current = observer;

      return () => {
        observer.disconnect();
      };
    } catch (error) {
      console.warn('Performance monitoring setup failed:', error);
    }
  }, [recordMetric]);

  // Memory monitoring
  useEffect(() => {
    const checkMemory = () => {
      if ('memory' in performance) {
        const memory = (performance as any).memory;
        recordMetric('memoryUsed', memory.usedJSHeapSize);
        recordMetric('memoryTotal', memory.totalJSHeapSize);
        recordMetric('memoryLimit', memory.jsHeapSizeLimit);
      }
    };

    // Check memory every 30 seconds
    const interval = setInterval(checkMemory, 30000);
    checkMemory(); // Initial check

    return () => clearInterval(interval);
  }, [recordMetric]);

  // Generate performance report
  const generateReport = useCallback((): PerformanceReport => {
    const now = Date.now();
    const last5MinMetrics = metricsRef.current.filter(m => now - m.timestamp < 5 * 60 * 1000);

    // Calculate page load time
    const pageLoadMetrics = last5MinMetrics.filter(m => m.name === 'pageLoadTime');
    const avgPageLoadTime = pageLoadMetrics.length > 0
      ? pageLoadMetrics.reduce((sum, m) => sum + m.value, 0) / pageLoadMetrics.length
      : 0;

    // Interaction metrics
    const lcpMetrics = last5MinMetrics.filter(m => m.name === 'LCP');
    const fidMetrics = last5MinMetrics.filter(m => m.name === 'FID');
    const clsMetrics = last5MinMetrics.filter(m => m.name === 'CLS');

    const interactionMetrics = {
      firstInput: fidMetrics.length > 0 ? fidMetrics[fidMetrics.length - 1].value : 0,
      cumulativeLayoutShift: clsMetrics.reduce((sum, m) => sum + m.value, 0),
      largestContentfulPaint: lcpMetrics.length > 0 ? lcpMetrics[lcpMetrics.length - 1].value : 0
    };

    // Render metrics
    const renderMetrics: RenderMetrics[] = componentName ? [{
      componentName,
      renderCount: renderCountRef.current,
      averageRenderTime: renderTimesRef.current.length > 0
        ? renderTimesRef.current.reduce((sum, time) => sum + time, 0) / renderTimesRef.current.length
        : 0,
      lastRenderTime: renderTimesRef.current[renderTimesRef.current.length - 1] || 0,
      propsChanges: last5MinMetrics.filter(m => m.name.endsWith('.propChanges')).length,
      stateChanges: last5MinMetrics.filter(m => m.name.endsWith('.stateChanges')).length
    }] : [];

    // Memory usage
    const memoryMetrics = last5MinMetrics.filter(m => m.name.startsWith('memory'));
    const latestMemoryUsed = memoryMetrics.filter(m => m.name === 'memoryUsed').slice(-1)[0]?.value || 0;
    const latestMemoryTotal = memoryMetrics.filter(m => m.name === 'memoryTotal').slice(-1)[0]?.value || 0;

    const memoryUsage = {
      used: latestMemoryUsed,
      total: latestMemoryTotal,
      percentage: latestMemoryTotal > 0 ? (latestMemoryUsed / latestMemoryTotal) * 100 : 0
    };

    // Network metrics
    const slowRequests = last5MinMetrics.filter(m => m.name === 'slowRequest');
    const requestMetrics = last5MinMetrics.filter(m => m.name.includes('request'));

    const networkMetrics = {
      slowRequests: slowRequests.length,
      averageRequestTime: slowRequests.length > 0
        ? slowRequests.reduce((sum, m) => sum + m.value, 0) / slowRequests.length
        : 0,
      failedRequests: requestMetrics.filter(m => m.metadata?.failed).length
    };

    return {
      pageLoadTime: avgPageLoadTime,
      interactionMetrics,
      renderMetrics,
      memoryUsage,
      networkMetrics
    };
  }, [componentName]);

  // Auto-generate reports
  useEffect(() => {
    const interval = setInterval(() => {
      const newReport = generateReport();
      setReport(newReport);
    }, 60000); // Every minute

    return () => clearInterval(interval);
  }, [generateReport]);

  // Performance optimizations detector
  const getOptimizationSuggestions = useCallback(() => {
    const suggestions: string[] = [];
    const currentReport = report || generateReport();

    if (currentReport.pageLoadTime > 3000) {
      suggestions.push('Page load time is slow. Consider code splitting and lazy loading.');
    }

    if (currentReport.interactionMetrics.firstInput > 100) {
      suggestions.push('First Input Delay is high. Optimize JavaScript execution.');
    }

    if (currentReport.interactionMetrics.cumulativeLayoutShift > 0.1) {
      suggestions.push('Layout shift detected. Set explicit dimensions for dynamic content.');
    }

    if (currentReport.memoryUsage.percentage > 80) {
      suggestions.push('High memory usage detected. Check for memory leaks.');
    }

    if (currentReport.renderMetrics.some(m => m.averageRenderTime > 16)) {
      suggestions.push('Slow component renders detected. Consider React.memo and optimization.');
    }

    if (currentReport.networkMetrics.slowRequests > 5) {
      suggestions.push('Multiple slow network requests. Implement request caching and optimization.');
    }

    return suggestions;
  }, [report, generateReport]);

  // Export performance data
  const exportMetrics = useCallback(() => {
    const exportData = {
      metrics: metricsRef.current,
      report: report || generateReport(),
      timestamp: new Date().toISOString(),
      userAgent: navigator.userAgent,
      url: window.location.href
    };

    return JSON.stringify(exportData, null, 2);
  }, [report, generateReport]);

  // Clear metrics
  const clearMetrics = useCallback(() => {
    metricsRef.current = [];
    setMetrics([]);
    renderCountRef.current = 0;
    renderTimesRef.current = [];
  }, []);

  return {
    // Current metrics
    metrics,
    report: report || generateReport(),
    
    // Recording functions
    recordMetric,
    measureComponent,
    startRender,
    endRender,
    
    // Analysis
    generateReport,
    getOptimizationSuggestions,
    
    // Utilities
    exportMetrics,
    clearMetrics,
    
    // Quick metrics
    averageRenderTime: renderTimesRef.current.length > 0
      ? renderTimesRef.current.reduce((sum, time) => sum + time, 0) / renderTimesRef.current.length
      : 0,
    renderCount: renderCountRef.current,
    isPerformanceGood: () => {
      const currentReport = report || generateReport();
      return currentReport.pageLoadTime < 2000 &&
             currentReport.interactionMetrics.firstInput < 100 &&
             currentReport.interactionMetrics.cumulativeLayoutShift < 0.1;
    }
  };
}

// HOC for automatic performance monitoring
export function withPerformanceMonitoring<P extends Record<string, any>>(
  Component: React.ComponentType<P>,
  componentName?: string
) {
  const WrappedComponent = React.forwardRef<any, P>((props, ref) => {
    const { measureComponent } = usePerformanceMonitor(componentName || Component.displayName || Component.name);
    const prevPropsRef = useRef<P>();

    useEffect(() => {
      const cleanup = measureComponent(componentName || Component.name || 'Unknown', props, prevPropsRef.current);
      prevPropsRef.current = props;
      return cleanup;
    });

    return <Component {...props} ref={ref} />;
  });

  WrappedComponent.displayName = `withPerformanceMonitoring(${componentName || Component.displayName || Component.name})`;
  
  return WrappedComponent;
}

// Hook for render optimization
export function useRenderOptimization<T>(value: T, deps?: React.DependencyList): T {
  const memoizedValue = React.useMemo(() => value, deps);
  const { recordMetric } = usePerformanceMonitor();
  
  // Track unnecessary re-renders
  const renderCount = useRef(0);
  renderCount.current++;
  
  useEffect(() => {
    if (renderCount.current > 1) {
      recordMetric('unnecessaryRender', renderCount.current);
    }
  });

  return memoizedValue;
}