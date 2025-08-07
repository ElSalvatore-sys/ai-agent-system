import React, { Component, ErrorInfo, ReactNode, useState, useCallback } from 'react';
import { AlertTriangle, RefreshCw, Bug, ArrowLeft } from 'lucide-react';
import { Button, Card } from '@/components/ui';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
  enableRecovery?: boolean;
  maxRetries?: number;
  recoveryStrategies?: RecoveryStrategy[];
}

interface State {
  hasError: boolean;
  error?: Error;
  errorInfo?: ErrorInfo;
  retryCount: number;
  lastErrorTime: number;
  recoveryAttempted: boolean;
  performanceContext?: any;
}

interface RecoveryStrategy {
  name: string;
  condition: (error: Error, errorInfo: ErrorInfo, state: State) => boolean;
  execute: () => Promise<boolean>;
  description: string;
}

interface ErrorReport {
  error: Error;
  errorInfo: ErrorInfo;
  timestamp: number;
  url: string;
  userAgent: string;
  performanceMetrics?: any;
  componentStack: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  recoverable: boolean;
}

export class ErrorBoundary extends Component<Props, State> {
  private retryTimeouts: NodeJS.Timeout[] = [];

  constructor(props: Props) {
    super(props);
    this.state = { 
      hasError: false,
      retryCount: 0,
      lastErrorTime: 0,
      recoveryAttempted: false
    };
  }

  public static getDerivedStateFromError(error: Error): Partial<State> {
    return { 
      hasError: true, 
      error,
      lastErrorTime: Date.now()
    };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('Enhanced ErrorBoundary caught an error:', error, errorInfo);
    
    this.setState({ 
      errorInfo,
      performanceContext: this.capturePerformanceContext()
    });

    // Generate error report
    const errorReport = this.generateErrorReport(error, errorInfo);
    this.reportError(errorReport);

    // Call external error handler
    this.props.onError?.(error, errorInfo);

    // Attempt automatic recovery if enabled
    if (this.props.enableRecovery && this.state.retryCount < (this.props.maxRetries || 3)) {
      this.attemptRecovery(error, errorInfo);
    }
  }

  private capturePerformanceContext() {
    try {
      return {
        memory: (performance as any).memory ? {
          used: (performance as any).memory.usedJSHeapSize,
          total: (performance as any).memory.totalJSHeapSize
        } : null,
        timing: performance.timing ? {
          loadTime: performance.timing.loadEventEnd - performance.timing.navigationStart,
          domReady: performance.timing.domContentLoadedEventEnd - performance.timing.navigationStart
        } : null,
        navigation: performance.navigation ? {
          type: performance.navigation.type,
          redirectCount: performance.navigation.redirectCount
        } : null
      };
    } catch (e) {
      return null;
    }
  }

  private generateErrorReport(error: Error, errorInfo: ErrorInfo): ErrorReport {
    const severity = this.classifyErrorSeverity(error);
    const recoverable = this.isRecoverable(error);

    return {
      error,
      errorInfo,
      timestamp: Date.now(),
      url: window.location.href,
      userAgent: navigator.userAgent,
      performanceMetrics: this.state.performanceContext,
      componentStack: errorInfo.componentStack,
      severity,
      recoverable
    };
  }

  private classifyErrorSeverity(error: Error): 'low' | 'medium' | 'high' | 'critical' {
    const errorMessage = error.message.toLowerCase();
    const errorStack = error.stack?.toLowerCase() || '';

    if (errorMessage.includes('network') || errorMessage.includes('fetch')) {
      return 'critical';
    }
    if (errorMessage.includes('cannot read') || errorMessage.includes('undefined')) {
      return 'high';
    }
    if (errorStack.includes('react') || errorStack.includes('component')) {
      return 'medium';
    }
    return 'low';
  }

  private isRecoverable(error: Error): boolean {
    const recoverableErrors = ['ChunkLoadError', 'TypeError', 'ReferenceError'];
    return recoverableErrors.some(type => 
      error.name === type || error.message.includes(type)
    );
  }

  private async attemptRecovery(error: Error, errorInfo: ErrorInfo) {
    const strategies = this.props.recoveryStrategies || this.getDefaultRecoveryStrategies();
    
    for (const strategy of strategies) {
      if (strategy.condition(error, errorInfo, this.state)) {
        try {
          console.log(`Attempting recovery strategy: ${strategy.name}`);
          const success = await strategy.execute();
          
          if (success) {
            console.log(`Recovery strategy ${strategy.name} succeeded`);
            this.setState({ 
              hasError: false, 
              error: undefined, 
              errorInfo: undefined,
              recoveryAttempted: true
            });
            return;
          }
        } catch (recoveryError) {
          console.error(`Recovery strategy ${strategy.name} failed:`, recoveryError);
        }
      }
    }

    this.setState(prev => ({ retryCount: prev.retryCount + 1 }));
  }

  private getDefaultRecoveryStrategies(): RecoveryStrategy[] {
    return [
      {
        name: 'Clear Cache and Reload',
        description: 'Clear browser cache and reload the page',
        condition: (error) => error.name === 'ChunkLoadError',
        execute: async () => {
          if ('caches' in window) {
            const cacheNames = await caches.keys();
            await Promise.all(cacheNames.map(name => caches.delete(name)));
          }
          window.location.reload();
          return true;
        }
      },
      {
        name: 'Retry with Delay',
        description: 'Wait and retry the operation',
        condition: () => true,
        execute: async () => {
          await new Promise(resolve => setTimeout(resolve, 2000));
          return false;
        }
      }
    ];
  }

  private async reportError(errorReport: ErrorReport) {
    try {
      if (import.meta.env.PROD) {
        await fetch('/api/errors', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(errorReport)
        });
      }
    } catch (reportingError) {
      console.error('Failed to report error:', reportingError);
    }
  }

  private handleRetry = () => {
    this.setState({ 
      hasError: false, 
      error: undefined, 
      errorInfo: undefined,
      retryCount: this.state.retryCount + 1,
      recoveryAttempted: false
    });
  };

  private handleReport = () => {
    if (this.state.error && this.state.errorInfo) {
      const report = this.generateErrorReport(this.state.error, this.state.errorInfo);
      this.reportError(report);
      alert('Error report sent. Thank you for helping us improve!');
    }
  };

  public componentWillUnmount() {
    this.retryTimeouts.forEach(timeout => clearTimeout(timeout));
  }

  public render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      const { error, errorInfo, retryCount, recoveryAttempted } = this.state;
      const maxRetries = this.props.maxRetries || 3;
      const canRetry = retryCount < maxRetries;

      return (
        <Card className="p-8 text-center max-w-lg mx-auto mt-8">
          <div className="flex flex-col items-center space-y-6">
            <div className="w-16 h-16 bg-red-100 dark:bg-red-900/20 rounded-full flex items-center justify-center">
              <AlertTriangle className="h-8 w-8 text-red-600 dark:text-red-400" />
            </div>
            
            <div>
              <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
                {recoveryAttempted ? 'Recovery Attempted' : 'Something went wrong'}
              </h2>
              <p className="text-gray-600 dark:text-gray-400 mb-4">
                {recoveryAttempted 
                  ? 'We tried to recover automatically, but the issue persists.'
                  : 'An unexpected error occurred. We apologize for the inconvenience.'
                }
              </p>

              {retryCount > 0 && (
                <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-md p-3 mb-4">
                  <p className="text-sm text-yellow-800 dark:text-yellow-200">
                    Retry attempt {retryCount} of {maxRetries}
                  </p>
                </div>
              )}

              {/* Show error details in development or when requested */}
              {(import.meta.env.DEV || retryCount > 1) && error && (
                <details className="text-left mb-4">
                  <summary className="cursor-pointer text-sm font-medium text-gray-700 dark:text-gray-300 mb-2 flex items-center gap-2">
                    <Bug className="h-4 w-4" />
                    Technical Details
                  </summary>
                  <div className="mt-2 space-y-2">
                    <div>
                      <p className="text-xs font-medium text-gray-700 dark:text-gray-300">Error Message:</p>
                      <pre className="text-xs text-gray-600 dark:text-gray-400 bg-gray-100 dark:bg-gray-900 p-2 rounded overflow-auto">
                        {error.message}
                      </pre>
                    </div>
                    {errorInfo?.componentStack && (
                      <div>
                        <p className="text-xs font-medium text-gray-700 dark:text-gray-300">Component Stack:</p>
                        <pre className="text-xs text-gray-600 dark:text-gray-400 bg-gray-100 dark:bg-gray-900 p-2 rounded overflow-auto max-h-32">
                          {errorInfo.componentStack}
                        </pre>
                      </div>
                    )}
                  </div>
                </details>
              )}
            </div>

            <div className="w-full space-y-3">
              {canRetry && (
                <Button 
                  onClick={this.handleRetry} 
                  className="w-full flex items-center justify-center gap-2"
                >
                  <RefreshCw className="h-4 w-4" />
                  {retryCount === 0 ? 'Try Again' : `Retry (${maxRetries - retryCount} left)`}
                </Button>
              )}
              
              <div className="flex gap-3">
                <Button 
                  onClick={() => window.location.reload()} 
                  variant="outline" 
                  className="flex-1 flex items-center justify-center gap-2"
                >
                  <RefreshCw className="h-4 w-4" />
                  Reload Page
                </Button>
                
                <Button 
                  onClick={this.handleReport}
                  variant="outline" 
                  className="flex-1 flex items-center justify-center gap-2"
                >
                  <Bug className="h-4 w-4" />
                  Report
                </Button>
              </div>

              <Button 
                onClick={() => window.history.back()}
                variant="ghost" 
                className="w-full flex items-center justify-center gap-2 text-sm"
              >
                <ArrowLeft className="h-4 w-4" />
                Go Back
              </Button>
            </div>

            {this.state.performanceContext && (
              <div className="text-xs text-gray-500 dark:text-gray-400 pt-4 border-t w-full">
                <p>Performance context captured for debugging</p>
              </div>
            )}
          </div>
        </Card>
      );
    }

    return this.props.children;
  }
}

// Enhanced Hook version for functional components
export const useErrorHandler = () => {
  const [errorHistory, setErrorHistory] = useState<ErrorReport[]>([]);

  const handleError = useCallback((error: Error, errorInfo?: ErrorInfo) => {
    console.error('Error caught by useErrorHandler:', error, errorInfo);
    
    const errorReport: ErrorReport = {
      error,
      errorInfo: errorInfo || {} as ErrorInfo,
      timestamp: Date.now(),
      url: window.location.href,
      userAgent: navigator.userAgent,
      componentStack: errorInfo?.componentStack || '',
      severity: classifyError(error),
      recoverable: isErrorRecoverable(error)
    };

    setErrorHistory(prev => [...prev.slice(-4), errorReport]);
    
    if (process.env.NODE_ENV === 'production') {
      reportErrorToService(errorReport);
    }
  }, []);

  const clearErrorHistory = useCallback(() => {
    setErrorHistory([]);
  }, []);

  return {
    handleError,
    errorHistory,
    clearErrorHistory,
    hasRecentErrors: errorHistory.length > 0,
    lastError: errorHistory[errorHistory.length - 1] || null
  };
};

// Utility functions
function classifyError(error: Error): 'low' | 'medium' | 'high' | 'critical' {
  const message = error.message.toLowerCase();
  
  if (message.includes('network') || message.includes('fetch')) return 'critical';
  if (message.includes('cannot read') || message.includes('undefined')) return 'high';
  if (message.includes('react') || message.includes('component')) return 'medium';
  
  return 'low';
}

function isErrorRecoverable(error: Error): boolean {
  const recoverableTypes = ['ChunkLoadError', 'TypeError', 'ReferenceError'];
  return recoverableTypes.some(type => error.name === type);
}

async function reportErrorToService(errorReport: ErrorReport) {
  try {
    await fetch('/api/errors', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(errorReport)
    });
  } catch (e) {
    console.error('Failed to report error:', e);
  }
}