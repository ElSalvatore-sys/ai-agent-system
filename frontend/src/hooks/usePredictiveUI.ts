import { useState, useEffect, useCallback, useRef } from 'react';
import { useLocation } from 'react-router-dom';
import { useQueryClient } from '@tanstack/react-query';

interface UserAction {
  type: 'click' | 'navigation' | 'input' | 'scroll' | 'hover' | 'focus';
  target: string;
  timestamp: number;
  context: {
    route: string;
    duration?: number;
    value?: string;
    metadata?: Record<string, any>;
  };
}

interface BehaviorPattern {
  id: string;
  actions: UserAction[];
  frequency: number;
  confidence: number;
  nextPrediction?: {
    action: string;
    probability: number;
    timing: number;
  };
}

interface UIAdaptation {
  component: string;
  changes: {
    position?: 'priority' | 'secondary' | 'hidden';
    variant?: 'default' | 'prominent' | 'minimal';
    prefetch?: string[];
    suggestions?: string[];
  };
  reason: string;
  confidence: number;
}

interface PredictiveUIState {
  patterns: BehaviorPattern[];
  adaptations: UIAdaptation[];
  predictions: {
    nextAction?: string;
    probability: number;
    suggestedActions: string[];
    timeToAction?: number;
  };
  userProfile: {
    expertise: 'beginner' | 'intermediate' | 'expert';
    preferences: Record<string, any>;
    usage: {
      totalSessions: number;
      avgSessionDuration: number;
      favoriteFeatures: string[];
      painPoints: string[];
    };
  };
}

export function usePredictiveUI() {
  const location = useLocation();
  const queryClient = useQueryClient();
  const [state, setState] = useState<PredictiveUIState>({
    patterns: [],
    adaptations: [],
    predictions: {
      probability: 0,
      suggestedActions: []
    },
    userProfile: {
      expertise: 'beginner',
      preferences: {},
      usage: {
        totalSessions: 0,
        avgSessionDuration: 0,
        favoriteFeatures: [],
        painPoints: []
      }
    }
  });

  const actionsBuffer = useRef<UserAction[]>([]);
  const sessionStartTime = useRef<number>(Date.now());
  const lastActionTime = useRef<number>(Date.now());
  const predictionModel = useRef<Map<string, number>>(new Map());

  // Load existing behavior data
  useEffect(() => {
    const savedData = localStorage.getItem('predictive-ui-data');
    if (savedData) {
      try {
        const parsed = JSON.parse(savedData);
        setState(prev => ({ ...prev, ...parsed }));
        // Initialize prediction model from saved patterns
        parsed.patterns?.forEach((pattern: BehaviorPattern) => {
          const key = pattern.actions.map(a => a.type + ':' + a.target).join('->');
          predictionModel.current.set(key, pattern.confidence);
        });
      } catch (error) {
        console.warn('Failed to load predictive UI data:', error);
      }
    }
  }, []);

  // Save behavior data periodically
  const saveData = useCallback(() => {
    localStorage.setItem('predictive-ui-data', JSON.stringify(state));
  }, [state]);

  useEffect(() => {
    const interval = setInterval(saveData, 30000); // Save every 30 seconds
    return () => clearInterval(interval);
  }, [saveData]);

  // Track user action
  const trackAction = useCallback((action: Omit<UserAction, 'timestamp' | 'context'> & { context?: Partial<UserAction['context']> }) => {
    const fullAction: UserAction = {
      ...action,
      timestamp: Date.now(),
      context: {
        route: location.pathname,
        duration: Date.now() - lastActionTime.current,
        ...action.context
      }
    };

    lastActionTime.current = Date.now();
    actionsBuffer.current.push(fullAction);

    // Keep only recent actions (last 100)
    if (actionsBuffer.current.length > 100) {
      actionsBuffer.current = actionsBuffer.current.slice(-100);
    }

    // Analyze patterns with sliding window
    if (actionsBuffer.current.length >= 3) {
      analyzePatterns();
    }

    // Update predictions
    updatePredictions();
  }, [location.pathname]);

  // Pattern analysis using simple sequence mining
  const analyzePatterns = useCallback(() => {
    const actions = actionsBuffer.current.slice(-10); // Look at last 10 actions
    const patterns: BehaviorPattern[] = [];

    // Find sequences of 2-4 actions
    for (let seqLength = 2; seqLength <= Math.min(4, actions.length); seqLength++) {
      for (let i = 0; i <= actions.length - seqLength; i++) {
        const sequence = actions.slice(i, i + seqLength);
        const patternId = sequence.map(a => `${a.type}:${a.target}`).join('->');
        
        // Check if we've seen this pattern before
        let existingPattern = patterns.find(p => p.id === patternId);
        if (!existingPattern) {
          existingPattern = {
            id: patternId,
            actions: sequence,
            frequency: 0,
            confidence: 0
          };
          patterns.push(existingPattern);
        }
        
        existingPattern.frequency++;
      }
    }

    // Calculate confidence and predictions
    patterns.forEach(pattern => {
      pattern.confidence = Math.min(100, pattern.frequency * 10);
      
      // Predict next action based on historical data
      const nextActions = actionsBuffer.current.filter((_, index) => {
        const prevSequence = actionsBuffer.current.slice(Math.max(0, index - pattern.actions.length), index);
        return prevSequence.length === pattern.actions.length &&
               prevSequence.every((action, i) => 
                 action.type === pattern.actions[i].type && 
                 action.target === pattern.actions[i].target
               );
      });

      if (nextActions.length > 0) {
        const mostCommonNext = getMostCommonAction(nextActions);
        pattern.nextPrediction = {
          action: `${mostCommonNext.type}:${mostCommonNext.target}`,
          probability: (nextActions.length / pattern.frequency) * 100,
          timing: calculateAverageTimingDelay(nextActions)
        };
      }
    });

    setState(prev => ({ ...prev, patterns }));
  }, []);

  // Update current predictions
  const updatePredictions = useCallback(() => {
    const recentActions = actionsBuffer.current.slice(-3);
    if (recentActions.length === 0) return;

    const currentSequence = recentActions.map(a => `${a.type}:${a.target}`).join('->');
    
    // Find matching patterns
    const matchingPatterns = state.patterns.filter(pattern => {
      const patternStart = pattern.id.substring(0, currentSequence.length);
      return patternStart === currentSequence && pattern.nextPrediction;
    });

    if (matchingPatterns.length > 0) {
      // Get highest confidence prediction
      const bestPrediction = matchingPatterns.reduce((best, current) => 
        current.confidence > best.confidence ? current : best
      );

      setState(prev => ({
        ...prev,
        predictions: {
          nextAction: bestPrediction.nextPrediction?.action,
          probability: bestPrediction.nextPrediction?.probability || 0,
          suggestedActions: matchingPatterns
            .filter(p => p.nextPrediction)
            .map(p => p.nextPrediction!.action)
            .slice(0, 3),
          timeToAction: bestPrediction.nextPrediction?.timing
        }
      }));

      // Generate UI adaptations based on predictions
      generateAdaptations(bestPrediction);
    }
  }, [state.patterns]);

  // Generate UI adaptations
  const generateAdaptations = useCallback((pattern: BehaviorPattern) => {
    const adaptations: UIAdaptation[] = [];

    if (!pattern.nextPrediction) return;

    const [actionType, target] = pattern.nextPrediction.action.split(':');
    
    // Promote frequently used components
    if (pattern.confidence > 70) {
      adaptations.push({
        component: target,
        changes: {
          position: 'priority',
          variant: 'prominent',
          prefetch: getRelatedQueries(target)
        },
        reason: `High confidence prediction (${pattern.confidence}%) for ${target}`,
        confidence: pattern.confidence
      });
    }

    // Suggest related actions
    if (pattern.nextPrediction.probability > 60) {
      const relatedActions = findRelatedActions(target);
      adaptations.push({
        component: 'suggestions',
        changes: {
          suggestions: relatedActions
        },
        reason: `Predicted next action with ${pattern.nextPrediction.probability}% probability`,
        confidence: pattern.nextPrediction.probability
      });
    }

    setState(prev => ({ ...prev, adaptations }));
  }, []);

  // User profiling
  const updateUserProfile = useCallback(() => {
    const sessionDuration = Date.now() - sessionStartTime.current;
    const actions = actionsBuffer.current;
    
    // Determine expertise level
    let expertise: 'beginner' | 'intermediate' | 'expert' = 'beginner';
    const uniqueFeatures = new Set(actions.map(a => a.target)).size;
    const actionSpeed = actions.length / (sessionDuration / 1000 / 60); // actions per minute

    if (uniqueFeatures > 10 && actionSpeed > 5) {
      expertise = 'expert';
    } else if (uniqueFeatures > 5 && actionSpeed > 2) {
      expertise = 'intermediate';
    }

    // Find favorite features
    const featureUsage = new Map<string, number>();
    actions.forEach(action => {
      featureUsage.set(action.target, (featureUsage.get(action.target) || 0) + 1);
    });

    const favoriteFeatures = Array.from(featureUsage.entries())
      .sort(([,a], [,b]) => b - a)
      .slice(0, 5)
      .map(([feature]) => feature);

    // Identify pain points (actions with high duration or repeated attempts)
    const painPoints = actions
      .filter(action => action.context.duration && action.context.duration > 5000)
      .map(action => action.target);

    setState(prev => ({
      ...prev,
      userProfile: {
        ...prev.userProfile,
        expertise,
        usage: {
          ...prev.userProfile.usage,
          totalSessions: prev.userProfile.usage.totalSessions + 1,
          avgSessionDuration: sessionDuration,
          favoriteFeatures,
          painPoints: Array.from(new Set(painPoints))
        }
      }
    }));
  }, []);

  // Session end tracking
  useEffect(() => {
    const handleBeforeUnload = () => {
      updateUserProfile();
      saveData();
    };

    const handleVisibilityChange = () => {
      if (document.hidden) {
        updateUserProfile();
      } else {
        sessionStartTime.current = Date.now();
      }
    };

    window.addEventListener('beforeunload', handleBeforeUnload);
    document.addEventListener('visibilitychange', handleVisibilityChange);

    return () => {
      window.removeEventListener('beforeunload', handleBeforeUnload);
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, [updateUserProfile, saveData]);

  // Prefetch predicted data
  const prefetchPredictedData = useCallback(() => {
    const prediction = state.predictions.nextAction;
    if (prediction && state.predictions.probability > 70) {
      const [, target] = prediction.split(':');
      const queries = getRelatedQueries(target);
      
      queries.forEach(queryKey => {
        queryClient.prefetchQuery({ 
          queryKey: [queryKey],
          staleTime: 5 * 60 * 1000 // 5 minutes
        });
      });
    }
  }, [state.predictions, queryClient]);

  useEffect(() => {
    const timer = setTimeout(prefetchPredictedData, 1000);
    return () => clearTimeout(timer);
  }, [prefetchPredictedData]);

  // Utility functions
  const getMostCommonAction = (actions: UserAction[]): UserAction => {
    const counts = new Map<string, { action: UserAction; count: number }>();
    
    actions.forEach(action => {
      const key = `${action.type}:${action.target}`;
      const existing = counts.get(key);
      if (existing) {
        existing.count++;
      } else {
        counts.set(key, { action, count: 1 });
      }
    });

    return Array.from(counts.values())
      .reduce((max, current) => current.count > max.count ? current : max)
      .action;
  };

  const calculateAverageTimingDelay = (actions: UserAction[]): number => {
    if (actions.length < 2) return 0;
    
    const delays = actions.slice(1).map((action, index) => 
      action.timestamp - actions[index].timestamp
    );
    
    return delays.reduce((sum, delay) => sum + delay, 0) / delays.length;
  };

  const getRelatedQueries = (target: string): string[] => {
    const queryMap: Record<string, string[]> = {
      'chat': ['messages', 'sessions', 'ai-models'],
      'dashboard': ['analytics', 'usage-stats', 'system-health'],
      'analytics': ['cost-data', 'performance-metrics', 'user-stats'],
      'settings': ['user-preferences', 'ai-models', 'system-config']
    };
    
    return queryMap[target] || [];
  };

  const findRelatedActions = (target: string): string[] => {
    const actionMap: Record<string, string[]> = {
      'chat': ['New Chat', 'Upload File', 'Voice Input', 'Export Chat'],
      'dashboard': ['View Analytics', 'Check Costs', 'System Status'],
      'analytics': ['Export Data', 'Filter Results', 'Compare Models'],
      'settings': ['Change Theme', 'Update Profile', 'Configure AI']
    };
    
    return actionMap[target] || [];
  };

  // Public API
  return {
    // Current state
    patterns: state.patterns,
    adaptations: state.adaptations,
    predictions: state.predictions,
    userProfile: state.userProfile,
    
    // Action tracking
    trackAction,
    
    // Smart suggestions
    getSuggestions: () => state.predictions.suggestedActions,
    getPredictedAction: () => state.predictions.nextAction,
    getConfidence: () => state.predictions.probability,
    
    // UI adaptations
    shouldPromote: (component: string) => 
      state.adaptations.some(a => a.component === component && a.changes.position === 'priority'),
    
    shouldMinimize: (component: string) =>
      state.adaptations.some(a => a.component === component && a.changes.position === 'hidden'),
    
    getVariant: (component: string): 'default' | 'prominent' | 'minimal' => {
      const adaptation = state.adaptations.find(a => a.component === component);
      return adaptation?.changes.variant || 'default';
    },
    
    // Utilities
    resetPatterns: () => setState(prev => ({ ...prev, patterns: [], adaptations: [] })),
    exportData: () => JSON.stringify(state, null, 2),
    importData: (data: string) => {
      try {
        const imported = JSON.parse(data);
        setState(imported);
      } catch (error) {
        console.error('Failed to import predictive UI data:', error);
      }
    }
  };
}