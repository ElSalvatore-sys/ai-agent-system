import { useEffect, useRef, useCallback, useState } from 'react';
import { useLocation } from 'react-router-dom';

interface AccessibilityPreferences {
  reduceMotion: boolean;
  highContrast: boolean;
  largeFonts: boolean;
  darkMode: boolean;
  screenReader: boolean;
  keyboardNavigation: boolean;
}

interface AccessibilityState {
  preferences: AccessibilityPreferences;
  announcements: string[];
  focusedElement: HTMLElement | null;
  keyboardTrapStack: HTMLElement[];
}

interface AccessibilitySettings {
  announcePageChanges: boolean;
  autoManageFocus: boolean;
  enableKeyboardTraps: boolean;
  respectSystemPreferences: boolean;
}

export function useAccessibility(settings: Partial<AccessibilitySettings> = {}) {
  const location = useLocation();
  const announcerRef = useRef<HTMLDivElement>();
  const focusHistoryRef = useRef<HTMLElement[]>([]);
  
  const [state, setState] = useState<AccessibilityState>({
    preferences: {
      reduceMotion: false,
      highContrast: false,
      largeFonts: false,
      darkMode: false,
      screenReader: false,
      keyboardNavigation: false
    },
    announcements: [],
    focusedElement: null,
    keyboardTrapStack: []
  });

  const {
    announcePageChanges = true,
    autoManageFocus = true,
    enableKeyboardTraps = true,
    respectSystemPreferences = true
  } = settings;

  // Initialize accessibility preferences
  useEffect(() => {
    const detectPreferences = () => {
      const preferences: AccessibilityPreferences = {
        reduceMotion: respectSystemPreferences && window.matchMedia('(prefers-reduced-motion: reduce)').matches,
        highContrast: respectSystemPreferences && window.matchMedia('(prefers-contrast: high)').matches,
        largeFonts: respectSystemPreferences && window.matchMedia('(prefers-reduced-data: reduce)').matches,
        darkMode: respectSystemPreferences && window.matchMedia('(prefers-color-scheme: dark)').matches,
        screenReader: detectScreenReader(),
        keyboardNavigation: false // Will be detected on first keyboard interaction
      };

      setState(prev => ({ ...prev, preferences }));
      applyAccessibilityStyles(preferences);
    };

    detectPreferences();

    // Listen for system preference changes
    if (respectSystemPreferences) {
      const mediaQueries = [
        window.matchMedia('(prefers-reduced-motion: reduce)'),
        window.matchMedia('(prefers-contrast: high)'),
        window.matchMedia('(prefers-color-scheme: dark)')
      ];

      const handleMediaChange = () => detectPreferences();
      mediaQueries.forEach(mq => mq.addEventListener('change', handleMediaChange));

      return () => {
        mediaQueries.forEach(mq => mq.removeEventListener('change', handleMediaChange));
      };
    }
  }, [respectSystemPreferences]);

  // Screen reader detection
  const detectScreenReader = (): boolean => {
    // Check for common screen reader indicators
    if (navigator.userAgent.includes('NVDA') || 
        navigator.userAgent.includes('JAWS') || 
        navigator.userAgent.includes('VoiceOver')) {
      return true;
    }

    // Check for screen reader API
    return !!(window as any).speechSynthesis && 
           'getVoices' in (window as any).speechSynthesis;
  };

  // Apply accessibility styles
  const applyAccessibilityStyles = (preferences: AccessibilityPreferences) => {
    const root = document.documentElement;
    
    // Reduced motion
    if (preferences.reduceMotion) {
      root.style.setProperty('--animation-duration', '0.01ms');
      root.style.setProperty('--transition-duration', '0.01ms');
    } else {
      root.style.removeProperty('--animation-duration');
      root.style.removeProperty('--transition-duration');
    }

    // High contrast
    if (preferences.highContrast) {
      root.classList.add('high-contrast');
    } else {
      root.classList.remove('high-contrast');
    }

    // Large fonts
    if (preferences.largeFonts) {
      root.style.setProperty('--font-scale', '1.2');
    } else {
      root.style.removeProperty('--font-scale');
    }

    // Add accessibility classes
    root.classList.toggle('reduce-motion', preferences.reduceMotion);
    root.classList.toggle('screen-reader', preferences.screenReader);
    root.classList.toggle('keyboard-navigation', preferences.keyboardNavigation);
  };

  // Create screen reader announcer
  useEffect(() => {
    if (!announcerRef.current) {
      const announcer = document.createElement('div');
      announcer.setAttribute('aria-live', 'polite');
      announcer.setAttribute('aria-atomic', 'true');
      announcer.setAttribute('aria-relevant', 'text');
      announcer.style.position = 'absolute';
      announcer.style.left = '-10000px';
      announcer.style.width = '1px';
      announcer.style.height = '1px';
      announcer.style.overflow = 'hidden';
      document.body.appendChild(announcer);
      announcerRef.current = announcer;
    }

    return () => {
      if (announcerRef.current) {
        document.body.removeChild(announcerRef.current);
      }
    };
  }, []);

  // Announce to screen readers
  const announce = useCallback((message: string, priority: 'polite' | 'assertive' = 'polite') => {
    if (!announcerRef.current) return;

    setState(prev => ({
      ...prev,
      announcements: [...prev.announcements.slice(-4), message] // Keep last 5
    }));

    announcerRef.current.setAttribute('aria-live', priority);
    announcerRef.current.textContent = message;

    // Clear after announcement
    setTimeout(() => {
      if (announcerRef.current) {
        announcerRef.current.textContent = '';
      }
    }, 1000);
  }, []);

  // Page change announcements
  useEffect(() => {
    if (announcePageChanges && state.preferences.screenReader) {
      const pageTitle = document.title;
      const pathname = location.pathname;
      
      // Determine page type
      let pageType = 'page';
      if (pathname.includes('chat')) pageType = 'chat interface';
      else if (pathname.includes('analytics')) pageType = 'analytics dashboard';
      else if (pathname.includes('settings')) pageType = 'settings panel';
      else if (pathname === '/') pageType = 'main dashboard';

      announce(`Navigated to ${pageTitle} ${pageType}`, 'assertive');
    }
  }, [location.pathname, announce, announcePageChanges, state.preferences.screenReader]);

  // Keyboard navigation detection
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Tab') {
        setState(prev => ({
          ...prev,
          preferences: { ...prev.preferences, keyboardNavigation: true }
        }));
      }
    };

    const handleMouseDown = () => {
      setState(prev => ({
        ...prev,
        preferences: { ...prev.preferences, keyboardNavigation: false }
      }));
    };

    document.addEventListener('keydown', handleKeyDown);
    document.addEventListener('mousedown', handleMouseDown);

    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      document.removeEventListener('mousedown', handleMouseDown);
    };
  }, []);

  // Focus management
  const manageFocus = useCallback((element: HTMLElement | string) => {
    let targetElement: HTMLElement | null = null;

    if (typeof element === 'string') {
      targetElement = document.querySelector(element);
    } else {
      targetElement = element;
    }

    if (targetElement && autoManageFocus) {
      // Save current focus
      const currentFocus = document.activeElement as HTMLElement;
      if (currentFocus) {
        focusHistoryRef.current.push(currentFocus);
      }

      targetElement.focus();
      setState(prev => ({ ...prev, focusedElement: targetElement }));
    }
  }, [autoManageFocus]);

  // Restore previous focus
  const restoreFocus = useCallback(() => {
    const previousElement = focusHistoryRef.current.pop();
    if (previousElement) {
      previousElement.focus();
      setState(prev => ({ ...prev, focusedElement: previousElement }));
    }
  }, []);

  // Keyboard trap management
  const trapFocus = useCallback((container: HTMLElement) => {
    if (!enableKeyboardTraps) return () => {};

    const focusableElements = container.querySelectorAll(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );
    
    const firstElement = focusableElements[0] as HTMLElement;
    const lastElement = focusableElements[focusableElements.length - 1] as HTMLElement;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Tab') {
        if (e.shiftKey) {
          if (document.activeElement === firstElement) {
            lastElement.focus();
            e.preventDefault();
          }
        } else {
          if (document.activeElement === lastElement) {
            firstElement.focus();
            e.preventDefault();
          }
        }
      }

      if (e.key === 'Escape') {
        restoreFocus();
      }
    };

    container.addEventListener('keydown', handleKeyDown);
    setState(prev => ({
      ...prev,
      keyboardTrapStack: [...prev.keyboardTrapStack, container]
    }));

    // Focus first element
    if (firstElement) {
      firstElement.focus();
    }

    return () => {
      container.removeEventListener('keydown', handleKeyDown);
      setState(prev => ({
        ...prev,
        keyboardTrapStack: prev.keyboardTrapStack.filter(el => el !== container)
      }));
    };
  }, [enableKeyboardTraps, restoreFocus]);

  // ARIA helpers
  const setAriaLabel = useCallback((element: HTMLElement, label: string) => {
    element.setAttribute('aria-label', label);
  }, []);

  const setAriaDescription = useCallback((element: HTMLElement, description: string) => {
    let descId = element.getAttribute('aria-describedby');
    if (!descId) {
      descId = `desc-${Math.random().toString(36).substr(2, 9)}`;
      element.setAttribute('aria-describedby', descId);
    }

    let descElement = document.getElementById(descId);
    if (!descElement) {
      descElement = document.createElement('div');
      descElement.id = descId;
      descElement.style.display = 'none';
      document.body.appendChild(descElement);
    }

    descElement.textContent = description;
  }, []);

  // Color contrast checking
  const checkColorContrast = useCallback((foreground: string, background: string): number => {
    const luminance1 = getRelativeLuminance(foreground);
    const luminance2 = getRelativeLuminance(background);
    const lighter = Math.max(luminance1, luminance2);
    const darker = Math.min(luminance1, luminance2);
    return (lighter + 0.05) / (darker + 0.05);
  }, []);

  // Update preferences
  const updatePreferences = useCallback((updates: Partial<AccessibilityPreferences>) => {
    setState(prev => {
      const newPreferences = { ...prev.preferences, ...updates };
      applyAccessibilityStyles(newPreferences);
      return { ...prev, preferences: newPreferences };
    });
  }, []);

  // Accessibility audit
  const runAccessibilityAudit = useCallback(() => {
    const issues: string[] = [];
    
    // Check for missing alt text
    const images = document.querySelectorAll('img:not([alt])');
    if (images.length > 0) {
      issues.push(`${images.length} images missing alt text`);
    }

    // Check for insufficient color contrast
    const elements = document.querySelectorAll('*');
    let contrastIssues = 0;
    elements.forEach(el => {
      const styles = window.getComputedStyle(el);
      const color = styles.color;
      const backgroundColor = styles.backgroundColor;
      
      if (color !== 'rgba(0, 0, 0, 0)' && backgroundColor !== 'rgba(0, 0, 0, 0)') {
        const contrast = checkColorContrast(color, backgroundColor);
        if (contrast < 4.5) { // WCAG AA standard
          contrastIssues++;
        }
      }
    });
    
    if (contrastIssues > 0) {
      issues.push(`${contrastIssues} elements with insufficient color contrast`);
    }

    // Check for missing headings
    const headings = document.querySelectorAll('h1, h2, h3, h4, h5, h6');
    if (headings.length === 0) {
      issues.push('No heading elements found');
    }

    // Check for missing landmarks
    const landmarks = document.querySelectorAll('main, nav, aside, header, footer, [role="main"], [role="navigation"], [role="complementary"], [role="banner"], [role="contentinfo"]');
    if (landmarks.length === 0) {
      issues.push('No landmark elements found');
    }

    return {
      score: Math.max(0, 100 - (issues.length * 10)),
      issues,
      passed: issues.length === 0
    };
  }, [checkColorContrast]);

  return {
    // State
    preferences: state.preferences,
    announcements: state.announcements,
    focusedElement: state.focusedElement,
    isKeyboardNavigating: state.preferences.keyboardNavigation,
    
    // Announcements
    announce,
    
    // Focus management
    manageFocus,
    restoreFocus,
    trapFocus,
    
    // ARIA helpers
    setAriaLabel,
    setAriaDescription,
    
    // Preferences
    updatePreferences,
    
    // Utilities
    checkColorContrast,
    runAccessibilityAudit,
    
    // Keyboard helpers
    onKeyDown: (callback: (e: KeyboardEvent) => void) => {
      useEffect(() => {
        document.addEventListener('keydown', callback);
        return () => document.removeEventListener('keydown', callback);
      }, [callback]);
    },
    
    // Screen reader helpers
    isScreenReaderActive: state.preferences.screenReader,
    announceError: (error: string) => announce(`Error: ${error}`, 'assertive'),
    announceSuccess: (message: string) => announce(`Success: ${message}`, 'polite'),
    announceLoading: (message: string) => announce(`Loading: ${message}`, 'polite')
  };
}

// Utility functions
function getRelativeLuminance(color: string): number {
  const rgb = parseColor(color);
  if (!rgb) return 0;

  const [r, g, b] = rgb.map(c => {
    c = c / 255;
    return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
  });

  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

function parseColor(color: string): [number, number, number] | null {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  if (!ctx) return null;

  ctx.fillStyle = color;
  const computedColor = ctx.fillStyle;
  
  const match = computedColor.match(/^#([0-9a-f]{6})$/i);
  if (match) {
    const hex = match[1];
    return [
      parseInt(hex.substr(0, 2), 16),
      parseInt(hex.substr(2, 2), 16),
      parseInt(hex.substr(4, 2), 16)
    ];
  }

  return null;
}

// SEO Hook
export function useSEO() {
  const location = useLocation();

  const updateMeta = useCallback((meta: {
    title?: string;
    description?: string;
    keywords?: string;
    canonical?: string;
    ogTitle?: string;
    ogDescription?: string;
    ogImage?: string;
    twitterCard?: 'summary' | 'summary_large_image';
  }) => {
    // Update title
    if (meta.title) {
      document.title = meta.title;
    }

    // Update meta tags
    const updateMetaTag = (name: string, content: string, property?: boolean) => {
      const selector = property ? `meta[property="${name}"]` : `meta[name="${name}"]`;
      let element = document.querySelector(selector) as HTMLMetaElement;
      
      if (!element) {
        element = document.createElement('meta');
        if (property) {
          element.setAttribute('property', name);
        } else {
          element.setAttribute('name', name);
        }
        document.head.appendChild(element);
      }
      
      element.setAttribute('content', content);
    };

    if (meta.description) updateMetaTag('description', meta.description);
    if (meta.keywords) updateMetaTag('keywords', meta.keywords);
    if (meta.ogTitle) updateMetaTag('og:title', meta.ogTitle, true);
    if (meta.ogDescription) updateMetaTag('og:description', meta.ogDescription, true);
    if (meta.ogImage) updateMetaTag('og:image', meta.ogImage, true);
    if (meta.twitterCard) updateMetaTag('twitter:card', meta.twitterCard);

    // Update canonical URL
    if (meta.canonical) {
      let canonical = document.querySelector('link[rel="canonical"]') as HTMLLinkElement;
      if (!canonical) {
        canonical = document.createElement('link');
        canonical.rel = 'canonical';
        document.head.appendChild(canonical);
      }
      canonical.href = meta.canonical;
    }
  }, []);

  const generateStructuredData = useCallback((data: any) => {
    let script = document.querySelector('script[type="application/ld+json"]');
    if (!script) {
      script = document.createElement('script');
      script.type = 'application/ld+json';
      document.head.appendChild(script);
    }
    script.textContent = JSON.stringify(data);
  }, []);

  // Auto-update based on route
  useEffect(() => {
    const routeTitle = getRouteMeta(location.pathname);
    updateMeta(routeTitle);
  }, [location.pathname, updateMeta]);

  return {
    updateMeta,
    generateStructuredData,
    setTitle: (title: string) => updateMeta({ title }),
    setDescription: (description: string) => updateMeta({ description })
  };
}

function getRouteMeta(pathname: string) {
  const routes: Record<string, any> = {
    '/': {
      title: 'AI Agent System - Intelligent Automation Dashboard',
      description: 'Advanced AI agent management system with real-time analytics and smart automation features.',
      keywords: 'AI, agents, automation, analytics, dashboard'
    },
    '/enhanced-chat': {
      title: 'Enhanced Chat - AI Agent System',
      description: 'Interactive AI chat interface with multi-model support and advanced features.',
      keywords: 'AI chat, artificial intelligence, conversation, multi-model'
    },
    '/analytics': {
      title: 'Analytics Dashboard - AI Agent System',
      description: 'Comprehensive analytics and performance metrics for AI agents and system usage.',
      keywords: 'analytics, metrics, performance, AI insights'
    },
    '/settings': {
      title: 'Settings - AI Agent System',
      description: 'Configure your AI agent system preferences and accessibility options.',
      keywords: 'settings, configuration, preferences, accessibility'
    }
  };

  return routes[pathname] || {
    title: 'AI Agent System',
    description: 'AI Agent System - Intelligent automation and analytics platform'
  };
}