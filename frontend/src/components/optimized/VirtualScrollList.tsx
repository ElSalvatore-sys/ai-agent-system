import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import { cn } from '@/utils/helpers';

interface VirtualScrollItem {
  id: string | number;
  height?: number;
  data: any;
}

interface VirtualScrollProps {
  items: VirtualScrollItem[];
  itemHeight?: number;
  containerHeight: number;
  overscan?: number;
  className?: string;
  renderItem: (item: VirtualScrollItem, index: number) => React.ReactNode;
  onScroll?: (scrollTop: number, scrollDirection: 'up' | 'down') => void;
  loadMore?: () => void;
  hasNextPage?: boolean;
  isLoadingMore?: boolean;
  scrollToIndex?: number;
  scrollToAlignment?: 'start' | 'center' | 'end' | 'auto';
}

interface ScrollMetrics {
  scrollTop: number;
  scrollHeight: number;
  clientHeight: number;
  startIndex: number;
  endIndex: number;
  visibleItems: VirtualScrollItem[];
}

export const VirtualScrollList = React.memo<VirtualScrollProps>(({
  items,
  itemHeight = 50,
  containerHeight,
  overscan = 5,
  className,
  renderItem,
  onScroll,
  loadMore,
  hasNextPage = false,
  isLoadingMore = false,
  scrollToIndex,
  scrollToAlignment = 'auto'
}) => {
  const [scrollTop, setScrollTop] = useState(0);
  const [isScrolling, setIsScrolling] = useState(false);
  const [lastScrollDirection, setLastScrollDirection] = useState<'up' | 'down'>('down');
  
  const scrollElementRef = useRef<HTMLDivElement>(null);
  const scrollTimeoutRef = useRef<NodeJS.Timeout>();
  const lastScrollTopRef = useRef(0);
  const measurementsRef = useRef<Map<string | number, number>>(new Map());

  // Dynamic height support
  const itemHeights = useMemo(() => {
    return items.map(item => 
      item.height || measurementsRef.current.get(item.id) || itemHeight
    );
  }, [items, itemHeight]);

  // Calculate total height
  const totalHeight = useMemo(() => {
    return itemHeights.reduce((sum, height) => sum + height, 0);
  }, [itemHeights]);

  // Calculate visible range
  const { startIndex, endIndex, visibleItems, offsetY } = useMemo(() => {
    let accumulatedHeight = 0;
    let startIdx = 0;
    let endIdx = 0;
    let startOffset = 0;

    // Find start index
    for (let i = 0; i < items.length; i++) {
      const itemHeight = itemHeights[i];
      if (accumulatedHeight + itemHeight > scrollTop) {
        startIdx = Math.max(0, i - overscan);
        startOffset = Math.max(0, accumulatedHeight - (overscan * itemHeight));
        break;
      }
      accumulatedHeight += itemHeight;
    }

    // Find end index
    accumulatedHeight = 0;
    for (let i = 0; i < items.length; i++) {
      if (i < startIdx) {
        accumulatedHeight += itemHeights[i];
        continue;
      }
      
      if (accumulatedHeight > containerHeight + (overscan * itemHeight)) {
        endIdx = Math.min(items.length - 1, i + overscan);
        break;
      }
      
      accumulatedHeight += itemHeights[i];
      endIdx = i;
    }

    const visible = items.slice(startIdx, endIdx + 1);

    return {
      startIndex: startIdx,
      endIndex: endIdx,
      visibleItems: visible,
      offsetY: startOffset
    };
  }, [items, itemHeights, scrollTop, containerHeight, overscan]);

  // Handle scroll events
  const handleScroll = useCallback((e: React.UIEvent<HTMLDivElement>) => {
    const element = e.currentTarget;
    const newScrollTop = element.scrollTop;
    
    setScrollTop(newScrollTop);
    setIsScrolling(true);
    
    // Determine scroll direction
    const direction = newScrollTop > lastScrollTopRef.current ? 'down' : 'up';
    setLastScrollDirection(direction);
    lastScrollTopRef.current = newScrollTop;
    
    // Call onScroll callback
    onScroll?.(newScrollTop, direction);
    
    // Clear scrolling state after a delay
    if (scrollTimeoutRef.current) {
      clearTimeout(scrollTimeoutRef.current);
    }
    
    scrollTimeoutRef.current = setTimeout(() => {
      setIsScrolling(false);
    }, 150);

    // Load more items when near bottom
    if (hasNextPage && !isLoadingMore && loadMore) {
      const { scrollTop, scrollHeight, clientHeight } = element;
      const scrollPercentage = (scrollTop + clientHeight) / scrollHeight;
      
      if (scrollPercentage > 0.8) { // 80% scrolled
        loadMore();
      }
    }
  }, [onScroll, hasNextPage, isLoadingMore, loadMore]);

  // Scroll to specific index
  const scrollToItem = useCallback((index: number, alignment: 'start' | 'center' | 'end' | 'auto' = 'auto') => {
    if (!scrollElementRef.current || index < 0 || index >= items.length) return;

    let targetScrollTop = 0;
    
    // Calculate position up to target index
    for (let i = 0; i < index; i++) {
      targetScrollTop += itemHeights[i];
    }

    const itemHeight = itemHeights[index];
    
    // Adjust based on alignment
    switch (alignment) {
      case 'center':
        targetScrollTop -= (containerHeight - itemHeight) / 2;
        break;
      case 'end':
        targetScrollTop -= containerHeight - itemHeight;
        break;
      case 'auto':
        const currentScrollTop = scrollElementRef.current.scrollTop;
        const currentScrollBottom = currentScrollTop + containerHeight;
        const itemTop = targetScrollTop;
        const itemBottom = targetScrollTop + itemHeight;
        
        // Only scroll if item is not visible
        if (itemTop < currentScrollTop) {
          // Item is above viewport
          targetScrollTop = itemTop;
        } else if (itemBottom > currentScrollBottom) {
          // Item is below viewport
          targetScrollTop = itemBottom - containerHeight;
        } else {
          // Item is already visible
          return;
        }
        break;
      // 'start' case uses calculated targetScrollTop as-is
    }

    scrollElementRef.current.scrollTo({
      top: Math.max(0, Math.min(targetScrollTop, totalHeight - containerHeight)),
      behavior: 'smooth'
    });
  }, [items.length, itemHeights, containerHeight, totalHeight]);

  // Handle scrollToIndex prop
  useEffect(() => {
    if (typeof scrollToIndex === 'number') {
      scrollToItem(scrollToIndex, scrollToAlignment);
    }
  }, [scrollToIndex, scrollToAlignment, scrollToItem]);

  // Measure item heights for dynamic sizing
  const measureItem = useCallback((id: string | number, height: number) => {
    measurementsRef.current.set(id, height);
  }, []);

  // Scroll metrics for debugging/analytics
  const scrollMetrics: ScrollMetrics = useMemo(() => ({
    scrollTop,
    scrollHeight: totalHeight,
    clientHeight: containerHeight,
    startIndex,
    endIndex,
    visibleItems
  }), [scrollTop, totalHeight, containerHeight, startIndex, endIndex, visibleItems]);

  // Cleanup timeout on unmount
  useEffect(() => {
    return () => {
      if (scrollTimeoutRef.current) {
        clearTimeout(scrollTimeoutRef.current);
      }
    };
  }, []);

  return (
    <div
      ref={scrollElementRef}
      className={cn(
        "overflow-auto",
        isScrolling && "scroll-smooth",
        className
      )}
      style={{ height: containerHeight }}
      onScroll={handleScroll}
      role="listbox"
      aria-label="Virtual scroll list"
    >
      <div
        style={{
          height: totalHeight,
          position: 'relative'
        }}
      >
        <div
          style={{
            transform: `translateY(${offsetY}px)`,
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0
          }}
        >
          {visibleItems.map((item, index) => {
            const actualIndex = startIndex + index;
            return (
              <VirtualScrollItem
                key={item.id}
                item={item}
                index={actualIndex}
                height={itemHeights[actualIndex]}
                isScrolling={isScrolling}
                measureHeight={measureItem}
              >
                {renderItem(item, actualIndex)}
              </VirtualScrollItem>
            );
          })}
          
          {/* Loading indicator */}
          {isLoadingMore && (
            <div className="flex items-center justify-center py-4">
              <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-500" />
              <span className="ml-2 text-sm text-gray-500">Loading more...</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
});

VirtualScrollList.displayName = 'VirtualScrollList';

// Individual item wrapper for measurements
interface VirtualScrollItemProps {
  item: VirtualScrollItem;
  index: number;
  height: number;
  isScrolling: boolean;
  measureHeight: (id: string | number, height: number) => void;
  children: React.ReactNode;
}

const VirtualScrollItem = React.memo<VirtualScrollItemProps>(({
  item,
  index,
  height,
  isScrolling,
  measureHeight,
  children
}) => {
  const itemRef = useRef<HTMLDivElement>(null);

  // Measure actual height if dynamic
  useEffect(() => {
    if (itemRef.current && !item.height) {
      const actualHeight = itemRef.current.getBoundingClientRect().height;
      if (actualHeight !== height) {
        measureHeight(item.id, actualHeight);
      }
    }
  }, [item.id, item.height, height, measureHeight]);

  return (
    <div
      ref={itemRef}
      style={{
        height: item.height || height,
        minHeight: item.height || height
      }}
      data-index={index}
      data-item-id={item.id}
      className={cn(
        "virtual-scroll-item",
        isScrolling && "pointer-events-none" // Disable interactions while scrolling
      )}
    >
      {children}
    </div>
  );
});

VirtualScrollItem.displayName = 'VirtualScrollItem';

// Hook for using virtual scroll
export function useVirtualScroll(items: VirtualScrollItem[], containerHeight: number) {
  const [scrollTop, setScrollTop] = useState(0);
  const [isScrolling, setIsScrolling] = useState(false);

  return {
    scrollTop,
    isScrolling,
    setScrollTop,
    setIsScrolling,
    totalItems: items.length,
    metrics: {
      scrollPercentage: Math.min(100, (scrollTop / (items.length * 50 - containerHeight)) * 100) || 0
    }
  };
}