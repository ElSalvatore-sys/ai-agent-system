import { useQueryClient } from '@tanstack/react-query';
import { useEffect } from 'react';

interface CacheEntry {
  key: string;
  prompt: string;
  modelId: string;
  response: string; // could be Message or AIResponse, kept generic
  createdAt: number;
}

export function useCacheManager() {
  const queryClient = useQueryClient();

  /* --------------------------- DEDUPLICATION ------------------------------ */
  const getCachedResponse = (prompt: string, modelId: string) => {
    const key = getKey(prompt, modelId);
    return queryClient.getQueryData<CacheEntry>(['ai-cache', key]);
  };

  const cacheResponse = (prompt: string, modelId: string, response: any) => {
    const key = getKey(prompt, modelId);
    const entry: CacheEntry = {
      key,
      prompt,
      modelId,
      response,
      createdAt: Date.now(),
    };
    queryClient.setQueryData(['ai-cache', key], entry);
  };

  /* -------------------------- PREDICTIVE PREFETCH ------------------------- */
  const prefetch = (prompt: string, modelId: string, fetcher: () => Promise<any>) => {
    const key = getKey(prompt, modelId);
    queryClient.prefetchQuery(['ai-cache', key], fetcher, {
      staleTime: 1000 * 60 * 60, // 1h
    });
  };

  /* ------------------------- INTELLIGENT INVALIDATION --------------------- */
  useEffect(() => {
    const interval = setInterval(() => {
      const queries = queryClient.getQueryCache().findAll({ predicate: (q) => q.queryKey[0] === 'ai-cache' });
      const now = Date.now();
      queries.forEach((q) => {
        const data = q.state.data as CacheEntry | undefined;
        if (data && now - data.createdAt > 1000 * 60 * 60 * 24) {
          // Invalidate after 24h
          queryClient.removeQueries({ queryKey: q.queryKey });
        }
      });
    }, 1000 * 60 * 10); // run every 10 min
    return () => clearInterval(interval);
  }, [queryClient]);

  /* ------------------------------- OFFLINE -------------------------------- */
  // react-query persists cache via window.localStorage hydration in main.tsx using PersistQueryClientProvider (TODO)

  return {
    getCachedResponse,
    cacheResponse,
    prefetch,
  };
}

function getKey(prompt: string, modelId: string) {
  const raw = `${modelId}:${prompt}`;
  // naive hash: djb2
  let hash = 5381;
  for (let i = 0; i < raw.length; i++) {
    hash = (hash * 33) ^ raw.charCodeAt(i);
  }
  return (hash >>> 0).toString(16);
}
