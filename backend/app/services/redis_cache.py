"""Redis Caching Layer for Model Responses and Metadata.

Provides comprehensive caching functionality for model responses, metadata,
and system metrics to improve performance and reduce redundant operations.
"""
from __future__ import annotations

import json
import pickle
import hashlib
import logging
from typing import Any, Optional, Dict, List, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

import redis.asyncio as redis
from redis.asyncio import Redis

from app.core.config import settings
from app.core.logger import LoggerMixin


logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry metadata"""
    key: str
    value: Any
    created_at: datetime
    expires_at: Optional[datetime]
    hit_count: int
    size_bytes: int
    tags: List[str]


@dataclass
class CacheStats:
    """Cache statistics"""
    total_keys: int
    total_memory_mb: float
    hit_rate: float
    miss_rate: float
    evictions: int
    expired_keys: int
    connections: int


class RedisCacheManager(LoggerMixin):
    """Redis-based caching manager for LLM system"""
    
    def __init__(self):
        super().__init__()
        self.redis_client: Optional[Redis] = None
        self.key_prefix = "llm_cache:"
        self.default_ttl = 3600  # 1 hour
        
        # Cache configuration
        self.max_key_length = 250
        self.compression_threshold = 1024  # Compress values larger than 1KB
        self.serialize_complex_types = True
        
        # Cache categories with different TTLs
        self.cache_configs = {
            "model_response": {"ttl": 1800, "compress": True},      # 30 minutes
            "model_metadata": {"ttl": 3600, "compress": False},     # 1 hour
            "system_metrics": {"ttl": 60, "compress": False},       # 1 minute
            "user_preferences": {"ttl": 7200, "compress": False},   # 2 hours
            "routing_decisions": {"ttl": 900, "compress": False},   # 15 minutes
            "container_status": {"ttl": 300, "compress": False},    # 5 minutes
            "gpu_metrics": {"ttl": 30, "compress": False},          # 30 seconds
            "model_health": {"ttl": 600, "compress": False},        # 10 minutes
        }
    
    async def initialize(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.from_url(
                settings.REDIS_URL,
                password=settings.REDIS_PASSWORD,
                decode_responses=False,  # We handle encoding/decoding manually
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Test connection
            await self.redis_client.ping()
            
            self.logger.info("Redis cache manager initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Redis cache: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown Redis connection"""
        if self.redis_client:
            await self.redis_client.close()
            self.logger.info("Redis cache manager shutdown")
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        category: str = "default",
        ttl: Optional[int] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """Set a value in cache"""
        try:
            full_key = self._build_key(category, key)
            
            # Get cache config for category
            config = self.cache_configs.get(category, {"ttl": self.default_ttl, "compress": False})
            actual_ttl = ttl or config["ttl"]
            
            # Serialize value
            serialized_value = await self._serialize_value(value, config.get("compress", False))
            
            # Store in Redis
            await self.redis_client.setex(full_key, actual_ttl, serialized_value)
            
            # Store metadata
            await self._store_metadata(full_key, len(serialized_value), actual_ttl, tags or [])
            
            self.logger.debug(f"Cached key: {full_key} (TTL: {actual_ttl}s)")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to set cache key {key}: {e}")
            return False
    
    async def get(self, key: str, category: str = "default") -> Optional[Any]:
        """Get a value from cache"""
        try:
            full_key = self._build_key(category, key)
            
            # Get from Redis
            serialized_value = await self.redis_client.get(full_key)
            
            if serialized_value is None:
                self.logger.debug(f"Cache miss: {full_key}")
                return None
            
            # Update hit count
            await self._increment_hit_count(full_key)
            
            # Deserialize value
            value = await self._deserialize_value(serialized_value)
            
            self.logger.debug(f"Cache hit: {full_key}")
            return value
            
        except Exception as e:
            self.logger.error(f"Failed to get cache key {key}: {e}")
            return None
    
    async def delete(self, key: str, category: str = "default") -> bool:
        """Delete a value from cache"""
        try:
            full_key = self._build_key(category, key)
            
            # Delete from Redis
            deleted = await self.redis_client.delete(full_key)
            
            # Delete metadata
            await self._delete_metadata(full_key)
            
            self.logger.debug(f"Deleted cache key: {full_key}")
            return deleted > 0
            
        except Exception as e:
            self.logger.error(f"Failed to delete cache key {key}: {e}")
            return False
    
    async def exists(self, key: str, category: str = "default") -> bool:
        """Check if a key exists in cache"""
        try:
            full_key = self._build_key(category, key)
            exists = await self.redis_client.exists(full_key)
            return exists > 0
        except Exception as e:
            self.logger.error(f"Failed to check cache key existence {key}: {e}")
            return False
    
    async def get_ttl(self, key: str, category: str = "default") -> Optional[int]:
        """Get TTL for a key"""
        try:
            full_key = self._build_key(category, key)
            ttl = await self.redis_client.ttl(full_key)
            return ttl if ttl > 0 else None
        except Exception as e:
            self.logger.error(f"Failed to get TTL for cache key {key}: {e}")
            return None
    
    async def extend_ttl(self, key: str, category: str = "default", additional_seconds: int = 3600) -> bool:
        """Extend TTL for a key"""
        try:
            full_key = self._build_key(category, key)
            current_ttl = await self.redis_client.ttl(full_key)
            
            if current_ttl > 0:
                new_ttl = current_ttl + additional_seconds
                await self.redis_client.expire(full_key, new_ttl)
                return True
            
            return False
        except Exception as e:
            self.logger.error(f"Failed to extend TTL for cache key {key}: {e}")
            return False
    
    async def get_by_pattern(self, pattern: str, category: str = "default") -> Dict[str, Any]:
        """Get all keys matching a pattern"""
        try:
            full_pattern = self._build_key(category, pattern)
            keys = await self.redis_client.keys(full_pattern)
            
            result = {}
            if keys:
                values = await self.redis_client.mget(keys)
                for key, value in zip(keys, values):
                    if value is not None:
                        # Extract original key from full key
                        original_key = key.decode() if isinstance(key, bytes) else key
                        original_key = original_key.replace(f"{self.key_prefix}{category}:", "")
                        
                        result[original_key] = await self._deserialize_value(value)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to get keys by pattern {pattern}: {e}")
            return {}
    
    async def delete_by_pattern(self, pattern: str, category: str = "default") -> int:
        """Delete all keys matching a pattern"""
        try:
            full_pattern = self._build_key(category, pattern)
            keys = await self.redis_client.keys(full_pattern)
            
            if keys:
                deleted = await self.redis_client.delete(*keys)
                
                # Delete metadata for all keys
                for key in keys:
                    await self._delete_metadata(key.decode() if isinstance(key, bytes) else key)
                
                self.logger.info(f"Deleted {deleted} cache keys matching pattern: {pattern}")
                return deleted
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Failed to delete keys by pattern {pattern}: {e}")
            return 0
    
    async def clear_category(self, category: str) -> int:
        """Clear all keys in a category"""
        return await self.delete_by_pattern("*", category)
    
    async def get_cache_stats(self) -> CacheStats:
        """Get cache statistics"""
        try:
            info = await self.redis_client.info("memory")
            keyspace = await self.redis_client.info("keyspace")
            stats = await self.redis_client.info("stats")
            
            # Calculate hit rate
            hits = stats.get("keyspace_hits", 0)
            misses = stats.get("keyspace_misses", 0)
            total_operations = hits + misses
            hit_rate = (hits / total_operations * 100) if total_operations > 0 else 0
            
            # Get total keys
            total_keys = 0
            for db_info in keyspace.values():
                if isinstance(db_info, dict) and "keys" in db_info:
                    total_keys += db_info["keys"]
            
            return CacheStats(
                total_keys=total_keys,
                total_memory_mb=info.get("used_memory", 0) / 1024 / 1024,
                hit_rate=hit_rate,
                miss_rate=100 - hit_rate,
                evictions=stats.get("evicted_keys", 0),
                expired_keys=stats.get("expired_keys", 0),
                connections=info.get("connected_clients", 0)
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get cache stats: {e}")
            return CacheStats(0, 0.0, 0.0, 0.0, 0, 0, 0)
    
    async def flush_expired_keys(self) -> int:
        """Manually flush expired keys"""
        try:
            # Get all keys with metadata
            metadata_keys = await self.redis_client.keys(f"{self.key_prefix}meta:*")
            
            expired_count = 0
            for meta_key in metadata_keys:
                # Check if the actual key still exists
                actual_key = meta_key.decode().replace(f"{self.key_prefix}meta:", "")
                
                if not await self.redis_client.exists(actual_key):
                    # Key expired, clean up metadata
                    await self.redis_client.delete(meta_key)
                    expired_count += 1
            
            self.logger.info(f"Cleaned up {expired_count} expired key metadata entries")
            return expired_count
            
        except Exception as e:
            self.logger.error(f"Failed to flush expired keys: {e}")
            return 0
    
    def _build_key(self, category: str, key: str) -> str:
        """Build full cache key"""
        # Ensure key length is within limits
        if len(key) > self.max_key_length:
            # Hash long keys
            key = hashlib.md5(key.encode()).hexdigest()
        
        return f"{self.key_prefix}{category}:{key}"
    
    async def _serialize_value(self, value: Any, compress: bool = False) -> bytes:
        """Serialize value for storage"""
        try:
            if isinstance(value, (str, int, float, bool)):
                serialized = json.dumps({"type": "json", "value": value}).encode()
            else:
                # Use pickle for complex types
                serialized = pickle.dumps({"type": "pickle", "value": value})
            
            # Compress if enabled and value is large enough
            if compress and len(serialized) > self.compression_threshold:
                import gzip
                serialized = gzip.compress(serialized)
                serialized = b"compressed:" + serialized
            
            return serialized
            
        except Exception as e:
            self.logger.error(f"Failed to serialize value: {e}")
            raise
    
    async def _deserialize_value(self, serialized: bytes) -> Any:
        """Deserialize value from storage"""
        try:
            # Check if compressed
            if serialized.startswith(b"compressed:"):
                import gzip
                serialized = gzip.decompress(serialized[11:])  # Remove "compressed:" prefix
            
            # Deserialize based on type
            data = pickle.loads(serialized) if serialized.startswith(b'\x80') else json.loads(serialized.decode())
            
            if isinstance(data, dict) and "type" in data:
                return data["value"]
            else:
                return data
                
        except Exception as e:
            self.logger.error(f"Failed to deserialize value: {e}")
            raise
    
    async def _store_metadata(self, key: str, size_bytes: int, ttl: int, tags: List[str]):
        """Store metadata for a cache key"""
        try:
            metadata = {
                "created_at": datetime.utcnow().isoformat(),
                "expires_at": (datetime.utcnow() + timedelta(seconds=ttl)).isoformat(),
                "size_bytes": size_bytes,
                "hit_count": 0,
                "tags": tags
            }
            
            meta_key = f"{self.key_prefix}meta:{key}"
            await self.redis_client.setex(
                meta_key, 
                ttl + 3600,  # Keep metadata a bit longer than actual data
                json.dumps(metadata).encode()
            )
            
        except Exception as e:
            self.logger.error(f"Failed to store metadata for key {key}: {e}")
    
    async def _delete_metadata(self, key: str):
        """Delete metadata for a cache key"""
        try:
            meta_key = f"{self.key_prefix}meta:{key}"
            await self.redis_client.delete(meta_key)
        except Exception as e:
            self.logger.error(f"Failed to delete metadata for key {key}: {e}")
    
    async def _increment_hit_count(self, key: str):
        """Increment hit count for a cache key"""
        try:
            meta_key = f"{self.key_prefix}meta:{key}"
            
            # Get current metadata
            meta_data = await self.redis_client.get(meta_key)
            if meta_data:
                metadata = json.loads(meta_data.decode())
                metadata["hit_count"] += 1
                
                # Update metadata
                await self.redis_client.setex(
                    meta_key,
                    await self.redis_client.ttl(meta_key),
                    json.dumps(metadata).encode()
                )
                
        except Exception as e:
            self.logger.error(f"Failed to increment hit count for key {key}: {e}")


class ModelResponseCache:
    """Specialized cache for model responses"""
    
    def __init__(self, cache_manager: RedisCacheManager):
        self.cache = cache_manager
    
    async def cache_response(
        self, 
        model_key: str, 
        prompt_hash: str, 
        response: str,
        tokens_used: int,
        cost: float,
        response_time: float
    ):
        """Cache a model response"""
        cache_key = f"{model_key}:{prompt_hash}"
        
        cache_data = {
            "response": response,
            "tokens_used": tokens_used,
            "cost": cost,
            "response_time": response_time,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.cache.set(
            cache_key, 
            cache_data, 
            category="model_response",
            tags=["model_response", model_key]
        )
    
    async def get_cached_response(self, model_key: str, prompt_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached model response"""
        cache_key = f"{model_key}:{prompt_hash}"
        return await self.cache.get(cache_key, category="model_response")
    
    def hash_prompt(self, prompt: str, system_prompt: str = "", temperature: float = 0.7) -> str:
        """Generate hash for prompt caching"""
        content = f"{prompt}|{system_prompt}|{temperature}"
        return hashlib.sha256(content.encode()).hexdigest()


# Global instances
_cache_manager: Optional[RedisCacheManager] = None
_response_cache: Optional[ModelResponseCache] = None


async def get_cache_manager() -> RedisCacheManager:
    """Get the global cache manager instance"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = RedisCacheManager()
        await _cache_manager.initialize()
    return _cache_manager


async def get_response_cache() -> ModelResponseCache:
    """Get the global response cache instance"""
    global _response_cache
    if _response_cache is None:
        cache_manager = await get_cache_manager()
        _response_cache = ModelResponseCache(cache_manager)
    return _response_cache