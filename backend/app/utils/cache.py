import json
import pickle
import hashlib
from typing import Any, Optional, Union, Dict, List
from datetime import datetime, timedelta
import redis.asyncio as redis

from app.core.config import settings
from app.core.logger import LoggerMixin

class CacheManager(LoggerMixin):
    """Redis-based caching manager with advanced features"""
    
    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.redis: Optional[redis.Redis] = None
        self.key_prefix = "ai_agent_system:"
        self.default_ttl = 3600  # 1 hour
    
    async def connect(self):
        """Connect to Redis"""
        try:
            self.redis = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=False,  # We'll handle encoding ourselves
                password=settings.REDIS_PASSWORD
            )
            
            # Test connection
            await self.redis.ping()
            self.logger.info("Connected to Redis cache")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    async def close(self):
        """Close Redis connection"""
        if self.redis:
            await self.redis.close()
            self.logger.info("Closed Redis connection")
    
    async def ping(self) -> bool:
        """Check if Redis is available"""
        try:
            if not self.redis:
                return False
            await self.redis.ping()
            return True
        except Exception:
            return False
    
    def _make_key(self, key: str) -> str:
        """Create prefixed cache key"""
        return f"{self.key_prefix}{key}"
    
    def _serialize(self, value: Any) -> bytes:
        """Serialize value for storage"""
        if isinstance(value, (str, int, float, bool)):
            return json.dumps(value).encode('utf-8')
        else:
            return pickle.dumps(value)
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value from storage"""
        try:
            # Try JSON first (for simple types)
            return json.loads(data.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Fall back to pickle
            return pickle.loads(data)
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self.redis:
            return None
        
        try:
            cache_key = self._make_key(key)
            data = await self.redis.get(cache_key)
            
            if data is None:
                return None
            
            return self._deserialize(data)
            
        except Exception as e:
            self.logger.warning(f"Failed to get cache key {key}: {e}")
            return None
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None
    ) -> bool:
        """Set value in cache"""
        if not self.redis:
            return False
        
        try:
            cache_key = self._make_key(key)
            serialized_value = self._serialize(value)
            ttl = ttl or self.default_ttl
            
            await self.redis.setex(cache_key, ttl, serialized_value)
            return True
            
        except Exception as e:
            self.logger.warning(f"Failed to set cache key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if not self.redis:
            return False
        
        try:
            cache_key = self._make_key(key)
            result = await self.redis.delete(cache_key)
            return result > 0
            
        except Exception as e:
            self.logger.warning(f"Failed to delete cache key {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        if not self.redis:
            return False
        
        try:
            cache_key = self._make_key(key)
            result = await self.redis.exists(cache_key)
            return result > 0
            
        except Exception as e:
            self.logger.warning(f"Failed to check cache key {key}: {e}")
            return False
    
    async def expire(self, key: str, ttl: int) -> bool:
        """Set expiration time for key"""
        if not self.redis:
            return False
        
        try:
            cache_key = self._make_key(key)
            result = await self.redis.expire(cache_key, ttl)
            return result
            
        except Exception as e:
            self.logger.warning(f"Failed to set expiration for key {key}: {e}")
            return False
    
    async def get_ttl(self, key: str) -> Optional[int]:
        """Get remaining TTL for key"""
        if not self.redis:
            return None
        
        try:
            cache_key = self._make_key(key)
            ttl = await self.redis.ttl(cache_key)
            return ttl if ttl > 0 else None
            
        except Exception as e:
            self.logger.warning(f"Failed to get TTL for key {key}: {e}")
            return None

class ConversationCache(CacheManager):
    """Specialized cache for conversation data"""
    
    def __init__(self, redis_url: str):
        super().__init__(redis_url)
        self.key_prefix = "ai_agent_system:conversations:"
        self.default_ttl = 7200  # 2 hours
    
    async def get_conversation(self, conversation_id: int) -> Optional[Dict]:
        """Get conversation from cache"""
        return await self.get(f"conv:{conversation_id}")
    
    async def set_conversation(
        self, 
        conversation_id: int, 
        conversation_data: Dict,
        ttl: Optional[int] = None
    ) -> bool:
        """Cache conversation data"""
        return await self.set(f"conv:{conversation_id}", conversation_data, ttl)
    
    async def get_messages(self, conversation_id: int) -> Optional[List[Dict]]:
        """Get conversation messages from cache"""
        return await self.get(f"messages:{conversation_id}")
    
    async def set_messages(
        self, 
        conversation_id: int, 
        messages: List[Dict],
        ttl: Optional[int] = None
    ) -> bool:
        """Cache conversation messages"""
        return await self.set(f"messages:{conversation_id}", messages, ttl)
    
    async def add_message(self, conversation_id: int, message: Dict) -> bool:
        """Add a single message to cached conversation"""
        try:
            messages = await self.get_messages(conversation_id) or []
            messages.append(message)
            return await self.set_messages(conversation_id, messages)
        except Exception as e:
            self.logger.warning(f"Failed to add message to cache: {e}")
            return False
    
    async def clear_conversation(self, conversation_id: int) -> bool:
        """Clear all cached data for a conversation"""
        try:
            await self.delete(f"conv:{conversation_id}")
            await self.delete(f"messages:{conversation_id}")
            return True
        except Exception as e:
            self.logger.warning(f"Failed to clear conversation cache: {e}")
            return False

class ModelCache(CacheManager):
    """Specialized cache for AI model responses"""
    
    def __init__(self, redis_url: str):
        super().__init__(redis_url)
        self.key_prefix = "ai_agent_system:models:"
        self.default_ttl = 1800  # 30 minutes
    
    def _hash_prompt(self, prompt: str, system_prompt: str = "", **params) -> str:
        """Create hash key for prompt and parameters"""
        content = f"{prompt}|{system_prompt}|{json.dumps(params, sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    async def get_response(
        self, 
        model: str, 
        prompt: str, 
        system_prompt: str = "",
        **params
    ) -> Optional[Dict]:
        """Get cached model response"""
        prompt_hash = self._hash_prompt(prompt, system_prompt, **params)
        cache_key = f"response:{model}:{prompt_hash}"
        return await self.get(cache_key)
    
    async def set_response(
        self, 
        model: str, 
        prompt: str, 
        response: Dict,
        system_prompt: str = "",
        ttl: Optional[int] = None,
        **params
    ) -> bool:
        """Cache model response"""
        prompt_hash = self._hash_prompt(prompt, system_prompt, **params)
        cache_key = f"response:{model}:{prompt_hash}"
        
        # Add metadata
        cache_data = {
            "response": response,
            "cached_at": datetime.utcnow().isoformat(),
            "model": model,
            "params": params
        }
        
        return await self.set(cache_key, cache_data, ttl)
    
    async def invalidate_model(self, model: str) -> int:
        """Invalidate all cached responses for a model"""
        if not self.redis:
            return 0
        
        try:
            pattern = self._make_key(f"response:{model}:*")
            keys = await self.redis.keys(pattern)
            
            if keys:
                deleted = await self.redis.delete(*keys)
                self.logger.info(f"Invalidated {deleted} cached responses for model {model}")
                return deleted
            
            return 0
            
        except Exception as e:
            self.logger.warning(f"Failed to invalidate model cache: {e}")
            return 0

class UserCache(CacheManager):
    """Specialized cache for user data and sessions"""
    
    def __init__(self, redis_url: str):
        super().__init__(redis_url)
        self.key_prefix = "ai_agent_system:users:"
        self.default_ttl = 3600  # 1 hour
    
    async def get_user(self, user_id: int) -> Optional[Dict]:
        """Get user data from cache"""
        return await self.get(f"user:{user_id}")
    
    async def set_user(
        self, 
        user_id: int, 
        user_data: Dict,
        ttl: Optional[int] = None
    ) -> bool:
        """Cache user data"""
        return await self.set(f"user:{user_id}", user_data, ttl)
    
    async def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session data from cache"""
        return await self.get(f"session:{session_id}")
    
    async def set_session(
        self, 
        session_id: str, 
        session_data: Dict,
        ttl: Optional[int] = None
    ) -> bool:
        """Cache session data"""
        ttl = ttl or settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
        return await self.set(f"session:{session_id}", session_data, ttl)
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete session from cache"""
        return await self.delete(f"session:{session_id}")
    
    async def get_user_preferences(self, user_id: int) -> Optional[Dict]:
        """Get user preferences from cache"""
        return await self.get(f"prefs:{user_id}")
    
    async def set_user_preferences(
        self, 
        user_id: int, 
        preferences: Dict,
        ttl: Optional[int] = None
    ) -> bool:
        """Cache user preferences"""
        return await self.set(f"prefs:{user_id}", preferences, ttl or 7200)

class RateLimitCache(CacheManager):
    """Specialized cache for rate limiting"""
    
    def __init__(self, redis_url: str):
        super().__init__(redis_url)
        self.key_prefix = "ai_agent_system:rate_limit:"
        self.default_ttl = settings.RATE_LIMIT_WINDOW
    
    async def increment_counter(self, client_id: str, window: int = None) -> int:
        """Increment rate limit counter for client"""
        if not self.redis:
            return 0
        
        try:
            cache_key = self._make_key(f"counter:{client_id}")
            window = window or settings.RATE_LIMIT_WINDOW
            
            # Use pipeline for atomic operations
            pipe = self.redis.pipeline()
            pipe.incr(cache_key)
            pipe.expire(cache_key, window)
            results = await pipe.execute()
            
            return results[0]
            
        except Exception as e:
            self.logger.warning(f"Failed to increment rate limit counter: {e}")
            return 0
    
    async def get_counter(self, client_id: str) -> int:
        """Get current counter value for client"""
        if not self.redis:
            return 0
        
        try:
            cache_key = self._make_key(f"counter:{client_id}")
            result = await self.redis.get(cache_key)
            return int(result) if result else 0
            
        except Exception as e:
            self.logger.warning(f"Failed to get rate limit counter: {e}")
            return 0
    
    async def reset_counter(self, client_id: str) -> bool:
        """Reset counter for client"""
        return await self.delete(f"counter:{client_id}")
    
    async def is_blocked(self, client_id: str) -> bool:
        """Check if client is blocked"""
        return await self.exists(f"blocked:{client_id}")
    
    async def block_client(self, client_id: str, duration: int) -> bool:
        """Block client for specified duration"""
        return await self.set(f"blocked:{client_id}", True, duration)

def cache_key_for_prompt(prompt: str, **kwargs) -> str:
    """Generate a cache key for a prompt and parameters"""
    key_data = f"{prompt}_{json.dumps(kwargs, sort_keys=True)}"
    return hashlib.md5(key_data.encode()).hexdigest()

async def cached_response(cache: CacheManager, key: str, func, ttl: int = 3600):
    """Decorator-like function for caching responses"""
    
    # Try to get from cache first
    cached = await cache.get(key)
    if cached is not None:
        return cached
    
    # Generate new response
    result = await func() if callable(func) else func
    
    # Cache the result
    await cache.set(key, result, ttl)
    
    return result