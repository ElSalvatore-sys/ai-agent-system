"""Advanced Caching System.

Enterprise-grade caching with semantic similarity, model weight caching,
persistent model instances, and intelligent context caching.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import pickle
import time
import threading
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import numpy as np

from app.core.logger import LoggerMixin
from app.core.config import settings
from app.services.redis_cache import get_cache_manager


class CacheStrategy(str, Enum):
    """Cache strategies for different content types"""
    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"
    SEMANTIC = "semantic"
    PREDICTIVE = "predictive"
    HYBRID = "hybrid"


class SimilarityMetric(str, Enum):
    """Similarity metrics for semantic caching"""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    JACCARD = "jaccard"
    LEVENSHTEIN = "levenshtein"
    SEMANTIC_HASH = "semantic_hash"


@dataclass
class CacheItem:
    """Advanced cache item with metadata"""
    key: str
    value: Any
    category: str
    created_at: datetime
    last_accessed: datetime
    access_count: int
    size_bytes: int
    ttl_seconds: Optional[int]
    embedding: Optional[np.ndarray] = None
    similarity_threshold: float = 0.8
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    compression_ratio: float = 1.0
    priority_score: float = 0.0


@dataclass
class ModelWeightCache:
    """Model weight caching configuration"""
    model_key: str
    weight_path: Path
    checksum: str
    size_mb: float
    quantization_type: str
    last_used: datetime
    reference_count: int = 0
    is_shared: bool = True
    optimization_applied: bool = False


@dataclass
class ConversationContext:
    """Conversation context for caching"""
    conversation_id: str
    user_id: int
    messages: List[Dict[str, Any]]
    context_window: int
    last_updated: datetime
    embedding: Optional[np.ndarray] = None
    summary: Optional[str] = None
    entities: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)


class SemanticSimilarityEngine:
    """Semantic similarity engine for intelligent caching"""
    
    def __init__(self):
        self.embedding_model = None
        self.embedding_dimension = 384  # Default for sentence-transformers
        self.similarity_threshold = 0.8
        self.max_embedding_cache = 10000
        self.embedding_cache: Dict[str, np.ndarray] = {}
        self._embedding_lock = threading.Lock()
    
    async def initialize(self):
        """Initialize the semantic similarity engine"""
        try:
            # Initialize sentence transformer for embeddings
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.embedding_dimension = self.embedding_model.get_sentence_embedding_dimension()
            
        except ImportError:
            # Fallback to TF-IDF or simple hashing
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.embedding_model = TfidfVectorizer(max_features=self.embedding_dimension)
    
    async def get_text_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text"""
        try:
            # Check cache first
            text_hash = hashlib.md5(text.encode()).hexdigest()
            
            with self._embedding_lock:
                if text_hash in self.embedding_cache:
                    return self.embedding_cache[text_hash]
            
            # Generate embedding
            if hasattr(self.embedding_model, 'encode'):
                # SentenceTransformer
                embedding = self.embedding_model.encode([text])[0]
            else:
                # TF-IDF fallback
                embedding = self.embedding_model.fit_transform([text]).toarray()[0]
            
            # Cache embedding
            with self._embedding_lock:
                if len(self.embedding_cache) >= self.max_embedding_cache:
                    # Remove oldest entry
                    oldest_key = next(iter(self.embedding_cache))
                    del self.embedding_cache[oldest_key]
                
                self.embedding_cache[text_hash] = embedding
            
            return embedding
            
        except Exception as e:
            # Fallback to simple hash-based embedding
            return self._fallback_embedding(text)
    
    def _fallback_embedding(self, text: str) -> np.ndarray:
        """Fallback embedding using hash-based approach"""
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        
        # Convert hash to fixed-size embedding
        embedding = np.array([
            int(text_hash[i:i+2], 16) for i in range(0, min(len(text_hash), self.embedding_dimension * 2), 2)
        ], dtype=np.float32)
        
        # Pad or truncate to desired dimension
        if len(embedding) < self.embedding_dimension:
            embedding = np.pad(embedding, (0, self.embedding_dimension - len(embedding)))
        else:
            embedding = embedding[:self.embedding_dimension]
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    async def calculate_similarity(
        self, 
        embedding1: np.ndarray, 
        embedding2: np.ndarray,
        metric: SimilarityMetric = SimilarityMetric.COSINE
    ) -> float:
        """Calculate similarity between embeddings"""
        try:
            if metric == SimilarityMetric.COSINE:
                return float(np.dot(embedding1, embedding2) / 
                           (np.linalg.norm(embedding1) * np.linalg.norm(embedding2)))
            
            elif metric == SimilarityMetric.EUCLIDEAN:
                distance = np.linalg.norm(embedding1 - embedding2)
                return 1.0 / (1.0 + distance)  # Convert distance to similarity
            
            else:
                # Default to cosine similarity
                return float(np.dot(embedding1, embedding2) / 
                           (np.linalg.norm(embedding1) * np.linalg.norm(embedding2)))
                
        except Exception as e:
            return 0.0
    
    async def find_similar_items(
        self, 
        query_embedding: np.ndarray,
        cached_embeddings: Dict[str, np.ndarray],
        threshold: float = 0.8,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Find similar items based on embeddings"""
        similarities = []
        
        for key, embedding in cached_embeddings.items():
            similarity = await self.calculate_similarity(query_embedding, embedding)
            if similarity >= threshold:
                similarities.append((key, similarity))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


class AdvancedCachingSystem(LoggerMixin):
    """Enterprise advanced caching system"""
    
    def __init__(self):
        super().__init__()
        self.redis_cache = None
        self.semantic_engine = SemanticSimilarityEngine()
        
        # Cache storage
        self.memory_cache: Dict[str, CacheItem] = {}
        self.model_weight_cache: Dict[str, ModelWeightCache] = {}
        self.conversation_contexts: Dict[str, ConversationContext] = {}
        
        # Cache configuration
        self.max_memory_cache_size = 1000
        self.max_model_weight_cache_gb = 50.0
        self.context_window_size = 10
        self.semantic_similarity_threshold = 0.8
        self.predictive_cache_enabled = True
        
        # Directories
        self.weight_cache_dir = Path("/var/cache/llm-weights")
        self.context_cache_dir = Path("/var/cache/llm-contexts")
        self.embedding_cache_dir = Path("/var/cache/llm-embeddings")
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._optimization_task: Optional[asyncio.Task] = None
        self._predictive_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        # Thread locks
        self._cache_lock = threading.RLock()
        self._weight_lock = threading.Lock()
        self._context_lock = threading.Lock()
        
        # Statistics
        self.stats = {
            "semantic_cache_hits": 0,
            "semantic_cache_misses": 0,
            "weight_cache_hits": 0,
            "weight_cache_misses": 0,
            "context_cache_hits": 0,
            "predictive_cache_hits": 0,
            "total_cache_size_mb": 0,
            "compression_savings_mb": 0,
            "cache_efficiency_percent": 0.0
        }
    
    async def initialize(self):
        """Initialize the advanced caching system"""
        try:
            # Initialize Redis cache
            self.redis_cache = await get_cache_manager()
            
            # Initialize semantic engine
            await self.semantic_engine.initialize()
            
            # Create cache directories
            self.weight_cache_dir.mkdir(parents=True, exist_ok=True)
            self.context_cache_dir.mkdir(parents=True, exist_ok=True)
            self.embedding_cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Load existing cache data
            await self._load_persistent_cache()
            
            # Start background tasks
            self._cleanup_task = asyncio.create_task(self._cache_cleanup_loop())
            self._optimization_task = asyncio.create_task(self._cache_optimization_loop())
            if self.predictive_cache_enabled:
                self._predictive_task = asyncio.create_task(self._predictive_caching_loop())
            
            self.logger.info("Advanced caching system initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize advanced caching system: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown the caching system"""
        self.logger.info("Shutting down advanced caching system")
        self._shutdown_event.set()
        
        # Cancel background tasks
        for task in [self._cleanup_task, self._optimization_task, self._predictive_task]:
            if task:
                task.cancel()
        
        # Save persistent cache data
        await self._save_persistent_cache()
    
    async def get_semantic(
        self, 
        query: str, 
        category: str = "default",
        similarity_threshold: Optional[float] = None
    ) -> Optional[Any]:
        """Get cached item using semantic similarity"""
        try:
            threshold = similarity_threshold or self.semantic_similarity_threshold
            
            # Generate query embedding
            query_embedding = await self.semantic_engine.get_text_embedding(query)
            
            # Find similar cached items
            with self._cache_lock:
                cached_embeddings = {
                    key: item.embedding 
                    for key, item in self.memory_cache.items()
                    if item.category == category and item.embedding is not None
                }
            
            similar_items = await self.semantic_engine.find_similar_items(
                query_embedding, cached_embeddings, threshold
            )
            
            if similar_items:
                # Return the most similar item
                best_key, similarity = similar_items[0]
                cached_item = self.memory_cache[best_key]
                
                # Update access metadata
                cached_item.last_accessed = datetime.utcnow()
                cached_item.access_count += 1
                
                self.stats["semantic_cache_hits"] += 1
                self.logger.debug(f"Semantic cache hit: {best_key} (similarity: {similarity:.3f})")
                
                return cached_item.value
            
            self.stats["semantic_cache_misses"] += 1
            return None
            
        except Exception as e:
            self.logger.error(f"Semantic cache get failed: {e}")
            return None
    
    async def set_semantic(
        self, 
        key: str, 
        value: Any, 
        query: str,
        category: str = "default",
        ttl_seconds: Optional[int] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """Set cached item with semantic embedding"""
        try:
            # Generate embedding for the query
            embedding = await self.semantic_engine.get_text_embedding(query)
            
            # Create cache item
            cache_item = CacheItem(
                key=key,
                value=value,
                category=category,
                created_at=datetime.utcnow(),
                last_accessed=datetime.utcnow(),
                access_count=1,
                size_bytes=self._calculate_size(value),
                ttl_seconds=ttl_seconds,
                embedding=embedding,
                tags=tags or [],
                metadata={"query": query}
            )
            
            # Store in memory cache
            with self._cache_lock:
                self.memory_cache[key] = cache_item
                
                # Enforce cache size limit
                if len(self.memory_cache) > self.max_memory_cache_size:
                    await self._evict_lru_items()
            
            # Also store in Redis for persistence
            await self.redis_cache.set(key, value, category, ttl_seconds, tags)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Semantic cache set failed: {e}")
            return False
    
    async def cache_model_weights(
        self, 
        model_key: str,
        weights_data: Any,
        quantization_type: str = "none",
        optimization_applied: bool = False
    ) -> str:
        """Cache model weights with deduplication and compression"""
        try:
            # Calculate checksum for deduplication
            weights_bytes = pickle.dumps(weights_data)
            checksum = hashlib.sha256(weights_bytes).hexdigest()
            
            # Check if weights already cached
            existing_cache = self._find_cached_weights_by_checksum(checksum)
            if existing_cache:
                existing_cache.reference_count += 1
                existing_cache.last_used = datetime.utcnow()
                self.stats["weight_cache_hits"] += 1
                return existing_cache.weight_path
            
            # Create cache entry
            weight_filename = f"{model_key}_{checksum[:16]}_{quantization_type}.weights"
            weight_path = self.weight_cache_dir / weight_filename
            
            # Compress and save weights
            compressed_data = await self._compress_weights(weights_bytes)
            
            with open(weight_path, 'wb') as f:
                f.write(compressed_data)
            
            # Create cache metadata
            cache_entry = ModelWeightCache(
                model_key=model_key,
                weight_path=weight_path,
                checksum=checksum,
                size_mb=len(compressed_data) / 1024 / 1024,
                quantization_type=quantization_type,
                last_used=datetime.utcnow(),
                reference_count=1,
                optimization_applied=optimization_applied
            )
            
            with self._weight_lock:
                self.model_weight_cache[checksum] = cache_entry
            
            # Enforce cache size limit
            await self._enforce_weight_cache_limit()
            
            self.stats["weight_cache_misses"] += 1
            self.logger.info(f"Cached model weights: {model_key} ({cache_entry.size_mb:.1f}MB)")
            
            return str(weight_path)
            
        except Exception as e:
            self.logger.error(f"Model weight caching failed: {e}")
            raise
    
    async def load_cached_weights(self, model_key: str, checksum: str) -> Optional[Any]:
        """Load cached model weights"""
        try:
            with self._weight_lock:
                cache_entry = self.model_weight_cache.get(checksum)
            
            if not cache_entry or not cache_entry.weight_path.exists():
                return None
            
            # Load and decompress weights
            with open(cache_entry.weight_path, 'rb') as f:
                compressed_data = f.read()
            
            weights_data = await self._decompress_weights(compressed_data)
            
            # Update access metadata
            cache_entry.last_used = datetime.utcnow()
            cache_entry.reference_count += 1
            
            self.stats["weight_cache_hits"] += 1
            return pickle.loads(weights_data)
            
        except Exception as e:
            self.logger.error(f"Weight loading failed: {e}")
            return None
    
    async def cache_conversation_context(
        self, 
        conversation_id: str,
        user_id: int,
        messages: List[Dict[str, Any]],
        generate_summary: bool = True
    ) -> bool:
        """Cache conversation context with intelligent summarization"""
        try:
            # Limit context window
            recent_messages = messages[-self.context_window_size:]
            
            # Generate context embedding
            context_text = " ".join([msg.get("content", "") for msg in recent_messages])
            context_embedding = await self.semantic_engine.get_text_embedding(context_text)
            
            # Generate summary if requested
            summary = None
            if generate_summary and len(messages) > self.context_window_size:
                summary = await self._generate_conversation_summary(messages)
            
            # Extract entities and topics
            entities = await self._extract_entities(context_text)
            topics = await self._extract_topics(context_text)
            
            # Create context object
            context = ConversationContext(
                conversation_id=conversation_id,
                user_id=user_id,
                messages=recent_messages,
                context_window=self.context_window_size,
                last_updated=datetime.utcnow(),
                embedding=context_embedding,
                summary=summary,
                entities=entities,
                topics=topics
            )
            
            with self._context_lock:
                self.conversation_contexts[conversation_id] = context
            
            # Persist to disk
            await self._save_conversation_context(context)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Conversation context caching failed: {e}")
            return False
    
    async def get_conversation_context(self, conversation_id: str) -> Optional[ConversationContext]:
        """Get cached conversation context"""
        try:
            with self._context_lock:
                context = self.conversation_contexts.get(conversation_id)
            
            if context:
                self.stats["context_cache_hits"] += 1
                return context
            
            # Try loading from disk
            context = await self._load_conversation_context(conversation_id)
            if context:
                with self._context_lock:
                    self.conversation_contexts[conversation_id] = context
                self.stats["context_cache_hits"] += 1
                return context
            
            return None
            
        except Exception as e:
            self.logger.error(f"Context retrieval failed: {e}")
            return None
    
    async def predictive_cache(
        self, 
        user_id: int,
        current_context: Dict[str, Any],
        prediction_horizon: int = 5
    ) -> List[str]:
        """Predictive caching based on user patterns"""
        try:
            if not self.predictive_cache_enabled:
                return []
            
            # Analyze user patterns
            user_patterns = await self._analyze_user_patterns(user_id)
            
            # Generate predictions
            predictions = await self._generate_cache_predictions(
                user_patterns, current_context, prediction_horizon
            )
            
            # Preload predicted content
            preloaded = []
            for prediction in predictions:
                if await self._preload_content(prediction):
                    preloaded.append(prediction["key"])
            
            if preloaded:
                self.logger.info(f"Predictively cached {len(preloaded)} items for user {user_id}")
            
            return preloaded
            
        except Exception as e:
            self.logger.error(f"Predictive caching failed: {e}")
            return []
    
    async def _compress_weights(self, weights_bytes: bytes) -> bytes:
        """Compress model weights"""
        try:
            import lz4.frame
            return lz4.frame.compress(weights_bytes)
        except ImportError:
            import gzip
            return gzip.compress(weights_bytes)
    
    async def _decompress_weights(self, compressed_data: bytes) -> bytes:
        """Decompress model weights"""
        try:
            import lz4.frame
            return lz4.frame.decompress(compressed_data)
        except ImportError:
            import gzip
            return gzip.decompress(compressed_data)
    
    def _find_cached_weights_by_checksum(self, checksum: str) -> Optional[ModelWeightCache]:
        """Find cached weights by checksum"""
        with self._weight_lock:
            return self.model_weight_cache.get(checksum)
    
    async def _enforce_weight_cache_limit(self):
        """Enforce weight cache size limit"""
        try:
            with self._weight_lock:
                total_size_gb = sum(cache.size_mb for cache in self.model_weight_cache.values()) / 1024
                
                if total_size_gb > self.max_model_weight_cache_gb:
                    # Sort by last used time and reference count
                    sorted_cache = sorted(
                        self.model_weight_cache.items(),
                        key=lambda x: (x[1].reference_count, x[1].last_used)
                    )
                    
                    # Remove least used items
                    removed_size = 0
                    for checksum, cache_entry in sorted_cache:
                        if total_size_gb - removed_size / 1024 <= self.max_model_weight_cache_gb * 0.9:
                            break
                        
                        # Remove cache file
                        if cache_entry.weight_path.exists():
                            cache_entry.weight_path.unlink()
                        
                        # Remove from cache
                        del self.model_weight_cache[checksum]
                        removed_size += cache_entry.size_mb
                        
                        self.logger.info(f"Evicted cached weights: {cache_entry.model_key}")
                        
        except Exception as e:
            self.logger.error(f"Weight cache limit enforcement failed: {e}")
    
    async def _evict_lru_items(self):
        """Evict least recently used items from memory cache"""
        try:
            with self._cache_lock:
                # Sort by last accessed time
                sorted_items = sorted(
                    self.memory_cache.items(),
                    key=lambda x: x[1].last_accessed
                )
                
                # Remove oldest 10% of items
                items_to_remove = len(sorted_items) // 10
                for i in range(items_to_remove):
                    key, _ = sorted_items[i]
                    del self.memory_cache[key]
                    
        except Exception as e:
            self.logger.error(f"LRU eviction failed: {e}")
    
    async def _generate_conversation_summary(self, messages: List[Dict[str, Any]]) -> str:
        """Generate conversation summary"""
        try:
            # Simple extractive summarization
            # In production, this could use a dedicated summarization model
            
            content_messages = [msg.get("content", "") for msg in messages if msg.get("role") == "user"]
            
            if not content_messages:
                return ""
            
            # Take key sentences from each message
            summary_parts = []
            for content in content_messages[-5:]:  # Last 5 user messages
                sentences = content.split('. ')
                if sentences:
                    summary_parts.append(sentences[0])
            
            return ". ".join(summary_parts)
            
        except Exception as e:
            self.logger.error(f"Summary generation failed: {e}")
            return ""
    
    async def _extract_entities(self, text: str) -> List[str]:
        """Extract entities from text"""
        try:
            # Simple entity extraction using basic patterns
            # In production, this could use NER models
            
            import re
            
            # Extract potential entities (capitalized words, numbers, etc.)
            entities = []
            
            # Capitalized words (potential names, places)
            entities.extend(re.findall(r'\b[A-Z][a-z]+\b', text))
            
            # Numbers and dates
            entities.extend(re.findall(r'\b\d+\b', text))
            
            # Email addresses
            entities.extend(re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text))
            
            # URLs
            entities.extend(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text))
            
            return list(set(entities))  # Remove duplicates
            
        except Exception as e:
            self.logger.error(f"Entity extraction failed: {e}")
            return []
    
    async def _extract_topics(self, text: str) -> List[str]:
        """Extract topics from text"""
        try:
            # Simple topic extraction using keyword frequency
            # In production, this could use topic modeling
            
            import re
            from collections import Counter
            
            # Remove stopwords and extract meaningful words
            stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'can', 'may', 'might', 'must', 'shall', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
            
            words = re.findall(r'\b[a-z]{3,}\b', text.lower())
            meaningful_words = [word for word in words if word not in stopwords]
            
            # Get most frequent words as topics
            word_counts = Counter(meaningful_words)
            topics = [word for word, count in word_counts.most_common(10) if count > 1]
            
            return topics
            
        except Exception as e:
            self.logger.error(f"Topic extraction failed: {e}")
            return []
    
    async def _save_conversation_context(self, context: ConversationContext):
        """Save conversation context to disk"""
        try:
            context_file = self.context_cache_dir / f"{context.conversation_id}.json"
            
            # Convert to serializable format
            context_data = {
                "conversation_id": context.conversation_id,
                "user_id": context.user_id,
                "messages": context.messages,
                "context_window": context.context_window,
                "last_updated": context.last_updated.isoformat(),
                "summary": context.summary,
                "entities": context.entities,
                "topics": context.topics,
                "embedding": context.embedding.tolist() if context.embedding is not None else None
            }
            
            with open(context_file, 'w') as f:
                json.dump(context_data, f)
                
        except Exception as e:
            self.logger.error(f"Context saving failed: {e}")
    
    async def _load_conversation_context(self, conversation_id: str) -> Optional[ConversationContext]:
        """Load conversation context from disk"""
        try:
            context_file = self.context_cache_dir / f"{conversation_id}.json"
            
            if not context_file.exists():
                return None
            
            with open(context_file, 'r') as f:
                context_data = json.load(f)
            
            # Convert back to ConversationContext
            embedding = None
            if context_data.get("embedding"):
                embedding = np.array(context_data["embedding"])
            
            return ConversationContext(
                conversation_id=context_data["conversation_id"],
                user_id=context_data["user_id"],
                messages=context_data["messages"],
                context_window=context_data["context_window"],
                last_updated=datetime.fromisoformat(context_data["last_updated"]),
                embedding=embedding,
                summary=context_data.get("summary"),
                entities=context_data.get("entities", []),
                topics=context_data.get("topics", [])
            )
            
        except Exception as e:
            self.logger.error(f"Context loading failed: {e}")
            return None
    
    async def _analyze_user_patterns(self, user_id: int) -> Dict[str, Any]:
        """Analyze user patterns for predictive caching"""
        try:
            # Analyze user's conversation patterns
            patterns = {
                "frequent_topics": [],
                "common_entities": [],
                "typical_session_length": 0,
                "preferred_models": [],
                "peak_usage_hours": [],
                "conversation_flow_patterns": []
            }
            
            # This would typically analyze historical data
            # For now, return basic patterns
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"User pattern analysis failed: {e}")
            return {}
    
    async def _generate_cache_predictions(
        self, 
        user_patterns: Dict[str, Any],
        current_context: Dict[str, Any],
        prediction_horizon: int
    ) -> List[Dict[str, Any]]:
        """Generate cache predictions based on patterns"""
        try:
            predictions = []
            
            # Simple prediction logic
            # In production, this could use ML models
            
            # Predict based on frequent topics
            for topic in user_patterns.get("frequent_topics", [])[:prediction_horizon]:
                predictions.append({
                    "key": f"topic_prediction_{topic}",
                    "type": "response",
                    "priority": 0.8,
                    "query": f"Tell me about {topic}"
                })
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Cache prediction generation failed: {e}")
            return []
    
    async def _preload_content(self, prediction: Dict[str, Any]) -> bool:
        """Preload content based on prediction"""
        try:
            # This would preload the predicted content
            # For now, just simulate success
            return True
            
        except Exception as e:
            self.logger.error(f"Content preloading failed: {e}")
            return False
    
    def _calculate_size(self, obj: Any) -> int:
        """Calculate approximate size of object in bytes"""
        try:
            return len(pickle.dumps(obj))
        except:
            return len(str(obj).encode('utf-8'))
    
    async def _load_persistent_cache(self):
        """Load persistent cache data"""
        try:
            # Load model weight cache metadata
            weight_cache_file = self.weight_cache_dir / "cache_metadata.json"
            if weight_cache_file.exists():
                with open(weight_cache_file, 'r') as f:
                    weight_metadata = json.load(f)
                
                for checksum, data in weight_metadata.items():
                    cache_entry = ModelWeightCache(
                        model_key=data["model_key"],
                        weight_path=Path(data["weight_path"]),
                        checksum=checksum,
                        size_mb=data["size_mb"],
                        quantization_type=data["quantization_type"],
                        last_used=datetime.fromisoformat(data["last_used"]),
                        reference_count=data.get("reference_count", 0)
                    )
                    
                    # Only keep if file still exists
                    if cache_entry.weight_path.exists():
                        self.model_weight_cache[checksum] = cache_entry
            
        except Exception as e:
            self.logger.error(f"Persistent cache loading failed: {e}")
    
    async def _save_persistent_cache(self):
        """Save persistent cache data"""
        try:
            # Save model weight cache metadata
            weight_metadata = {}
            for checksum, cache_entry in self.model_weight_cache.items():
                weight_metadata[checksum] = {
                    "model_key": cache_entry.model_key,
                    "weight_path": str(cache_entry.weight_path),
                    "size_mb": cache_entry.size_mb,
                    "quantization_type": cache_entry.quantization_type,
                    "last_used": cache_entry.last_used.isoformat(),
                    "reference_count": cache_entry.reference_count
                }
            
            weight_cache_file = self.weight_cache_dir / "cache_metadata.json"
            with open(weight_cache_file, 'w') as f:
                json.dump(weight_metadata, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Persistent cache saving failed: {e}")
    
    async def _cache_cleanup_loop(self):
        """Background cache cleanup task"""
        while not self._shutdown_event.is_set():
            try:
                await self._cleanup_expired_items()
                await self._cleanup_orphaned_files()
                await asyncio.sleep(300)  # Run every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Cache cleanup error: {e}")
                await asyncio.sleep(300)
    
    async def _cache_optimization_loop(self):
        """Background cache optimization task"""
        while not self._shutdown_event.is_set():
            try:
                await self._optimize_cache_efficiency()
                await self._update_cache_statistics()
                await asyncio.sleep(600)  # Run every 10 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Cache optimization error: {e}")
                await asyncio.sleep(600)
    
    async def _predictive_caching_loop(self):
        """Background predictive caching task"""
        while not self._shutdown_event.is_set():
            try:
                await self._run_predictive_caching()
                await asyncio.sleep(1800)  # Run every 30 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Predictive caching error: {e}")
                await asyncio.sleep(1800)
    
    async def _cleanup_expired_items(self):
        """Clean up expired cache items"""
        try:
            current_time = datetime.utcnow()
            
            with self._cache_lock:
                expired_keys = []
                
                for key, item in self.memory_cache.items():
                    if item.ttl_seconds:
                        expiry_time = item.created_at + timedelta(seconds=item.ttl_seconds)
                        if current_time > expiry_time:
                            expired_keys.append(key)
                
                for key in expired_keys:
                    del self.memory_cache[key]
                
                if expired_keys:
                    self.logger.info(f"Cleaned up {len(expired_keys)} expired cache items")
                    
        except Exception as e:
            self.logger.error(f"Expired item cleanup failed: {e}")
    
    async def _cleanup_orphaned_files(self):
        """Clean up orphaned cache files"""
        try:
            # Clean up weight cache files not in metadata
            for weight_file in self.weight_cache_dir.glob("*.weights"):
                checksum = weight_file.stem.split('_')[-2] if '_' in weight_file.stem else None
                if checksum and checksum not in self.model_weight_cache:
                    weight_file.unlink()
                    self.logger.info(f"Removed orphaned weight file: {weight_file.name}")
                    
        except Exception as e:
            self.logger.error(f"Orphaned file cleanup failed: {e}")
    
    async def _optimize_cache_efficiency(self):
        """Optimize cache efficiency"""
        try:
            # Calculate cache hit rates
            total_hits = (self.stats["semantic_cache_hits"] + 
                         self.stats["weight_cache_hits"] + 
                         self.stats["context_cache_hits"])
            
            total_requests = (total_hits + 
                            self.stats["semantic_cache_misses"] + 
                            self.stats["weight_cache_misses"])
            
            if total_requests > 0:
                efficiency = (total_hits / total_requests) * 100
                self.stats["cache_efficiency_percent"] = efficiency
                
                if efficiency < 50:  # Low cache efficiency
                    self.logger.warning(f"Low cache efficiency: {efficiency:.1f}%")
                    # Could implement cache strategy adjustments here
                    
        except Exception as e:
            self.logger.error(f"Cache efficiency optimization failed: {e}")
    
    async def _update_cache_statistics(self):
        """Update cache statistics"""
        try:
            # Calculate total cache size
            memory_size = sum(item.size_bytes for item in self.memory_cache.values()) / 1024 / 1024
            weight_size = sum(cache.size_mb for cache in self.model_weight_cache.values())
            
            self.stats["total_cache_size_mb"] = memory_size + weight_size
            
        except Exception as e:
            self.logger.error(f"Statistics update failed: {e}")
    
    async def _run_predictive_caching(self):
        """Run predictive caching for active users"""
        try:
            # This would identify active users and run predictive caching
            # For now, just log the activity
            self.logger.debug("Running predictive caching cycle")
            
        except Exception as e:
            self.logger.error(f"Predictive caching run failed: {e}")
    
    async def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        return {
            "stats": self.stats,
            "memory_cache_size": len(self.memory_cache),
            "weight_cache_size": len(self.model_weight_cache),
            "context_cache_size": len(self.conversation_contexts),
            "cache_directories": {
                "weights": str(self.weight_cache_dir),
                "contexts": str(self.context_cache_dir),
                "embeddings": str(self.embedding_cache_dir)
            },
            "semantic_engine": {
                "embedding_dimension": self.semantic_engine.embedding_dimension,
                "cached_embeddings": len(self.semantic_engine.embedding_cache)
            }
        }


# Global instance
_caching_system: Optional[AdvancedCachingSystem] = None


async def get_advanced_caching_system() -> AdvancedCachingSystem:
    """Get the global advanced caching system instance"""
    global _caching_system
    if _caching_system is None:
        _caching_system = AdvancedCachingSystem()
        await _caching_system.initialize()
    return _caching_system