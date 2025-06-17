import hashlib
import json
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Cache entry with TTL support"""
    data: Any
    timestamp: float
    ttl: float = 3600  # 1 hour default
    
    def is_expired(self) -> bool:
        return time.time() - self.timestamp > self.ttl

class QueryCache:
    """In-memory cache for query results with TTL"""
    
    def __init__(self, max_size: int = 1000, default_ttl: float = 3600):
        self.cache: Dict[str, CacheEntry] = {}
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.hits = 0
        self.misses = 0
    
    def _generate_key(self, query: str, top_k: int = 5) -> str:
        """Generate cache key from query parameters"""
        key_data = {"query": query.lower().strip(), "top_k": top_k}
        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
    
    def get(self, query: str, top_k: int = 5) -> Optional[Any]:
        """Get cached result if exists and not expired"""
        key = self._generate_key(query, top_k)
        
        if key in self.cache:
            entry = self.cache[key]
            if not entry.is_expired():
                self.hits += 1
                logger.debug(f"Cache hit for query: {query[:50]}...")
                return entry.data
            else:
                # Remove expired entry
                del self.cache[key]
        
        self.misses += 1
        return None
    
    def put(self, query: str, result: Any, top_k: int = 5, ttl: Optional[float] = None):
        """Cache query result"""
        key = self._generate_key(query, top_k)
        
        # Implement LRU eviction if cache is full
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k].timestamp)
            del self.cache[oldest_key]
        
        self.cache[key] = CacheEntry(
            data=result,
            timestamp=time.time(),
            ttl=ttl or self.default_ttl
        )
        logger.debug(f"Cached result for query: {query[:50]}...")
    
    def clear(self):
        """Clear all cache entries"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.2f}%",
            "cache_size": len(self.cache),
            "max_size": self.max_size
        }
