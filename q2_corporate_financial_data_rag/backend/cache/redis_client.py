import asyncio
import json
import hashlib
from typing import Any, Optional, Dict, List, Union
from datetime import datetime, timedelta
import redis.asyncio as aioredis
import structlog
from config import settings

logger = structlog.get_logger(__name__)


class RedisClient:
    """Async Redis client for Financial Intelligence RAG System"""
    
    def __init__(self):
        self.redis: Optional[aioredis.Redis] = None
        self.connection_pool = None
        
    async def connect(self) -> None:
        """Initialize Redis connection with connection pooling"""
        try:
            self.redis = aioredis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True,
                max_connections=20,
                retry_on_timeout=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                health_check_interval=30
            )
            
            # Test connection
            await self.redis.ping()
            logger.info("Redis connection established successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Close Redis connection"""
        if self.redis:
            await self.redis.close()
            logger.info("Redis connection closed")
    
    def _generate_cache_key(self, prefix: str, *args: str) -> str:
        """Generate consistent cache key"""
        key_parts = [settings.CACHE_KEY_PREFIX, prefix] + list(args)
        key = ":".join(key_parts)
        return key
    
    def _hash_query(self, query: str) -> str:
        """Generate hash for query string"""
        return hashlib.md5(query.encode()).hexdigest()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            if not self.redis:
                await self.connect()
            
            value = await self.redis.get(key)
            if value:
                return json.loads(value)
            return None
            
        except Exception as e:
            logger.warning(f"Cache get failed for key {key}: {e}")
            return None
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None
    ) -> bool:
        """Set value in cache with TTL"""
        try:
            if not self.redis:
                await self.connect()
            
            serialized_value = json.dumps(value, default=str)
            
            if ttl:
                await self.redis.setex(key, ttl, serialized_value)
            else:
                await self.redis.set(key, serialized_value)
            
            return True
            
        except Exception as e:
            logger.warning(f"Cache set failed for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            if not self.redis:
                await self.connect()
            
            result = await self.redis.delete(key)
            return result > 0
            
        except Exception as e:
            logger.warning(f"Cache delete failed for key {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        try:
            if not self.redis:
                await self.connect()
            
            return await self.redis.exists(key) > 0
            
        except Exception as e:
            logger.warning(f"Cache exists check failed for key {key}: {e}")
            return False
    
    async def get_ttl(self, key: str) -> int:
        """Get TTL for key"""
        try:
            if not self.redis:
                await self.connect()
            
            return await self.redis.ttl(key)
            
        except Exception as e:
            logger.warning(f"Cache TTL check failed for key {key}: {e}")
            return -1
    
    async def increment(self, key: str, amount: int = 1) -> int:
        """Increment counter"""
        try:
            if not self.redis:
                await self.connect()
            
            return await self.redis.incr(key, amount)
            
        except Exception as e:
            logger.warning(f"Cache increment failed for key {key}: {e}")
            return 0
    
    async def expire(self, key: str, ttl: int) -> bool:
        """Set expiration for key"""
        try:
            if not self.redis:
                await self.connect()
            
            return await self.redis.expire(key, ttl)
            
        except Exception as e:
            logger.warning(f"Cache expire failed for key {key}: {e}")
            return False
    
    # Financial-specific caching methods
    
    async def cache_query_result(
        self, 
        query: str, 
        result: Dict[str, Any], 
        is_realtime: bool = True
    ) -> bool:
        """Cache query result with appropriate TTL"""
        query_hash = self._hash_query(query)
        cache_key = self._generate_cache_key("query", query_hash)
        
        ttl = settings.CACHE_TTL_REALTIME if is_realtime else settings.CACHE_TTL_HISTORICAL
        
        # Add metadata
        cached_data = {
            "result": result,
            "cached_at": datetime.utcnow().isoformat(),
            "query": query,
            "is_realtime": is_realtime
        }
        
        return await self.set(cache_key, cached_data, ttl)
    
    async def get_cached_query_result(self, query: str) -> Optional[Dict[str, Any]]:
        """Get cached query result"""
        query_hash = self._hash_query(query)
        cache_key = self._generate_cache_key("query", query_hash)
        
        cached_data = await self.get(cache_key)
        if cached_data:
            logger.info(f"Cache hit for query: {query[:50]}...")
            return cached_data
        
        logger.info(f"Cache miss for query: {query[:50]}...")
        return None
    
    async def cache_financial_metrics(
        self, 
        company: str, 
        metrics: Dict[str, Any]
    ) -> bool:
        """Cache financial metrics for company"""
        cache_key = self._generate_cache_key("metrics", company.lower())
        
        cached_data = {
            "metrics": metrics,
            "cached_at": datetime.utcnow().isoformat(),
            "company": company
        }
        
        return await self.set(cache_key, cached_data, settings.FINANCIAL_METRICS_CACHE_TTL)
    
    async def get_cached_financial_metrics(self, company: str) -> Optional[Dict[str, Any]]:
        """Get cached financial metrics"""
        cache_key = self._generate_cache_key("metrics", company.lower())
        return await self.get(cache_key)
    
    async def cache_company_data(
        self, 
        company: str, 
        data: Dict[str, Any]
    ) -> bool:
        """Cache company data"""
        cache_key = self._generate_cache_key("company", company.lower())
        
        cached_data = {
            "data": data,
            "cached_at": datetime.utcnow().isoformat(),
            "company": company
        }
        
        return await self.set(cache_key, cached_data, settings.CACHE_TTL_HISTORICAL)
    
    async def get_cached_company_data(self, company: str) -> Optional[Dict[str, Any]]:
        """Get cached company data"""
        cache_key = self._generate_cache_key("company", company.lower())
        return await self.get(cache_key)
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            if not self.redis:
                await self.connect()
            
            info = await self.redis.info()
            
            stats = {
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory_human", "0"),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "total_commands_processed": info.get("total_commands_processed", 0),
            }
            
            # Calculate hit ratio
            hits = stats["keyspace_hits"]
            misses = stats["keyspace_misses"]
            total = hits + misses
            
            if total > 0:
                stats["hit_ratio"] = round((hits / total) * 100, 2)
            else:
                stats["hit_ratio"] = 0.0
            
            return stats
            
        except Exception as e:
            logger.warning(f"Failed to get cache stats: {e}")
            return {}
    
    async def clear_cache_pattern(self, pattern: str) -> int:
        """Clear cache keys matching pattern"""
        try:
            if not self.redis:
                await self.connect()
            
            keys = await self.redis.keys(pattern)
            if keys:
                return await self.redis.delete(*keys)
            return 0
            
        except Exception as e:
            logger.warning(f"Failed to clear cache pattern {pattern}: {e}")
            return 0


# Global Redis client instance
redis_client = RedisClient()


async def get_redis_client() -> RedisClient:
    """Get Redis client instance"""
    if not redis_client.redis:
        await redis_client.connect()
    return redis_client 