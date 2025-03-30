"""
Caching system for reliability processing.

This module provides the CacheManager class, which stores and retrieves
validation results to improve performance for repeated validations.
"""

import hashlib
import time
from typing import Any, Dict, Optional, Tuple, Union

class CacheManager:
    """
    Manager for caching validation results.
    
    The CacheManager stores validation results with configurable expiry
    times to improve performance for repeated validations.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CacheManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize the cache manager state."""
        self.cache = {}
        self.default_expiry = 3600  # 1 hour default
    
    def reset(self):
        """Reset the cache manager state."""
        self._initialize()
    
    def generate_cache_key(self, data: Any) -> str:
        """
        Generate a cache key for the given data.
        
        Args:
            data: Data to generate a key for
            
        Returns:
            Cache key as a string
        """
        # Convert data to string and hash it
        try:
            data_str = str(data)
        except Exception:
            # If data can't be converted to string, use its type
            data_str = str(type(data))
        
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def set(self, key: str, value: Any, expiry: Optional[int] = None) -> bool:
        """
        Store a value in the cache with the given key and expiry.
        
        Args:
            key: Cache key
            value: Value to store
            expiry: Expiry time in seconds (None for default)
            
        Returns:
            True if successful, False otherwise
        """
        expiry_time = expiry if expiry is not None else self.default_expiry
        expiry_timestamp = time.time() + expiry_time
        
        self.cache[key] = {
            "value": value,
            "expiry": expiry_timestamp,
        }
        
        return True
    
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value, or None if not found or expired
        """
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        current_time = time.time()
        
        # Check if expired
        if current_time > entry["expiry"]:
            del self.cache[key]
            return None
        
        return entry["value"]
    
    def set_with_data(self, data: Any, value: Any, expiry: Optional[int] = None) -> str:
        """
        Store a value in the cache with a key generated from the data.
        
        Args:
            data: Data to generate a key from
            value: Value to store
            expiry: Expiry time in seconds (None for default)
            
        Returns:
            Cache key
        """
        key = self.generate_cache_key(data)
        self.set(key, value, expiry)
        return key
    
    def get_with_data(self, data: Any) -> Optional[Any]:
        """
        Retrieve a value from the cache with a key generated from the data.
        
        Args:
            data: Data to generate a key from
            
        Returns:
            Cached value, or None if not found or expired
        """
        key = self.generate_cache_key(data)
        return self.get(key)
    
    def invalidate(self, key: str) -> bool:
        """
        Invalidate a cached value.
        
        Args:
            key: Cache key
            
        Returns:
            True if successful, False if key not found
        """
        if key in self.cache:
            del self.cache[key]
            return True
        return False
    
    def invalidate_with_data(self, data: Any) -> bool:
        """
        Invalidate a cached value with a key generated from the data.
        
        Args:
            data: Data to generate a key from
            
        Returns:
            True if successful, False if key not found
        """
        key = self.generate_cache_key(data)
        return self.invalidate(key)
    
    def invalidate_all(self):
        """Invalidate all cached values."""
        self.cache.clear()
    
    def cleanup_expired(self):
        """Remove all expired entries from the cache."""
        current_time = time.time()
        expired_keys = []
        
        for key, entry in self.cache.items():
            if current_time > entry["expiry"]:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
    
    def set_default_expiry(self, seconds: int):
        """
        Set the default expiry time for cache entries.
        
        Args:
            seconds: Default expiry time in seconds
        """
        self.default_expiry = seconds
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get statistics about the cache.
        
        Returns:
            Dictionary with cache statistics
        """
        current_time = time.time()
        expired_count = 0
        
        for entry in self.cache.values():
            if current_time > entry["expiry"]:
                expired_count += 1
        
        return {
            "total_entries": len(self.cache),
            "expired_entries": expired_count,
            "valid_entries": len(self.cache) - expired_count,
        }


def save_to_cache_with_expiry(data: Any, key: str, expiry: int = 3600) -> bool:
    """
    Save data to cache with expiry time.
    
    Args:
        data: Data to cache
        key: Cache key
        expiry: Expiry time in seconds
        
    Returns:
        True if successful, False otherwise
    """
    cache_manager = CacheManager()
    return cache_manager.set(key, data, expiry)


def get_from_cache_with_expiry(key: str) -> Optional[Any]:
    """
    Get data from cache.
    
    Args:
        key: Cache key
        
    Returns:
        Cached data, or None if not found or expired
    """
    cache_manager = CacheManager()
    return cache_manager.get(key)


def save_validation_result(result: Any, context: Any, expiry: int = 3600) -> str:
    """
    Save a validation result to cache.
    
    Args:
        result: Validation result to cache
        context: Context used for validation
        expiry: Expiry time in seconds
        
    Returns:
        Cache key
    """
    cache_manager = CacheManager()
    # Create a composite key from result and context
    composite_data = {"result": str(result), "context": str(context)}
    return cache_manager.set_with_data(composite_data, result, expiry)


def get_validation_result(result: Any, context: Any) -> Optional[Any]:
    """
    Get a cached validation result.
    
    Args:
        result: Result to check
        context: Context used for validation
        
    Returns:
        Cached validation result, or None if not found or expired
    """
    cache_manager = CacheManager()
    # Create a composite key from result and context
    composite_data = {"result": str(result), "context": str(context)}
    return cache_manager.get_with_data(composite_data)
