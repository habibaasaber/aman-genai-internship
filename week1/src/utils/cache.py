import json
import os
import hashlib
from typing import Any, Callable, Dict, Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ResponseCache:
    """
    A simple file-based JSON cache for storing LLM responses.
    This prevents unnecessary repeated API calls during testing.
    """
    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            
    def _generate_key(self, **kwargs) -> str:
        """Generates a unique MD5 hash for the given kwargs."""
        # Convert kwargs to a sorted JSON string to ensure consistency
        serialized = json.dumps(kwargs, sort_keys=True)
        return hashlib.md5(serialized.encode('utf-8')).hexdigest()
        
    def _get_cache_path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.json")

    def get(self, **kwargs) -> Optional[Dict[str, Any]]:
        """Retrieves a cached response if it exists."""
        key = self._generate_key(**kwargs)
        path = self._get_cache_path(key)
        
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    logger.debug(f"Cache hit for key {key}")
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read cache {path}: {e}")
        return None

    def set(self, result: Dict[str, Any], **kwargs) -> None:
        """Saves a response to the cache."""
        key = self._generate_key(**kwargs)
        path = self._get_cache_path(key)
        
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to write cache {path}: {e}")

# Global cache instance
cache = ResponseCache()
