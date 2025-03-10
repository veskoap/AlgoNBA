"""
Cache management utilities for AlgoNBA to avoid redundant processing.
"""
import os
import pickle
import json
import hashlib
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

class CacheManager:
    """
    Manages data caching operations to avoid redundant processing and downloads.
    Provides mechanisms to save and load various data types with versioning.
    """
    
    def __init__(self, cache_dir: str = None):
        """
        Initialize the cache manager.
        
        Args:
            cache_dir: Base directory for cached data. If None, will use an appropriate
                     default directory based on the environment (Colab or local).
        """
        # Set up appropriate cache directory for current environment
        self.cache_dir = self._determine_cache_dir(cache_dir)
        self._ensure_cache_dir()
        self.cache_registry = self._load_registry()
        
    def _determine_cache_dir(self, cache_dir: str = None) -> str:
        """
        Determine the appropriate cache directory based on environment.
        
        Args:
            cache_dir: User-specified cache directory or None
            
        Returns:
            str: Appropriate cache directory path
        """
        import os
        import platform
        
        # If user specified a directory, use that
        if cache_dir is not None:
            return cache_dir
            
        # Check if running in Google Colab
        try:
            import google.colab
            is_colab = True
        except ImportError:
            is_colab = False
            
        if is_colab:
            # In Colab, try to mount Drive first for persistence if not already mounted
            drive_mount_path = '/content/drive'
            drive_cache_path = '/content/drive/MyDrive/AlgoNBA/cache'
            local_cache_path = '/content/algoNBA_cache'
            
            # Attempt to mount Drive if not already mounted
            if not os.path.exists(drive_mount_path):
                print("Google Drive not mounted. Attempting to mount...")
                try:
                    from google.colab import drive
                    drive.mount(drive_mount_path)
                    print("Google Drive mounted successfully")
                except Exception as e:
                    print(f"Error mounting Google Drive: {e}")
                    print(f"Using local cache at {local_cache_path} instead")
                    return local_cache_path
            
            # If Drive is now mounted (or was already), try to use it
            if os.path.exists(drive_mount_path):
                try:
                    # Try creating our cache directory
                    os.makedirs(drive_cache_path, exist_ok=True)
                    print(f"Using Google Drive cache at {drive_cache_path}")
                    return drive_cache_path
                except Exception as e:
                    print(f"Could not create cache in Google Drive: {e}")
                    print(f"Using local cache at {local_cache_path} instead")
                    return local_cache_path
            
            # Fallback to local Colab cache
            print(f"Using local cache at {local_cache_path}")
            return local_cache_path
            
        else:
            # Running locally, check the platform
            system = platform.system()
            machine = platform.machine()
            
            # Base path depends on the platform
            if system == 'Darwin':  # macOS
                # Check if running on Apple Silicon (like M1)
                if machine == 'arm64':
                    # Use a project-relative path for portability
                    return os.path.abspath('data/cache')
                else:
                    # Intel Mac - use same path
                    return os.path.abspath('data/cache')
                    
            elif system == 'Linux':
                # Use a project-relative path but ensure it's absolute
                return os.path.abspath('data/cache')
                
            elif system == 'Windows':
                # Use a project-relative path but ensure it's absolute
                return os.path.abspath('data/cache')
                
            else:
                # Fallback for unknown systems
                return os.path.abspath('data/cache')
        
    def _ensure_cache_dir(self) -> None:
        """Create cache directory structure if it doesn't exist."""
        # Create main cache dir
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Create subdirectories for different data types
        subdirs = ['games', 'features', 'models', 'training', 'predictions', 
                  'player_availability', 'player_impact']
        for subdir in subdirs:
            os.makedirs(os.path.join(self.cache_dir, subdir), exist_ok=True)
    
    def _load_registry(self) -> Dict:
        """Load cache registry or create a new one if it doesn't exist."""
        registry_path = os.path.join(self.cache_dir, 'registry.json')
        if os.path.exists(registry_path):
            try:
                with open(registry_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                # If registry is corrupted, create a new one
                return self._initialize_registry()
        else:
            return self._initialize_registry()
    
    def _initialize_registry(self) -> Dict:
        """Create a new registry structure."""
        registry = {
            'version': '1.0',
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'entries': {}
        }
        self._save_registry(registry)
        return registry
    
    def _save_registry(self, registry: Dict) -> None:
        """Save registry to disk."""
        registry['last_updated'] = datetime.now().isoformat()
        registry_path = os.path.join(self.cache_dir, 'registry.json')
        with open(registry_path, 'w') as f:
            json.dump(registry, f, indent=2)
    
    def _generate_cache_key(self, data_type: str, params: Dict) -> str:
        """
        Generate a unique cache key based on data type and parameters.
        
        Args:
            data_type: Type of data (games, features, etc.)
            params: Parameters that uniquely identify the data
            
        Returns:
            A unique hash key for the data
        """
        # Convert parameters to a stable string representation
        param_str = json.dumps(params, sort_keys=True)
        # Generate hash
        key = hashlib.md5(f"{data_type}_{param_str}".encode()).hexdigest()
        return key
    
    def get_cache_path(self, data_type: str, cache_key: str) -> str:
        """
        Get the file path for a cache entry.
        
        Args:
            data_type: Type of data (games, features, etc.)
            cache_key: Unique cache key
            
        Returns:
            Path to the cached file
        """
        return os.path.join(self.cache_dir, data_type, f"{cache_key}.pkl")
    
    def has_cache(self, data_type: str, params: Dict) -> bool:
        """
        Check if data is available in cache.
        
        Args:
            data_type: Type of data (games, features, etc.)
            params: Parameters that uniquely identify the data
            
        Returns:
            True if data is cached, False otherwise
        """
        cache_key = self._generate_cache_key(data_type, params)
        cache_path = self.get_cache_path(data_type, cache_key)
        
        # Check registry for metadata about this entry
        entry_key = f"{data_type}_{cache_key}"
        registry_has_entry = (entry_key in self.cache_registry.get('entries', {}))
        
        # Check if file exists
        file_exists = os.path.exists(cache_path)
        
        return registry_has_entry and file_exists
    
    def get_cache(self, data_type: str, params: Dict) -> Optional[Any]:
        """
        Retrieve data from cache.
        
        Args:
            data_type: Type of data (games, features, etc.)
            params: Parameters that uniquely identify the data
            
        Returns:
            Cached data if available, None otherwise
        """
        if not self.has_cache(data_type, params):
            return None
        
        cache_key = self._generate_cache_key(data_type, params)
        cache_path = self.get_cache_path(data_type, cache_key)
        
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
                
            # Update access timestamp in registry
            entry_key = f"{data_type}_{cache_key}"
            if entry_key in self.cache_registry.get('entries', {}):
                self.cache_registry['entries'][entry_key]['last_accessed'] = datetime.now().isoformat()
                self._save_registry(self.cache_registry)
                
            return data
        except (pickle.PickleError, FileNotFoundError):
            # If file is corrupted, remove it from registry
            entry_key = f"{data_type}_{cache_key}"
            if entry_key in self.cache_registry.get('entries', {}):
                del self.cache_registry['entries'][entry_key]
                self._save_registry(self.cache_registry)
            
            # Try to delete the file
            try:
                os.remove(cache_path)
            except FileNotFoundError:
                pass
                
            return None
    
    def set_cache(self, data_type: str, params: Dict, data: Any, metadata: Optional[Dict] = None) -> None:
        """
        Store data in cache.
        
        Args:
            data_type: Type of data (games, features, etc.)
            params: Parameters that uniquely identify the data
            data: Data to cache
            metadata: Optional metadata about the data
        """
        cache_key = self._generate_cache_key(data_type, params)
        cache_path = self.get_cache_path(data_type, cache_key)
        
        # Ensure directory exists before writing
        cache_dir = os.path.dirname(cache_path)
        os.makedirs(cache_dir, exist_ok=True)
        
        # Save data to file
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
        
        # Update registry
        entry_key = f"{data_type}_{cache_key}"
        timestamp = datetime.now().isoformat()
        
        if 'entries' not in self.cache_registry:
            self.cache_registry['entries'] = {}
            
        self.cache_registry['entries'][entry_key] = {
            'data_type': data_type,
            'params': params,
            'created_at': timestamp,
            'last_accessed': timestamp,
            'file_path': cache_path
        }
        
        if metadata:
            self.cache_registry['entries'][entry_key]['metadata'] = metadata
            
        self._save_registry(self.cache_registry)
    
    def invalidate_cache(self, data_type: str, params: Dict) -> bool:
        """
        Invalidate a specific cache entry.
        
        Args:
            data_type: Type of data (games, features, etc.)
            params: Parameters that uniquely identify the data
            
        Returns:
            True if cache was invalidated, False otherwise
        """
        if not self.has_cache(data_type, params):
            return False
        
        cache_key = self._generate_cache_key(data_type, params)
        cache_path = self.get_cache_path(data_type, cache_key)
        entry_key = f"{data_type}_{cache_key}"
        
        # Remove file
        try:
            os.remove(cache_path)
        except FileNotFoundError:
            pass
        
        # Update registry
        if entry_key in self.cache_registry.get('entries', {}):
            del self.cache_registry['entries'][entry_key]
            self._save_registry(self.cache_registry)
            
        return True
    
    def clear_cache_type(self, data_type: str) -> int:
        """
        Clear all cache entries of a specific type.
        
        Args:
            data_type: Type of data to clear (games, features, etc.)
            
        Returns:
            Number of entries cleared
        """
        if 'entries' not in self.cache_registry:
            return 0
            
        # Find all entries of this type
        entries_to_remove = []
        for entry_key, entry_data in self.cache_registry['entries'].items():
            if entry_data['data_type'] == data_type:
                entries_to_remove.append(entry_key)
                # Remove file
                try:
                    os.remove(entry_data['file_path'])
                except FileNotFoundError:
                    pass
        
        # Update registry
        for entry_key in entries_to_remove:
            del self.cache_registry['entries'][entry_key]
            
        self._save_registry(self.cache_registry)
        return len(entries_to_remove)
    
    def clear_all_cache(self) -> int:
        """
        Clear all cache entries.
        
        Returns:
            Number of entries cleared
        """
        if 'entries' not in self.cache_registry:
            return 0
            
        # Remove all cache files
        entry_count = len(self.cache_registry['entries'])
        for _, entry_data in self.cache_registry['entries'].items():
            try:
                os.remove(entry_data['file_path'])
            except FileNotFoundError:
                pass
        
        # Reset registry
        self.cache_registry = self._initialize_registry()
        return entry_count
    
    def get_cache_stats(self) -> Dict:
        """
        Get statistics about the cache.
        
        Returns:
            Dictionary with cache statistics
        """
        if 'entries' not in self.cache_registry:
            return {"total_entries": 0, "by_type": {}}
            
        entries = self.cache_registry['entries']
        
        # Count entries by type
        by_type = {}
        for _, entry_data in entries.items():
            data_type = entry_data['data_type']
            if data_type not in by_type:
                by_type[data_type] = 0
            by_type[data_type] += 1
        
        # Calculate total size
        total_size = 0
        for _, entry_data in entries.items():
            try:
                file_path = entry_data['file_path']
                if os.path.exists(file_path):
                    total_size += os.path.getsize(file_path)
            except (FileNotFoundError, KeyError):
                pass
        
        return {
            "total_entries": len(entries),
            "by_type": by_type,
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024)
        }
    
    def get_cache_keys(self, data_type: Optional[str] = None) -> List[str]:
        """
        Get all cache keys, optionally filtered by data type.
        
        Args:
            data_type: Optional data type to filter by
            
        Returns:
            List of cache keys
        """
        if 'entries' not in self.cache_registry:
            return []
            
        keys = []
        for entry_key, entry_data in self.cache_registry['entries'].items():
            if data_type is None or entry_data['data_type'] == data_type:
                # Extract just the hash part from the entry_key
                key = entry_key.split('_', 1)[1] if '_' in entry_key else entry_key
                keys.append(key)
        
        return keys

    def is_cache_stale(self, data_type: str, params: Dict, max_age_days: int) -> bool:
        """
        Check if a cache entry is older than a specific age.
        
        Args:
            data_type: Type of data (games, features, etc.)
            params: Parameters that uniquely identify the data
            max_age_days: Maximum age in days
            
        Returns:
            True if cache is stale, False otherwise
        """
        if not self.has_cache(data_type, params):
            return True
            
        cache_key = self._generate_cache_key(data_type, params)
        entry_key = f"{data_type}_{cache_key}"
        
        if entry_key not in self.cache_registry.get('entries', {}):
            return True
            
        entry = self.cache_registry['entries'][entry_key]
        created_at = datetime.fromisoformat(entry['created_at'])
        
        # Calculate age in days
        age_days = (datetime.now() - created_at).days
        
        return age_days > max_age_days