"""Provides a sharded Zarr store."""

__version__ = "0.1.1"

from typing import Dict, Optional, Tuple, Callable
from pathlib import PurePosixPath
import itertools
import functools
import json

import zarr.storage
from zarr.storage import array_meta_key, group_meta_key, attrs_key
_meta_keys = set([array_meta_key, group_meta_key, attrs_key])
from zarr.util import normalize_storage_path

def array_shard_directory_store(prefix: str, **kwargs):
    """Creates a DirectoryStore based on the provided prefix path when passed a string of chunk dimensions.
    
    For use in ShardedStore array_shards."""
    @functools.wraps(zarr.storage.DirectoryStore)
    def wrapper(chunk_dims: str):
        return zarr.storage.DirectoryStore(f'{prefix}/{chunk_dims}.zarr', dimension_separator='/', **kwargs)

    return wrapper

def array_shard_zip_store(prefix: str, **kwargs):
    """Creates a ZipStore based on the provided prefix path when passed a string of chunk dimensions.
    
    For use in ShardedStore array_shards."""
    @functools.wraps(zarr.storage.ZipStore)
    def wrapper(chunk_dims: str):
        return zarr.storage.ZipStore(f'{prefix}/{chunk_dims}.zarr.zip', dimension_separator='/', **kwargs)

    return wrapper

class ShardedStore(zarr.storage.Store):
    """Store composed of a base store and additional component stores."""

    def __init__(self,
         base: zarr.storage.BaseStore,
         shards: Optional[Dict[str, zarr.storage.BaseStore]] = None,
         array_shards: Optional[Dict[str, Tuple[int, Callable]]] = None):
        """Created the sharded store, a store composed of multiple component stores.
        
        Paths for the shard stores are "mounted" on the base store. Shard store paths must
        not overlap directly, but a ShardedStore can be composed of other ShardedStores.
        
        Parameters
        ----------
       
        base: zarr.storage.BaseStore
            Base store. Paths that do not map into the other components shared are stored here.
            
        shards: dict[path, zarr.storage.BaseStore], optional
            Mapping of paths to shard stores.
            
        array_shards: dict[path, func], optional
            Shard for arrays when writing to a store. Mapping of array paths to the number of sharded 
            dimensions and the function that will produce stores for remaining chunk dimensions. The sharded
            chunk dimensions will be passed to this function as a `/`-separated string. For example "0/0", "0/1", etc.
            Mostly commonly, this function is created with array_shard_directory_store.
        """
        self.base = base
        self.shards = {}
        self.array_shards = {}

        self._mount_paths = []
        mount_paths = []
        if shards:
            for p, s in shards.items():
                norm = normalize_storage_path(p)
                self.shards[norm] = s
                self._mount_paths.append(norm)
                mount_paths.append(PurePosixPath(norm))
        self._array_mount_paths = []
        array_mount_paths = []
        if array_shards:
            for p, s in array_shards.items():
                norm = normalize_storage_path(p)
                self.array_shards[norm] = s
                self._array_mount_paths.append(norm)
                array_mount_paths.append(PurePosixPath(norm))

        for mpa, mpb in itertools.permutations(mount_paths + array_mount_paths, 2):
                if  mpa in mpb.parents:
                    raise RuntimeError(f'{mpb} is a subgroup of {mpa} -- not supported')

        mount_path_lengths = [len(str(mp)) for mp in self._mount_paths + self._array_mount_paths]
        self._mount_path_lengths = set(mount_path_lengths)
        self._mount_paths_per_length = {}
        for mount_path in self._mount_paths + self._array_mount_paths:
            length = len(mount_path)
            mount_paths = self._mount_paths_per_length.get(length, set())
            mount_paths.add(mount_path)
            self._mount_paths_per_length[length] = mount_paths
        self._min_mount_path_length = min(mount_path_lengths)

    def _shard_for_key(self, key: str) -> Tuple[zarr.storage.BaseStore, str, bool]:
        norm_key = normalize_storage_path(key)
        if len(norm_key) <= self._min_mount_path_length:
            return self.base, norm_key, False
        norm_key_length = len(norm_key)
        for mount_path_length in self._mount_path_lengths:
            if norm_key_length <= mount_path_length + 1:
                continue
            norm_key_prefix = norm_key[:mount_path_length]
            mount_paths = self._mount_paths_per_length[mount_path_length]
            if norm_key_prefix in mount_paths:
                if norm_key_prefix in self.array_shards:
                    return self.array_shards[norm_key_prefix], str(PurePosixPath(norm_key).relative_to(PurePosixPath(norm_key_prefix))), True

                return self.shards[norm_key_prefix], str(PurePosixPath(norm_key).relative_to(PurePosixPath(norm_key_prefix))), False
        return self.base, norm_key, False
        
 
    def is_readable(self):
        return all([self.base.is_readable(),] + list(map(lambda x: x.is_readable(), self.shards.values())))

    def is_writeable(self):
        return all([self.base.is_writeable(),] + list(map(lambda x: x.is_writeable(), self.shards.values())))

    def is_listable(self):
        return all([self.base.is_listable(),] + list(map(lambda x: x.is_listable(), self.shards.values())))

    def is_erasable(self):
        return all([self.base.is_erasable(),] + list(map(lambda x: x.is_erasable(), self.shards.values())))

    def close(self):
        for shard in self.shards.values():
            shard.close()
        self.base.close()

    def __delitem__(self, key):
        shard, new_key, _ = self._shard_for_key(key)
        del shard[new_key]

    def __getitem__(self, key):
        shard, new_key, is_array_shard = self._shard_for_key(key)
        print(shard, new_key, is_array_shard)
        if is_array_shard:
            if new_key in _meta_keys:
                return self.base[key]
            print(shard[1])
            return shard[1][new_key]
        return shard[new_key]

    def __setitem__(self, key, value):
        print(key)
        shard, new_key, is_array_shard = self._shard_for_key(key)
        print(shard, new_key, is_array_shard)
        if is_array_shard:
            print(shard, new_key, array_meta_key)
            if new_key in _meta_keys:
                print(new_key, value)
                print(new_key, json.loads(value))
            pass
        shard[new_key] = value

    @staticmethod
    def _shard_iter(shards):
        for mount, shard in shards.items():
            for k in iter(shard):
                yield mount + '/' + k

    def __iter__(self):
        return itertools.chain(self.base, self._shard_iter(self.shards))

    def __len__(self):
        return sum([len(self.base),] + [len(s) for s in self.shards.values()])
