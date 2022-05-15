"""Provides a sharded Zarr store."""

__version__ = "0.2.0"

from typing import Any, Dict, Optional, Tuple, Callable
from pathlib import PurePosixPath, Path
import itertools
import functools
import json
import math

import numcodecs as codecs
from numpy import array

import zarr.storage
from zarr.storage import array_meta_key, group_meta_key, attrs_key
_meta_keys = set([array_meta_key, group_meta_key, attrs_key])
from zarr.util import normalize_storage_path, normalize_dimension_separator

def array_shard_directory_store(prefix: str, **kwargs):
    """Creates a DirectoryStore based on the provided prefix path when passed a string of chunk dimensions.
    
    For use in ShardedStore array_shards."""
    @functools.wraps(zarr.storage.DirectoryStore)
    def wrapper(chunk_dims: str):
        return zarr.storage.DirectoryStore(f'{prefix}/{chunk_dims}.zarr', **kwargs)

    return wrapper

def array_shard_zip_store(prefix: str, **kwargs):
    """Creates a ZipStore based on the provided prefix path when passed a string of chunk dimensions.
    
    For use in ShardedStore array_shards."""
    @functools.wraps(zarr.storage.ZipStore)
    def wrapper(chunk_dims: str):
        Path(f'{prefix}/{chunk_dims}').parent.mkdir(parents=True, exist_ok=True)
        return zarr.storage.ZipStore(f'{prefix}/{chunk_dims}.zarr.zip', **kwargs)

    return wrapper

class ShardedStore(zarr.storage.Store):
    """Store composed of a base store and additional component stores."""

    def __init__(self,
         base: zarr.storage.BaseStore,
         shards: Optional[Dict[str, zarr.storage.BaseStore]] = None,
         array_shard_funcs: Optional[Dict[str, Tuple[int, Callable]]] = None,
         dimension_separator: Optional[str] = None,
         ):
        """Created the sharded store, a store composed of multiple component stores.
        
        Paths for the shard stores are "mounted" on the base store. Shard store paths must
        not overlap directly, but a ShardedStore can be composed of other ShardedStores.
        
        Parameters
        ----------
       
        base: zarr.storage.BaseStore
            Base store. Paths that do not map into the other components shared are stored here.
            
        shards: dict[path, zarr.storage.BaseStore], optional
            Mapping of paths to shard stores.
            
        array_shard_funcs: dict[path, (int, func)], optional
            Shard for arrays when writing to a store. Mapping of array paths to the number of sharded 
            dimensions and the function that will produce stores for remaining chunk dimensions. The sharded
            chunk dimensions will be passed to this function as a `/`-separated string. For example "0/0", "0/1", etc.
            for a dimension of 2.
            Mostly commonly, this `func` is `shardedstore.array_shard_directory_store`.

        dimension_separator : {'.', '/'}, optional
            Separator placed between the dimensions of a chunk. The base store and and array_shard_funcs store must
            use the same dimension_separator.
        """
        self.base = base
        self.shards = {}
        self.array_shards = {}
        self.array_shard_dims = {}
        self.array_shard_funcs = {}

        if dimension_separator is None:
            dimension_separator = getattr(base, "_dimension_separator", None)
        dimension_separator = normalize_dimension_separator(dimension_separator)
        self._dimension_separator = dimension_separator
        if hasattr(base, '_dimension_separator'):
            if base._dimension_separator != dimension_separator:
                raise ValueError('ShardedStore and base store must use the same dimension_separator')

        self._mount_paths = []
        self._array_mount_paths = []

        mount_paths = []
        if shards:
            for p, s in shards.items():
                norm = normalize_storage_path(p)
                self.shards[norm] = s
                self._mount_paths.append(norm)
                mount_paths.append(PurePosixPath(norm))
        array_mount_paths = []
        if array_shard_funcs:
            for p, s in array_shard_funcs.items():
                norm = normalize_storage_path(p)
                self.array_shard_dims[norm] = s[0]
                self.array_shard_funcs[norm] = s[1]
                self.array_shards[norm] = {}
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
        if len(mount_path_lengths):
            self._min_mount_path_length = min(mount_path_lengths)

    def _shard_for_key(self, key: str, value: bytes = None) -> Tuple[zarr.storage.BaseStore, str]:
        norm_key = normalize_storage_path(key)

        if len(norm_key) <= self._min_mount_path_length:
            return self.base, norm_key 

        norm_key_length = len(norm_key)
        for mount_path_length in self._mount_path_lengths:
            if norm_key_length <= mount_path_length + 1:
                continue

            norm_key_prefix = norm_key[:mount_path_length]
            mount_paths = self._mount_paths_per_length[mount_path_length]
            if norm_key_prefix in mount_paths:
                if norm_key_prefix in self.array_shards:
                    shard_dims = self.array_shard_dims[norm_key_prefix]
                    shard_path_length = shard_dims + shard_dims
                    if mount_path_length + shard_path_length >= norm_key_length:
                        raise ValueError('Array shard requested for array path with insufficient chunked dims')
                    postfix = str(PurePosixPath(norm_key).relative_to(PurePosixPath(norm_key_prefix)))
                    if postfix in _meta_keys:
                        if postfix == array_meta_key and value is not None:
                            array_meta_str = value.decode()
                            array_meta = json.loads(array_meta_str)

                            chunks = array_meta['chunks']
                            if (any([c != 1 for c in chunks[:shard_dims]])):
                                raise ValueError(f'Shared chunk dimensions must be 1, received: {chunks[:shard_dims]}')
                            array_shard_func = self.array_shard_funcs[norm_key_prefix]
                            array_shards = self.array_shards[norm_key_prefix]

                            array_meta['chunks'] = chunks[shard_dims:]

                            chunk_shard_shape = [c for c in array_meta['shape'][:shard_dims]]
                            prod = math.prod(array_meta['shape'][:shard_dims+1])
                            array_meta['shape'] = array_meta['shape'][shard_dims:]
                            array_meta['shape'][0] = prod

                            array_meta.pop('zarr_format', None)
                            array_meta['compressor'] = codecs.registry.get_codec(array_meta['compressor'])

                            for chunk_shard in itertools.product(*(range(s) for s in chunk_shard_shape)):
                                chunk_prefix = '/'.join([str(c) for c in chunk_shard])
                                array_shard = array_shard_func(chunk_prefix)
                                if hasattr(array_shard, '_dimension_separator'):
                                    if array_shard._dimension_separator != self._dimension_separator:
                                        raise ValueError('Array shard store must use the same dimension_separator as the ShardedStore')
                                zarr.storage.init_array(array_shard, overwrite=True, **array_meta)
                                array_shards[chunk_prefix] = array_shard
                            self.array_shards[norm_key_prefix] = array_shards
                        return self.base, norm_key 

                    chunk_prefix = postfix[:shard_path_length-1]
                    array_shards = self.array_shards[norm_key_prefix]
                    if chunk_prefix in array_shards:
                        remaining_chunks = postfix[shard_path_length:]
                        return array_shards[chunk_prefix], remaining_chunks 

                if norm_key_prefix in self.shards:
                    return self.shards[norm_key_prefix], str(PurePosixPath(norm_key).relative_to(PurePosixPath(norm_key_prefix))) 
        return self.base, norm_key
    
    def _get_shards_status(self, status_method):
        base_status = [getattr(self.base, status_method)(),]
        shards_status = list(map(lambda x: getattr(x, status_method)(), self.shards.values()))
        array_shards_status = []
        for array_shards in self.array_shards.values():
            for array_shard in self.array_shards.values():
                status = list(map(lambda x: getattr(x, status_method)(), array_shard.values()))
                array_shards_status = array_shards_status + status
        return all(base_status + shards_status + array_shards_status)
        
    def is_readable(self):
        return self._get_shards_status('is_readable')

    def is_writeable(self):
        return self._get_shards_status('is_writeable')

    def is_listable(self):
        return self._get_shards_status('is_listable')

    def is_erasable(self):
        return self._get_shards_status('is_erasable')

    def close(self):
        for shard in self.shards.values():
            shard.close()
        for array_shards in self.array_shards.values():
            for array_shard in array_shards.values():
                array_shard.close()
        self.base.close()

    def __delitem__(self, key):
        shard, new_key = self._shard_for_key(key)
        del shard[new_key]

    def __getitem__(self, key):
        shard, new_key = self._shard_for_key(key)
        return shard[new_key]

    def __setitem__(self, key, value):
        shard, new_key = self._shard_for_key(key, value)
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
