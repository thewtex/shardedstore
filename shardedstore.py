"""Provides a sharded Zarr store."""

__version__ = "0.1.0"

from typing import Dict, Optional, Tuple
from pathlib import PurePosixPath
import itertools

import zarr.storage
from zarr.util import normalize_storage_path

class ShardedStore(zarr.storage.Store):
    """Store composed of a base store and additional component stores."""

    def __init__(self, base: zarr.storage.BaseStore, shards: Optional[Dict[str, zarr.storage.BaseStore]] = None):
        """Created the shared store. Paths for the shard stores are "mounted" on the base store."""
        self.base = base
        self.shards = {}
        self._mount_points = []
        if shards:
            for p, s in shards.items():
                norm = normalize_storage_path(p)
                self.shards[norm] = s
                self._mount_points.append(PurePosixPath(p))

        for mpa, mpb in itertools.permutations(self._mount_points, 2):
                if  mpa in mpb.parents:
                    raise RuntimeError(f'{mpb} is a subgroup of {mpa} -- not supported')

        mount_point_lengths = [len(str(mp)) for mp in self._mount_points]
        self._min_mount_point_length = min(mount_point_lengths)

    def _shard_for_key(self, key: str) -> Tuple[zarr.storage.BaseStore, str]:
        norm_key = normalize_storage_path(key)
        if len(norm_key) <= self._min_mount_point_length:
            return self.base, norm_key
        path = PurePosixPath(norm_key)
        parents = path.parents
        for mp in self._mount_points:
            if mp in parents:
                return self.shards[str(mp)], str(path.relative_to(mp))
        return self.base, norm_key
        
 
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
        shard, new_key = self._shard_for_key(key)
        del shard[new_key]

    def __getitem__(self, key):
        shard, new_key = self._shard_for_key(key)
        return shard[new_key]

    def __setitem__(self, key, value):
        shard, new_key = self._shard_for_key(key)
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