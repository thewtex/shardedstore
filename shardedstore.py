"""Provides a sharded Zarr store."""

__version__ = "0.1.0"

from typing import Dict, Optional
from pathlib import Path

import zarr.storage

class ShardedStore(zarr.storage.Store):
    """Store composed of a base store and additional component stores."""

    def __init__(self, base: zarr.storage.BaseStore, shards: Optional[Dict[str, zarr.storage.BaseStore]] = None):
        """Created the shared store. Paths for the shard stores are "mounted" on the base store."""
        self.base = base
        if shards is None:
            self.shards = {}
        self.shards = shards

        self._mount_points = [Path(s) for s in shards.keys()]
        for ia, mpa in enumerate(self._mount_points):
            for ib, mpb in enumerate(self._mount_points):
                if ia != ib and mpa in mpb.parents:
                        raise RuntimeError(f'{mpb} is a subgroup of {mpa} -- not supported')
 
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
        # todo
        pass

    def __getitem__(self, key):
        # todo
        pass

    def __setitem__(self, key, value):
        # todo
        pass

    def __iter__(self):
        # todo
        pass

    def __len__(self):
        # todo
        pass