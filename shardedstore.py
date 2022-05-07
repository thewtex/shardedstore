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
        def check_overlapping(mount_points, index):
            mp = mount_points[index]
            for ii, mount_point in enumerate(mount_points):
                if ii != index:
                    if mp in mount_point.parents:
                        raise RuntimeError(f'{mp} is a subgroup of {mount_point} -- not supported')
            if index+1 < len(mount_points):
                check_overlapping(mount_points, index+1)
        if len(self._mount_points):
            check_overlapping(self._mount_points, 0)

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