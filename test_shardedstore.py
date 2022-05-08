import tempfile
import os

import numpy as np
import xarray as xr
from datatree import DataTree
import datatree

from shardedstore import ShardedStore

from zarr.storage import DirectoryStore, ZipStore

def test_shardedstore():
    with tempfile.TemporaryDirectory(prefix='test_shardedstore') as folder:
        base_store = DirectoryStore(os.path.join(folder, "base.zarr"), dimension_separator='/')
        shard1 = DirectoryStore(os.path.join(folder, "shard1.zarr"), dimension_separator='/')
        shard2 = DirectoryStore(os.path.join(folder, "shard2.zarr"), dimension_separator='/')

        try:
            sharded_store = ShardedStore(base_store, {'simulation': shard1, 'simulation/fine': shard2})
            assert False
        except RuntimeError as e:
            # shards must not be overlapping
            pass

        sharded_store = ShardedStore(base_store, {'people': shard1, 'simulation/fine': shard2})

        assert sharded_store.is_readable()
        assert sharded_store.is_writeable()
        assert sharded_store.is_listable()
        assert sharded_store.is_erasable()

        assert sharded_store._shard_for_key('test')[0] == base_store
        assert sharded_store._shard_for_key('test')[1] == 'test'

        assert sharded_store._shard_for_key('t/e')[0] == base_store
        assert sharded_store._shard_for_key('t/e')[1] == 't/e'

        assert sharded_store._shard_for_key('people/bob')[0] == shard1
        assert sharded_store._shard_for_key('people/bob')[1] == 'bob'

        assert sharded_store._shard_for_key('simulation')[0] == base_store
        assert sharded_store._shard_for_key('simulation')[1] == 'simulation'

        assert sharded_store._shard_for_key('simulation/fine')[0] == base_store
        assert sharded_store._shard_for_key('simulation/fine')[1] == 'simulation/fine'

        assert sharded_store._shard_for_key('simulation/fine/.zarray')[0] == shard2
        assert sharded_store._shard_for_key('simulation/fine/.zarray')[1] == '.zarray'


        base_content = 'base_content'.encode()
        sharded_store['base'] = base_content
        assert sharded_store['base'] == base_content

        shard1_content = 'shard1_content'.encode()
        sharded_store['people/shard1'] = shard1_content
        assert sharded_store['people/shard1'] == shard1_content

        shard2_content = 'shard2_content'.encode()
        sharded_store['simulation/fine/shard2'] = shard2_content
        assert sharded_store['simulation/fine/shard2'] == shard2_content

        assert len(sharded_store) == 3
        expected = ['base', 'people/shard1', 'simulation/fine/shard2']
        for i, k in enumerate(sharded_store):
            assert expected[i] == k

        del sharded_store['base']
        del sharded_store['people/shard1']
        assert len(sharded_store) == 1
        expected = ['simulation/fine/shard2']
        for i, k in enumerate(sharded_store):
            assert expected[i] == k

        sharded_store.close()

def test_datatree_shardedstore():
    with tempfile.TemporaryDirectory(prefix='test_datatree_shardedstore') as folder:
        base_store = DirectoryStore(os.path.join(folder, "base.zarr"), dimension_separator='/')
        shard1 = DirectoryStore(os.path.join(folder, "shard1.zarr"), dimension_separator='/')
        shard2 = DirectoryStore(os.path.join("shard2.zarr"), dimension_separator='/')

        # xarray-datatree Quick Overview
        data = xr.DataArray(np.random.randn(3, 3, 5), dims=("x", "y", "z"), coords={"x": [4, 10, 20]})
        data = data.chunk(2)
        ds = xr.Dataset(dict(foo=data, bar=("x", [0, 1, 2]), baz=np.pi))
        ds2 = ds.interp(coords={"x": [10, 12, 14, 16, 18, 20]})
        ds3 = xr.Dataset(
            dict(people=["alice", "bob"], heights=("people", [1.57, 1.82])),
            coords={"species": "human"},
            )
        dt = DataTree.from_dict({"simulation/coarse": ds, "simulation/fine": ds2, "/": ds3})

        single_store = DirectoryStore(os.path.join(folder, "single.zarr"), dimension_separator='/')
        dt.to_zarr(single_store)
        
        sharded_store = ShardedStore(base_store, {'people': shard1, 'simulation/fine': shard2})
        dt.to_zarr(sharded_store)

        from_single = datatree.open_datatree(single_store, engine='zarr')
        from_sharded = datatree.open_datatree(sharded_store, engine='zarr')

        assert from_single.identical(from_sharded)