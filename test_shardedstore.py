import tempfile
import os

import pytest

import numpy as np
import xarray as xr
from datatree import DataTree
import datatree
import json

from shardedstore import (
    ShardedStore,
    array_shard_directory_store,
    to_zip_store_with_prefix,
)

from zarr.storage import DirectoryStore


def test_shardedstore():
    with tempfile.TemporaryDirectory(prefix="test_shardedstore") as folder:
        base_store = DirectoryStore(
            os.path.join(folder, "base.zarr"), dimension_separator="/"
        )
        shard1 = DirectoryStore(
            os.path.join(folder, "shard1.zarr"), dimension_separator="/"
        )
        shard2 = DirectoryStore(
            os.path.join(folder, "shard2.zarr"), dimension_separator="/"
        )
        array_shards1 = array_shard_directory_store(
            os.path.join(folder, "array_shard1")
        )

        try:
            sharded_store = ShardedStore(
                base_store, {"simulation": shard1, "simulation/fine": shard2}
            )
            assert False
        except RuntimeError as e:
            # shards must not be overlapping
            pass
        try:
            sharded_store = ShardedStore(
                base_store,
                {"simulation": shard1},
                {"simulation/fine": (2, array_shards1)},
            )
            assert False
        except RuntimeError as e:
            # shards must not be overlapping
            pass
        try:
            sharded_store = ShardedStore(
                base_store,
                {"simulation": shard1, "simulation/fine": shard2},
                dimension_separator=".",
            )
            assert False
        except ValueError as e:
            # shards must use the same dimension_separator
            pass

        sharded_store = ShardedStore(
            base_store, {"people": shard1, "simulation/fine": shard2}
        )

        assert sharded_store.is_readable()
        assert sharded_store.is_writeable()
        assert sharded_store.is_listable()
        assert sharded_store.is_erasable()

        assert sharded_store._shard_for_key("test")[0] == base_store
        assert sharded_store._shard_for_key("test")[1] == "test"

        assert sharded_store._shard_for_key("t/e")[0] == base_store
        assert sharded_store._shard_for_key("t/e")[1] == "t/e"

        assert sharded_store._shard_for_key("people/bob")[0] == shard1
        assert sharded_store._shard_for_key("people/bob")[1] == "bob"

        assert sharded_store._shard_for_key("simulation")[0] == base_store
        assert sharded_store._shard_for_key("simulation")[1] == "simulation"

        assert sharded_store._shard_for_key("simulation/fine")[0] == base_store
        assert sharded_store._shard_for_key("simulation/fine")[1] == "simulation/fine"

        assert sharded_store._shard_for_key("simulation/fine/line")[0] == shard2
        assert sharded_store._shard_for_key("simulation/fine/line")[1] == "line"

        base_content = "base_content".encode()
        sharded_store["base"] = base_content
        assert sharded_store["base"] == base_content

        shard1_content = "shard1_content".encode()
        sharded_store["people/shard1"] = shard1_content
        assert sharded_store["people/shard1"] == shard1_content

        shard2_content = "shard2_content".encode()
        sharded_store["simulation/fine/shard2"] = shard2_content
        assert sharded_store["simulation/fine/shard2"] == shard2_content

        assert len(sharded_store) == 3
        expected = ["base", "people/shard1", "simulation/fine/shard2"]
        for i, k in enumerate(sharded_store):
            assert expected[i] == k

        del sharded_store["base"]
        del sharded_store["people/shard1"]
        assert len(sharded_store) == 1
        expected = ["simulation/fine/shard2"]
        for i, k in enumerate(sharded_store):
            assert expected[i] == k

        config = sharded_store.get_config()
        config_str = json.dumps(config)
        config = json.loads(config_str)
        sharded_store = ShardedStore.from_config(config)

        sharded_store.close()


@pytest.mark.parametrize("dimension_separator", ["/", ".", None])
def test_datatree_shardedstore(dimension_separator):
    with tempfile.TemporaryDirectory(prefix="test_datatree_shardedstore") as folder:
        base_store = DirectoryStore(
            os.path.join(folder, "base.zarr"), dimension_separator=dimension_separator
        )
        shard1 = DirectoryStore(
            os.path.join(folder, "shard1.zarr"), dimension_separator=dimension_separator
        )
        shard2 = DirectoryStore(
            os.path.join(folder, "shard2.zarr"), dimension_separator=dimension_separator
        )
        array_shards1 = array_shard_directory_store(
            os.path.join(folder, "array_shards1"),
            dimension_separator=dimension_separator,
        )
        array_shards2 = array_shard_directory_store(
            os.path.join(folder, "array_shards2"),
            dimension_separator=dimension_separator,
        )

        # xarray-datatree Quick Overview
        data = xr.DataArray(
            np.random.randn(3, 3, 5), dims=("x", "y", "z"), coords={"x": [4, 10, 20]}
        )
        data = data.chunk([1, 2, 2])
        ds = xr.Dataset(dict(foo=data, bar=("x", [0, 1, 2]), baz=np.pi))
        ds2 = ds.interp(coords={"x": [10, 12, 14, 16, 18, 20]})
        ds2 = ds2.chunk({"x": 1, "y": 1, "z": 2})
        ds3 = xr.Dataset(
            dict(people=["alice", "bob"], heights=("people", [1.57, 1.82])),
            coords={"species": "human"},
        )
        dt = DataTree.from_dict(
            {"simulation/coarse": ds, "simulation/fine": ds2, "/": ds3}
        )

        single_store = DirectoryStore(
            os.path.join(folder, "single.zarr"), dimension_separator=dimension_separator
        )
        dt.to_zarr(single_store)

        sharded_store = ShardedStore(
            base_store,
            {"people": shard1, "species": shard2},
            {
                "simulation/coarse/foo": (1, array_shards1),
                "simulation/fine/foo": (2, array_shards2),
            },
        )
        dt.to_zarr(sharded_store)

        config = sharded_store.get_config()
        config_str = json.dumps(config)
        config = json.loads(config_str)
        sharded_store = ShardedStore.from_config(config)

        from_single = datatree.open_datatree(single_store, engine="zarr").compute()
        from_sharded = datatree.open_datatree(sharded_store, engine="zarr").compute()

        assert from_single.identical(from_sharded)

        to_zip_stores = to_zip_store_with_prefix(os.path.join(folder, "zip_stores"))
        zip_sharded_stores = sharded_store.map_shards(to_zip_stores)

        # raises `ValueError: Attempt to use ZIP archive that was already closed` ?
        # Requires: https://github.com/xarray-contrib/datatree/pull/90
        # from_zip_sharded = datatree.open_datatree(zip_sharded_stores, engine='zarr', mode='r').compute()

        zip_sharded_stores.close()
        sharded_store.close()
