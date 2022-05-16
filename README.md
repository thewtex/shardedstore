# shardedstore

[![image](https://img.shields.io/pypi/v/shardedstore.svg)](https://pypi.python.org/pypi/shardedstore/)
[![Test](https://github.com/thewtex/shardedstore/actions/workflows/test.yml/badge.svg)](https://github.com/thewtex/shardedstore/actions/workflows/test.yml)
[![DOI](https://zenodo.org/badge/489549406.svg)](https://zenodo.org/badge/latestdoi/489549406)

Provides a sharded Zarr store.

## Features

- For large Zarr stores, avoid an excessive number of objects or extremely large objects, which avoids filesystem inode usage and object store limitations.
- Performance-sensitive implementation.
- Use existing Zarr v2 stores.
- Mix and match shard store types.
- Serialize and deserialize the ShardedStore in JSON.
- Shard groups or array chunks.
- Easily run transformations on store shards.

## Installation

```sh
pip install shardedstore
```

## Example

```python
from zarr.storage import DirectoryStore
from shardedstore import ShardedStore, array_shard_directory_store, to_zip_store_with_prefix

# xarray example, but works with zarr in general
import xarray as xr
from datatree import DataTree, open_datatree
import json
import numpy as np
import os

base_store = DirectoryStore("base.zarr")
shard1 = DirectoryStore("shard1.zarr")
shard2 = DirectoryStore("shard2.zarr")
array_shards1 = array_shard_directory_store("array_shards1")
array_shards2 = array_shard_directory_store("array_shards2")

# xarray-datatree Quick Overview
data = xr.DataArray(np.random.randn(2, 3), dims=("x", "y"), coords={"x": [10, 20]})
# Sharded array dimensions must have a chunk shape of 1.
data = data.chunk([1,2])
ds = xr.Dataset(dict(foo=data, bar=("x", [1, 2]), baz=np.pi))
ds2 = ds.interp(coords={"x": [10, 12, 14, 16, 18, 20]})
ds2 = ds2.chunk({'x':1, 'y':2})
ds3 = xr.Dataset(
    dict(people=["alice", "bob"], heights=("people", [1.57, 1.82])),
    coords={"species": "human"},
    )
dt = DataTree.from_dict({"simulation/coarse": ds, "simulation/fine": ds2, "/": ds3})

# A monolithic store
single_store = DirectoryStore("single.zarr")
dt.to_zarr(single_store)

# A sharded store demonstrating sharding on groups and arrays. The arrays are sharded over 1 dimension.
sharded_store = ShardedStore(base_store,
    {'people': shard1, 'species': shard2},
    {'simulation/coarse/foo': (1, array_shards1), 'simulation/fine/foo': (1, array_shards2)})
dt.to_zarr(sharded_store)

# Serialize / deserialize
config = sharded_store.get_config()
config_str = json.dumps(config)
config = json.loads(config_str)
sharded_store = ShardedStore.from_config(config)

from_single = datatree.open_datatree(single_store, engine='zarr').compute()
from_sharded = datatree.open_datatree(sharded_store, engine='zarr').compute()

assert from_single.identical(from_sharded)

# Run transformations over component shards with `map_shards`
to_zip_stores = to_zip_store_with_prefix("zip_stores")
zip_sharded_stores = sharded_store.map_shards(to_zip_stores)
```

## Development

Contributions are welcome and appreciated.

```
git clone https://github.com/thewtex/shardedstore
cd shardedstore
pip install -e ".[test]"
pytest
```
