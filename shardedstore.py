"""Provides a sharded Zarr store."""

__version__ = "0.3.0"

from typing import Any, Dict, Optional, Tuple, Callable
from pathlib import PurePosixPath, Path
import itertools
import functools
import json
import math
import importlib

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
        return zarr.storage.DirectoryStore(f"{prefix}/{chunk_dims}.zarr", **kwargs)

    return wrapper


def to_zip_store(
    prefix: str,
    shard_store: zarr.storage.BaseStore,
    shard_path: str,
    chunk_dims: Optional[str],
):
    """Convert stores to a zip store at the provided prefix."""
    if chunk_dims:
        store_path = f"{prefix}/{shard_path}/{chunk_dims}.zarr.zip"
    elif shard_path:
        store_path = f"{prefix}/{shard_path}.zarr.zip"
    else:
        store_path = f"{prefix}.zarr.zip"
    Path(store_path).parent.mkdir(parents=True, exist_ok=True)

    dimension_separator = getattr(shard_store, "_dimension_separator", None)
    zip_store = zarr.storage.ZipStore(
        store_path, mode="a", dimension_separator=dimension_separator
    )
    for k in shard_store:
        zip_store[k] = shard_store[k]

    zip_store.flush()
    return zip_store


def to_zip_store_with_prefix(prefix: str):
    """Convert stores to a zip store at the provided prefix.

    For use in `ShardedStore.map_shards`."""

    @functools.wraps(to_zip_store)
    def wrapper(
        shard_store: zarr.storage.BaseStore, shard_path: str, chunk_dims: Optional[str]
    ):
        return to_zip_store(prefix, shard_store, shard_path, chunk_dims)

    return wrapper


class ShardedStore(zarr.storage.Store):
    """Store composed of a base store and additional component stores."""

    def _update_internal_state(self):
        self._mount_paths = []
        self._array_mount_paths = []

        mount_paths = []
        for shard_path in self.shards:
            self._mount_paths.append(shard_path)
            mount_paths.append(PurePosixPath(shard_path))
        array_mount_paths = []
        for shard_path in self.array_shards:
            self._array_mount_paths.append(shard_path)
            array_mount_paths.append(PurePosixPath(shard_path))

        for mpa, mpb in itertools.permutations(mount_paths + array_mount_paths, 2):
            if mpa in mpb.parents:
                raise RuntimeError(f"{mpb} is a subgroup of {mpa} -- not supported")

        mount_path_lengths = [
            len(str(mp)) for mp in self._mount_paths + self._array_mount_paths
        ]
        self._mount_path_lengths = set(mount_path_lengths)
        self._mount_paths_per_length = {}
        for mount_path in self._mount_paths + self._array_mount_paths:
            length = len(mount_path)
            mount_paths = self._mount_paths_per_length.get(length, set())
            mount_paths.add(mount_path)
            self._mount_paths_per_length[length] = mount_paths
        if len(mount_path_lengths):
            self._min_mount_path_length = min(mount_path_lengths)

    def __init__(
        self,
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
        if hasattr(base, "_dimension_separator"):
            if base._dimension_separator != dimension_separator:
                raise ValueError(
                    "ShardedStore and base store must use the same dimension_separator"
                )

        if shards:
            for p, s in shards.items():
                norm = normalize_storage_path(p)
                self.shards[norm] = s

        if array_shard_funcs:
            for p, s in array_shard_funcs.items():
                norm = normalize_storage_path(p)
                self.array_shard_dims[norm] = s[0]
                self.array_shard_funcs[norm] = s[1]
                self.array_shards[norm] = {}

        self._update_internal_state()

    @staticmethod
    def _get_store_config(store):
        if hasattr(store, "get_config"):
            return store.get_config()

        store_name = store.__class__.__name__
        store_module = store.__module__
        store_package = store_module.split(".")[0]
        store_package = importlib.import_module(store_package)

        config_args = []
        config_kwargs = {}
        for k in store.__dict__:
            if k == "_dimension_separator":
                config_kwargs["dimension_separator"] = store._dimension_separator
            # Avoid clobbering on re-open
            elif k == "mode" and k in "wx":
                pass
            elif k in ("path", "base"):
                config_args.append(getattr(store, k))
            elif not k.startswith("_"):
                val = getattr(store, k)
                try:
                    json_serializable = json.dumps(val)
                    config_kwargs[k] = val
                except (TypeError, OverflowError):
                    pass
        config = {
            "args": config_args,
            "kwargs": config_kwargs,
        }
        storeconfig = {
            "name": store_name,
            "module": store_module,
            "config": config,
        }
        if hasattr(store_package, "__version__"):
            storeconfig["version"] = store_package.__version__

        return storeconfig

    def get_config(self):
        """Returns a dictionary with configuration parameters for the store and store shards.
        Can be used with `from_config` to create a read-only instance of the store.
        All values compatible with JSON encoding."""

        shard_config = {}
        for shard_path in self.shards:
            shard_config[shard_path] = self._get_store_config(self.shards[shard_path])

        array_shard_dims_config = {}
        for shard_path in self.array_shard_dims:
            array_shard_dims_config[shard_path] = self.array_shard_dims[shard_path]

        array_shards_config = {}
        for array_shards_path in self.array_shards:
            array_shard_config = {}
            for array_shard_path in self.array_shards[array_shards_path]:
                array_shard_config[array_shard_path] = self._get_store_config(
                    self.array_shards[array_shards_path][array_shard_path]
                )
            array_shards_config[array_shards_path] = array_shard_config

        config = {
            "args": [
                self._get_store_config(self.base),
            ],
            "kwargs": {
                "dimension_separator": self._dimension_separator,
                "shards": shard_config,
                "array_shard_dims": array_shard_dims_config,
                "array_shards": array_shards_config,
            },
        }
        storeconfig = {
            "version": __version__,
            "name": self.__class__.__name__,
            "module": self.__module__,
            "config": config,
        }

        return storeconfig

    @staticmethod
    def _from_store_config(config):
        mod = importlib.import_module(config["module"])
        store_cls = getattr(mod, config["name"])
        if hasattr(store_cls, "from_config"):
            return store_cls.from_config(config)
        return store_cls(*config["config"]["args"], **config["config"]["kwargs"])

    @classmethod
    def from_config(cls, config):
        """Instantiate a sharded store from its `get_config` configuration."""

        if config["name"] != cls.__name__:
            raise ValueError(f"Config provided is not for {cls.__name__}")

        config = config["config"]
        base = cls._from_store_config(config["args"][0])
        shards = {}
        if "shards" in config["kwargs"]:
            for shard_path in config["kwargs"]["shards"]:
                shards[shard_path] = cls._from_store_config(
                    config["kwargs"]["shards"][shard_path]
                )

        sharded_store = cls(base, shards=shards)

        array_shards = {}
        array_shard_dims = {}
        if "array_shards" in config["kwargs"]:
            array_shards_config = config["kwargs"]["array_shards"]
            for array_shards_path in array_shards_config:
                array_shards[array_shards_path] = {}
                array_shard_config = array_shards_config[array_shards_path]
                for array_shard_path in array_shard_config:
                    array_shard_dims[array_shards_path] = (
                        len(array_shard_path) + 1
                    ) // 2
                    array_shard = cls._from_store_config(
                        array_shard_config[array_shard_path]
                    )
                    array_shards[array_shards_path][array_shard_path] = array_shard
        sharded_store.array_shards = array_shards
        sharded_store.array_shard_dims = array_shard_dims
        sharded_store._update_internal_state()

        return sharded_store

    def map_shards(
        self, func: Callable[[zarr.storage.BaseStore, str, str], zarr.storage.BaseStore]
    ):
        """Run the provided function on each shard in the store.

        The function should take the store, shard path, and optional array chunk path as inputs and return a store as an output.

        Returns a new ShardedStore with the resulting output stores."""
        base = func(self.base, "", None)
        shards = {}
        for path in self.shards:
            shards[path] = func(self.shards[path], path, None)
        array_shard_funcs = {}
        for p, sf in self.array_shard_funcs:
            array_shard_funcs[p] = (self.array_shard_dims[p], sf)
        sharded_store = self.__class__(
            base,
            shards=shards,
            array_shard_funcs=array_shard_funcs,
            dimension_separator=self._dimension_separator,
        )

        array_shards = {}
        array_shard_dims = {}
        for array_shards_path in self.array_shards:
            array_shards[array_shards_path] = {}
            for array_shard_path in self.array_shards[array_shards_path]:
                array_shard_dims[array_shards_path] = (len(array_shard_path) + 1) // 2
                array_shard = func(
                    self.array_shards[array_shards_path][array_shard_path],
                    array_shards_path,
                    array_shard_path,
                )
                array_shards[array_shards_path][array_shard_path] = array_shard
        sharded_store.array_shards = array_shards
        sharded_store.array_shard_dims = array_shard_dims

        return sharded_store

    def _shard_for_key(
        self, key: str, value: bytes = None
    ) -> Tuple[zarr.storage.BaseStore, str]:
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
                        raise ValueError(
                            "Array shard requested for array path with insufficient chunked dims"
                        )
                    postfix = str(
                        PurePosixPath(norm_key).relative_to(
                            PurePosixPath(norm_key_prefix)
                        )
                    )
                    if postfix in _meta_keys:
                        if postfix == array_meta_key and value is not None:
                            array_meta_str = value.decode()
                            array_meta = json.loads(array_meta_str)

                            chunks = array_meta["chunks"]
                            if any([c != 1 for c in chunks[:shard_dims]]):
                                raise ValueError(
                                    f"Shared chunk dimensions must be 1, received: {chunks[:shard_dims]}"
                                )
                            array_shard_func = self.array_shard_funcs[norm_key_prefix]
                            array_shards = self.array_shards[norm_key_prefix]

                            array_meta["chunks"] = chunks[shard_dims:]

                            chunk_shard_shape = [
                                c for c in array_meta["shape"][:shard_dims]
                            ]
                            prod = math.prod(array_meta["shape"][: shard_dims + 1])
                            array_meta["shape"] = array_meta["shape"][shard_dims:]
                            array_meta["shape"][0] = prod

                            array_meta.pop("zarr_format", None)
                            array_meta["compressor"] = codecs.registry.get_codec(
                                array_meta["compressor"]
                            )

                            prefix_separator = self._dimension_separator
                            if not prefix_separator:
                                prefix_separator = "/"
                            for chunk_shard in itertools.product(
                                *(range(s) for s in chunk_shard_shape)
                            ):
                                chunk_prefix = prefix_separator.join(
                                    [str(c) for c in chunk_shard]
                                )
                                array_shard = array_shard_func(chunk_prefix)
                                if hasattr(array_shard, "_dimension_separator"):
                                    if (
                                        array_shard._dimension_separator
                                        != self._dimension_separator
                                    ):
                                        raise ValueError(
                                            "Array shard store must use the same dimension_separator as the ShardedStore"
                                        )
                                zarr.storage.init_array(
                                    array_shard, overwrite=True, **array_meta
                                )
                                array_shards[chunk_prefix] = array_shard
                            self.array_shards[norm_key_prefix] = array_shards
                        return self.base, norm_key

                    chunk_prefix = postfix[: shard_path_length - 1]
                    array_shards = self.array_shards[norm_key_prefix]
                    if chunk_prefix in array_shards:
                        remaining_chunks = postfix[shard_path_length:]
                        return array_shards[chunk_prefix], remaining_chunks

                if norm_key_prefix in self.shards:
                    return self.shards[norm_key_prefix], str(
                        PurePosixPath(norm_key).relative_to(
                            PurePosixPath(norm_key_prefix)
                        )
                    )
        return self.base, norm_key

    def _get_shards_status(self, status_method):
        base_status = [
            getattr(self.base, status_method)(),
        ]
        shards_status = list(
            map(lambda x: getattr(x, status_method)(), self.shards.values())
        )
        array_shards_status = []
        for array_shards in self.array_shards.values():
            for array_shard in array_shards:
                status = list(
                    map(lambda x: getattr(x, status_method)(), array_shard.values())
                )
                array_shards_status = array_shards_status + status
        return all(base_status + shards_status + array_shards_status)

    def is_readable(self):
        return self._get_shards_status("is_readable")

    def is_writeable(self):
        return self._get_shards_status("is_writeable")

    def is_listable(self):
        return self._get_shards_status("is_listable")

    def is_erasable(self):
        return self._get_shards_status("is_erasable")

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
                yield mount + "/" + k

    def __iter__(self):
        return itertools.chain(self.base, self._shard_iter(self.shards))

    def __len__(self):
        return sum(
            [
                len(self.base),
            ]
            + [len(s) for s in self.shards.values()]
        )
