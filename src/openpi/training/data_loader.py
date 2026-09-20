from collections.abc import Iterator, Sequence
import logging
import multiprocessing
import os
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class IndexedActionDataset(Dataset):
    """Explicit observation/target mapping over an existing LeRobot image dataset.

    Sparse actions have no fixed time interval. Their validated archive contains
    the action chunks directly; LeRobot timestamp offsets are not used.
    """

    def __init__(self, dataset: Dataset, path: str, horizon: int, *, require_validated: bool = True):
        self._dataset = dataset
        with np.load(path, allow_pickle=False) as archive:
            self.rows = {key: archive[key].copy() for key in archive.files}
        if require_validated and not bool(self.rows["validated"].item()):
            raise ValueError("Sparse dataset completion/alignment audit has not passed; training is disabled")
        indices = self.rows["observation_indices"]
        n = len(indices)
        if n == 0 or indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("Indexed observations must be a nonempty integer vector")
        if np.any(indices < 0) or np.any(indices >= len(dataset)) or np.any(np.diff(indices) <= 0):
            raise ValueError("Indexed observations must be sorted, unique, and within the source dataset")
        actions, states = self.rows["actions"], self.rows["states"]
        if actions.ndim != 3 or actions.shape[:2] != (n, horizon) or states.ndim != 2 or len(states) != n:
            raise ValueError("Indexed state/action shapes do not match the observation count and horizon")
        if not np.isfinite(actions).all() or not np.isfinite(states).all():
            raise ValueError("Indexed states and actions must be finite")
        columns = self.rows["state_columns"]
        if columns.shape != (states.shape[1],) or not np.issubdtype(columns.dtype, np.integer):
            raise ValueError("State column mapping does not match the indexed states")
        targets, padding = self.rows["source_action_indices"], self.rows["actions_is_pad"]
        bounds = self.rows["episode_bounds"]
        if targets.shape != (n, horizon) or padding.shape != targets.shape or bounds.shape != (n, 2):
            raise ValueError("Indexed target provenance/padding shapes are invalid")
        if not np.issubdtype(targets.dtype, np.integer) or padding.dtype != np.bool_:
            raise ValueError("Indexed target provenance must be integer and padding must be boolean")
        if np.any(indices < bounds[:, 0]) or np.any(indices >= bounds[:, 1]):
            raise ValueError("Indexed observation crosses its source episode boundary")
        if np.any(bounds[:, 0] < 0) or np.any(bounds[:, 1] > len(dataset)) or np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError("Indexed episode bounds must lie within the source dataset")
        provenance_keys = {"source_episode_bounds", "source_observation_indices"}
        if provenance_keys.intersection(self.rows) and not provenance_keys.issubset(self.rows):
            raise ValueError("Compact observations require both raw episode and observation provenance")
        source_bounds = self.rows.get("source_episode_bounds", bounds)
        source_indices = self.rows.get("source_observation_indices", indices)
        if (
            source_bounds.shape != (n, 2)
            or source_indices.shape != (n,)
            or not np.issubdtype(source_bounds.dtype, np.integer)
            or not np.issubdtype(source_indices.dtype, np.integer)
            or np.any(source_bounds[:, 0] < 0)
            or np.any(source_bounds[:, 0] >= source_bounds[:, 1])
            or np.any(source_indices < source_bounds[:, 0])
            or np.any(source_indices >= source_bounds[:, 1])
            or np.any(np.diff(source_indices) <= 0)
        ):
            raise ValueError("Invalid raw provenance for compact observations")
        if np.any(targets < source_bounds[:, :1]) or np.any(targets >= source_bounds[:, 1:]):
            raise ValueError("Indexed target crosses its source episode boundary")
        if np.any(targets[:, 0] < source_indices):
            raise ValueError("Indexed first target precedes its observation")
        if np.any(np.diff(targets, axis=1) < 0) or np.any(np.diff(padding.astype(int), axis=1) < 0):
            raise ValueError("Indexed targets must be ordered with padding only at the end")
        if np.any(padding[:, 0]) or self.rows["episode_indices"].shape != (n,):
            raise ValueError("Every observation needs a real first target and episode identity")
        for row in np.flatnonzero(padding.any(axis=1)):
            last = int(np.flatnonzero(~padding[row])[-1])
            if not np.array_equal(actions[row, last:], np.broadcast_to(actions[row, last], actions[row, last:].shape)):
                raise ValueError("Padded targets must repeat the final real target")

    def __len__(self):
        return len(self.rows["observation_indices"])

    def __getitem__(self, index):
        item = dict(self._dataset[int(self.rows["observation_indices"][index])])
        if int(item["episode_index"]) != int(self.rows["episode_indices"][index]):
            raise ValueError("Sparse archive and source dataset episode identities differ")
        state = np.asarray(item["state"])[self.rows["state_columns"]]
        if not np.array_equal(state, self.rows["states"][index]):
            raise ValueError("Sparse archive and source observation states differ")
        if "source_observation_indices" in self.rows:
            if int(np.asarray(item["source_row"]).item()) != int(self.rows["source_observation_indices"][index]):
                raise ValueError("Compact image row and raw observation provenance differ")
            if not np.array_equal(np.asarray(item["cartesian_position"]), self.rows["cartesian_positions"][index]):
                raise ValueError("Compact Cartesian context differs from its indexed source")
        item["state"] = state.copy()
        item["actions"] = self.rows["actions"][index].copy()
        item["actions_is_pad"] = self.rows["actions_is_pad"][index].copy()
        return item


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class RawFrameChunkDataset(Dataset):
    """Observations are rows of a raw HDF5 recording; labels come from a validated dense index archive.

    Frames are never copied: each item reads one `pixels[row]` frame from the recording. The file handle is
    opened lazily in the process that reads, so forked loader workers never share an HDF5 handle.
    """

    def __init__(self, archive_path: str, *, require_validated: bool = True):
        import h5py  # noqa: F401  (imported here so the dependency is only needed for this branch)

        with np.load(archive_path, allow_pickle=False) as archive:
            self.rows = {key: archive[key] for key in archive.files}
        rows = self.rows
        if require_validated and not bool(rows["validated"]):
            raise ValueError("Dense archive has not been validated")
        self.horizon = int(rows["horizon"])
        self.source_path = str(rows["source_path"])
        self.prompt = str(rows["prompt"])
        observations = rows["source_observation_indices"]
        targets = rows["source_action_indices"]
        padding = rows["actions_is_pad"]
        actions = rows["actions"]
        states = rows["states"]
        bounds = rows["source_episode_bounds"]
        n = len(observations)
        if observations.ndim != 1 or n == 0 or np.any(observations[1:] <= observations[:-1]):
            raise ValueError("Dense observations must be a nonempty strictly increasing row index array")
        if actions.shape != (n, self.horizon, 4) or not np.isfinite(actions).all():
            raise ValueError("Dense actions must be finite (n, horizon, 4) reference poses")
        if states.ndim != 2 or states.shape[0] != n or not np.isfinite(states).all():
            raise ValueError("Dense states must be a finite (n, dim) array")
        if targets.shape != (n, self.horizon) or padding.shape != (n, self.horizon) or bounds.shape != (n, 2):
            raise ValueError("Dense target rows, padding and episode bounds must match the observations")
        if np.any(observations < bounds[:, 0]) or np.any(observations >= bounds[:, 1]):
            raise ValueError("Every observation must lie inside its episode")
        real = ~padding
        if np.any(targets[real] <= np.broadcast_to(observations[:, None], targets.shape)[real]) or np.any(
            targets >= bounds[:, 1:2]
        ):
            raise ValueError("Every real target row must follow its observation inside the same episode")
        if np.any(np.diff(targets, axis=1) < 0):
            raise ValueError("Target rows must be non-decreasing along the chunk")
        lengths = (~padding).sum(axis=1)
        if not np.array_equal(padding, np.arange(self.horizon)[None] >= lengths[:, None]):
            raise ValueError("Padding may occur only as an episode-tail suffix")
        if np.any(targets[padding] != bounds[:, 1:2].repeat(self.horizon, axis=1)[padding] - 1):
            raise ValueError("Padded slots must repeat the episode's last row")
        if not np.isin(actions[..., 3], (0, 1)).all():
            raise ValueError("Dense jaw intent must be binary")
        self._pid = None
        self._handle = None

    def __len__(self) -> int:
        return len(self.rows["source_observation_indices"])

    def _frames(self):
        import h5py

        if self._handle is None or self._pid != os.getpid():
            self._handle = h5py.File(self.source_path, "r")
            self._pid = os.getpid()
        return self._handle["pixels"]

    def __getitem__(self, index: SupportsIndex) -> dict:
        i = int(index)
        row = int(self.rows["source_observation_indices"][i])
        return {
            "image": np.asarray(self._frames()[row]),
            "state": self.rows["states"][i].copy(),
            "actions": self.rows["actions"][i].copy(),
            "actions_is_pad": self.rows["actions_is_pad"][i].copy(),
            "prompt": self.prompt,
            "episode_index": int(self.rows["episode_indices"][i]),
            "source_row": row,
        }


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def create_torch_dataset(
    data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)
    if data_config.dense_archive_path is not None:
        if data_config.action_chunks_path is not None or data_config.frame_indices_path is not None:
            raise ValueError("Dense raw-frame archives replace LeRobot chunk selection entirely")
        return RawFrameChunkDataset(data_config.dense_archive_path)

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        delta_timestamps=None
        if data_config.action_chunks_path
        else {key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys},
    )

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    if data_config.action_chunks_path is not None:
        if data_config.frame_indices_path is not None:
            raise ValueError("Use explicit sparse observations or dense frame selection, not both")
        dataset = IndexedActionDataset(dataset, data_config.action_chunks_path, action_horizon)

    if data_config.frame_indices_path is not None:
        indices = np.load(data_config.frame_indices_path, allow_pickle=False)
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer) or indices.size == 0:
            raise ValueError("Frame selection must be a nonempty one-dimensional integer array")
        if np.any(indices < 0) or np.any(indices >= len(dataset)) or np.any(indices[1:] <= indices[:-1]):
            raise ValueError("Frame selection must be sorted, unique, and within the full dataset")
        dataset = torch.utils.data.Subset(dataset, indices.tolist())

    return dataset


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        datasets=data_config.datasets,
    )


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    if data_config.rlds_data_dir is not None:
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    dataset = create_torch_dataset(data_config, action_horizon, model_config)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]
