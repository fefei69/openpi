import dataclasses

import jax
import numpy as np
import pytest

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def test_frame_selection_preserves_global_chunk_indices(monkeypatch, tmp_path):
    class Metadata:
        def __init__(self, repo_id):
            self.fps = 30

    class Native:
        def __init__(self, repo_id, delta_timestamps):
            self.horizon = len(delta_timestamps["actions"])

        def __len__(self):
            return 20

        def __getitem__(self, index):
            # Two ten-row episodes; queries must remain bounded by the original episode.
            end = (index // 10 + 1) * 10 - 1
            return {"actions": np.minimum(np.arange(index, index + self.horizon), end)}

    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDatasetMetadata", Metadata)
    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDataset", Native)
    path = tmp_path / "indices.npy"
    np.save(path, [3, 9, 17])
    config = _config.DataConfig(repo_id="local/test", frame_indices_path=str(path))
    selected = _data_loader.create_torch_dataset(config, 4, pi0_config.Pi0Config())
    np.testing.assert_array_equal(selected[1]["actions"], [9, 9, 9, 9])
    np.testing.assert_array_equal(selected[2]["actions"], [17, 18, 19, 19])
    for invalid in ([3, 3], [4, 2], [-1], [20], [], [1.5]):
        np.save(path, invalid)
        with pytest.raises(ValueError, match="Frame selection"):
            _data_loader.create_torch_dataset(config, 4, pi0_config.Pi0Config())


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)
