import dataclasses
import json

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from examples.hanoi.pipeline import deliver
from examples.hanoi.pipeline import manage
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import normalize
from openpi.training import checkpoints
from openpi.training import config
from openpi.training import utils


def test_inference_export_survives_native_pruning_and_restores_ema(tmp_path):
    model = nnx.Linear(3, 2, rngs=nnx.Rngs(0))
    params = nnx.state(model)
    ema = jax.tree.map(lambda value: value + 2.0, params)
    optimizer = optax.adam(1e-3)
    with at.disable_typechecking():
        state = utils.TrainState(
            step=jnp.array(2),
            params=params,
            model_def=nnx.graphdef(model),
            opt_state=optimizer.init(params),
            tx=optimizer,
            ema_decay=0.99,
            ema_params=ema,
        )
    stats = {"state": normalize.NormStats(mean=np.zeros(7), std=np.ones(7))}

    class Loader:
        def data_config(self):
            return config.DataConfig(asset_id="hanoi", norm_stats=stats)

    manager, _ = checkpoints.initialize_checkpoint_dir(
        tmp_path / "run", keep_period=None, overwrite=False, resume=False
    )
    checkpoints.save_state(manager, state, Loader(), 1)
    abandoned = tmp_path / "run/exports/.1-interrupted.partial"
    abandoned.mkdir(parents=True)
    (abandoned / "orphan").write_text("interrupted export")
    exported = checkpoints.export_policy_checkpoint(manager, 1)
    assert not abandoned.exists()
    assert manage.complete_step(tmp_path / "run") == 1
    assert not (exported / "train_state").exists()
    manifest = json.loads((exported / "export.json").read_text())
    for name in manifest["files"]:
        assert (exported / name).stat().st_ino == (tmp_path / "run/1" / name).stat().st_ino
    checkpoints.save_state(manager, dataclasses.replace(state, step=jnp.array(3)), Loader(), 2)
    manager.wait_until_finished()
    assert not (tmp_path / "run/1").exists()
    restored = _model.restore_params(exported / "params", restore_type=np.ndarray)
    for expected, actual in zip(jax.tree.leaves(ema.to_pure_dict()), jax.tree.leaves(restored), strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert (exported / "assets/hanoi/norm_stats.json").is_file()
    restored_state = checkpoints.restore_state(manager, state, Loader())
    assert int(restored_state.step) == 3
    assert deliver.optimizer_step(tmp_path / "run/2") == 3
    second = checkpoints.export_policy_checkpoint(manager, 2)
    assert checkpoints.export_policy_checkpoint(manager, 2) == second
    manager.close()


def test_export_rejects_uncommitted_checkpoint(tmp_path):
    manager, _ = checkpoints.initialize_checkpoint_dir(
        tmp_path / "run", keep_period=None, overwrite=False, resume=False
    )
    with pytest.raises(ValueError, match="not complete"):
        checkpoints.export_policy_checkpoint(manager, 123)
    assert not (tmp_path / "run/exports/123").exists()
    manager.close()
