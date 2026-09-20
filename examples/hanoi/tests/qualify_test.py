import dataclasses
import types

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from examples.hanoi.data import dataset
from examples.hanoi.training import qualify
from openpi.policies import hanoi_policy
from openpi.shared import array_typing as at
from openpi.training import checkpoints
from openpi.training import config as training_config
from openpi.training import utils


def test_restored_optimizer_matches_uninterrupted_compiled_step(tmp_path):
    model = nnx.Linear(3, 2, rngs=nnx.Rngs(0))
    params = nnx.state(model)
    optimizer = optax.adam(1e-3)
    with at.disable_typechecking():
        state = utils.TrainState(
            step=jnp.array(0),
            params=params,
            model_def=nnx.graphdef(model),
            opt_state=optimizer.init(params),
            tx=optimizer,
            ema_decay=0.99,
            ema_params=jax.tree.map(lambda value: value.copy(), params),
        )

    def step(state):
        gradients = jax.grad(lambda p: sum(jnp.sum(x**2) for x in jax.tree.leaves(p)))(state.params)
        updates, opt_state = state.tx.update(gradients, state.opt_state, state.params)
        params = optax.apply_updates(state.params, updates)
        ema = jax.tree.map(lambda old, new: 0.99 * old + 0.01 * new, state.ema_params, params)
        return dataclasses.replace(state, step=state.step + 1, params=params, opt_state=opt_state, ema_params=ema)

    shardings = jax.tree.map(lambda value: value.sharding, state)
    compiled = jax.jit(step, in_shardings=(shardings,), out_shardings=shardings, donate_argnums=(0,))
    state = compiled(state)
    template = qualify.restore_template(state)
    assert template.tx is state.tx
    assert all(isinstance(value, jax.ShapeDtypeStruct) for value in jax.tree.leaves(template))
    loader = types.SimpleNamespace(data_config=training_config.DataConfig)
    manager, _ = checkpoints.initialize_checkpoint_dir(
        tmp_path / "run", keep_period=None, overwrite=False, resume=False
    )
    try:
        checkpoints.save_state(manager, state, loader, 0)
        manager.wait_until_finished()
        expected = jax.device_get(compiled(state))
        del state
        restored = checkpoints.restore_state(manager, template, loader)
        actual = jax.device_get(compiled(restored))
        assert int(actual.step) == 2
        for want, got in zip(jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True):
            np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-7)
    finally:
        manager.close()


@pytest.mark.parametrize("changed", ["selection", "parquet", "normalization"])
def test_validation_gate_rejects_data_changes_before_gpu_work(tmp_path, monkeypatch, changed):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "data/hanoi"
    (root / "indices").mkdir(parents=True)
    audit_path = root / "audit.json"
    audit_path.write_text("{}")
    indices = root / "indices/multitask_train.npy"
    np.save(indices, [1, 3])
    parquet = tmp_path / "dataset/episode.parquet"
    parquet.parent.mkdir()
    parquet.write_bytes(b"verified dataset placeholder")
    conversion = {
        "contract": hanoi_policy.CONTRACT,
        "rows": 4,
        "dataset_root": str(parquet.parent),
        "selection_sha256": {indices.name: dataset.sha256(indices)},
        "parquet_sha256": {parquet.name: dataset.sha256(parquet)},
    }
    dataset.write_json(root / "conversion.json", conversion)
    dataset.write_json(
        root / "data_validation.json",
        {
            "passed": True,
            "rows_checked": 4,
            "conversion_sha256": dataset.sha256(root / "conversion.json"),
            "verified_parquet_files": {
                parquet.name: {"bytes": parquet.stat().st_size, "mtime_ns": parquet.stat().st_mtime_ns}
            },
        },
    )
    dataset.write_json(root / "execution_validation.json", {"passed": True, "audit_sha256": dataset.sha256(audit_path)})
    configs = {}
    for task in dataset.TASKS:
        name = f"pi05_hanoi_{task}"
        assets = tmp_path / "assets" / name
        norm = assets / "local/hanoi/norm_stats.json"
        norm.parent.mkdir(parents=True)
        norm.write_text("{}")
        configs[name] = types.SimpleNamespace(
            name=name,
            assets_dirs=assets,
            data=types.SimpleNamespace(repo_id="local/hanoi"),
            model=types.SimpleNamespace(action_horizon=63),
        )
        dataset.write_json(
            norm.parent / "provenance.json",
            {
                "config": name,
                "audit_sha256": dataset.sha256(audit_path),
                "norm_stats_sha256": dataset.sha256(norm),
                "horizon": 63,
                "includes_terminal_holds": True,
            },
        )
    monkeypatch.setattr(training_config, "get_config", configs.__getitem__)
    assert qualify.require_validated_data() == conversion
    if changed == "selection":
        np.save(indices, [2, 3])
        message = "anchor selection changed"
    elif changed == "parquet":
        parquet.write_bytes(b"changed")
        message = "Converted data changed"
    else:
        norm.write_text('{"changed": true}')
        message = "Normalization provenance differs"
    with pytest.raises(ValueError, match=message):
        qualify.require_validated_data()
