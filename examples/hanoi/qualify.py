"""Qualify a real GPU profile with the unchanged full pi0.5 training step."""

import dataclasses
import functools
import gc
import json
import logging
import pathlib
import time

import jax
import jax.numpy as jnp
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tyro

from examples.hanoi import dataset
from examples.hanoi import storage
from examples.hanoi import telemetry
from openpi.models import model as _model
from openpi.policies import hanoi_policy
from openpi.policies import policy_config
from openpi.shared import nnx_utils
from openpi.training import checkpoints
from openpi.training import config as _config
from openpi.training import data_loader
from openpi.training import sharding
from scripts import train


def require_validated_data() -> dict:
    root = pathlib.Path("data/hanoi")
    conversion = json.loads((root / "conversion.json").read_text())
    data_validation = json.loads((root / "data_validation.json").read_text())
    execution_validation = json.loads((root / "execution_validation.json").read_text())
    if not data_validation["passed"] or not execution_validation["passed"]:
        raise ValueError("Data and command-contract checks must pass before allocating training work")
    if data_validation["conversion_sha256"] != dataset.sha256(root / "conversion.json"):
        raise ValueError("Data validation does not match this conversion")
    if execution_validation["audit_sha256"] != dataset.sha256(root / "audit.json"):
        raise ValueError("Command validation does not match this source audit")
    if conversion["contract"] != hanoi_policy.CONTRACT or data_validation["rows_checked"] != conversion["rows"]:
        raise ValueError("The complete converted dataset must match the current observation/action contract")
    for name, digest in conversion["selection_sha256"].items():
        if dataset.sha256(root / "indices" / name) != digest:
            raise ValueError(f"Train/validation/test anchor selection changed: {name}")
    verified_files = data_validation["verified_parquet_files"]
    if set(verified_files) != set(conversion["parquet_sha256"]):
        raise ValueError("Converted file inventory changed after data verification")
    for name, metadata in verified_files.items():
        current = (pathlib.Path(conversion["dataset_root"]) / name).stat()
        if current.st_size != metadata["bytes"] or current.st_mtime_ns != metadata["mtime_ns"]:
            raise ValueError(f"Converted data changed after verification: {name}")
    for task in dataset.TASKS:
        config = _config.get_config(f"pi05_hanoi_{task}")
        assets = config.assets_dirs / config.data.repo_id
        provenance = json.loads((assets / "provenance.json").read_text())
        if (
            provenance["config"] != config.name
            or provenance["audit_sha256"] != execution_validation["audit_sha256"]
            or provenance["norm_stats_sha256"] != dataset.sha256(assets / "norm_stats.json")
            or provenance["horizon"] != config.model.action_horizon
            or not provenance["includes_terminal_holds"]
        ):
            raise ValueError(f"Normalization provenance differs from the validated dataset: {task}")
    return conversion


def save_interval(step_seconds: float) -> int:
    maximum = max(1, min(1000, int(300 / step_seconds)))
    return max(i for i in range(1, maximum + 1) if 5000 % i == 0)


def restore_template(state):
    """Keep the compiled optimizer metadata and shardings without retaining device buffers."""
    return jax.tree.map(lambda value: jax.ShapeDtypeStruct(value.shape, value.dtype, sharding=value.sharding), state)


def main(
    profile: str,
    exp_name: str,
    output_path: pathlib.Path,
    fsdp_devices: int = 1,
    config_name: str = "pi05_hanoi_multitask",
):
    logging.basicConfig(level=logging.INFO, force=True)
    conversion = require_validated_data()
    code_identity = dataset.training_code_identity()
    lock_identity = dataset.sha256(pathlib.Path("uv.lock"))
    if jax.default_backend() != "gpu" or jax.device_count() != fsdp_devices:
        raise ValueError("Qualification requires exactly the requested number of actual GPUs")
    gpu_info = telemetry.hardware_info(telemetry.allocated_devices())
    config = dataclasses.replace(_config.get_config(config_name), exp_name=exp_name, fsdp_devices=fsdp_devices)
    # Qualification never resumes or replaces a production experiment.
    manager, _ = checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir, keep_period=None, overwrite=False, resume=False
    )
    begin = time.monotonic()
    mesh = sharding.make_mesh(fsdp_devices)
    batch_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    loader = data_loader.create_data_loader(config, sharding=batch_sharding, shuffle=True)
    iterator = iter(loader)
    batch = next(iterator)
    rng, init_rng = jax.random.split(jax.random.key(config.seed))
    state, state_sharding = train.init_train_state(config, init_rng, mesh, resume=False)
    jax.block_until_ready(state)
    compiled_step = jax.jit(
        functools.partial(train.train_step, config),
        in_shardings=(replicated, state_sharding, batch_sharding),
        out_shardings=(state_sharding, replicated),
        donate_argnums=(1,),
    )
    timings = []
    startup_seconds = None
    for step in range(110):
        started = time.monotonic()
        with sharding.set_mesh(mesh):
            state, info = compiled_step(rng, state, batch)
        info = jax.device_get(info)
        if not all(np.isfinite(value).all() for value in info.values()):
            raise FloatingPointError(f"Nonfinite qualification metrics at step {step}: {info}")
        batch = next(iterator)
        if step >= 10:
            timings.append(time.monotonic() - started)
        if step == 9:
            startup_seconds = time.monotonic() - begin
        if step % 10 == 0:
            logging.info("Qualification step %d: %s", step, info)
    memory = {str(device): device.memory_stats() for device in jax.local_devices()}
    started = time.monotonic()
    checkpoints.save_state(manager, state, loader, 109)
    manager.wait_until_finished()
    save_seconds = time.monotonic() - started
    exported = checkpoints.export_policy_checkpoint(manager, 109)
    full_checkpoint_bytes = storage.logical_bytes(config.checkpoint_dir / "109")
    inference_checkpoint_bytes = storage.logical_bytes(exported)
    # Release live arrays before reloading to test restore without requiring two full states in VRAM.
    # Reinitializing here creates new static optimizer functions incompatible with compiled_step.
    shape = restore_template(state)
    del state
    gc.collect()
    restored = checkpoints.restore_state(manager, shape, loader)
    if int(restored.step) != 110:
        raise ValueError("Checkpoint restore lost optimizer progress")
    with sharding.set_mesh(mesh):
        restored, info = compiled_step(rng, restored, batch)
    if not all(np.isfinite(value).all() for value in jax.device_get(info).values()):
        raise FloatingPointError("The restored optimizer did not produce a finite training step")
    # Inference uses EMA parameters, matching the exported policy checkpoint.
    restored_optimizer_step = int(restored.step)
    del restored, compiled_step, shape, state_sharding, info
    jax.clear_caches()
    gc.collect()
    model = config.model.load(_model.restore_params(exported / "params", dtype=jnp.bfloat16))
    model.eval()
    # Exercise the evaluation forward batch on the restored export as well as
    # single-observation serving. This catches evaluation-only memory failures.
    loss = nnx_utils.module_jit(model.compute_loss, static_argnames=("train",))
    evaluation_loss = np.asarray(loss(jax.random.key(41), batch[0], batch[1], train=False))
    if evaluation_loss.shape != (config.batch_size, 63) or not np.isfinite(evaluation_loss).all():
        raise ValueError("Restored EMA evaluation forward pass failed")
    del loss
    jax.clear_caches()
    sample = nnx_utils.module_jit(model.sample_actions)
    observation = jax.tree.map(lambda value: value[:1], batch[0])
    predictions = np.asarray(sample(jax.random.key(42), observation, num_steps=10))
    if predictions.shape != (1, 63, 32) or not np.isfinite(predictions).all():
        raise ValueError("Restored EMA inference failed")
    physical_observation = jax.tree.map(lambda value: value[:8], batch[0])
    physical_predictions = np.asarray(sample(jax.random.key(43), physical_observation, num_steps=10))
    if physical_predictions.shape != (8, 63, 32) or not np.isfinite(physical_predictions).all():
        raise ValueError("Restored EMA physical-metric sampling failed")
    del sample, model
    jax.clear_caches()
    gc.collect()
    # Exercise the repository's actual serving factory with exported assets and
    # canonical camera/state inputs. No websocket server or robot driver is started.
    native = LeRobotDataset(conversion["repo_id"])
    episode = conversion["episodes"][0]
    row = native[episode["global_start"]]
    policy = policy_config.create_trained_policy(config, exported, sample_kwargs={"num_steps": 10})
    serving = policy.infer(
        {
            "observation/image": hanoi_policy.parse_image(row["image"]),
            "observation/state": np.asarray(row["state"]),
            "prompt": hanoi_policy.PROMPTS[episode["direction"]],
        },
        noise=np.zeros((63, 32), dtype=np.float32),
    )
    if serving["actions"].shape != (63, 4) or not np.isfinite(serving["actions"]).all():
        raise ValueError("The native serving factory failed the exported Hanoi contract")
    p95 = float(np.percentile(timings, 95))
    if code_identity != dataset.training_code_identity() or lock_identity != dataset.sha256(pathlib.Path("uv.lock")):
        raise ValueError("Code or environment changed while qualification was running")
    dataset.write_json(
        output_path,
        {
            "qualified": True,
            "profile": profile,
            "fsdp_devices": fsdp_devices,
            "gpu_info": gpu_info,
            "config": config_name,
            "global_batch": config.batch_size,
            "horizon": config.model.action_horizon,
            "measured_steps": len(timings),
            "step_seconds_median": float(np.median(timings)),
            "step_seconds_p95": p95,
            "startup_seconds": startup_seconds,
            "checkpoint_save_seconds": save_seconds,
            "full_checkpoint_bytes": full_checkpoint_bytes,
            "inference_checkpoint_bytes": inference_checkpoint_bytes,
            "save_interval": save_interval(p95),
            "memory_stats": memory,
            "checkpoint_restore_passed": True,
            "restored_optimizer_step": restored_optimizer_step,
            "ema_inference_passed": True,
            "ema_evaluation_forward_passed": True,
            "ema_evaluation_sampling_passed": True,
            "native_serving_factory_passed": True,
            "checkpoint_dir": str(config.checkpoint_dir),
            "lockfile_sha256": lock_identity,
            "training_code_sha256": code_identity,
        },
    )
    manager.close()


if __name__ == "__main__":
    tyro.cli(main)
