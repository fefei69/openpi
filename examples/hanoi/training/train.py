"""Launch one qualified Hanoi run through the repository's JAX trainer."""

import dataclasses
import json
import logging
import pathlib

import filelock
import jax
import tyro

from examples.hanoi.data import dataset
from examples.hanoi.pipeline import storage as _storage
from examples.hanoi.pipeline import telemetry
from examples.hanoi.training import qualify
from openpi.training import config as _config
from scripts import train


def main(config_name: str, exp_name: str, qualification_path: pathlib.Path):
    logging.basicConfig(level=logging.INFO, force=True)
    conversion = qualify.require_validated_data()
    qualification = json.loads(qualification_path.read_text())
    if not qualification["qualified"] or qualification["lockfile_sha256"] != dataset.sha256(pathlib.Path("uv.lock")):
        raise ValueError("This environment has not passed GPU qualification")
    if qualification["training_code_sha256"] != dataset.training_code_identity():
        raise ValueError("Training implementation changed since GPU qualification")
    if jax.default_backend() != "gpu" or jax.device_count() != qualification["fsdp_devices"]:
        raise ValueError("Allocated GPUs do not match the qualified profile")
    devices = telemetry.allocated_devices()
    gpu_info = telemetry.hardware_info(devices)

    # Device UUIDs may differ, but type, VRAM, and driver must match the tested profile.
    def hardware(text: str) -> list[tuple[str, ...]]:
        return sorted(
            tuple(part.strip() for i, part in enumerate(line.split(",")) if i != 1) for line in text.splitlines()
        )

    if hardware(gpu_info) != hardware(qualification["gpu_info"]):
        raise ValueError("GPU type, VRAM, or driver differs from the qualified hardware")
    config = dataclasses.replace(
        _config.get_config(config_name),
        exp_name=exp_name,
        fsdp_devices=qualification["fsdp_devices"],
        save_interval=qualification["save_interval"],
        resume=True,
    )
    if config_name not in {f"pi05_hanoi_{task}" for task in dataset.TASKS}:
        raise ValueError("Managed training is restricted to the three Hanoi configurations")
    contract_identity = {
        "config": config_name,
        "exp_name": exp_name,
        "contract": conversion["contract"],
        "conversion_sha256": dataset.sha256(pathlib.Path("data/hanoi/conversion.json")),
        "lockfile_sha256": dataset.sha256(pathlib.Path("uv.lock")),
        "training_code_sha256": dataset.training_code_identity(),
        "norm_stats_sha256": dataset.sha256(config.assets_dirs / config.data.repo_id / "norm_stats.json"),
        "batch_size": config.batch_size,
        "seed": config.seed,
        "num_train_steps": config.num_train_steps,
    }
    config.checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
    with (
        _storage.pipeline_lock(exp_name),
        filelock.FileLock(str(config.checkpoint_dir) + ".writer.lock", timeout=0),
    ):
        _storage.validate(qualification_path, exp_name, output_path=pathlib.Path("data/hanoi/storage_validation.json"))
        identity_path = config.checkpoint_dir / "hanoi_identity.json"
        if identity_path.exists():
            if json.loads(identity_path.read_text()) != contract_identity:
                raise ValueError("Refusing to resume a checkpoint with a different training/data identity")
        elif config.checkpoint_dir.exists() and any(config.checkpoint_dir.iterdir()):
            raise ValueError("Refusing to adopt an existing experiment without a managed identity")
        placement_path = config.checkpoint_dir / "hanoi_placement.json"
        if placement_path.exists():
            previous = json.loads(placement_path.read_text())
            previous_devices = previous["fsdp_devices"]
            if previous_devices != config.fsdp_devices and previous_devices not in qualification.get(
                "cross_mesh_restore_from", []
            ):
                raise ValueError("Changing device count requires a qualified cross-mesh checkpoint restore")
        dataset.write_json(identity_path, contract_identity)
        dataset.write_json(
            placement_path,
            {
                "fsdp_devices": config.fsdp_devices,
                "qualification_path": str(qualification_path.resolve()),
                "gpu_info": gpu_info,
            },
        )
        log_path = pathlib.Path("data/hanoi/runs") / exp_name / config_name / "telemetry.jsonl"
        with telemetry.monitor(devices, config.checkpoint_dir, log_path):
            train.main(config)


if __name__ == "__main__":
    tyro.cli(main)
