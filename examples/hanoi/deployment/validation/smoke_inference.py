"""Load a Hanoi checkpoint and run local inference without robot dependencies."""

import dataclasses
import datetime
import hashlib
import importlib.metadata
import json
import logging
import pathlib
import time

import jax
import numpy as np
import tyro

from examples.hanoi.evaluation import metrics as evaluation
from openpi.policies import hanoi_policy
from openpi.policies import policy_config
from openpi.training import config as _config


def main(
    checkpoint_dir: pathlib.Path = pathlib.Path("checkpoints/pi05_hanoi_aaaa_to_cccc/hanoi_20260914/exports/29999"),
    config_name: str = "pi05_hanoi_aaaa_to_cccc",
    direction: str = "aaaa_to_cccc",
    episode_path: pathlib.Path | None = None,
    output_dir: pathlib.Path = pathlib.Path("data/hanoi/local_smoke"),
    repeats: int = 3,
):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    if repeats < 2:
        raise ValueError("Use at least two requests to measure warm inference and repeatability")
    if direction not in hanoi_policy.PROMPTS or config_name not in (
        f"pi05_hanoi_{direction}",
        "pi05_hanoi_multitask",
    ):
        raise ValueError("Choose a Hanoi configuration and its matching direction")
    evaluation.validate_export(checkpoint_dir)
    config = _config.get_config(config_name)
    prompt = hanoi_policy.PROMPTS[direction]
    requests = {
        "dummy": {
            "observation/image": np.full((224, 224, 3), 128, dtype=np.uint8),
            "observation/state": np.array([0.4, 0.0, 0.2, 0.0, 0.0, 0.0, 0.034], dtype=np.float32),
            "prompt": prompt,
        }
    }
    reference = None
    anchor = None
    if episode_path is not None:
        import h5py

        with h5py.File(episode_path, "r") as episode:
            age = episode["command_monotonic_ns"][:] - episode["image_receipt_monotonic_ns"][:]
            eligible = np.flatnonzero((age >= 0) & (age <= 50_000_000))
            if not len(eligible):
                raise ValueError("The debug episode has no eligible observation anchors")
            anchor = int(eligible[0])
            requests["recorded"] = {
                "observation/image": episode["pixels"][anchor],
                "observation/state": episode["proprio"][anchor, :7],
                "prompt": prompt,
            }
            stop = min(anchor + 63, len(episode["action_abs"]))
            reference = episode["action_abs"][anchor:stop]
            reference = np.concatenate([reference, np.repeat(reference[-1:], 63 - len(reference), axis=0)])
    norm_path = checkpoint_dir / "assets" / hanoi_policy.REPO_ID / "norm_stats.json"
    report = {
        "started_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "checkpoint": str(checkpoint_dir.resolve()),
        "config": config_name,
        "model": dataclasses.asdict(config.model),
        "direction": direction,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("jax", "jaxlib", "flax", "orbax-checkpoint", "numpy", "torch", "transformers")
        },
        "normalization_sha256": hashlib.sha256(norm_path.read_bytes()).hexdigest(),
        "export_sha256": hashlib.sha256((checkpoint_dir / "export.json").read_bytes()).hexdigest(),
        "sampling_steps": 10,
        "numpy_noise_seed": 44,
        "episode_path": str(episode_path.resolve()) if episode_path else None,
        "recorded_anchor": anchor,
        "hardware_executed": False,
        "requests": {},
    }
    logging.info("Loading %s on %s", checkpoint_dir, report["devices"])
    started = time.perf_counter()
    policy = policy_config.create_trained_policy(config, checkpoint_dir, sample_kwargs={"num_steps": 10})
    report["load_seconds"] = time.perf_counter() - started
    if policy.metadata != hanoi_policy.CONTRACT:
        raise ValueError("Served metadata does not match the Hanoi contract")
    noise = np.random.default_rng(44).standard_normal((63, 32)).astype(np.float32)
    arrays = {"noise": noise}
    if reference is not None:
        arrays["recorded_reference"] = reference
    for name, observation in requests.items():
        durations = []
        first = None
        for repeat in range(repeats):
            started = time.perf_counter()
            actions = np.asarray(policy.infer(observation, noise=noise)["actions"])
            durations.append(time.perf_counter() - started)
            if actions.shape != (63, 4) or not np.isfinite(actions).all():
                raise ValueError(f"{name}: expected finite (63, 4) absolute XYZ/jaw actions")
            if first is None:
                first = actions.copy()
            else:
                np.testing.assert_allclose(actions, first, atol=1e-6, rtol=1e-5)
                np.testing.assert_array_equal(actions[:, 3] >= 0.5, first[:, 3] >= 0.5)
            logging.info("%s request %d: shape=%s, %.3f s", name, repeat + 1, actions.shape, durations[-1])
        arrays[f"{name}_image"] = observation["observation/image"]
        arrays[f"{name}_state"] = observation["observation/state"]
        arrays[f"{name}_actions"] = first
        result = {
            "shape": list(first.shape),
            "finite": True,
            "repeatability_passed": True,
            "request_seconds": durations,
            "warm_median_seconds": float(np.median(durations[1:])),
            "first_action": first[0].tolist(),
        }
        if name == "recorded":
            result["debug_mean_xyz_error_mm"] = float(
                np.linalg.norm(first[:, :3] - reference[:, :3], axis=1).mean() * 1000
            )
            result["debug_jaw_agreement"] = float(np.mean((first[:, 3] >= 0.5) == (reference[:, 3] >= 0.5)))
        report["requests"][name] = result
    report["passed"] = True
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "inference_samples.npz", **arrays)
    (output_dir / "inference_report.json").write_text(json.dumps(report, indent=2) + "\n")
    logging.info("PASS: report and input/output samples saved to %s", output_dir)


if __name__ == "__main__":
    tyro.cli(main)
