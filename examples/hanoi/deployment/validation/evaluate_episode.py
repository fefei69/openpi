"""Measure a selected policy against one transferred HDF5 demonstration, without hardware."""

import datetime
import hashlib
import importlib.metadata
import json
import logging
import pathlib
import time

import h5py
import jax
import numpy as np
import tyro

from examples.hanoi.evaluation import metrics as evaluation
from openpi.policies import hanoi_policy
from openpi.policies import policy_config
from openpi.training import config as _config


def distribution(values: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def metrics(predicted: np.ndarray, target: np.ndarray, valid: np.ndarray) -> dict:
    native = evaluation.Metrics()
    native.update(predicted, target, valid)
    errors = np.linalg.norm(predicted[..., :3] - target[..., :3], axis=-1) * 1000
    mean_per_anchor = (errors * valid).sum(axis=1) / valid.sum(axis=1)
    confusion = native.confusion
    return {
        **native.result(),
        "xyz_error_mm": distribution(errors[valid]),
        "mean_chunk_xyz_error_mm": distribution(mean_per_anchor),
        "jaw_accuracy": float(np.trace(confusion) / confusion.sum()),
        "jaw_close_recall": float(confusion[0, 0] / confusion[0].sum()) if confusion[0].sum() else None,
        "jaw_open_recall": float(confusion[1, 1] / confusion[1].sum()) if confusion[1].sum() else None,
        "xyz_within_mm": {str(bound): float(np.mean(errors[valid] <= bound)) for bound in (1, 2, 5, 10)},
    }


def main(
    episode_path: pathlib.Path = pathlib.Path("data/hanoi/deployment_debug/aaaa_to_cccc_episode_000/episode.h5"),
    checkpoint_dir: pathlib.Path = pathlib.Path("checkpoints/pi05_hanoi_aaaa_to_cccc/hanoi_20260914/exports/29999"),
    config_name: str = "pi05_hanoi_aaaa_to_cccc",
    output_dir: pathlib.Path = pathlib.Path("data/hanoi/episode_accuracy"),
    max_anchors: int = 0,
    seed: int = 44,
):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    if max_anchors < 0:
        raise ValueError("max_anchors must be zero (all) or positive")
    metadata = json.loads(episode_path.with_suffix(".json").read_text())
    direction = metadata["provenance"]["direction"]
    if config_name not in (f"pi05_hanoi_{direction}", "pi05_hanoi_multitask"):
        raise ValueError("The checkpoint configuration does not match this episode's direction")
    if metadata["contract"] != hanoi_policy.CONTRACT or metadata["prompt"] != hanoi_policy.PROMPTS[direction]:
        raise ValueError("The transferred episode has a different observation/action contract")
    evaluation.validate_export(checkpoint_dir)
    with h5py.File(episode_path, "r") as episode:
        if len(episode["ep_len"]) != 1 or int(episode["ep_offset"][0]) != 0:
            raise ValueError("Expected a single episode with rebased offsets")
        states = episode["proprio"][:, :7]
        actions = episode["action_abs"][:]
        move_ids = episode["move_idx"][:]
        age = episode["command_monotonic_ns"][:] - episode["image_receipt_monotonic_ns"][:]
        eligible = np.flatnonzero((age >= 0) & (age <= 50_000_000))
        np.testing.assert_array_equal(eligible, metadata["eligible_local_indices"])
        if not len(eligible) or int(episode["ep_len"][0]) != len(actions):
            raise ValueError("Episode length or eligible anchor count is invalid")
        indices = eligible
        if max_anchors and len(indices) > max_anchors:
            indices = indices[np.linspace(0, len(indices) - 1, max_anchors, dtype=int)]
        future = indices[:, None] + np.arange(63)
        valid = future < len(actions)
        target = actions[np.minimum(future, len(actions) - 1)]
        predicted = np.empty(target.shape, dtype=np.float64)
        durations = np.empty(len(indices))
        noises = np.empty((len(indices), 63, 32), dtype=np.float32)
        output_dir.mkdir(parents=True, exist_ok=True)
        config = _config.get_config(config_name)
        started = time.perf_counter()
        policy = policy_config.create_trained_policy(config, checkpoint_dir, sample_kwargs={"num_steps": 10})
        load_seconds = time.perf_counter() - started
        logging.info("Loaded checkpoint; evaluating %d/%d fresh anchors", len(indices), len(eligible))
        for number, anchor in enumerate(indices):
            observation = {
                "observation/image": episode["pixels"][int(anchor)],
                "observation/state": states[anchor],
                "prompt": metadata["prompt"],
            }
            # Key noise by source row so a subset uses the same draws as a full run.
            noise = np.random.default_rng(np.random.SeedSequence([seed, int(anchor)])).standard_normal((63, 32))
            noises[number] = noise.astype(np.float32)
            request_started = time.perf_counter()
            result = np.asarray(policy.infer(observation, noise=noises[number])["actions"])
            durations[number] = time.perf_counter() - request_started
            if result.shape != (63, 4) or not np.isfinite(result).all():
                raise ValueError(f"Invalid prediction at row {anchor}")
            predicted[number] = result
            count = number + 1
            if count % 250 == 0 or count == len(indices):
                elapsed = time.perf_counter() - started
                partial = metrics(predicted[:count, :9], target[:count, :9], valid[:count, :9])
                progress = {
                    "completed": count,
                    "total": len(indices),
                    "elapsed_seconds": elapsed,
                    "prefix9_mean_xyz_mm": partial["mean_valid_xyz_mm"],
                    "prefix9_jaw_accuracy": partial["jaw_accuracy"],
                }
                (output_dir / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")
                logging.info(
                    "Progress %d/%d, %.1f s, prefix XYZ %.3f mm",
                    count,
                    len(indices),
                    elapsed,
                    partial["mean_valid_xyz_mm"],
                )
        elapsed = time.perf_counter() - started
    hold = np.broadcast_to(states[indices, None, :3], predicted[..., :3].shape)
    errors = np.linalg.norm(predicted[..., :3] - target[..., :3], axis=-1) * 1000
    hold_errors = np.linalg.norm(hold - target[..., :3], axis=-1) * 1000
    norm_path = checkpoint_dir / "assets" / hanoi_policy.REPO_ID / "norm_stats.json"
    report = {
        "completed_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "episode_path": str(episode_path.resolve()),
        "provenance": metadata["provenance"],
        "config": config_name,
        "checkpoint": str(checkpoint_dir.resolve()),
        "normalization_sha256": hashlib.sha256(norm_path.read_bytes()).hexdigest(),
        "export_sha256": hashlib.sha256((checkpoint_dir / "export.json").read_bytes()).hexdigest(),
        "source_rows": len(actions),
        "eligible_anchors": len(eligible),
        "evaluated_anchors": len(indices),
        "excluded_stale_anchors": len(actions) - len(eligible),
        "sampling_steps": 10,
        "numpy_seed": seed,
        "noise_scheme": "One draw per anchor, NumPy SeedSequence([seed, source_row])",
        "evaluation_mode": "Recorded RGB and measured state at each anchor; predictions are never fed back as observations",
        "terminal_padding_excluded": True,
        "hardware_executed": False,
        "backend": jax.default_backend(),
        "versions": {
            name: importlib.metadata.version(name) for name in ("jax", "jaxlib", "numpy", "flax", "orbax-checkpoint")
        },
        "first_reference": metrics(predicted[:, :1], target[:, :1], valid[:, :1]),
        "prefix_9": metrics(predicted[:, :9], target[:, :9], valid[:, :9]),
        "horizon_63": metrics(predicted, target, valid),
        "hold_position_baseline_xyz_mm": {
            name: float(
                ((hold_errors[:, :length] * valid[:, :length]).sum(axis=1) / valid[:, :length].sum(axis=1)).mean()
            )
            for name, length in (("first_reference", 1), ("prefix_9", 9), ("horizon_63", 63))
        },
        "by_move": {
            str(move): {
                "prefix_9": metrics(
                    predicted[move_ids[indices] == move, :9],
                    target[move_ids[indices] == move, :9],
                    valid[move_ids[indices] == move, :9],
                ),
                "horizon_63": metrics(
                    predicted[move_ids[indices] == move],
                    target[move_ids[indices] == move],
                    valid[move_ids[indices] == move],
                ),
            }
            for move in np.unique(move_ids[indices])
        },
        "by_horizon": [
            {
                "reference": offset + 1,
                "seconds": (offset + 1) / 30,
                "count": int(valid[:, offset].sum()),
                "xyz_error_mm": distribution(errors[valid[:, offset], offset]),
                "jaw_accuracy": float(
                    np.mean((predicted[valid[:, offset], offset, 3] >= 0.5) == target[valid[:, offset], offset, 3])
                ),
            }
            for offset in range(63)
        ],
        "timing": {
            "load_seconds": load_seconds,
            "total_seconds": elapsed,
            "warm_request_seconds": distribution(durations[1:] if len(durations) > 1 else durations),
        },
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(
        output_dir / "predictions.npz",
        indices=indices,
        predictions=predicted,
        targets=target,
        valid=valid,
        states=states[indices],
        moves=move_ids[indices],
        noise=noises,
        request_seconds=durations,
    )
    logging.info(
        "Completed: first %.3f mm, prefix %.3f mm, full %.3f mm; saved to %s",
        report["first_reference"]["mean_valid_xyz_mm"],
        report["prefix_9"]["mean_valid_xyz_mm"],
        report["horizon_63"]["mean_valid_xyz_mm"],
        output_dir,
    )


if __name__ == "__main__":
    tyro.cli(main)
