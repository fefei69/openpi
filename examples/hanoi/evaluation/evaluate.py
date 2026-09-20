"""Select an EMA snapshot on validation, evaluate test once, then compact candidates."""

import collections
import dataclasses
import gc
import json
import logging
import pathlib
import time

import filelock
import jax
import jax.numpy as jnp
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import torch
import tyro

from examples.hanoi.data import dataset
from examples.hanoi.evaluation import metrics as evaluation
from examples.hanoi.evaluation import verify_serving
from examples.hanoi.pipeline import storage
from examples.hanoi.training import qualify
from openpi import transforms
from openpi.models import model as _model
from openpi.policies import hanoi_policy
from openpi.shared import nnx_utils
from openpi.training import checkpoints
from openpi.training import config as _config
from openpi.training import data_loader


def collate(rows: list[dict]) -> dict:
    return jax.tree.map(lambda *values: np.stack(values), *rows)


def make_loader(data: data_loader.Dataset, batch_size: int, workers: int):
    return torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        drop_last=False,
        collate_fn=collate,
        multiprocessing_context="spawn" if workers else None,
        persistent_workers=workers > 0,
    )


def evaluate_checkpoint(
    config: _config.TrainConfig, snapshot: pathlib.Path, split: str, manifest: dict, *, batch_size: int = 32
) -> dict:
    started = time.perf_counter()
    task = config.name.removeprefix("pi05_hanoi_")
    selection_path = pathlib.Path("data/hanoi/indices") / f"{task}_{split}.npy"
    indices = np.load(selection_path, allow_pickle=False)
    data_config = config.data.create(config.assets_dirs, config.model)
    data_config = dataclasses.replace(
        data_config,
        frame_indices_path=str(selection_path),
        norm_stats=checkpoints.load_norm_stats(snapshot / "assets", data_config.asset_id),
    )
    selected = data_loader.create_torch_dataset(data_config, 63, config.model)
    transformed = data_loader.transform_dataset(selected, data_config)
    model = config.model.load(_model.restore_params(snapshot / "params", dtype=jnp.bfloat16))
    model.eval()
    loss = nnx_utils.module_jit(model.compute_loss, static_argnames=("train",))
    mesh = jax.sharding.Mesh(np.array(jax.devices()), ("eval_batch",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("eval_batch"))
    flow_sum, flow_count = 0.0, 0
    flow_by_episode = collections.defaultdict(lambda: [0.0, 0])
    flow_by_direction = collections.defaultdict(lambda: [0.0, 0])
    ends = np.array([ep["global_start"] + ep["length"] for ep in manifest["episodes"]])
    loader = make_loader(transformed, batch_size, config.num_workers)
    flow_started = time.perf_counter()
    for number, batch in enumerate(loader):
        padded, count = evaluation.pad_batch(batch, batch_size)
        observation = _model.Observation.from_dict(
            jax.device_put({k: v for k, v in padded.items() if k != "actions"}, sharding)
        )
        rng = jax.random.fold_in(jax.random.key(42), number)
        values = np.asarray(loss(rng, observation, jax.device_put(padded["actions"], sharding), train=False))[:count]
        per_anchor = values.mean(axis=-1)
        if not np.isfinite(per_anchor).all():
            raise FloatingPointError("Nonfinite validation/test flow loss")
        episode_ids = np.searchsorted(ends, indices[flow_count : flow_count + count], side="right")
        for episode_id in np.unique(episode_ids):
            mask = episode_ids == episode_id
            flow_by_episode[int(episode_id)][0] += float(per_anchor[mask].sum(dtype=np.float64))
            flow_by_episode[int(episode_id)][1] += int(mask.sum())
            direction = manifest["episodes"][episode_id]["direction"]
            flow_by_direction[direction][0] += float(per_anchor[mask].sum(dtype=np.float64))
            flow_by_direction[direction][1] += int(mask.sum())
        flow_sum += float(per_anchor.sum(dtype=np.float64))
        flow_count += count
        if number % 100 == 0:
            logging.info(
                "%s %s flow anchors %d/%d in %.1f seconds",
                snapshot.name,
                split,
                flow_count,
                len(indices),
                time.perf_counter() - flow_started,
            )
    if flow_count != len(indices):
        raise ValueError("Evaluation dropped or repeated anchors")
    flow_finished = time.perf_counter()
    del loader, transformed, selected, loss
    gc.collect()
    native = LeRobotDataset(manifest["repo_id"], delta_timestamps={"actions": [i / 30 for i in range(63)]})
    physical_indices = evaluation.sampled_indices(indices, manifest["episodes"])
    input_transform = transforms.compose(
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            transforms.Normalize(data_config.norm_stats, use_quantiles=True),
            *data_config.model_transforms.inputs,
        ]
    )
    output_transform = transforms.compose(
        [
            transforms.Unnormalize(data_config.norm_stats, use_quantiles=True),
            *data_config.data_transforms.outputs,
        ]
    )
    sample = nnx_utils.module_jit(model.sample_actions)
    metrics = collections.defaultdict(evaluation.Metrics)
    swap_metrics = evaluation.Metrics()
    sample_batch = 8
    for offset in range(0, len(physical_indices), sample_batch):
        selected_indices = physical_indices[offset : offset + sample_batch]
        episode_ids = np.searchsorted(ends, selected_indices, side="right")
        inputs, targets, masks, swapped = [], [], [], []
        for index, episode_id in zip(selected_indices, episode_ids, strict=True):
            episode = manifest["episodes"][episode_id]
            row = native[int(index)]
            row["prompt"] = hanoi_policy.PROMPTS[episode["direction"]]
            targets.append(np.asarray(row["actions"]).copy())
            masks.append(~np.asarray(row["actions_is_pad"]))
            inputs.append(input_transform(dict(row)))
            if task == "multitask":
                other = next(direction for direction in dataset.DIRECTIONS if direction != episode["direction"])
                swapped.append(input_transform({**row, "prompt": hanoi_policy.PROMPTS[other]}))
        batch, count = evaluation.pad_batch(collate(inputs), sample_batch)
        # Repeat the same explicit noise for the correct/swapped prompt comparison.
        noise = jax.random.normal(jax.random.fold_in(jax.random.key(43), offset), (sample_batch, 63, 32))
        noise = jax.device_put(noise, sharding)
        observation = _model.Observation.from_dict(
            jax.device_put({k: v for k, v in batch.items() if k != "actions"}, sharding)
        )
        predicted = np.asarray(sample(jax.random.key(0), observation, num_steps=10, noise=noise))
        absolute = output_transform({"state": batch["state"].copy(), "actions": predicted.copy()})["actions"][:count]
        target, valid = np.stack(targets), np.stack(masks)
        metrics["all"].update(absolute, target, valid)
        for episode_id in np.unique(episode_ids):
            mask = episode_ids == episode_id
            direction = manifest["episodes"][episode_id]["direction"]
            metrics[f"episode_{episode_id}"].update(absolute[mask], target[mask], valid[mask])
            metrics[direction].update(absolute[mask], target[mask], valid[mask])
        if swapped:
            swap_batch, _ = evaluation.pad_batch(collate(swapped), sample_batch)
            swap_observation = _model.Observation.from_dict(
                jax.device_put({k: v for k, v in swap_batch.items() if k != "actions"}, sharding)
            )
            swap_prediction = np.asarray(sample(jax.random.key(0), swap_observation, num_steps=10, noise=noise))
            swap_absolute = output_transform({"state": swap_batch["state"].copy(), "actions": swap_prediction.copy()})[
                "actions"
            ][:count]
            swap_metrics.update(swap_absolute, target, valid)
    physical_finished = time.perf_counter()
    result = {
        "split": split,
        "flow_loss": flow_sum / flow_count,
        "flow_anchors": flow_count,
        "flow_by_episode": {
            str(key): {"loss": value[0] / value[1], "anchors": value[1]} for key, value in flow_by_episode.items()
        },
        "flow_by_direction": {
            key: {"loss": value[0] / value[1], "anchors": value[1]} for key, value in flow_by_direction.items()
        },
        "physical": {key: metric.result() for key, metric in metrics.items()},
        "sampling_steps": 10,
        "flow_seed": 42,
        "sampling_seed": 43,
        "flow_batch_size": batch_size,
    }
    if task == "multitask":
        result["prompt_swap_diagnostic"] = swap_metrics.result()
    del sample, model, native
    jax.clear_caches()
    gc.collect()
    finished = time.perf_counter()
    # Host conversions above synchronize device results. Timings include data I/O
    # and compilation; these are stage durations, not isolated inference latency.
    result["timing_seconds"] = {
        "setup": flow_started - started,
        "flow": flow_finished - flow_started,
        "physical": physical_finished - flow_finished,
        "cleanup": finished - physical_finished,
        "total": finished - started,
    }
    logging.info("%s %s evaluation timings: %s", snapshot.name, split, result["timing_seconds"])
    return result


def main(config_name: str, exp_name: str):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    if jax.default_backend() != "gpu":
        raise ValueError("Full checkpoint evaluation requires a GPU allocation")
    manifest = qualify.require_validated_data()
    config = dataclasses.replace(_config.get_config(config_name), exp_name=exp_name)
    if config_name not in {f"pi05_hanoi_{task}" for task in dataset.TASKS}:
        raise ValueError("Evaluation is restricted to the three Hanoi models")
    results_dir = pathlib.Path("data/hanoi/runs") / exp_name / config_name
    results_dir.mkdir(parents=True, exist_ok=True)
    training_identity_path = config.checkpoint_dir / "hanoi_identity.json"
    training_identity = json.loads(training_identity_path.read_text())
    if (
        training_identity["config"] != config_name
        or training_identity["exp_name"] != exp_name
        or training_identity["training_code_sha256"] != dataset.training_code_identity()
        or training_identity["lockfile_sha256"] != dataset.sha256(pathlib.Path("uv.lock"))
        or training_identity["conversion_sha256"] != dataset.sha256(pathlib.Path("data/hanoi/conversion.json"))
    ):
        raise ValueError("Evaluation implementation/environment/data differs from the trained checkpoint")
    identity = {
        "config": config_name,
        "exp_name": exp_name,
        "conversion_sha256": dataset.sha256(pathlib.Path("data/hanoi/conversion.json")),
        "evaluation_code_sha256": dataset.sha256(pathlib.Path(__file__)),
        "metrics_code_sha256": dataset.sha256(pathlib.Path(evaluation.__file__)),
        "serving_code_sha256": dataset.sha256(pathlib.Path(verify_serving.__file__)),
        "training_identity_sha256": dataset.sha256(training_identity_path),
    }
    with storage.pipeline_lock(exp_name), filelock.FileLock(str(results_dir / "evaluation.lock"), timeout=0):
        if (results_dir / "complete.json").exists():
            complete = json.loads((results_dir / "complete.json").read_text())
            if complete["identity"] != identity:
                raise ValueError("Completed evaluation identity changed; do not silently rerun test")
            for name, digest in complete["result_sha256"].items():
                if dataset.sha256(results_dir / name) != digest:
                    raise ValueError("A completed evaluation result changed")
            selected_path = pathlib.Path(complete["selected_checkpoint"])
            if dataset.sha256(selected_path / "export.json") != complete["export_sha256"]:
                raise ValueError("Selected inference snapshot changed")
            evaluation.compact_exports(config.checkpoint_dir / "exports", selected_path)
            return
        snapshots = sorted((config.checkpoint_dir / "exports").glob("[0-9]*"), key=lambda path: int(path.name))
        if not snapshots or not any(int(path.name) == config.num_train_steps - 1 for path in snapshots):
            raise ValueError("Training has not produced its complete final EMA snapshot")
        validation = []
        for snapshot in snapshots:
            evaluation.validate_export(snapshot)
            norm_path = snapshot / "assets" / config.data.repo_id / "norm_stats.json"
            if dataset.sha256(norm_path) != training_identity["norm_stats_sha256"]:
                raise ValueError("Exported normalization assets differ from the training identity")
            snapshot_identity = {**identity, "export_sha256": dataset.sha256(snapshot / "export.json")}
            output = results_dir / f"validation_{snapshot.name}.json"
            if output.exists():
                result = json.loads(output.read_text())
                if result["identity"] != snapshot_identity:
                    raise ValueError("Existing validation results have a different identity")
            else:
                result = {
                    "identity": snapshot_identity,
                    "step": int(snapshot.name),
                    **evaluate_checkpoint(config, snapshot, "val", manifest),
                }
                dataset.write_json(output, result)
            validation.append(result)
        selected = min(validation, key=lambda value: (value["flow_loss"], value["step"]))
        selected_path = config.checkpoint_dir / "exports" / str(selected["step"])
        selection = {
            "identity": identity,
            "step": selected["step"],
            "criterion": "validation_flow_loss",
            "validation_flow_loss": selected["flow_loss"],
            "checkpoint": str(selected_path.resolve()),
        }
        selection_path = results_dir / "selection.json"
        if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
            raise ValueError("Selection is locked before test; refusing to change it")
        dataset.write_json(selection_path, selection)
        test_path = results_dir / "test.json"
        if test_path.exists():
            test = json.loads(test_path.read_text())
            if test["identity"] != selected["identity"] or test["step"] != selected["step"]:
                raise ValueError("Existing test result belongs to a different selected snapshot")
        else:
            test = {
                "identity": selected["identity"],
                "step": selected["step"],
                **evaluate_checkpoint(config, selected_path, "test", manifest),
            }
            dataset.write_json(test_path, test)
        serving_path = results_dir / "serving_validation.json"
        serving = verify_serving.verify(config, selected_path, manifest, serving_path)
        if not serving["passed"]:
            raise ValueError("Selected checkpoint failed serving parity")
        # Persist the completed result before pruning only this run's unselected exports.
        complete = {
            "identity": identity,
            "selected_checkpoint": str(selected_path.resolve()),
            "selected_step": selected["step"],
            "test_flow_loss": test["flow_loss"],
            "test_physical": test["physical"],
            "hardware_success_measured": False,
            "serving_parity_passed": True,
            "export_sha256": dataset.sha256(selected_path / "export.json"),
            "result_sha256": {
                path.name: dataset.sha256(path)
                for path in [selection_path, test_path, serving_path, *sorted(results_dir.glob("validation_*.json"))]
            },
        }
        dataset.write_json(results_dir / "complete.json", complete)
        evaluation.compact_exports(config.checkpoint_dir / "exports", selected_path)


if __name__ == "__main__":
    tyro.cli(main)
