"""Check the converted rows, episode boundaries, images, and training/serving parity."""

import contextlib
import dataclasses
import json
import logging
import pathlib

import h5py
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import pyarrow.parquet as pq
import tyro

from examples.hanoi.data import compute_norm_stats
from examples.hanoi.data import dataset
from openpi import transforms
from openpi.policies import hanoi_policy
from openpi.training import config as _config
from openpi.training import data_loader


def main(
    data_dir: pathlib.Path = pathlib.Path("/scratch/cw5167/datasets"),
    manifest_path: pathlib.Path = pathlib.Path("data/hanoi/conversion.json"),
):
    logging.basicConfig(level=logging.INFO)
    manifest = json.loads(manifest_path.read_text())
    native = LeRobotDataset(manifest["repo_id"], delta_timestamps={"actions": [i / 30 for i in range(63)]})
    config = _config.get_config("pi05_hanoi_multitask")
    data_config = config.data.create(config.assets_dirs, config.model)
    training_transform = transforms.compose(
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            transforms.Normalize(data_config.norm_stats, use_quantiles=True),
            *data_config.model_transforms.inputs,
        ]
    )
    serving_transform = transforms.compose(
        [
            *data_config.data_transforms.inputs,
            transforms.Normalize(data_config.norm_stats, use_quantiles=True),
            *data_config.model_transforms.inputs,
        ]
    )
    checked_images = 0
    with contextlib.ExitStack() as stack:
        handles = {d: stack.enter_context(h5py.File(dataset.source_path(data_dir, d), "r")) for d in dataset.DIRECTIONS}
        for episode in manifest["episodes"]:
            handle = handles[episode["direction"]]
            start, stop = episode["source_start"], episode["source_start"] + episode["length"]
            proprio, actions = handle["proprio"][start:stop], handle["action_abs"][start:stop]
            parquet = pathlib.Path(manifest["dataset_root"]) / native.meta.get_data_file_path(episode["episode_index"])
            columns = pq.read_table(parquet, columns=["state", "actions", "episode_index", "frame_index", "index"])
            np.testing.assert_array_equal(np.array(columns["state"].to_pylist(), dtype=np.float32), proprio[:, :7])
            np.testing.assert_array_equal(np.array(columns["actions"].to_pylist(), dtype=np.float32), actions)
            np.testing.assert_array_equal(columns["episode_index"].to_numpy(), np.full(7201, episode["episode_index"]))
            np.testing.assert_array_equal(columns["frame_index"].to_numpy(), np.arange(7201))
            np.testing.assert_array_equal(columns["index"].to_numpy(), np.arange(7201) + episode["global_start"])
            anchors = np.unique(
                np.r_[np.linspace(0, 7200, 12, dtype=int), 29, 30, 155, 156, 197, 198, 386, 387, 416, 417, 7199]
            )
            for anchor in anchors:
                row = native[episode["global_start"] + int(anchor)]
                row["prompt"] = hanoi_policy.PROMPTS[episode["direction"]]
                rgb = handle["pixels"][start + anchor]
                np.testing.assert_array_equal(hanoi_policy.parse_image(row["image"]), rgb)
                np.testing.assert_array_equal(row["actions"].numpy(), dataset.action_chunk(actions, int(anchor)))
                np.testing.assert_array_equal(row["actions_is_pad"].numpy(), anchor + np.arange(63) >= 7201)
                serving = {"observation/image": rgb, "observation/state": proprio[anchor, :7], "prompt": row["prompt"]}
                trained = training_transform(dict(row))
                served = serving_transform(serving)
                for key in ("state", "tokenized_prompt", "tokenized_prompt_mask"):
                    np.testing.assert_array_equal(trained[key], served[key])
                for key in trained["image"]:
                    np.testing.assert_array_equal(trained["image"][key], served["image"][key])
                    assert trained["image_mask"][key] == served["image_mask"][key]
                numeric = compute_norm_stats.numeric_batch(proprio, actions, np.array([anchor]))
                repacked = transforms.compose(
                    [*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs]
                )(dict(row))
                for key in ("state", "actions"):
                    np.testing.assert_array_equal(numeric[key][0], repacked[key])
                inverse = transforms.compose(
                    [
                        transforms.Unnormalize(data_config.norm_stats, use_quantiles=True),
                        *data_config.data_transforms.outputs,
                    ]
                )({"state": trained["state"].copy(), "actions": trained["actions"].copy()})
                np.testing.assert_allclose(inverse["actions"], row["actions"].numpy(), atol=1e-7, rtol=0)
                checked_images += 1
            logging.info(
                "Verified full numeric rows and sampled image/chunk parity for episode %d", episode["episode_index"]
            )
    for task in dataset.TASKS:
        for split in dataset.SPLITS:
            path = manifest_path.parent / "indices" / f"{task}_{split}.npy"
            selected_config = dataclasses.replace(data_config, frame_indices_path=str(path))
            selected = data_loader.create_torch_dataset(selected_config, 63, config.model)
            indices = np.load(path, allow_pickle=False)
            for offset in (0, len(indices) // 2, len(indices) - 1):
                np.testing.assert_array_equal(selected[offset]["actions"], native[int(indices[offset])]["actions"])
    dataset.write_json(
        manifest_path.parent / "data_validation.json",
        {
            "passed": True,
            "rows_checked": manifest["rows"],
            "image_and_chunk_probes": checked_images,
            "conversion_sha256": dataset.sha256(manifest_path),
            "training_serving_parity": True,
            "verified_parquet_files": {
                name: {
                    "bytes": (pathlib.Path(manifest["dataset_root"]) / name).stat().st_size,
                    "mtime_ns": (pathlib.Path(manifest["dataset_root"]) / name).stat().st_mtime_ns,
                }
                for name in manifest["parquet_sha256"]
            },
        },
    )


if __name__ == "__main__":
    tyro.cli(main)
