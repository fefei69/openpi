"""Compute exact training-anchor statistics without decoding unused RGB images.

Uses the same XYZ delta transform as the native loader, including all 63 targets and
terminal holds. verify_data checks this numeric path against native LeRobot samples.
"""

import contextlib
import json
import logging
import pathlib

import h5py
import numpy as np
import tyro

from examples.hanoi import dataset
from openpi import transforms
from openpi.shared import normalize


def numeric_batch(proprio: np.ndarray, actions: np.ndarray, anchors: np.ndarray) -> dict:
    indices = np.minimum(anchors[:, None] + np.arange(63), len(actions) - 1)
    batch = {"state": proprio[anchors, :7].copy(), "actions": actions[indices].copy()}
    return transforms.DeltaActions(transforms.make_bool_mask(3, -1))(batch)


def main(
    data_dir: pathlib.Path = pathlib.Path("/scratch/cw5167/datasets"),
    audit_path: pathlib.Path = pathlib.Path("data/hanoi/audit.json"),
    assets_dir: pathlib.Path = pathlib.Path("assets"),
):
    logging.basicConfig(level=logging.INFO)
    manifest = json.loads(audit_path.read_text())
    stats = {task: {key: normalize.RunningStats() for key in ("state", "actions")} for task in dataset.TASKS}
    counts = dict.fromkeys(dataset.TASKS, 0)
    with contextlib.ExitStack() as stack:
        handles = {d: stack.enter_context(h5py.File(dataset.source_path(data_dir, d), "r")) for d in dataset.DIRECTIONS}
        for episode in manifest["episodes"]:
            if episode["split"] != "train":
                continue
            handle = handles[episode["direction"]]
            start, stop = episode["source_start"], episode["source_start"] + episode["length"]
            proprio, actions = handle["proprio"][start:stop], handle["action_abs"][start:stop]
            anchors = np.flatnonzero(dataset.eligible_anchors(handle, start, stop))
            for offset in range(0, len(anchors), 256):
                selected = anchors[offset : offset + 256]
                batch = numeric_batch(proprio, actions, selected)
                for task in (episode["direction"], "multitask"):
                    for key, value in batch.items():
                        stats[task][key].update(value.astype(np.float64))
                    counts[task] += len(selected)
            logging.info("Computed statistics for training episode %d", episode["episode_index"])
    for task in dataset.TASKS:
        config_name = f"pi05_hanoi_{task}"
        output = assets_dir / config_name / manifest["repo_id"]
        normalize.save(output, {key: value.get_statistics() for key, value in stats[task].items()})
        dataset.write_json(
            output / "provenance.json",
            {
                "config": config_name,
                "audit_sha256": dataset.sha256(audit_path),
                "anchors": counts[task],
                "horizon": 63,
                "includes_terminal_holds": True,
                "norm_stats_sha256": dataset.sha256(output / "norm_stats.json"),
            },
        )
        logging.info("Saved %s statistics from %d training anchors", task, counts[task])


if __name__ == "__main__":
    tyro.cli(main)
