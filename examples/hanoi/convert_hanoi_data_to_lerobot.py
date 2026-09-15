"""Validate then convert the two Hanoi HDF5 files with the pinned LeRobot writer.

Run with python -m examples.hanoi.convert_hanoi_data_to_lerobot. Raw files are read only;
new destinations are created by default. --resume verifies source identity and preserves
interrupted output before continuing from the last committed episode.
"""

import contextlib
import datetime
import json
import logging
import pathlib
import shutil

import filelock
import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
import tyro

from examples.hanoi import dataset as hanoi_dataset
from openpi.policies import hanoi_policy


def main(
    data_dir: pathlib.Path = pathlib.Path("/scratch/cw5167/datasets"),
    output_dir: pathlib.Path = pathlib.Path("data/hanoi"),
    *,
    audit_only: bool = False,
    resume: bool = False,
):
    logging.basicConfig(level=logging.INFO, force=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    with filelock.FileLock(str(output_dir / "conversion.lock"), timeout=0):
        convert(data_dir, output_dir, audit_only=audit_only, resume=resume)


def convert(data_dir: pathlib.Path, output_dir: pathlib.Path, *, audit_only: bool, resume: bool):
    logging.info("Auditing source labels, routes, timestamps, endpoints, and SHA-256 identities")
    manifest = hanoi_dataset.audit(data_dir)
    previous = output_dir / "audit.json"
    destination = HF_LEROBOT_HOME / hanoi_policy.REPO_ID
    if destination.exists() and (not resume or not previous.exists() or json.loads(previous.read_text()) != manifest):
        raise FileExistsError(f"Existing conversion requires --resume with the identical audited source: {destination}")
    hanoi_dataset.write_json(output_dir / "audit.json", manifest)
    if audit_only:
        return
    if destination.exists():
        metadata = LeRobotDatasetMetadata(hanoi_policy.REPO_ID)
        committed = metadata.total_episodes
        if metadata.total_frames != committed * 7201:
            raise ValueError("Partially committed metadata requires repair before resume")
        expected = {destination / metadata.get_data_file_path(index) for index in range(committed)}
        orphaned = [path for path in (destination / "data").rglob("*.parquet") if path not in expected]
        if (destination / "images").exists():
            orphaned.append(destination / "images")
        if orphaned:
            preserved = output_dir / "interrupted" / datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%S%f")
            preserved.mkdir(parents=True)
            for path in orphaned:
                shutil.move(str(path), str(preserved / path.name))
        converted = LeRobotDataset(hanoi_policy.REPO_ID)
        converted.start_image_writer(num_threads=8)
        logging.info("Resuming after %d committed episodes", committed)
    else:
        converted = LeRobotDataset.create(
            repo_id=hanoi_policy.REPO_ID,
            robot_type="trossen_wxai_cartesian",
            fps=30,
            use_videos=False,
            features={
                "image": {"dtype": "image", "shape": (224, 224, 3), "names": ["height", "width", "channel"]},
                "state": {"dtype": "float32", "shape": (7,), "names": hanoi_policy.CONTRACT["state"]},
                "actions": {"dtype": "float32", "shape": (4,), "names": hanoi_policy.CONTRACT["actions"]},
            },
            image_writer_threads=8,
        )
    first_episode = converted.meta.total_episodes
    try:
        with contextlib.ExitStack() as stack:
            handles = {
                d: stack.enter_context(h5py.File(hanoi_dataset.source_path(data_dir, d), "r"))
                for d in hanoi_dataset.DIRECTIONS
            }
            for episode in manifest["episodes"]:
                if episode["episode_index"] < first_episode:
                    continue
                handle = handles[episode["direction"]]
                start = episode["source_start"]
                stop = start + episode["length"]
                states = handle["proprio"][start:stop, :7]
                actions = handle["action_abs"][start:stop]
                # Bound decoded image memory independently of episode length.
                for block_start in range(start, stop, 128):
                    pixels = handle["pixels"][block_start : min(block_start + 128, stop)]
                    for local_index, image in enumerate(pixels, start=block_start - start):
                        converted.add_frame(
                            {
                                "image": image,
                                "state": states[local_index],
                                "actions": actions[local_index],
                                "task": hanoi_policy.PROMPTS[episode["direction"]],
                            }
                        )
                converted.save_episode()
                logging.info("Converted episode %d/100", episode["episode_index"] + 1)
    finally:
        converted.stop_image_writer()
    hanoi_dataset.write_indices(manifest, data_dir, output_dir / "indices")
    manifest["dataset_root"] = str(destination.resolve())
    manifest["selection_sha256"] = {
        path.name: hanoi_dataset.sha256(path) for path in sorted((output_dir / "indices").glob("*.npy"))
    }
    manifest["parquet_sha256"] = {
        str(path.relative_to(destination)): hanoi_dataset.sha256(path)
        for path in sorted((destination / "data").rglob("*.parquet"))
    }
    if converted.meta.total_frames != manifest["rows"] or converted.meta.total_episodes != 100:
        raise ValueError("Converted row or episode count does not match the source")
    hanoi_dataset.write_json(output_dir / "conversion.json", manifest)
    logging.info("Conversion complete: %s", destination)


if __name__ == "__main__":
    tyro.cli(main)
