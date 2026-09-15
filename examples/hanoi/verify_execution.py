"""Replay all teacher actions through the hardware-free deployment command builder."""

import contextlib
import json
import pathlib

import h5py
import numpy as np
import tyro

from examples.hanoi import dataset
from examples.hanoi import execution


def main(
    data_dir: pathlib.Path = pathlib.Path("/scratch/cw5167/datasets"),
    audit_path: pathlib.Path = pathlib.Path("data/hanoi/audit.json"),
):
    manifest = json.loads(audit_path.read_text())
    maxima = np.zeros(4)
    count, gripper_count = 0, 0
    with contextlib.ExitStack() as stack:
        handles = {d: stack.enter_context(h5py.File(dataset.source_path(data_dir, d), "r")) for d in dataset.DIRECTIONS}
        for episode in manifest["episodes"]:
            start, length = episode["source_start"], episode["length"]
            actions = handles[episode["direction"]]["action_abs"][start : start + length]
            executor = execution.ReferenceExecutor(actions[0, :3].astype(np.float64))
            tick = 0
            events = []
            while tick < length - 1:
                # Exercise every inference offset relative to the existing command boundary.
                delay = count % 9
                anchor = max(executor.fresh_after_tick, tick - delay, 0)
                command = executor.plan(
                    dataset.action_chunk(actions, anchor),
                    observation_tick=anchor,
                    now_tick=tick,
                    image_age_s=0.01,
                    task=episode["direction"],
                )
                if command.kind == "cartesian":
                    expected = actions[tick : tick + command.ticks, :3]
                    actual = command.sample(np.arange(1, command.ticks + 1) / command.ticks)
                    maxima[0] = max(maxima[0], np.linalg.norm(actual - expected, axis=-1).max())
                    for derivative in range(1, 4):
                        maxima[derivative] = max(
                            maxima[derivative],
                            np.linalg.norm(command.sample(np.linspace(0, 1, 101), derivative), axis=-1).max(),
                        )
                    count += 1
                elif command.kind == "gripper":
                    events.append((tick, command.ticks, command.jaw_open))
                    gripper_count += 1
                if command.ticks <= 0:
                    raise ValueError("Executor did not advance")
                tick += command.ticks
            expected_events = [
                (m * 480 + offset, duration, jaw)
                for m in range(15)
                for offset, duration, jaw in ((156, 42, False), (387, 30, True))
            ]
            if events != expected_events:
                raise ValueError(f"Gripper event timing changed in episode {episode['episode_index']}")
    if maxima[0] > 0.0001 or np.any(maxima[1:] > execution.LIMITS):
        raise ValueError(f"Teacher command equivalence failed: {maxima}")
    dataset.write_json(
        audit_path.parent / "execution_validation.json",
        {
            "passed": True,
            "audit_sha256": dataset.sha256(audit_path),
            "cartesian_commands": count,
            "gripper_events": gripper_count,
            "max_reference_error_m": maxima[0],
            "max_speed_m_s": maxima[1],
            "max_acceleration_m_s2": maxima[2],
            "max_jerk_m_s3": maxima[3],
            "hardware_validated": False,
        },
    )


if __name__ == "__main__":
    tyro.cli(main)
