"""Diagnostic policy input: recorded velocity matched to live task progress."""

import dataclasses
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from openpi_client import hanoi

from examples.hanoi.deployment.async_inference import Observation


class RecordedVelocity:
    """Match forward within the current recorded open/closed-jaw interval.

    XYZ selects a recorded measured pose; only that row's measured velocity is
    substituted. Gripper commands advance the interval, disambiguating a descent
    from a lift at the same XYZ. The live gripper dwell replaces the recorded one.
    No recorded position, image, jaw value, or action is sent to the robot.
    """

    def __init__(self, states: np.ndarray, actions: np.ndarray):
        self.states = np.asarray(states, dtype=np.float32).copy()
        actions = np.asarray(actions)
        if (
            self.states.ndim != 2
            or self.states.shape[1] != 7
            or len(self.states) == 0
            or actions.shape != (len(self.states), 4)
            or not np.isfinite(self.states).all()
            or not np.isfinite(actions).all()
        ):
            raise ValueError("Recorded velocity requires finite aligned (N, 7) states and (N, 4) actions")
        jaws = actions[:, 3] >= 0.5
        if not jaws[0]:
            raise ValueError("Recorded velocity experiment requires an open-jaw episode start")
        self.boundaries = np.r_[0, np.flatnonzero(jaws[1:] != jaws[:-1]) + 1, len(jaws)]
        self.jaws = jaws[self.boundaries[:-1]]
        self.motion_starts = self.boundaries[:-1].copy()
        for phase in range(1, len(self.jaws)):
            start, end = self.boundaries[phase : phase + 2]
            moving = np.flatnonzero(np.linalg.norm(actions[start:end, :3] - actions[start, :3], axis=1) > 1e-6)
            if len(moving):
                self.motion_starts[phase] = start + moving[0]
        self.phase = 0
        self.row = 0
        self.sha256 = hashlib.sha256(self.states.tobytes() + actions.astype(np.float32).tobytes()).hexdigest()

    @classmethod
    def load(cls, path: Path):
        metadata = json.loads(path.with_suffix(".json").read_text())
        if metadata["contract"] != hanoi.CONTRACT or metadata["prompt"] != hanoi.PROMPTS["aaaa_to_cccc"]:
            raise ValueError("Recorded velocity requires the matching forward Hanoi episode")
        with h5py.File(path, "r") as episode:
            return cls(episode["proprio"][:, :7], episode["action_abs"][:])

    def gripper_command(self, *, jaw_open: bool):
        if jaw_open == self.jaws[self.phase]:
            return
        if self.phase + 1 == len(self.jaws):
            raise ValueError("No further gripper phase in the recorded velocity episode")
        self.phase += 1
        self.row = int(self.motion_starts[self.phase])

    def apply(self, observation: Observation) -> Observation:
        measured = np.asarray(observation.data["observation/state"])
        end = self.boundaries[self.phase + 1]
        distances = np.linalg.norm(self.states[self.row : end, :3] - measured[:3], axis=1)
        self.row += int(np.argmin(distances))
        state = measured.copy()
        state[3:6] = self.states[self.row, 3:6]
        return dataclasses.replace(
            observation,
            data={**observation.data, "observation/state": state},
            velocity_override={
                "source": "recorded",
                "reference_row": self.row,
                "gripper_phase": self.phase,
                "match_distance_mm": float(np.linalg.norm(self.states[self.row, :3] - measured[:3]) * 1000),
                "recorded_xyz_m": self.states[self.row, :3].tolist(),
                "measured_state": measured.tolist(),
                "policy_velocity_m_s": state[3:6].tolist(),
            },
        )
