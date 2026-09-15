"""Hardware-free command builder for the shared 30 Hz Cartesian action contract.

Cartesian outputs describe a 0.3 s quintic with explicit endpoint derivatives. A
future driver client must preserve these derivatives and the committed interval;
calling a rest-to-rest endpoint API is not equivalent. This module moves no robot.
"""

import dataclasses
import math

import numpy as np

RATE_HZ = 30
PREFIX = 9
LIMITS = (0.145, 0.23, 1.3)
_VANDER = np.polynomial.polynomial.polyvander(np.linspace(0, 1, PREFIX + 1), 5)
_FIT = np.linalg.pinv(_VANDER)
_ENDPOINT_INVERSE = np.linalg.inv(np.array([[1, 1, 1], [3, 4, 5], [6, 12, 20]], dtype=float))


@dataclasses.dataclass(frozen=True)
class Command:
    kind: str
    start_tick: int
    ticks: int
    coefficients: np.ndarray | None = None
    jaw_open: bool | None = None

    def sample(self, fraction: np.ndarray, derivative: int = 0) -> np.ndarray:
        if self.coefficients is None:
            raise ValueError("Only Cartesian commands have a trajectory")
        coefficients = self.coefficients.copy()
        for _ in range(derivative):
            coefficients = coefficients[1:] * np.arange(1, len(coefficients))[:, None] / (self.ticks / RATE_HZ)
        return np.polynomial.polynomial.polyvander(fraction, len(coefficients) - 1) @ coefficients


def fit_cartesian(
    targets: np.ndarray, position: np.ndarray, velocity: np.ndarray, acceleration: np.ndarray, start_tick: int
) -> Command:
    """Estimate endpoint derivatives from all nine references; retain the prior command's initial derivatives."""
    if targets.shape != (PREFIX, 3) or not np.isfinite(targets).all():
        raise ValueError("A Cartesian prefix requires nine finite XYZ references")
    duration = PREFIX / RATE_HZ
    fitted = _FIT @ np.vstack([position, targets])
    end_velocity = np.arange(1, 6) @ fitted[1:] / duration
    end_acceleration = (np.arange(2, 6) * np.arange(1, 5)) @ fitted[2:] / duration**2
    coefficients = np.zeros((6, 3))
    coefficients[0] = position
    coefficients[1] = velocity * duration
    coefficients[2] = acceleration * duration**2 / 2
    coefficients[3:] = _ENDPOINT_INVERSE @ np.stack(
        [
            targets[-1] - coefficients[:3].sum(axis=0),
            duration * end_velocity - coefficients[1] - 2 * coefficients[2],
            duration**2 * end_acceleration - 2 * coefficients[2],
        ]
    )
    command = Command("cartesian", start_tick, PREFIX, coefficients)
    reconstruction = command.sample(np.arange(1, PREFIX + 1) / PREFIX)
    if np.linalg.norm(reconstruction - targets, axis=-1).max() > 0.0001:
        raise ValueError("Predicted references cannot be represented by a continuous prefix within 0.1 mm")
    # Dense verification for this offline builder; live integration also needs calibrated
    # workspace/IK and continuous trajectory bounds, independently of these sampled checks.
    for derivative, limit in enumerate(LIMITS, start=1):
        if np.linalg.norm(command.sample(np.linspace(0, 1, 101), derivative), axis=-1).max() > limit:
            raise ValueError(f"Predicted trajectory exceeds derivative-{derivative} limit {limit}")
    return command


@dataclasses.dataclass
class ReferenceExecutor:
    position: np.ndarray
    velocity: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(3))
    acceleration: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(3))
    jaw_open: bool = True
    available_tick: int = 0
    fresh_after_tick: int = 0
    task: str = ""

    def plan(
        self, actions: np.ndarray, *, observation_tick: int, now_tick: float, image_age_s: float, task: str
    ) -> Command:
        if self.task and task != self.task:
            raise ValueError("Reset the executor and queued actions before changing tasks")
        self.task = task
        if not math.isfinite(now_tick) or now_tick < observation_tick:
            raise ValueError("Invalid observation/command time")
        if now_tick < self.available_tick:
            return Command("wait", math.ceil(now_tick), self.available_tick - math.ceil(now_tick))
        if observation_tick < self.fresh_after_tick:
            raise ValueError("A fresh observation is required after gripper actuation")
        if not 0 <= image_age_s <= 0.05 or not math.isfinite(now_tick) or now_tick < observation_tick:
            raise ValueError("Stale image or invalid observation/command time")
        actions = np.asarray(actions, dtype=np.float64)
        if actions.shape != (63, 4) or not np.isfinite(actions).all():
            raise ValueError("Expected 63 finite absolute XYZ/jaw references")
        # Labels are next-reference values: row k is due at observation_tick + k + 1.
        tick = math.ceil(now_tick)
        elapsed = tick - observation_tick
        if elapsed + PREFIX > len(actions):
            raise ValueError("Prediction is expired; obtain a fresh observation")
        future = actions[elapsed:]
        jaw_open = bool(future[0, 3] >= 0.5)
        if jaw_open != self.jaw_open:
            ticks = 30 if jaw_open else 42
            if len(future) < ticks or np.any((future[:ticks, 3] >= 0.5) != jaw_open):
                raise ValueError("Jaw intent changes before the commissioned dwell can finish")
            if np.linalg.norm(future[:ticks, :3] - self.position, axis=-1).max() > 0.0001:
                raise ValueError("Cartesian motion requested during gripper actuation")
            self.jaw_open = jaw_open
            self.velocity, self.acceleration = np.zeros(3), np.zeros(3)
            self.available_tick = self.fresh_after_tick = tick + ticks
            return Command("gripper", tick, ticks, jaw_open=jaw_open)
        moving = np.flatnonzero(np.linalg.norm(future[:, :3] - self.position, axis=-1) > 2e-8)
        if len(moving) == 0 or moving[0] > 0:
            ticks = min(PREFIX, int(moving[0])) if len(moving) else PREFIX
            self.velocity, self.acceleration = np.zeros(3), np.zeros(3)
            self.available_tick = tick + ticks
            return Command("hold", tick, ticks)
        if np.any((future[:PREFIX, 3] >= 0.5) != self.jaw_open):
            raise ValueError("A Cartesian prefix crosses a gripper event; replan at the existing command boundary")
        command = fit_cartesian(future[:PREFIX, :3], self.position, self.velocity, self.acceleration, tick)
        self.position = future[PREFIX - 1, :3].copy()
        self.velocity = command.sample(np.array([1.0]), 1)[0]
        self.acceleration = command.sample(np.array([1.0]), 2)[0]
        self.available_tick = tick + PREFIX
        return command
