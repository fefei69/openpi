"""Command builders for Hanoi.

PolicyExecutor turns learned targets into timed rest-to-rest moves.
ReferenceExecutor retains strict reconstruction for offline teacher validation.
Both produce commands without touching hardware.
"""

import dataclasses
import math

import numpy as np

RATE_HZ = 30
PREFIX = 9
LIMITS = (0.145, 0.23, 1.3)
# Slow arm travel without delaying stop commands or gripper release.
ARM_MOTION_TIME_SCALE = 2
# Live close ramps the gripping effort more gently than the recorded 1.2 s close.
GRIPPER_CLOSE_S = 2.4
GRIPPER_CLOSE_SETTLE_S = 0.2
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


def move_to_target(position: np.ndarray, target: np.ndarray, start_tick: int) -> Command:
    """A bounded rest-to-rest move, stretched to half speed for arm travel."""
    delta = np.asarray(target, dtype=float) - position
    distance = float(np.linalg.norm(delta))
    # Exact peak derivatives of s(u) = 10*u^3 - 15*u^4 + 6*u^5.
    duration = max(
        PREFIX / RATE_HZ,
        1.875 * distance / LIMITS[0],
        math.sqrt((10 / math.sqrt(3)) * distance / LIMITS[1]),
        np.cbrt(60 * distance / LIMITS[2]),
    )
    ticks = math.ceil(duration * RATE_HZ) * ARM_MOTION_TIME_SCALE
    coefficients = np.zeros((6, 3))
    coefficients[0] = position
    coefficients[3:] = np.array([10, -15, 6])[:, None] * delta
    return Command("cartesian", start_tick, ticks, coefficients)


@dataclasses.dataclass
class PolicyExecutor(ReferenceExecutor):
    """Execute learned targets without requiring recorder-exact intermediate samples.

    Each Cartesian move ends at rest, so the arm can wait for the next prediction
    or operate its gripper. The strict ReferenceExecutor remains for offline
    reconstruction of recorded teacher trajectories.
    """

    # A jaw transition commits its paired XYZ first, then actuates at that point.
    pending_jaw_open: bool | None = None

    def plan(
        self, actions: np.ndarray, *, observation_tick: int, now_tick: float, image_age_s: float, task: str
    ) -> Command:
        if self.task and task != self.task:
            raise ValueError("Reset the executor and queued actions before changing tasks")
        if not math.isfinite(now_tick) or now_tick < observation_tick:
            raise ValueError("Invalid observation/command time")
        if now_tick < self.available_tick:
            return Command("wait", math.ceil(now_tick), self.available_tick - math.ceil(now_tick))
        tick = math.ceil(now_tick)
        if self.pending_jaw_open is not None:
            # Finish the already committed move-and-grip operation even if its
            # source prediction has expired while the slow approach completed.
            return self._gripper_command(jaw_open=self.pending_jaw_open, tick=tick)
        if observation_tick < self.fresh_after_tick:
            raise ValueError("A fresh observation is required after gripper actuation")
        if not 0 <= image_age_s <= 0.05:
            raise ValueError("Stale image")
        actions = np.asarray(actions, dtype=float)
        if actions.shape != (63, 4) or not np.isfinite(actions).all():
            raise ValueError("Expected 63 finite absolute XYZ/jaw references")
        elapsed = tick - observation_tick
        if elapsed >= len(actions):
            raise ValueError("Prediction is expired; obtain a fresh observation")
        self.task = task
        future = actions[elapsed:]
        jaw_open = bool(future[0, 3] >= 0.5)
        if jaw_open != self.jaw_open:
            target = future[0, :3]
            if np.linalg.norm(target - self.position) > 1e-6:
                command = move_to_target(self.position, target, tick)
                self.position = target.copy()
                self.velocity, self.acceleration = np.zeros(3), np.zeros(3)
                self.available_tick = tick + command.ticks
                self.pending_jaw_open = jaw_open
                return command
            return self._gripper_command(jaw_open=jaw_open, tick=tick)
        prefix = future[:PREFIX]
        changes = np.flatnonzero((prefix[:, 3] >= 0.5) != self.jaw_open)
        if len(changes):
            prefix = prefix[: changes[0]]
        target = prefix[-1, :3]
        if np.linalg.norm(target - self.position) <= 1e-6:
            command = Command("hold", tick, len(prefix))
        else:
            command = move_to_target(self.position, target, tick)
            self.position = target.copy()
        self.velocity, self.acceleration = np.zeros(3), np.zeros(3)
        self.available_tick = tick + command.ticks
        return command

    def _gripper_command(self, *, jaw_open: bool, tick: int) -> Command:
        # The client enforces this dwell, then obtains a fresh observation.
        ticks = 30 if jaw_open else math.ceil((GRIPPER_CLOSE_S + GRIPPER_CLOSE_SETTLE_S) * RATE_HZ)
        self.pending_jaw_open = None
        self.jaw_open = jaw_open
        self.velocity, self.acceleration = np.zeros(3), np.zeros(3)
        self.available_tick = self.fresh_after_tick = tick + ticks
        return Command("gripper", tick, ticks, jaw_open=jaw_open)
