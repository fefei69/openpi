"""Executor for the dense contract five: 10 Hz reference poses tracked in short segments.

A chunk holds 30 absolute reference poses, one every three 30 Hz ticks after the observation
tick. At each boundary the executor takes the rows not yet due, executes the next three (0.3 s)
as one quintic segment that starts from the commanded state and ends at the third row with the
velocity implied by the neighbouring rows, and hands the rest back for the next re-plan. A jaw
change is executed at rest at its row's pose, followed by the commissioned dwell. Segments whose
derivatives exceed the motion limits are stretched in time (up to ``max_stretch``) before being
refused. No hardware here; ``TrossenArm.dispatch`` sends the segment's endpoint, duration and
feedforward end derivatives to the driver.
"""

import dataclasses
import math

import numpy as np

from examples.hanoi.deployment.execution import GRIPPER_CLOSE_S
from examples.hanoi.deployment.execution import GRIPPER_CLOSE_SETTLE_S
from examples.hanoi.deployment.execution import LIMITS
from examples.hanoi.deployment.execution import RATE_HZ
from examples.hanoi.deployment.execution import Command

HORIZON = 30
ROW_TICKS = 3  # 10 Hz rows on the 30 Hz control clock
PREFIX = 3  # the contract's execution prefix: the shortest re-plan interval the model was validated for
# Rows executed per segment. The model's state input carries no velocity, so it predicts the
# demonstration's cruise speed whatever the arm is doing; with 3-row (0.3 s) segments any slowdown
# made the next rows unreachable within the limits and the executor braked into 0.1 s stop-starts
# (live run 1: 90 brakes in 134 segments). Re-planning the same chunks with 9-row segments gave none.
EXECUTED_ROWS = 9
MAX_IMAGE_AGE_S = 0.05
_ENDPOINT_INVERSE = np.linalg.inv(np.array([[1, 1, 1], [3, 4, 5], [6, 12, 20]], dtype=float))
_SAMPLES = np.linspace(0, 1, 121)
# The recorded legs peak exactly at the limits; a quintic between two at-limit boundary states can
# overshoot them by a hair. Same tolerance as hardware.validate_trajectory.
LIMIT_TOLERANCE = 1.02
# Tracking segments use the recorder's limits with a margin: the recorded legs peak at exactly the
# velocity and acceleration limits (0.145 m/s, 0.23 m/s^2), and sub-segments fitted around those
# peaks overshoot them by up to about 10%; the jerk limit of 1.3 m/s^3 was chosen for whole 2 s
# legs, and stitching 0.3 s quintics whose boundary derivatives come from rows with 0.7 mm noise
# needs about 20 m/s^3 (jerk scales with mismatch / duration^3). At 0.3 m/s^2 the motion is still
# gentle (0.03 g); the driver smooths its own profile. Slowing a segment down cannot fix an
# acceleration overshoot when the arm is already faster than the target needs.
TRACK_LIMITS = (0.16, 0.30, 20.0)


def quintic_segment(p0, v0, a0, p1, v1, a1, ticks: int, start_tick: int) -> Command:
    """The unique quintic through the given boundary state and derivatives over ``ticks``."""
    duration = ticks / RATE_HZ
    coefficients = np.zeros((6, 3))
    coefficients[0] = p0
    coefficients[1] = v0 * duration
    coefficients[2] = a0 * duration**2 / 2
    coefficients[3:] = _ENDPOINT_INVERSE @ np.stack(
        [
            p1 - coefficients[:3].sum(axis=0),
            duration * v1 - coefficients[1] - 2 * coefficients[2],
            duration**2 * a1 - 2 * coefficients[2],
        ]
    )
    return Command("cartesian", start_tick, ticks, coefficients)


def rest_to_rest_ticks(distance_m: float) -> int:
    """Ticks a rest-to-rest quintic needs over ``distance_m`` within the motion limits (the recorder's rule)."""
    duration = max(
        ROW_TICKS / RATE_HZ,
        1.875 * distance_m / LIMITS[0],
        math.sqrt((10 / math.sqrt(3)) * distance_m / LIMITS[1]),
        np.cbrt(60 * distance_m / LIMITS[2]),
    )
    return math.ceil(duration * RATE_HZ)


def end_derivatives(rows: np.ndarray, end: int, position: np.ndarray, estimator: str = "fit5"):
    """Velocity and acceleration at row ``end`` implied by the chunk rows, clipped to the limits.

    ``central``: central differences (noisy acceleration from 0.7 mm row noise).
    ``velocity_only``: central-difference velocity, zero acceleration.
    ``fit5``: least-squares quadratic through up to five rows around ``end`` (the executor's position
    stands in for the row before the first), evaluated at ``end``.
    """
    row_s = ROW_TICKS / RATE_HZ
    points = np.vstack([position[None, :], rows])  # index 0 is "row -1": where the arm is now
    centre = end + 1
    if estimator in ("central", "velocity_only"):
        lo, hi = centre - 1, min(centre + 1, len(points) - 1)
        velocity = (points[hi] - points[lo]) / ((hi - lo) * row_s)
        acceleration = (points[hi] - 2 * points[centre] + points[lo]) / row_s**2 if estimator == "central" else np.zeros(3)
    else:
        lo, hi = max(0, centre - 2), min(len(points) - 1, centre + 2)
        t = (np.arange(lo, hi + 1) - centre) * row_s
        coefficients = np.polynomial.polynomial.polyfit(t, points[lo:hi + 1], min(2, hi - lo))
        velocity = coefficients[1] if len(coefficients) > 1 else np.zeros(3)
        acceleration = 2 * coefficients[2] if len(coefficients) > 2 else np.zeros(3)
    speed = np.linalg.norm(velocity)
    if speed > LIMITS[0]:
        velocity = velocity * (LIMITS[0] / speed)
    magnitude = np.linalg.norm(acceleration)
    if magnitude > LIMITS[1]:
        acceleration = acceleration * (LIMITS[1] / magnitude)
    return velocity, acceleration


def carry_violation(command: Command, *, min_z_m: float, lateral_m: float = 0.01) -> float | None:
    """Lowest height at which a segment travels sideways by more than ``lateral_m``, if below ``min_z_m``.

    The recording carries every ring at the 191 mm hover and only descends above the target column;
    a sideways move with the jaw closed below the release height would drag the ring across the pegs.
    This is a safety check on the model's plan, not a correction of it: the client stops the run.
    """
    if command.kind != "cartesian" or min_z_m <= 0:
        return None
    points = command.sample(_SAMPLES)
    lateral = np.linalg.norm(points[:, :2] - points[0, :2], axis=1)
    low = (lateral > lateral_m) & (points[:, 2] < min_z_m)
    return float(points[low, 2].min()) if low.any() else None


def within_limits(command: Command, limits=TRACK_LIMITS) -> bool:
    return all(
        np.linalg.norm(command.sample(_SAMPLES, derivative), axis=-1).max() <= limit * LIMIT_TOLERANCE
        for derivative, limit in enumerate(limits, start=1)
    )


@dataclasses.dataclass
class DenseExecutor:
    position: np.ndarray
    velocity: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(3))
    acceleration: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(3))
    jaw_open: bool = True
    available_tick: int = 0
    fresh_after_tick: int = 0
    task: str = ""
    pending_jaw_open: bool | None = None
    # Chunk length the server announces (30 or 16 rows at 10 Hz for the two trained variants).
    horizon: int = HORIZON
    # Rows executed per segment before re-planning (0.1 s each).
    prefix: int = EXECUTED_ROWS
    limits: tuple = TRACK_LIMITS
    max_stretch: float = 6.0
    stop_tolerance_m: float = 0.0005
    # How the segment's end velocity and acceleration are read off the chunk rows (see end_derivatives).
    estimator: str = "fit5"
    last_stretch: float = 1.0
    # Set when the references could not be followed within the limits and the executor chose to
    # brake to rest instead; the next chunk is then predicted from a stopped arm.
    last_braked: bool = False

    def plan(self, actions: np.ndarray, *, observation_tick: int, now_tick: float, image_age_s: float, task: str) -> Command:
        if self.task and task != self.task:
            raise ValueError("Reset the executor and queued actions before changing tasks")
        if not math.isfinite(now_tick) or now_tick < observation_tick:
            raise ValueError("Invalid observation/command time")
        if now_tick < self.available_tick:
            return Command("wait", math.ceil(now_tick), self.available_tick - math.ceil(now_tick))
        tick = math.ceil(now_tick)
        if self.pending_jaw_open is not None:
            return self._gripper_command(jaw_open=self.pending_jaw_open, tick=tick)
        if observation_tick < self.fresh_after_tick:
            raise ValueError("A fresh observation is required after gripper actuation")
        if not 0 <= image_age_s <= MAX_IMAGE_AGE_S:
            raise ValueError("Stale image")
        actions = np.asarray(actions, dtype=float)
        if actions.shape != (self.horizon, 4) or not np.isfinite(actions).all():
            raise ValueError(f"Expected {self.horizon} finite absolute XYZ/jaw references")
        # Row k is due at observation_tick + ROW_TICKS * (k + 1); skip the rows already due.
        elapsed_rows = (tick - observation_tick) // ROW_TICKS
        if elapsed_rows >= self.horizon:
            raise ValueError("Prediction is expired; obtain a fresh observation")
        self.task = task
        future = actions[elapsed_rows:]
        jaw_open = bool(future[0, 3] >= 0.5)
        if jaw_open != self.jaw_open:
            target = future[0, :3]
            distance = float(np.linalg.norm(target - self.position))
            if distance > 0.001 or np.linalg.norm(self.velocity) > 0.001:
                # Align at rest on the row's pose before the grip, at the recorder's rest-to-rest pace.
                try:
                    command = self._segment(target, np.zeros(3), np.zeros(3), tick, self._stop_ticks(distance))
                except ValueError:
                    return self._brake(tick)
                self.pending_jaw_open = jaw_open
                return command
            return self._gripper_command(jaw_open=jaw_open, tick=tick)
        prefix = future[: self.prefix]
        changes = np.flatnonzero((prefix[:, 3] >= 0.5) != self.jaw_open)
        stopping = len(changes) > 0
        if stopping:
            prefix = prefix[: changes[0]]
        end = len(prefix) - 1
        target = prefix[end, :3]
        still = np.linalg.norm(prefix[:, :3] - self.position, axis=-1).max() <= self.stop_tolerance_m
        if still and np.linalg.norm(self.velocity) <= 0.001:
            self.velocity, self.acceleration = np.zeros(3), np.zeros(3)
            self.available_tick = tick + len(prefix) * ROW_TICKS
            return Command("hold", tick, len(prefix) * ROW_TICKS)
        following = end + 1
        row_s = ROW_TICKS / RATE_HZ
        if stopping or following >= len(future) or np.linalg.norm(future[following, :3] - target) <= self.stop_tolerance_m:
            # Coming to rest: a jaw event follows, the chunk ends, or the references stay put.
            nominal = max(len(prefix) * ROW_TICKS, self._stop_ticks(float(np.linalg.norm(target - self.position))))
            try:
                return self._segment(target, np.zeros(3), np.zeros(3), tick, nominal)
            except ValueError:
                return self._brake(tick)
        end_velocity, end_acceleration = end_derivatives(future[:, :3], end, self.position, self.estimator)
        try:
            return self._segment(target, end_velocity, end_acceleration, tick, self._due_ticks(observation_tick, elapsed_rows + len(prefix), tick))
        except ValueError:
            return self._brake(tick)

    def _brake(self, tick: int) -> Command:
        """Shortest smooth stop along the current velocity; a fresh chunk from a stopped arm is always feasible."""
        speed = float(np.linalg.norm(self.velocity))
        nominal = max(ROW_TICKS, math.ceil(speed / LIMITS[1] * RATE_HZ))
        stretch = 1.0
        while True:
            ticks = math.ceil(nominal * stretch)
            target = self.position + self.velocity * (ticks / RATE_HZ) / 2
            command = quintic_segment(self.position, self.velocity, self.acceleration, target, np.zeros(3), np.zeros(3), ticks, tick)
            if within_limits(command, self.limits) or stretch > 10:
                break
            stretch *= 1.25
        self.last_stretch, self.last_braked = stretch, True
        self.position, self.velocity, self.acceleration = target, np.zeros(3), np.zeros(3)
        self.available_tick = tick + ticks
        return command

    @staticmethod
    def _due_ticks(observation_tick: int, rows_done: int, tick: int) -> int:
        """Ticks until the last executed row is due, so the arm keeps the reference timeline."""
        return max(ROW_TICKS, observation_tick + ROW_TICKS * rows_done - tick)

    def _stop_ticks(self, distance_m: float) -> int:
        braking = 2 * float(np.linalg.norm(self.velocity)) / LIMITS[1]
        return max(rest_to_rest_ticks(distance_m), math.ceil(braking * RATE_HZ))

    def _segment(self, target, end_velocity, end_acceleration, tick: int, nominal_ticks: int,
                 max_stretch: float | None = None) -> Command:
        max_stretch = self.max_stretch if max_stretch is None else max_stretch
        stretch = 1.0
        while True:
            ticks = math.ceil(nominal_ticks * stretch)
            command = quintic_segment(
                self.position, self.velocity, self.acceleration, target,
                end_velocity / stretch, end_acceleration / stretch**2, ticks, tick,
            )
            if within_limits(command, self.limits):
                break
            stretch *= 1.25
            if stretch > max_stretch:
                raise ValueError(
                    f"Predicted references exceed the motion limits even at {max_stretch:g}x duration"
                )
        self.last_stretch = stretch
        self.last_braked = False
        self.position = np.asarray(target, dtype=float).copy()
        self.velocity = command.sample(np.array([1.0]), 1)[0]
        self.acceleration = command.sample(np.array([1.0]), 2)[0]
        self.available_tick = tick + ticks
        return command

    def _gripper_command(self, *, jaw_open: bool, tick: int) -> Command:
        ticks = 30 if jaw_open else math.ceil((GRIPPER_CLOSE_S + GRIPPER_CLOSE_SETTLE_S) * RATE_HZ)
        self.pending_jaw_open = None
        self.jaw_open = jaw_open
        self.velocity, self.acceleration = np.zeros(3), np.zeros(3)
        self.available_tick = self.fresh_after_tick = tick + ticks
        return Command("gripper", tick, ticks, jaw_open=jaw_open)
