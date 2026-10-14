"""Scripted board reset between trials: the arm puts the rings back on the start tower, with no policy involved.

After a trial the board tracker knows where every ring is. ``reset_moves`` plans the shortest legal sequence of
moves from that board back to the start tower (the same breadth-first search the progress metric uses), and
``execute_reset`` has the arm perform each move as scripted pick-and-place from the recorded peg hovers: descend
from the 191 mm hover to the ring's grasp level, close, lift back to the hover, travel to the target peg at hover
height, descend to the 152 mm release height, open, lift. Every leg is a rest-to-rest quintic within the motion
limits, validated against the workspace like a policy command, and the arrival of every leg and the stroke after
every close are checked; any surprise stops the reset and hands the board back to the operator.
"""

import dataclasses
import logging
import math
import time

import numpy as np

from examples.hanoi.deployment.cosmos_client import START_POSES
from examples.hanoi.deployment.dense_execution import TRACK_LIMITS
from examples.hanoi.deployment.dense_execution import quintic_segment
from examples.hanoi.deployment.dense_execution import rest_to_rest_ticks
from examples.hanoi.deployment.execution import Command
from examples.hanoi.deployment.execution import GRIPPER_CLOSE_S
from examples.hanoi.deployment.execution import GRIPPER_CLOSE_SETTLE_S
from examples.hanoi.deployment.execution import RATE_HZ
from examples.hanoi.deployment.hardware import check_grasp
from examples.hanoi.deployment.hardware import validate_trajectory
from examples.hanoi.deployment.progress import LEVELS_MM
from examples.hanoi.deployment.progress import board_string
from examples.hanoi.deployment.progress import path_moves
from examples.hanoi.deployment.progress import shortest_path

PEG_HOVER_XYZ = {peg: np.array(START_POSES[f"above_peg_{peg.lower()}"][0]) for peg in "ABC"}
HOVER_Z = 0.191  # the recorded carry height
RELEASE_Z = 0.152  # where the recordings open the jaw, whatever the stack height
ARRIVAL_TOLERANCE_M = 0.008  # the tracking watchdog's limit
SETTLE_S = 0.15


def reset_moves(final_board: dict, start_board: str) -> list | None:
    """(ring, source, target) moves from ``final_board`` back to the start tower; None if the board is not a legal Hanoi state."""
    path = shortest_path(final_board, start_board)
    return None if path is None else path_moves(path)


def grasp_z(rings_on_peg: int) -> float:
    """Tool height to grasp the top ring of a peg holding ``rings_on_peg`` rings."""
    if not 1 <= rings_on_peg <= len(LEVELS_MM):
        raise ValueError(f"No ring to grasp on a peg holding {rings_on_peg}")
    return float(LEVELS_MM[rings_on_peg - 1]) / 1000


@dataclasses.dataclass
class ScriptedMover:
    """Rest-to-rest legs and gripper commands on an enabled arm, each one checked before the next."""

    arm: object
    xyz_min: np.ndarray
    xyz_max: np.ndarray
    limits: tuple = TRACK_LIMITS
    log: object = None
    legs: list = dataclasses.field(default_factory=list)

    def _note(self, event, **fields):
        if self.log is not None:
            self.log(event, **fields)
        self.legs.append({"event": event, **fields})

    def position(self) -> np.ndarray:
        return np.asarray(self.arm.read(require_orientation=True)[:3], dtype=float)

    def leg(self, target) -> None:
        """One rest-to-rest quintic from the measured position to ``target``, dispatched and waited out."""
        start = self.position()
        target = np.asarray(target, dtype=float)
        ticks = max(rest_to_rest_ticks(float(np.linalg.norm(target - start))), 3)
        command = quintic_segment(start, np.zeros(3), np.zeros(3), target, np.zeros(3), np.zeros(3), ticks, 0)
        validate_trajectory(command, self.xyz_min, self.xyz_max, self.limits)
        self.arm.dispatch(command, self.xyz_min, self.xyz_max, self.limits)
        time.sleep(ticks / RATE_HZ + SETTLE_S)
        arrived = self.position()
        error = float(np.linalg.norm(arrived - target))
        self._note("reset_leg", target_xyz_m=target.tolist(), arrived_xyz_m=arrived.tolist(), error_mm=error * 1000, duration_s=ticks / RATE_HZ)
        if error > ARRIVAL_TOLERANCE_M:
            raise RuntimeError(f"Reset leg ended {error * 1000:.1f} mm from its target (limit {ARRIVAL_TOLERANCE_M * 1000:.0f} mm)")

    def gripper(self, jaw_open: bool) -> float:
        ticks = 30 if jaw_open else math.ceil((GRIPPER_CLOSE_S + GRIPPER_CLOSE_SETTLE_S) * RATE_HZ)
        self.arm.dispatch(Command("gripper", 0, ticks, jaw_open=jaw_open), self.xyz_min, self.xyz_max, self.limits)
        time.sleep(ticks / RATE_HZ + SETTLE_S)
        stroke = float(self.arm.read(require_orientation=False)[6])
        self._note("reset_gripper", jaw_open=jaw_open, stroke_m=stroke)
        if not jaw_open:
            check_grasp(stroke)  # closed on nothing: stop rather than carry air
        return stroke

    def move_ring(self, source: str, target: str, rings_on_source: int) -> None:
        """Pick the top ring of ``source`` and place it on ``target``, hover to hover."""
        src, dst = PEG_HOVER_XYZ[source], PEG_HOVER_XYZ[target]
        self.leg([src[0], src[1], HOVER_Z])
        self.leg([src[0], src[1], grasp_z(rings_on_source)])
        self.gripper(jaw_open=False)
        self.leg([src[0], src[1], HOVER_Z])
        self.leg([dst[0], dst[1], HOVER_Z])
        self.leg([dst[0], dst[1], RELEASE_Z])
        self.gripper(jaw_open=True)
        self.leg([dst[0], dst[1], HOVER_Z])


def execute_reset(arm, bounds, final_board: dict, start_board: str, log=None) -> dict:
    """Put the rings back on the start tower from ``final_board``; the arm must be initialised and enabled at a hover.

    Returns a report with the moves made; raises if a leg misses or a grasp closes on nothing, with the board as
    far as it got in ``report["board"]`` for the operator.
    """
    moves = reset_moves(final_board, start_board)
    if moves is None:
        raise ValueError(f"Board {final_board} is not a legal Hanoi state; reset it by hand")
    stacks = {p: list(v) for p, v in final_board.items()}
    mover = ScriptedMover(arm, np.asarray(bounds[0], float), np.asarray(bounds[1], float), log=log)
    report = {"from_board": board_string(final_board), "to_board": start_board, "planned_moves": len(moves), "moves": [], "board": None}
    try:
        for ring, source, target in moves:
            logging.info("Reset: ring %d %s->%s", ring, source, target)
            mover.move_ring(source, target, len(stacks[source]))
            stacks[target].append(stacks[source].pop())
            report["moves"].append({"ring": ring, "from": source, "to": target})
    finally:
        report["board"] = {p: list(v) for p, v in stacks.items()}
        report["legs"] = mover.legs
    return report
