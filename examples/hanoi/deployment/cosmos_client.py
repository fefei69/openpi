"""Waypoint_v4 deployment client: observe, infer, move one waypoint, re-observe.

Companion to client.py (pi0.5 roundtrip). Same robot environment, camera, arm, workspace
bounds, watchdogs and recovery; a different policy contract. It drives either waypoint_v4
server: ``cosmos_policy.experiments.robot.hanoi.serve_waypoint`` in the Cosmos checkout (port
8001, identity key ``cosmos_hanoi``) or ``examples.hanoi.deployment.serve_waypoint`` here, the
pi0.5 checkpoint trained on the same dataset (port 8000, key ``hanoi_waypoint``). Both return
eight absolute destinations; this client commits only the
first, waits for the move and any gripper dwell to finish, then takes a fresh observation.
There is no 30 Hz action timeline. Feedback, tracking and missed-grasp checks still run at
30 Hz while a command executes.

Modes:
  replay  stream rows of a single-episode Cosmos extract (no ROS, no arm); exercises the
          transport, the contract check and command validation against the workspace.
  shadow  live camera and arm feedback, predictions logged, no motion.
  live    initialize at the recorded v4 episode start, then policy control for --duration-s.
"""

import dataclasses
import json
import logging
import math
from pathlib import Path
import signal
import time
from typing import Literal

import numpy as np
from openpi_client import hanoi
from openpi_client import msgpack_numpy
import tyro
from websockets.sync.client import connect

from examples.hanoi.deployment.async_inference import InferenceRecorder
from examples.hanoi.deployment.async_inference import Observation
from examples.hanoi.deployment.bag import start_bag
from examples.hanoi.deployment.execution import ARM_MOTION_TIME_SCALE
from examples.hanoi.deployment.execution import GRIPPER_CLOSE_S
from examples.hanoi.deployment.execution import GRIPPER_CLOSE_SETTLE_S
from examples.hanoi.deployment.execution import RATE_HZ
from examples.hanoi.deployment.execution import Command
from examples.hanoi.deployment.execution import move_to_target
from examples.hanoi.deployment.hardware import INITIAL_JAW_M
from examples.hanoi.deployment.hardware import MIN_GRASP_STROKE_M
from examples.hanoi.deployment.hardware import MissedGraspError
from examples.hanoi.deployment.hardware import RosCamera
from examples.hanoi.deployment.hardware import TrossenArm
from examples.hanoi.deployment.hardware import check_grasp
from examples.hanoi.deployment.hardware import validate_trajectory

PROMPT = hanoi.PROMPTS["aaaa_to_cccc"]
MAX_IMAGE_AGE_S = hanoi.CONTRACT["max_image_age_s"]
# Selected export of run hanoi_cosmos_waypoint_v4_20260917 (selection.json: iter_000008000.pt).
SELECTED_EXPORT_SHA256 = "64208a3da1806524e073052a37f971336fa40d8df466b14ce3a1c089bad20b1f"
# pi0.5 waypoint_v4 export 29999 (hanoi_waypoint_full_20260917): SHA-256 of its export.json manifest.
SELECTED_PI05_EXPORT_SHA256 = "61b39665ee49aee3c6ad7749559c7f13cf671c34fcb4aba2ca9c691d34d2cdf4"
SELECTED_EXPORTS = {"cosmos": SELECTED_EXPORT_SHA256, "pi05": SELECTED_PI05_EXPORT_SHA256}
# Metadata key each server publishes its identity under, and the policy family it means.
IDENTITY_KEYS = {"cosmos_hanoi": "cosmos", "hanoi_waypoint": "pi05"}
# Fields of the waypoint_v4 deployment contract this client relies on (metadata.json["deployment"]).
EXPECTED_CONTRACT = {
    "version": 4,
    "robot": "trossen_wxai_single",
    "action_horizon": 8,
    "execution_prefix": 1,
    "state": [f"joint_{i}_rad" for i in range(6)] + ["jaw_stroke_m"],
    "actions": ["destination_x_m", "destination_y_m", "destination_z_m", "jaw_open_after_arrival"],
    "frame": "commissioned_base_tool_frame",
    "orientation_rpy_rad": [0.0, math.pi / 4, 0.0],
    "rgb_topic": hanoi.CONTRACT["rgb_topic"],
    "rgb_crop_xywh": [151, 90, 360, 360],
    "max_image_age_s": MAX_IMAGE_AGE_S,
    "jaw_open_stroke_m": 0.034,
    "jaw_open_duration_s": 1.0,
    "jaw_close_effort_n": -20.0,
    "jaw_close_duration_s": 1.2,
    "jaw_close_settle_s": 0.2,
    "joint_order": "trossen_arm_driver_arm_indices_0_to_5",
    "internal_xyz_encoding": "relative_to_measured_observation_xyz",
}
EXECUTION_ADAPTER = "{family}_waypoint_v4_commit_{commit_count}_rest_to_rest"


def execution_adapter(config, family: str = "cosmos") -> str:
    name = EXECUTION_ADAPTER.format(family=family, commit_count=config.commit_count)
    return name + ("_snapped_ablation" if config.snap_to_recorded_destinations else "")
# Start pose of every waypoint_v4 training episode: mean of the 40 episode starts in train.npz
# (starts spread at most 0.45 mm and 0.0008 rad), behind peg B with the jaw open 0.034 m. The
# pi0.5 client's rod-A hover (492.6, -56.2, 191.1) is a carry pose in this recording: 238 training
# windows begin there and none of them precedes a grasp, while the grasp hover over A is at
# x = 496.1 mm. Starting there made every grasp 3.5 mm short in x (README, "Start pose").
V4_START_XYZ_M = (0.413924, 0.015821, 0.191204)
V4_START_JOINTS_RAD = (0.05512, 1.43154, 1.21085, -0.5648, 0.03789, 0.0403)
# The recorded grasp hover over peg A: where every episode arrives after its first leg and from which the
# first grasp descends (mean of the 40 first grasp windows; spread under 0.7 mm and 0.0014 rad). Not the
# carry hover at (492.3, -56.3), which the recording only releases from.
HOVER_A_XYZ_M = (0.496236, -0.057392, 0.191068)
HOVER_A_JOINTS_RAD = (-0.15583, 1.76183, 1.55087, -0.58042, -0.11006, -0.11089)
START_POSES = {
    "episode_start": (V4_START_XYZ_M, V4_START_JOINTS_RAD),  # behind peg B, how every training episode begins
    "above_peg_a": (HOVER_A_XYZ_M, HOVER_A_JOINTS_RAD),  # the grasp hover over A, for comparisons that start there
}
# Phase bands of the recorded destinations: grasps below 100 mm, releases 120-170 mm, hovers above.
GRASP_MAX_Z = 0.10
RELEASE_MIN_Z = 0.12
HOVER_MIN_Z = 0.17


class CommandRejected(ValueError):
    """A predicted destination failed workspace or motion validation; the arm did not move."""


@dataclasses.dataclass
class Config:
    mode: Literal["replay", "shadow", "live"] = "live"
    server: str = "ws://127.0.0.1:8001"
    duration_s: float = 60.0
    # Single-episode extract written by cosmos-policy's examples/hanoi/extract_episode.py.
    episode: Path = Path("../cosmos-policy/data/hanoi_cosmos/exports_local/hanoi_episode_040.h5")
    # Replay: explicit rows, else every --replay-stride rows from --start-row.
    replay_rows: tuple[int, ...] = ()
    replay_stride: int = 300
    start_row: int = 0
    output: Path = Path("data/hanoi/deployment")
    robot_ip: str = "192.168.1.3"
    camera_topic: str = hanoi.CONTRACT["rgb_topic"]
    # Bounds derived from this policy's own training labels (see cosmos_workspace.json).
    workspace: Path | None = dataclasses.field(default_factory=lambda: Path(__file__).with_name("cosmos_workspace.json"))
    initial_jaw: Literal["open", "closed"] = "open"
    # Where the arm starts before the first prediction: the recorded v4 episode start, not the
    # pi0.5 rod-A hover (a carry pose the model never grasps from).
    start: Literal["episode_start", "above_peg_a"] = "episode_start"
    # The six joint angles are the model's state input, so the aligned start is also checked
    # against the recorded start joints; 0.05 rad is about 3 degrees.
    max_start_joint_error_rad: float = 0.05
    # Waypoints of each eight-waypoint chunk executed before the next observation. 1 is the
    # contract's execution prefix. 3 (hover, gripper action, lift) was tried in runs 10 to 12 and
    # drifted faster: the open-loop waypoints carry the offset at prediction time unchanged, while
    # a re-plan corrects part of it at most steps. Kept as an option for ablations.
    commit_count: int = 1
    min_grasp_stroke_m: float = MIN_GRASP_STROKE_M
    # "selected" means the selected export of whichever family the server announces (Cosmos
    # iter_000008000.pt or the pi0.5 export 29999); an empty string skips the identity check
    # (replay/shadow experiments only); anything else is compared verbatim.
    expected_export_sha256: str = "selected"
    inference_timeout_s: float = 5.0
    warmup_timeout_s: float = 60.0
    # In-flight deviation from the quintic reference. A loaded lift lags the reference by
    # several millimetres (controller compliance with a ring on the peg), so this is looser
    # than the pi0.5 client's 3 mm, whose post-grasp moves were a few millimetres each.
    max_tracking_error_m: float = 0.008
    # Position error accepted after a move's duration plus the settle window.
    max_arrival_error_m: float = 0.003
    arrival_settle_s: float = 0.5
    # Bounded wait for a camera frame received after the last command completed.
    fresh_frame_timeout_s: float = 2.0
    # ABLATION ONLY, off by default: snap each committed destination onto the recorded grid the
    # server publishes (hovers by height, releases and grasps to the nearest recorded point).
    # This replaces the model's millimetre-level output with task knowledge, so it must never be
    # on when measuring the policy. It exists to separate "wrong point" from "imprecise point".
    snap_to_recorded_destinations: bool = False
    snap_max_mm: float = 6.0
    # A grasp height farther than this from the nearest ring level means the model is unsure
    # of the stack height; stop instead of guessing (levels are 10.2 mm apart).
    grasp_level_max_dz_mm: float = 4.0
    # When the duration ends: finish a placement in progress (a held ring is not released
    # mid-carry) for at most this long, then open the gripper and return to joint home.
    return_home_after_duration: bool = True
    finish_grace_s: float = 30.0
    # Record a ROS 2 bag of the full-frame camera stream for the whole run, from robot initialization
    # to the return home, under <run>/camera_bag (mcap). The camera publishes 640 x 480 rgb8 at 60 Hz:
    # 55 MB/s raw, about 24 MB/s with the default zstd_fast preset (7 GB for five minutes). Live and
    # shadow modes only; the recorder is a separate process and does not touch the control loop.
    record_bag: bool = False
    bag_topics: tuple[str, ...] = ()  # default: the camera image topic and its camera_info
    bag_storage_preset: Literal["none", "fastwrite", "zstd_fast", "zstd_small"] = "zstd_fast"


# ---- contract, transport ----


def check_contract(metadata, *, expected_export_sha256: str | None = None) -> dict:
    """Reject any server that is not a selected waypoint_v4 policy; returns its identity plus
    ``policy_family`` ("cosmos" or "pi05"). ``expected_export_sha256`` may be "selected"."""
    family = identity = None
    if isinstance(metadata, dict):
        for key, name in IDENTITY_KEYS.items():
            if isinstance(metadata.get(key), dict):
                family, identity = name, metadata[key]
                break
    if not isinstance(identity, dict) or not isinstance(identity.get("contract"), dict):
        raise ValueError("Server is not a Hanoi waypoint_v4 policy server")
    if expected_export_sha256 == "selected":
        expected_export_sha256 = SELECTED_EXPORTS[family]
    contract = identity["contract"]
    for key, expected in EXPECTED_CONTRACT.items():
        actual = contract.get(key)
        if key == "orientation_rpy_rad":
            same = isinstance(actual, (list, tuple)) and len(actual) == 3 and np.allclose(actual, expected)
        else:
            same = actual == expected
        if not same:
            raise ValueError(f"Contract mismatch for {key}: server {actual!r}, client {expected!r}")
    if identity.get("prompt") != PROMPT:
        raise ValueError("Server prompt is not the trained AAAA-to-CCCC instruction")
    if int(identity.get("commit_count", 0)) != 1:
        raise ValueError("Server does not commit exactly one destination per observation")
    if expected_export_sha256 and identity.get("export_sha256") != expected_export_sha256:
        raise ValueError(
            f"Server export {identity.get('export_sha256')!r} is not the selected checkpoint {expected_export_sha256}"
        )
    return {**identity, "policy_family": family}


class CosmosWebSocketPolicy:
    """Bounded-time OpenPI-protocol transport for the waypoint server."""

    def __init__(self, uri: str, *, timeout_s: float, warmup_timeout_s: float, expected_export_sha256: str | None):
        self.timeout_s, self.warmup_timeout_s, self.first = timeout_s, warmup_timeout_s, True
        self.ws = connect(uri, compression=None, max_size=16 * 1024 * 1024, open_timeout=5, close_timeout=0.2)
        try:
            self.metadata = msgpack_numpy.unpackb(self.ws.recv(timeout=5))
            self.identity = check_contract(self.metadata, expected_export_sha256=expected_export_sha256)
        except BaseException:
            self.ws.close()
            raise

    def infer(self, observation: dict) -> tuple[np.ndarray, dict, dict | None]:
        self.ws.send(msgpack_numpy.packb(observation))
        reply = self.ws.recv(timeout=self.warmup_timeout_s if self.first else self.timeout_s)
        self.first = False
        if isinstance(reply, str):
            raise RuntimeError(f"Policy server failed: {reply}")
        reply = msgpack_numpy.unpackb(reply)
        actions = np.asarray(reply["actions"], dtype=np.float64)
        if actions.shape != (8, 4) or not np.isfinite(actions).all():
            raise ValueError("Expected eight finite absolute XYZ/jaw destinations")
        if int(reply.get("commit_count", 1)) != 1:
            raise ValueError("Server changed the commit count")
        dream = None
        if "future_image" in reply:  # server started with --dream
            future = np.asarray(reply["future_image"])
            if future.shape != (224, 224, 3) or future.dtype != np.uint8:
                raise ValueError("Server dream must be a 224 x 224 x 3 uint8 image")
            value = float(reply.get("value", np.nan))
            if not np.isfinite(value):
                raise ValueError("Server dream lacks a finite value estimate")
            dream = {"future_image": future, "value": value}
        return actions, dict(reply.get("server_timing", {})), dream

    def close(self):
        self.ws.close()


def save_dream(output: Path, request_id: int, dream: dict | None) -> str | None:
    """Write the server's predicted future frame next to the request's input; None when absent."""
    if dream is None:
        return None
    from PIL import Image

    folder = output / "inference_dreams"
    folder.mkdir(exist_ok=True)
    name = f"inference_dreams/{request_id:06d}.png"
    Image.fromarray(dream["future_image"]).save(output / name)
    return name


# ---- planning ----


def stop_actions(status: str, *, live: bool, return_home_after_duration: bool) -> tuple[bool, bool]:
    """(open the gripper, return to joint home) for a terminal status.

    Failures hold in place so the scene can be inspected; a missed grasp and a
    completed duration, and Ctrl-C release and go home.
    """
    if not live:
        return False, False
    if status == "missed_grasp":
        return True, True
    if status == "operator_stop":
        return True, True  # Ctrl-C: open, then return to joint home like every other stop.
    if status == "duration_reached" and return_home_after_duration:
        return True, True
    if status == "rejected_command":
        return True, True  # Nothing was dispatched; the arm rests at its last verified arrival.
    if status == "failed":
        return True, True  # An unexpected exception: open and home rather than hold a ring mid-carry.
    if status == "task_solved":
        return True, True
    return False, False


def plan_waypoint(position: np.ndarray, jaw_open: bool, target: np.ndarray, *, tolerance_m: float = 1e-6) -> list:
    """Commands for the committed destination: a rest-to-rest move, then a jaw change, if each differs."""
    target = np.asarray(target, dtype=float)
    if target.shape != (4,) or not np.isfinite(target).all():
        raise ValueError("A destination is XYZ in metres plus a jaw intent")
    steps = []
    if np.linalg.norm(target[:3] - position) > tolerance_m:
        steps.append(("move", target[:3].copy()))
    intent = bool(target[3] >= 0.5)
    if intent != jaw_open:
        steps.append(("gripper", intent))
    return steps


def snap_destination(xyz, grid, *, max_snap_m: float, max_grasp_dz_m: float) -> tuple[np.ndarray, str, float]:
    """Map a predicted destination onto the recorded grid; returns (xyz, phase, distance) or raises ValueError.

    Hovers are transit points, so only their height is a task constant. Releases and grasps
    snap to the nearest recorded point of their phase. A grasp between two ring levels is refused.
    """
    xyz = np.asarray(xyz, dtype=float)
    grid = np.asarray(grid, dtype=float)
    if xyz.shape != (3,) or not np.isfinite(xyz).all():
        raise ValueError("A destination is XYZ in metres")
    z = xyz[2]
    if z >= HOVER_MIN_Z:
        hover_z = float(grid[:, 2].max())
        return np.array([xyz[0], xyz[1], hover_z]), "hover", abs(z - hover_z)
    if z < GRASP_MAX_Z:
        # Grasp: the peg column is chosen laterally, then the ring level by height, so a
        # lateral miss and an ambiguous stack height are reported as different faults.
        grasps = grid[grid[:, 2] < GRASP_MAX_Z]
        if not len(grasps):
            raise ValueError("The recorded grid has no grasp destinations")
        lateral = np.linalg.norm(grasps[:, :2] - xyz[:2], axis=1)
        j = int(np.argmin(lateral))
        if lateral[j] > max_snap_m:
            raise ValueError(
                f"Nearest recorded grasp column is {lateral[j] * 1000:.1f} mm away laterally (limit {max_snap_m * 1000:.1f} mm)"
            )
        levels = grasps[np.linalg.norm(grasps[:, :2] - grasps[j, :2], axis=1) < 1e-6]
        k = int(np.argmin(np.abs(levels[:, 2] - z)))
        dz = abs(levels[k, 2] - z)
        if dz > max_grasp_dz_m:
            raise ValueError(
                f"Grasp height {z * 1000:.1f} mm is {dz * 1000:.1f} mm from the nearest ring level "
                f"(limit {max_grasp_dz_m * 1000:.1f} mm); the stack height is ambiguous"
            )
        return levels[k].copy(), "grasp", float(np.linalg.norm(levels[k] - xyz))
    if z >= RELEASE_MIN_Z:
        releases = grid[(grid[:, 2] >= RELEASE_MIN_Z) & (grid[:, 2] < HOVER_MIN_Z)]
        if not len(releases):
            raise ValueError("The recorded grid has no release destinations")
        distances = np.linalg.norm(releases - xyz, axis=1)
        j = int(np.argmin(distances))
        if distances[j] > max_snap_m:
            raise ValueError(
                f"Nearest recorded release destination is {distances[j] * 1000:.1f} mm away (limit {max_snap_m * 1000:.1f} mm)"
            )
        return releases[j].copy(), "release", float(distances[j])
    raise ValueError(f"Destination height {z * 1000:.1f} mm lies between the recorded grasp and release heights")


def gripper_command(jaw_open: bool, tick: int) -> Command:
    if jaw_open:
        ticks = math.ceil(hanoi.CONTRACT["jaw_open_duration_s"] * RATE_HZ)
    else:
        ticks = math.ceil((GRIPPER_CLOSE_S + GRIPPER_CLOSE_SETTLE_S) * RATE_HZ)
    return Command("gripper", tick, ticks, jaw_open=jaw_open)


# ---- observations ----


def observation_data(image: np.ndarray, joints: np.ndarray, jaw_stroke_m: float, xyz: np.ndarray) -> dict:
    return {
        "observation/image": np.asarray(image, dtype=np.uint8),
        "observation/state": np.r_[np.asarray(joints, dtype=np.float32), np.float32(jaw_stroke_m)],
        "observation/cartesian_position": np.asarray(xyz, dtype=np.float32),
        "prompt": PROMPT,
    }


class ReplayEpisode:
    """Rows of a Cosmos single-episode extract as observations; no hardware."""

    def __init__(self, path: Path):
        import h5py

        self.h5 = h5py.File(path, "r")
        for name in ("pixels", "joint_positions", "proprio", "command_monotonic_ns", "image_receipt_monotonic_ns"):
            if name not in self.h5:
                raise ValueError(f"Extract lacks {name}: {path}")
        self.length = len(self.h5["pixels"])
        if self.h5["joint_positions"].shape != (self.length, 6) or self.h5["proprio"].shape[1] < 7:
            raise ValueError("Extract must have six joint angles and an eight-column proprio table")

    def observe(self, tick: int, row: int) -> tuple[Observation, np.ndarray]:
        row = min(max(row, 0), self.length - 1)
        proprio = np.asarray(self.h5["proprio"][row], dtype=float)
        joints = np.asarray(self.h5["joint_positions"][row], dtype=float)
        age = (int(self.h5["command_monotonic_ns"][row]) - int(self.h5["image_receipt_monotonic_ns"][row])) / 1e9
        captured = time.monotonic()
        data = observation_data(self.h5["pixels"][row], joints, proprio[6], proprio[:3])
        return Observation(tick, captured, captured - age, data), proprio[:7].astype(np.float32)

    def close(self):
        self.h5.close()


def start_joint_report(joints, expected=V4_START_JOINTS_RAD, *, max_error_rad: float) -> dict:
    """Compare the aligned start joints with the recorded episode start; XYZ alone can hide a
    different arm configuration, and joints are what the model observes."""
    joints = np.asarray(joints, dtype=float)
    expected = np.asarray(expected, dtype=float)
    if joints.shape != (6,) or not np.isfinite(joints).all():
        raise ValueError(f"Start joints must be six finite radians, got {joints.tolist()}")
    error = np.abs(joints - expected)
    report = {
        "measured_joints_rad": joints.tolist(),
        "recorded_joints_rad": expected.tolist(),
        "max_error_rad": float(error.max()),
        "max_error_joint": int(error.argmax()),
        "limit_rad": float(max_error_rad),
    }
    if error.max() > max_error_rad:
        raise ValueError(
            f"Start joints differ from the recorded v4 start by {error.max():.4f} rad at joint "
            f"{int(error.argmax())} (limit {max_error_rad:g}): measured {joints.round(4).tolist()}, "
            f"recorded {expected.round(4).tolist()}. The XYZ matched, so the arm reached the pose in a "
            "different configuration or the tool orientation differs from the recording."
        )
    return report


def observe_hardware(arm, camera, tick, *, require_orientation, jaw_closed, min_grasp_stroke_m):
    state, joints = arm.read_joints(require_orientation=require_orientation)
    if jaw_closed:
        check_grasp(float(state[6]), min_grasp_stroke_m)
    frame = camera.latest()
    captured = time.monotonic()
    return Observation(tick, captured, frame.received_at, observation_data(frame.rgb, joints, state[6], state[:3])), state


def fresh_observation(observe, *, after: float, timeout_s: float):
    """Wait for a frame received after ``after`` and younger than the contract age."""
    deadline = time.monotonic() + timeout_s
    while True:
        observation, state = observe()
        if 0 <= observation.image_age_s <= MAX_IMAGE_AGE_S and observation.image_received_at >= after:
            return observation, state
        if time.monotonic() >= deadline:
            raise RuntimeError("No fresh camera frame received after the last command completed")
        time.sleep(0.01)


# ---- main ----


def main(config: Config):
    if not 1 <= config.commit_count <= 8:
        raise ValueError("--commit-count must be between 1 and 8, the chunk length")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not np.isfinite(config.duration_s) or config.duration_s <= 0 or config.start_row < 0:
        raise ValueError("Duration must be finite and positive; start row must be nonnegative")
    if min(config.inference_timeout_s, config.warmup_timeout_s, config.max_tracking_error_m,
           config.max_arrival_error_m, config.arrival_settle_s, config.fresh_frame_timeout_s) <= 0:
        raise ValueError("Timeouts and tolerances must be positive")
    if not np.isfinite(config.min_grasp_stroke_m) or not 0 < config.min_grasp_stroke_m < INITIAL_JAW_M:
        raise ValueError("Minimum grasp stroke must be finite, positive, and below the open stroke")
    if config.replay_stride < 1:
        raise ValueError("Replay stride must be positive")
    if not np.isfinite(config.finish_grace_s) or config.finish_grace_s < 0:
        raise ValueError("Finish grace must be finite and nonnegative")
    if min(config.snap_max_mm, config.grasp_level_max_dz_mm) <= 0:
        raise ValueError("Snap tolerances must be positive")
    bounds = None
    if config.workspace:
        workspace = json.loads(config.workspace.read_text())
        bounds = np.asarray([workspace["xyz_min_m"], workspace["xyz_max_m"]], dtype=float)
        if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[0] >= bounds[1]):
            raise ValueError("Workspace needs finite XYZ minimum/maximum triples")
    if bounds is None:
        raise ValueError("All modes validate commands against --workspace XYZ bounds")
    output = config.output / f"cosmos_{config.mode}_{time.time_ns()}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(dataclasses.asdict(config), default=str, indent=2) + "\n")
    events = (output / "events.jsonl").open("w")
    bag = None
    arm = camera = policy = recorder = episode = None
    counts = {"observations": 0, "predictions": 0, "moves": 0, "gripper_commands": 0, "holds": 0,
              "accepted_commands": 0, "rejected_commands": 0, "stale_frames": 0}
    latencies, inference_times = [], []
    status = "initializing"
    jaw_closed = config.initial_jaw == "closed"
    release_on_stop = return_home_on_stop = gripper_released = returned_home = False
    missed_grasp_stroke_m = None
    previous_sigint = signal.getsignal(signal.SIGINT)
    live = config.mode == "live"

    def log(event, **fields):
        events.write(json.dumps({"event": event, "monotonic_s": time.monotonic(), **fields}) + "\n")
        events.flush()

    def run_command(command: Command, started_at: float, *, expected_end: np.ndarray | None):
        """Poll feedback at 30 Hz through the command's duration, then a settle window; live only."""
        duration = command.ticks / RATE_HZ
        worst = 0.0
        while True:
            now = time.monotonic()
            fraction = min(1.0, (now - started_at) / duration)
            state, _ = arm.read_joints(require_orientation=True)
            if jaw_closed:
                check_grasp(float(state[6]), config.min_grasp_stroke_m)
            error = None
            if command.kind == "cartesian":
                expected = command.sample(np.array([fraction]))[0]
                error = float(np.linalg.norm(state[:3] - expected))
                worst = max(worst, error)
                if error > config.max_tracking_error_m:
                    raise RuntimeError(
                        f"Tracking error {error * 1000:.3f} mm at {fraction * 100:.0f}% of the move "
                        f"exceeds the {config.max_tracking_error_m * 1000:.1f} mm limit"
                    )
            log("tick", kind=command.kind, fraction=fraction, state=state.tolist(), tracking_error_m=error)
            if fraction >= 1.0:
                if expected_end is None:
                    return state
                arrival = float(np.linalg.norm(state[:3] - expected_end))
                if arrival <= config.max_arrival_error_m:
                    log("arrived", kind=command.kind, arrival_error_m=arrival, worst_tracking_error_m=worst,
                        settle_s=now - started_at - duration)
                    return state
                if now - started_at - duration >= config.arrival_settle_s:
                    raise RuntimeError(
                        f"Arrival error {arrival * 1000:.3f} mm after the {config.arrival_settle_s:.1f} s settle "
                        f"window exceeds the {config.max_arrival_error_m * 1000:.1f} mm limit"
                    )
            time.sleep(1 / RATE_HZ)

    try:
        if config.mode == "replay":
            episode = ReplayEpisode(config.episode)
            rows = list(config.replay_rows) or list(range(config.start_row, episode.length, config.replay_stride))
            if any(not 0 <= row < episode.length for row in rows):
                raise ValueError("Replay rows must lie inside the extract")
        else:
            if config.record_bag:
                topics = config.bag_topics or (config.camera_topic, config.camera_topic.rsplit("/", 1)[0] + "/camera_info")
                bag = start_bag(output / "camera_bag", topics, storage_preset=config.bag_storage_preset)
                log("bag_started", directory=str(bag.directory), topics=list(bag.topics), pid=bag.process.pid, wall_s=time.time())
            camera = RosCamera(config.camera_topic)
            start_xyz, start_joints = START_POSES[config.start]
            arm = TrossenArm(config.robot_ip, initial_xyz=np.array(start_xyz))
            if live:
                signal.signal(signal.SIGINT, signal.default_int_handler)
                logging.info("Moving arm to the recorded %s pose %s and aligning XYZ within 0.5 mm",
                             config.start, np.round(start_xyz, 4).tolist())
                log("robot_initialized", **arm.initialize())

        def observe(tick=0):
            if episode is not None:
                return episode.observe(tick, rows[0])
            return observe_hardware(arm, camera, tick, require_orientation=live, jaw_closed=jaw_closed,
                                    min_grasp_stroke_m=config.min_grasp_stroke_m)

        if episode is None:
            initial, state = fresh_observation(observe, after=0.0, timeout_s=5.0)
            orientation_error_deg = float(np.rad2deg(arm.orientation_error_rad))
            log("initial_tool_orientation", measured_rotvec_rad=arm.measured_orientation.tolist(),
                expected_rotvec_rad=arm.orientation.tolist(), error_deg=orientation_error_deg, required_for_motion=live)
            if live:
                log("initial_pose_verified", **arm.verify_initial_pose(state, jaw_open=not jaw_closed))
        else:
            initial, state = episode.observe(0, rows[0])
        np.savez_compressed(output / "initial_observation.npz", **initial.data)
        from PIL import Image

        Image.fromarray(initial.data["observation/image"]).save(output / "camera_crop.png")

        policy = CosmosWebSocketPolicy(config.server, timeout_s=config.inference_timeout_s,
                                       warmup_timeout_s=config.warmup_timeout_s,
                                       expected_export_sha256=config.expected_export_sha256 or None)
        (output / "server_metadata.json").write_text(json.dumps(policy.metadata, indent=2, default=str) + "\n")
        grid = policy.identity.get("destinations")
        if config.snap_to_recorded_destinations and (not isinstance(grid, list) or not grid):
            raise ValueError("The snapping ablation needs a server that publishes the recorded destination grid")
        logging.info("Server: %s waypoint_v4 policy, export %s on %s", policy.identity["policy_family"],
                     policy.identity["export_sha256"][:16], policy.identity.get("gpu"))
        recorder = InferenceRecorder(output)
        logging.info("Warming up transport without motion")
        policy.infer(initial.data)  # Discarded; the server warmed the model at startup.

        position = None
        if live:
            initial, state = fresh_observation(observe, after=0.0, timeout_s=5.0)
            log("policy_start_pose_verified", **arm.verify_initial_pose(state, jaw_open=not jaw_closed))
            log("start_joints_verified", **start_joint_report(
                initial.data["observation/state"][:6], START_POSES[config.start][1], max_error_rad=config.max_start_joint_error_rad))
            if np.any(state[:3] < bounds[0]) or np.any(state[:3] > bounds[1]):
                raise ValueError("Initial pose is outside configured workspace")
            if not jaw_closed and abs(state[6] - INITIAL_JAW_M) > 0.003:
                raise ValueError("Initial open-jaw state does not match measured stroke")
            if jaw_closed and state[6] >= 0.03:
                raise ValueError("Initial closed-jaw intent conflicts with measured open stroke")
            position = state[:3].copy()
            arm.enable()
        jaw_open = not jaw_closed
        epoch = time.monotonic()
        fresh_after = 0.0
        request_id = 0
        replay_index = 0
        status = "running"
        logging.info("Running %s for %.1f seconds; writing %s", config.mode, config.duration_s, output)

        def keep_running():
            elapsed = time.monotonic() - epoch
            if elapsed < config.duration_s:
                return True
            # Past the duration: let the policy place a held ring instead of dropping it mid-carry.
            return live and jaw_closed and elapsed < config.duration_s + config.finish_grace_s

        while keep_running():
            tick = int((time.monotonic() - epoch) * RATE_HZ)
            if episode is not None:
                if replay_index >= len(rows):
                    status = "replay_exhausted"
                    break
                observation, state = episode.observe(tick, rows[replay_index])
                replay_index += 1
                if not 0 <= observation.image_age_s <= MAX_IMAGE_AGE_S:
                    counts["stale_frames"] += 1
                    log("stale_row", row=rows[replay_index - 1], image_age_s=observation.image_age_s)
                    continue
                position = state[:3].astype(float).copy()
                jaw_open = bool(state[6] >= 0.03)
            else:
                observation, state = fresh_observation(observe, after=fresh_after, timeout_s=config.fresh_frame_timeout_s)
                if not live:
                    position = state[:3].astype(float).copy()
                    jaw_open = bool(state[6] >= 0.03)
            counts["observations"] += 1
            recorder.save_input(request_id, observation)
            started = time.monotonic()
            actions, timing, dream = policy.infer(observation.data)
            received = time.monotonic()
            dream_file = save_dream(output, request_id, dream)
            if dream_file:
                counts["dreams"] = counts.get("dreams", 0) + 1
            counts["predictions"] += 1
            latencies.append(received - observation.captured_at)
            inference_times.append(received - started)
            recorder.write("inference_response", request_id, started_at_s=started, received_at_s=received,
                           inference_s=received - started, latency_s=latencies[-1], actions=actions.tolist(),
                           server_timing=timing, dream_file=dream_file, value=(dream["value"] if dream else None))
            log("prediction", request_id=request_id, tick=tick, row=(rows[replay_index - 1] if episode else None),
                latency_s=latencies[-1], inference_s=inference_times[-1], actions=actions.tolist())
            executed = 0
            aborted = False
            for chunk_index, target in enumerate(actions[: config.commit_count]):
                if chunk_index and not keep_running():
                    log("chunk_cut", request_id=request_id, chunk_index=chunk_index, reason="duration reached")
                    break
                steps = plan_waypoint(position, jaw_open, target)
                if not steps:
                    counts["holds"] += 1
                    log("hold", request_id=request_id, chunk_index=chunk_index,
                        tick=int((time.monotonic() - epoch) * RATE_HZ), position=position.tolist(), jaw_open=jaw_open)
                    continue
                for kind, value in steps:
                    tick = int((time.monotonic() - epoch) * RATE_HZ)
                    if kind == "move":
                        if config.snap_to_recorded_destinations:
                            raw = value
                            try:
                                value, phase, distance = snap_destination(
                                    raw, policy.identity["destinations"], max_snap_m=config.snap_max_mm / 1000,
                                    max_grasp_dz_m=config.grasp_level_max_dz_mm / 1000)
                            except ValueError as exc:
                                counts["rejected_commands"] += 1
                                log("rejected", request_id=request_id, chunk_index=chunk_index, tick=tick, reason=str(exc), target_xyz_m=raw.tolist())
                                if live:
                                    raise CommandRejected(str(exc)) from exc
                                aborted = True
                                break
                            log("snapped", request_id=request_id, chunk_index=chunk_index, phase=phase, raw_xyz_m=raw.tolist(),
                                snapped_xyz_m=value.tolist(), distance_m=distance)
                            if np.linalg.norm(value - position) <= 1e-4:
                                counts["holds"] += 1
                                log("hold", request_id=request_id, chunk_index=chunk_index, tick=tick, position=position.tolist(), jaw_open=jaw_open,
                                    reason="snapped destination equals the current position")
                                continue
                        command = move_to_target(position, value, tick)
                        try:
                            validate_trajectory(command, *bounds)
                        except ValueError as exc:
                            counts["rejected_commands"] += 1
                            log("rejected", request_id=request_id, chunk_index=chunk_index, tick=tick, reason=str(exc), target_xyz_m=value.tolist())
                            if live:
                                raise CommandRejected(str(exc)) from exc
                            aborted = True
                            break
                        expected_end = value
                    else:
                        if live and abs(state[6] - INITIAL_JAW_M) > 0.003 and jaw_open:
                            raise ValueError("Gripper is not at its commanded open stroke before a jaw change")
                        command = gripper_command(value, tick)
                        expected_end = None
                    command_started = time.monotonic()
                    if live:
                        arm.dispatch(command, *bounds)
                    counts["accepted_commands"] += 1
                    counts["moves" if kind == "move" else "gripper_commands"] += 1
                    log("command", request_id=request_id, chunk_index=chunk_index, tick=tick, kind=command.kind, ticks=command.ticks,
                        duration_s=command.ticks / RATE_HZ, motion=live,
                        target_xyz_m=(value.tolist() if kind == "move" else position.tolist()),
                        jaw_open=(value if kind == "gripper" else None))
                    if live:
                        if kind == "gripper" and not value:
                            jaw_closed = True  # Missed-grasp checks start with the close command.
                        state = run_command(command, command_started, expected_end=expected_end)
                        if kind == "move":
                            position = value.copy()
                        else:
                            jaw_open = value
                            jaw_closed = not value
                            if value and abs(state[6] - INITIAL_JAW_M) > 0.003:
                                raise ValueError("Gripper did not reach its commanded open stroke")
                    else:
                        if kind == "move":
                            position = value.copy()
                        else:
                            jaw_open = value
                if aborted:
                    break
                executed += 1
            request_id += 1
            fresh_after = time.monotonic()
            if not executed:
                time.sleep(0.1)
        if status == "running":
            status = "duration_reached"  # Time limit is not a declaration of task success.
            log("duration_reached", elapsed_s=time.monotonic() - epoch, jaw_closed=jaw_closed,
                grace_expired=live and jaw_closed)
            if live and jaw_closed:
                logging.warning("Grace period expired with the ring still held; releasing in place before homing")
    except CommandRejected as exc:
        status = "rejected_command"
        log("stopped_on_rejected_command", error=str(exc))
        logging.error("Stopping: %s", exc)
        logging.info("Opening the gripper and returning to joint home")
    except MissedGraspError as exc:
        status = "missed_grasp"
        missed_grasp_stroke_m = exc.stroke_m
        log("missed_grasp", stroke_m=exc.stroke_m, minimum_m=exc.minimum_m, error=str(exc))
        logging.error("%s", exc)
    except KeyboardInterrupt:
        status = "operator_stop"
        log("operator_stop")
        logging.info("Operator stopped the policy; holding arm position and opening gripper")
    except BaseException as exc:
        status = "failed"
        log("failure", error=repr(exc))
        raise
    finally:
        release_on_stop, return_home_on_stop = stop_actions(
            status, live=live and arm is not None, return_home_after_duration=config.return_home_after_duration
        )
        if arm is not None:
            stage = "hold"
            try:
                if release_on_stop:
                    stage = "gripper_release"
                    log("gripper_release_started", reason=status)
                    log("gripper_release_finished", **arm.stop_and_release())
                    gripper_released = True
                    if return_home_on_stop:
                        stage = "return_home"
                        log("return_home_started", reason=status)
                        log("return_home_finished", joints_rad=arm.go_home())
                        returned_home = True
                else:
                    arm.hold()
            except (Exception, KeyboardInterrupt) as exc:
                status = "cleanup_failed"
                log(f"{stage}_failed", error=repr(exc))
                logging.exception("Could not finish robot cleanup; attempting a position hold")
                try:
                    arm.hold()
                except (Exception, KeyboardInterrupt) as hold_exc:
                    log("cleanup_failure", resource="hold", error=repr(hold_exc))
        bag_report = None
        if bag is not None:  # after the arm cleanup, so the return home is in the bag
            try:
                bag_report = bag.stop()
                log("bag_stopped", **bag_report)
                logging.info("Camera bag: %.0f s, %.2f GB, %s", bag_report["duration_s"], bag_report["size_bytes"] / 1e9, bag_report["outcome"])
            except (Exception, KeyboardInterrupt) as exc:
                log("cleanup_failure", resource="bag", error=repr(exc))
        for resource in (policy, camera, arm, episode):
            if resource is not None:
                try:
                    resource.close()
                except (Exception, KeyboardInterrupt) as exc:
                    status = "cleanup_failed"
                    log("cleanup_failure", resource=type(resource).__name__, error=repr(exc))
        summary = {
            "status": status,
            "mode": config.mode,
            "execution_adapter": execution_adapter(config, policy.identity["policy_family"] if policy is not None else "cosmos"),
            "policy_family": policy.identity.get("policy_family") if policy is not None else None,
            "server_export_sha256": policy.identity.get("export_sha256") if policy is not None else None,
            "arm_motion_time_scale": ARM_MOTION_TIME_SCALE,
            "gripper_close_s": GRIPPER_CLOSE_S,
            "gripper_close_settle_s": GRIPPER_CLOSE_SETTLE_S,
            "return_home_requested": return_home_on_stop,
            "returned_home": returned_home,
            "gripper_release_requested": release_on_stop,
            "gripper_released": gripper_released,
            "missed_grasp_stroke_m": missed_grasp_stroke_m,
            **counts,
            "latency_median_s": float(np.median(latencies)) if latencies else None,
            "latency_p95_s": float(np.percentile(latencies, 95)) if latencies else None,
            "inference_median_s": float(np.median(inference_times)) if inference_times else None,
            "task_success": False if missed_grasp_stroke_m is not None else None,
            "camera_bag": bag_report,
        }
        alignment = getattr(arm, "initial_proprio_alignment", None)
        if isinstance(alignment, dict):
            summary["initial_proprio_alignment"] = alignment
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        events.close()
        signal.signal(signal.SIGINT, previous_sigint)
        logging.info("Result: %s", summary)
        if status == "cleanup_failed":
            raise RuntimeError("Deployment cleanup failed; see events.jsonl")


if __name__ == "__main__":
    main(tyro.cli(Config))
