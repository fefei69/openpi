"""Dense contract-five deployment client: 10 Hz reference poses tracked in 0.3 s segments.

Server: ``examples.hanoi.deployment.serve_dense`` (pi0.5 dense checkpoint, port 8000). Same robot
environment, camera, arm, workspace bounds, watchdogs and recovery as the other clients, and the
same v4 episode start pose and joint-space start check as cosmos_client. Unlike the waypoint
clients, observations are taken while the arm moves, one inference is kept in flight, and at every
segment boundary the newest chunk's not-yet-due rows are executed as one tracking segment
(``dense_execution.DenseExecutor``). Raw model output is executed; nothing snaps or corrects it.

Modes:
  replay  stream rows of a raw single-episode extract at 30 Hz (no ROS, no arm); exercises the
          transport, the contract check, the executor and command validation, and reports the
          policy's slot-1 error against the recorded reference poses.
  shadow  live camera and arm feedback, predictions logged, no motion.
  live    initialize at the recorded v4 episode start, then policy control for --duration-s.
"""

import dataclasses
import json
import logging
from pathlib import Path
import signal
import time
from typing import Literal

import numpy as np
from openpi_client import hanoi
from openpi_client import msgpack_numpy
import tyro
from websockets.sync.client import connect

from examples.hanoi.deployment.async_inference import ActionBuffer
from examples.hanoi.deployment.async_inference import InferenceWorker
from examples.hanoi.deployment.async_inference import Observation
from examples.hanoi.deployment.bag import start_bag
from examples.hanoi.deployment.cosmos_client import START_POSES
from examples.hanoi.deployment.cosmos_client import CommandRejected
from examples.hanoi.deployment.cosmos_client import start_joint_report
from examples.hanoi.deployment.cosmos_client import stop_actions
from examples.hanoi.deployment.dense_execution import EXECUTED_ROWS
from examples.hanoi.deployment.dense_execution import HORIZON
from examples.hanoi.deployment.dense_execution import PREFIX
from examples.hanoi.deployment.dense_execution import ROW_TICKS
from examples.hanoi.deployment.dense_execution import TRACK_LIMITS
from examples.hanoi.deployment.dense_execution import DenseExecutor
from examples.hanoi.deployment.dense_execution import carry_violation
from examples.hanoi.deployment.execution import GRIPPER_CLOSE_S
from examples.hanoi.deployment.execution import GRIPPER_CLOSE_SETTLE_S
from examples.hanoi.deployment.execution import RATE_HZ
from examples.hanoi.deployment.hardware import INITIAL_JAW_M
from examples.hanoi.deployment.hardware import MIN_GRASP_STROKE_M
from examples.hanoi.deployment.hardware import MissedGraspError
from examples.hanoi.deployment.hardware import RosCamera
from examples.hanoi.deployment.hardware import TrossenArm
from examples.hanoi.deployment.hardware import check_grasp
from examples.hanoi.deployment.hardware import validate_trajectory

PROMPT = hanoi.PROMPTS["aaaa_to_cccc"]
MAX_IMAGE_AGE_S = hanoi.CONTRACT["max_image_age_s"]
# Selected exports per trained variant (both step 29999): SHA-256 of the export.json manifest.
SELECTED_EXPORTS = {
    "pi05_hanoi_dense_aaaa_to_cccc": "2084b7375cb8b445e56e8833afc37820bb12a638c673012454cbe4d622e49059",  # 30-step chunk
    "pi05_hanoi_dense_h16_aaaa_to_cccc": "1d8cb8e63fee3cfaea629c0cd953b3f1b309e150d33ecc6213b0b0bdd61288e9",  # 16-step chunk
    # Cosmos dense v5, video init, cycle 2, iter 16000 (weights file SHA-256): 16-step chunk, prefix 8.
    "cosmos_hanoi_dense_v5_h16": "ab1a1ccfa5675c7ee49102e043a14141810994f54a44ff8f9368389632c301b7",
}
SELECTED_EXPORT_SHA256 = SELECTED_EXPORTS["pi05_hanoi_dense_aaaa_to_cccc"]
ALLOWED_HORIZONS = (30, 16)
# Execution prefixes the trained contracts declare: 3 (pi0.5, 0.1 s inference) and 8 (Cosmos, 0.5 s).
ALLOWED_PREFIXES = (3, 8)
EXPECTED_CONTRACT = {
    "version": 5,
    "robot": "trossen_wxai_single",
    "reference_rate_hz": RATE_HZ // ROW_TICKS,
    "state": [f"joint_{i}_rad" for i in range(6)] + ["jaw_stroke_m"],
    "actions": ["reference_x_m", "reference_y_m", "reference_z_m", "jaw_open_intent"],
    "internal_xyz_encoding": "absolute",
    "frame": "commissioned_base_tool_frame",
    "orientation_rpy_rad": [0.0, np.pi / 4, 0.0],
    "rgb_crop_xywh": [151, 90, 360, 360],
    "max_image_age_s": MAX_IMAGE_AGE_S,
    "jaw_open_stroke_m": INITIAL_JAW_M,
}
EXECUTION_ADAPTER = "{family}_v5_h{horizon}_track_{rows}rows"


@dataclasses.dataclass
class Config:
    mode: Literal["replay", "shadow", "live"] = "live"
    server: str = "ws://127.0.0.1:8000"
    duration_s: float = 180.0
    # Raw single-episode extract (pixels, joint_positions, proprio, reference_pose, image timing).
    episode: Path = Path("../cosmos-policy/data/hanoi_cosmos/exports_local/hanoi_episode_040.h5")
    start_row: int = 0
    output: Path = Path("data/hanoi/deployment")
    robot_ip: str = "192.168.1.3"
    camera_topic: str = hanoi.CONTRACT["rgb_topic"]
    # Bounds derived from the waypoint_v4 labels; the dense reference paths lie inside the same box.
    workspace: Path | None = dataclasses.field(default_factory=lambda: Path(__file__).with_name("cosmos_workspace.json"))
    initial_jaw: Literal["open", "closed"] = "open"
    min_grasp_stroke_m: float = MIN_GRASP_STROKE_M
    # "selected" is the selected export of the variant the server announces; "" skips the identity
    # check (replay/shadow only); anything else is compared verbatim.
    expected_export_sha256: str = "selected"
    inference_timeout_s: float = 2.0
    warmup_timeout_s: float = 90.0
    # Measured position against the dispatched segment; the driver follows the same quintic.
    max_tracking_error_m: float = 0.008
    max_tick_lateness_s: float = 0.005
    # Measured tool tilt allowed while moving. The commissioned gate is 2 degrees; a loaded lateral carry
    # reached 2.05 and stopped run 2 mid-carry. Each tick logs the error.
    max_orientation_error_deg: float = 3.0
    # A sideways move with a ring held below this height is rejected and the run stops (0 disables).
    # The recording carries at 191 mm and releases at 150 mm; run 2 (h16) travelled between pegs at 151.
    carry_min_z_m: float = 0.17
    # Rows of each chunk executed per segment (0.1 s each) before re-planning; see dense_execution.
    prefix_rows: int = EXECUTED_ROWS
    # Segments whose derivatives exceed the motion limits are slowed down up to this factor.
    max_segment_stretch: float = 6.0
    # episode_start: behind peg B, how every training episode begins. above_peg_a: the recorded grasp
    # hover over A, for comparisons with methods that start there; the first chunk is then the descent.
    start: Literal["episode_start", "above_peg_a"] = "above_peg_a"
    max_start_joint_error_rad: float = 0.05
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
    identity = metadata.get("hanoi_dense") if isinstance(metadata, dict) else None
    if not isinstance(identity, dict) or not isinstance(identity.get("contract"), dict):
        raise ValueError("Server is not a Hanoi dense contract-five policy server")
    contract = identity["contract"]
    for key, expected in EXPECTED_CONTRACT.items():
        actual = contract.get(key)
        if key == "orientation_rpy_rad":
            same = isinstance(actual, (list, tuple)) and len(actual) == 3 and np.allclose(actual, expected)
        else:
            same = actual == expected
        if not same:
            raise ValueError(f"Contract mismatch for {key}: server {actual!r}, client {expected!r}")
    if contract.get("execution_prefix") not in ALLOWED_PREFIXES:
        raise ValueError(f"Contract mismatch for execution_prefix: server {contract.get('execution_prefix')!r}, client {ALLOWED_PREFIXES}")
    if contract.get("action_horizon") not in ALLOWED_HORIZONS:
        raise ValueError(f"Contract mismatch for action_horizon: server {contract.get('action_horizon')!r}, client {ALLOWED_HORIZONS}")
    if identity.get("prompt") != PROMPT:
        raise ValueError("Server prompt is not the trained AAAA-to-CCCC instruction")
    if expected_export_sha256 == "selected":
        expected_export_sha256 = SELECTED_EXPORTS.get(identity.get("config_name"))
        if expected_export_sha256 is None:
            raise ValueError(f"No selected export for server config {identity.get('config_name')!r}")
    if expected_export_sha256 and identity.get("export_sha256") != expected_export_sha256:
        raise ValueError("Server export is not the selected checkpoint")
    return {**identity, "action_horizon": int(contract["action_horizon"]), "execution_prefix": int(contract["execution_prefix"]),
            "policy_family": identity.get("model", "pi05_dense")}


class DenseWebSocketPolicy:
    """Bounded-time OpenPI-protocol transport used by the inference worker."""

    def __init__(self, uri: str, *, timeout_s: float, warmup_timeout_s: float, expected_export_sha256: str | None):
        self.timeout_s, self.warmup_timeout_s, self.first = timeout_s, warmup_timeout_s, True
        self.ws = connect(uri, compression=None, max_size=16 * 1024 * 1024, open_timeout=5, close_timeout=0.2)
        try:
            self.metadata = msgpack_numpy.unpackb(self.ws.recv(timeout=5))
            self.identity = check_contract(self.metadata, expected_export_sha256=expected_export_sha256)
            self.horizon = self.identity["action_horizon"]
            self.execution_prefix = self.identity["execution_prefix"]
        except BaseException:
            self.ws.close()
            raise

    def infer(self, observation: dict) -> dict:
        self.ws.send(msgpack_numpy.packb(observation))
        reply = self.ws.recv(timeout=self.warmup_timeout_s if self.first else self.timeout_s)
        self.first = False
        if isinstance(reply, str):
            raise RuntimeError(f"Policy server failed: {reply}")
        reply = msgpack_numpy.unpackb(reply)
        if int(reply.get("execution_prefix", self.execution_prefix)) != self.execution_prefix or int(reply.get("reference_rate_hz", 10)) != RATE_HZ // ROW_TICKS:
            raise ValueError("Server changed the execution prefix or reference rate")
        return reply

    def close(self):
        self.ws.close()


# ---- observations ----


def observation_data(image: np.ndarray, joints: np.ndarray, jaw_stroke_m: float, xyz: np.ndarray) -> dict:
    """The dense contract's request; the measured XYZ rides along for analysis only."""
    return {
        "observation/image": np.ascontiguousarray(image, dtype=np.uint8),
        "observation/state": np.r_[np.asarray(joints, np.float32), np.float32(jaw_stroke_m)].astype(np.float32),
        "observation/cartesian_position": np.asarray(xyz, np.float32),
        "prompt": PROMPT,
    }


class ReplayEpisode:
    """Rows of a raw single-episode extract, streamed on the control clock."""

    def __init__(self, path: Path):
        import h5py

        self.h5 = h5py.File(path, "r")
        self.length = len(self.h5["proprio"])
        self.reference = np.asarray(self.h5["reference_pose"][:, :3], dtype=np.float64)
        # Commanded derivatives, the replay analogue of the live executor's continuity state.
        self.reference_velocity = np.gradient(self.reference, 1 / RATE_HZ, axis=0)
        self.reference_acceleration = np.gradient(self.reference_velocity, 1 / RATE_HZ, axis=0)

    def observe(self, tick: int, row: int) -> tuple[Observation, np.ndarray]:
        row = min(row, self.length - 1)
        joints = np.asarray(self.h5["joint_positions"][row], np.float32)
        proprio = np.asarray(self.h5["proprio"][row], np.float64)
        image = np.asarray(self.h5["pixels"][row])
        age = (int(self.h5["command_monotonic_ns"][row]) - int(self.h5["image_receipt_monotonic_ns"][row])) / 1e9
        captured = time.monotonic()
        state = np.r_[proprio[:6], proprio[6]].astype(np.float32)
        return Observation(tick, captured, captured - age, observation_data(image, joints, proprio[6], proprio[:3])), state

    def commanded_state(self, row: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        row = min(row, self.length - 1)
        return self.reference[row].copy(), self.reference_velocity[row].copy(), self.reference_acceleration[row].copy()

    def slot_errors_mm(self, row: int, actions: np.ndarray) -> list:
        """Per-slot XYZ error against the recorded reference three rows per slot ahead."""
        rows = np.minimum(row + ROW_TICKS * np.arange(1, len(actions) + 1), self.length - 1)
        return (np.linalg.norm(np.asarray(actions)[:, :3] - self.reference[rows], axis=-1) * 1000).round(3).tolist()

    def close(self):
        self.h5.close()


def observe_hardware(arm, camera, tick, *, require_orientation, jaw_closed, min_grasp_stroke_m):
    state, joints = arm.read_joints(require_orientation=require_orientation)
    if jaw_closed:
        check_grasp(float(state[6]), min_grasp_stroke_m)
    frame = camera.latest()
    captured = time.monotonic()
    return Observation(tick, captured, frame.received_at, observation_data(frame.rgb, joints, state[6], state[:3])), state


# ---- main ----


def main(config: Config):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not np.isfinite(config.duration_s) or config.duration_s <= 0 or config.start_row < 0:
        raise ValueError("Duration must be finite and positive; start row must be nonnegative")
    if min(config.inference_timeout_s, config.warmup_timeout_s, config.max_tracking_error_m,
           config.max_tick_lateness_s, config.max_segment_stretch) <= 0:
        raise ValueError("Timeouts and tolerances must be positive")
    if not PREFIX <= config.prefix_rows <= 12:
        raise ValueError(f"--prefix-rows must be between the contract's {PREFIX} and 12")
    if not np.isfinite(config.min_grasp_stroke_m) or not 0 < config.min_grasp_stroke_m < INITIAL_JAW_M:
        raise ValueError("Minimum grasp stroke must be finite, positive, and below the open stroke")
    bounds = None
    if config.workspace:
        workspace = json.loads(config.workspace.read_text())
        bounds = np.asarray([workspace["xyz_min_m"], workspace["xyz_max_m"]], dtype=float)
        if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[0] >= bounds[1]):
            raise ValueError("Workspace needs finite XYZ minimum/maximum triples")
    live = config.mode == "live"
    if live and bounds is None:
        raise ValueError("Live motion requires --workspace with XYZ bounds")
    output = config.output / f"dense_{config.mode}_{time.time_ns()}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(dataclasses.asdict(config), default=str, indent=2) + "\n")
    events = (output / "events.jsonl").open("w")
    bag = None
    arm = camera = worker = episode = policy = None
    counts = {"ticks": 0, "predictions": 0, "accepted_predictions": 0, "segments": 0, "stretched_segments": 0,
              "brakes": 0, "holds": 0, "gripper_commands": 0, "accepted_commands": 0, "rejected_commands": 0,
              "stale_frames": 0}
    latencies, lateness, tracking, slot1_errors = [], [], [], []
    status = "initializing"
    last_command = None
    command_started_at = busy_until = fresh_image_after = 0.0
    jaw_closed = config.initial_jaw == "closed"
    gripper_released = returned_home = False
    missed_grasp_stroke_m = None
    previous_sigint = signal.getsignal(signal.SIGINT)

    def log(event, **fields):
        events.write(json.dumps({"event": event, "monotonic_s": time.monotonic(), **fields}) + "\n")
        events.flush()

    def observe(tick):
        if episode is not None:
            return episode.observe(tick, config.start_row + tick)
        return observe_hardware(arm, camera, tick, require_orientation=live, jaw_closed=jaw_closed,
                                min_grasp_stroke_m=config.min_grasp_stroke_m)

    try:
        if config.mode == "replay":
            episode = ReplayEpisode(config.episode)
            if config.start_row + int(config.duration_s * RATE_HZ) >= episode.length:
                raise ValueError("Requested replay extends beyond the recorded episode")
        else:
            if config.record_bag:
                topics = config.bag_topics or (config.camera_topic, config.camera_topic.rsplit("/", 1)[0] + "/camera_info")
                bag = start_bag(output / "camera_bag", topics, storage_preset=config.bag_storage_preset)
                log("bag_started", directory=str(bag.directory), topics=list(bag.topics), pid=bag.process.pid)
            camera = RosCamera(config.camera_topic)
            start_xyz, start_joints = START_POSES[config.start]
            arm = TrossenArm(config.robot_ip, initial_xyz=np.array(start_xyz))
            arm.orientation_limit_deg = config.max_orientation_error_deg
            if live:
                signal.signal(signal.SIGINT, signal.default_int_handler)
                logging.info("Moving arm to the recorded %s pose %s and aligning XYZ within 0.5 mm",
                             config.start, np.round(start_xyz, 4).tolist())
                log("robot_initialized", **arm.initialize())
        deadline = time.monotonic() + 5
        while True:
            initial, state = observe(0)
            if 0 <= initial.image_age_s <= MAX_IMAGE_AGE_S:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError("Waiting for a fresh camera frame")
            time.sleep(0.01)
        if arm is not None:
            log("initial_tool_orientation", measured_rotvec_rad=arm.measured_orientation.tolist(),
                expected_rotvec_rad=arm.orientation.tolist(), error_deg=float(np.rad2deg(arm.orientation_error_rad)),
                required_for_motion=live)
            if live:
                log("initial_pose_verified", **arm.verify_initial_pose(state, jaw_open=not jaw_closed))
        np.savez_compressed(output / "initial_observation.npz", **initial.data)
        from PIL import Image

        Image.fromarray(initial.data["observation/image"]).save(output / "camera_crop.png")

        policy = DenseWebSocketPolicy(config.server, timeout_s=config.inference_timeout_s,
                                      warmup_timeout_s=config.warmup_timeout_s,
                                      expected_export_sha256=config.expected_export_sha256 or None)
        (output / "server_metadata.json").write_text(json.dumps(policy.metadata, indent=2, default=str) + "\n")
        logging.info("Server: %s (%d-step chunks, prefix %d), export %s on %s", policy.identity.get("config_name"), policy.horizon,
                     policy.execution_prefix, policy.identity["export_sha256"][:16], policy.identity.get("gpu"))
        if not policy.execution_prefix <= config.prefix_rows <= policy.horizon - 4:
            raise ValueError(f"--prefix-rows {config.prefix_rows} must lie between the server's execution prefix "
                             f"{policy.execution_prefix} and its chunk length minus four rows of latency slack ({policy.horizon - 4})")
        worker = InferenceWorker(policy, record_dir=output, horizon=policy.horizon)
        logging.info("Warming up transport without motion")
        worker.submit(initial)
        deadline = time.monotonic() + config.warmup_timeout_s + 1
        while worker.take() is None:
            if time.monotonic() > deadline:
                raise TimeoutError("Policy warmup timed out")
            time.sleep(0.01)
        generation = worker.invalidate()  # Never execute the warm-up result.

        initial, state = observe(0)
        jaw_open = not jaw_closed
        if live:
            log("policy_start_pose_verified", **arm.verify_initial_pose(state, jaw_open=jaw_open))
            log("start_joints_verified", **start_joint_report(
                initial.data["observation/state"][:6], START_POSES[config.start][1], max_error_rad=config.max_start_joint_error_rad))
            if not 0 <= initial.image_age_s <= MAX_IMAGE_AGE_S:
                raise ValueError("Camera stale after warmup")
            if np.any(state[:3] < bounds[0]) or np.any(state[:3] > bounds[1]):
                raise ValueError("Initial pose is outside configured workspace")
            if jaw_open and abs(state[6] - INITIAL_JAW_M) > 0.003:
                raise ValueError("Initial open-jaw state does not match measured stroke")
            if not jaw_open and state[6] >= 0.03:
                raise ValueError("Initial closed-jaw intent conflicts with measured open stroke")
            arm.enable()
        executor = DenseExecutor(state[:3].astype(float).copy(), jaw_open=jaw_open, horizon=policy.horizon,
                                 prefix=config.prefix_rows, max_stretch=config.max_segment_stretch)
        log("reference_initialized", measured_state=state.tolist(), jaw_open=jaw_open)
        buffer = ActionBuffer(executor, generation=generation)
        epoch = time.monotonic()
        next_tick = 0
        duration_ticks = int(config.duration_s * RATE_HZ)
        grace_ticks = int(config.finish_grace_s * RATE_HZ)
        status = "running"
        logging.info("Running %s for %.1f seconds; writing %s", config.mode, config.duration_s, output)
        while True:
            due = epoch + next_tick / RATE_HZ
            time.sleep(max(0.0, due - time.monotonic()))
            now = time.monotonic()
            tick = max(next_tick, int((now - epoch) * RATE_HZ))
            lag = now - (epoch + tick / RATE_HZ)
            lateness.append(lag)
            if live and (tick != next_tick or lag > config.max_tick_lateness_s):
                log("control_late", expected_tick=next_tick, actual_tick=tick, lateness_s=now - due)
            next_tick = tick + 1
            counts["ticks"] += 1
            observation, state = observe(tick)
            log("tick", tick=tick, lateness_s=lag, state=state.tolist(), image_age_s=observation.image_age_s,
                orientation_error_deg=(float(np.rad2deg(arm.orientation_error_rad)) if arm is not None else None))
            if live and last_command is not None:
                if last_command.kind == "cartesian":
                    fraction = np.clip((now - command_started_at) / (last_command.ticks / RATE_HZ), 0, 1)
                    expected = last_command.sample(np.array([fraction]))[0]
                else:
                    expected = buffer.executor.position
                error = float(np.linalg.norm(state[:3] - expected))
                tracking.append(error)
                if error > config.max_tracking_error_m:
                    raise RuntimeError(f"Tracking error {error * 1000:.3f} mm exceeds the {config.max_tracking_error_m * 1000:.1f} mm limit")
            prediction = worker.take()
            if prediction is not None:
                counts["predictions"] += 1
                latencies.append(prediction.received_at - prediction.observation.captured_at)
                accepted = buffer.accept(prediction)
                counts["accepted_predictions"] += int(accepted)
                fields = {}
                if episode is not None:
                    errors = episode.slot_errors_mm(config.start_row + prediction.observation.tick, prediction.actions)
                    slot1_errors.append(errors[0])
                    fields["reference_error_mm"] = errors
                log("prediction", request_id=prediction.request_id, tick=tick, observation_tick=prediction.observation.tick,
                    latency_s=latencies[-1], inference_s=prediction.inference_s, accepted=accepted,
                    actions=prediction.actions.tolist(), **fields)
            dwelling = last_command is not None and last_command.kind == "gripper" and now < busy_until
            if not 0 <= observation.image_age_s <= MAX_IMAGE_AGE_S:
                counts["stale_frames"] += 1
                if live:
                    raise RuntimeError("Live camera frame exceeded the 50 ms age limit")
            elif not dwelling and observation.image_received_at >= fresh_image_after:
                worker.submit(observation)
            if now < busy_until:
                continue
            # Segment boundary: end of the run is decided here so a dispatched segment always completes.
            if tick >= duration_ticks and not (live and jaw_closed and tick < duration_ticks + grace_ticks):
                status = "duration_reached"
                log("duration_reached", elapsed_s=now - epoch, jaw_closed=jaw_closed, grace_expired=live and jaw_closed)
                if live and jaw_closed:
                    logging.warning("Grace period expired with the ring still held; releasing in place before homing")
                break
            if episode is not None:
                # The recording does not follow hypothetical commands: plan from the recorded commanded
                # state, which is what the live executor's continuity provides.
                position, velocity, acceleration = episode.commanded_state(config.start_row + tick)
                buffer.executor.position, buffer.executor.velocity, buffer.executor.acceleration = position, velocity, acceleration
            elif not live:
                buffer.executor.position = state[:3].astype(float).copy()
                buffer.executor.velocity = np.zeros(3)
                buffer.executor.acceleration = np.zeros(3)
            try:
                proposal = buffer.propose(tick)
                if proposal is None:
                    continue
                command, proposed_executor = proposal
                source = buffer.prediction
                if command.kind == "gripper" and live and np.linalg.norm(buffer.executor.velocity) > 0.002:
                    raise ValueError("Reference trajectory must stop before a gripper command")
                if live and last_command is not None and last_command.kind == "gripper" and last_command.jaw_open \
                        and abs(state[6] - INITIAL_JAW_M) > 0.003:
                    raise ValueError("Gripper did not reach its commanded open stroke")
                if command.kind == "cartesian" and bounds is not None:
                    validate_trajectory(command, *bounds, TRACK_LIMITS)
                if live and jaw_closed and command.kind == "cartesian":
                    low = carry_violation(command, min_z_m=config.carry_min_z_m)
                    if low is not None:
                        raise ValueError(f"Sideways carry with the ring held at {low * 1000:.0f} mm, below the "
                                         f"{config.carry_min_z_m * 1000:.0f} mm carry height")
            except ValueError as exc:
                if "expired" in str(exc):  # the newest chunk is older than its horizon: wait for a fresh one
                    counts["expired_predictions"] = counts.get("expired_predictions", 0) + 1
                    buffer.prediction = None
                    continue
                counts["rejected_commands"] += 1
                log("rejected", tick=tick, request_id=buffer.prediction.request_id if buffer.prediction else None,
                    reason=str(exc))
                if live:
                    raise CommandRejected(str(exc)) from exc
                buffer.prediction = None
                buffer.executor = DenseExecutor(state[:3].astype(float).copy(), jaw_open=bool(state[6] >= 0.03),
                                                available_tick=tick, horizon=policy.horizon, prefix=config.prefix_rows,
                                                max_stretch=config.max_segment_stretch)
                continue
            command_started_at = time.monotonic()
            if live:
                arm.dispatch(command, *bounds, TRACK_LIMITS)
            buffer.commit(command, proposed_executor)
            last_command = command
            busy_until = command_started_at + command.ticks / RATE_HZ
            counts["accepted_commands"] += 1
            if command.kind == "cartesian":
                counts["segments"] += 1
                counts["stretched_segments"] += int(proposed_executor.last_stretch > 1.0)
                counts["brakes"] += int(proposed_executor.last_braked)
            elif command.kind == "gripper":
                counts["gripper_commands"] += 1
            else:
                counts["holds"] += 1
            log("command", request_id=source.request_id, observation_tick=source.observation.tick,
                elapsed_rows=(tick - source.observation.tick) // ROW_TICKS, tick=tick, kind=command.kind,
                ticks=command.ticks, duration_s=command.ticks / RATE_HZ, motion=live,
                target_xyz_m=buffer.executor.position.tolist(), end_velocity_m_s=buffer.executor.velocity.tolist(),
                stretch=proposed_executor.last_stretch if command.kind == "cartesian" else None,
                braked=proposed_executor.last_braked if command.kind == "cartesian" else None,
                jaw_open=command.jaw_open, gripper_alignment=proposed_executor.pending_jaw_open is not None)
            if command.kind == "gripper":
                jaw_closed = not command.jaw_open
                buffer.generation = worker.invalidate()
                fresh_image_after = busy_until
    except CommandRejected as exc:
        status = "rejected_command"
        log("stopped_on_rejected_command", error=str(exc))
        logging.error("Stopping: %s", exc)
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
            status, live=live and arm is not None, return_home_after_duration=config.return_home_after_duration)
        if arm is not None:
            stage = "hold"
            try:
                if release_on_stop:
                    stage = "gripper_release"
                    log("gripper_release_started", reason=status)
                    recovery = arm.stop_and_release()
                    gripper_released = True
                    log("gripper_release_finished", **recovery)
                    if return_home_on_stop:
                        stage = "return_home"
                        log("return_home_started", reason=status)
                        logging.info("Returning to joint home")
                        home_joints = arm.go_home()
                        returned_home = True
                        log("return_home_finished", joints_rad=home_joints)
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
        for resource, method in ((worker, "close"), (camera, "close"), (arm, "close"), (episode, "close")):
            if resource is not None:
                try:
                    getattr(resource, method)()
                except (Exception, KeyboardInterrupt) as exc:
                    status = "cleanup_failed"
                    log("cleanup_failure", resource=method, error=repr(exc))
        summary = {
            "status": status,
            "mode": config.mode,
            "execution_adapter": EXECUTION_ADAPTER.format(family=policy.identity["policy_family"] if policy is not None else "?",
                                                          horizon=policy.horizon if policy is not None else "?", rows=config.prefix_rows),
            "prefix_rows": config.prefix_rows,
            "policy_family": policy.identity["policy_family"] if policy is not None else None,
            "execution_prefix": policy.execution_prefix if policy is not None else None,
            "start": config.start,
            "config_name": policy.identity.get("config_name") if policy is not None else None,
            "action_horizon": policy.horizon if policy is not None else None,
            "server_export_sha256": policy.identity.get("export_sha256") if policy is not None else None,
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
            "tracking_error_p95_mm": float(np.percentile(tracking, 95) * 1000) if tracking else None,
            "tracking_error_max_mm": float(max(tracking) * 1000) if tracking else None,
            "replay_slot1_error_mean_mm": float(np.mean(slot1_errors)) if slot1_errors else None,
            "max_tick_lateness_s": max(lateness, default=0.0),
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
