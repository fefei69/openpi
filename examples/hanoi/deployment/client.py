"""Async Hanoi deployment: recorded replay, live shadow, or explicit arm motion.

Run as a module from the repository root. See this folder's README.md for environment and
execution limitations. Live mode initializes the arm before running the policy.
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
import tyro

from examples.hanoi.deployment.async_inference import ActionBuffer
from examples.hanoi.deployment.async_inference import InferenceWorker
from examples.hanoi.deployment.async_inference import Observation
from examples.hanoi.deployment.async_inference import WebSocketPolicy
from examples.hanoi.deployment.async_inference import reference_executor
from examples.hanoi.deployment.execution import ARM_MOTION_TIME_SCALE
from examples.hanoi.deployment.execution import GRIPPER_CLOSE_S
from examples.hanoi.deployment.execution import GRIPPER_CLOSE_SETTLE_S
from examples.hanoi.deployment.execution import PolicyExecutor
from examples.hanoi.deployment.hardware import MIN_GRASP_STROKE_M
from examples.hanoi.deployment.hardware import MissedGraspError
from examples.hanoi.deployment.hardware import RosCamera
from examples.hanoi.deployment.hardware import TrossenArm
from examples.hanoi.deployment.hardware import check_grasp
from examples.hanoi.deployment.hardware import validate_trajectory
from examples.hanoi.deployment.recorded_velocity import RecordedVelocity


@dataclasses.dataclass
class Config:
    mode: Literal["replay", "shadow", "live"] = "live"
    server: str = "ws://127.0.0.1:8000"
    duration_s: float = 30.0
    episode: Path = Path("data/hanoi/deployment_debug/aaaa_to_cccc_episode_000/episode.h5")
    # Diagnostic override of policy state[3:6]; hardware feedback remains measured.
    velocity_source: Literal["measured", "recorded"] = "measured"
    start_row: int = 0
    output: Path = Path("data/hanoi/deployment")
    robot_ip: str = "192.168.1.3"
    camera_topic: str = hanoi.CONTRACT["rgb_topic"]
    # Recorded task envelope in the collection base/tool frame; override if needed.
    workspace: Path | None = dataclasses.field(default_factory=lambda: Path(__file__).with_name("workspace.json"))
    # Known starting gripper intent; open is also checked against measured stroke.
    initial_jaw: Literal["open", "closed"] = "open"
    # Collector's minimum valid grasp stroke; monitored throughout a closed grip.
    min_grasp_stroke_m: float = MIN_GRASP_STROKE_M
    # Maximum wait per warmed inference request.
    inference_timeout_s: float = 1.0
    # Separate first-inference compilation timeout; its output is discarded.
    warmup_timeout_s: float = 90.0
    # Maximum permitted measured/reference position difference.
    max_tracking_error_m: float = 0.003
    # Host scheduling delay above which to log a timing diagnostic.
    max_tick_lateness_s: float = 0.005


def main(config: Config):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not np.isfinite(config.duration_s) or config.duration_s <= 0 or config.start_row < 0:
        raise ValueError("Duration must be finite and positive; start row must be nonnegative")
    if not np.isfinite(
        [config.inference_timeout_s, config.warmup_timeout_s, config.max_tracking_error_m, config.max_tick_lateness_s]
    ).all():
        raise ValueError("Timeouts and tolerances must be finite")
    if (
        min(
            config.inference_timeout_s, config.warmup_timeout_s, config.max_tracking_error_m, config.max_tick_lateness_s
        )
        <= 0
    ):
        raise ValueError("Timeouts and tolerances must be positive")
    if not np.isfinite(config.min_grasp_stroke_m) or not 0 < config.min_grasp_stroke_m < 0.034:
        raise ValueError("Minimum grasp stroke must be finite, positive, and below the open stroke")
    bounds = None
    if config.workspace:
        workspace = json.loads(config.workspace.read_text())
        bounds = np.asarray([workspace["xyz_min_m"], workspace["xyz_max_m"]], dtype=float)
        if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[0] >= bounds[1]):
            raise ValueError("Workspace needs finite XYZ minimum/maximum triples")
    if config.mode == "live" and bounds is None:
        raise ValueError("Live motion requires --workspace with XYZ bounds")
    velocity_reference = None
    if config.velocity_source == "recorded":
        if config.mode == "replay" or config.initial_jaw != "open":
            raise ValueError("Recorded velocity requires a live camera and an open-jaw start")
        # Validate and load the reference before constructing any hardware.
        velocity_reference = RecordedVelocity.load(config.episode)
    # Each run owns its artifacts; never overwrite a previous deployment record.
    output = config.output / f"{config.mode}_{time.time_ns()}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(dataclasses.asdict(config), default=str, indent=2) + "\n")
    events = (output / "events.jsonl").open("w")
    arm = camera = worker = episode = None
    counts = {"ticks": 0, "predictions": 0, "accepted_commands": 0, "rejected_commands": 0, "stale_frames": 0}
    latencies, lateness = [], []
    status = "initializing"
    last_command = None
    command_started_at = 0.0
    busy_until = 0.0
    fresh_image_after = 0.0
    jaw_closed = False
    release_on_stop = False
    return_home_on_stop = False
    gripper_released = False
    returned_home = False
    missed_grasp_stroke_m = None
    previous_sigint = signal.getsignal(signal.SIGINT)

    def log(event, **fields):
        events.write(json.dumps({"event": event, "monotonic_s": time.monotonic(), **fields}) + "\n")
        events.flush()

    def observe(tick):
        if episode is not None:
            row = min(config.start_row + tick, len(episode["proprio"]) - 1)
            state = episode["proprio"][row, :7]
            image = episode["pixels"][row]
            age = (int(episode["command_monotonic_ns"][row]) - int(episode["image_receipt_monotonic_ns"][row])) / 1e9
            captured = time.monotonic()
            received = captured - age
        else:
            state = arm.read(require_orientation=config.mode == "live")
            if config.mode == "live" and jaw_closed:
                check_grasp(float(state[6]), config.min_grasp_stroke_m)
            frame = camera.latest()
            image, received = frame.rgb, frame.received_at
            captured = time.monotonic()
        return Observation(
            tick,
            captured,
            received,
            {"observation/image": image, "observation/state": state, "prompt": hanoi.PROMPTS["aaaa_to_cccc"]},
        )

    def policy_observation(observation):
        return observation if velocity_reference is None else velocity_reference.apply(observation)

    try:
        if velocity_reference is not None:
            logging.info("Velocity diagnostic: recorded velocity; LIVE images, XYZ, and jaw stroke")
            log(
                "velocity_override_enabled",
                episode=str(config.episode.resolve()),
                reference_sha256=velocity_reference.sha256,
                alignment="nearest_measured_xyz_forward_within_gripper_phase",
            )
        if config.mode == "replay":
            import h5py

            episode = h5py.File(config.episode, "r")
            metadata = json.loads(config.episode.with_suffix(".json").read_text())
            if metadata["contract"] != hanoi.CONTRACT or metadata["prompt"] != hanoi.PROMPTS["aaaa_to_cccc"]:
                raise ValueError("Replay requires the matching forward Hanoi episode")
            if config.start_row + int(config.duration_s * 30) >= len(episode["proprio"]):
                raise ValueError("Requested replay extends beyond the recorded episode")
        else:
            camera = RosCamera(config.camera_topic)
            arm = TrossenArm(config.robot_ip)
            if config.mode == "live":
                # Keep Ctrl-C in Python's try/finally path even if a hardware or
                # ROS library installed its own signal handler while connecting.
                signal.signal(signal.SIGINT, signal.default_int_handler)
                logging.info("Moving arm to the rod-A setup pose and aligning XYZ within 0.5 mm")
                log("robot_initialized", **arm.initialize())
        # Wait for an initial image before policy inference.
        deadline = time.monotonic() + 5
        while True:
            try:
                initial = observe(0)
                if not 0 <= initial.image_age_s <= 0.05:
                    raise RuntimeError("Waiting for a fresh camera frame")
                break
            except RuntimeError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.01)
        if arm is not None:
            orientation_error_deg = float(np.rad2deg(arm.orientation_error_rad))
            log(
                "initial_tool_orientation",
                measured_rotvec_rad=arm.measured_orientation.tolist(),
                expected_rotvec_rad=arm.orientation.tolist(),
                error_deg=orientation_error_deg,
                required_for_motion=config.mode == "live",
            )
            logging.info(
                "Measured tool orientation (angle-axis rad): %s; difference from dataset orientation: %.2f degrees",
                arm.measured_orientation.tolist(),
                orientation_error_deg,
            )
            if config.mode == "shadow" and orientation_error_deg > 2:
                logging.warning(
                    "Shadow mode: tool orientation is outside the live client's 2-degree tolerance. "
                    "Continuing inference without motion; the arm has not been initialized to the dataset pose."
                )
            if config.mode == "live":
                log(
                    "initial_pose_verified",
                    **arm.verify_initial_pose(initial.data["observation/state"], jaw_open=config.initial_jaw == "open"),
                )
        np.savez_compressed(output / "initial_observation.npz", **initial.data)
        from PIL import Image

        Image.fromarray(initial.data["observation/image"]).save(output / "camera_crop.png")
        policy = WebSocketPolicy(
            config.server, timeout_s=config.inference_timeout_s, warmup_timeout_s=config.warmup_timeout_s
        )
        (output / "server_metadata.json").write_text(json.dumps(policy.metadata, indent=2) + "\n")
        if config.mode == "live":
            identity = policy.metadata.get("hanoi_deployment", {})
            expected = {
                "config_name": "pi05_hanoi_aaaa_to_cccc",
                "export_sha256": "38b1e39be734df97a49836f8ede9238f1e97335af724e4a8448a50c9bf5b4b88",
                "normalization_sha256": "ec82035dd64b6715d75644addc7b040bb4d347ec3a3f79b730ba7930590c1c48",
                "num_steps": 10,
            }
            if identity != expected:
                policy.close()
                raise ValueError("Live mode requires the audited forward identity from examples.hanoi.deployment.serve")
        worker = InferenceWorker(policy, record_dir=output)
        logging.info("Saving every inference input and reply under %s", output)
        logging.info("Warming up policy without motion (first JAX compilation may take about 25 seconds)")
        worker.submit(policy_observation(initial))
        deadline = time.monotonic() + config.warmup_timeout_s + 1
        while worker.take() is None:
            if time.monotonic() > deadline:
                raise TimeoutError("Policy warmup timed out")
            time.sleep(0.01)
        generation = worker.invalidate()  # Never execute the compile/warmup result.
        initial = observe(0)
        state = initial.data["observation/state"]
        jaw_open = config.initial_jaw == "open"
        if config.mode == "live":
            log("policy_start_pose_verified", **arm.verify_initial_pose(state, jaw_open=jaw_open))
            if not 0 <= initial.image_age_s <= 0.05:
                raise ValueError("Camera stale after warmup")
            if np.any(state[:3] < bounds[0]) or np.any(state[:3] > bounds[1]):
                raise ValueError("Initial pose is outside configured workspace")
            if jaw_open and abs(state[6] - 0.034) > 0.003:
                raise ValueError("Initial open-jaw state does not match measured stroke")
            if not jaw_open and state[6] >= 0.03:
                raise ValueError("Initial closed-jaw intent conflicts with measured open stroke")
            executor = reference_executor(state, jaw_open=jaw_open)
            log(
                "reference_initialized",
                measured_state=state.tolist(),
                reference_velocity_m_s=executor.velocity.tolist(),
                reference_acceleration_m_s2=executor.acceleration.tolist(),
            )
            arm.enable()
        else:
            executor = PolicyExecutor(state[:3].copy(), jaw_open=jaw_open)
        buffer = ActionBuffer(executor, generation=generation)
        epoch = time.monotonic()
        next_tick = 0
        status = "running"
        logging.info("Running %s for %.1f seconds; writing %s", config.mode, config.duration_s, output)
        while next_tick < config.duration_s * 30:
            due = epoch + next_tick / 30
            # Let the controller finish the dispatched move before selecting a new target.
            if last_command is not None and next_tick >= buffer.executor.available_tick:
                due = max(due, busy_until)
            time.sleep(max(0.0, due - time.monotonic()))
            now = time.monotonic()
            tick = max(next_tick, int((now - epoch) * 30))
            lag = now - (epoch + tick / 30)
            lateness.append(lag)
            if config.mode == "live" and (tick != next_tick or lag > config.max_tick_lateness_s):
                log("control_late", expected_tick=next_tick, actual_tick=tick, lateness_s=now - due)
            next_tick = tick + 1
            counts["ticks"] += 1
            observation = observe(tick)
            state = observation.data["observation/state"]
            log("tick", tick=tick, lateness_s=lag, state=state.tolist(), image_age_s=observation.image_age_s)
            if config.mode == "live" and last_command is not None:
                if last_command.kind == "cartesian":
                    fraction = np.clip((now - command_started_at) / (last_command.ticks / 30), 0, 1)
                    expected = last_command.sample(np.array([fraction]))[0]
                else:
                    expected = buffer.executor.position
                error = float(np.linalg.norm(state[:3] - expected))
                if error > config.max_tracking_error_m:
                    raise RuntimeError(f"Tracking error {error * 1000:.3f} mm exceeds configured limit")
            prediction = worker.take()
            if prediction is not None:
                counts["predictions"] += 1
                latencies.append(prediction.received_at - prediction.observation.captured_at)
                accepted = buffer.accept(prediction)
                log(
                    "prediction",
                    request_id=prediction.request_id,
                    tick=tick,
                    observation_tick=prediction.observation.tick,
                    latency_s=latencies[-1],
                    inference_s=prediction.inference_s,
                    accepted=accepted,
                    actions=prediction.actions.tolist(),
                )
            dwelling = last_command is not None and last_command.kind == "gripper" and now < busy_until
            if not 0 <= observation.image_age_s <= 0.05:
                counts["stale_frames"] += 1
                if config.mode == "live":
                    raise RuntimeError("Live camera frame exceeded 50 ms age limit")
            elif not dwelling and observation.image_received_at >= fresh_image_after:
                worker.submit(policy_observation(observation))
            if now < busy_until:
                continue
            if config.mode != "live":
                # Recorded/live shadow feedback does not follow our hypothetical commands.
                # Assess each boundary at the current measured pose, not an old rejected pose.
                buffer.executor.position = state[:3].copy()
                buffer.executor.velocity = np.zeros(3)
                buffer.executor.acceleration = np.zeros(3)
            try:
                proposal = buffer.propose(tick)
                if proposal is None:
                    # Every target move ends at rest; wait there for the next result.
                    continue
                command, proposed_executor = proposal
                source_prediction = buffer.prediction
                if (
                    config.mode == "live"
                    and command.kind in {"gripper", "hold"}
                    and np.linalg.norm(buffer.executor.velocity) > 0.002
                ):
                    raise ValueError("Reference trajectory must stop before a gripper or hold command")
                if (
                    config.mode == "live"
                    and last_command is not None
                    and last_command.kind == "gripper"
                    and last_command.jaw_open
                    and abs(state[6] - 0.034) > 0.003
                ):
                    raise ValueError("Gripper did not reach its commanded open stroke")
                if command.kind == "cartesian" and bounds is not None:
                    validate_trajectory(command, *bounds)
            except ValueError as exc:
                counts["rejected_commands"] += 1
                log(
                    "rejected",
                    tick=tick,
                    request_id=buffer.prediction.request_id if buffer.prediction is not None else None,
                    reason=str(exc),
                )
                if config.mode == "live":
                    raise
                # Shadow/replay evaluate again from measured state after a rejection.
                buffer.prediction = None
                buffer.executor = PolicyExecutor(state[:3].copy(), jaw_open=bool(state[6] >= 0.03), available_tick=tick)
                continue
            command_started_at = time.monotonic()
            if config.mode == "live" and command_started_at - (epoch + tick / 30) > config.max_tick_lateness_s:
                log("dispatch_late", tick=tick, lateness_s=command_started_at - (epoch + tick / 30))
            if arm is not None and config.mode == "live":
                arm.dispatch(command, *bounds)
            buffer.commit(command, proposed_executor)
            last_command = command
            busy_until = command_started_at + command.ticks / 30
            counts["accepted_commands"] += 1
            log(
                "command",
                request_id=source_prediction.request_id,
                observation_tick=source_prediction.observation.tick,
                generation=source_prediction.observation.generation,
                tick=tick,
                kind=command.kind,
                ticks=command.ticks,
                dispatch_s=time.monotonic() - command_started_at,
                motion=config.mode == "live",
                target_xyz_m=buffer.executor.position.tolist(),
                jaw_open=command.jaw_open,
                gripper_alignment=proposed_executor.pending_jaw_open is not None,
                duration_s=command.ticks / 30,
            )
            if command.kind == "gripper":
                jaw_closed = not command.jaw_open
                if velocity_reference is not None:
                    velocity_reference.gripper_command(jaw_open=command.jaw_open)
                buffer.generation = worker.invalidate()
                fresh_image_after = busy_until
        status = "duration_reached"  # Time limit is not a declaration of task success.
    except MissedGraspError as exc:
        status = "missed_grasp"
        missed_grasp_stroke_m = exc.stroke_m
        release_on_stop = True
        return_home_on_stop = True
        log("missed_grasp", stroke_m=exc.stroke_m, minimum_m=exc.minimum_m, error=str(exc))
        logging.error("%s", exc)
        logging.info("Missed grasp: opening gripper, then returning to joint zero")
    except KeyboardInterrupt:
        status = "operator_stop"
        release_on_stop = config.mode == "live"
        log("operator_stop")
        logging.info("Operator stopped the policy; holding arm position and opening gripper")
    except BaseException as exc:
        status = "failed"
        log("failure", error=repr(exc))
        raise
    finally:
        # Stop and release before a missed-grasp return to joint zero.
        if arm is not None:
            cleanup_stage = "hold"
            try:
                if release_on_stop:
                    cleanup_stage = "gripper_release"
                    log("gripper_release_started", reason=status)
                    recovery = arm.stop_and_release()
                    gripper_released = True
                    log("gripper_release_finished", **recovery)
                    if return_home_on_stop:
                        cleanup_stage = "return_home"
                        log("return_home_started", reason=status)
                        logging.info("Recovery: returning to joint zero")
                        home_joints = arm.go_home()
                        returned_home = True
                        log("return_home_finished", joints_rad=home_joints)
                else:
                    arm.hold()
            except (Exception, KeyboardInterrupt) as exc:
                status = "cleanup_failed"
                log(f"{cleanup_stage}_failed", error=repr(exc))
                logging.exception("Could not finish robot cleanup; attempting a position hold")
                try:
                    arm.hold()
                except (Exception, KeyboardInterrupt) as hold_exc:
                    log("cleanup_failure", resource="hold", error=repr(hold_exc))
        for resource, method in (
            (worker, "close"),
            (camera, "close"),
            (arm, "close"),
            (episode, "close"),
        ):
            if resource is not None:
                try:
                    getattr(resource, method)()
                except (Exception, KeyboardInterrupt) as exc:
                    status = "cleanup_failed"
                    log("cleanup_failure", resource=method, error=repr(exc))
                    logging.exception("Cleanup failed; check the robot and use its external stop if needed")
        summary = {
            "status": status,
            "mode": config.mode,
            "execution_adapter": "endpoint_rest_to_rest_v2",
            "velocity_source": config.velocity_source,
            "velocity_reference_sha256": velocity_reference.sha256 if velocity_reference is not None else None,
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
            "max_tick_lateness_s": max(lateness, default=0.0),
            "task_success": False if missed_grasp_stroke_m is not None else None,
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
