"""Small ROS image and Trossen adapters; hardware imports are lazy."""

import dataclasses
import logging
import threading
import time

import cv2
import numpy as np
from openpi_client import hanoi

from examples.hanoi.deployment.execution import ARM_MOTION_TIME_SCALE
from examples.hanoi.deployment.execution import GRIPPER_CLOSE_S
from examples.hanoi.deployment.execution import LIMITS
from examples.hanoi.deployment.execution import Command

# User-specified rod-A setup: K4326 medoid 1296, observation 87660.
INITIAL_XYZ = np.array([0.492297590, -0.056030598, 0.191169396])
INITIAL_ROTVEC = np.array([0.0, 0.785398163, 0.0])
INITIAL_POSITION_TOLERANCE_M = 0.0005
INITIAL_ALIGNMENT_CORRECTIONS = 3
INITIAL_MAX_COMPENSATION_M = 0.005
INITIAL_JAW_M = 0.034  # Recorded open_fraction * max_stroke.
HOME_JOINTS = np.zeros(6)
# Collector's missed-grasp threshold: 20% of the calibrated 0.04 m max stroke.
MIN_GRASP_STROKE_M = 0.008


class MissedGraspError(RuntimeError):
    def __init__(self, stroke_m: float, minimum_m: float):
        self.stroke_m, self.minimum_m = float(stroke_m), float(minimum_m)
        super().__init__(
            f"Missed/slipped grasp or rod grip: jaw stroke {stroke_m * 1000:.2f} mm "
            f"is at/below {minimum_m * 1000:.2f} mm."
        )


def check_grasp(stroke_m: float, minimum_m: float = MIN_GRASP_STROKE_M):
    if stroke_m <= minimum_m:
        raise MissedGraspError(stroke_m, minimum_m)


@dataclasses.dataclass(frozen=True)
class Frame:
    rgb: np.ndarray
    received_at: float
    sequence: int
    header_time_s: float


class RosCamera:
    def __init__(self, topic: str = hanoi.CONTRACT["rgb_topic"]):
        import rclpy
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy
        from rclpy.qos import HistoryPolicy
        from rclpy.qos import QoSProfile
        from rclpy.qos import ReliabilityPolicy
        from sensor_msgs.msg import Image

        self.lock = threading.Lock()
        self.frame = None
        self.error = None
        self.sequence = 0
        self.context = Context()
        rclpy.init(context=self.context)
        self.node = Node("hanoi_policy_camera", context=self.context)
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.subscription = self.node.create_subscription(Image, topic, self._receive, qos)
        self.executor = SingleThreadedExecutor(context=self.context)
        self.executor.add_node(self.node)
        self.thread = threading.Thread(target=self.executor.spin, name="hanoi-camera", daemon=True)
        self.thread.start()

    def _receive(self, message):
        received = time.monotonic()  # Before decoding; never use a remote clock for freshness.
        try:
            rgb = hanoi.decode_ros_rgb(
                message.data, height=message.height, width=message.width, step=message.step, encoding=message.encoding
            )
            rgb = hanoi.preprocess_camera(rgb)
            stamp = message.header.stamp
            with self.lock:
                self.sequence += 1
                self.frame = Frame(rgb, received, self.sequence, stamp.sec + stamp.nanosec / 1e9)
        except Exception as exc:
            with self.lock:
                self.error = exc

    def latest(self) -> Frame:
        with self.lock:
            if self.error is not None:
                raise RuntimeError("Camera decode failed") from self.error
            if self.frame is None:
                raise RuntimeError("No camera frame received")
            return self.frame

    def close(self):
        self.executor.shutdown(timeout_sec=2)
        self.thread.join(timeout=2)
        self.node.destroy_node()
        self.context.shutdown()


def _critical_points(coefficients: np.ndarray) -> np.ndarray:
    roots = np.polynomial.polynomial.polyroots(coefficients)
    return np.array([0.0, 1.0, *[r.real for r in roots if abs(r.imag) < 1e-8 and 0 < r.real < 1]])


def validate_trajectory(command: Command, xyz_min: np.ndarray, xyz_max: np.ndarray, limits=LIMITS):
    """Check polynomial extrema, including between the offline builder's samples."""
    coefficients = command.coefficients.copy()
    for axis in range(3):
        points = _critical_points(np.polynomial.polynomial.polyder(coefficients[:, axis]))
        positions = command.sample(points)[:, axis]
        if positions.min() < xyz_min[axis] or positions.max() > xyz_max[axis]:
            raise ValueError(
                f"Trajectory leaves configured Cartesian workspace on {'XYZ'[axis]}: "
                f"range [{positions.min():.5f}, {positions.max():.5f}] m, "
                f"allowed [{xyz_min[axis]:.5f}, {xyz_max[axis]:.5f}] m"
            )
    duration = command.ticks / 30
    for order, limit in enumerate(limits, start=1):
        coefficients = np.polynomial.polynomial.polyder(coefficients, axis=0) / duration
        norm_squared = np.zeros(2 * len(coefficients) - 1)
        for c in coefficients.T:
            squared = np.polynomial.polynomial.polymul(c, c)
            norm_squared[: len(squared)] += squared
        points = _critical_points(np.polynomial.polynomial.polyder(norm_squared))
        # 2% tolerance: the recorded legs peak exactly at the limits, and a tracking segment fitted
        # between two at-limit boundary states can overshoot them by a hair.
        if np.linalg.norm(command.sample(points, order), axis=-1).max() > limit * 1.02:
            raise ValueError(
                f"Trajectory derivative-{order} peak "
                f"{np.linalg.norm(command.sample(points, order), axis=-1).max():.5f} exceeds limit {limit}"
            )


class TrossenArm:
    """WXAI base/tool Cartesian control, with no automatic home or error clear.

    Construction configures the driver without calling mode or motion setters.
    Motion is enabled by the explicit initialization command or the live client.
    """

    def __init__(self, ip: str, *, driver=None, api=None, initial_xyz=INITIAL_XYZ):
        if api is None:
            import trossen_arm as api
        self.api = api
        self.driver = driver if driver is not None else api.TrossenArmDriver()
        self.driver.configure(
            api.Model.wxai_v0, api.StandardEndEffector.wxai_v0_follower, ip, clear_error=False, timeout=5.0
        )
        self.enabled = False
        self.last_id = None
        self.last_changed = time.monotonic()
        self.orientation = INITIAL_ROTVEC.copy()  # This pure-pitch RPY equals angle-axis.
        # Measured tool orientation is required within this angle whenever motion is enabled.
        self.orientation_limit_deg = 2.0
        # Start pose for initialize()/verify_initial_pose(): the pi0.5 rod-A hover by default;
        # the Cosmos client passes its own recording's episode start.
        self.initial_xyz = np.array(initial_xyz, dtype=float)
        if self.initial_xyz.shape != (3,) or not np.isfinite(self.initial_xyz).all():
            raise ValueError(f"Start pose must be three finite metres, got {initial_xyz!r}")
        self.measured_orientation = None
        self.orientation_error_rad = None
        self.angular_velocity = None
        self.last_initial_pose_report = None
        self.initial_proprio_alignment = None

    def read(self, *, require_orientation: bool = True) -> np.ndarray:
        """Read measured state; shadow callers may inspect an uninitialized pose.

        Controller errors, invalid feedback, and stale feedback remain fatal in
        every mode. The orientation requirement is retained by default for motion.
        """
        return self._read(require_orientation=require_orientation)[0]

    def read_joints(self, *, require_orientation: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Measured state plus the six arm joint angles (rad, driver order 0-5), one snapshot.

        The Cosmos waypoint client observes joint angles instead of XYZ velocity;
        both come from the same RobotOutput so state and Cartesian context agree.
        """
        state, output = self._read(require_orientation=require_orientation)
        joints = np.asarray(output.joint.arm.positions, dtype=float)
        if joints.shape != (6,) or not np.isfinite(joints).all():
            raise ValueError("Invalid robot joint feedback")
        return state, joints

    def _read(self, *, require_orientation: bool):
        error = self.driver.get_error_information()
        if error != "No error":
            raise RuntimeError(f"Trossen controller error: {error}")
        output = self.driver.get_robot_output()
        now = time.monotonic()
        if output.header.id != self.last_id:
            self.last_id, self.last_changed = output.header.id, now
        if now - self.last_changed > 0.1:
            raise RuntimeError("Trossen feedback stopped updating")
        pose = np.asarray(output.cartesian.positions, dtype=float)
        velocity = np.asarray(output.cartesian.velocities, dtype=float)
        if pose.shape != (6,) or velocity.shape != (6,) or not np.isfinite(np.r_[pose, velocity]).all():
            raise ValueError("Invalid robot feedback")
        state = np.r_[pose[:3], velocity[:3], output.joint.gripper.position].astype(np.float32)
        if not np.isfinite(state).all():
            raise ValueError("Invalid robot feedback")
        self.measured_orientation = pose[3:].copy()
        self.angular_velocity = velocity[3:].copy()
        # Compare rotations, not rotation-vector coordinates: equivalent angle-axis
        # representations can differ by a full turn.
        measured_rotation = cv2.Rodrigues(self.measured_orientation)[0]
        expected_rotation = cv2.Rodrigues(self.orientation)[0]
        cosine = (np.trace(expected_rotation.T @ measured_rotation) - 1) / 2
        self.orientation_error_rad = float(np.arccos(np.clip(cosine, -1, 1)))
        if require_orientation and self.orientation_error_rad > np.deg2rad(self.orientation_limit_deg):
            raise ValueError(
                f"Tool orientation differs from dataset (0, pi/4, 0) by "
                f"{np.rad2deg(self.orientation_error_rad):.2f} degrees (maximum {self.orientation_limit_deg:g} degrees); "
                f"measured angle-axis radians: {self.measured_orientation.tolist()}. "
                "The arm must reach the dataset orientation before policy actions."
            )
        return state, output

    def verify_initial_pose(self, state: np.ndarray, *, jaw_open: bool = True) -> dict:
        """Check readback against the desired start pose, never its compensated command."""
        position_error = float(np.linalg.norm(state[:3] - self.initial_xyz))
        orientation_error = float(np.rad2deg(self.orientation_error_rad))
        speed = float(np.linalg.norm(state[3:6]))
        angular_speed = float(np.linalg.norm(self.angular_velocity))
        jaw_ok = abs(state[6] - INITIAL_JAW_M) <= 0.003 if jaw_open else state[6] < 0.03
        report = {
            "target_xyz_m": self.initial_xyz.tolist(),
            "measured_xyz_m": state[:3].tolist(),
            "target_rotvec_rad": self.orientation.tolist(),
            "measured_rotvec_rad": self.measured_orientation.tolist(),
            "target_jaw_m": INITIAL_JAW_M if jaw_open else None,
            "measured_jaw_m": float(state[6]),
            "position_error_mm": position_error * 1000,
            "position_tolerance_mm": INITIAL_POSITION_TOLERANCE_M * 1000,
            "orientation_error_deg": orientation_error,
            "linear_speed_m_s": speed,
            "angular_speed_rad_s": angular_speed,
        }
        if self.initial_proprio_alignment is not None:
            report["initial_proprio_alignment"] = self.initial_proprio_alignment
        self.last_initial_pose_report = report
        if position_error > INITIAL_POSITION_TOLERANCE_M or orientation_error > 2 or not jaw_ok:
            reason = (self.initial_proprio_alignment or {}).get("stop_reason", "")
            raise ValueError(
                f"Initial pose verification failed: XYZ error {position_error * 1000:.3f} mm "
                f"(limit {INITIAL_POSITION_TOLERANCE_M * 1000:g}), "
                f"orientation error {orientation_error:.3f} deg (limit 2), "
                f"jaw {state[6]:.5f} m (expected {'0.034 +/- 0.003' if jaw_open else '< 0.03'}), "
                f"measured XYZ {state[:3].tolist()}, rotation vector {self.measured_orientation.tolist()}. {reason}"
            )
        return report

    def go_home(self):
        """Return to the recorder's zero-joint home and check the result."""
        self.read(require_orientation=False)
        self.enable()
        self.driver.set_arm_positions(HOME_JOINTS.tolist(), 3.2 * ARM_MOTION_TIME_SCALE, blocking=True)
        time.sleep(0.2)
        self.read(require_orientation=False)
        joints = np.asarray(self.driver.get_arm_positions(), dtype=float)
        if (
            joints.shape != (6,)
            or not np.isfinite(joints).all()
            or np.max(np.abs(joints - HOME_JOINTS)) > np.deg2rad(2)
        ):
            raise ValueError(f"Arm did not reach the joint home pose: {joints.tolist()}")
        return joints.tolist()

    def initialize(self) -> dict:
        """Home, move to the start pose, open the jaw, and correct XYZ to within 0.5 mm."""
        self.last_initial_pose_report = None
        self.initial_proprio_alignment = None
        logging.info("Arm travel uses %sx durations (half speed)", ARM_MOTION_TIME_SCALE)
        state = self.read(require_orientation=False)
        if state[6] < INITIAL_JAW_M - 0.003:
            # Release a possible rod grip before joint homing on a new launch.
            self.stop_and_release()
        self.go_home()
        self.driver.set_cartesian_positions(
            np.r_[self.initial_xyz, self.orientation].tolist(),
            self.api.InterpolationSpace.cartesian,
            goal_time=2.104 * ARM_MOTION_TIME_SCALE,
            blocking=True,
            num_trajectory_check_samples=101,
        )
        self.driver.set_gripper_mode(self.api.Mode.position)
        self.driver.set_gripper_position(INITIAL_JAW_M, 1.0, blocking=True)
        time.sleep(0.5)
        command = self.initial_xyz.copy()
        alignment = {
            "target_recorded_proprio_xyz": self.initial_xyz.tolist(),
            "tolerance_mm": INITIAL_POSITION_TOLERANCE_M * 1000,
            "maximum_corrections": INITIAL_ALIGNMENT_CORRECTIONS,
            "maximum_compensation_mm": INITIAL_MAX_COMPENSATION_M * 1000,
            "corrections": [],
            "final_error_mm": None,
            "passed": False,
        }
        self.initial_proprio_alignment = alignment
        for attempt in range(INITIAL_ALIGNMENT_CORRECTIONS + 1):
            state = self.read(require_orientation=False)
            residual = self.initial_xyz - state[:3]
            error = float(np.linalg.norm(residual))
            alignment.update(final_error_mm=error * 1000, passed=error <= INITIAL_POSITION_TOLERANCE_M)
            if attempt:
                alignment["corrections"].append(
                    {
                        "attempt": attempt,
                        "command_xyz": command.tolist(),
                        "readback_xyz": state[:3].tolist(),
                        "residual_mm": error * 1000,
                    }
                )
            logging.info(
                "Initial alignment after %d corrections: %.3f mm from desired XYZ (limit %.1f mm)",
                attempt,
                error * 1000,
                INITIAL_POSITION_TOLERANCE_M * 1000,
            )
            if error <= INITIAL_POSITION_TOLERANCE_M:
                break
            if self.orientation_error_rad > np.deg2rad(2) or abs(state[6] - INITIAL_JAW_M) > 0.003:
                break  # XYZ correction cannot repair an orientation or jaw error.
            if attempt == INITIAL_ALIGNMENT_CORRECTIONS:
                alignment["stop_reason"] = "Reached the three-correction limit."
                break
            next_command = command + residual
            if np.linalg.norm(next_command - self.initial_xyz) > INITIAL_MAX_COMPENSATION_M:
                alignment["stop_reason"] = "Required cumulative XYZ compensation exceeds 5 mm."
                break
            command = next_command
            self.driver.set_cartesian_positions(
                np.r_[command, self.orientation].tolist(),
                self.api.InterpolationSpace.cartesian,
                goal_time=2.104 * ARM_MOTION_TIME_SCALE,
                blocking=True,
                num_trajectory_check_samples=101,
            )
            time.sleep(0.2)
        return self.verify_initial_pose(state)

    def enable(self):
        if self.enabled:
            return
        # A partially failed mode switch also needs a best-effort stop attempt.
        self.enabled = True
        self.driver.set_arm_modes(self.api.Mode.position)

    def dispatch(self, command: Command, xyz_min: np.ndarray, xyz_max: np.ndarray, limits=LIMITS):
        if not self.enabled:
            raise RuntimeError("Arm motion has not been enabled")
        if command.kind == "cartesian":
            validate_trajectory(command, xyz_min, xyz_max, limits)
            endpoint = command.sample(np.array([1.0]))[0]
            velocity = command.sample(np.array([1.0]), 1)[0]
            acceleration = command.sample(np.array([1.0]), 2)[0]
            self.driver.set_cartesian_positions(
                np.r_[endpoint, self.orientation].tolist(),
                self.api.InterpolationSpace.cartesian,
                goal_time=command.ticks / 30,
                blocking=False,
                goal_feedforward_velocities=np.r_[velocity, np.zeros(3)].tolist(),
                goal_feedforward_accelerations=np.r_[acceleration, np.zeros(3)].tolist(),
                num_trajectory_check_samples=101,
            )
        elif command.kind == "gripper":
            if command.jaw_open:
                self.driver.set_gripper_mode(self.api.Mode.position)
                self.driver.set_gripper_position(0.034, 1.0, blocking=False)
            else:
                self.driver.set_gripper_mode(self.api.Mode.external_effort)
                self.driver.set_gripper_external_effort(-20.0, GRIPPER_CLOSE_S, blocking=False)
        elif command.kind not in {"hold", "wait"}:
            raise ValueError(f"Unknown command kind: {command.kind}")

    def hold(self, *, wait: bool = True):
        """Best-effort software stop; preserves gripper mode/effort.

        This is a joint position hold, not an independent emergency stop. It
        requires a working driver/connection and must be commissioned on hardware.
        """
        if self.enabled:
            self.driver.set_arm_positions(self.driver.get_arm_positions(), 0.3, blocking=False)
            if wait:
                time.sleep(0.35)  # Let the driver's background loop finish before cleanup.
                self.enabled = False

    def stop_and_release(self) -> dict:
        """Hold the current arm position and open the gripper in place.

        Uses robot feedback only; no lift or return-home trajectory is sent.
        """
        self.read(require_orientation=False)
        self.enable()
        # Release immediately while the arm's short hold command completes.
        self.hold(wait=False)
        self.read(require_orientation=False)
        logging.info("Recovery: holding arm position and opening gripper")
        self.driver.set_gripper_mode(self.api.Mode.position)
        self.driver.set_gripper_position(INITIAL_JAW_M, 1.0, blocking=True)
        state = self.read(require_orientation=False)
        if abs(state[6] - INITIAL_JAW_M) > 0.003:
            raise RuntimeError(f"Recovery stopped: gripper did not open (stroke {state[6] * 1000:.2f} mm)")
        return {"released_jaw_m": float(state[6]), "stopped_xyz_m": state[:3].tolist()}

    def close(self):
        self.driver.cleanup()
