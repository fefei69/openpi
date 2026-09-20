"""Initialization and live startup checks with a fake driver; never connect hardware."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from examples.hanoi.deployment import client
from examples.hanoi.deployment import hardware
from examples.hanoi.deployment import initialize


@pytest.fixture
def robot(monkeypatch):
    clock = SimpleNamespace(now=0.0)

    def sleep(seconds):
        clock.now += seconds

    monkeypatch.setattr(hardware, "time", SimpleNamespace(monotonic=lambda: clock.now, sleep=sleep))
    driver = Mock()
    driver.get_error_information.return_value = "No error"
    output = SimpleNamespace(
        header=SimpleNamespace(id=0),
        cartesian=SimpleNamespace(positions=np.array([0.3, 0, 0.25, 0, 0, 0]), velocities=np.zeros(6)),
        joint=SimpleNamespace(gripper=SimpleNamespace(position=0.034)),
    )
    joints = np.zeros(6)

    def feedback():
        output.header.id += 1
        return output

    def arm_positions(target, goal_time, *, blocking):
        joints[:] = target
        if np.all(joints == 0):
            output.cartesian.positions[:] = [0.3, 0, 0.25, 0, 0, 0]
        if blocking:
            clock.now += goal_time

    def cartesian_positions(target, _space, *, goal_time, blocking, **_kwargs):
        output.cartesian.positions[:] = target
        joints[:] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        if blocking:
            clock.now += goal_time

    def gripper_position(target, goal_time, *, blocking):
        output.joint.gripper.position = target
        if blocking:
            clock.now += goal_time

    driver.get_robot_output.side_effect = feedback
    driver.get_arm_positions.side_effect = lambda: joints.copy()
    driver.set_arm_positions.side_effect = arm_positions
    driver.set_cartesian_positions.side_effect = cartesian_positions
    driver.set_gripper_position.side_effect = gripper_position
    arm = hardware.TrossenArm("unused", driver=driver, api=Mock())
    monkeypatch.setattr(initialize, "TrossenArm", lambda ip, **kwargs: arm)
    return arm, driver, output, clock


def read_report(tmp_path):
    return json.loads(next(tmp_path.glob("initial_pose_*.json")).read_text())


def test_pose_test_records_target_readback_then_returns_home(tmp_path, robot):
    _, driver, output, clock = robot
    result = initialize.main(initialize.Config(output=tmp_path))
    assert result["status"] == "passed"
    assert result["returned_home"] is True
    np.testing.assert_array_equal(hardware.INITIAL_XYZ, [0.492297590, -0.056030598, 0.191169396])
    np.testing.assert_allclose(result["measured_xyz_m"], hardware.INITIAL_XYZ)
    assert result["orientation_error_deg"] == pytest.approx(0)
    assert result["measured_jaw_m"] == pytest.approx(0.034)
    assert result["initial_proprio_alignment"]["corrections"] == []
    assert result["initial_proprio_alignment"]["passed"] is True
    assert result["initial_proprio_alignment"]["final_error_mm"] <= 0.5
    assert clock.now == pytest.approx(6.4 + 0.2 + 4.208 + 1.0 + 0.5 + 6.4 + 0.2)
    assert [call[0] for call in driver.method_calls if call[0].startswith("set_")] == [
        "set_arm_modes",
        "set_arm_positions",
        "set_cartesian_positions",
        "set_gripper_mode",
        "set_gripper_position",
        "set_arm_positions",
    ]
    assert driver.set_arm_positions.call_args_list[-1].args[0] == [0.0] * 6
    assert driver.set_arm_positions.call_args_list[-1].args[1] == 6.4
    assert driver.set_arm_positions.call_args_list[-1].kwargs["blocking"] is True
    assert driver.set_cartesian_positions.call_args.kwargs["goal_time"] == 4.208
    assert driver.set_cartesian_positions.call_args.kwargs["blocking"] is True
    assert driver.set_gripper_position.call_args.kwargs["blocking"] is True
    driver.cleanup.assert_called_once()
    assert read_report(tmp_path) == result
    np.testing.assert_allclose(output.cartesian.positions, [0.3, 0, 0.25, 0, 0, 0])


def test_velocity_noise_does_not_fail_a_correct_pose(tmp_path, robot):
    _, _, output, _ = robot
    output.cartesian.velocities[:] = [0.005, -0.003, 0.002, 0.04, -0.04, 0.02]
    result = initialize.main(initialize.Config(output=tmp_path))
    assert result["status"] == "passed"
    assert result["linear_speed_m_s"] > 0.002
    assert result["angular_speed_rad_s"] > 0.02
    assert result["returned_home"] is True


@pytest.mark.parametrize("offset_m", [0.0004, 0.0012])
def test_alignment_compensates_readback_bias_only_when_needed(tmp_path, robot, offset_m):
    _, driver, output, _ = robot
    original = driver.set_cartesian_positions.side_effect

    def biased_cartesian(target, *args, **kwargs):
        original(target, *args, **kwargs)
        output.cartesian.positions[0] += offset_m

    driver.set_cartesian_positions.side_effect = biased_cartesian
    result = initialize.main(initialize.Config(output=tmp_path))
    alignment = result["initial_proprio_alignment"]
    expected_corrections = int(offset_m > 0.0005)
    assert len(alignment["corrections"]) == expected_corrections
    assert driver.set_cartesian_positions.call_count == 1 + expected_corrections
    assert alignment["final_error_mm"] <= 0.5
    assert alignment["passed"] is True
    assert result["returned_home"] is True
    np.testing.assert_array_equal(result["target_xyz_m"], hardware.INITIAL_XYZ)
    for call in driver.set_cartesian_positions.call_args_list:
        np.testing.assert_array_equal(call.args[0][3:], hardware.INITIAL_ROTVEC)
        assert call.kwargs["blocking"] is True
        assert call.kwargs["goal_time"] == 4.208
        assert call.kwargs["num_trajectory_check_samples"] == 101
    if expected_corrections:
        corrected = driver.set_cartesian_positions.call_args.args[0][:3]
        np.testing.assert_allclose(corrected, hardware.INITIAL_XYZ - [offset_m, 0, 0], atol=3e-8)
        setters = [call[0] for call in driver.method_calls if call[0].startswith("set_")]
        assert setters.index("set_gripper_position") < len(setters) - 1 - setters[::-1].index("set_cartesian_positions")


def test_alignment_accumulates_corrections_against_original_desired_pose(robot):
    arm, driver, output, _ = robot
    original = driver.set_cartesian_positions.side_effect

    def partial_response(target, *args, **kwargs):
        original(target, *args, **kwargs)
        output.cartesian.positions[:3] = (
            hardware.INITIAL_XYZ + 0.5 * (np.asarray(target[:3]) - hardware.INITIAL_XYZ) + [0.0012, 0, 0]
        )

    driver.set_cartesian_positions.side_effect = partial_response
    report = arm.initialize()
    alignment = report["initial_proprio_alignment"]
    assert len(alignment["corrections"]) == 2
    assert alignment["final_error_mm"] == pytest.approx(0.3, abs=0.0001)
    assert alignment["passed"] is True
    commands = driver.set_cartesian_positions.call_args_list
    np.testing.assert_allclose(commands[1].args[0][:3], hardware.INITIAL_XYZ - [0.0012, 0, 0], atol=3e-8)
    np.testing.assert_allclose(commands[2].args[0][:3], hardware.INITIAL_XYZ - [0.0018, 0, 0], atol=3e-8)
    assert np.linalg.norm(np.array(report["measured_xyz_m"]) - commands[-1].args[0][:3]) > 0.0005


@pytest.mark.parametrize(
    ("offset_m", "corrections", "reason"), [(0.001, 3, "three-correction"), (0.002, 2, "exceeds 5 mm")]
)
def test_alignment_stops_at_correction_or_compensation_limit(tmp_path, robot, offset_m, corrections, reason):
    _, driver, output, _ = robot
    original = driver.set_cartesian_positions.side_effect

    def no_correction_response(target, *args, **kwargs):
        original(target, *args, **kwargs)
        output.cartesian.positions[:3] = hardware.INITIAL_XYZ + np.array([offset_m, 0, 0])

    driver.set_cartesian_positions.side_effect = no_correction_response
    with pytest.raises(ValueError, match=reason):
        initialize.main(initialize.Config(output=tmp_path))
    report = read_report(tmp_path)
    alignment = report["initial_proprio_alignment"]
    assert report["returned_home"] is True
    assert alignment["passed"] is False
    assert alignment["final_error_mm"] == pytest.approx(offset_m * 1000, abs=0.0001)
    assert len(alignment["corrections"]) == corrections
    assert driver.set_cartesian_positions.call_count == corrections + 1
    for call in driver.set_cartesian_positions.call_args_list:
        assert np.linalg.norm(np.asarray(call.args[0][:3]) - hardware.INITIAL_XYZ) <= 0.005


@pytest.mark.parametrize("failure", ["position", "orientation", "gripper"])
def test_wrong_pose_fails_but_still_returns_home(tmp_path, robot, failure):
    _, driver, output, _ = robot
    original = driver.set_gripper_position.side_effect

    def inject_failure(*args, **kwargs):
        original(*args, **kwargs)
        if failure == "position":
            output.cartesian.positions[0] += 0.01
        elif failure == "orientation":
            output.cartesian.positions[4] = 0
        else:
            output.joint.gripper.position = 0

    driver.set_gripper_position.side_effect = inject_failure
    with pytest.raises(ValueError, match="Initial pose verification failed"):
        initialize.main(initialize.Config(output=tmp_path))
    report = read_report(tmp_path)
    assert report["status"] == "failed"
    assert report["returned_home"] is True
    assert "measured_xyz_m" in report
    assert "Initial pose verification failed" in report["error"]
    assert driver.set_cartesian_positions.call_count == 1
    assert driver.set_arm_positions.call_args_list[-1].args[0] == [0.0] * 6
    driver.cleanup.assert_called_once()


def test_failed_home_does_not_send_cartesian_goal(tmp_path, robot):
    _, driver, _, _ = robot
    driver.get_arm_positions.side_effect = lambda: np.ones(6)
    with pytest.raises(ValueError, match="joint home pose"):
        initialize.main(initialize.Config(output=tmp_path))
    driver.set_cartesian_positions.assert_not_called()
    driver.set_gripper_position.assert_not_called()
    driver.cleanup.assert_called_once()


def test_controller_fault_holds_instead_of_starting_a_return_move(tmp_path, robot):
    failure = RuntimeError("controller error")
    _, driver, _, _ = robot
    driver.set_cartesian_positions.side_effect = failure
    with pytest.raises(type(failure)):
        initialize.main(initialize.Config(output=tmp_path))
    driver.set_gripper_position.assert_not_called()
    assert driver.set_arm_positions.call_count == 2  # home, then a short hold
    assert driver.set_arm_positions.call_args.args[1] == 0.3
    driver.cleanup.assert_called_once()
    assert read_report(tmp_path)["returned_home"] is False


def test_fault_before_initialization_never_enables_arm(tmp_path, robot):
    _, driver, _, _ = robot
    driver.get_error_information.return_value = "Controller fault"
    with pytest.raises(RuntimeError, match="Controller fault"):
        initialize.main(initialize.Config(output=tmp_path))
    assert not [call for call in driver.method_calls if call[0].startswith("set_")]
    driver.cleanup.assert_called_once()


def test_cleanup_failure_does_not_report_pass(tmp_path, robot):
    _, driver, _, _ = robot
    driver.cleanup.side_effect = RuntimeError("cleanup failed")
    with pytest.raises(RuntimeError, match="cleanup failed"):
        initialize.main(initialize.Config(output=tmp_path))
    assert read_report(tmp_path)["status"] == "failed"


def test_failed_return_home_does_not_report_pass(tmp_path, robot):
    _, driver, _, _ = robot
    original = driver.set_arm_positions.side_effect

    def refuse_return(target, goal_time, **kwargs):
        if driver.set_arm_positions.call_count != 2:
            original(target, goal_time, **kwargs)

    driver.set_arm_positions.side_effect = refuse_return
    with pytest.raises(ValueError, match="joint home pose"):
        initialize.main(initialize.Config(output=tmp_path))
    report = read_report(tmp_path)
    assert report["status"] == "failed"
    assert report["returned_home"] is False
    assert driver.set_arm_positions.call_args.args[1] == 0.3  # fallback hold
    driver.cleanup.assert_called_once()


def test_live_client_initializes_from_home_before_policy(tmp_path, monkeypatch, robot):
    arm, driver, output, _ = robot
    camera = Mock()
    camera.latest.side_effect = lambda: SimpleNamespace(
        rgb=np.zeros((224, 224, 3), dtype=np.uint8), received_at=client.time.monotonic()
    )
    policy = Mock(side_effect=RuntimeError("offline policy stub reached"))
    monkeypatch.setattr(client, "TrossenArm", lambda ip, **kwargs: arm)
    monkeypatch.setattr(client, "RosCamera", lambda topic: camera)
    monkeypatch.setattr(client, "WebSocketPolicy", policy)
    config = client.Config(output=tmp_path)
    assert config.mode == "live"
    assert config.duration_s == 30
    with pytest.raises(RuntimeError, match="offline policy stub reached"):
        client.main(config)
    policy.assert_called_once()
    driver.set_cartesian_positions.assert_called_once()
    np.testing.assert_allclose(output.cartesian.positions[:3], hardware.INITIAL_XYZ)
    # Client cleanup holds the current pose; it doesn't run the test's return home.
    np.testing.assert_allclose(driver.set_arm_positions.call_args.args[0], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    camera.close.assert_called_once()
    driver.cleanup.assert_called_once()


def test_live_client_rejects_unaligned_pose_before_inference(tmp_path, monkeypatch, robot):
    arm, driver, output, _ = robot
    original = driver.set_cartesian_positions.side_effect

    def no_correction_response(target, *args, **kwargs):
        original(target, *args, **kwargs)
        output.cartesian.positions[:3] = hardware.INITIAL_XYZ + np.array([0.001, 0, 0])

    driver.set_cartesian_positions.side_effect = no_correction_response
    camera, policy = Mock(), Mock()
    monkeypatch.setattr(client, "TrossenArm", lambda ip, **kwargs: arm)
    monkeypatch.setattr(client, "RosCamera", lambda topic: camera)
    monkeypatch.setattr(client, "WebSocketPolicy", policy)
    with pytest.raises(ValueError, match="three-correction limit"):
        client.main(client.Config(output=tmp_path))
    policy.assert_not_called()
    camera.latest.assert_not_called()
    summary = json.loads(next(tmp_path.glob("live_*/summary.json")).read_text())
    assert summary["status"] == "failed"
    assert summary["accepted_commands"] == 0
    assert summary["initial_proprio_alignment"]["passed"] is False
    assert summary["initial_proprio_alignment"]["final_error_mm"] == pytest.approx(1.0, abs=0.0001)
    driver.cleanup.assert_called_once()
    camera.close.assert_called_once()


@pytest.mark.parametrize(
    ("kind", "reference_moving"),
    [
        ("hold", False),
        ("gripper", False),
        ("gripper", True),
        ("cartesian", False),
        ("gripper_alignment", False),
        ("late", False),
        ("miss", False),
        ("miss_recorded_velocity", False),
        ("miss_release_failure", False),
        ("miss_home_failure", False),
        ("slip", False),
        ("interrupt", False),
    ],
)
def test_live_loop_with_nonzero_velocity_feedback(tmp_path, monkeypatch, robot, kind, reference_moving):
    import dataclasses

    from examples.hanoi.deployment.async_inference import Prediction

    arm, driver, output, clock = robot
    # Actual startup velocity from the reported run: norm 2.43 mm/s.
    velocity = np.array([-0.0019453523, 0.0002347865, -0.0014328236])
    output.cartesian.velocities[:3] = velocity
    monkeypatch.setattr(
        client,
        "time",
        SimpleNamespace(
            monotonic=hardware.time.monotonic,
            sleep=hardware.time.sleep,
            time_ns=lambda: 1,
        ),
    )
    camera = Mock()
    closed_at = None
    recovering = False
    original_open = driver.set_gripper_position.side_effect
    original_feedback = driver.get_robot_output.side_effect
    original_arm_positions = driver.set_arm_positions.side_effect

    def close_gripper(*args, **kwargs):
        nonlocal closed_at
        if kind == "gripper_alignment":
            np.testing.assert_allclose(
                output.cartesian.positions[:3], hardware.INITIAL_XYZ + np.array([0, -0.0007, -0.0008]), atol=1e-7
            )
        closed_at = clock.now
        output.joint.gripper.position = 0.0049 if kind.startswith("miss") else 0.015

    def open_gripper(*args, **kwargs):
        nonlocal recovering
        if closed_at is not None:
            recovering = True
            if kind == "miss_release_failure":
                return  # The readback stays closed: homing must not start.
        original_open(*args, **kwargs)

    def arm_positions(target, *args, **kwargs):
        if kind == "miss_home_failure" and closed_at is not None and np.all(np.asarray(target) == 0):
            raise RuntimeError("Home command failed")
        original_arm_positions(target, *args, **kwargs)

    def feedback():
        if kind == "slip" and closed_at is not None and not recovering and clock.now - closed_at > 3.0:
            output.joint.gripper.position = 0.007784731686115265
        return original_feedback()

    driver.set_gripper_external_effort.side_effect = close_gripper
    driver.set_gripper_position.side_effect = open_gripper
    driver.get_robot_output.side_effect = feedback
    driver.set_arm_positions.side_effect = arm_positions

    def frame():
        if kind == "interrupt" and closed_at is not None:
            import signal

            signal.raise_signal(signal.SIGINT)
        if kind == "late":
            clock.now += 0.01
        return SimpleNamespace(rgb=np.zeros((224, 224, 3), dtype=np.uint8), received_at=clock.now)

    camera.latest.side_effect = frame
    policy = Mock()
    policy.metadata = {
        "hanoi_deployment": {
            "config_name": "pi05_hanoi_aaaa_to_cccc",
            "export_sha256": "38b1e39be734df97a49836f8ede9238f1e97335af724e4a8448a50c9bf5b4b88",
            "normalization_sha256": "ec82035dd64b6715d75644addc7b040bb4d347ec3a3f79b730ba7930590c1c48",
            "num_steps": 10,
        }
    }

    def infer(data):
        if kind == "gripper_alignment":
            target = data["observation/state"][:3] + [0, -0.0007, -0.0008]
            return {"actions": np.tile(np.r_[target, 0.0], (63, 1))}
        if kind == "cartesian":
            fixture = json.loads((Path(__file__).parent / "fixtures/first_live_prediction.json").read_text())
            actions = np.array(fixture["actions"])
            # Exercise the recorded small movement relative to the new setup pose.
            actions[:, :3] += data["observation/state"][:3] - np.array(fixture["state"])[:3]
            return {"actions": actions}
        return {"actions": np.tile(np.r_[data["observation/state"][:3], float(kind in {"hold", "late"})], (63, 1))}

    policy.infer.side_effect = infer

    class ImmediateWorker:
        def __init__(self, policy, *, record_dir=None):
            self.policy, self.result, self.generation = policy, None, 0
            self.request_id = 0

        def submit(self, observation):
            observation = dataclasses.replace(observation, generation=self.generation)
            self.result = Prediction(
                observation, self.policy.infer(observation.data)["actions"], clock.now, 0, self.request_id
            )
            self.request_id += 1

        def take(self):
            result, self.result = self.result, None
            return result

        def invalidate(self):
            self.generation += 1
            self.result = None
            return self.generation

        def close(self):
            self.policy.close()

    monkeypatch.setattr(client, "TrossenArm", lambda ip, **kwargs: arm)
    monkeypatch.setattr(client, "RosCamera", lambda topic: camera)
    monkeypatch.setattr(client, "WebSocketPolicy", lambda *args, **kwargs: policy)
    monkeypatch.setattr(client, "InferenceWorker", ImmediateWorker)
    if reference_moving:
        factory = client.reference_executor

        def moving_reference(*args, **kwargs):
            executor = factory(*args, **kwargs)
            executor.velocity[0] = 0.01
            return executor

        monkeypatch.setattr(client, "reference_executor", moving_reference)
    duration_s = {"slip": 3.4, "gripper_alignment": 0.9}.get(kind, 0.1)
    config = client.Config(output=tmp_path, duration_s=duration_s)
    if kind == "miss_recorded_velocity":
        from examples.hanoi.deployment.recorded_velocity import RecordedVelocity

        recorded_states = np.tile(np.r_[hardware.INITIAL_XYZ, [0.04, 0.01, -0.05], 0.034], (3, 1))
        recorded_states[2, 2] += 0.0001
        recorded_actions = np.c_[recorded_states[:, :3], [1, 0, 0]]
        monkeypatch.setattr(
            client.RecordedVelocity, "load", lambda path: RecordedVelocity(recorded_states, recorded_actions)
        )
        config.velocity_source = "recorded"
    if kind in {"miss_release_failure", "miss_home_failure"}:
        with pytest.raises(RuntimeError, match="Deployment cleanup failed"):
            client.main(config)
        summary = json.loads(next(tmp_path.glob("live_*/summary.json")).read_text())
        assert summary["status"] == "cleanup_failed"
        assert summary["gripper_released"] is (kind == "miss_home_failure")
        assert summary["return_home_requested"] is True
        assert summary["returned_home"] is False
        assert summary["task_success"] is False
        assert driver.set_arm_positions.call_args.args[1] == 0.3  # Hold after cleanup failure.
        homes = [call for call in driver.set_arm_positions.call_args_list if np.all(np.asarray(call.args[0]) == 0)]
        assert len(homes) == (2 if kind == "miss_home_failure" else 1)  # Includes startup home.
        events = [json.loads(line) for line in next(tmp_path.glob("live_*/events.jsonl")).read_text().splitlines()]
        expected = "return_home_failed" if kind == "miss_home_failure" else "gripper_release_failed"
        assert any(row["event"] == expected for row in events)
        if kind == "miss_release_failure":
            assert not any(row["event"] == "return_home_started" for row in events)
    elif kind in {"miss", "slip", "interrupt", "miss_recorded_velocity"}:
        client.main(config)
        summary = json.loads(next(tmp_path.glob("live_*/summary.json")).read_text())
        assert summary["status"] == ("operator_stop" if kind == "interrupt" else "missed_grasp")
        assert summary["return_home_requested"] is (kind != "interrupt")
        assert summary["returned_home"] is (kind != "interrupt")
        assert summary["gripper_released"] is True
        if kind == "interrupt":
            assert driver.set_arm_positions.call_args.args[1] == 0.3
        else:
            assert driver.set_arm_positions.call_args.args[0] == [0.0] * 6
            assert driver.set_arm_positions.call_args.kwargs["blocking"] is True
            setters = [call for call in driver.method_calls if call[0].startswith("set_")]
            assert setters[-2][0] == "set_gripper_position"
            assert setters[-1][0] == "set_arm_positions"
            assert summary["task_success"] is False
        assert driver.set_cartesian_positions.call_count == 1  # Initial setup only.
        events = [json.loads(line) for line in next(tmp_path.glob("live_*/events.jsonl")).read_text().splitlines()]
        stop = next(i for i, row in enumerate(events) if row["event"] in {"missed_grasp", "operator_stop"})
        assert not any(row["event"] == "command" for row in events[stop:])
        if kind != "interrupt":
            recovery_events = [row["event"] for row in events[stop:]]
            assert recovery_events == [
                "missed_grasp",
                "gripper_release_started",
                "gripper_release_finished",
                "return_home_started",
                "return_home_finished",
            ]
        if kind == "slip":
            assert summary["accepted_commands"] > 1  # check also runs after the closing dwell
    elif reference_moving:
        with pytest.raises(ValueError, match="Reference trajectory must stop"):
            client.main(config)
        driver.set_gripper_external_effort.assert_not_called()
    else:
        client.main(config)
        summary = json.loads(next(tmp_path.glob("live_*/summary.json")).read_text())
        assert summary["status"] == "duration_reached"
        assert summary["accepted_commands"] == (2 if kind == "gripper_alignment" else 1)
        if kind in {"gripper", "gripper_alignment"}:
            driver.set_gripper_external_effort.assert_called_once_with(-20.0, 2.4, blocking=False)
        events = [json.loads(line) for line in next(tmp_path.glob("live_*/events.jsonl")).read_text().splitlines()]
        if kind == "gripper_alignment":
            commands = [event for event in events if event["event"] == "command"]
            assert [event["kind"] for event in commands] == ["cartesian", "gripper"]
            assert [event["gripper_alignment"] for event in commands] == [True, False]
            assert commands[0]["request_id"] == commands[1]["request_id"]
            assert commands[0]["target_xyz_m"] == commands[1]["target_xyz_m"]
            assert commands[1]["tick"] >= commands[0]["tick"] + commands[0]["ticks"]
            assert commands[1]["jaw_open"] is False
        if kind == "cartesian":
            assert driver.set_cartesian_positions.call_count == 2  # initialization plus learned target
            assert driver.set_cartesian_positions.call_args.kwargs["blocking"] is False
        if kind == "late":
            assert any(event["event"] == "dispatch_late" for event in events)
        reference = next(event for event in events if event["event"] == "reference_initialized")
        np.testing.assert_array_equal(reference["reference_velocity_m_s"], np.zeros(3))
        np.testing.assert_array_equal(reference["reference_acceleration_m_s2"], np.zeros(3))
    for call in policy.infer.call_args_list:
        expected_velocity = [0.04, 0.01, -0.05] if kind == "miss_recorded_velocity" else velocity
        np.testing.assert_allclose(call.args[0]["observation/state"][3:6], expected_velocity)
    if kind == "miss_recorded_velocity":
        events = [json.loads(line) for line in next(tmp_path.glob("live_*/events.jsonl")).read_text().splitlines()]
        for event in events:
            if event["event"] == "tick":
                np.testing.assert_allclose(event["state"][3:6], velocity)  # Feedback remains measured.
        assert summary["velocity_source"] == "recorded"
        assert summary["returned_home"] is True  # Guard still uses the real jaw, not the reference jaw.
    driver.cleanup.assert_called_once()
    policy.close.assert_called_once()


def test_saved_run_detects_delayed_grasp_slip():
    fixture = json.loads((Path(__file__).parent / "fixtures/missed_grasp_feedback.json").read_text())
    for row in fixture["samples"]:
        if row["tick"] < fixture["first_missed_tick"]:
            hardware.check_grasp(row["stroke_m"])
        else:
            with pytest.raises(hardware.MissedGraspError):
                hardware.check_grasp(row["stroke_m"])


def prepare_low_pose(robot):
    arm, driver, output, _ = robot
    driver.set_cartesian_positions(
        [0.496, -0.056, 0.09, 0, np.pi / 4, 0],
        arm.api.InterpolationSpace.cartesian,
        goal_time=1.0,
        blocking=True,
    )
    output.joint.gripper.position = 0.0049
    arm.enable()
    driver.reset_mock()
    return arm, driver, output


def test_recovery_stops_and_releases_in_place(robot):
    arm, driver, output = prepare_low_pose(robot)
    before = output.cartesian.positions.copy()
    report = arm.stop_and_release()
    setters = [call[0] for call in driver.method_calls if call[0].startswith("set_")]
    assert setters == [
        "set_arm_positions",
        "set_gripper_mode",
        "set_gripper_position",
    ]
    driver.set_cartesian_positions.assert_not_called()
    np.testing.assert_allclose(output.cartesian.positions, before)
    assert driver.set_arm_positions.call_count == 1
    np.testing.assert_allclose(driver.set_arm_positions.call_args.args[0], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert report["released_jaw_m"] == pytest.approx(0.034)
    assert driver.set_arm_positions.call_args_list[0].args[1] == 0.3  # Prompt stop before release.
    driver.set_gripper_position.assert_called_once_with(0.034, 1.0, blocking=True)


def test_failed_release_keeps_arm_in_place(robot):
    arm, driver, _ = prepare_low_pose(robot)
    driver.set_gripper_position.side_effect = lambda *args, **kwargs: None
    with pytest.raises(RuntimeError, match="gripper did not open"):
        arm.stop_and_release()
    driver.set_cartesian_positions.assert_not_called()
    assert driver.set_arm_positions.call_count == 1  # initial hold only


def test_ctrl_c_in_initialization_test_stops_in_place(tmp_path, robot):
    _, driver, _, _ = robot
    original = driver.set_cartesian_positions.side_effect

    def interrupt_once(*args, **kwargs):
        if driver.set_cartesian_positions.call_count == 1:
            raise KeyboardInterrupt
        return original(*args, **kwargs)

    driver.set_cartesian_positions.side_effect = interrupt_once
    result = initialize.main(initialize.Config(output=tmp_path))
    assert result["status"] == "operator_stop"
    assert result["returned_home"] is False
    assert result["released_jaw_m"] == pytest.approx(0.034)
    assert driver.set_arm_positions.call_count == 2  # Startup home, then hold on Ctrl-C.
    assert driver.set_arm_positions.call_args.args[1] == 0.3
    driver.cleanup.assert_called_once()


def test_startup_releases_previous_rod_grip_before_homing(robot):
    arm, driver, _ = prepare_low_pose(robot)
    arm.initialize()
    calls = [call for call in driver.method_calls if call[0].startswith("set_")]
    opened = next(i for i, call in enumerate(calls) if call[0] == "set_gripper_position")
    setup = next(i for i, call in enumerate(calls) if call[0] == "set_cartesian_positions")
    homed = next(i for i, call in enumerate(calls) if call[0] == "set_arm_positions" and call.args[1] == 6.4)
    assert opened < homed < setup
    assert driver.set_cartesian_positions.call_count == 1  # Setup only; no recovery lift.
    np.testing.assert_allclose(arm.read()[:3], hardware.INITIAL_XYZ)
