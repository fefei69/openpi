"""CPU tests for the Cosmos waypoint client: planning, contract check, transport, snapshots."""

import json
import math
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import h5py
import numpy as np
import pytest
from websockets.sync.server import serve

from examples.hanoi.deployment import cosmos_client
from examples.hanoi.deployment import hardware
from examples.hanoi.deployment.cosmos_client import EXPECTED_CONTRACT
from examples.hanoi.deployment.cosmos_client import PROMPT
from examples.hanoi.deployment.cosmos_client import CosmosWebSocketPolicy
from examples.hanoi.deployment.cosmos_client import ReplayEpisode
from examples.hanoi.deployment.cosmos_client import check_contract
from examples.hanoi.deployment.cosmos_client import gripper_command
from examples.hanoi.deployment.cosmos_client import plan_waypoint
from examples.hanoi.deployment.cosmos_client import snap_destination
from examples.hanoi.deployment.cosmos_client import stop_actions
from openpi_client import msgpack_numpy


GRID = [  # a subset of the waypoint_v4 grid, metres
    [0.4926, -0.0562, 0.1911], [0.4961, -0.0572, 0.1911], [0.4926, -0.0562, 0.1506],
    [0.4961, -0.0572, 0.0877], [0.4961, -0.0572, 0.0775], [0.4961, -0.0572, 0.0678],
    [0.4920, 0.0137, 0.1911], [0.4920, 0.0137, 0.1501], [0.4950, 0.0137, 0.0775],
]


def metadata(family="cosmos", **overrides):
    identity = {
        "contract": {**EXPECTED_CONTRACT, "recording": "hanoi_wm_roundtrip_20260915_223424"},
        "prompt": PROMPT,
        "commit_count": 1,
        "export_sha256": "abc",
        "gpu": "test",
        "destinations": GRID,
    }
    identity.update(overrides)
    return {{"cosmos": "cosmos_hanoi", "pi05": "hanoi_waypoint"}[family]: identity}


def test_plan_orders_move_before_gripper_and_holds_when_nothing_changes():
    position = np.array([0.49, -0.05, 0.19])
    assert plan_waypoint(position, True, [0.49, -0.05, 0.19, 1.0]) == []
    steps = plan_waypoint(position, True, [0.49, 0.01, 0.07, 0.0])
    assert [kind for kind, _ in steps] == ["move", "gripper"]
    np.testing.assert_array_equal(steps[0][1], [0.49, 0.01, 0.07])
    assert steps[1][1] is False
    assert plan_waypoint(position, False, [0.49, -0.05, 0.19, 0.6]) == [("gripper", True)]
    assert plan_waypoint(position, True, [0.49, -0.05, 0.19, 0.49]) == [("gripper", False)]
    with pytest.raises(ValueError):
        plan_waypoint(position, True, [0.49, -0.05, np.nan, 1.0])


def test_gripper_command_uses_the_live_dwell_timings():
    assert gripper_command(True, 5).ticks == 30
    assert gripper_command(False, 5).ticks == math.ceil((2.4 + 0.2) * 30) == 78
    assert gripper_command(False, 5).kind == "gripper" and gripper_command(False, 5).jaw_open is False


def test_contract_check_accepts_the_selected_server_and_rejects_deviations():
    assert check_contract(metadata(), expected_export_sha256="abc")["gpu"] == "test"
    assert check_contract(metadata())["policy_family"] == "cosmos"
    assert check_contract(metadata(family="pi05"))["policy_family"] == "pi05"
    # "selected" resolves to the family's own export hash.
    check_contract(metadata(family="pi05", export_sha256=cosmos_client.SELECTED_PI05_EXPORT_SHA256), expected_export_sha256="selected")
    check_contract(metadata(export_sha256=cosmos_client.SELECTED_EXPORT_SHA256), expected_export_sha256="selected")
    with pytest.raises(ValueError, match="not the selected checkpoint"):
        check_contract(metadata(family="pi05", export_sha256=cosmos_client.SELECTED_EXPORT_SHA256), expected_export_sha256="selected")
    # Float orientation survives serialization noise; everything else is exact.
    wobble = metadata()
    wobble["cosmos_hanoi"]["contract"]["orientation_rpy_rad"] = [0.0, 0.7853981633974483, 1e-12]
    check_contract(wobble)
    # The raw default never reads the recorded grid, so a server that omits it is accepted.
    no_grid = metadata()
    del no_grid["cosmos_hanoi"]["destinations"]
    check_contract(no_grid)
    cases = [
        ({"a": 1}, "not a Hanoi waypoint"),
        (metadata(contract={**EXPECTED_CONTRACT, "version": 3}), "mismatch for version"),
        (metadata(contract={**EXPECTED_CONTRACT, "action_horizon": 63}), "mismatch for action_horizon"),
        (metadata(contract={**EXPECTED_CONTRACT, "state": ["x_m", "y_m"]}), "mismatch for state"),
        (metadata(contract={**EXPECTED_CONTRACT, "rgb_crop_xywh": [0, 0, 480, 480]}), "mismatch for rgb_crop"),
        (metadata(prompt="Move all four rings from peg C to peg A following Tower of Hanoi rules."), "prompt"),
        (metadata(commit_count=2), "exactly one destination"),
    ]
    for bad, message in cases:
        with pytest.raises(ValueError, match=message):
            check_contract(bad)
    with pytest.raises(ValueError, match="not the selected checkpoint"):
        check_contract(metadata(), expected_export_sha256="other")


def test_transport_checks_identity_and_parses_replies():
    replies = [
        msgpack_numpy.packb({"actions": np.ones((8, 4), np.float32), "commit_count": 1, "server_timing": {"infer_ms": 1}}),
        msgpack_numpy.packb({"actions": np.ones((8, 4), np.float32), "commit_count": 1, "server_timing": {"infer_ms": 1},
                             "future_image": np.full((224, 224, 3), 7, np.uint8), "value": 0.25}),
        "Traceback: boom",
    ]

    def handler(ws):
        ws.send(msgpack_numpy.packb(metadata()))
        for reply in replies:
            request = msgpack_numpy.unpackb(ws.recv())
            assert request["observation/state"].shape == (7,)
            ws.send(reply)

    with serve(handler, "127.0.0.1", 0, compression=None) as server:
        port = server.socket.getsockname()[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        with pytest.raises(ValueError, match="not the selected checkpoint"):
            CosmosWebSocketPolicy(f"ws://127.0.0.1:{port}", timeout_s=2, warmup_timeout_s=2, expected_export_sha256="zzz")
        policy = CosmosWebSocketPolicy(f"ws://127.0.0.1:{port}", timeout_s=2, warmup_timeout_s=2, expected_export_sha256="abc")
        request = {"observation/image": np.zeros((224, 224, 3), np.uint8), "observation/state": np.zeros(7, np.float32),
                   "observation/cartesian_position": np.zeros(3, np.float32), "prompt": PROMPT}
        actions, timing, dream = policy.infer(request)
        assert actions.shape == (8, 4) and actions.dtype == np.float64 and timing == {"infer_ms": 1} and dream is None
        actions, timing, dream = policy.infer(request)
        assert dream["value"] == 0.25 and dream["future_image"].shape == (224, 224, 3) and dream["future_image"].dtype == np.uint8
        with pytest.raises(RuntimeError, match="Policy server failed"):
            policy.infer(request)
        policy.close()
        server.shutdown()


def fake_output(joints=None, xyz=(0.49, -0.056, 0.191)):
    joint = SimpleNamespace(gripper=SimpleNamespace(position=0.034))
    if joints is not None:
        joint.arm = SimpleNamespace(positions=joints)
    return SimpleNamespace(
        header=SimpleNamespace(id=1),
        cartesian=SimpleNamespace(positions=np.array([*xyz, 0.0, math.pi / 4, 0.0]), velocities=np.zeros(6)),
        joint=joint,
    )


def fake_arm(output, **arm_kwargs):
    api = SimpleNamespace(
        Model=SimpleNamespace(wxai_v0=1),
        StandardEndEffector=SimpleNamespace(wxai_v0_follower=2),
        Mode=SimpleNamespace(position=3, external_effort=4),
        InterpolationSpace=SimpleNamespace(cartesian=5),
    )
    driver = MagicMock()
    driver.get_error_information.return_value = "No error"
    driver.get_robot_output.return_value = output
    return hardware.TrossenArm("192.168.1.3", driver=driver, api=api, **arm_kwargs)


def test_read_joints_returns_the_same_snapshot_and_read_is_unchanged():
    joints = [0.05, 1.43, 1.21, -0.56, 0.038, 0.04]
    arm = fake_arm(fake_output(joints))
    state, measured = arm.read_joints()
    np.testing.assert_allclose(state, [0.49, -0.056, 0.191, 0, 0, 0, 0.034])
    np.testing.assert_allclose(measured, joints)
    assert arm.driver.get_robot_output.call_count == 1
    # The pi0.5 path never touches the arm joints, so the older fakes still work.
    np.testing.assert_allclose(fake_arm(fake_output()).read(), [0.49, -0.056, 0.191, 0, 0, 0, 0.034])
    with pytest.raises(ValueError, match="joint feedback"):
        fake_arm(fake_output([0.1, np.nan, 0, 0, 0, 0])).read_joints()


def test_replay_episode_builds_contract_observations(tmp_path):
    path = tmp_path / "episode.h5"
    with h5py.File(path, "w") as h5:
        h5["pixels"] = np.full((3, 224, 224, 3), 9, np.uint8)
        h5["joint_positions"] = np.arange(18, dtype=np.float32).reshape(3, 6) / 10
        proprio = np.zeros((3, 8), np.float32)
        proprio[:, :3] = [0.49, -0.05, 0.19]
        proprio[:, 3:6] = 123  # velocity never reaches the policy
        proprio[:, 6] = 0.034
        proprio[:, 7] = 999  # legacy commanded gripper never reaches the policy
        h5["proprio"] = proprio
        h5["command_monotonic_ns"] = np.array([40_000_000, 90_000_000, 20_000_000], np.int64)
        h5["image_receipt_monotonic_ns"] = np.array([0, 0, 0], np.int64)
    episode = ReplayEpisode(path)
    observation, state = episode.observe(7, 1)
    data = observation.data
    assert data["observation/state"].dtype == np.float32 and data["observation/state"].shape == (7,)
    np.testing.assert_allclose(data["observation/state"], np.r_[np.arange(6, 12) / 10, 0.034], atol=1e-6)
    np.testing.assert_allclose(data["observation/cartesian_position"], [0.49, -0.05, 0.19], atol=1e-6)
    assert data["observation/image"].shape == (224, 224, 3) and data["prompt"] == PROMPT
    assert observation.image_age_s == pytest.approx(0.09, abs=1e-6)  # stale rows are the caller's decision
    assert episode.observe(0, 0)[0].image_age_s == pytest.approx(0.04, abs=1e-6)
    np.testing.assert_allclose(state[:3], [0.49, -0.05, 0.19], atol=1e-6)
    episode.close()


def test_client_never_imports_the_model_environment():
    import sys

    assert "torch" not in sys.modules and "cosmos_policy" not in sys.modules
    assert cosmos_client.execution_adapter(cosmos_client.Config()).startswith("cosmos_waypoint_v4")


def test_terminal_status_decides_release_and_return_home():
    # (release gripper, return to joint home)
    assert stop_actions("duration_reached", live=True, return_home_after_duration=True) == (True, True)
    assert stop_actions("duration_reached", live=True, return_home_after_duration=False) == (False, False)
    assert stop_actions("missed_grasp", live=True, return_home_after_duration=False) == (True, True)
    assert stop_actions("operator_stop", live=True, return_home_after_duration=True) == (True, True)
    assert stop_actions("failed", live=True, return_home_after_duration=True) == (True, True)
    assert stop_actions("cleanup_failed", live=True, return_home_after_duration=True) == (False, False)
    assert stop_actions("rejected_command", live=True, return_home_after_duration=False) == (True, True)
    for status in ("duration_reached", "missed_grasp", "operator_stop", "replay_exhausted"):
        assert stop_actions(status, live=False, return_home_after_duration=True) == (False, False)


def test_snapping_is_phase_aware_and_refuses_ambiguous_grasps():
    kw = dict(max_snap_m=0.006, max_grasp_dz_m=0.004)
    # Hover: height only; the lateral prediction is kept (transit point).
    xyz, phase, d = snap_destination([0.4925, -0.0551, 0.1935], GRID, **kw)
    np.testing.assert_allclose(xyz, [0.4925, -0.0551, 0.1911]); assert phase == "hover" and d == pytest.approx(0.0024)
    # Release and grasp: nearest recorded point of that phase (run 3's actual predictions).
    xyz, phase, _ = snap_destination([0.4887, 0.0130, 0.1533], GRID, **kw)
    np.testing.assert_allclose(xyz, [0.4920, 0.0137, 0.1501]); assert phase == "release"
    xyz, phase, _ = snap_destination([0.4927, -0.0589, 0.0810], GRID, **kw)
    np.testing.assert_allclose(xyz, [0.4961, -0.0572, 0.0775]); assert phase == "grasp"
    xyz, _, _ = snap_destination([0.4925, -0.0548, 0.0878], GRID, **kw)
    np.testing.assert_allclose(xyz, [0.4961, -0.0572, 0.0877])
    with pytest.raises(ValueError, match="ambiguous"):
        snap_destination([0.4926, -0.0559, 0.0826], GRID, **kw)  # midway between two ring levels
    with pytest.raises(ValueError, match="between the recorded"):
        snap_destination([0.4926, -0.0559, 0.1100], GRID, **kw)
    with pytest.raises(ValueError, match="mm away"):
        snap_destination([0.4926, 0.0500, 0.1501], GRID, **kw)  # between pegs
    with pytest.raises(ValueError, match="XYZ"):
        snap_destination([0.49, np.nan, 0.15], GRID, **kw)


def test_raw_model_output_is_the_default():
    assert cosmos_client.Config().snap_to_recorded_destinations is False
    # Re-observe after every waypoint (the contract's execution prefix); the adapter name records the count.
    assert cosmos_client.Config().commit_count == 1
    assert cosmos_client.execution_adapter(cosmos_client.Config()) == "cosmos_waypoint_v4_commit_1_rest_to_rest"
    assert cosmos_client.execution_adapter(cosmos_client.Config(commit_count=3)) == "cosmos_waypoint_v4_commit_3_rest_to_rest"
    assert cosmos_client.execution_adapter(cosmos_client.Config(), "pi05") == "pi05_waypoint_v4_commit_1_rest_to_rest"
    one = cosmos_client.Config(commit_count=1, snap_to_recorded_destinations=True)
    assert cosmos_client.execution_adapter(one) == "cosmos_waypoint_v4_commit_1_rest_to_rest_snapped_ablation"


def test_start_pose_is_the_recorded_v4_episode_start_and_is_checked_in_joint_space():
    config = cosmos_client.Config()
    bounds = json.loads(config.workspace.read_text())
    for xyz, joints in cosmos_client.START_POSES.values():
        assert all(lo <= v <= hi for v, lo, hi in zip(xyz, bounds["xyz_min_m"], bounds["xyz_max_m"])) and len(joints) == 6
    start_xyz = cosmos_client.START_POSES[config.start][0]
    assert config.start == "episode_start" and start_xyz == cosmos_client.V4_START_XYZ_M
    # Behind peg B, not the pi0.5 rod-A hover the model only ever carries a ring to.
    assert np.linalg.norm(np.subtract(start_xyz, hardware.INITIAL_XYZ)) > 0.05
    joints = list(cosmos_client.V4_START_JOINTS_RAD)
    arm = fake_arm(fake_output(joints, xyz=start_xyz), initial_xyz=np.array(start_xyz))
    report = arm.verify_initial_pose(arm.read())
    assert report["target_xyz_m"] == list(start_xyz) and report["position_error_mm"] < 0.001
    # The pi0.5 default is untouched: the same readback fails against the rod-A pose.
    default_arm = fake_arm(fake_output(joints, xyz=start_xyz))
    np.testing.assert_allclose(default_arm.initial_xyz, hardware.INITIAL_XYZ)
    with pytest.raises(ValueError, match="Initial pose verification failed"):
        default_arm.verify_initial_pose(default_arm.read())
    with pytest.raises(ValueError, match="three finite"):
        fake_arm(fake_output(), initial_xyz=[0.4, np.nan, 0.19])
    # Joint-space check against the recorded start joints, for either start pose.
    report = cosmos_client.start_joint_report(joints, max_error_rad=0.05)
    hover = cosmos_client.start_joint_report(cosmos_client.HOVER_A_JOINTS_RAD, cosmos_client.HOVER_A_JOINTS_RAD, max_error_rad=0.05)
    assert hover["max_error_rad"] == 0.0
    assert report["max_error_rad"] == 0.0 and report["limit_rad"] == 0.05
    off = np.add(joints, [0, 0, 0.08, 0, 0, 0])
    with pytest.raises(ValueError, match="recorded v4 start by 0.0800 rad at joint 2"):
        cosmos_client.start_joint_report(off, max_error_rad=0.05)
    with pytest.raises(ValueError, match="six finite"):
        cosmos_client.start_joint_report(joints[:5], max_error_rad=0.05)


def test_save_dream_writes_the_future_frame_next_to_the_inputs(tmp_path):
    assert cosmos_client.save_dream(tmp_path, 3, None) is None
    frame = np.full((224, 224, 3), 7, np.uint8)
    name = cosmos_client.save_dream(tmp_path, 3, {"future_image": frame, "value": 0.5})
    assert name == "inference_dreams/000003.png"
    from PIL import Image

    np.testing.assert_array_equal(np.asarray(Image.open(tmp_path / name)), frame)


def test_pi05_waypoint_server_wrapper_binarizes_jaw_intent_and_commits_one():
    from examples.hanoi.deployment.serve_waypoint import WaypointPolicy

    class Fake:
        def infer(self, obs):
            actions = np.ones((8, 4), np.float32)
            actions[:, 3] = [0.98, 0.51, 0.49, 0.02, 1.0, 0.0, 0.5, 0.3]
            return {"actions": actions, "policy_timing": {"infer_ms": 5}}

    reply = WaypointPolicy(Fake()).infer({})
    assert reply["commit_count"] == 1 and reply["actions"].dtype == np.float32
    np.testing.assert_array_equal(reply["actions"][:, 3], [1, 1, 0, 0, 1, 0, 1, 0])

    class Bad:
        def infer(self, obs):
            return {"actions": np.zeros((63, 4), np.float32)}

    with pytest.raises(ValueError, match="eight finite"):
        WaypointPolicy(Bad()).infer({})
