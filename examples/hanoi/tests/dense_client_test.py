"""Hardware-free tests for the dense contract-five executor, contract check and transport."""
import threading

import numpy as np
import pytest
from openpi_client import msgpack_numpy
from websockets.sync.server import serve

from examples.hanoi.deployment import dense_client
from examples.hanoi.deployment import dense_execution
from examples.hanoi.deployment.async_inference import InferenceWorker
from examples.hanoi.deployment.dense_execution import DenseExecutor
from examples.hanoi.deployment.execution import LIMITS
from examples.hanoi.deployment.hardware import validate_trajectory

START = np.array([0.45, 0.0, 0.19])


def leg(distance=(0.15, 0.0, 0.0), duration_s=1.94, n=30, offset_s=0.0, jaw=1.0):
    """10 Hz samples of a rest-to-rest quintic leg like the recorder's, starting ``offset_s`` into it."""
    t = offset_s + np.arange(1, n + 1) * 0.1
    u = np.clip(t / duration_s, 0, 1)
    s = 10 * u**3 - 15 * u**4 + 6 * u**5
    return np.c_[START + s[:, None] * np.asarray(distance), np.full(n, jaw)]


def leg_state(distance, duration_s, t):
    """Position, velocity and acceleration of that leg at time ``t``."""
    u = t / duration_s
    d = np.asarray(distance)
    return (START + (10 * u**3 - 15 * u**4 + 6 * u**5) * d,
            (30 * u**2 - 60 * u**3 + 30 * u**4) / duration_s * d,
            (60 * u - 180 * u**2 + 120 * u**3) / duration_s**2 * d)


def test_segments_track_a_recorded_leg_with_continuous_velocity_and_skip_elapsed_rows():
    ex = DenseExecutor(START.copy(), prefix=3)
    cmd = ex.plan(leg(), observation_tick=0, now_tick=0, image_age_s=0.01, task="f")
    assert cmd.kind == "cartesian" and ex.last_stretch <= 1.6  # leg start from rest, near the jerk limit
    validate_trajectory(cmd, START - 1, START + 1, dense_execution.TRACK_LIMITS)
    np.testing.assert_allclose(ex.position, leg()[2, :3], atol=1e-9)
    # Mid-leg (cruise): the segment is exactly three rows, and starts with the previous end velocity.
    p, v, a = leg_state((0.15, 0, 0), 1.94, 0.9)
    ex = DenseExecutor(p.copy(), velocity=v.copy(), acceleration=a.copy(), prefix=3)
    cmd = ex.plan(leg(offset_s=0.9), observation_tick=0, now_tick=0, image_age_s=0.01, task="f")
    assert cmd.ticks == 9 and ex.last_stretch == 1.0
    assert np.linalg.norm(cmd.sample(np.array([0.0]), 1)[0] - v) < 1e-9
    _, v_end, _ = leg_state((0.15, 0, 0), 1.94, 1.2)
    assert np.linalg.norm(ex.velocity - v_end) < 0.003  # central-difference estimate of the demonstrated velocity
    assert ex.plan(leg(offset_s=0.9), observation_tick=0, now_tick=5, image_age_s=0.01, task="f").kind == "wait"
    cmd = ex.plan(leg(offset_s=0.9), observation_tick=0, now_tick=9, image_age_s=0.01, task="f")
    np.testing.assert_allclose(ex.position, leg(offset_s=0.9)[5, :3], atol=1e-9)  # rows 0-2 elapsed, 3-5 executed
    validate_trajectory(cmd, START - 1, START + 1, dense_execution.TRACK_LIMITS)


def test_expired_stale_and_hold():
    ex = DenseExecutor(START.copy(), prefix=3)
    with pytest.raises(ValueError, match="expired"):
        ex.plan(leg(), observation_tick=0, now_tick=90, image_age_s=0.01, task="f")
    with pytest.raises(ValueError, match="Stale"):
        ex.plan(leg(), observation_tick=0, now_tick=0, image_age_s=0.2, task="f")
    still = leg(distance=(0, 0, 0))
    cmd = ex.plan(still, observation_tick=0, now_tick=0, image_age_s=0.01, task="f")
    assert cmd.kind == "hold" and cmd.ticks == 9


def test_jaw_change_aligns_at_rest_then_dwells_and_needs_a_fresh_observation():
    ex = DenseExecutor(START.copy(), prefix=3)
    close = leg(distance=(0, 0, 0), jaw=0.0)
    close[:, :3] = START + [0.004, 0, 0]
    cmd = ex.plan(close, observation_tick=0, now_tick=0, image_age_s=0.01, task="f")
    assert cmd.kind == "cartesian" and ex.pending_jaw_open is False and ex.last_stretch == 1.0
    assert cmd.ticks == dense_execution.rest_to_rest_ticks(0.004) and np.linalg.norm(ex.velocity) < 1e-9
    validate_trajectory(cmd, START - 1, START + 1, dense_execution.TRACK_LIMITS)
    cmd = ex.plan(close, observation_tick=0, now_tick=ex.available_tick, image_age_s=0.01, task="f")
    assert (cmd.kind, cmd.ticks, cmd.jaw_open) == ("gripper", 78, False)
    with pytest.raises(ValueError, match="fresh observation"):
        ex.plan(close, observation_tick=0, now_tick=ex.available_tick, image_age_s=0.01, task="f")
    # A close in the third row: the prefix stops short, at rest, so the next boundary sees it at row 0.
    ex = DenseExecutor(START.copy(), prefix=3)
    mixed = leg()
    mixed[2:, 3] = 0.0
    cmd = ex.plan(mixed, observation_tick=0, now_tick=0, image_age_s=0.01, task="f")
    assert cmd.kind == "cartesian" and np.linalg.norm(ex.velocity) < 1e-9
    np.testing.assert_allclose(ex.position, mixed[1, :3], atol=1e-9)


def test_fast_references_are_stretched_then_refused():
    fast = leg(distance=(0.15, 0, 0), duration_s=0.8)  # far faster than the recorder ever moved
    ex = DenseExecutor(START.copy(), prefix=3, max_stretch=6.0)
    cmd = ex.plan(fast, observation_tick=0, now_tick=0, image_age_s=0.01, task="f")
    assert cmd.kind == "cartesian" and 1.0 < ex.last_stretch <= 6.0 and dense_execution.within_limits(cmd)
    # Beyond the allowed stretch the executor brakes to rest where it is instead of refusing.
    p, v, a = leg_state((0.15, 0, 0), 1.94, 0.9)
    ex = DenseExecutor(p.copy(), velocity=v.copy(), acceleration=a.copy(), prefix=3, max_stretch=1.0)
    reversing = leg(distance=(-0.15, 0, 0), duration_s=0.8, offset_s=0.1)
    reversing[:, :3] += p - START  # rows start where the arm is, then head back fast
    cmd = ex.plan(reversing, observation_tick=0, now_tick=0, image_age_s=0.01, task="f")
    assert cmd.kind == "cartesian" and ex.last_braked and np.linalg.norm(ex.velocity) < 1e-9
    # The stop runs on along the cruise direction, as far as the jerk limit needs, and no further.
    assert 0.02 < ex.position[0] - p[0] < 0.12 and abs(ex.position[1] - p[1]) < 1e-9
    assert dense_execution.within_limits(cmd)
    validate_trajectory(cmd, START - 1, START + 1, dense_execution.TRACK_LIMITS)


def metadata(horizon=30, prefix=3, config=None, **overrides):
    config = config or ("pi05_hanoi_dense_aaaa_to_cccc" if horizon == 30 else "pi05_hanoi_dense_h16_aaaa_to_cccc")
    identity = {"contract": {**dense_client.EXPECTED_CONTRACT, "action_horizon": horizon, "execution_prefix": prefix, "recording": "r"},
                "prompt": dense_client.PROMPT, "config_name": config,
                "export_sha256": dense_client.SELECTED_EXPORTS.get(config, "x"), "gpu": "test", "num_steps": 10}
    identity.update(overrides)
    return {"hanoi_dense": identity}


def test_contract_check():
    assert dense_client.check_contract(metadata(), expected_export_sha256="selected")["action_horizon"] == 30
    assert dense_client.check_contract(metadata(horizon=16), expected_export_sha256="selected")["action_horizon"] == 16
    with pytest.raises(ValueError, match="not the selected"):  # the 16-step hash under the 30-step config name
        dense_client.check_contract(metadata(export_sha256=dense_client.SELECTED_EXPORTS["pi05_hanoi_dense_h16_aaaa_to_cccc"]), expected_export_sha256="selected")
    cosmos = dense_client.check_contract(metadata(horizon=16, prefix=8, config="cosmos_hanoi_dense_v5_h16", model="cosmos_dense"), expected_export_sha256="selected")
    assert cosmos["execution_prefix"] == 8 and cosmos["policy_family"] == "cosmos_dense"
    with pytest.raises(ValueError, match="mismatch for execution_prefix"):
        dense_client.check_contract(metadata(prefix=5), expected_export_sha256="selected")
    ex = DenseExecutor(START.copy(), horizon=16, prefix=3)
    assert ex.plan(leg(n=16), observation_tick=0, now_tick=0, image_age_s=0.01, task="f").kind == "cartesian"
    with pytest.raises(ValueError, match="expired"):
        ex.plan(leg(n=16), observation_tick=0, now_tick=48, image_age_s=0.01, task="f")
    for bad, message in [({"a": 1}, "not a Hanoi dense"),
                         (metadata(horizon=8), "mismatch for action_horizon"),
                         (metadata(contract={**dense_client.EXPECTED_CONTRACT, "internal_xyz_encoding": "relative"}), "mismatch for internal"),
                         (metadata(prompt="x"), "prompt"),
                         (metadata(export_sha256="other"), "not the selected")]:
        with pytest.raises(ValueError, match=message):
            dense_client.check_contract(bad, expected_export_sha256="selected")


def test_transport_and_worker_accept_thirty_row_chunks():
    def handler(ws):
        ws.send(msgpack_numpy.packb(metadata()))
        request = msgpack_numpy.unpackb(ws.recv())
        assert request["observation/state"].shape == (7,)
        ws.send(msgpack_numpy.packb({"actions": leg().astype(np.float32), "reference_rate_hz": 10, "execution_prefix": 3,
                                     "server_timing": {"infer_ms": 1}}))

    with serve(handler, "127.0.0.1", 0, compression=None) as server:
        port = server.socket.getsockname()[1]
        threading.Thread(target=server.serve_forever, daemon=True).start()
        policy = dense_client.DenseWebSocketPolicy(f"ws://127.0.0.1:{port}", timeout_s=2, warmup_timeout_s=2,
                                                   expected_export_sha256="selected")
        worker = InferenceWorker(policy, horizon=30)
        obs = dense_client.observation_data(np.zeros((224, 224, 3), np.uint8), np.zeros(6), 0.034, START)
        from examples.hanoi.deployment.async_inference import Observation
        import time
        now = time.monotonic()
        worker.submit(Observation(0, now, now - 0.01, obs))
        deadline = time.monotonic() + 5
        while (prediction := worker.take()) is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert prediction is not None and prediction.actions.shape == (30, 4)
        worker.close()
        server.shutdown()


def test_default_nine_row_segments_follow_a_leg_without_braking():
    ex = DenseExecutor(START.copy())
    assert ex.prefix == 9
    cmd = ex.plan(leg(), observation_tick=0, now_tick=0, image_age_s=0.01, task="f")
    assert cmd.kind == "cartesian" and not ex.last_braked and cmd.ticks >= 27
    np.testing.assert_allclose(ex.position, leg()[8, :3], atol=1e-9)
    validate_trajectory(cmd, START - 1, START + 1, dense_execution.TRACK_LIMITS)


def test_carry_rule_flags_sideways_travel_below_the_carry_height_only():
    from examples.hanoi.deployment.dense_execution import carry_violation, quintic_segment
    p = np.array([0.49, 0.086, 0.191])
    high = quintic_segment(p, np.zeros(3), np.zeros(3), p + [0, -0.07, 0], np.zeros(3), np.zeros(3), 60, 0)
    assert carry_violation(high, min_z_m=0.17) is None  # lateral at the hover height
    descent = quintic_segment(p, np.zeros(3), np.zeros(3), p + [0, 0, -0.04], np.zeros(3), np.zeros(3), 30, 0)
    assert carry_violation(descent, min_z_m=0.17) is None  # straight down over the column
    diagonal = quintic_segment(p, np.zeros(3), np.zeros(3), p + [0, -0.03, -0.04], np.zeros(3), np.zeros(3), 30, 0)
    low = carry_violation(diagonal, min_z_m=0.17)
    assert low is not None and 0.15 < low < 0.17
    assert carry_violation(diagonal, min_z_m=0.0) is None  # disabled
