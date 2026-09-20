"""Timing, cancellation, protocol, and hardware-command tests without a robot."""

import contextlib
import dataclasses
import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from openpi_client import hanoi
from openpi_client import msgpack_numpy
import pytest
from websockets.sync.server import serve

from examples.hanoi.deployment.async_inference import ActionBuffer
from examples.hanoi.deployment.async_inference import InferenceWorker
from examples.hanoi.deployment.async_inference import Observation
from examples.hanoi.deployment.async_inference import Prediction
from examples.hanoi.deployment.async_inference import WebSocketPolicy
from examples.hanoi.deployment.execution import Command
from examples.hanoi.deployment.execution import PolicyExecutor
from examples.hanoi.deployment.execution import ReferenceExecutor
from examples.hanoi.deployment.hardware import TrossenArm
from examples.hanoi.deployment.hardware import validate_trajectory

POSITION = np.array([0.45, 0.0, 0.15])


def observation(tick=0, generation=0):
    return Observation(
        tick,
        1.0,
        0.98,
        {
            "prompt": hanoi.PROMPTS["aaaa_to_cccc"],
            "observation/state": np.r_[POSITION, np.zeros(3), 0.034],
            "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
        },
        generation,
    )


def prediction(tick=0, *, generation=0, moving=False, jaw_open=True):
    actions = np.tile(np.r_[POSITION, float(jaw_open)], (63, 1))
    if moving:
        actions[:, 0] += (tick + np.arange(1, 64)) * 0.0001
    return Prediction(observation(tick, generation), actions, 1.1, 0.1)


def wait_for(predicate):
    deadline = time.monotonic() + 2
    while not predicate():
        if time.monotonic() > deadline:
            pytest.fail("Timed out waiting for worker")
        time.sleep(0.002)


def test_worker_latest_only_reset_and_owned_arrays(tmp_path):
    entered, release = threading.Event(), threading.Event()
    calls = []

    class Policy:
        def infer(self, data):
            calls.append(data["observation/state"].copy())
            entered.set()
            assert release.wait(2)
            return {"actions": prediction().actions}

        def close(self):
            release.set()

    worker = InferenceWorker(Policy(), record_dir=tmp_path)
    try:
        worker.submit(observation())
        assert entered.wait(2)
        worker.submit(observation(1))
        generation = worker.invalidate()
        obs = observation(2)
        worker.submit(obs)
        obs.data["observation/state"][0] = 99  # Caller mutation cannot change pending request.
        obs.data["observation/image"][:] = 255
        release.set()
        wait_for(lambda: len(calls) == 2 and worker.result is not None)
        result = worker.take()
        assert result.observation.generation == generation
        assert result.observation.tick == 2
        assert result.request_id == 1
        assert calls[1][0] == POSITION[0]
        assert worker.take() is None
    finally:
        worker.close()
    assert not worker.thread.is_alive()
    records = [json.loads(line) for line in (tmp_path / "inferences.jsonl").read_text().splitlines()]
    requests = [row for row in records if row["event"] == "inference_request"]
    replies = [row for row in records if row["event"] == "inference_response"]
    assert [row["observation_tick"] for row in requests] == [0, 2]  # Superseded pending tick 1 was never sent.
    assert [row["generation"] for row in requests] == [0, 1]
    assert [row["request_id"] for row in replies] == [0, 1]  # Save even the invalidated first result.
    for row, sent_state in zip(requests, calls, strict=True):
        with np.load(tmp_path / row["input_file"], allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["observation/state"], sent_state)
            np.testing.assert_array_equal(saved["observation/image"], np.zeros((224, 224, 3), dtype=np.uint8))
            assert saved["prompt"].item() == hanoi.PROMPTS["aaaa_to_cccc"]
        np.testing.assert_array_equal(row["state"], sent_state)
    assert len(list((tmp_path / "inference_inputs").glob("*.npz"))) == 2


def test_failed_inference_retains_exact_input_before_call(tmp_path):
    policy = Mock()
    obs = observation(17)
    obs.data["observation/image"][:] = np.arange(224, dtype=np.uint8)[None, :, None]

    def infer(data):
        # The input is already on disk even if the request never gets a reply.
        with np.load(tmp_path / "inference_inputs/000000.npz", allow_pickle=False) as saved:
            for key in ("observation/image", "observation/state"):
                np.testing.assert_array_equal(saved[key], data[key])
                assert saved[key].dtype == data[key].dtype
        raise TimeoutError("simulated inference timeout")

    policy.infer.side_effect = infer
    worker = InferenceWorker(policy, record_dir=tmp_path)
    try:
        worker.submit(obs)
        wait_for(lambda: worker.error is not None)
        with pytest.raises(RuntimeError, match="worker failed"):
            worker.take()
    finally:
        worker.close()
    records = [json.loads(line) for line in (tmp_path / "inferences.jsonl").read_text().splitlines()]
    assert [row["event"] for row in records] == ["inference_request", "inference_error"]
    assert records[0]["observation_tick"] == 17
    assert records[1]["request_id"] == records[0]["request_id"] == 0
    assert "simulated inference timeout" in records[1]["error"]
    assert records[1]["cancelled"] is False


@pytest.mark.parametrize("actions", [np.zeros((9, 4)), np.full((63, 4), np.nan)])
def test_worker_rejects_malformed_output(actions):
    policy = Mock()
    policy.infer.return_value = {"actions": actions}
    worker = InferenceWorker(policy)
    try:
        worker.submit(observation())
        wait_for(lambda: worker.error is not None)
        with pytest.raises(RuntimeError, match="worker failed"):
            worker.take()
    finally:
        worker.close()


def test_worker_rejects_stale_image():
    worker = InferenceWorker(Mock())
    try:
        with pytest.raises(ValueError, match="stale"):
            worker.submit(dataclasses.replace(observation(), image_received_at=0.9))
    finally:
        worker.close()


def test_latency_skip_transaction_and_committed_interval():
    executor = ReferenceExecutor(POSITION + np.array([0.0004, 0, 0]), velocity=np.array([0.003, 0, 0]))
    buffer = ActionBuffer(executor)
    assert buffer.accept(prediction(moving=True))
    command, proposed = buffer.propose(4)
    np.testing.assert_allclose(command.sample(np.array([1.0]))[0], POSITION + np.array([0.0013, 0, 0]))
    assert command.start_tick == 4
    assert proposed.available_tick == 13
    # Before a successful dispatch/commit, the reference has not advanced.
    assert buffer.executor is executor
    assert executor.available_tick == 0
    buffer.commit(command, proposed)
    assert buffer.accept(prediction(8, moving=True))
    assert buffer.propose(12) is None
    next_command, _ = buffer.propose(13)
    np.testing.assert_allclose(next_command.sample(np.array([0.0]))[0], command.sample(np.array([1.0]))[0])
    assert not buffer.accept(prediction(7, moving=True))


def test_expired_chunk_cannot_be_replayed():
    buffer = ActionBuffer(ReferenceExecutor(POSITION.copy()))
    buffer.accept(prediction())
    with pytest.raises(ValueError, match="expired"):
        buffer.propose(55)


def test_slowed_policy_move_waits_for_completion_while_accepting_new_predictions():
    from examples.hanoi.deployment.async_inference import reference_executor

    buffer = ActionBuffer(reference_executor(np.r_[POSITION, np.zeros(3), 0.034], jaw_open=True))
    assert buffer.accept(prediction(moving=True))
    command, proposed = buffer.propose(0)
    assert command.ticks >= 18  # Minimum Cartesian duration is now 0.6 seconds.
    buffer.commit(command, proposed)
    assert buffer.accept(prediction(command.ticks - 1, moving=True))
    assert buffer.propose(command.ticks - 1) is None
    next_command, _ = buffer.propose(command.ticks)
    np.testing.assert_allclose(next_command.sample(np.array([0.0]))[0], command.sample(np.array([1.0]))[0])


def test_gripper_dwell_and_generation_barrier():
    buffer = ActionBuffer(PolicyExecutor(POSITION.copy()))
    buffer.accept(prediction(jaw_open=False))
    command, proposed = buffer.propose(0)
    assert command.kind == "gripper"
    assert command.ticks == 78
    buffer.commit(command, proposed)
    buffer.generation = 1
    assert buffer.prediction is None
    assert not buffer.accept(prediction(1, jaw_open=False))  # Old in-flight result.
    assert not buffer.accept(prediction(77, generation=1, jaw_open=False))
    assert buffer.propose(77) is None
    assert buffer.accept(prediction(78, generation=1, jaw_open=False))
    command, _ = buffer.propose(78)
    assert command.kind == "hold"  # Repeated close must not restart dwell.


def test_grasp_pair_keeps_original_prediction_and_survives_approach_longer_than_chunk():
    buffer = ActionBuffer(PolicyExecutor(POSITION.copy()))
    source = prediction(jaw_open=False)
    source.actions[:, 2] -= 0.1
    assert buffer.accept(source)
    approach, proposed = buffer.propose(0)
    assert approach.kind == "cartesian"
    assert approach.ticks > 63
    # Failed dispatch must not schedule either the new endpoint or the close.
    assert buffer.executor.pending_jaw_open is None
    np.testing.assert_array_equal(buffer.executor.position, POSITION)
    buffer.commit(approach, proposed)
    assert not buffer.accept(prediction(approach.ticks - 1, jaw_open=True))
    assert buffer.prediction is source
    assert buffer.propose(approach.ticks - 1) is None
    close, proposed = buffer.propose(approach.ticks)
    assert close.kind == "gripper"
    assert close.jaw_open is False
    # Failed close dispatch also leaves the pending operation intact.
    assert buffer.executor.pending_jaw_open is False
    buffer.commit(close, proposed)
    assert buffer.prediction is None
    buffer.generation = 1
    end = approach.ticks + close.ticks
    assert not buffer.accept(prediction(end - 1, generation=1, jaw_open=False))
    assert buffer.accept(prediction(end, generation=1, jaw_open=False))


def test_reset_cancels_pending_grasp():
    buffer = ActionBuffer(PolicyExecutor(POSITION.copy()))
    buffer.accept(prediction(moving=True, jaw_open=False))
    command, proposed = buffer.propose(0)
    buffer.commit(command, proposed)
    assert buffer.executor.pending_jaw_open is False
    buffer.reset(PolicyExecutor(POSITION.copy()), 1)
    assert buffer.executor.pending_jaw_open is None
    assert buffer.propose(command.ticks) is None


def test_reset_discards_old_task_actions():
    buffer = ActionBuffer(ReferenceExecutor(POSITION.copy()))
    buffer.accept(prediction())
    buffer.reset(ReferenceExecutor(POSITION.copy()), 1)
    assert buffer.propose(0) is None
    assert not buffer.accept(prediction(10))


def test_trossen_nonblocking_derivatives_and_gripper_calls():
    driver = Mock()
    api = SimpleNamespace(
        Model=SimpleNamespace(wxai_v0=1),
        StandardEndEffector=SimpleNamespace(wxai_v0_follower=2),
        Mode=SimpleNamespace(position=3, external_effort=4),
        InterpolationSpace=SimpleNamespace(cartesian=5),
    )
    arm = TrossenArm("unused", driver=driver, api=api)
    driver.set_arm_modes.assert_not_called()
    arm.enable()
    coefficients = np.zeros((6, 3))
    coefficients[0] = POSITION
    coefficients[1] = [0.0009, 0.0, 0.0]
    command = Command("cartesian", 0, 9, coefficients)
    arm.dispatch(command, np.array([0.4, -0.1, 0.1]), np.array([0.5, 0.1, 0.2]))
    args, kwargs = driver.set_cartesian_positions.call_args
    np.testing.assert_allclose(args[0][:3], POSITION + np.array([0.0009, 0, 0]))
    assert kwargs["goal_time"] == 0.3
    assert kwargs["blocking"] is False
    np.testing.assert_allclose(kwargs["goal_feedforward_velocities"], [0.003, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(kwargs["goal_feedforward_accelerations"], np.zeros(6))
    arm.dispatch(Command("gripper", 9, 78, jaw_open=False), None, None)
    driver.set_gripper_external_effort.assert_called_once_with(-20.0, 2.4, blocking=False)
    arm.dispatch(Command("gripper", 87, 30, jaw_open=True), None, None)
    driver.set_gripper_position.assert_called_once_with(0.034, 1.0, blocking=False)


def test_workspace_checks_interior_extremum():
    coefficients = np.zeros((6, 3))
    coefficients[0] = POSITION
    coefficients[1, 0] = 0.008
    coefficients[2, 0] = -0.008
    command = Command("cartesian", 0, 9, coefficients)
    with pytest.raises(ValueError, match="workspace"):
        validate_trajectory(command, np.array([0.4, -0.1, 0.1]), np.array([0.451, 0.1, 0.2]))


@pytest.mark.parametrize("wrong_contract", [False, True])
def test_real_websocket_contract_and_numpy_roundtrip(wrong_contract):
    def handler(ws):
        contract = {**hanoi.CONTRACT, "action_horizon": 9} if wrong_contract else hanoi.CONTRACT
        ws.send(msgpack_numpy.packb(contract))
        if not wrong_contract:
            obs = msgpack_numpy.unpackb(ws.recv())
            assert obs["observation/image"].shape == (224, 224, 3)
            ws.send(msgpack_numpy.packb({"actions": prediction().actions}))

    with serve(handler, "127.0.0.1", 0) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        uri = f"ws://127.0.0.1:{server.socket.getsockname()[1]}"
        try:
            if wrong_contract:
                with pytest.raises(ValueError, match="contract mismatch"):
                    WebSocketPolicy(uri)
            else:
                policy = WebSocketPolicy(uri)
                try:
                    np.testing.assert_array_equal(policy.infer(observation().data)["actions"], prediction().actions)
                finally:
                    policy.close()
        finally:
            server.shutdown()
            thread.join(2)


def test_close_interrupts_blocked_websocket_receive(tmp_path):
    entered = threading.Event()

    def handler(ws):
        ws.send(msgpack_numpy.packb(hanoi.CONTRACT))
        ws.recv()
        entered.set()
        with contextlib.suppress(Exception):
            ws.recv()  # Wait for client close, never return an inference response.

    with serve(handler, "127.0.0.1", 0) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        worker = InferenceWorker(
            WebSocketPolicy(f"ws://127.0.0.1:{server.socket.getsockname()[1]}"), record_dir=tmp_path
        )
        try:
            worker.submit(observation())
            assert entered.wait(2)
            start = time.monotonic()
            worker.close()
            assert time.monotonic() - start < 1
            assert not worker.thread.is_alive()
        finally:
            server.shutdown()
            thread.join(2)
    records = [json.loads(line) for line in (tmp_path / "inferences.jsonl").read_text().splitlines()]
    assert [row["event"] for row in records] == ["inference_request", "inference_error"]
    assert records[-1]["cancelled"] is True
    assert (tmp_path / records[0]["input_file"]).is_file()


def test_full_control_loop_with_slow_policy(tmp_path):
    import json

    import h5py

    from examples.hanoi.deployment.client import Config
    from examples.hanoi.deployment.client import main

    episode = tmp_path / "episode.h5"
    with h5py.File(episode, "w") as f:
        f["proprio"] = np.tile(np.r_[POSITION, np.zeros(3), 0.034, 0.034], (60, 1))
        f["pixels"] = np.broadcast_to(np.arange(60, dtype=np.uint8)[:, None, None, None], (60, 224, 224, 3))
        f["command_monotonic_ns"] = np.arange(60) * 33_333_333
        f["image_receipt_monotonic_ns"] = f["command_monotonic_ns"][:] - 10_000_000
    episode.with_suffix(".json").write_text(
        json.dumps({"contract": hanoi.CONTRACT, "prompt": hanoi.PROMPTS["aaaa_to_cccc"]})
    )

    received_inputs = []

    def handler(ws):
        ws.send(msgpack_numpy.packb(hanoi.CONTRACT))
        with contextlib.suppress(Exception):
            while True:
                received_inputs.append(msgpack_numpy.unpackb(ws.recv()))
                time.sleep(0.08)  # Longer than two control ticks.
                ws.send(msgpack_numpy.packb({"actions": prediction().actions}))

    with serve(handler, "127.0.0.1", 0) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            main(
                Config(
                    mode="replay",
                    episode=episode,
                    duration_s=0.7,
                    output=tmp_path / "runs",
                    server=f"ws://127.0.0.1:{server.socket.getsockname()[1]}",
                )
            )
        finally:
            server.shutdown()
            thread.join(2)
    summary = json.loads(next((tmp_path / "runs").glob("*/summary.json")).read_text())
    assert summary["status"] == "duration_reached"
    assert summary["ticks"] == 21
    assert 2 <= summary["accepted_commands"] <= 3
    assert summary["predictions"] < summary["ticks"]
    assert summary["rejected_commands"] == 0
    run = next((tmp_path / "runs").iterdir())
    records = [json.loads(line) for line in (run / "inferences.jsonl").read_text().splitlines()]
    requests = [row for row in records if row["event"] == "inference_request"]
    replies = {row["request_id"]: row for row in records if row["event"] == "inference_response"}
    assert requests[0]["request_id"] == 0  # Warmup is recorded too.
    assert requests[0]["generation"] == 0
    assert len(received_inputs) >= len(replies) >= 2
    assert len(requests) == len(list((run / "inference_inputs").glob("*.npz")))
    assert len(requests) < summary["ticks"]  # Record requests, not every queued control observation.
    for row, sent in zip(requests, received_inputs, strict=False):
        with np.load(run / row["input_file"], allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["observation/image"], sent["observation/image"])
            np.testing.assert_array_equal(saved["observation/state"], sent["observation/state"])
            assert saved["prompt"].item() == sent["prompt"]
            assert saved["observation/image"][0, 0, 0] == row["observation_tick"]
    events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
    for event in (row for row in events if row["event"] in {"command", "prediction"}):
        assert event["request_id"] in replies
        assert event["request_id"] != 0  # Warmup is never executed.
        assert event["observation_tick"] == requests[event["request_id"]]["observation_tick"]


def test_robot_feedback_is_measured_and_watchdog_detects_stall(monkeypatch):
    from examples.hanoi.deployment import hardware

    clock = [1.0]
    monkeypatch.setattr(hardware.time, "monotonic", lambda: clock[0])
    driver = Mock()
    driver.get_error_information.return_value = "No error"
    driver.get_robot_output.return_value = SimpleNamespace(
        header=SimpleNamespace(id=1),
        cartesian=SimpleNamespace(
            positions=np.r_[POSITION, 0.0, np.pi / 4, 0.0], velocities=[0.001, 0.002, 0.003, 0, 0, 0]
        ),
        joint=SimpleNamespace(gripper=SimpleNamespace(position=0.012)),
    )
    arm = TrossenArm("unused", driver=driver, api=Mock())
    state = arm.read()
    np.testing.assert_allclose(state, [*POSITION, 0.001, 0.002, 0.003, 0.012])
    clock[0] += 0.11
    with pytest.raises(RuntimeError, match="stopped updating"):
        arm.read()


@pytest.fixture
def uninitialized_arm():
    driver = Mock()
    driver.get_error_information.return_value = "No error"
    output = SimpleNamespace(
        header=SimpleNamespace(id=0),
        cartesian=SimpleNamespace(positions=np.r_[POSITION, np.zeros(3)], velocities=np.zeros(6)),
        joint=SimpleNamespace(gripper=SimpleNamespace(position=0.034)),
    )

    def feedback():
        output.header.id += 1
        return output

    driver.get_robot_output.side_effect = feedback
    return TrossenArm("unused", driver=driver, api=Mock()), driver, output


@pytest.mark.parametrize("mode", ["shadow", "live"])
def test_client_shadow_feedback_and_failed_live_initialization(tmp_path, monkeypatch, caplog, uninitialized_arm, mode):
    import json

    from examples.hanoi.deployment import client

    arm, driver, _ = uninitialized_arm
    camera = Mock()
    camera.latest.side_effect = lambda: SimpleNamespace(
        rgb=np.zeros((224, 224, 3), dtype=np.uint8), received_at=time.monotonic()
    )
    policy = Mock()
    policy.metadata = hanoi.CONTRACT
    policy.infer.return_value = {"actions": prediction().actions}
    policy_factory = Mock(return_value=policy)
    monkeypatch.setattr(client, "RosCamera", lambda topic: camera)
    monkeypatch.setattr(client, "TrossenArm", lambda ip: arm)
    monkeypatch.setattr(client, "WebSocketPolicy", policy_factory)
    workspace = tmp_path / "bounds.json"
    workspace.write_text(json.dumps({"xyz_min_m": [0.4, -0.1, 0.1], "xyz_max_m": [0.5, 0.1, 0.2]}))
    config = client.Config(mode=mode, workspace=workspace, duration_s=0.1, output=tmp_path / mode)
    if mode == "live":
        arm.initialize = Mock(side_effect=ValueError("Initialization failed"))
        with pytest.raises(ValueError, match="Initialization failed"):
            client.main(config)
        arm.initialize.assert_called_once()
        policy_factory.assert_not_called()
    else:
        client.main(config)
        summary_path = next(config.output.glob("*/summary.json"))
        summary = json.loads(summary_path.read_text())
        assert summary["status"] == "duration_reached"
        assert summary["predictions"] > 0
        assert "Shadow mode: tool orientation" in caplog.text
        events = [json.loads(line) for line in (summary_path.parent / "events.jsonl").read_text().splitlines()]
        orientation = next(event for event in events if event["event"] == "initial_tool_orientation")
        assert orientation["error_deg"] == pytest.approx(45)
        assert orientation["measured_rotvec_rad"] == [0, 0, 0]
        assert not orientation["required_for_motion"]
    # Shadow and a rejected live startup must never enable or move the robot.
    assert not [call for call in driver.method_calls if call[0].startswith("set_")]
    driver.cleanup.assert_called_once()
    camera.close.assert_called_once()


def test_equivalent_wrapped_orientation_passes_motion_check(uninitialized_arm):
    arm, _, output = uninitialized_arm
    output.cartesian.positions[3:] = [0, np.pi / 4 + 2 * np.pi, 0]
    state = arm.read()
    np.testing.assert_allclose(state[:3], POSITION)
    assert arm.orientation_error_rad == pytest.approx(0, abs=1e-7)


def test_shadow_still_rejects_controller_fault_and_nonfinite_feedback(uninitialized_arm):
    arm, driver, output = uninitialized_arm
    driver.get_error_information.return_value = "Controller fault"
    with pytest.raises(RuntimeError, match="Controller fault"):
        arm.read(require_orientation=False)
    driver.get_error_information.return_value = "No error"
    output.cartesian.positions[0] = np.nan
    with pytest.raises(ValueError, match="Invalid robot feedback"):
        arm.read(require_orientation=False)


@pytest.mark.parametrize("state", [np.zeros(6), np.r_[np.zeros(3), np.nan, np.zeros(3)]])
def test_reference_start_rejects_invalid_measured_state(state):
    from examples.hanoi.deployment.async_inference import reference_executor

    with pytest.raises(ValueError, match="seven measured state values"):
        reference_executor(state, jaw_open=True)
