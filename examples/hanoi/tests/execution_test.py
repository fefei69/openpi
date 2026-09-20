import numpy as np
import pytest

from examples.hanoi.deployment import execution


def hold_actions(jaw: float = 1.0) -> np.ndarray:
    return np.tile([0.4, 0.02, 0.15, jaw], (63, 1))


def test_gripper_dwell_requires_fresh_observation_and_does_not_repeat():
    executor = execution.ReferenceExecutor(np.array([0.4, 0.02, 0.15]))
    close = hold_actions(0)
    command = executor.plan(close, observation_tick=0, now_tick=0, image_age_s=0.01, task="forward")
    assert (command.kind, command.ticks) == ("gripper", 42)
    assert executor.plan(close, observation_tick=1, now_tick=1, image_age_s=0.01, task="forward").kind == "wait"
    with pytest.raises(ValueError, match="fresh observation"):
        executor.plan(close, observation_tick=0, now_tick=42, image_age_s=0.01, task="forward")
    assert executor.plan(close, observation_tick=42, now_tick=42, image_age_s=0.01, task="forward").kind == "hold"
    command = executor.plan(hold_actions(), observation_tick=51, now_tick=51, image_age_s=0.01, task="forward")
    assert (command.kind, command.ticks) == ("gripper", 30)


@pytest.mark.parametrize(("age", "tick"), [(0.051, 0), (-0.01, 0), (0.01, 60), (0.01, -1)])
def test_stale_and_expired_predictions_rejected(age, tick):
    executor = execution.ReferenceExecutor(np.array([0.4, 0.02, 0.15]))
    with pytest.raises(ValueError, match="[Ss]tale|expired|[Ii]nvalid"):
        executor.plan(hold_actions(), observation_tick=0, now_tick=tick, image_age_s=age, task="forward")


def test_task_reset_and_motion_during_jaw_rejected():
    executor = execution.ReferenceExecutor(np.array([0.4, 0.02, 0.15]))
    executor.plan(hold_actions(), observation_tick=0, now_tick=0, image_age_s=0.01, task="forward")
    with pytest.raises(ValueError, match="Reset"):
        executor.plan(hold_actions(), observation_tick=9, now_tick=9, image_age_s=0.01, task="reverse")
    actions = hold_actions(0)
    actions[10, 0] += 0.01
    with pytest.raises(ValueError, match="during gripper"):
        executor.plan(actions, observation_tick=9, now_tick=9, image_age_s=0.01, task="forward")


def test_quintic_preserves_nonzero_boundary_derivatives():
    coefficients = np.array([[0.4, 0.02, 0.15], [0.01, 0, 0], [0.001, 0, 0], [0.002, 0, 0], [-0.001, 0, 0], [0, 0, 0]])
    expected = np.polynomial.polynomial.polyvander(np.arange(1, 10) / 9, 5) @ coefficients
    command = execution.fit_cartesian(expected, coefficients[0], coefficients[1] / 0.3, 2 * coefficients[2] / 0.09, 0)
    np.testing.assert_allclose(command.coefficients, coefficients, atol=1e-10)
    np.testing.assert_allclose(command.sample(np.array([0.0]), 1)[0], coefficients[1] / 0.3, atol=1e-10)


def test_limits_reject_unrepresentable_prediction():
    targets = np.tile([0.4, 0.02, 0.15], (9, 1))
    targets[4, 0] += 0.1
    with pytest.raises(ValueError, match="continuous prefix"):
        execution.fit_cartesian(targets, np.array([0.4, 0.02, 0.15]), np.zeros(3), np.zeros(3), 0)


def test_actual_failed_live_prediction_becomes_bounded_endpoint_move():
    import json
    from pathlib import Path

    from examples.hanoi.deployment.async_inference import ActionBuffer
    from examples.hanoi.deployment.async_inference import Observation
    from examples.hanoi.deployment.async_inference import Prediction
    from examples.hanoi.deployment.async_inference import reference_executor
    from examples.hanoi.deployment.hardware import validate_trajectory

    fixture = json.loads((Path(__file__).parent / "fixtures/first_live_prediction.json").read_text())
    state, actions = np.array(fixture["state"]), np.array(fixture["actions"])
    observation = Observation(fixture["observation_tick"], 1.0, 0.99, {"prompt": "forward"})
    buffer = ActionBuffer(reference_executor(state, jaw_open=True))
    buffer.accept(Prediction(observation, actions, 1.106, 0.106))
    command, proposed = buffer.propose(fixture["dispatch_tick"])
    assert command.kind == "cartesian"
    np.testing.assert_allclose(command.sample(np.array([1.0]))[0], actions[12, :3])
    np.testing.assert_array_equal(buffer.executor.position, state[:3])  # not committed yet
    np.testing.assert_array_equal(proposed.velocity, np.zeros(3))
    bounds = json.loads((Path(__file__).parents[1] / "deployment/workspace.json").read_text())
    validate_trajectory(command, np.array(bounds["xyz_min_m"]), np.array(bounds["xyz_max_m"]))
    assert command.ticks == 22  # Half-speed setting doubles the former 11-tick move.


@pytest.mark.parametrize("distance", [0.0002, 0.005, 0.03, 0.1])
def test_endpoint_move_duration_respects_speed_acceleration_and_jerk(distance):
    position = np.array([0.4, 0.02, 0.15])
    target = position + distance * np.array([1, -2, 2]) / 3
    command = execution.move_to_target(position, target, 0)
    np.testing.assert_allclose(command.sample(np.array([0.0, 1.0])), [position, target], atol=1e-12)
    for order in (1, 2):
        np.testing.assert_allclose(command.sample(np.array([0.0, 1.0]), order), 0, atol=1e-12)
    for order, limit in enumerate(execution.LIMITS, 1):
        slower_limit = limit / execution.ARM_MOTION_TIME_SCALE**order
        assert np.linalg.norm(command.sample(np.linspace(0, 1, 1001), order), axis=-1).max() <= slower_limit + 1e-10


def test_learned_gripper_request_ignores_future_xyz_drift_and_enforces_actual_dwell():
    executor = execution.PolicyExecutor(np.array([0.4, 0.02, 0.15]))
    actions = hold_actions(0)
    actions[1:, :3] += 0.001
    actions[10:, 3] = 1  # Model changes its mind before the physical close finishes.
    command = executor.plan(actions, observation_tick=0, now_tick=0, image_age_s=0.01, task="forward")
    assert (command.kind, command.ticks) == ("gripper", 78)
    np.testing.assert_array_equal(executor.position, [0.4, 0.02, 0.15])
    assert executor.plan(actions, observation_tick=77, now_tick=77, image_age_s=0.01, task="forward").kind == "wait"
    with pytest.raises(ValueError, match="fresh observation"):
        executor.plan(actions, observation_tick=0, now_tick=78, image_age_s=0.01, task="forward")
    assert (
        executor.plan(hold_actions(0), observation_tick=78, now_tick=78, image_age_s=0.01, task="forward").kind
        == "hold"
    )


@pytest.mark.parametrize(
    ("position", "target"),
    [
        # Paired close targets discarded in the three runs from 2026-09-15.
        ([0.496103510, -0.056187024, 0.089156030], [0.496183547, -0.056930061, 0.088339231]),
        ([0.496103186, -0.056296712, 0.089679252], [0.496113638, -0.056439048, 0.089165385]),
        ([0.496109748, -0.057193626, 0.089116981], [0.496114534, -0.057111330, 0.088814838]),
    ],
)
@pytest.mark.parametrize("jaw_open", [False, True])
def test_gripper_reaches_paired_xyz_before_actuating(position, target, jaw_open):
    executor = execution.PolicyExecutor(np.array(position), jaw_open=not jaw_open)
    actions = np.tile([*target, float(jaw_open)], (63, 1))
    actions[:4, :3] = position  # These rows elapsed during inference.
    actions[5:, 2] += 0.02  # Later lift references must not move the grasp point.
    command = executor.plan(actions, observation_tick=0, now_tick=4, image_age_s=0.01, task="forward")
    assert command.kind == "cartesian"
    np.testing.assert_allclose(command.sample(np.array([0.0, 1.0])), [position, target], atol=1e-12)
    assert executor.jaw_open is not jaw_open
    assert executor.pending_jaw_open is jaw_open
    end = command.start_tick + command.ticks
    assert executor.plan(actions, observation_tick=0, now_tick=end - 1, image_age_s=0.01, task="forward").kind == "wait"
    command = executor.plan(actions, observation_tick=0, now_tick=end, image_age_s=0.01, task="forward")
    assert command.kind == "gripper"
    assert command.jaw_open is jaw_open
    assert command.ticks == (30 if jaw_open else 78)
    assert executor.pending_jaw_open is None
    np.testing.assert_allclose(executor.position, target)
    with pytest.raises(ValueError, match="fresh observation"):
        executor.plan(actions, observation_tick=0, now_tick=end + command.ticks, image_age_s=0.01, task="forward")


def test_target_selection_stops_before_upcoming_jaw_transition():
    executor = execution.PolicyExecutor(np.array([0.4, 0.02, 0.15]))
    actions = hold_actions()
    actions[:, 0] += np.arange(1, 64) * 0.001
    actions[3:, 3] = 0
    command = executor.plan(actions, observation_tick=0, now_tick=0, image_age_s=0.01, task="forward")
    assert command.kind == "cartesian"
    np.testing.assert_allclose(command.sample(np.array([1.0]))[0], actions[2, :3])


@pytest.mark.parametrize(("age", "tick"), [(0.051, 0), (0.01, 63), (0.01, -1)])
def test_policy_executor_rejects_stale_or_expired_actions(age, tick):
    executor = execution.PolicyExecutor(np.array([0.4, 0.02, 0.15]))
    with pytest.raises(ValueError, match="[Ss]tale|expired|[Ii]nvalid"):
        executor.plan(hold_actions(), observation_tick=0, now_tick=tick, image_age_s=age, task="forward")


def test_target_outside_workspace_is_still_rejected():
    from examples.hanoi.deployment.hardware import validate_trajectory

    command = execution.move_to_target(np.array([0.4, 0.02, 0.15]), np.array([0.6, 0.02, 0.15]), 0)
    with pytest.raises(ValueError, match="workspace"):
        validate_trajectory(command, np.array([0.3, -0.1, 0.1]), np.array([0.5, 0.1, 0.2]))
