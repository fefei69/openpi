import numpy as np
import pytest

from examples.hanoi import execution


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
