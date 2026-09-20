import numpy as np
import pytest

from examples.hanoi.evaluation import metrics as evaluation


def test_physical_metrics_ignore_terminal_padding_and_use_both_jaw_classes():
    truth = np.zeros((2, 3, 4))
    truth[0, 1:, 3] = 1
    predicted = truth.copy()
    predicted[0, :, 0] = [0.001, 0.002, 0.003]
    predicted[1, :, 0] = [0.004, 100, 100]
    predicted[0, 1, 3] = 0
    valid = np.array([[True, True, True], [True, False, False]])
    metrics = evaluation.Metrics()
    metrics.update(predicted, truth, valid)
    result = metrics.result()
    assert result["first_xyz_mm"] == pytest.approx(2.5)
    assert result["mean_valid_xyz_mm"] == pytest.approx(3.0)
    assert result["last_valid_xyz_mm"] == pytest.approx(3.5)
    assert result["jaw_confusion_true_rows_predicted_columns"] == [[2, 0], [1, 1]]
    assert result["jaw_balanced_accuracy"] == 0.75


def test_partial_batch_keeps_real_count_and_sample_selection_is_episode_balanced():
    batch = {"state": np.arange(6).reshape(3, 2), "image": {"base": np.ones((3, 2, 2, 3))}}
    padded, count = evaluation.pad_batch(batch, 4)
    assert count == 3
    np.testing.assert_array_equal(padded["state"][:count], batch["state"])
    np.testing.assert_array_equal(padded["state"][3], batch["state"][-1])
    indices = np.r_[np.arange(10), np.array([12, 19])]
    episodes = [{"global_start": 0, "length": 10}, {"global_start": 10, "length": 10}]
    np.testing.assert_array_equal(evaluation.sampled_indices(indices, episodes, 3), [0, 4, 9, 12, 19])
