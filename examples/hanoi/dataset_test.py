import numpy as np

from examples.hanoi import dataset


def test_terminal_chunk_holds_last_reference():
    actions = np.arange(24, dtype=np.float32).reshape(6, 4)
    chunk = dataset.action_chunk(actions, 4)
    np.testing.assert_array_equal(chunk[0], actions[4])
    np.testing.assert_array_equal(chunk[1:], np.tile(actions[-1], (62, 1)))


def test_routes_reject_buried_ring_and_smaller_destination():
    assert dataset.legal_route(np.array([[0, 0, 0, 0], [1, 0, 0, 0]]))
    assert not dataset.legal_route(np.array([[0, 0, 0, 0], [0, 1, 0, 0]]))
    assert not dataset.legal_route(np.array([[1, 0, 0, 0], [1, 1, 0, 0]]))
    assert [dataset.split_for_pair(k) for k in (0, 39, 40, 44, 45, 49)] == [
        "train",
        "train",
        "val",
        "val",
        "test",
        "test",
    ]
