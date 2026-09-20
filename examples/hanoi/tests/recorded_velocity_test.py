"""Recorded velocity substitution and phase alignment, without hardware."""

import json

import h5py
import numpy as np
from openpi_client import hanoi
import pytest

from examples.hanoi.deployment import client
from examples.hanoi.deployment.async_inference import InferenceRecorder
from examples.hanoi.deployment.async_inference import Observation
from examples.hanoi.deployment.recorded_velocity import RecordedVelocity


@pytest.fixture
def reference():
    xyz = np.array(
        [
            [0.49, 0, 0.19],
            [0.49, 0, 0.09],
            [0.49, 0, 0.087],
            [0.49, 0, 0.087],
            [0.49, 0, 0.087],
            [0.49, 0, 0.09],
            [0.49, 0, 0.15],
            [0.4, 0, 0.09],
            [0.4, 0, 0.09],
            [0.4, 0, 0.15],
        ],
        dtype=np.float32,
    )
    jaw = np.array([1, 1, 1, 0, 0, 0, 0, 0, 1, 1])
    states = np.c_[xyz, np.zeros((10, 3)), np.where(jaw, 0.034, 0.015)].astype(np.float32)
    states[:, 5] = [-0.01, -0.04, -0.001, 0, 0, 0.05, 0.08, -0.04, 0, 0.08]
    return RecordedVelocity(states, np.c_[xyz, jaw])


def observation(xyz=(0.49, 0, 0.09), *, tick=0):
    return Observation(
        tick,
        1.0,
        0.99,
        {
            "observation/state": np.array([*xyz, 0.001, 0.002, 0.003, 0.033], dtype=np.float32),
            "observation/image": np.full((224, 224, 3), 17, dtype=np.uint8),
            "prompt": hanoi.PROMPTS["aaaa_to_cccc"],
        },
    )


def test_only_velocity_changes_and_exact_inputs_and_measured_state_are_recorded(reference, tmp_path):
    live = observation()
    original = live.data["observation/state"].copy()
    modified = reference.apply(live)
    assert modified.velocity_override["reference_row"] == 1
    np.testing.assert_array_equal(modified.data["observation/state"][[0, 1, 2, 6]], original[[0, 1, 2, 6]])
    np.testing.assert_array_equal(modified.data["observation/state"][3:6], reference.states[1, 3:6])
    np.testing.assert_array_equal(live.data["observation/state"], original)
    assert modified.data["observation/image"] is live.data["observation/image"]
    assert modified.data["prompt"] == live.data["prompt"]
    assert modified.tick == live.tick
    assert modified.image_age_s == live.image_age_s
    InferenceRecorder(tmp_path).save_input(0, modified)
    with np.load(tmp_path / "inference_inputs/000000.npz") as archive:
        np.testing.assert_array_equal(archive["observation/state"], modified.data["observation/state"])
        np.testing.assert_array_equal(archive["observation/image"], live.data["observation/image"])
        assert set(archive.files) == set(live.data)
    log = json.loads((tmp_path / "inferences.jsonl").read_text())
    np.testing.assert_array_equal(log["velocity_override"]["measured_state"], original)
    assert log["velocity_override"]["reference_row"] == 1


def test_position_alignment_does_not_follow_wall_clock_or_cross_gripper_events(reference):
    first = reference.apply(observation(tick=0))
    delayed = reference.apply(observation(tick=900))
    assert first.velocity_override["reference_row"] == delayed.velocity_override["reference_row"] == 1
    assert delayed.data["observation/state"][5] < 0  # Descending while open.
    reference.gripper_command(jaw_open=False)
    lifted = reference.apply(observation(tick=1000))
    assert lifted.velocity_override["reference_row"] == 5  # Skip the completed closing dwell.
    assert lifted.data["observation/state"][5] > 0  # Same XYZ, now lifting with closed intent.
    reference.gripper_command(jaw_open=False)
    assert reference.phase == 1  # Repeated intent cannot advance twice.


def test_alignment_does_not_jump_backwards_to_a_previous_leg(reference):
    reference.apply(observation((0.49, 0, 0.087)))
    assert reference.row == 2
    assert reference.apply(observation((0.49, 0, 0.19))).velocity_override["reference_row"] == 2
    reference.gripper_command(jaw_open=False)
    reference.apply(observation((0.4, 0, 0.09)))
    assert reference.row == 7
    reference.gripper_command(jaw_open=True)
    assert reference.row == 9


def test_load_validates_episode_before_hardware(tmp_path, monkeypatch, reference):
    path = tmp_path / "episode.h5"
    actions = np.c_[reference.states[:, :3], [1, 1, 1, 0, 0, 0, 0, 0, 1, 1]]
    with h5py.File(path, "w") as episode:
        episode["proprio"] = reference.states
        episode["action_abs"] = actions
    path.with_suffix(".json").write_text(
        json.dumps({"contract": hanoi.CONTRACT, "prompt": hanoi.PROMPTS["aaaa_to_cccc"]})
    )
    loaded = RecordedVelocity.load(path)
    np.testing.assert_array_equal(loaded.states, reference.states)
    assert loaded.sha256 == reference.sha256
    path.with_suffix(".json").write_text(json.dumps({"contract": {}, "prompt": "wrong"}))

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid reference must fail before connecting hardware")

    monkeypatch.setattr(client, "TrossenArm", forbidden)
    monkeypatch.setattr(client, "RosCamera", forbidden)
    with pytest.raises(ValueError, match="matching forward"):
        client.main(client.Config(episode=path, velocity_source="recorded", output=tmp_path / "runs"))
