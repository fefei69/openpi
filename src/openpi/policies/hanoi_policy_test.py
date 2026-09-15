import numpy as np
import pytest

from openpi import transforms
from openpi.policies import hanoi_policy


def test_ros_stride_crop_and_lerobot_roundtrip():
    rng = np.random.default_rng(42)
    rgb = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
    padded = np.full((480, 1928), 123, dtype=np.uint8)
    padded[:, :1920] = rgb.reshape(480, 1920)
    decoded = hanoi_policy.decode_ros_rgb(padded.tobytes(), height=480, width=640, step=1928, encoding="rgb8")
    np.testing.assert_array_equal(decoded, rgb)
    cropped = hanoi_policy.preprocess_camera(decoded)
    lerobot_image = np.moveaxis(cropped.astype(np.float32) / 255, -1, 0)
    np.testing.assert_array_equal(hanoi_policy.parse_image(lerobot_image), cropped)
    with pytest.raises(ValueError, match="before cropping"):
        hanoi_policy.preprocess_camera(cropped)


def test_measured_state_only_and_missing_cameras():
    data = {"observation/image": np.zeros((224, 224, 3), dtype=np.uint8), "observation/state": np.zeros(7)}
    observation = hanoi_policy.HanoiInputs()(data)
    assert observation["image_mask"] == {
        "base_0_rgb": True,
        "left_wrist_0_rgb": False,
        "right_wrist_0_rgb": False,
    }
    data["observation/state"] = np.zeros(8)
    with pytest.raises(ValueError, match="seven"):
        hanoi_policy.HanoiInputs()(data)


def test_delta_quantile_inverse_preserves_xyz_and_jaw():
    state = np.array([0.4, 0.02, 0.15, 0.1, -0.1, 0.01, 0.034], np.float32)
    actions = np.tile(np.array([0.41, -0.01, 0.2, 1], np.float32), (63, 1))
    actions[31:, 3] = 0
    data = {
        "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/state": state,
        "actions": actions,
    }
    processed = hanoi_policy.HanoiInputs()(data)
    mask = transforms.make_bool_mask(3, -1)
    processed = transforms.DeltaActions(mask)(processed)
    np.testing.assert_allclose(processed["actions"][0], [0.01, -0.03, 0.05, 1], atol=1e-7)
    stats = {
        key: transforms.NormStats(mean=np.zeros(dim), std=np.ones(dim), q01=-np.ones(dim), q99=np.ones(dim))
        for key, dim in (("state", 7), ("actions", 4))
    }
    processed = transforms.Normalize(stats, use_quantiles=True)(processed)
    processed = transforms.PadStatesAndActions(32)(processed)
    processed = transforms.Unnormalize(stats, use_quantiles=True)(processed)
    processed = transforms.AbsoluteActions(mask)(processed)
    result = hanoi_policy.HanoiOutputs()(processed)
    np.testing.assert_allclose(result["actions"], actions, atol=1e-7)
    np.testing.assert_array_equal(data["actions"], actions)
