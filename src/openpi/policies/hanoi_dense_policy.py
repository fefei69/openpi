"""Dense contract five: every fresh recorded row is an observation; labels are future commanded poses."""

import dataclasses
import hashlib
import json
import pathlib

import numpy as np

from openpi import transforms
from openpi.policies import hanoi_policy

HORIZON = 30
FRAMESKIP = 3
EXECUTION_PREFIX = 3
REFERENCE_RATE_HZ = 10
SAMPLING_STEPS = 10
# Frames are read from the raw recording; this label only names the normalization assets and loader branch.
REPO_ID = "local/hanoi_dense_v5_raw"
ASSET_ID = "local/hanoi_dense_v5"
CONFIG_NAME = "pi05_hanoi_dense_aaaa_to_cccc"
RECORDING = "hanoi_wm_roundtrip_20260915_223424"
PROMPT = hanoi_policy.PROMPTS["aaaa_to_cccc"]
CONTRACT = {
    "version": 5,
    "robot": "trossen_wxai_single",
    "reference_rate_hz": REFERENCE_RATE_HZ,
    "action_horizon": HORIZON,
    "execution_prefix": EXECUTION_PREFIX,
    "state": [*[f"joint_{i}_rad" for i in range(6)], "jaw_stroke_m"],
    "joint_order": "trossen_arm_driver_arm_indices_0_to_5",
    "actions": ["reference_x_m", "reference_y_m", "reference_z_m", "jaw_open_intent"],
    "internal_xyz_encoding": "absolute",
    "frame": "commissioned_base_tool_frame",
    "orientation_rpy_rad": [0.0, np.pi / 4, 0.0],
    "rgb_topic": hanoi_policy.CONTRACT["rgb_topic"],
    "rgb_crop_xywh": [151, 90, 360, 360],
    "max_image_age_s": 0.05,
    "jaw_open_stroke_m": 0.034,
    "jaw_open_duration_s": 1.0,
    "jaw_close_effort_n": -20.0,
    "jaw_close_duration_s": 1.2,
    "jaw_close_settle_s": 0.2,
    "training_observation_alignment": "every non-stale row; chunk starts three rows after the observation",
    "training_deployment_timing": "moving observations in training; asynchronous chunk execution at deployment",
    "recording": RECORDING,
    "prompt": PROMPT,
}

# Comparison variants share the contract except for the chunk length; each has its own archive root.
VARIANTS = {
    CONFIG_NAME: {"horizon": HORIZON, "data_root": "data/hanoi/dense_v5_pi05"},
    "pi05_hanoi_dense_h16_aaaa_to_cccc": {"horizon": 16, "data_root": "data/hanoi/dense_v5_pi05_h16"},
}


def contract_for(config_name: str) -> dict:
    return {**CONTRACT, "action_horizon": VARIANTS[config_name]["horizon"]}


@dataclasses.dataclass(frozen=True)
class HanoiDenseInputs(transforms.DataTransformFn):
    """Single external camera, seven measured joint values, and absolute reference-pose labels."""

    horizon: int = HORIZON

    def __call__(self, data: dict) -> dict:
        image = hanoi_policy.parse_image(data["observation/image"])
        state = np.asarray(data["observation/state"], dtype=np.float32)
        if state.shape != (7,) or not np.isfinite(state).all():
            raise ValueError("Dense Hanoi state requires six joint angles and the measured jaw stroke")
        result = {
            "state": state.copy(),
            "image": {
                "base_0_rgb": image,
                "left_wrist_0_rgb": np.zeros_like(image),
                "right_wrist_0_rgb": np.zeros_like(image),
            },
            "image_mask": {"base_0_rgb": np.True_, "left_wrist_0_rgb": np.False_, "right_wrist_0_rgb": np.False_},
        }
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.shape != (self.horizon, 4) or not np.isfinite(actions).all():
                raise ValueError(f"Dense Hanoi labels require {self.horizon} finite absolute reference poses")
            if not np.isin(actions[:, 3], (0, 1)).all():
                raise ValueError("Recorded jaw intent must be binary")
            result["actions"] = actions.copy()
        if "prompt" in data:
            result["prompt"] = data["prompt"]
        return result


@dataclasses.dataclass(frozen=True)
class HanoiDenseOutputs(transforms.DataTransformFn):
    """Absolute reference poses with the jaw intent thresholded, plus the timing the client executes with."""

    horizon: int = HORIZON

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        if actions.ndim < 2 or actions.shape[-2] != self.horizon or actions.shape[-1] < 4:
            raise ValueError(f"Dense Hanoi policy must return {self.horizon} reference poses")
        actions = np.array(actions[..., :4], dtype=np.float32)
        if not np.isfinite(actions).all():
            raise ValueError("Dense Hanoi prediction is not finite")
        actions[..., 3] = (actions[..., 3] >= 0.5).astype(np.float32)
        return {"actions": actions, "reference_rate_hz": REFERENCE_RATE_HZ, "execution_prefix": EXECUTION_PREFIX}


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def serving_metadata(train_config, checkpoint_dir: pathlib.Path | str) -> dict:
    """Identity published by the policy server next to the static contract."""
    checkpoint_dir = pathlib.Path(checkpoint_dir)
    export = checkpoint_dir / "export.json"
    normalization = checkpoint_dir / "assets" / ASSET_ID / "norm_stats.json"
    return {
        **train_config.policy_metadata,
        "hanoi_dense": {
            "contract": contract_for(train_config.name),
            "prompt": PROMPT,
            "config_name": train_config.name,
            "checkpoint": str(checkpoint_dir.resolve()),
            "export_sha256": _sha256(export) if export.exists() else None,
            "export_step": json.loads(export.read_text())["step"] if export.exists() else None,
            "normalization_sha256": _sha256(normalization) if normalization.exists() else None,
            "num_steps": SAMPLING_STEPS,
        },
    }
