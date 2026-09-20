"""Joint observations with sparse Cartesian destinations for the recollected Hanoi data."""

import dataclasses

import numpy as np

from openpi import transforms
from openpi.policies import hanoi_policy
from openpi.policies import hanoi_sparse_policy

HORIZON = 8
REPO_ID = "local/hanoi_joint_20260915_aaaa_to_cccc"
ASSET_ID = "local/hanoi_joint_v3"
CONFIG_NAME = "pi05_hanoi_joint_aaaa_to_cccc"
CONTRACT = {
    **hanoi_sparse_policy.CONTRACT,
    "version": 3,
    "state": [*[f"joint_{i}_rad" for i in range(6)], "jaw_stroke_m"],
    "joint_order": "trossen_arm_driver_arm_indices_0_to_5",
    "cartesian_context": "observation/cartesian_position: measured XYZ metres from the same SDK snapshot",
    "cartesian_context_role": "action encoding/decoding only; not a model state input",
    "recording": "hanoi_wm_roundtrip_20260915_223424",
}


@dataclasses.dataclass(frozen=True)
class HanoiJointInputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        image = hanoi_policy.parse_image(data["observation/image"])
        state = np.asarray(data["observation/state"], dtype=np.float32)
        xyz = np.asarray(data["observation/cartesian_position"], dtype=np.float32)
        if state.shape != (7,) or not np.isfinite(state).all():
            raise ValueError("Joint Hanoi state requires six joint angles and measured gripper stroke")
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            raise ValueError("Joint Hanoi requires a separate measured Cartesian XYZ context")
        result = {
            "state": state.copy(),
            "cartesian_position": xyz.copy(),
            "image": {
                "base_0_rgb": image,
                "left_wrist_0_rgb": np.zeros_like(image),
                "right_wrist_0_rgb": np.zeros_like(image),
            },
            "image_mask": {"base_0_rgb": np.True_, "left_wrist_0_rgb": np.False_, "right_wrist_0_rgb": np.False_},
        }
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.shape != (HORIZON, 4) or not np.isfinite(actions).all():
                raise ValueError("Joint Hanoi labels require eight finite Cartesian/gripper targets")
            if not np.isin(actions[:, 3], (0, 1)).all():
                raise ValueError("Recorded gripper intent must be binary")
            result["actions"] = actions.copy()
        if "prompt" in data:
            result["prompt"] = data["prompt"]
        return result


@dataclasses.dataclass(frozen=True)
class CartesianActions(transforms.DataTransformFn):
    """Use measured XYZ context, never joint angles, to change action coordinates."""

    absolute: bool = False

    def __call__(self, data: dict) -> dict:
        if "actions" not in data:
            return data
        actions = np.asarray(data["actions"]).copy()
        xyz = np.asarray(data["cartesian_position"])
        if xyz.shape != (*actions.shape[:-2], 3) or not np.isfinite(xyz).all():
            raise ValueError("Cartesian context must match the action batch dimensions")
        actions[..., :3] += (1 if self.absolute else -1) * xyz[..., None, :]
        return {**data, "actions": actions}
