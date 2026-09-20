"""Version-two Hanoi observations and fixed chunks of sparse Cartesian goals."""

import dataclasses

import numpy as np

from openpi import transforms
from openpi.policies import hanoi_policy

HORIZON = 8
ASSET_ID = "local/hanoi_sparse_v2"
CONTRACT = {
    **hanoi_policy.CONTRACT,
    "version": 2,
    "action_horizon": HORIZON,
    "execution_prefix": 1,
    "reference_rate_hz": None,
    "state": ["x_m", "y_m", "z_m", "jaw_stroke_m"],
    "actions": ["destination_x_m", "destination_y_m", "destination_z_m", "jaw_open_after_arrival"],
    "internal_xyz_encoding": "relative_to_measured_observation_xyz",
    "waypoint_completion": "arrival_and_controller_ready_then_changed_gripper_intent_and_dwell",
    "inference_timing": "fresh_observation_after_committed_waypoint_completion",
    "training_observation_alignment": "nearby_real_moving_observation_after_nominal_reference_and_gripper_dwell",
    "training_deployment_timing": "explicit_moving_demonstration_to_stopped_inference_approximation",
    "elapsed_time_target_skipping": False,
    "terminal_padding": "repeat_final_target_in_loss_exclude_padding_from_accuracy",
    "termination": "external_goal_confirmation",
    "arrival_tolerance_m": None,
}


@dataclasses.dataclass(frozen=True)
class HanoiSparseInputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        image = hanoi_policy.parse_image(data["observation/image"])
        state = np.asarray(data["observation/state"], dtype=np.float32)
        if state.shape != (4,) or not np.isfinite(state).all():
            raise ValueError("Sparse Hanoi state requires measured XYZ and gripper stroke (four values)")
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
            if actions.shape != (HORIZON, 4) or not np.isfinite(actions).all():
                raise ValueError("Sparse Hanoi labels require eight finite XYZ/gripper targets")
            if not np.isin(actions[:, 3], (0, 1)).all():
                raise ValueError("Recorded gripper intent must be binary")
            result["actions"] = actions.copy()
        if "prompt" in data:
            result["prompt"] = data["prompt"]
        return result


@dataclasses.dataclass(frozen=True)
class HanoiSparseOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        if actions.ndim < 2 or actions.shape[-2] != HORIZON or actions.shape[-1] < 4:
            raise ValueError("Sparse Hanoi policy must return eight targets")
        if not np.isfinite(actions[..., :4]).all():
            raise ValueError("Sparse Hanoi prediction is not finite")
        return {"actions": actions[..., :4]}
