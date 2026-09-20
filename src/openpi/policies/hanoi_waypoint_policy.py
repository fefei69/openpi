"""Version-four joint-state Hanoi contract using recorded motion-leg destinations."""

from openpi.policies import hanoi_joint_policy

HORIZON = hanoi_joint_policy.HORIZON
REPO_ID = "local/hanoi_waypoint_20260915_aaaa_to_cccc"
ASSET_ID = "local/hanoi_waypoint_v4"
CONFIG_NAME = "pi05_hanoi_waypoint_aaaa_to_cccc"
CONTRACT = {
    **hanoi_joint_policy.CONTRACT,
    "version": 4,
    "target_extraction": "recorded_leg_endpoints_and_gripper_events_merge_arrive_then_grip",
    "intermediate_motion_noise": "present_in_recorded_observations_not_predicted_as_arbitrary_destinations",
    "recorded_path_deviation_budget_m": 0.0025,
}
