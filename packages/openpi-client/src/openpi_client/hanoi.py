"""Dependency-light Hanoi contract shared by training and robot clients."""

import cv2
import numpy as np

PROMPTS = {
    "aaaa_to_cccc": "Move all four rings from peg A to peg C following Tower of Hanoi rules.",
    "cccc_to_aaaa": "Move all four rings from peg C to peg A following Tower of Hanoi rules.",
}
REPO_ID = "local/hanoi_roundtrip_20260910"
CONTRACT = {
    "version": 1,
    "robot": "trossen_wxai_single",
    "reference_rate_hz": 30,
    "action_horizon": 63,
    "execution_prefix": 9,
    "state": ["x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s", "jaw_stroke_m"],
    "actions": ["next_reference_x_m", "next_reference_y_m", "next_reference_z_m", "jaw_open_intent"],
    "frame": "commissioned_base_tool_frame",
    "orientation_rpy_rad": [0.0, np.pi / 4, 0.0],
    "rgb_topic": "/camera/camera/color/image_raw",
    "rgb_crop_xywh": [151, 90, 360, 360],
    "max_image_age_s": 0.05,
    "jaw_open_stroke_m": 0.034,
    "jaw_open_duration_s": 1.0,
    "jaw_close_effort_n": -20.0,
    "jaw_close_duration_s": 1.2,
    "jaw_close_settle_s": 0.2,
}


def preprocess_camera(rgb: np.ndarray) -> np.ndarray:
    """Apply the collection crop once, to a full-resolution ROS RGB frame."""
    rgb = np.asarray(rgb)
    if rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8:
        raise ValueError("Camera input must be a 480x640 uint8 RGB image before cropping")
    return cv2.resize(rgb[90:450, 151:511], (224, 224), interpolation=cv2.INTER_AREA)


def decode_ros_rgb(data: bytes, *, height: int, width: int, step: int, encoding: str) -> np.ndarray:
    """Decode sensor_msgs/Image fields without importing ROS; honor row padding."""
    if encoding != "rgb8" or height != 480 or width != 640 or step < width * 3:
        raise ValueError("Expected a 640x480 rgb8 ROS image with a valid byte stride")
    if len(data) != height * step:
        raise ValueError("ROS image byte length does not match height * step")
    rows = np.frombuffer(data, dtype=np.uint8).reshape(height, step)
    return rows[:, : width * 3].reshape(height, width, 3).copy()


def parse_image(image: np.ndarray) -> np.ndarray:
    """Recover exact RGB bytes from LeRobot's float CHW or serving's uint8 HWC."""
    image = np.asarray(image)
    if image.shape == (3, 224, 224):
        image = np.moveaxis(image, 0, -1)
    if image.shape != (224, 224, 3):
        raise ValueError(f"Expected an already cropped 224x224 RGB image, got {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError("Floating point RGB images must be finite and within [0, 1]")
        image = np.rint(image * 255).astype(np.uint8)
    if image.dtype != np.uint8:
        raise ValueError("RGB images must be uint8 or normalized floating point")
    return image
