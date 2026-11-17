"""OpenPI observation/action transforms for the diagnostic OpenArm scene.

Both camera inputs are fixed external views. The showcase front view occupies OpenPI's
left-wrist image slot solely as a model input convention; it is not wrist data.
The adapter requires OpenArm-specific training and normalization statistics.
It does not make the existing LIBERO checkpoint compatible with OpenArm.
"""
from dataclasses import dataclass

import numpy as np


ACTION_LOW = np.array([-1.39626, -0.174533, -1.5708, 0.0, -1.5708, -0.785398, -1.5708, -0.7854], dtype=np.float32)
ACTION_HIGH = np.array([3.49066, 3.31613, 1.5708, 2.44346, 1.5708, 0.785398, 1.5708, 0.0], dtype=np.float32)


def _image(value):
    arr = np.asarray(value)
    if arr.ndim != 3:
        raise ValueError(f"Expected a three-dimensional image, got {arr.shape}")
    if arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = arr.transpose(1, 2, 0)
    if arr.shape[-1] != 3:
        raise ValueError(f"Expected RGB image, got {arr.shape}")
    if np.issubdtype(arr.dtype, np.floating):
        if not np.isfinite(arr).all() or np.any(arr < 0) or np.any(arr > 1):
            raise ValueError("Floating point images must be finite and in [0, 1]")
        arr = np.rint(arr * 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        raise ValueError(f"Expected uint8 or normalized float image, got {arr.dtype}")
    return np.ascontiguousarray(arr)


@dataclass(frozen=True)
class OpenArmInputs:
    """Convert repacked LeRobot samples or live observations to OpenPI inputs."""

    def __call__(self, data):
        state = np.asarray(data["observation/state"], dtype=np.float32)
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError("OpenArm state must contain 8 finite positions")
        table = _image(data["observation/table_image"])
        front = _image(data["observation/front_image"])
        if table.shape != front.shape:
            raise ValueError("Camera images must have matching shapes")
        result = {
            "state": state,
            "image": {"base_0_rgb": table, "left_wrist_0_rgb": front,
                      "right_wrist_0_rgb": np.zeros_like(table)},
            "image_mask": {"base_0_rgb": np.True_, "left_wrist_0_rgb": np.True_,
                           "right_wrist_0_rgb": np.False_},
        }
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.ndim < 1 or actions.shape[-1] != 8 or not np.isfinite(actions).all():
                raise ValueError("Training actions must end in 8 finite actuator commands")
            result["actions"] = actions
        if "prompt" in data:
            result["prompt"] = data["prompt"]
        return result


@dataclass(frozen=True)
class OpenArmOutputs:
    """Return absolute joint/gripper commands clipped to simulation limits."""

    def __call__(self, data):
        actions = np.asarray(data["actions"], dtype=np.float32)
        if actions.ndim < 1 or actions.shape[-1] < 8 or not np.isfinite(actions).all():
            raise ValueError("Model actions must have at least 8 finite coordinates")
        return {"actions": np.clip(actions[..., :8], ACTION_LOW, ACTION_HIGH)}
