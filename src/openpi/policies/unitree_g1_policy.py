import dataclasses
from collections.abc import Sequence

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _get_first_available(data: dict, keys: Sequence[str]) -> np.ndarray | None:
    for key in keys:
        if key in data:
            return data[key]
    return None


@dataclasses.dataclass(frozen=True)
class UnitreeG1Inputs(transforms.DataTransformFn):
    """Maps Unitree G1 LeRobot frames to OpenPI's expected observation format."""

    model_type: _model.ModelType
    base_image_keys: Sequence[str] = (
        "observation.images.cam_left_high",
        "observation.images.cam_high",
        "observation.images.cam_right_high",
    )
    left_wrist_image_keys: Sequence[str] = (
        "observation.images.cam_left_wrist",
        "observation.images.cam_wrist",
    )
    right_wrist_image_keys: Sequence[str] = ("observation.images.cam_right_wrist",)
    state_key: str = "observation.state"
    action_key: str = "action"

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data[self.state_key])

        base_image_raw = _get_first_available(data, self.base_image_keys)
        if base_image_raw is None:
            raise ValueError(f"Expected one of {self.base_image_keys}, got keys={tuple(sorted(data.keys()))}")
        base_image = _parse_image(base_image_raw)

        left_wrist_raw = _get_first_available(data, self.left_wrist_image_keys)
        right_wrist_raw = _get_first_available(data, self.right_wrist_image_keys)

        left_wrist_image = _parse_image(left_wrist_raw) if left_wrist_raw is not None else np.zeros_like(base_image)
        right_wrist_image = _parse_image(right_wrist_raw) if right_wrist_raw is not None else np.zeros_like(base_image)

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                left_mask = np.bool_(left_wrist_raw is not None)
                right_mask = np.bool_(right_wrist_raw is not None)
            case _model.ModelType.PI0_FAST:
                # FAST models do not mask padding cameras in the existing OpenPI adapters.
                left_mask = np.True_
                right_mask = np.True_
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": left_mask,
                "right_wrist_0_rgb": right_mask,
            },
        }

        actions = data.get("actions", data.get(self.action_key))
        if actions is not None:
            inputs["actions"] = np.asarray(actions)

        prompt = data.get("prompt", data.get("task"))
        if prompt is not None:
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class UnitreeG1Outputs(transforms.DataTransformFn):
    action_dim: int

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
