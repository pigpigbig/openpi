import numpy as np

from openpi.models import model as _model
from openpi.policies import unitree_g1_policy


def test_unitree_g1_inputs_maps_images_and_actions():
    transform = unitree_g1_policy.UnitreeG1Inputs(model_type=_model.ModelType.PI05)

    data = {
        "observation.state": np.arange(28, dtype=np.float32),
        "observation.images.cam_left_high": np.ones((3, 8, 6), dtype=np.float32),
        "observation.images.cam_left_wrist": np.full((8, 6, 3), 7, dtype=np.uint8),
        "observation.images.cam_right_wrist": np.full((8, 6, 3), 9, dtype=np.uint8),
        "action": np.ones((16, 28), dtype=np.float32),
        "prompt": "toast the bread",
    }

    out = transform(data)

    assert out["state"].shape == (28,)
    assert out["image"]["base_0_rgb"].shape == (8, 6, 3)
    assert out["image"]["left_wrist_0_rgb"].shape == (8, 6, 3)
    assert out["image"]["right_wrist_0_rgb"].shape == (8, 6, 3)
    assert bool(out["image_mask"]["left_wrist_0_rgb"])
    assert bool(out["image_mask"]["right_wrist_0_rgb"])
    assert out["actions"].shape == (16, 28)
    assert out["prompt"] == "toast the bread"


def test_unitree_g1_inputs_pads_missing_wrist_camera():
    transform = unitree_g1_policy.UnitreeG1Inputs(model_type=_model.ModelType.PI05)

    data = {
        "observation.state": np.arange(16, dtype=np.float32),
        "observation.images.cam_high": np.full((8, 6, 3), 5, dtype=np.uint8),
        "observation.images.cam_left_wrist": np.full((8, 6, 3), 7, dtype=np.uint8),
    }

    out = transform(data)

    assert bool(out["image_mask"]["left_wrist_0_rgb"])
    assert not bool(out["image_mask"]["right_wrist_0_rgb"])
    assert np.all(out["image"]["right_wrist_0_rgb"] == 0)


def test_unitree_g1_outputs_trim_padding_dims():
    transform = unitree_g1_policy.UnitreeG1Outputs(action_dim=28)

    data = {"actions": np.ones((16, 32), dtype=np.float32)}
    out = transform(data)

    assert out["actions"].shape == (16, 28)
