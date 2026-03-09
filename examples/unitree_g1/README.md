# Unitree G1 Finetuning

This is the narrowest OpenPI path for fine-tuning `pi05_base` on public Unitree G1 data without rewriting the trainer:

1. Keep using the Unitree LeRobot datasets as-is.
2. Use the OpenPI `LeRobotDataset` loader.
3. Map Unitree's cameras and joint vectors into OpenPI's 3-camera, padded-action format.

The repository now includes an example config for the public dexterous-hand dataset:

- Dataset: `unitreerobotics/G1_Dex3_ToastedBread_Dataset`
- OpenPI config: `pi05_unitree_g1_dex3_toastedbread`

There is also a lighter dex1 example:

- Dataset: `unitreerobotics/G1_Dex1_MountCamera_Dataset`
- OpenPI config: `pi05_unitree_g1_dex1_mount_camera`

The adapter assumes:

- `observation.state` is the current G1 joint state.
- `action` is an absolute joint-position target sequence from the dataset.
- `observation.images.cam_left_high` is the primary scene camera.
- `observation.images.cam_left_wrist` and `observation.images.cam_right_wrist` are the wrist cameras.

OpenPI only supports 3 image slots for `pi0/pi05`, so this adapter uses:

- `cam_left_high` (or `cam_high`) -> `base_0_rgb`
- `cam_left_wrist` -> `left_wrist_0_rgb`
- `cam_right_wrist` -> `right_wrist_0_rgb`

`cam_right_high` is ignored by default. If your dataset has a better primary head camera, change the
`base_image_keys` in `LeRobotUnitreeG1DataConfig`.

## Public datasets

The Unitree Hugging Face org currently exposes multiple G1 datasets. Good starting points:

- `unitreerobotics/G1_Dex3_ToastedBread_Dataset`
- `unitreerobotics/G1_Dex1_MountCamera_Dataset`
- The `unitreerobotics/G1_Dex3_datasets` collection, which groups 13 Dex3 datasets

If you want a broader multi-task policy, merge several Unitree datasets into one LeRobot repo first, then point the
same config at that merged repo.

## Training

Compute normalization statistics first:

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_unitree_g1_dex3_toastedbread
```

Run JAX training:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_unitree_g1_dex3_toastedbread \
  --exp-name g1_toastedbread_ft \
  --overwrite
```

The config uses `pi05_base` weights and pads the 28-D Dex3 state/action vectors to OpenPI's 32-D `pi05` action space.

## Switching datasets

For another Unitree G1 dataset, copy one of the example config entries in
`src/openpi/training/config.py` and change:

- `data.repo_id`
- `data.action_dim`
- `data.base_image_keys`, if the preferred head camera differs
