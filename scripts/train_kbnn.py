"""
Train a KBNN head on frozen pi0.5 (pi05_libero) features.

Inputs:
- A Torch-converted pi05_libero checkpoint (from `examples/convert_jax_model_to_pytorch.py`):
  expects `model.safetensors` in `--checkpoint-dir`.
- KBNN initialization weights extracted from the pi05 action head linear projection:
  `kbnn_weights/action_out_proj_weight.npy` and `kbnn_weights/action_out_proj_bias.npy`.
- A trajectory dataset collected by `examples/libero/collect_kbnn_data.py`:
  `data/libero/kbnn_dataset/env_*/ep_*.npz`

What it does:
- Freezes pi05 (Torch) and uses it only as a feature extractor.
- For each training sample (camshift view), computes the 1024-d feature right before
  the action output projection (the action-expert “action token” representation).
- Trains a small MLP (KBNN mean-weights only) to map that feature → 32-d action vector.

Notes:
- LIBERO actions are 7-D, but pi05 is configured with `action_dim=32`. The websocket
  policy server returns only the first 7 dims (the ones actually consumed by LIBERO).
  This script supervises only the first 7 dims: `loss(pred[:, :7], action7)`.
- Full-covariance KBNN Bayesian updates are infeasible at these dimensions (multi‑TiB
  covariance). This script uses `cov_mode=diag` and standard backprop on the mean weights.

Example:
  uv run scripts/train_kbnn.py \\
    --checkpoint-dir /media/data-ssd-2/qiaoan_ckpt/pi05_libero_pytorch \\
    --kbnn-weights kbnn_weights \\
    --data-root data/libero/kbnn_dataset \\
    --geometry 1025,2050,2050,32 \\
    --device cuda \\
    --epochs 5 \\
    --sample-steps 32 \\
    --output kbnn_checkpoint.pt
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset

# Ensure repo root on path so `KBNN2.py` and `weight_initialization.py` are importable.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from KBNN2 import KBNN  # noqa: E402
from openpi.models import pi0_config  # noqa: E402
from openpi.models_pytorch import pi0_pytorch  # noqa: E402
from weight_initialization import initialize_weights  # noqa: E402


class KBNNDataset(Dataset):
    def __init__(self, root: str):
        self.files = sorted(glob.glob(os.path.join(root, "env_*", "ep_*.npz")))
        if not self.files:
            raise ValueError(f"No npz files found under {root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        with np.load(self.files[idx]) as data:
            camshift = data["camshift_images"]  # (T, H, W, 3) uint8
            wrist = data["wrist_images"]  # (T, H, W, 3) uint8
            states = data["states"]  # (T, 8)
            actions = data["actions"]  # (T, 7)
            prompt = str(data["prompt"]) if "prompt" in data else ""
            env_id = int(data["env_id"]) if "env_id" in data else -1
        return camshift, wrist, states, actions, prompt, env_id


def _load_action_out_proj_with_bias(weight_path: str, bias_path: str) -> torch.Tensor:
    w = np.load(weight_path)  # (1024, 32) in y = xW convention (row-vector x)
    b = np.load(bias_path)  # (32,)
    w_with_bias = np.vstack([w, b[None, :]])  # (1025, 32) for x_aug=[x,1]
    return torch.tensor(w_with_bias, dtype=torch.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", required=True, help="Torch pi05_libero checkpoint dir (has model.safetensors)")
    ap.add_argument("--kbnn-weights", default="kbnn_weights", help="Dir with action_out_proj_weight.npy and bias.npy")
    ap.add_argument("--data-root", default="data/libero/kbnn_dataset", help="Root of collected npz data")
    ap.add_argument(
        "--geometry",
        default="1025,2050,2050,32",
        help="Comma-separated dims for `weight_initialization` (first dim includes bias).",
    )
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--sample-steps", type=int, default=32, help="Sample up to N steps per episode (0 = all).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", default="kbnn_checkpoint.pt", help="Path to save trained KBNN weights")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    rng = np.random.default_rng(args.seed)

    # Load frozen pi05 model (Torch).
    cfg = pi0_config.Pi0Config(pi05=True)
    model = pi0_pytorch.PI0Pytorch(cfg).to(device)
    safetensor_path = os.path.join(args.checkpoint_dir, "model.safetensors")
    if not os.path.isfile(safetensor_path):
        raise FileNotFoundError(f"Expected `model.safetensors` in {args.checkpoint_dir}")
    state_dict = load_file(safetensor_path, device=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    if not hasattr(model, "encode_for_actions"):
        raise AttributeError(
            "PI0Pytorch missing encode_for_actions; expected encode_for_actions(images, wrist_images, states)."
        )

    # KBNN geometry + initialization.
    geom_with_bias = [int(x) for x in args.geometry.split(",")]
    if len(geom_with_bias) < 2:
        raise ValueError("--geometry must contain at least input and output dims")
    input_with_bias = geom_with_bias[0]
    input_dim = input_with_bias - 1
    expected_feat_dim = int(getattr(model.action_out_proj, "in_features", input_dim))
    if input_dim != expected_feat_dim:
        raise ValueError(
            f"Geometry mismatch: geometry[0]-1={input_dim}, but pi05 action_out_proj expects {expected_feat_dim}."
        )
    kbnn_geom = [input_dim, *geom_with_bias[1:]]

    w_path = Path(args.kbnn_weights) / "action_out_proj_weight.npy"
    b_path = Path(args.kbnn_weights) / "action_out_proj_bias.npy"
    w_with_bias = _load_action_out_proj_with_bias(str(w_path), str(b_path))
    init_mws = initialize_weights(w_with_bias, geom_with_bias)

    kbnn = KBNN(kbnn_geom, dtype=torch.float32, device=device, cov_mode="diag")
    with torch.no_grad():
        for i, mw in enumerate(init_mws):
            kbnn.mws[i].copy_(mw.to(device=device, dtype=kbnn.mws[i].dtype))
    kbnn.mws = [torch.nn.Parameter(mw) for mw in kbnn.mws]

    # Data.
    ds = KBNNDataset(args.data_root)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    opt = torch.optim.Adam(kbnn.mws, lr=args.lr)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(args.epochs):
        running = 0.0
        n_batches = 0
        for camshift, wrist, states, actions, _prompt, _env_id in dl:
            # Shapes: (B, T, ...)
            B, T = actions.shape[:2]
            if B != 1 and args.sample_steps:
                raise ValueError("Use --batch-size 1 when --sample-steps != 0 (episodes can have variable lengths).")

            camshift_np = camshift.numpy()
            wrist_np = wrist.numpy()
            states_np = states.numpy()
            actions_np = actions.numpy()

            if args.sample_steps and args.sample_steps > 0 and T > args.sample_steps:
                idx = rng.choice(T, size=args.sample_steps, replace=False)
                idx.sort()
                camshift_np = camshift_np[:, idx]
                wrist_np = wrist_np[:, idx]
                states_np = states_np[:, idx]
                actions_np = actions_np[:, idx]

            acts7 = torch.as_tensor(actions_np.reshape(-1, actions_np.shape[-1]), device=device, dtype=torch.float32)

            with torch.no_grad():
                feats = model.encode_for_actions(camshift_np, wrist_np, states_np).to(device=device, dtype=torch.float32)

            preds32 = kbnn.forward_deterministic(feats)
            loss = loss_fn(preds32[:, :7], acts7)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running += float(loss.detach().cpu())
            n_batches += 1

        print(f"epoch {epoch+1}/{args.epochs} loss={running / max(1, n_batches):.6f}")

    torch.save(
        {
            "geometry_with_bias": geom_with_bias,
            "kbnn_geometry": kbnn_geom,
            "cov_mode": "diag",
            "mws": [w.detach().cpu() for w in kbnn.mws],
        },
        args.output,
    )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

