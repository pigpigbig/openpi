"""
Train a KBNN head on frozen pi0.5 (pi05_libero) features.

Requirements:
- Torch-converted pi05_libero checkpoint (from examples/convert_jax_model_to_pytorch.py).
- KBNN2.py and weight_initialization.py available on PYTHONPATH (placed in repo root).
- Dataset collected via collect_kbnn_data.py: data/libero/kbnn_dataset/env_*/ep_*.npz

This script:
- Loads the frozen pi05 model (Torch).
- Extracts the 1024-d pre-action_head features for camshift_images via a model helper
  `encode_for_actions(images, wrist_images, states)` that must be implemented in PI0Pytorch.
- Initializes KBNN from action_out_proj weights using weight_initialization.initialize_weights.
- Trains only KBNN on (camshift_feature -> action) pairs.

Run example:
    python scripts/train_kbnn.py \\
      --checkpoint-dir /media/data-ssd-2/qiaoan_ckpt/pi05_libero_pytorch \\
      --kbnn-weights kbnn_weights \\
      --data-root data/libero/kbnn_dataset \\
      --geometry 1024,2050,2050,32 \\
      --device cuda
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Ensure repo root on path so KBNN2.py and weight_initialization.py are importable.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from KBNN2 import KBNN
from weight_initialization import initialize_weights
from openpi.models_pytorch import pi0_pytorch
from openpi.models import pi0_config


class KBNNDataset(Dataset):
    def __init__(self, root: str):
        self.files = sorted(glob.glob(os.path.join(root, "env_*", "ep_*.npz")))
        if not self.files:
            raise ValueError(f"No npz files found under {root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        camshift = data["camshift_images"]  # (T, H, W, 3) uint8
        wrist = data["wrist_images"]        # (T, H, W, 3) uint8
        states = data["states"]             # (T, 8)
        actions = data["actions"]           # (T, 7)
        return camshift, wrist, states, actions


def load_action_head_with_bias(weight_path: str, bias_path: str) -> torch.Tensor:
    w = np.load(weight_path)  # (1024, 32)
    b = np.load(bias_path)    # (32,)
    w_with_bias = np.vstack([w, b[None, :]])  # (1025, 32) in y = xW convention
    return torch.tensor(w_with_bias, dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", required=True, help="Torch pi05_libero checkpoint dir")
    ap.add_argument("--kbnn-weights", default="kbnn_weights", help="Dir with action_out_proj_weight.npy and bias.npy")
    ap.add_argument("--data-root", default="data/libero/kbnn_dataset", help="Root of collected npz data")
    ap.add_argument("--geometry", default="1024,2050,2050,32", help="Comma-separated layer sizes (input excludes bias)")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"

    # Load frozen pi05 model (Torch)
    cfg = pi0_config.Pi0Config(pi05=True)
    model = pi0_pytorch.PI0Pytorch(cfg).to(device)
    # Load weights
    ckpt = torch.load(os.path.join(args.checkpoint_dir, "pytorch_model.bin"), map_location=device)
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Geometry and KBNN init
    geom = [int(x) for x in args.geometry.split(",")]
    w_path = Path(args.kbnn_weights) / "action_out_proj_weight.npy"
    b_path = Path(args.kbnn_weights) / "action_out_proj_bias.npy"
    w_with_bias = load_action_head_with_bias(str(w_path), str(b_path))
    init_weights = initialize_weights(w_with_bias, [geom[0] + 1, *geom[1:]])

    kbnn = KBNN([geom[0], *geom[1:]], dtype=torch.float32, device=device)
    # Load init weights into KBNN.mws
    for i, mw in enumerate(init_weights):
        kbnn.mws[i] = mw.to(device)

    ds = KBNNDataset(args.data_root)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    # Optimizer over KBNN means only
    opt = torch.optim.Adam([w for w in kbnn.mws], lr=args.lr)

    # Training loop: for each sample, extract features from camshift images via model.encode_for_actions
    # NOTE: encode_for_actions must be implemented in PI0Pytorch to return the 1024-d pre-head features.
    for epoch in range(args.epochs):
        for camshift, wrist, states, actions in dl:
            B, T = actions.shape[:2]
            # shape: (B, T, ...)
            camshift = camshift.numpy()
            wrist = wrist.numpy()
            states_np = states.numpy()
            acts = torch.as_tensor(actions.view(B * T, -1), device=device, dtype=torch.float32)

            # features: (B*T, 1024) expected
            if not hasattr(model, "encode_for_actions"):
                raise AttributeError("PI0Pytorch missing encode_for_actions; please implement to return 1024-d features.")
            feats = model.encode_for_actions(camshift, wrist, states_np)
            feats = feats.to(device)

            opt.zero_grad()
            loss_accum = 0.0
            for i in range(feats.shape[0]):
                kbnn.forward(feats[i])
                kbnn.backward(acts[i])
                loss_accum += torch.sum((feats[i]) * 0)  # dummy tensor to keep graph; updates happen inside KBNN
            loss_accum.backward()
            opt.step()
        print(f"Epoch {epoch+1}/{args.epochs} done")

    torch.save({"mws": [w.cpu() for w in kbnn.mws], "sws": [s.cpu() for s in kbnn.sws]}, "kbnn_checkpoint.pt")
    print("Saved kbnn_checkpoint.pt")


if __name__ == "__main__":
    main()
