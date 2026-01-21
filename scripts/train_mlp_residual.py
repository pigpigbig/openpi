"""
Train a small MLP on extracted (x, y) pairs.

Inputs:
  - Dataset dir from extract_kbnn_pairs.py (contains shard_*.npz + meta.pt).
Outputs:
  - Torch checkpoint containing MLP weights + meta for inference.
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class PairDataset(Dataset):
    def __init__(self, root: str):
        self.files = sorted(glob.glob(os.path.join(root, "shard_*.npz")))
        if not self.files:
            raise ValueError(f"No shard_*.npz files found under {root}")
        xs = []
        ys = []
        for path in self.files:
            data = np.load(path)
            xs.append(data["x"])
            ys.append(data["y"])
        self.x = np.concatenate(xs, axis=0)
        self.y = np.concatenate(ys, axis=0)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def build_mlp(input_dim: int, output_dim: int, hidden_dim: int, layers: int) -> torch.nn.Module:
    blocks = []
    in_dim = input_dim
    for _ in range(layers):
        blocks.append(torch.nn.Linear(in_dim, hidden_dim))
        blocks.append(torch.nn.ReLU())
        in_dim = hidden_dim
    blocks.append(torch.nn.Linear(in_dim, output_dim))
    return torch.nn.Sequential(*blocks)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True, help="Directory with shard_*.npz + meta.pt")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2, help="Number of hidden layers")
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--output", default="mlp_residual.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"

    meta_path = Path(args.dataset_dir) / "meta.pt"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta.pt at {meta_path}")
    meta = torch.load(meta_path, map_location="cpu")

    dataset = PairDataset(args.dataset_dir)
    n_total = len(dataset)
    n_val = int(n_total * args.val_split)
    n_train = n_total - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    input_dim = int(meta["proj_dim"])
    output_dim = int(meta["out_dim"])
    model = build_mlp(input_dim, output_dim, args.hidden, args.layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x = torch.as_tensor(x, device=device, dtype=torch.float32)
            y = torch.as_tensor(y, device=device, dtype=torch.float32)
            pred = model(x)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss.detach().cpu())
            global_step += 1
            if global_step % args.log_every == 0:
                logging.info(
                    "[mlp] epoch=%d step=%d train_loss=%.6f",
                    epoch + 1,
                    global_step,
                    running / args.log_every,
                )
                running = 0.0

        if n_val > 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for x, y in val_loader:
                    x = torch.as_tensor(x, device=device, dtype=torch.float32)
                    y = torch.as_tensor(y, device=device, dtype=torch.float32)
                    pred = model(x)
                    val_losses.append(float(loss_fn(pred, y).detach().cpu()))
            logging.info("[mlp] epoch=%d val_loss=%.6f", epoch + 1, float(np.mean(val_losses)))

    ckpt = {
        "mlp_state": model.state_dict(),
        "mlp_hidden": args.hidden,
        "mlp_layers": args.layers,
        "proj_matrix": meta["proj_matrix"],
        "feature_mean": meta["feature_mean"],
        "feature_std": meta["feature_std"],
        "target_mean": meta["target_mean"],
        "target_std": meta["target_std"],
        "residual_scale": meta["residual_scale"],
        "proj_dim": meta["proj_dim"],
        "out_dim": meta["out_dim"],
        "horizon": meta["horizon"],
        "feature_dim": meta["feature_dim"],
    }
    torch.save(ckpt, args.output)
    logging.info("[mlp] saved checkpoint to %s", args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
