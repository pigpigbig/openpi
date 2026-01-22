"""
Train KBNN (Kalman-style updates) on extracted (x, y) pair shards.

Dataset format: output of scripts/extract_kbnn_pairs.py
  - shard_*.npz with x: (N, proj_dim), y: (N, out_dim)
  - meta.pt with proj_matrix, feature_mean/std, target_mean/std, residual_scale, etc.

This keeps the KBNN workflow separate from train_kbnn.py.
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from KBNN_old import KBNN as KBNNOld


def _load_shards(dataset_dir: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(dataset_dir, "shard_*.npz")))
    if not files:
        raise ValueError(f"No shard_*.npz files found under {dataset_dir}")
    return files


def _save_kbnn(
    path: str,
    kbnn: KBNNOld,
    meta: dict,
    proj_dim: int,
    kbnn_hidden: int,
    out_dim: int,
) -> None:
    torch.save(
        {
            "geometry_with_bias": [proj_dim + 1, kbnn_hidden, out_dim],
            "kbnn_geometry": [proj_dim, kbnn_hidden, out_dim],
            "cov_mode": getattr(kbnn, "cov_mode", "full"),
            "mws": [w.detach().cpu().T for w in kbnn.mw],
            "feature_mean": meta["feature_mean"],
            "feature_std": meta["feature_std"],
            "proj_matrix": meta["proj_matrix"],
            "proj_dim": proj_dim,
            "kbnn_hidden": kbnn_hidden,
            "kbnn_out_dim": out_dim,
            "residual_scale": meta.get("residual_scale", 1.0),
            "target_mean": meta.get("target_mean"),
            "target_std": meta.get("target_std"),
        },
        path,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True, help="Directory with shard_*.npz + meta.pt")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--init-cov", type=float, default=1e-2)
    ap.add_argument("--kbnn-noise", type=float, default=0.0, help="Noise term inside KBNN forward pass")
    ap.add_argument("--kbnn-normalise", action="store_true", help="Enable KBNN normalization (scales ma/Ca)")
    ap.add_argument("--kbnn-hidden", type=int, default=2048, help="Hidden dimension for KBNN")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle samples within each shard")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=0, help="Save checkpoint every N steps (0 disables)")
    ap.add_argument("--output", default="kbnn_from_pairs.pt")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"

    meta_path = Path(args.dataset_dir) / "meta.pt"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta.pt at {meta_path}")
    meta = torch.load(meta_path, map_location="cpu")

    proj_dim = int(meta["proj_dim"])
    out_dim = int(meta["out_dim"])
    kbnn_hidden = int(meta.get("kbnn_hidden", 64))
    if args.kbnn_hidden != kbnn_hidden:
        logging.info(
            "[kbnn_pairs] overriding kbnn_hidden from meta (%d) to %d",
            kbnn_hidden,
            args.kbnn_hidden,
        )
        kbnn_hidden = args.kbnn_hidden

    kbnn = KBNNOld(
        [proj_dim, kbnn_hidden, out_dim],
        act_fun=["relu", "relu", "linear"],
        no_bias=False,
        noise=args.kbnn_noise,
        normalise=args.kbnn_normalise,
        verbose=False,
        device=torch.device(device),
        init_cov=args.init_cov,
    )

    shards = _load_shards(args.dataset_dir)

    global_step = 0
    for epoch in range(args.epochs):
        running = 0.0
        count = 0
        for shard in shards:
            data = np.load(shard)
            x_np = data["x"]
            y_np = data["y"]
            if args.shuffle:
                idx = np.random.permutation(x_np.shape[0])
                x_np = x_np[idx]
                y_np = y_np[idx]
            x = torch.as_tensor(x_np, device=device, dtype=torch.float32)
            y = torch.as_tensor(y_np, device=device, dtype=torch.float32)

            for i in range(x.shape[0]):
                xi = x[i].unsqueeze(0)
                yi = y[i].unsqueeze(0)
                pred, _, _, _ = kbnn.single_forward_pass(xi, training=False)
                loss = torch.mean((pred - yi) ** 2)
                kbnn.train(xi, yi)
                running += float(loss.detach().cpu())
                count += 1
                global_step += 1
                if args.log_every > 0 and global_step % args.log_every == 0:
                    logging.info(
                        "[kbnn_pairs] epoch=%d step=%d loss=%.6f",
                        epoch + 1,
                        global_step,
                        running / max(count, 1),
                    )
                    running = 0.0
                    count = 0
                if args.save_every > 0 and global_step % args.save_every == 0:
                    save_path = f"{Path(args.output).with_suffix('')}_step{global_step}.pt"
                    _save_kbnn(save_path, kbnn, meta, proj_dim, kbnn_hidden, out_dim)
                    logging.info("[kbnn_pairs] saved %s", save_path)

        if count > 0:
            logging.info(
                "[kbnn_pairs] epoch=%d avg_loss=%.6f",
                epoch + 1,
                running / max(count, 1),
            )

    _save_kbnn(args.output, kbnn, meta, proj_dim, kbnn_hidden, out_dim)
    logging.info("[kbnn_pairs] saved %s", args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
