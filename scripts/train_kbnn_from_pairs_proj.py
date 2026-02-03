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
import math

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from KBNN2 import KBNN as KBNNFull


def _load_shards(dataset_dir: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(dataset_dir, "shard_*.npz")))
    if not files:
        raise ValueError(f"No shard_*.npz files found under {dataset_dir}")
    return files


def _save_kbnn(
    path: str,
    kbnn: KBNNFull,
    meta: dict,
    proj_dim: int,
    kbnn_hidden: int,
    out_dim: int,
    proj_matrix: torch.Tensor | None = None,
) -> None:
    torch.save(
        {
            "geometry_with_bias": [proj_dim + 1, kbnn_hidden, out_dim],
            "kbnn_geometry": [proj_dim, kbnn_hidden, out_dim],
            "cov_mode": getattr(kbnn, "cov_mode", "full"),
            "mws": [w.detach().cpu() for w in kbnn.mws],
            "feature_mean": meta["feature_mean"],
            "feature_std": meta["feature_std"],
            "proj_matrix": proj_matrix if proj_matrix is not None else meta["proj_matrix"],
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
    ap.add_argument("--kbnn-hidden", type=int, default=2048, help="Hidden dimension for KBNN")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle samples within each shard")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=0, help="Save checkpoint every N steps (0 disables)")
    ap.add_argument("--explode-norm", type=float, default=1e6, help="Stop and log if any weight norm exceeds this")
    ap.add_argument("--output", default="kbnn_from_pairs.pt")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"

    meta_path = Path(args.dataset_dir) / "meta.pt"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta.pt at {meta_path}")
    meta = torch.load(meta_path, map_location="cpu")

    proj_dim = 256
    feature_dim = int(meta["feature_dim"])
    horizon = int(meta["horizon"])
    out_dim = int(meta["out_dim"])
    kbnn_hidden = int(meta.get("kbnn_hidden", 64))
    if args.kbnn_hidden != kbnn_hidden:
        logging.info(
            "[kbnn_pairs] overriding kbnn_hidden from meta (%d) to %d",
            kbnn_hidden,
            args.kbnn_hidden,
        )
        kbnn_hidden = args.kbnn_hidden

    if proj_dim * kbnn_hidden * out_dim > 1_000_000:
        logging.warning(
            "[kbnn_pairs] KBNN2 full-covariance can be very memory heavy with proj_dim=%d, hidden=%d, out_dim=%d.",
            proj_dim,
            kbnn_hidden,
            out_dim,
        )

    kbnn = KBNNFull(
        [proj_dim, kbnn_hidden, out_dim],
        init_cov=args.init_cov,
        device=device,
    )

    shards = _load_shards(args.dataset_dir)

    global_step = 0
    explode = False
    # Project raw flat features into proj_dim (shape: [proj_dim, flat_dim])
    proj_matrix = np.transpose(np.random.randn(proj_dim, feature_dim * horizon).astype(np.float32) / math.sqrt(
        feature_dim * horizon
    ))
    for epoch in range(args.epochs):
        running = 0.0
        count = 0
        for shard in shards:
            data = np.load(shard)
            if data["x"].shape[1] != feature_dim * horizon:
                raise ValueError(
                    f"Expected raw flat features of shape (N, {feature_dim * horizon}) "
                    f"but got {data['x'].shape}. If your shards are already projected, "
                    "do not use train_kbnn_from_pairs_proj.py."
                )
            x_np = data["x"] @ proj_matrix
            y_np = data["y"]
            if args.shuffle:
                idx = np.random.permutation(x_np.shape[0])
                x_np = x_np[idx]
                y_np = y_np[idx]

            feature_mean = torch.as_tensor(x_np.mean(axis=0), dtype=torch.float32, device=device)
            feature_std = torch.as_tensor(x_np.std(axis=0), dtype=torch.float32, device=device)
            target_mean = torch.as_tensor(y_np.mean(axis=0), dtype=torch.float32, device=device)
            target_std = torch.as_tensor(y_np.std(axis=0), dtype=torch.float32, device=device)

            feature_std = torch.clamp(feature_std, min=1e-8)
            target_std = torch.clamp(target_std, min=1e-8)

            x = torch.as_tensor(x_np, device=device, dtype=torch.float32)
            y = torch.as_tensor(y_np, device=device, dtype=torch.float32)

            x = (x - feature_mean) / feature_std
            y = (y - target_mean) / target_std

            for i in range(x.shape[0]):
                xi = x[i]
                yi = y[i]
                cache = kbnn.forward(xi)
                pred = cache["mus"][-1]
                loss = torch.mean((pred - yi) ** 2)
                kbnn.backward(yi)
                mw_norms = [float(torch.linalg.norm(w).detach().cpu()) for w in kbnn.mws]
                max_mw_norm = max(mw_norms) if mw_norms else 0.0
                if not torch.isfinite(loss) or max_mw_norm > args.explode_norm:
                    logging.warning(
                        "[kbnn_pairs] explosion detected: epoch=%d step=%d shard=%s idx=%d loss=%s max_mw_norm=%.6e",
                        epoch + 1,
                        global_step + 1,
                        Path(shard).name,
                        i,
                        float(loss.detach().cpu()) if torch.isfinite(loss) else "nan/inf",
                        max_mw_norm,
                    )
                    explode_path = f"{Path(args.output).with_suffix('')}_explode_step{global_step + 1}.pt"
                    meta["feature_mean"] = feature_mean.detach().cpu()
                    meta["feature_std"] = feature_std.detach().cpu()
                    meta["target_mean"] = target_mean.detach().cpu()
                    meta["target_std"] = target_std.detach().cpu()
                    _save_kbnn(
                        explode_path,
                        kbnn,
                        meta,
                        proj_dim,
                        kbnn_hidden,
                        out_dim,
                        proj_matrix=torch.as_tensor(proj_matrix),
                    )
                    logging.warning("[kbnn_pairs] saved %s", explode_path)
                    explode = True
                    break
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
                    meta["feature_mean"] = feature_mean.detach().cpu()
                    meta["feature_std"] = feature_std.detach().cpu()
                    meta["target_mean"] = target_mean.detach().cpu()
                    meta["target_std"] = target_std.detach().cpu()
                    _save_kbnn(
                        save_path,
                        kbnn,
                        meta,
                        proj_dim,
                        kbnn_hidden,
                        out_dim,
                        proj_matrix=torch.as_tensor(proj_matrix),
                    )
                    logging.info("[kbnn_pairs] saved %s", save_path)
            if explode:
                break
        if explode:
            break

        if count > 0:
            logging.info(
                "[kbnn_pairs] epoch=%d avg_loss=%.6f",
                epoch + 1,
                running / max(count, 1),
            )

    meta["feature_mean"] = feature_mean.detach().cpu()
    meta["feature_std"] = feature_std.detach().cpu()
    meta["target_mean"] = target_mean.detach().cpu()
    meta["target_std"] = target_std.detach().cpu()
    _save_kbnn(args.output, kbnn, meta, proj_dim, kbnn_hidden, out_dim, proj_matrix=torch.as_tensor(proj_matrix))
    logging.info("[kbnn_pairs] saved %s", args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
