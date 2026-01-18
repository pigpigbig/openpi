#!/usr/bin/env python3
"""
Print the Frobenius norm of proj_matrix for a range of KBNN checkpoints.

Example:
  python scripts/print_proj_norms.py --start 0 --end 9
"""

import argparse
import os

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="kbnn_checkpoint_step", help="Checkpoint filename prefix")
    ap.add_argument("--suffix", default=".pt", help="Checkpoint filename suffix")
    ap.add_argument("--start", type=int, default=0, help="Start index (inclusive)")
    ap.add_argument("--end", type=int, default=9, help="End index (inclusive)")
    args = ap.parse_args()

    for idx in range(args.start, args.end + 1):
        path = f"{args.prefix}{idx}{args.suffix}"
        if not os.path.exists(path):
            print(f"{path}: MISSING")
            continue
        ckpt = torch.load(path, map_location="cpu")
        proj = ckpt.get("proj_matrix")
        if proj is None:
            print(f"{path}: proj_matrix MISSING")
            continue
        norm = torch.linalg.norm(proj).item()
        print(f"{path}: proj_matrix_norm={norm:.6f}")


if __name__ == "__main__":
    main()
