"""
Extract (x, y) pairs used by KBNN training from LIBERO .npz episodes.

x: projected features (proj_matrix @ flatten(action_out_proj input)), optionally normalized
y: residual target (noise - actions_t - base_out), optionally normalized and scaled

This dataset is meant for sanity-checking trainability with alternative models.
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import random
import math
from pathlib import Path

import numpy as np
import torch

from openpi.policies import policy_config as _policy_config
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config


class _HeadInputCatcher:
    def __init__(self):
        self.last_in = None

    def __call__(self, _module, inputs):
        self.last_in = inputs[0].detach()


def _load_episode_files(data_root: str, env_ids: list[int] | None = None) -> list[str]:
    if env_ids:
        files = []
        for env_id in env_ids:
            files.extend(sorted(glob.glob(os.path.join(data_root, f"env_{env_id:02d}", "ep_*.npz"))))
    else:
        files = sorted(glob.glob(os.path.join(data_root, "env_*", "ep_*.npz")))
    if not files:
        raise ValueError(f"No episode npz files found under {data_root}")
    return files


def _pad_actions_to_32(action7: np.ndarray) -> np.ndarray:
    out = np.zeros((action7.shape[0], 32), dtype=np.float32)
    out[:, :7] = action7.astype(np.float32)
    return out


def _make_action_chunk(actions32: np.ndarray, t: int, horizon: int) -> np.ndarray:
    T = actions32.shape[0]
    chunk = np.zeros((horizon, 32), dtype=np.float32)
    for k in range(horizon):
        idx = min(T - 1, t + k)
        chunk[k] = actions32[idx]
    return chunk


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", required=True, help="Converted PyTorch ckpt dir (model.safetensors)")
    ap.add_argument("--norm-stats-assets-dir", required=True, help="Assets dir containing <asset_id>/norm_stats.json")
    ap.add_argument("--policy-config", default="pi05_libero", help="OpenPI config name")
    ap.add_argument("--data-root", default="data/libero/kbnn_dataset", help="Dataset root")
    ap.add_argument("--output-dir", default="data/libero/kbnn_pairs", help="Output directory for x/y shards")
    ap.add_argument("--env-ids", type=int, nargs="*", default=None)
    ap.add_argument("--samples", type=int, default=0, help="Total samples to extract (0 = use all episodes once)")
    ap.add_argument("--proj-dim", type=int, default=256)
    ap.add_argument(
        "--no-proj",
        action="store_true",
        help="Skip projection matrix and use raw flattened features as x.",
    )
    ap.add_argument("--normalize-target", action="store_true", help="Normalize y using target mean/std")
    ap.add_argument("--no-feature-normalization", action="store_true", help="Disable feature normalization")
    ap.add_argument("--feature-stats-samples", type=int, default=0, help="Samples to estimate feature stats")
    ap.add_argument("--target-stats-samples", type=int, default=0, help="Samples to estimate target stats")
    ap.add_argument("--residual-scale", type=float, default=0.1, help="Scale applied to y residuals")
    ap.add_argument("--init-from", default=None, help="Optional KBNN checkpoint to reuse proj/stats")
    ap.add_argument("--shard-size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"

    train_config = _config.get_config(args.policy_config)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if data_config.asset_id is None:
        raise ValueError("Train config has no asset_id; cannot load norm stats.")
    norm_stats = _checkpoints.load_norm_stats(Path(args.norm_stats_assets_dir), data_config.asset_id)

    policy = _policy_config.create_trained_policy(
        train_config,
        args.checkpoint_dir,
        norm_stats=norm_stats,
        pytorch_device=device,
    )
    model = policy._model  # noqa: SLF001
    model.eval()

    catcher = _HeadInputCatcher()
    hook_handle = model.action_out_proj.register_forward_pre_hook(catcher)

    files = _load_episode_files(args.data_root, args.env_ids)
    horizon = int(getattr(train_config.model, "action_horizon", 10))
    feat_dim = 1024
    flat_dim = horizon * feat_dim
    out_dim = horizon * 32
    proj_dim = flat_dim if args.no_proj else args.proj_dim
    if args.no_proj:
        logging.info("[pairs] --no-proj enabled; using flat_dim=%d and skipping projection.", flat_dim)

    ckpt = None
    if args.init_from:
        ckpt = torch.load(args.init_from, map_location="cpu")

    if args.no_proj:
        if args.init_from and (ckpt.get("no_proj") is not True or ckpt.get("proj_matrix") is not None):
            raise ValueError(
                "--no-proj requires an init_from checkpoint created with --no-proj (no_proj=True)."
            )
        proj_matrix = None
    elif args.init_from and ckpt.get("proj_matrix") is not None:
        proj_matrix = ckpt["proj_matrix"].to(dtype=torch.float32, device=device)
        if proj_matrix.shape != (proj_dim, flat_dim):
            raise ValueError(
                "proj_matrix shape mismatch: "
                f"checkpoint has {tuple(proj_matrix.shape)}, expected {(proj_dim, flat_dim)}"
            )
    else:
        proj_matrix = torch.randn(proj_dim, flat_dim, device=device) / math.sqrt(flat_dim)

    def sample_training_point(ep_path: str | None = None):
        if ep_path is None:
            ep_path = random.choice(files)
        with np.load(ep_path) as data:
            camshift_images = data["camshift_images"]
            wrist_images = data["wrist_images"]
            states = data["states"]
            actions7 = data["actions"]
            prompt = str(data["prompt"]) if "prompt" in data else ""
        T = actions7.shape[0]
        t = random.randrange(T)
        obs = {
            "observation/image": camshift_images[t],
            "observation/wrist_image": wrist_images[t],
            "observation/state": states[t],
            "prompt": prompt,
        }
        actions32 = _pad_actions_to_32(actions7)
        act_chunk = _make_action_chunk(actions32, t, horizon)
        return obs, act_chunk

    def _prepare_inputs(obs):
        inputs = policy._input_transform(obs)  # noqa: SLF001
        allowed = {
            "state",
            "image",
            "image_mask",
            "tokenized_prompt",
            "tokenized_prompt_mask",
            "token_ar_mask",
            "token_loss_mask",
        }
        inputs = {k: v for k, v in inputs.items() if k in allowed}
        import jax

        inputs = jax.tree.map(lambda x: np.asarray(x), inputs)
        inputs_t = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(device)[None, ...], inputs)
        from openpi.models import model as _model

        return _model.Observation.from_dict(inputs_t)

    def _sample_feature(obs, act_chunk, *, noise=None, time=None):
        observation = _prepare_inputs(obs)
        actions_t = torch.from_numpy(act_chunk)[None, ...].to(device, dtype=torch.float32)
        if noise is None:
            noise = torch.randn_like(actions_t)
            noise[..., 7:] = 0.0
        if time is None:
            time = torch.rand((1,), device=device, dtype=torch.float32) * 0.999 + 0.001
        with torch.no_grad():
            _ = model(observation, actions_t, noise=noise, time=time)
        if catcher.last_in is None:
            raise RuntimeError("Failed to capture action_out_proj input.")
        return catcher.last_in.squeeze(0).to(dtype=torch.float32, device=device), noise, actions_t

    # Feature stats for normalization of projected features.
    if args.no_feature_normalization:
        feature_mean = torch.zeros(proj_dim, dtype=torch.float32, device=device)
        feature_std = torch.ones(proj_dim, dtype=torch.float32, device=device)
        logging.info("[pairs] feature normalization disabled (mean=0, std=1)")
    else:
        ckpt_feature_mean = ckpt.get("feature_mean") if ckpt else None
        ckpt_feature_std = ckpt.get("feature_std") if ckpt else None
        if ckpt_feature_mean is not None and ckpt_feature_std is not None:
            if ckpt_feature_mean.numel() != proj_dim or ckpt_feature_std.numel() != proj_dim:
                raise ValueError(
                    "feature stats size mismatch for --no-proj: "
                    f"checkpoint has {ckpt_feature_mean.numel()}, expected {proj_dim}."
                )
            feature_mean = ckpt_feature_mean.to(dtype=torch.float32, device=device)
            feature_std = ckpt_feature_std.to(dtype=torch.float32, device=device)
            logging.info("[pairs] using feature stats from %s", args.init_from)
        else:
            stats_samples = args.feature_stats_samples or min(len(files), 1000)
            sum_x = torch.zeros(proj_dim, dtype=torch.float64, device=device)
            sum_x2 = torch.zeros(proj_dim, dtype=torch.float64, device=device)
            for _ in range(stats_samples):
                obs, act_chunk = sample_training_point()
                feats, _, _ = _sample_feature(obs, act_chunk)
                x_flat = feats.reshape(-1)
                x_proj = x_flat if proj_matrix is None else (proj_matrix @ x_flat)
                sum_x += x_proj.double()
                sum_x2 += (x_proj.double() ** 2)
            feature_mean = (sum_x / stats_samples).float()
            feature_var = (sum_x2 / stats_samples - feature_mean.double() ** 2).clamp_min(1e-12)
            feature_std = torch.sqrt(feature_var).float().clamp_min(1e-6)
            logging.info(
                "[pairs] feature stats: count=%s mean_abs=%.6f std_mean=%.6f",
                stats_samples,
                float(feature_mean.abs().mean()),
                float(feature_std.mean()),
            )

    # Target stats for normalization of residual targets.
    if args.normalize_target:
        ckpt_target_mean = ckpt.get("target_mean") if ckpt else None
        ckpt_target_std = ckpt.get("target_std") if ckpt else None
        if ckpt_target_mean is not None and ckpt_target_std is not None:
            target_mean = ckpt_target_mean.to(dtype=torch.float32, device=device)
            target_std = ckpt_target_std.to(dtype=torch.float32, device=device)
            logging.info("[pairs] using target stats from %s", args.init_from)
        else:
            stats_samples = args.target_stats_samples or min(len(files), 1000)
            sum_y = torch.zeros(out_dim, dtype=torch.float64, device=device)
            sum_y2 = torch.zeros(out_dim, dtype=torch.float64, device=device)
            for _ in range(stats_samples):
                obs, act_chunk = sample_training_point()
                feats, noise, actions_t = _sample_feature(obs, act_chunk)
                base_out = model.action_out_proj(feats).detach()
                target = (noise - actions_t).squeeze(0)
                y_flat = (target - base_out).reshape(-1)
                sum_y += y_flat.double()
                sum_y2 += (y_flat.double() ** 2)
            target_mean = (sum_y / stats_samples).float()
            target_var = (sum_y2 / stats_samples - target_mean.double() ** 2).clamp_min(1e-12)
            target_std = torch.sqrt(target_var).float().clamp_min(1e-6)
            logging.info(
                "[pairs] target stats: count=%s mean_abs=%.6f std_mean=%.6f",
                stats_samples,
                float(target_mean.abs().mean()),
                float(target_std.mean()),
            )
    else:
        target_mean = torch.zeros(out_dim, dtype=torch.float32, device=device)
        target_std = torch.ones(out_dim, dtype=torch.float32, device=device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "meta.pt"
    torch.save(
        {
            "proj_matrix": proj_matrix.detach().cpu() if proj_matrix is not None else None,
            "feature_mean": feature_mean.detach().cpu(),
            "feature_std": feature_std.detach().cpu(),
            "target_mean": target_mean.detach().cpu(),
            "target_std": target_std.detach().cpu(),
            "residual_scale": args.residual_scale,
            "proj_dim": proj_dim,
            "out_dim": out_dim,
            "horizon": horizon,
            "feature_dim": feat_dim,
            "normalize_target": bool(args.normalize_target),
            "feature_normalization": not args.no_feature_normalization,
            "no_proj": bool(args.no_proj),
            "policy_config": args.policy_config,
            "checkpoint_dir": args.checkpoint_dir,
            "norm_stats_assets_dir": args.norm_stats_assets_dir,
            "seed": args.seed,
        },
        meta_path,
    )
    logging.info("[pairs] wrote meta to %s", meta_path)

    total_samples = args.samples if args.samples > 0 else len(files)
    logging.info("[pairs] extracting %d samples (shard_size=%d)", total_samples, args.shard_size)

    shard_x = []
    shard_y = []
    shard_idx = 0
    for idx in range(total_samples):
        if args.samples > 0:
            obs, act_chunk = sample_training_point()
        else:
            obs, act_chunk = sample_training_point(files[idx])
        feats, noise, actions_t = _sample_feature(obs, act_chunk)
        x_flat = feats.reshape(-1)
        x_proj = x_flat if proj_matrix is None else (proj_matrix @ x_flat)
        x_proj = (x_proj - feature_mean) / feature_std
        base_out = model.action_out_proj(feats).detach()
        target = (noise - actions_t).squeeze(0)
        y_flat = (target - base_out).reshape(-1)
        if args.normalize_target:
            y_flat = (y_flat - target_mean) / target_std
        y_flat = y_flat * float(args.residual_scale)

        shard_x.append(x_proj.detach().cpu().numpy())
        shard_y.append(y_flat.detach().cpu().numpy())

        if len(shard_x) >= args.shard_size:
            shard_path = out_dir / f"shard_{shard_idx:04d}.npz"
            np.savez_compressed(shard_path, x=np.stack(shard_x, axis=0), y=np.stack(shard_y, axis=0))
            logging.info("[pairs] wrote %s (%d samples)", shard_path, len(shard_x))
            shard_x, shard_y = [], []
            shard_idx += 1

    if shard_x:
        shard_path = out_dir / f"shard_{shard_idx:04d}.npz"
        np.savez_compressed(shard_path, x=np.stack(shard_x, axis=0), y=np.stack(shard_y, axis=0))
        logging.info("[pairs] wrote %s (%d samples)", shard_path, len(shard_x))

    hook_handle.remove()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
