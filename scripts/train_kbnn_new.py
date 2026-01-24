"""
Train KBNN as a drop-in replacement for pi05's denoising head (action_out_proj).

Key idea:
- In pi05, `action_out_proj` outputs the denoiser prediction (v_t / u_t) used inside diffusion rollout.
- DO NOT train KBNN to output final env action directly. Train residual for the denoiser head target:
      loss = MSE( base_out + kbnn(x),  u_t )
  where u_t = noise - actions, and base_out is the original model.action_out_proj output.

This script trains a residual KBNN on:
      y = (u_t - base_out)
so replacing action_out_proj with (base_out + kbnn) is correct.

Inputs:
- Converted PyTorch checkpoint directory (has model.safetensors + config.json)
- Norm stats assets directory from original checkpoint step dir (contains <asset_id>/norm_stats.json)
- Dataset collected by examples/libero/collect_kbnn_data.py:
    data/libero/kbnn_dataset/env_*/ep_*.npz

Output:
- kbnn_checkpoint.pt compatible with scripts/serve_kbnn_policy.py (contains mws + proj_matrix + stats)

Example:
  uv run scripts/train_kbnn.py \
    --checkpoint-dir /media/data-ssd-2/qiaoan_ckpt/pi05_libero_pytorch \
    --norm-stats-assets-dir /media/data-ssd-2/qiaoan_ckpt/pi05_29999/assets \
    --data-root data/libero/kbnn_dataset \
    --epochs 1 \
    --steps-per-epoch 2000 \
    --proj-dim 2048 \
    --kbnn-hidden 64 \
    --cov-mode diag \
    --output kbnn_checkpoint.pt
"""

from __future__ import annotations

import argparse
import glob
import logging
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

# Ensure repo root on path so `KBNN_old.py` is importable.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from KBNN_new import KBNN as KBNNNew  # noqa: E402

from openpi.policies import policy_config as _policy_config  # noqa: E402
from openpi.training import checkpoints as _checkpoints  # noqa: E402
from openpi.training import config as _config  # noqa: E402


class _HeadInputCatcher:
    """Capture the tensor input to model.action_out_proj (pre-hook)."""

    def __init__(self):
        self.last_in = None

    def __call__(self, _module, inputs):
        # inputs is a tuple; first entry is tensor [B,H,D]
        self.last_in = inputs[0].detach()


def _load_episode_files(data_root: str, env_ids: list[int] | None = None) -> list[str]:
    if env_ids:
        files: list[str] = []
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
    """Window the per-step action stream into an action-horizon chunk."""
    T = actions32.shape[0]
    chunk = np.zeros((horizon, 32), dtype=np.float32)
    for k in range(horizon):
        idx = min(T - 1, t + k)
        chunk[k] = actions32[idx]
    return chunk


def main() -> None:
    ap = argparse.ArgumentParser()

    # model + data
    ap.add_argument("--checkpoint-dir", required=True, help="Converted PyTorch ckpt dir (model.safetensors)")
    ap.add_argument("--norm-stats-assets-dir", required=True, help="Assets dir containing <asset_id>/norm_stats.json")
    ap.add_argument("--policy-config", default="pi05_libero", help="OpenPI config name")
    ap.add_argument("--data-root", default="data/libero/kbnn_dataset", help="Collected dataset root")
    ap.add_argument("--env-ids", type=int, nargs="*", default=None, help="Optional env ids to train on.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")

    # sampling/training
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--steps-per-epoch", type=int, default=2000, help="Random samples per epoch")
    ap.add_argument("--use-each-episode-once", action="store_true", help="Use each episode file once per epoch")
    ap.add_argument("--encode-chunk-steps", type=int, default=1, help="Compatibility arg (unused).")

    # KBNN geometry
    ap.add_argument("--geometry", default=None, help="Optional 'proj_dim,kbnn_hidden' (output always horizon*32)")
    ap.add_argument("--proj-dim", type=int, default=2048, help="Projection dimension for flattened features")
    ap.add_argument("--kbnn-hidden", type=int, default=64, help="Hidden dimension for KBNN")

    # feature normalization
    ap.add_argument("--feature-stats-samples", type=int, default=0)
    ap.add_argument("--no-feature-normalization", action="store_true")

    # residual target options
    ap.add_argument("--residual-scale", type=float, default=1.0)
    ap.add_argument("--normalize-target", action="store_true")

    # init / resume
    ap.add_argument("--init-from", default=None, help="Optional KBNN checkpoint to initialize from")
    ap.add_argument("--init-cov", type=float, default=1.0, help="Initial diagonal covariance for KBNN weights.")
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--save-initial", action="store_true")
    ap.add_argument("--output", default="kbnn_checkpoint.pt")
    ap.add_argument("--kbnn-verbose", action="store_true")

    # ----- NEW: stability knobs (match new KBNN_old.py) -----
    ap.add_argument("--cov-mode", default="diag", choices=["diag", "full"], help="KBNN covariance mode")
    ap.add_argument("--obs-var", type=float, default=1e-4, help="Posterior output var floor (avoid perfect measurement)")
    ap.add_argument("--var-floor", type=float, default=1e-6, help="Min clamp for Ca/Cy denominators")
    ap.add_argument("--jitter", type=float, default=1e-8, help="PSD jitter for Cw (full mode)")
    ap.add_argument("--process-noise", type=float, default=1e-6, help="Covariance inflation to prevent overconfidence")
    ap.add_argument("--gain-clip", type=float, default=50.0, help="Clamp Kalman gain magnitude")
    ap.add_argument("--kbnn-forward-noise", type=float, default=1e-4, help="Additive forward-pass noise to Cy")

    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    logging.info("[train] device=%s", device)

    # Build a Policy so we reuse the exact same preprocessing + normalization as inference.
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

    if not hasattr(model, "action_out_proj"):
        raise AttributeError("Loaded torch model has no action_out_proj")

    catcher = _HeadInputCatcher()
    hook_handle = model.action_out_proj.register_forward_pre_hook(catcher)

    files = _load_episode_files(args.data_root, args.env_ids)

    horizon = int(getattr(train_config.model, "action_horizon", 10))
    feat_dim = int(getattr(model.action_out_proj, "in_features", 1024))
    flat_dim = horizon * feat_dim
    out_dim = horizon * 32

    logging.info("[train] horizon=%d feat_dim=%d flat_dim=%d out_dim=%d", horizon, feat_dim, flat_dim, out_dim)

    # Parse geometry override
    proj_dim = args.proj_dim
    kbnn_hidden = args.kbnn_hidden
    if args.geometry:
        parts = [int(x) for x in args.geometry.split(",") if x.strip()]
        if len(parts) >= 2:
            proj_dim, kbnn_hidden = parts[0], parts[1]
            logging.info("[kbnn] geometry override: proj_dim=%d kbnn_hidden=%d", proj_dim, kbnn_hidden)
        else:
            raise ValueError("--geometry must be 'proj_dim,kbnn_hidden'")

    geom_no_bias = [proj_dim, kbnn_hidden, out_dim]
    geom_with_bias = [proj_dim + 1, kbnn_hidden, out_dim]

    # ---------- episode sampling ----------
    def sample_training_point(ep_path: str | None = None):
        if ep_path is None:
            ep_path = random.choice(files)
        with np.load(ep_path) as data:
            camshift_images = data["camshift_images"]  # (T,224,224,3)
            wrist_images = data["wrist_images"]        # (T,224,224,3)
            states = data["states"]                    # (T,8)
            actions7 = data["actions"]                 # (T,7)
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
        act_chunk = _make_action_chunk(actions32, t, horizon)  # (H,32)
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

        # shape: [H,1024]
        return catcher.last_in.squeeze(0).to(dtype=torch.float32, device=device)

    # ---------- initialize KBNN mean weights ----------
    ckpt = None
    if args.init_from:
        ckpt = torch.load(args.init_from, map_location="cpu")
        if "mws" not in ckpt:
            raise ValueError(f"{args.init_from} missing 'mws' list.")
        init_mws = [w.to(dtype=torch.float32).T for w in ckpt["mws"]]
        logging.info("[kbnn] init from %s", args.init_from)
    else:
        # Residual KBNN: start at 0 so baseline stays unchanged.
        init_mws = []
        for i in range(len(geom_no_bias) - 1):
            init_mws.append(torch.zeros((geom_no_bias[i] + 1, geom_no_bias[i + 1]), dtype=torch.float32))
        logging.info("[kbnn] init from zeros (residual KBNN)")

    # ---------- projection matrix ----------
    if args.init_from:
        proj_matrix = ckpt.get("proj_matrix")
        if proj_matrix is None:
            raise ValueError(f"{args.init_from} missing proj_matrix; refusing to create a new one.")
        proj_matrix = proj_matrix.to(dtype=torch.float32, device=device)
        if proj_matrix.shape != (proj_dim, flat_dim):
            raise ValueError(
                "proj_matrix shape mismatch: "
                f"checkpoint has {tuple(proj_matrix.shape)}, expected {(proj_dim, flat_dim)}"
            )
    else:
        proj_matrix = torch.randn(proj_dim, flat_dim, device=device) / math.sqrt(flat_dim)

    # ---------- iterator for stats ----------
    def _stats_iter():
        if args.feature_stats_samples > 0:
            stats_samples = args.feature_stats_samples
            return stats_samples, (sample_training_point() for _ in range(stats_samples))
        if args.use_each_episode_once:
            stats_files = files[:]
            random.shuffle(stats_files)
            stats_samples = len(stats_files)
            return stats_samples, (sample_training_point(ep_path) for ep_path in stats_files)
        stats_samples = args.steps_per_epoch
        return stats_samples, (sample_training_point() for _ in range(stats_samples))

    # ---------- feature stats (proj) ----------
    if args.no_feature_normalization:
        feature_mean = torch.zeros(proj_dim, dtype=torch.float32, device=device)
        feature_std = torch.ones(proj_dim, dtype=torch.float32, device=device)
        logging.info("[kbnn] feature normalization disabled (mean=0, std=1)")
    else:
        ckpt_feature_mean = None
        ckpt_feature_std = None
        if ckpt is not None and "feature_mean" in ckpt and "feature_std" in ckpt:
            ckpt_feature_mean = ckpt["feature_mean"].to(dtype=torch.float32, device=device)
            ckpt_feature_std = ckpt["feature_std"].to(dtype=torch.float32, device=device)
            logging.info("[kbnn] using feature stats from %s", args.init_from)

        if ckpt_feature_mean is not None and ckpt_feature_std is not None:
            feature_mean = ckpt_feature_mean
            feature_std = ckpt_feature_std
        else:
            stats_samples, stats_iter = _stats_iter()
            sum_x = torch.zeros(proj_dim, dtype=torch.float64, device=device)
            sum_x2 = torch.zeros(proj_dim, dtype=torch.float64, device=device)

            count = 0
            for _ in range(stats_samples):
                obs, act_chunk = next(stats_iter)
                feats = _sample_feature(obs, act_chunk)          # (H,1024)
                x_flat = feats.reshape(-1)                        # (H*1024,)
                x_proj = proj_matrix @ x_flat                     # (proj_dim,)
                sum_x += x_proj.double()
                sum_x2 += (x_proj.double() ** 2)
                count += 1

            feature_mean = (sum_x / max(count, 1)).float()
            feature_var = (sum_x2 / max(count, 1) - feature_mean.double() ** 2).clamp_min(1e-12)
            feature_std = torch.sqrt(feature_var).float().clamp_min(1e-6)

            logging.info(
                "[kbnn] feature stats (proj): count=%d mean_abs=%.6f std_mean=%.6f",
                count,
                float(feature_mean.abs().mean()),
                float(feature_std.mean()),
            )

    # ---------- target stats for normalization ----------
    if args.normalize_target:
        ckpt_target_mean = None
        ckpt_target_std = None
        if ckpt is not None and "target_mean" in ckpt and "target_std" in ckpt:
            ckpt_target_mean = ckpt["target_mean"].to(dtype=torch.float32, device=device)
            ckpt_target_std = ckpt["target_std"].to(dtype=torch.float32, device=device)
            logging.info("[kbnn] using target stats from %s", args.init_from)

        if ckpt_target_mean is not None and ckpt_target_std is not None:
            target_mean = ckpt_target_mean
            target_std = ckpt_target_std
        else:
            stats_samples, stats_iter = _stats_iter()
            sum_y = torch.zeros(out_dim, dtype=torch.float64, device=device)
            sum_y2 = torch.zeros(out_dim, dtype=torch.float64, device=device)
            count = 0

            for _ in range(stats_samples):
                obs, act_chunk = next(stats_iter)

                actions_t = torch.from_numpy(act_chunk)[None, ...].to(device, dtype=torch.float32)
                noise = torch.randn_like(actions_t)
                noise[..., 7:] = 0.0
                time = torch.rand((1,), device=device, dtype=torch.float32) * 0.999 + 0.001

                feats = _sample_feature(obs, act_chunk, noise=noise, time=time)
                x_raw = feats.to(dtype=torch.float32, device=device)

                base_out = model.action_out_proj(x_raw).detach()                 # (H,32)
                target = (noise - actions_t).squeeze(0).to(torch.float32)        # (H,32)
                y_flat = (target - base_out).reshape(-1)                         # (H*32,)

                sum_y += y_flat.double()
                sum_y2 += (y_flat.double() ** 2)
                count += 1

            target_mean = (sum_y / max(count, 1)).float()
            target_var = (sum_y2 / max(count, 1) - target_mean.double() ** 2).clamp_min(1e-12)
            target_std = torch.sqrt(target_var).float().clamp_min(1e-6)

            logging.info(
                "[kbnn] target stats: count=%d mean_abs=%.6f std_mean=%.6f",
                count,
                float(target_mean.abs().mean()),
                float(target_std.mean()),
            )
    else:
        target_mean = torch.zeros(out_dim, dtype=torch.float32, device=device)
        target_std = torch.ones(out_dim, dtype=torch.float32, device=device)

    # ---------- create KBNN ----------
    kbnn = KBNNNew(
        geom_no_bias,
        act_fun=["relu", "relu", "linear"],
        weight_prior=[mw.to(device=device) for mw in init_mws],
        no_bias=False,
        noise=float(args.kbnn_forward_noise),        # IMPORTANT: forward-pass noise to avoid tiny Cy
        verbose=args.kbnn_verbose,
        device=torch.device(device),
        init_cov=args.init_cov,

        # stability knobs (KBNN_new.py)
        cov_mode=args.cov_mode,
        obs_var=float(args.obs_var),
        var_floor=float(args.var_floor),
        jitter=float(args.jitter),
        process_noise=float(args.process_noise),
        gain_clip=float(args.gain_clip),
        rho_cya=0.999,
    )

    logging.info(
        "[kbnn] cov_mode=%s obs_var=%g var_floor=%g process_noise=%g gain_clip=%g forward_noise=%g",
        args.cov_mode,
        args.obs_var,
        args.var_floor,
        args.process_noise,
        args.gain_clip,
        args.kbnn_forward_noise,
    )

    # ---------- save checkpoint ----------
    def _save_kbnn(path: str) -> None:
        torch.save(
            {
                "geometry_with_bias": geom_with_bias,
                "kbnn_geometry": geom_no_bias,
                "cov_mode": getattr(kbnn, "cov_mode", "diag"),
                "mws": [w.detach().cpu().T for w in kbnn.mw],
                "feature_mean": feature_mean.detach().cpu(),
                "feature_std": feature_std.detach().cpu(),
                "proj_matrix": proj_matrix.detach().cpu(),
                "proj_dim": proj_dim,
                "kbnn_hidden": kbnn_hidden,
                "kbnn_out_dim": out_dim,
                "residual_scale": args.residual_scale,
                "target_mean": target_mean.detach().cpu(),
                "target_std": target_std.detach().cpu(),
                "obs_var": float(args.obs_var),
                "var_floor": float(args.var_floor),
                "process_noise": float(args.process_noise),
                "gain_clip": float(args.gain_clip),
                "kbnn_forward_noise": float(args.kbnn_forward_noise),
            },
            path,
        )

    if args.save_initial:
        _save_kbnn(args.output)
        logging.info("[kbnn] saved initial checkpoint to %s", args.output)
        hook_handle.remove()
        return

    # ---------- train loop ----------
    global_step = 0

    for epoch in range(args.epochs):
        if args.use_each_episode_once:
            epoch_files = files[:]
            random.shuffle(epoch_files)
            steps = len(epoch_files)
            sample_iter = (sample_training_point(ep_path) for ep_path in epoch_files)
        else:
            steps = args.steps_per_epoch
            sample_iter = (sample_training_point() for _ in range(steps))

        for step in range(steps):
            obs, act_chunk = next(sample_iter)

            actions_t = torch.from_numpy(act_chunk)[None, ...].to(device, dtype=torch.float32)
            noise = torch.randn_like(actions_t)
            noise[..., 7:] = 0.0
            time = torch.rand((1,), device=device, dtype=torch.float32) * 0.999 + 0.001

            feats = _sample_feature(obs, act_chunk, noise=noise, time=time)  # (H,1024)
            x_raw = feats.to(dtype=torch.float32, device=device)

            # ---- project features -> x (1,proj_dim) ----
            x_flat = x_raw.reshape(-1)  # (H*1024,)
            x_proj = (proj_matrix @ x_flat).to(dtype=torch.float32, device=device)
            x_proj = (x_proj - feature_mean) / feature_std
            x = x_proj.unsqueeze(0)

            # ---- compute residual target y (1,out_dim) ----
            base_out = model.action_out_proj(x_raw).detach()                      # (H,32)
            target = (noise - actions_t).squeeze(0).to(dtype=torch.float32)       # (H,32)
            y_flat = (target - base_out).reshape(-1).to(dtype=torch.float32)      # (H*32,)

            if args.normalize_target:
                y_flat = (y_flat - target_mean) / target_std

            y_flat = y_flat * float(args.residual_scale)
            y = y_flat.unsqueeze(0)

            # sanity checks
            if not torch.isfinite(x).all() or not torch.isfinite(y).all():
                logging.warning("[train] non-finite x/y at step=%d; skipping update", global_step)
                global_step += 1
                continue

            # prediction
            pred, _, _, _ = kbnn.single_forward_pass(x, training=False)
            loss = torch.mean((pred - y) ** 2).item()

            if step == 0:
                logging.info(
                    "[train] epoch=%d step=%d loss=%.6f | ||x||=%.4f ||y||=%.4f ||pred||=%.4f ||base_out||=%.4f",
                    epoch + 1,
                    step + 1,
                    loss,
                    float(torch.linalg.norm(x).detach().cpu()),
                    float(torch.linalg.norm(y).detach().cpu()),
                    float(torch.linalg.norm(pred).detach().cpu()),
                    float(torch.linalg.norm(base_out).detach().cpu()),
                )

            # train one step
            kbnn.train(x, y)

            if (global_step + 1) % 50 == 0:
                mw_norms = [float(torch.linalg.norm(w).detach().cpu()) for w in kbnn.mw]
                logging.info(
                    "[train] step=%d loss=%.6f mw_norms=%s",
                    global_step + 1,
                    loss,
                    [round(v, 3) for v in mw_norms],
                )

            global_step += 1

            if args.save_every > 0 and global_step % args.save_every == 0:
                save_path = f"{Path(args.output).with_suffix('')}_step{global_step}.pt"
                _save_kbnn(save_path)
                logging.info("[kbnn] saved %s", save_path)

    hook_handle.remove()

    _save_kbnn(args.output)
    logging.info("[kbnn] saved final checkpoint to %s", args.output)


if __name__ == "__main__":
    main()
