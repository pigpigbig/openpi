"""
Train KBNN as a drop-in replacement for pi05's denoising head (action_out_proj).

Why this exists:
- In pi05, `action_out_proj` does NOT output the final environment action directly.
  It outputs the denoiser prediction (v_t / u_t) used inside the diffusion rollout.
- If you train KBNN to regress directly to the final 7-D LIBERO env action and then
  replace `action_out_proj`, performance can collapse (e.g., 0% success) because the
  head is trained for the wrong target.

This script trains KBNN with the SAME objective as pi05 training:
    loss = MSE(u_t, v_t_pred)
where u_t = noise - actions, and v_t_pred comes from the model (with KBNN in place
of the linear action_out_proj).

Inputs:
- Converted PyTorch checkpoint directory (has `model.safetensors` + `config.json`).
- Norm stats assets directory from the original checkpoint step dir (contains
  `<asset_id>/norm_stats.json`). Example: `/media/.../pi05_29999/assets`.
- Dataset from `examples/libero/collect_kbnn_data.py`:
    data/libero/kbnn_dataset/env_*/ep_*.npz
  Must include:
    camshift_images: (T,224,224,3) uint8
    wrist_images:    (T,224,224,3) uint8
    states:          (T,8) float32
    actions:         (T,7) float32   (env actions)
    prompt:          str

Output:
- A `kbnn_checkpoint.pt` compatible with `scripts/serve_kbnn_policy.py` (contains `mws`).

Example:
  uv run scripts/train_kbnn.py \\
    --checkpoint-dir /media/data-ssd-2/qiaoan_ckpt/pi05_libero_pytorch \\
    --norm-stats-assets-dir /media/data-ssd-2/qiaoan_ckpt/pi05_29999/assets \\
    --data-root data/libero/kbnn_dataset \\
    --epochs 1 \\
    --steps-per-epoch 2000 \\
    --encode-chunk-steps 1 \\
    --output kbnn_checkpoint.pt
"""

from __future__ import annotations

import argparse
import logging
import glob
import os
import random
import sys
import math
from pathlib import Path

import numpy as np
import torch

# Ensure repo root on path so `KBNN_old.py` and `weight_initialization.py` are importable.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weight_initialization import initialize_weights  # noqa: E402
from KBNN_old import KBNN as KBNNOld  # noqa: E402

from openpi.policies import policy_config as _policy_config  # noqa: E402
from openpi.training import checkpoints as _checkpoints  # noqa: E402
from openpi.training import config as _config  # noqa: E402


class _HeadInputCatcher:
    def __init__(self):
        self.last_in = None

    def __call__(self, _module, inputs):
        # inputs is a tuple; first entry is tensor [B,H,D]
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
    """Window the per-step action stream into an action-horizon chunk."""
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
    ap.add_argument("--data-root", default="data/libero/kbnn_dataset", help="Collected dataset root")
    ap.add_argument("--kbnn-weights", default="kbnn_weights", help="Dir with action_out_proj_weight.npy and bias.npy")
    ap.add_argument("--geometry", default="2048,512", help="KBNN proj_dim,hidden_dim (output is horizon*32)")
    ap.add_argument("--proj-dim", type=int, default=2048, help="Projection dimension for flattened features")
    ap.add_argument("--kbnn-hidden", type=int, default=64, help="Hidden dimension for KBNN")
    ap.add_argument(
        "--env-ids",
        type=int,
        nargs="*",
        default=None,
        help="Optional env ids to train on (e.g. --env-ids 0 1).",
    )
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--steps-per-epoch", type=int, default=2000, help="Random samples per epoch")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument(
        "--feature-stats-samples",
        type=int,
        default=0,
        help="Samples to estimate feature mean/std (0 = one sample per episode if --use-each-episode-once, else steps-per-epoch).",
    )
    ap.add_argument("--save-every", type=int, default=500, help="Save a KBNN checkpoint every N steps")
    ap.add_argument(
        "--use-each-episode-once",
        action="store_true",
        help="If set, each epoch will use each episode file once (one random timestep per episode).",
    )
    ap.add_argument(
        "--init-from",
        default=None,
        help="Optional KBNN checkpoint to initialize from (e.g. kbnn_checkpoint_noop.pt).",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", default="kbnn_checkpoint.pt")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"

    # Build a Policy so we reuse the exact same LIBERO preprocessing + normalization as inference.
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

    def sample_training_point(ep_path: str | None = None):
        if ep_path is None:
            ep_path = random.choice(files)
        with np.load(ep_path) as data:
            camshift_images = data["camshift_images"]  # (T,224,224,3)
            wrist_images = data["wrist_images"]
            states = data["states"]  # (T,8)
            actions7 = data["actions"]  # (T,7)
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
        return catcher.last_in.squeeze(0).to(dtype=torch.float32, device=device)

    # Initialize KBNN weights with feature-normalized correction.
    ckpt = None
    if args.init_from:
        ckpt = torch.load(args.init_from, map_location="cpu")
    geom_parts = [int(x) for x in args.geometry.split(",")] if args.geometry else []
    proj_dim = args.proj_dim
    kbnn_hidden = args.kbnn_hidden
    if len(geom_parts) >= 2:
        proj_dim = geom_parts[0]
        kbnn_hidden = geom_parts[1]
        if len(geom_parts) > 2:
            logging.info("[kbnn] geometry has extra values; using only first two: %s", geom_parts[:2])

    feat_dim = 1024
    flat_dim = horizon * feat_dim
    out_dim = horizon * 32
    geom_no_bias = [proj_dim, kbnn_hidden, out_dim]
    geom_with_bias = [proj_dim + 1, kbnn_hidden, out_dim]
    if args.init_from:
        if "mws" not in ckpt:
            raise ValueError(f"{args.init_from} missing 'mws' list.")
        init_mws = [w.to(dtype=torch.float32).T for w in ckpt["mws"]]
        logging.info("[kbnn] init from %s", args.init_from)
    else:
        # Start from zero-mean KBNN so baseline behavior is unchanged.
        init_mws = []
        for i in range(len(geom_no_bias) - 1):
            init_mws.append(torch.zeros((geom_no_bias[i] + 1, geom_no_bias[i + 1]), dtype=torch.float32))
        logging.info("[kbnn] init from zeros (residual KBNN)")

    if args.init_from:
        proj_matrix = ckpt.get("proj_matrix")
        if proj_matrix is None:
            raise ValueError(f"{args.init_from} missing proj_matrix; refusing to create a new one.")
        proj_matrix = proj_matrix.to(dtype=torch.float32, device=device)
        if proj_matrix.shape != (proj_dim, flat_dim):
            raise ValueError(
                "proj_matrix shape mismatch: "
                f"checkpoint has {tuple(proj_matrix.shape)}, "
                f"expected {(proj_dim, flat_dim)}."
            )
    else:
        proj_matrix = torch.randn(proj_dim, flat_dim, device=device) / math.sqrt(flat_dim)

    # Feature stats for normalization of projected features.
    ckpt_feature_mean = None
    ckpt_feature_std = None
    if ckpt is not None:
        if "feature_mean" in ckpt and "feature_std" in ckpt:
            ckpt_feature_mean = ckpt["feature_mean"].to(dtype=torch.float32, device=device)
            ckpt_feature_std = ckpt["feature_std"].to(dtype=torch.float32, device=device)
            logging.info("[kbnn] using feature stats from %s", args.init_from)

    if ckpt_feature_mean is not None and ckpt_feature_std is not None:
        feature_mean = ckpt_feature_mean
        feature_std = ckpt_feature_std
    else:
        if args.feature_stats_samples > 0:
            stats_samples = args.feature_stats_samples
            stats_iter = (sample_training_point() for _ in range(stats_samples))
        elif args.use_each_episode_once:
            stats_files = files[:]
            random.shuffle(stats_files)
            stats_samples = len(stats_files)
            stats_iter = (sample_training_point(ep_path) for ep_path in stats_files)
        else:
            stats_samples = args.steps_per_epoch
            stats_iter = (sample_training_point() for _ in range(stats_samples))

        sum_x = torch.zeros(proj_dim, dtype=torch.float64, device=device)
        sum_x2 = torch.zeros(proj_dim, dtype=torch.float64, device=device)
        count = 0
        for _ in range(stats_samples):
            obs, act_chunk = next(stats_iter)
            feats = _sample_feature(obs, act_chunk)
            feats = feats.reshape(-1, feats.shape[-1])
            x_flat = feats.reshape(-1).to(dtype=torch.float32, device=device)
            x_proj = proj_matrix @ x_flat
            sum_x += x_proj.double()
            sum_x2 += (x_proj.double() ** 2)
            count += 1

        feature_mean = (sum_x / max(count, 1)).float()
        feature_var = (sum_x2 / max(count, 1) - feature_mean.double() ** 2).clamp_min(1e-12)
        feature_std = torch.sqrt(feature_var).float().clamp_min(1e-6)
        logging.info(
            "[kbnn] feature stats (proj): count=%s mean_abs=%.6f std_mean=%.6f",
            count,
            float(feature_mean.abs().mean()),
            float(feature_std.mean()),
        )
        logging.info(
            "[kbnn] feature_mean[:5]=%s feature_std[:5]=%s",
            feature_mean[:5].tolist(),
            feature_std[:5].tolist(),
        )

    kbnn = KBNNOld(
        geom_no_bias,
        act_fun=["relu", "relu", "linear"],
        weight_prior=[mw.to(device=device) for mw in init_mws],
        no_bias=False,
        noise=0.0,
        verbose=False,
        device=torch.device(device),
    )
    if args.lr != 0.0:
        logging.info("[kbnn] lr=%s (unused; KBNN uses Kalman update)", args.lr)
    def _save_kbnn(path: str) -> None:
        torch.save(
            {
                "geometry_with_bias": geom_with_bias,
                "kbnn_geometry": geom_no_bias,
                "cov_mode": getattr(kbnn, "cov_mode", "full"),
                "mws": [w.detach().cpu().T for w in kbnn.mw],
                "feature_mean": feature_mean.detach().cpu(),
                "feature_std": feature_std.detach().cpu(),
                "proj_matrix": proj_matrix.detach().cpu(),
                "proj_dim": proj_dim,
                "kbnn_hidden": kbnn_hidden,
                "kbnn_out_dim": out_dim,
            },
            path,
        )

    global_step = 0
    for epoch in range(args.epochs):
        running = 0.0
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
            feats = _sample_feature(obs, act_chunk, noise=noise, time=time)
            x_raw = feats.to(dtype=torch.float32, device=device)
            if (global_step + 1) % 100 == 0:
                print(f"[kbnn] x_raw shape: {tuple(x_raw.shape)}")
            x_flat = x_raw.reshape(-1)
            x_proj = (proj_matrix @ x_flat).to(dtype=torch.float32, device=kbnn.device)
            x_proj = (x_proj - feature_mean) / feature_std
            x = x_proj.unsqueeze(0)
            base_out = model.action_out_proj(x_raw).detach()
            target = (noise - actions_t).squeeze(0).to(dtype=torch.float32, device=kbnn.device)
            y_flat = (target - base_out).reshape(-1).to(dtype=torch.float32, device=kbnn.device)
            y = y_flat.unsqueeze(0)
            if (global_step + 1) % 100 == 0:
                logging.info(
                    "[kbnn] sample shapes: x=%s y=%s x_mean=%.6f x_std=%.6f y_mean=%.6f y_std=%.6f",
                    tuple(x.shape),
                    tuple(y.shape),
                    float(x.mean()),
                    float(x.std(unbiased=False)),
                    float(y.mean()),
                    float(y.std(unbiased=False)),
                )
            pred, _, _, _ = kbnn.single_forward_pass(x, training=False)
            loss = torch.mean((pred - y) ** 2)
            kbnn.train(x, y)
            running += float(loss.detach().cpu())
            global_step += 1
            if (step + 1) % 100 == 0:
                print(f"epoch {epoch+1}/{args.epochs} step {step+1}/{steps} loss={running/100:.6f}")
                running = 0.0
            if args.save_every > 0 and global_step % args.save_every == 0:
                save_path = f"{Path(args.output).with_suffix('')}_step{global_step}.pt"
                _save_kbnn(save_path)
                print(f"Saved {save_path}")

    hook_handle.remove()

    # Save just the KBNN head weights in the same format expected by serve_kbnn_policy.py.
    _save_kbnn(args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
