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
  uv run scripts/train_kbnn_diffusion.py \\
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
import glob
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

# Ensure repo root on path so `KBNN2.py` and `weight_initialization.py` are importable.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weight_initialization import initialize_weights  # noqa: E402

from openpi.policies import policy_config as _policy_config  # noqa: E402
from openpi.training import checkpoints as _checkpoints  # noqa: E402
from openpi.training import config as _config  # noqa: E402


class KBNNActionHead(torch.nn.Module):
    """KBNN MLP head: (width=1024) -> (32). Uses weights with explicit bias columns."""

    def __init__(self, mws: list[torch.Tensor]):
        super().__init__()
        self.mws = torch.nn.ParameterList([torch.nn.Parameter(w.clone().detach()) for w in mws])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, D)
        if x.ndim != 3:
            raise ValueError(f"Expected (B,H,D) got {tuple(x.shape)}")
        b, h, d = x.shape
        hidden = x.reshape(b * h, d).to(dtype=torch.float32)
        for i, mw in enumerate(self.mws):
            ones = torch.ones(hidden.shape[0], 1, dtype=hidden.dtype, device=hidden.device)
            hidden = torch.cat([hidden, ones], dim=1) @ mw.T
            if i != len(self.mws) - 1:
                hidden = torch.relu(hidden)
        return hidden.reshape(b, h, -1)


def _load_episode_files(data_root: str) -> list[str]:
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
    ap.add_argument("--geometry", default="1025,2050,2050,32", help="KBNN geometry for initialization (includes bias)")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--steps-per-epoch", type=int, default=2000, help="Random samples per epoch")
    ap.add_argument("--lr", type=float, default=1e-4)
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

    # Initialize KBNN weights from the original linear projection weights.
    geom_with_bias = [int(x) for x in args.geometry.split(",")]
    w = np.load(str(Path(args.kbnn_weights) / "action_out_proj_weight.npy"))  # (1024, 32)
    b = np.load(str(Path(args.kbnn_weights) / "action_out_proj_bias.npy"))  # (32,)
    w_with_bias = torch.tensor(np.vstack([w, b[None, :]]), dtype=torch.float32)
    init_mws = initialize_weights(w_with_bias, geom_with_bias)

    kbnn_head = KBNNActionHead([mw.to(device) for mw in init_mws]).to(device)
    if not hasattr(model, "action_out_proj"):
        raise AttributeError("Loaded torch model has no action_out_proj")
    model.action_out_proj = kbnn_head

    # Freeze everything except the KBNN head.
    for p in model.parameters():
        p.requires_grad_(False)
    for p in kbnn_head.parameters():
        p.requires_grad_(True)

    opt = torch.optim.AdamW(kbnn_head.parameters(), lr=args.lr)

    files = _load_episode_files(args.data_root)
    horizon = int(getattr(train_config.model, "action_horizon", 10))

    def sample_training_point():
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

    for epoch in range(args.epochs):
        running = 0.0
        for step in range(args.steps_per_epoch):
            obs, act_chunk = sample_training_point()

            # Apply the same transforms used by the policy server (parse/normalize/tokenize).
            inputs = policy._input_transform(obs)  # noqa: SLF001
            inputs = {k: np.asarray(v) for k, v in inputs.items()}

            # Policy.infer would add batch dim and convert to torch; replicate that for training.
            inputs_t = torch.utils._pytree.tree_map(  # type: ignore[attr-defined]
                lambda x: torch.from_numpy(np.array(x)).to(device)[None, ...], inputs
            )
            from openpi.models import model as _model  # local import to avoid circulars

            observation = _model.Observation.from_dict(inputs_t)
            actions_t = torch.from_numpy(act_chunk)[None, ...].to(device, dtype=torch.float32)  # (1,H,32)

            # Use noise with padded dims masked to zero to avoid learning spurious padding behavior.
            noise = torch.randn_like(actions_t)
            noise[..., 7:] = 0.0

            # Use a fixed random time per sample (in [0,1]); keep float32.
            time = torch.rand((1,), device=device, dtype=torch.float32) * 0.999 + 0.001

            loss_map = model(observation, actions_t, noise=noise, time=time)  # (1,H,32) loss per-dim
            loss = loss_map[..., :7].mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running += float(loss.detach().cpu())
            if (step + 1) % 100 == 0:
                print(f"epoch {epoch+1}/{args.epochs} step {step+1}/{args.steps_per_epoch} loss={running/100:.6f}")
                running = 0.0

    # Save just the KBNN head weights in the same format expected by serve_kbnn_policy.py.
    torch.save(
        {
            "geometry_with_bias": geom_with_bias,
            "kbnn_geometry": [geom_with_bias[0] - 1, *geom_with_bias[1:]],
            "cov_mode": "diag",
            "mws": [p.detach().cpu() for p in kbnn_head.mws],
        },
        args.output,
    )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

