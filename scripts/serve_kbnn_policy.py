import dataclasses
import logging
import socket
from pathlib import Path

import torch
import tyro

from openpi.training import checkpoints as _checkpoints
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class KBNNActionHead(torch.nn.Module):
    """Replace pi05's `action_out_proj` with a KBNN MLP.

    Expects weights in the same format as `kbnn_checkpoint.pt` saved by `scripts/train_kbnn.py`,
    i.e. a list of layer mean weights `mws`, each shaped (out_dim, in_dim+1) including bias.
    """

    def __init__(
        self,
        mws: list[torch.Tensor],
        feature_mean: torch.Tensor | None = None,
        feature_std: torch.Tensor | None = None,
        target_mean: torch.Tensor | None = None,
        target_std: torch.Tensor | None = None,
        residual_scale: float | None = None,
        kbnn_scale: float = 1.0,
        proj_matrix: torch.Tensor | None = None,
        horizon: int | None = None,
        feature_dim: int = 1024,
    ):
        super().__init__()
        self.mws = torch.nn.ParameterList([torch.nn.Parameter(w.clone().detach()) for w in mws])
        if feature_mean is not None and feature_std is not None:
            self.register_buffer("feature_mean", feature_mean.clone().detach())
            self.register_buffer("feature_std", feature_std.clone().detach())
        else:
            self.feature_mean = None
            self.feature_std = None
        if proj_matrix is not None:
            self.register_buffer("proj_matrix", proj_matrix.clone().detach())
        else:
            self.proj_matrix = None
        if target_mean is not None and target_std is not None:
            self.register_buffer("target_mean", target_mean.clone().detach())
            self.register_buffer("target_std", target_std.clone().detach())
        else:
            self.target_mean = None
            self.target_std = None
        if residual_scale is not None:
            self.register_buffer("residual_scale", torch.tensor(float(residual_scale), dtype=torch.float32))
        else:
            self.residual_scale = None
        self.register_buffer("kbnn_scale", torch.tensor(float(kbnn_scale), dtype=torch.float32))
        self.horizon = horizon
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, D) or (N, D)
        orig_shape = x.shape
        if x.ndim == 3:
            batch_size, horizon, width = x.shape
        elif x.ndim == 2:
            # Treat as a single batch with horizon=1.
            batch_size, width = x.shape
            horizon = 1
            x = x.unsqueeze(1)
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got shape={orig_shape}")

        hidden = x.to(dtype=torch.float32)
        if self.proj_matrix is not None:
            if self.horizon is None:
                self.horizon = horizon
            flat = hidden.reshape(hidden.shape[0], -1)
            hidden = flat @ self.proj_matrix.T
        if self.feature_mean is not None and self.feature_std is not None:
            hidden = (hidden - self.feature_mean) / self.feature_std
        for layer_idx, mw in enumerate(self.mws):
            ones = torch.ones(hidden.shape[0], 1, dtype=hidden.dtype, device=hidden.device)
            hidden_aug = torch.cat([hidden, ones], dim=1)  # (N, in+1)
            hidden = hidden_aug @ mw.T  # (N, out)
            if layer_idx != len(self.mws) - 1:
                hidden = torch.relu(hidden)

        if self.residual_scale is not None and float(self.residual_scale) != 0.0:
            hidden = hidden / self.residual_scale
        if self.target_mean is not None and self.target_std is not None:
            hidden = hidden * self.target_std + self.target_mean
        if float(self.kbnn_scale) != 1.0:
            hidden = hidden * self.kbnn_scale

        if len(orig_shape) == 3 or (len(orig_shape) == 2 and self.proj_matrix is not None):
            return hidden.reshape(batch_size, horizon, -1)
        return hidden


class ResidualActionHead(torch.nn.Module):
    def __init__(self, base_head: torch.nn.Module, kbnn_head: KBNNActionHead, debug_every: int = 0):
        super().__init__()
        self.base_head = base_head
        self.kbnn_head = kbnn_head
        self._debug_every = max(0, int(debug_every))
        self._debug_step = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_head(x)
        residual = self.kbnn_head(x)
        out = base + residual
        if self._debug_every > 0:
            self._debug_step += 1
            if self._debug_step % self._debug_every == 0:
                with torch.no_grad():
                    base_norm = float(torch.linalg.norm(base))
                    residual_norm = float(torch.linalg.norm(residual))
                    out_norm = float(torch.linalg.norm(out))
                    if base.ndim == 3:
                        base_7 = base[0, :, :7]
                        residual_7 = residual[0, :, :7]
                        out_7 = out[0, :, :7]
                    else:
                        base_7 = base[:1, :7]
                        residual_7 = residual[:1, :7]
                        out_7 = out[:1, :7]
                    z_mean = float(out_7[..., 2].mean())
                    grip_mean = float(out_7[..., 6].mean())
                    logging.info(
                        "[kbnn_debug] step=%d base_norm=%.6f residual_norm=%.6f out_norm=%.6f z_mean=%.6f grip_mean=%.6f",
                        self._debug_step,
                        base_norm,
                        residual_norm,
                        out_norm,
                        z_mean,
                        grip_mean,
                    )
        return out


def _load_kbnn_mws(kbnn_checkpoint: str, device: str):
    ckpt = torch.load(kbnn_checkpoint, map_location="cpu")
    if "mws" not in ckpt:
        raise ValueError(f"{kbnn_checkpoint} missing 'mws' (expected output of scripts/train_kbnn.py)")
    mws = [w.to(dtype=torch.float32, device=device) for w in ckpt["mws"]]
    feature_mean = ckpt.get("feature_mean")
    feature_std = ckpt.get("feature_std")
    target_mean = ckpt.get("target_mean")
    target_std = ckpt.get("target_std")
    residual_scale = ckpt.get("residual_scale", 1.0)
    proj_matrix = ckpt.get("proj_matrix")
    if feature_mean is not None and feature_std is not None:
        feature_mean = feature_mean.to(dtype=torch.float32, device=device)
        feature_std = feature_std.to(dtype=torch.float32, device=device)
    else:
        feature_mean = None
        feature_std = None
    if target_mean is not None and target_std is not None:
        target_mean = target_mean.to(dtype=torch.float32, device=device)
        target_std = target_std.to(dtype=torch.float32, device=device)
    else:
        target_mean = None
        target_std = None
    if proj_matrix is not None:
        proj_matrix = proj_matrix.to(dtype=torch.float32, device=device)
    return mws, feature_mean, feature_std, target_mean, target_std, residual_scale, proj_matrix


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint directory."""

    # Training config name (e.g., "pi05_libero").
    config: str
    # Converted PyTorch checkpoint directory (must contain model.safetensors).
    dir: str


@dataclasses.dataclass
class Args:
    """Serve a pi05 policy with an optional KBNN action head."""

    # Port to serve the policy on.
    port: int = 8000

    # If provided, injected when "prompt" missing.
    default_prompt: str | None = None

    # Device for torch policy: "cuda", "cuda:0", or "cpu".
    pytorch_device: str | None = None

    # Assets directory containing norm stats for the LIBERO dataset (e.g. from the original JAX checkpoint step dir).
    # This should point to the `assets/` folder that contains `<asset_id>/norm_stats.json`.
    # Example: `/media/data-ssd-2/qiaoan_ckpt/pi05_29999/assets`
    norm_stats_assets_dir: str | None = None

    # KBNN checkpoint (recommended: output of scripts/train_kbnn.py). If None, serve baseline pi05.
    kbnn_checkpoint: str | None = None

    # Disable KBNN even if `kbnn_checkpoint` is provided.
    disable_kbnn: bool = False

    # Scale residual at inference time (post-unnormalization).
    kbnn_scale: float = 1.0

    # Record the policy's behavior for debugging.
    record: bool = False

    # Log residual/base norms every N forwards (0 disables).
    debug_every: int = 0

    # Which policy checkpoint to load.
    policy: Checkpoint = dataclasses.field(default_factory=lambda: Checkpoint(config="pi05_libero", dir=""))


def main(args: Args) -> None:
    if not args.policy.dir:
        raise ValueError("--policy.dir is required and must point to a converted PyTorch checkpoint directory.")
    checkpoint_dir = Path(args.policy.dir)
    if not (checkpoint_dir / "model.safetensors").exists():
        raise FileNotFoundError(f"Expected {checkpoint_dir}/model.safetensors (convert the checkpoint to PyTorch first).")

    train_config = _config.get_config(args.policy.config)

    # Load norm stats:
    # The converted PyTorch checkpoint directory may not include `assets/`, so allow pointing to the
    # original step checkpoint's assets dir (or any compatible assets dir).
    norm_stats = None
    if args.norm_stats_assets_dir is not None:
        data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
        if data_config.asset_id is None:
            raise ValueError("Train config has no asset_id; cannot load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(Path(args.norm_stats_assets_dir), data_config.asset_id)

    policy = _policy_config.create_trained_policy(
        train_config,
        str(checkpoint_dir),
        default_prompt=args.default_prompt,
        norm_stats=norm_stats,
        pytorch_device=args.pytorch_device,
    )

    # Replace the action head with KBNN.
    use_kbnn = (not args.disable_kbnn) and (args.kbnn_checkpoint is not None)
    if use_kbnn:
        device = policy._pytorch_device  # noqa: SLF001
        mws, feature_mean, feature_std, target_mean, target_std, residual_scale, proj_matrix = _load_kbnn_mws(
            args.kbnn_checkpoint,
            device=device,
        )
        if target_mean is None or target_std is None:
            logging.warning(
                "KBNN checkpoint missing target_mean/target_std; residual unnormalization is disabled."
            )
        kbnn_head = KBNNActionHead(
            mws,
            feature_mean=feature_mean,
            feature_std=feature_std,
            target_mean=target_mean,
            target_std=target_std,
            residual_scale=residual_scale,
            kbnn_scale=args.kbnn_scale,
            proj_matrix=proj_matrix,
            horizon=int(getattr(train_config.model, "action_horizon", 10)),
        ).to(device)

        # Under the hood, the PyTorch model is PI0Pytorch and uses `action_out_proj` for (width->32).
        model = policy._model  # noqa: SLF001
        if not hasattr(model, "action_out_proj"):
            raise AttributeError("Loaded policy model does not have `action_out_proj`; cannot attach KBNN head.")
        model.action_out_proj = ResidualActionHead(
            model.action_out_proj,
            kbnn_head,
            debug_every=args.debug_every,
        ).to(device)
        logging.info("Attached KBNN action head from %s", args.kbnn_checkpoint)
    else:
        logging.info("Serving baseline pi05 (no KBNN).")

    policy_metadata = policy.metadata | {
        "kbnn_enabled": bool(use_kbnn),
        "kbnn_checkpoint": args.kbnn_checkpoint if use_kbnn else None,
    }

    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s, port: %s)", hostname, local_ip, args.port)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
