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


class MLPActionHead(torch.nn.Module):
    def __init__(
        self,
        mlp: torch.nn.Module,
        proj_matrix: torch.Tensor,
        feature_mean: torch.Tensor,
        feature_std: torch.Tensor,
        target_mean: torch.Tensor | None,
        target_std: torch.Tensor | None,
        residual_scale: float,
        mlp_scale: float,
        horizon: int,
    ):
        super().__init__()
        self.mlp = mlp
        self.register_buffer("proj_matrix", proj_matrix.clone().detach())
        self.register_buffer("feature_mean", feature_mean.clone().detach())
        self.register_buffer("feature_std", feature_std.clone().detach())
        if target_mean is not None and target_std is not None:
            self.register_buffer("target_mean_buf", target_mean.clone().detach())
            self.register_buffer("target_std_buf", target_std.clone().detach())
        else:
            self.target_mean_buf = None
            self.target_std_buf = None
        self.register_buffer("residual_scale", torch.tensor(float(residual_scale), dtype=torch.float32))
        self.register_buffer("mlp_scale", torch.tensor(float(mlp_scale), dtype=torch.float32))
        self.horizon = horizon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, D) or (N, D)
        if x.ndim == 2:
            x = x.unsqueeze(1)
        batch_size, horizon, width = x.shape
        flat = x.reshape(batch_size, -1)
        proj = flat @ self.proj_matrix.T
        proj = (proj - self.feature_mean) / self.feature_std
        hidden = self.mlp(proj)
        if float(self.residual_scale) != 0.0:
            hidden = hidden / self.residual_scale
        if self.target_mean_buf is not None and self.target_std_buf is not None:
            hidden = hidden * self.target_std_buf + self.target_mean_buf
        if float(self.mlp_scale) != 1.0:
            hidden = hidden * self.mlp_scale
        return hidden.reshape(batch_size, horizon, -1)


class ResidualActionHead(torch.nn.Module):
    def __init__(self, base_head: torch.nn.Module, mlp_head: MLPActionHead, debug_every: int = 0):
        super().__init__()
        self.base_head = base_head
        self.mlp_head = mlp_head
        self._debug_every = max(0, int(debug_every))
        self._debug_step = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_head(x)
        residual = self.mlp_head(x)
        out = base + residual
        if self._debug_every > 0:
            self._debug_step += 1
            if self._debug_step % self._debug_every == 0:
                with torch.no_grad():
                    base_norm = float(torch.linalg.norm(base))
                    residual_norm = float(torch.linalg.norm(residual))
                    out_norm = float(torch.linalg.norm(out))
                    if base.ndim == 3:
                        out_7 = out[0, :, :7]
                    else:
                        out_7 = out[:1, :7]
                    z_mean = float(out_7[..., 2].mean())
                    grip_mean = float(out_7[..., 6].mean())
                    logging.info(
                        "[mlp_debug] step=%d base_norm=%.6f residual_norm=%.6f out_norm=%.6f z_mean=%.6f grip_mean=%.6f",
                        self._debug_step,
                        base_norm,
                        residual_norm,
                        out_norm,
                        z_mean,
                        grip_mean,
                    )
        return out


def _load_mlp_checkpoint(path: str, device: str):
    ckpt = torch.load(path, map_location="cpu")
    required = ["mlp_state", "proj_matrix", "feature_mean", "feature_std", "proj_dim", "out_dim", "horizon"]
    for key in required:
        if key not in ckpt:
            raise ValueError(f"{path} missing '{key}'")
    return ckpt


def _build_mlp(input_dim: int, output_dim: int, hidden: int, layers: int) -> torch.nn.Module:
    blocks = []
    in_dim = input_dim
    for _ in range(layers):
        blocks.append(torch.nn.Linear(in_dim, hidden))
        blocks.append(torch.nn.ReLU())
        in_dim = hidden
    blocks.append(torch.nn.Linear(in_dim, output_dim))
    return torch.nn.Sequential(*blocks)


@dataclasses.dataclass
class Checkpoint:
    config: str
    dir: str


@dataclasses.dataclass
class Args:
    port: int = 8000
    default_prompt: str | None = None
    pytorch_device: str | None = None
    norm_stats_assets_dir: str | None = None
    mlp_checkpoint: str | None = None
    disable_mlp: bool = False
    record: bool = False
    mlp_scale: float = 1.0
    # Log residual/base norms every N forwards (0 disables).
    debug_every: int = 0
    policy: Checkpoint = dataclasses.field(default_factory=lambda: Checkpoint(config="pi05_libero", dir=""))


def main(args: Args) -> None:
    if not args.policy.dir:
        raise ValueError("--policy.dir is required and must point to a converted PyTorch checkpoint directory.")
    checkpoint_dir = Path(args.policy.dir)
    if not (checkpoint_dir / "model.safetensors").exists():
        raise FileNotFoundError(f"Expected {checkpoint_dir}/model.safetensors")

    train_config = _config.get_config(args.policy.config)

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

    use_mlp = (not args.disable_mlp) and (args.mlp_checkpoint is not None)
    if use_mlp:
        device = policy._pytorch_device  # noqa: SLF001
        ckpt = _load_mlp_checkpoint(args.mlp_checkpoint, device=device)
        proj_matrix = ckpt["proj_matrix"].to(dtype=torch.float32, device=device)
        feature_mean = ckpt["feature_mean"].to(dtype=torch.float32, device=device)
        feature_std = ckpt["feature_std"].to(dtype=torch.float32, device=device)
        target_mean = ckpt.get("target_mean")
        target_std = ckpt.get("target_std")
        if target_mean is not None and target_std is not None:
            target_mean = target_mean.to(dtype=torch.float32, device=device)
            target_std = target_std.to(dtype=torch.float32, device=device)
        residual_scale = ckpt.get("residual_scale", 1.0)
        hidden = int(ckpt.get("mlp_hidden", 256))
        layers = int(ckpt.get("mlp_layers", 2))
        mlp = _build_mlp(int(ckpt["proj_dim"]), int(ckpt["out_dim"]), hidden, layers).to(device)
        mlp.load_state_dict(ckpt["mlp_state"])
        mlp.eval()

        mlp_head = MLPActionHead(
            mlp,
            proj_matrix=proj_matrix,
            feature_mean=feature_mean,
            feature_std=feature_std,
            target_mean=target_mean,
            target_std=target_std,
            residual_scale=float(residual_scale),
            mlp_scale=float(args.mlp_scale),
            horizon=int(ckpt["horizon"]),
        ).to(device)

        model = policy._model  # noqa: SLF001
        if not hasattr(model, "action_out_proj"):
            raise AttributeError("Loaded policy model does not have `action_out_proj`.")
        model.action_out_proj = ResidualActionHead(
            model.action_out_proj,
            mlp_head,
            debug_every=args.debug_every,
        ).to(device)
        logging.info("Attached MLP residual head from %s", args.mlp_checkpoint)
    else:
        logging.info("Serving baseline pi05 (no MLP).")

    policy_metadata = policy.metadata | {
        "mlp_enabled": bool(use_mlp),
        "mlp_checkpoint": args.mlp_checkpoint if use_mlp else None,
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
