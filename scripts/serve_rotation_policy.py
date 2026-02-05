import dataclasses
import logging
import socket
from pathlib import Path
from typing import List, Optional

import torch
import tyro

from openpi.training import checkpoints as _checkpoints
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class DummyKBNNResidualRotationHead(torch.nn.Module):
    """Apply (pi05 + kbnn(pi05)) -> rotate -> pad to 32 dims.

    The "KBNN" here is a dummy 7->7 module that operates on the pi05 output
    (first 7 dims of the 32-dim action).
    """

    def __init__(
        self,
        base_head: torch.nn.Module,
        rotation_matrix: torch.Tensor | None,
        kbnn_scale: float = 1.0,
        disable_kbnn: bool = False,
        debug_every: int = 0,
    ):
        super().__init__()
        self.base_head = base_head
        self.rotation_matrix = rotation_matrix
        self.kbnn_scale = float(kbnn_scale)
        self.disable_kbnn = disable_kbnn
        self._debug_every = max(0, int(debug_every))
        self._debug_step = 0

    def _kbnn(self, action7: torch.Tensor) -> torch.Tensor:
        # Dummy KBNN: identity scaled output (can be replaced later).
        return action7 * self.kbnn_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_head(x)
        base7 = base[..., :7]
        if self.disable_kbnn:
            updated7 = base7
        else:
            updated7 = base7 + self._kbnn(base7)

        if self.rotation_matrix is not None:
            rot = self.rotation_matrix
            updated7 = updated7.clone()
            if updated7.ndim == 3:
                updated7 = updated7 @ rot.T
            elif updated7.ndim == 2:
                updated7 = updated7 @ rot.T
            else:
                raise ValueError(f"Expected 2D or 3D action tensor, got {updated7.shape}")

        out = torch.zeros_like(base)
        out[..., :7] = updated7

        if self._debug_every > 0:
            self._debug_step += 1
            if self._debug_step % self._debug_every == 0:
                with torch.no_grad():
                    base_norm = float(torch.linalg.norm(base))
                    out_norm = float(torch.linalg.norm(out))
                    z_mean = float(updated7[..., 2].mean())
                    grip_mean = float(updated7[..., 6].mean())
                    logging.info(
                        "[kbnn_debug] step=%d base_norm=%.6f out_norm=%.6f z_mean=%.6f grip_mean=%.6f",
                        self._debug_step,
                        base_norm,
                        out_norm,
                        z_mean,
                        grip_mean,
                    )
        return out


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint directory."""

    config: str
    dir: str


@dataclasses.dataclass
class Args:
    """Serve a pi05 policy with dummy KBNN residual + rotation matrix metadata."""

    port: int = 8000
    default_prompt: str | None = None
    pytorch_device: str | None = None
    norm_stats_assets_dir: str | None = None
    disable_kbnn: bool = False
    kbnn_scale: float = 1.0
    record: bool = False
    debug_every: int = 0
    # Rotation matrix to include in server metadata (row-major 49 floats for 7x7).
    rotation_matrix: Optional[List[float]] = None
    policy: Checkpoint = dataclasses.field(default_factory=lambda: Checkpoint(config="pi05_libero", dir=""))


def main(args: Args) -> None:
    if not args.policy.dir:
        raise ValueError("--policy.dir is required and must point to a converted PyTorch checkpoint directory.")
    checkpoint_dir = Path(args.policy.dir)
    if not (checkpoint_dir / "model.safetensors").exists():
        raise FileNotFoundError(f"Expected {checkpoint_dir}/model.safetensors (convert the checkpoint to PyTorch first).")

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

    model = policy._model  # noqa: SLF001
    rotation_tensor = None
    if args.rotation_matrix is not None:
        if len(args.rotation_matrix) != 49:
            raise ValueError("rotation_matrix must have 49 floats (row-major 7x7).")
        rotation_tensor = torch.tensor(args.rotation_matrix, dtype=torch.float32).reshape(7, 7)

    model.action_out_proj = DummyKBNNResidualRotationHead(
        base_head=model.action_out_proj,
        rotation_matrix=rotation_tensor,
        kbnn_scale=args.kbnn_scale,
        disable_kbnn=args.disable_kbnn,
        debug_every=args.debug_every,
    )

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    policy_metadata = dict(policy.metadata)
    if args.rotation_matrix is not None:
        policy_metadata["rotation_matrix"] = rotation_tensor.cpu().numpy()

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
