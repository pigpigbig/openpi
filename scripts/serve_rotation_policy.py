import dataclasses
import logging
import socket
from pathlib import Path
from typing import List, Optional
from KBNN2 import KBNN

import torch
import tyro

from openpi.training import checkpoints as _checkpoints
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config

# TODO: Change to run with 100 KBNN checkpoints and plot them, rather than running one at the each time
# See train_kbnn_from_action for training detail
# See checkpoints in kbnn_weights_step_0
# step -1: apply normalization and denormalization for input and output of kbnn
# step 0: run 50 tests for checkpoint 58 <- Goal for Feb 6
# step 1: run 50 tests each, for each checkpoint from 18 to 58
# step 2: test the same thing on conventional NN
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
        kbnn_in_mean: torch.Tensor | None = None,
        kbnn_in_std: torch.Tensor | None = None,
        kbnn_out_mean: torch.Tensor | None = None,
        kbnn_out_std: torch.Tensor | None = None,
        kbnn_model: KBNN | None = None,
    ):
        super().__init__()
        self.base_head = base_head
        self.rotation_matrix = rotation_matrix
        self.kbnn_scale = float(kbnn_scale)
        self.disable_kbnn = disable_kbnn
        self._debug_every = max(0, int(debug_every))
        self._debug_step = 0
        self.kbnn_model = kbnn_model
        if kbnn_in_mean is not None and kbnn_in_std is not None:
            self.register_buffer("kbnn_in_mean", kbnn_in_mean.clone().detach())
            self.register_buffer("kbnn_in_std", kbnn_in_std.clone().detach())
        else:
            self.kbnn_in_mean = None
            self.kbnn_in_std = None
        if kbnn_out_mean is not None and kbnn_out_std is not None:
            self.register_buffer("kbnn_out_mean", kbnn_out_mean.clone().detach())
            self.register_buffer("kbnn_out_std", kbnn_out_std.clone().detach())
        else:
            self.kbnn_out_mean = None
            self.kbnn_out_std = None

    def _kbnn(self, action7: torch.Tensor) -> torch.Tensor:
        # Dummy KBNN: identity scaled output (can be replaced later).
        x = action7
        if self.kbnn_in_mean is not None and self.kbnn_in_std is not None:
            mean = self.kbnn_in_mean
            std = self.kbnn_in_std
            mask = std > 1e-6
            x = torch.where(mask, (x - mean) / std, x)
        if self.kbnn_model is None:
            y = x * self.kbnn_scale
        else:
            flat = x.reshape(-1, x.shape[-1])
            outs = []
            with torch.no_grad():
                for i in range(flat.shape[0]):
                    cache = self.kbnn_model.forward(flat[i])
                    outs.append(cache["mus"][-1])
            y = torch.stack(outs, dim=0).reshape(x.shape)
        if self.kbnn_out_mean is not None and self.kbnn_out_std is not None:
            mean = self.kbnn_out_mean
            std = self.kbnn_out_std
            mask = std > 1e-6
            y = torch.where(mask, y * std + mean, y)
        return y

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
            # TODO: put rotation matrix to the left
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
                    diff = updated7 - base7
                    diff_norm = float(torch.linalg.norm(diff))
                    diff_mean = float(diff.abs().mean())
                    diff_max = float(diff.abs().max())
                    z_mean = float(updated7[..., 2].mean())
                    grip_mean = float(updated7[..., 6].mean())
                    logging.info(
                        "[kbnn_debug] step=%d base_norm=%.6f out_norm=%.6f diff_norm=%.6f diff_mean=%.6f diff_max=%.6f z_mean=%.6f grip_mean=%.6f",
                        self._debug_step,
                        base_norm,
                        out_norm,
                        diff_norm,
                        diff_mean,
                        diff_max,
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
    kbnn_checkpoint: str | None = None
    disable_kbnn: bool = False
    kbnn_scale: float = 1.0
    kbnn_in_mean: Optional[List[float]] = None
    kbnn_in_std: Optional[List[float]] = None
    kbnn_out_mean: Optional[List[float]] = None
    kbnn_out_std: Optional[List[float]] = None
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
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device(args.pytorch_device or "cpu")
    if args.pytorch_device == "cuda" and model_device.type != "cuda":
        logging.warning("[serve_rotation] pytorch_device=cuda requested but model is on %s", model_device)
    rotation_tensor = None
    if args.rotation_matrix is not None:
        if len(args.rotation_matrix) != 49:
            raise ValueError("rotation_matrix must have 49 floats (row-major 7x7).")
        rotation_tensor = torch.tensor(args.rotation_matrix, dtype=torch.float32, device=model_device).reshape(7, 7)

    kbnn_model = None
    if args.kbnn_checkpoint:
        kbnn_ckpt = torch.load(args.kbnn_checkpoint, map_location="cpu")
        if rotation_tensor is None and "rotation" in kbnn_ckpt:
            rotation_tensor = kbnn_ckpt["rotation"].to(dtype=torch.float32, device=model_device)
        if args.kbnn_in_mean is None and "ref_mean" in kbnn_ckpt:
            args.kbnn_in_mean = kbnn_ckpt["ref_mean"].reshape(-1).tolist()
        if args.kbnn_in_std is None and "ref_std" in kbnn_ckpt:
            args.kbnn_in_std = kbnn_ckpt["ref_std"].reshape(-1).tolist()
        if args.kbnn_out_mean is None and "target_mean" in kbnn_ckpt:
            args.kbnn_out_mean = kbnn_ckpt["target_mean"].reshape(-1).tolist()
        if args.kbnn_out_std is None and "target_std" in kbnn_ckpt:
            args.kbnn_out_std = kbnn_ckpt["target_std"].reshape(-1).tolist()
        if "mws" in kbnn_ckpt and "sws" in kbnn_ckpt:
            layers = []
            for mw in kbnn_ckpt["mws"]:
                in_dim = mw.shape[1] - 1
                out_dim = mw.shape[0]
                if not layers:
                    layers.append(in_dim)
                layers.append(out_dim)
            kbnn_model = KBNN(layers, dtype=torch.float32, device=str(model_device))
            kbnn_model.mws = [w.to(dtype=torch.float32, device=model_device) for w in kbnn_ckpt["mws"]]
            kbnn_model.sws = [w.to(dtype=torch.float32, device=model_device) for w in kbnn_ckpt["sws"]]

    def _tensor_or_none(values: Optional[List[float]], name: str) -> torch.Tensor | None:
        if values is None:
            return None
        if len(values) != 7:
            raise ValueError(f"{name} must have 7 floats (got {len(values)})")
        return torch.tensor(values, dtype=torch.float32)

    kbnn_in_mean = _tensor_or_none(args.kbnn_in_mean, "kbnn_in_mean")
    kbnn_in_std = _tensor_or_none(args.kbnn_in_std, "kbnn_in_std")
    kbnn_out_mean = _tensor_or_none(args.kbnn_out_mean, "kbnn_out_mean")
    kbnn_out_std = _tensor_or_none(args.kbnn_out_std, "kbnn_out_std")
    if kbnn_in_mean is not None:
        kbnn_in_mean = kbnn_in_mean.to(model_device)
    if kbnn_in_std is not None:
        kbnn_in_std = kbnn_in_std.to(model_device)
    if kbnn_out_mean is not None:
        kbnn_out_mean = kbnn_out_mean.to(model_device)
    if kbnn_out_std is not None:
        kbnn_out_std = kbnn_out_std.to(model_device)
    action_head = DummyKBNNResidualRotationHead(
        base_head=model.action_out_proj,
        rotation_matrix=rotation_tensor,
        kbnn_scale=args.kbnn_scale,
        disable_kbnn=args.disable_kbnn,
        debug_every=args.debug_every,
        kbnn_in_mean=kbnn_in_mean,
        kbnn_in_std=kbnn_in_std,
        kbnn_out_mean=kbnn_out_mean,
        kbnn_out_std=kbnn_out_std,
        kbnn_model=kbnn_model,
    )
    action_head = action_head.to(model_device)
    model.action_out_proj = action_head

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
