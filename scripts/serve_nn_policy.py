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


class SimpleNN(torch.nn.Module):
    """7 -> 50 -> 40 -> 7 MLP (matches train_kbnn_from_distribution.py)."""

    def __init__(self, input_dim: int = 7, output_dim: int = 7):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, 50)
        self.fc2 = torch.nn.Linear(50, 40)
        self.fc3 = torch.nn.Linear(40, output_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class NNRotationProcessor(torch.nn.Module):
    """Apply (pi05 actions + nn(pi05 actions)) -> rotate -> pad to 32 dims."""

    def __init__(
        self,
        nn_model: torch.nn.Module | None,
        rotation_matrix: torch.Tensor | None,
        nn_scale: float = 1.0,
        disable_nn: bool = False,
        nn_in_mean: torch.Tensor | None = None,
        nn_in_std: torch.Tensor | None = None,
        nn_out_mean: torch.Tensor | None = None,
        nn_out_std: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.nn_model = nn_model
        self.rotation_matrix = rotation_matrix
        self.nn_scale = float(nn_scale)
        self.disable_nn = disable_nn
        if nn_in_mean is not None and nn_in_std is not None:
            self.register_buffer("nn_in_mean", nn_in_mean.clone().detach())
            self.register_buffer("nn_in_std", nn_in_std.clone().detach())
        else:
            self.nn_in_mean = None
            self.nn_in_std = None
        if nn_out_mean is not None and nn_out_std is not None:
            self.register_buffer("nn_out_mean", nn_out_mean.clone().detach())
            self.register_buffer("nn_out_std", nn_out_std.clone().detach())
        else:
            self.nn_out_mean = None
            self.nn_out_std = None

    def _nn(self, action7: torch.Tensor) -> torch.Tensor:
        x = action7
        if self.nn_in_mean is not None and self.nn_in_std is not None:
            mean = self.nn_in_mean
            std = self.nn_in_std
            mask = std > 1e-6
            x = torch.where(mask, (x - mean) / std, x)
        if self.nn_model is None:
            y = x
        else:
            y = self.nn_model(x)
        if self.nn_out_mean is not None and self.nn_out_std is not None:
            mean = self.nn_out_mean
            std = self.nn_out_std
            mask = std > 1e-6
            y = torch.where(mask, y * std + mean, y)
        if float(self.nn_scale) != 1.0:
            y = y * self.nn_scale
        return y

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        base7 = actions[..., :7]
        if self.disable_nn:
            updated7 = base7
        else:
            updated7 = base7 + self._nn(base7)

        if self.rotation_matrix is not None:
            rot = self.rotation_matrix
            updated7 = updated7.clone()
            if updated7.ndim == 3:
                updated7 = updated7 @ rot.T
            elif updated7.ndim == 2:
                updated7 = updated7 @ rot.T
            else:
                raise ValueError(f"Expected 2D or 3D action tensor, got {updated7.shape}")

        out = actions.clone()
        out[..., :7] = updated7
        return out


def _load_nn_checkpoint(path: str) -> dict:
    return torch.load(path, map_location="cpu")


def _looks_like_state_dict(obj: dict) -> bool:
    if not isinstance(obj, dict) or not obj:
        return False
    for key, value in obj.items():
        if not isinstance(key, str):
            return False
        if not torch.is_tensor(value):
            return False
        if not (key.endswith(".weight") or key.endswith(".bias")):
            return False
    return True


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
    nn_checkpoint: str | None = None
    disable_nn: bool = False
    nn_scale: float = 1.0
    nn_in_mean: Optional[List[float]] = None
    nn_in_std: Optional[List[float]] = None
    nn_out_mean: Optional[List[float]] = None
    nn_out_std: Optional[List[float]] = None
    record: bool = False
    debug_every: int = 0
    rotation_matrix: Optional[List[float]] = None
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

    model = policy._model  # noqa: SLF001
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device(args.pytorch_device or "cpu")
    if args.pytorch_device == "cuda" and model_device.type != "cuda":
        logging.warning("[serve_nn] pytorch_device=cuda requested but model is on %s", model_device)

    rotation_tensor = None
    if args.rotation_matrix is not None:
        if len(args.rotation_matrix) != 49:
            raise ValueError("rotation_matrix must have 49 floats (row-major 7x7).")
        rotation_tensor = torch.tensor(args.rotation_matrix, dtype=torch.float32, device=model_device).reshape(7, 7)

    nn_model = None
    if args.nn_checkpoint:
        ckpt = _load_nn_checkpoint(args.nn_checkpoint)
        if rotation_tensor is None and "rotation" in ckpt:
            rotation_tensor = ckpt["rotation"].to(dtype=torch.float32, device=model_device)
        if args.nn_in_mean is None and "ref_mean" in ckpt:
            args.nn_in_mean = ckpt["ref_mean"].reshape(-1).tolist()
        if args.nn_in_std is None and "ref_std" in ckpt:
            args.nn_in_std = ckpt["ref_std"].reshape(-1).tolist()
        if args.nn_out_mean is None and "target_mean" in ckpt:
            args.nn_out_mean = ckpt["target_mean"].reshape(-1).tolist()
        if args.nn_out_std is None and "target_std" in ckpt:
            args.nn_out_std = ckpt["target_std"].reshape(-1).tolist()

        state = None
        for key in ("nn_state", "model_state", "state_dict", "mlp_state", "model_state_dict"):
            if key in ckpt:
                state = ckpt[key]
                break
        if state is None and _looks_like_state_dict(ckpt):
            state = ckpt
        if state is None:
            raise ValueError(
                f"{args.nn_checkpoint} missing NN weights. "
                "Expected one of: nn_state, model_state, state_dict, mlp_state, "
                "or a raw state_dict (fc*.weight/bias)."
            )
        nn_model = SimpleNN(7, 7).to(model_device)
        nn_model.load_state_dict(state)
        nn_model.eval()

    def _tensor_or_none(values: Optional[List[float]], name: str) -> torch.Tensor | None:
        if values is None:
            return None
        if len(values) != 7:
            raise ValueError(f"{name} must have 7 floats (got {len(values)})")
        return torch.tensor(values, dtype=torch.float32, device=model_device)

    nn_in_mean = _tensor_or_none(args.nn_in_mean, "nn_in_mean")
    nn_in_std = _tensor_or_none(args.nn_in_std, "nn_in_std")
    nn_out_mean = _tensor_or_none(args.nn_out_mean, "nn_out_mean")
    nn_out_std = _tensor_or_none(args.nn_out_std, "nn_out_std")

    processor = NNRotationProcessor(
        nn_model=nn_model,
        rotation_matrix=rotation_tensor,
        nn_scale=args.nn_scale,
        disable_nn=args.disable_nn,
        nn_in_mean=nn_in_mean,
        nn_in_std=nn_in_std,
        nn_out_mean=nn_out_mean,
        nn_out_std=nn_out_std,
    ).to(model_device)

    orig_sample_actions = policy._sample_actions  # noqa: SLF001
    debug_every = max(0, int(args.debug_every))
    debug_step = 0

    def _sample_actions_with_nn(device, observation, **kwargs):
        nonlocal debug_step
        actions = orig_sample_actions(device, observation, **kwargs)
        out = processor(actions)
        if debug_every > 0:
            debug_step += 1
            if debug_step % debug_every == 0:
                with torch.no_grad():
                    base7 = actions[..., :7]
                    updated7 = out[..., :7]
                    diff = updated7 - base7
                    diff_norm = float(torch.linalg.norm(diff))
                    diff_mean = float(diff.abs().mean())
                    diff_max = float(diff.abs().max())
                    diff_per_dim = diff.mean(dim=tuple(range(diff.ndim - 1))).tolist()
                    diff_sample = diff.detach().cpu().tolist()
                    logging.info("[nn_debug] step=%d diff=%s", debug_step, diff_per_dim)
        return out

    policy._sample_actions = _sample_actions_with_nn  # noqa: SLF001

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
