import dataclasses
import enum
import logging
import socket

import numpy as np
import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    config: str
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_data_collection script."""

    env: EnvMode = EnvMode.ALOHA_SIM
    default_prompt: str | None = None
    port: int = 8000
    record: bool = False
    record_action_expert_io: bool = False
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(config="pi05_aloha", dir="gs://openpi-assets/checkpoints/pi05_base"),
    EnvMode.ALOHA_SIM: Checkpoint(config="pi0_aloha_sim", dir="gs://openpi-assets/checkpoints/pi0_aloha_sim"),
    EnvMode.DROID: Checkpoint(config="pi05_droid", dir="gs://openpi-assets/checkpoints/pi05_droid"),
    EnvMode.LIBERO: Checkpoint(config="pi05_libero", dir="gs://openpi-assets/checkpoints/pi05_libero"),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    if args.record_action_expert_io:
        if not getattr(policy, "_is_pytorch_model", False):
            raise ValueError("record_action_expert_io requires a PyTorch policy.")
        model = getattr(policy, "_model", None)
        if model is None or not hasattr(model, "action_in_proj") or not hasattr(model, "action_out_proj"):
            raise ValueError("Policy model missing action_in_proj/action_out_proj; cannot record expert IO.")

        capture: dict[str, np.ndarray | None] = {
            "action_expert_in": None,
            "action_out_proj_in": None,
            "action_out_proj_out": None,
        }

        def _hook_action_in(_module, _inputs, output):
            capture["action_expert_in"] = output.detach().cpu().numpy()

        def _hook_action_out(_module, inputs, output):
            capture["action_out_proj_in"] = inputs[0].detach().cpu().numpy()
            capture["action_out_proj_out"] = output.detach().cpu().numpy()

        model.action_in_proj.register_forward_hook(_hook_action_in)
        model.action_out_proj.register_forward_hook(_hook_action_out)

        policy_metadata = dict(policy_metadata)
        policy_metadata["action_out_proj_weight"] = model.action_out_proj.weight.detach().cpu().numpy()
        policy_metadata["action_out_proj_bias"] = model.action_out_proj.bias.detach().cpu().numpy()

        class _ExpertIOPolicy(_policy.BasePolicy):
            def __init__(self, base_policy: _policy.BasePolicy):
                self._base = base_policy

            def infer(self, obs: dict) -> dict:  # type: ignore[override]
                capture["action_expert_in"] = None
                capture["action_out_proj_in"] = None
                capture["action_out_proj_out"] = None
                out = self._base.infer(obs)
                if capture["action_expert_in"] is not None:
                    out["action_expert_in"] = capture["action_expert_in"]
                if capture["action_out_proj_in"] is not None:
                    out["action_out_proj_in"] = capture["action_out_proj_in"]
                if capture["action_out_proj_out"] is not None:
                    out["action_out_proj_out"] = capture["action_out_proj_out"]
                return out

            def reset(self) -> None:
                return self._base.reset()

        policy = _ExpertIOPolicy(policy)

    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

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
