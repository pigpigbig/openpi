from __future__ import annotations

import dataclasses

import tyro

from openpi.training import config as _config
import train as train_script


@dataclasses.dataclass
class Args:
    exp_name: str
    config_name: str = "pi05_libero_lora"
    project_name: str = "openpi"
    checkpoint_base_dir: str = "./checkpoints"
    assets_base_dir: str = "./assets"
    seed: int = 42
    batch_size: int = 32
    num_workers: int = 2
    num_train_steps: int = 30_000
    log_interval: int = 100
    save_interval: int = 1_000
    keep_period: int | None = 5_000
    overwrite: bool = False
    resume: bool = False
    wandb_enabled: bool = True
    fsdp_devices: int = 1


def main(args: Args) -> None:
    base = _config.get_config(args.config_name)
    cfg = dataclasses.replace(
        base,
        exp_name=args.exp_name,
        project_name=args.project_name,
        checkpoint_base_dir=args.checkpoint_base_dir,
        assets_base_dir=args.assets_base_dir,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_train_steps=args.num_train_steps,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        keep_period=args.keep_period,
        overwrite=args.overwrite,
        resume=args.resume,
        wandb_enabled=args.wandb_enabled,
        fsdp_devices=args.fsdp_devices,
    )
    train_script.main(cfg)


if __name__ == "__main__":
    main(tyro.cli(Args))
