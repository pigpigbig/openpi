from __future__ import annotations

import dataclasses
import functools
import logging
from pathlib import Path

from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np
import tqdm_loggable.auto as tqdm
import tyro
import wandb

import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.sharding as sharding
import openpi.training.swag_lora as swag_lora
import openpi.training.utils as training_utils
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

    swag_start_step: int = 20_000
    swag_collect_interval: int = 1_000
    swag_max_num_models: int = 10
    swag_var_clamp: float = 1e-30


def _build_config(args: Args) -> _config.TrainConfig:
    base = _config.get_config(args.config_name)
    return dataclasses.replace(
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
        ema_decay=None,
    )


def _save_swag(collector: swag_lora.SWAGLoRACollector, checkpoint_dir: Path) -> None:
    state_path, summary_path = swag_lora.save_collector(collector, checkpoint_dir)
    logging.info("Saved SWAG-LoRA state to %s and %s", state_path, summary_path)


def main(args: Args) -> None:
    config = _build_config(args)

    train_script.init_logging()
    logging.info("Starting SWAG-LoRA training with config=%s exp_name=%s", config.name, config.exp_name)
    logging.info(
        "SWAG settings: start_step=%d collect_interval=%d max_num_models=%d",
        args.swag_start_step,
        args.swag_collect_interval,
        args.swag_max_num_models,
    )

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    train_script.init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = _data_loader.create_data_loader(config, sharding=data_sharding, shuffle=True)
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info("Initialized data loader:\n%s", training_utils.array_tree_to_info(batch))

    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = train_script.init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info("Initialized train state:\n%s", training_utils.array_tree_to_info(train_state.params))

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    collector = swag_lora.load_collector(config.checkpoint_dir) or swag_lora.SWAGLoRACollector(
        max_num_models=args.swag_max_num_models,
        var_clamp=args.swag_var_clamp,
    )
    if collector.n_models > 0:
        logging.info(
            "Restored SWAG-LoRA collector with n_models=%d last_collect_step=%d",
            collector.n_models,
            collector.last_collect_step,
        )

    ptrain_step = jax.jit(
        functools.partial(train_script.train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)

        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []

        if (
            step >= args.swag_start_step
            and step % args.swag_collect_interval == 0
            and step > collector.last_collect_step
        ):
            collector.collect_from_state(train_state.params, step)
            logging.info(
                "Collected SWAG-LoRA snapshot at step=%d (n_models=%d)",
                step,
                collector.n_models,
            )
            wandb.log({"swag_lora/n_models": collector.n_models}, step=step)
            _save_swag(collector, config.checkpoint_dir)

        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)
            if collector.n_models > 0:
                _save_swag(collector, config.checkpoint_dir)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()

    if collector.n_models == 0:
        logging.warning(
            "Training completed without collecting any SWAG-LoRA snapshots. "
            "Check swag_start_step=%d against num_train_steps=%d.",
            args.swag_start_step,
            config.num_train_steps,
        )
    else:
        _save_swag(collector, config.checkpoint_dir)


if __name__ == "__main__":
    main(tyro.cli(Args))
