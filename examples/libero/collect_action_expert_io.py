"""
Collect action-expert I/O from the vanilla pi05 policy (no camshift, no KBNN).

For each policy inference step, this saves:
  - action_expert_in: (H, 1024) from action_in_proj (last denoise step)
  - action_out_proj_in: (H, 1024) from action_out_proj input (suffix_out, last denoise step)
  - action_out_proj_out: (H, 32) from action_out_proj output (v_t, last denoise step)
  - actions: (H, 32) final actions returned by the policy

Also saves action_out_proj weights/bias to meta.pt once per run.
"""

from __future__ import annotations

import collections
import dataclasses
import logging
import pathlib
from typing import List, Optional

import numpy as np
import torch
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
import tqdm
import tyro

from openpi.policies import policy_config as _policy_config
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config

LIBERO_ENV_RESOLUTION = 256
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]


@dataclasses.dataclass
class Args:
    # Model checkpoint
    checkpoint_dir: str
    norm_stats_assets_dir: str
    policy_config: str = "pi05_libero"
    device: str = "cuda"

    # LIBERO evaluation
    task_suite_name: str = "libero_10"
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    env_ids: Optional[List[int]] = None
    max_steps: int = 520

    # Policy preprocessing
    resize_size: int = 224
    replan_steps: int = 5

    # Output
    output_dir: str = "data/libero/action_expert_io"

    # Misc
    seed: int = 7
    log_level: str = "INFO"


def _get_libero_env(task, resolution, seed):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat):
    quat = np.array(quat, dtype=np.float64)
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(max(1.0 - quat[3] * quat[3], 0.0))
    if np.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float64)
    return (quat[:3] * 2.0 * np.arccos(quat[3])) / den


def main(args: Args) -> None:
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), force=True)

    train_config = _config.get_config(args.policy_config)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if data_config.asset_id is None:
        raise ValueError("Train config has no asset_id; cannot load norm stats.")
    norm_stats = _checkpoints.load_norm_stats(pathlib.Path(args.norm_stats_assets_dir), data_config.asset_id)

    policy = _policy_config.create_trained_policy(
        train_config,
        args.checkpoint_dir,
        norm_stats=norm_stats,
        pytorch_device=args.device,
    )
    model = policy._model  # noqa: SLF001
    model.eval()

    # Capture hooks (last denoise step only).
    capture: dict[str, torch.Tensor | None] = {
        "action_in": None,
        "action_out_in": None,
        "action_out": None,
    }

    def _hook_action_in(_module, _inputs, output):
        capture["action_in"] = output.detach().cpu()

    def _hook_action_out(_module, inputs, output):
        capture["action_out_in"] = inputs[0].detach().cpu()
        capture["action_out"] = output.detach().cpu()

    h1 = model.action_in_proj.register_forward_hook(_hook_action_in)
    h2 = model.action_out_proj.register_forward_hook(_hook_action_out)

    out_root = pathlib.Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    # Save action_out_proj weights/bias once.
    meta_path = out_root / "meta.pt"
    if not meta_path.exists():
        torch.save(
            {
                "action_out_proj_weight": model.action_out_proj.weight.detach().cpu(),
                "action_out_proj_bias": model.action_out_proj.bias.detach().cpu(),
            },
            meta_path,
        )

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    if args.env_ids:
        task_ids = [tid for tid in args.env_ids if 0 <= tid < num_tasks_in_suite]
        if not task_ids:
            raise ValueError(f"No valid env_ids provided (0..{num_tasks_in_suite - 1}).")
    else:
        task_ids = list(range(num_tasks_in_suite))

    for task_id in tqdm.tqdm(task_ids):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env_dir = out_root / f"env_{task_id:02d}"
        env_dir.mkdir(parents=True, exist_ok=True)

        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])

            action_plan = collections.deque()
            t = 0
            done = False

            expert_in_list = []
            out_in_list = []
            out_list = []
            actions_list = []

            while t < args.max_steps + args.num_steps_wait:
                if t < args.num_steps_wait:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, args.resize_size, args.resize_size))
                wrist_img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                )

                if not action_plan:
                    element = {
                        "observation/image": img,
                        "observation/wrist_image": wrist_img,
                        "observation/state": np.concatenate(
                            (
                                obs["robot0_eef_pos"],
                                _quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            )
                        ).astype(np.float32),
                        "prompt": str(task_description),
                    }

                    capture["action_in"] = None
                    capture["action_out_in"] = None
                    capture["action_out"] = None
                    outputs = policy.infer(element)
                    action_chunk = outputs["actions"]
                    action_plan.extend(action_chunk[: args.replan_steps])

                    if capture["action_in"] is None or capture["action_out"] is None:
                        logging.warning(
                            "[expert_io] Missing hook captures at env %d ep %d step %d",
                            task_id,
                            episode_idx,
                            t,
                        )
                    else:
                        expert_in_list.append(capture["action_in"][0].numpy())
                        out_in_list.append(capture["action_out_in"][0].numpy())
                        out_list.append(capture["action_out"][0].numpy())
                        actions_list.append(np.asarray(action_chunk))

                action = action_plan.popleft()
                obs, reward, done, info = env.step(action.tolist())
                if done:
                    break
                t += 1

            ep_path = env_dir / f"ep_{episode_idx:04d}.npz"
            np.savez_compressed(
                ep_path,
                action_expert_in=np.asarray(expert_in_list, dtype=np.float32),
                action_out_proj_in=np.asarray(out_in_list, dtype=np.float32),
                action_out_proj_out=np.asarray(out_list, dtype=np.float32),
                actions=np.asarray(actions_list, dtype=np.float32),
                env_id=task_id,
                prompt=str(task_description),
            )

    h1.remove()
    h2.remove()


if __name__ == "__main__":
    tyro.cli(main)
