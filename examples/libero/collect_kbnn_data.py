"""
Collect successful LIBERO episodes with the original camera angle for KBNN training.

Runs a pi0.5 policy via websocket, gathers observations/actions for a set of envs,
and saves only successful episodes to disk as compressed .npz files.
"""

import collections
import dataclasses
import logging
import pathlib
from typing import Optional

import imageio
import numpy as np
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_ENV_RESOLUTION = 256
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]


@dataclasses.dataclass
class Args:
    # Model server
    host: str = "0.0.0.0"
    port: int = 8000

    # Data collection targets
    task_suite_name: str = "libero_10"  # use libero_10 to enumerate 10 envs
    train_envs: int = 8  # first N envs for training; remaining are held out
    successes_per_env: int = 200
    num_steps_wait: int = 10
    max_steps: int = 400  # safety cap per episode (incl. wait steps)

    # Preprocessing
    resize_size: int = 224

    # Output
    output_dir: str = "data/libero/kbnn_dataset"
    save_videos: bool = False
    video_every: int = 50

    # Misc
    seed: int = 7
    log_level: str = "INFO"


def _get_libero_env(task, resolution, seed):
    """Construct LIBERO env with default camera (no shift)."""
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env


def collect(args: Args) -> None:
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}, n_tasks={num_tasks_in_suite}")

    out_root = pathlib.Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Train envs: first N; others implicitly held out
    env_ids = list(range(min(args.train_envs, num_tasks_in_suite)))

    total_success = 0
    for env_id in env_ids:
        task = task_suite.get_task(env_id)
        init_states = task_suite.get_task_init_states(env_id)
        successes = 0
        attempts = 0
        env_dir = out_root / f"env_{env_id:02d}"
        env_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"[env {env_id}] Collecting {args.successes_per_env} successes")

        while successes < args.successes_per_env:
            attempts += 1
            env = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed + attempts)
            obs = env.reset()
            # Use cycling initial states if fewer than attempts
            init_state = init_states[(attempts - 1) % len(init_states)]
            obs = env.set_init_state(init_state)

            action_plan = collections.deque()
            t = 0
            done = False

            frames = []
            wrist_frames = []
            states = []
            actions = []

            while t < args.max_steps:
                # let objects settle
                if t < args.num_steps_wait:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                # preprocess images (no rotation)
                img = np.ascontiguousarray(obs["agentview_image"])
                wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"])
                img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, args.resize_size, args.resize_size))
                wrist_img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                )

                frames.append(img)
                wrist_frames.append(wrist_img)
                states.append(
                    np.concatenate(
                        (
                            obs["robot0_eef_pos"],
                            obs["robot0_eef_quat"],
                            obs["robot0_gripper_qpos"],
                        )
                    ).astype(np.float32)
                )

                if not action_plan:
                    element = {
                        "observation/image": img,
                        "observation/wrist_image": wrist_img,
                        "observation/state": states[-1],
                        "prompt": str(task.language),
                    }
                    action_chunk = client.infer(element)["actions"]
                    action_plan.extend(action_chunk)

                action = action_plan.popleft()
                actions.append(np.array(action, dtype=np.float32))

                obs, reward, done, info = env.step(action.tolist())
                t += 1

                if done:
                    break

            if done:
                successes += 1
                total_success += 1
                ep_path = env_dir / f"ep_{successes:04d}.npz"
                np.savez_compressed(
                    ep_path,
                    images=np.stack(frames, axis=0),
                    wrist_images=np.stack(wrist_frames, axis=0),
                    states=np.stack(states, axis=0),
                    actions=np.stack(actions, axis=0),
                    prompt=str(task.language),
                    env_id=env_id,
                )

                if args.save_videos and successes % max(1, args.video_every) == 0:
                    video_path = ep_path.with_suffix(".mp4")
                    imageio.mimwrite(video_path, [np.asarray(x) for x in frames], fps=10)

                logging.info(
                    f"[env {env_id}] success {successes}/{args.successes_per_env} "
                    f"(attempt {attempts}, steps {t}, total_success={total_success})"
                )
            else:
                logging.info(f"[env {env_id}] attempt {attempts} failed (timeout or no done)")


if __name__ == "__main__":
    tyro.cli(collect)
