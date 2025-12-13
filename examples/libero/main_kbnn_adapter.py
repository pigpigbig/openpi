import collections
import dataclasses
import logging
import math
import pathlib

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
import torch
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

from kbnn_adapter import KBNNAdapter, KBNNConfig

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    # Model server parameters
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    # LIBERO environment-specific parameters
    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10
    num_trials_per_task: int = 50

    # Physics twist parameters (optional)
    friction_scale: float = 1.0
    mass_scale: float = 1.0

    # KBNN parameters
    kbnn_hidden1: int = 64
    kbnn_hidden2: int = 64
    kbnn_prior_var: float = 1e-2
    kbnn_obs_noise: float = 1e-3
    kbnn_jitter: float = 1e-6
    kbnn_device: str = "cpu"  # keep CPU by default; eval is dominated by sim + policy server
    online_update: bool = False

    # Videos / utils
    video_out_path: str = "data/libero/videos_kbnn"
    save_video_every: int = 25
    seed: int = 7


def _apply_physics_twist(env, friction_scale: float, mass_scale: float):
    sim = None
    if hasattr(env, "sim"):
        sim = env.sim
    elif hasattr(env, "env") and hasattr(env.env, "sim"):
        sim = env.env.sim

    if sim is None:
        logging.warning("Could not locate sim on env; skipping physics twist.")
        return

    model = getattr(sim, "model", None)
    if model is None:
        logging.warning("Sim has no 'model' attribute; skipping physics twist.")
        return

    if friction_scale != 1.0 and hasattr(model, "geom_friction"):
        try:
            model.geom_friction[:] *= float(friction_scale)
            logging.info(f"[PHYS] scaled geom_friction by {friction_scale}")
        except Exception as e:
            logging.warning(f"[PHYS] failed to scale geom_friction: {e}")

    if mass_scale != 1.0 and hasattr(model, "body_mass"):
        try:
            if model.body_mass.shape[0] > 1:
                model.body_mass[1:] *= float(mass_scale)
            else:
                model.body_mass[:] *= float(mass_scale)
            logging.info(f"[PHYS] scaled body_mass by {mass_scale}")
        except Exception as e:
            logging.warning(f"[PHYS] failed to scale body_mass: {e}")


def _get_libero_env(task, resolution, seed, friction_scale: float, mass_scale: float):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # affects object positions even when using fixed initial state
    _apply_physics_twist(env, friction_scale, mass_scale)
    return env, task_description


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(float(quat[3]))) / den


def _kbnn_input_from_obs(obs) -> np.ndarray:
    """
    KBNN input x (8-dim):
      [eef_pos(3), eef_axisangle(3), gripper_qpos(1), 1.0]
    We append a constant 1.0 feature (NOT a bias term; KBNN bias is disabled in kbnn_adapter.py now).
    """
    state7 = np.concatenate(
        (
            obs["robot0_eef_pos"],
            _quat2axisangle(obs["robot0_eef_quat"]),
            np.atleast_1d(obs["robot0_gripper_qpos"]).reshape(-1)[:1],
        )
    ).astype(np.float32)
    if state7.shape[0] != 7:
        # be defensive if gripper_qpos is different shape
        state7 = state7[:7] if state7.shape[0] > 7 else np.pad(state7, (0, 7 - state7.shape[0]))
    return np.concatenate([state7, np.array([1.0], dtype=np.float32)], axis=0)


def eval_libero(args: Args) -> None:
    np.random.seed(args.seed)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220
    elif args.task_suite_name == "libero_object":
        max_steps = 280
    elif args.task_suite_name == "libero_goal":
        max_steps = 300
    elif args.task_suite_name == "libero_10":
        max_steps = 520
    elif args.task_suite_name == "libero_90":
        max_steps = 400
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    kbnn_cfg = KBNNConfig(
        in_dim=8,
        out_dim=7,
        hidden_dims=(args.kbnn_hidden1, args.kbnn_hidden2),
        prior_var=args.kbnn_prior_var,
        obs_noise=args.kbnn_obs_noise,
        jitter=args.kbnn_jitter,
        device=args.kbnn_device,
    )
    kbnn = KBNNAdapter(kbnn_cfg)
    logging.info(
        f"[KBNN] init in_dim={kbnn_cfg.in_dim}, hidden={kbnn_cfg.hidden_dims}, out_dim={kbnn_cfg.out_dim}, "
        f"use_bias={kbnn_cfg.use_bias}, device={kbnn_cfg.device}"
    )

    # Start evaluation (keep original logging behavior)
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(
            task,
            LIBERO_ENV_RESOLUTION,
            args.seed,
            args.friction_scale,
            args.mass_scale,
        )

        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            env.reset()
            action_plan = collections.deque()

            obs = env.set_init_state(initial_states[episode_idx])

            t = 0
            replay_images = []

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    replay_images.append(img)

                    if not action_plan:
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    np.atleast_1d(obs["robot0_gripper_qpos"]).reshape(-1),
                                )
                            ),
                            "prompt": str(task_description),
                        }

                        action_chunk = client.infer(element)["actions"]
                        assert len(action_chunk) >= args.replan_steps, (
                            f"We want to replan every {args.replan_steps} steps, "
                            f"but policy only predicts {len(action_chunk)} steps."
                        )

                        # ----- KBNN residual on top of policy actions -----
                        x = _kbnn_input_from_obs(obs)  # numpy (8,)
                        x_t = torch.from_numpy(x).float()  # torch (8,)
                        mu_delta, _ = kbnn.predict(x_t)  # torch (7,)
                        delta = mu_delta.detach().cpu().numpy().astype(np.float32)

                        # Apply the same residual to each of the next replan steps
                        for a in action_chunk[: args.replan_steps]:
                            a = np.asarray(a, dtype=np.float32)
                            if a.shape[0] == 7:
                                action_plan.append(a + delta)
                            else:
                                action_plan.append(a)

                    action = np.asarray(action_plan.popleft(), dtype=np.float32)

                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    done = False
                    break

            task_episodes += 1
            total_episodes += 1

            # Save videos every N episodes (and also always save last episode if desired later)
            if args.save_video_every > 0 and (episode_idx % args.save_video_every == 0):
                suffix = "success" if done else "failure"
                task_segment = task_description.replace(" ", "_")
                imageio.mimwrite(
                    pathlib.Path(args.video_out_path)
                    / f"rollout_{task_segment}_ep{episode_idx:03d}_{suffix}.mp4",
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )

            # Keep original logging
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)