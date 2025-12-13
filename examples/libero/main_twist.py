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
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

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

    # Physics twist parameters
    friction_scale: float = 1.0
    mass_scale: float = 1.0

    # Utils
    video_out_path: str = "data/libero/videos"
    seed: int = 7


def _get_sim(env):
    """Best-effort helper to find the underlying MuJoCo sim (used only for physics tweaks)."""
    if hasattr(env, "sim"):
        return env.sim
    if hasattr(env, "env") and hasattr(env.env, "sim"):
        return env.env.sim
    return None


def _apply_physics_twist(env, friction_scale: float, mass_scale: float):
    """
    Scale friction and body masses on the underlying MuJoCo model.

    Also logs before/after stats so we can verify the change actually applied.
    This function is safe to call multiple times: we maintain a baseline copy
    of the original parameters and always scale relative to that.
    """
    sim = _get_sim(env)
    if sim is None:
        logging.warning("Could not locate sim on env; skipping physics twist.")
        return

    model = getattr(sim, "model", None)
    if model is None:
        logging.warning("Sim has no 'model' attribute; skipping physics twist.")
        return

    # ---- Baseline capture / refresh ----
    # Friction
    if hasattr(model, "geom_friction"):
        if (
            not hasattr(env, "_baseline_geom_friction")
            or env._baseline_geom_friction is None
            or env._baseline_geom_friction.shape != model.geom_friction.shape
        ):
            env._baseline_geom_friction = model.geom_friction.copy()
            logging.info(
                f"[TWIST] Captured baseline geom_friction stats: "
                f"min={env._baseline_geom_friction.min():.4f}, "
                f"max={env._baseline_geom_friction.max():.4f}"
            )

    # Mass
    if hasattr(model, "body_mass"):
        if (
            not hasattr(env, "_baseline_body_mass")
            or env._baseline_body_mass is None
            or env._baseline_body_mass.shape != model.body_mass.shape
        ):
            env._baseline_body_mass = model.body_mass.copy()
            logging.info(
                f"[TWIST] Captured baseline body_mass stats: "
                f"min={env._baseline_body_mass.min():.4f}, "
                f"max={env._baseline_body_mass.max():.4f}"
            )

    # ---- Apply scaling relative to baseline ----
    # Scale friction
    if hasattr(model, "geom_friction") and hasattr(env, "_baseline_geom_friction"):
        base_f = env._baseline_geom_friction
        before_min = float(model.geom_friction.min())
        before_max = float(model.geom_friction.max())

        model.geom_friction[:] = base_f * friction_scale

        after_min = float(model.geom_friction.min())
        after_max = float(model.geom_friction.max())
        logging.info(
            f"[TWIST] geom_friction scaled by {friction_scale}: "
            f"min {before_min:.4f} -> {after_min:.4f}, "
            f"max {before_max:.4f} -> {after_max:.4f}"
        )

    # Scale body masses
    if hasattr(model, "body_mass") and hasattr(env, "_baseline_body_mass"):
        base_m = env._baseline_body_mass
        before_min = float(model.body_mass.min())
        before_max = float(model.body_mass.max())

        # skip world body 0 if present
        if base_m.shape[0] > 1:
            model.body_mass[1:] = base_m[1:] * mass_scale
        else:
            model.body_mass[:] = base_m * mass_scale

        after_min = float(model.body_mass.min())
        after_max = float(model.body_mass.max())
        logging.info(
            f"[TWIST] body_mass scaled by {mass_scale}: "
            f"min {before_min:.4f} -> {after_min:.4f}, "
            f"max {before_max:.4f} -> {after_max:.4f}"
        )


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

    total_episodes, total_successes = 0, 0
    per_task_stats = []

    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)

        # NOTE: no physics twist here anymore; we apply it after each reset
        env, task_description = _get_libero_env(
            task,
            LIBERO_ENV_RESOLUTION,
            args.seed,
        )

        task_episodes, task_successes = 0, 0
        task_step_sum = 0  # sum of episode lengths (control steps only)
        first_success_trial = None  # 1-based index of first successful episode

        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment (may do a hard reset / recreate sim)
            env.reset()

            # Re-apply physics twist on the current MuJoCo model
            _apply_physics_twist(env, args.friction_scale, args.mass_scale)

            action_plan = collections.deque()

            # Initialize from stored initial state
            obs = env.set_init_state(initial_states[episode_idx])

            t = 0
            replay_images = []

            # episode length (count only control steps after num_steps_wait)
            control_steps = 0

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    if t < args.num_steps_wait:
                        # warmup steps (not counted)
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # From here on, each step is a "control step"
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
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }

                        action_chunk = client.infer(element)["actions"]
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()

                    # Step the environment with actual control
                    obs, reward, done, info = env.step(action.tolist())
                    control_steps += 1

                    if done:
                        task_successes += 1
                        total_successes += 1
                        if first_success_trial is None:
                            first_success_trial = episode_idx + 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            task_step_sum += control_steps
            logging.info(f"Episode control steps (after wait): {control_steps}")

            # Save a replay video only every 25 runs (per task)
            if episode_idx % 25 == 0:
                suffix = "success" if done else "failure"
                task_segment = task_description.replace(" ", "_")
                out_path = pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_ep{episode_idx:03d}_{suffix}.mp4"
                logging.info(f"Saving replay video for episode {episode_idx} to {out_path}")
                imageio.mimwrite(
                    out_path,
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )

            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Per-task stats
        task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0.0
        avg_episode_len = task_step_sum / task_episodes if task_episodes > 0 else 0.0

        logging.info(f"Current task success rate: {task_success_rate}")
        logging.info(f"Average control steps per episode for this task: {avg_episode_len:.1f}")
        if first_success_trial is not None:
            logging.info(f"Trials to first success for this task: {first_success_trial}")
        else:
            logging.info("Trials to first success for this task: NEVER (no success)")

        per_task_stats.append(
            {
                "task_id": int(task_id),
                "task_description": str(task_description),
                "episodes": int(task_episodes),
                "successes": int(task_successes),
                "success_rate": task_success_rate,
                "avg_episode_len": avg_episode_len,
                "trials_to_first_success": first_success_trial,
            }
        )

    # Final summary
    logging.info("========================================")
    logging.info("Per-task summary (physics-sensitive metrics):")
    for stats in per_task_stats:
        trials_str = (
            str(stats["trials_to_first_success"])
            if stats["trials_to_first_success"] is not None
            else "NEVER"
        )
        logging.info(
            f"[Task {stats['task_id']:02d}] "
            f"success_rate={stats['success_rate']*100:.1f}% "
            f"({stats['successes']}/{stats['episodes']}), "
            f"avg_episode_len={stats['avg_episode_len']:.1f}, "
            f"trials_to_first_success={trials_str}, "
            f"description='{stats['task_description']}'"
        )

    logging.info("========================================")
    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    # NOTE: no physics twist here; we apply it after each reset instead
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)