# examples/libero/main_log.py
import collections
import dataclasses
import logging
import math
import pathlib
import json
from typing import Optional

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
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Logging
    #################################################################################################################
    log_dir: Optional[str] = "data/libero/io_logs"  # Set to None to disable logs
    log_images: bool = True                         # Save the preprocessed images we send to the policy
    log_debug_last_layer: bool = False              # If server returns last-layer input/output, save them

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos
    seed: int = 7  # Random Seed (for reproducibility)


def _save_io_bundle(
    root: pathlib.Path,
    stem: str,
    prompt: str,
    state_vec: np.ndarray,
    img: np.ndarray,
    wrist_img: np.ndarray,
    replan_steps: int,
    action_chunk: np.ndarray,
    debug_last_layer_input: Optional[np.ndarray] = None,
    debug_last_layer_output: Optional[np.ndarray] = None,
    save_images: bool = True,
) -> None:
    root.mkdir(parents=True, exist_ok=True)

    # Images (uint8)
    if save_images:
        try:
            imageio.imwrite(root / f"{stem}_img.png", img)
            imageio.imwrite(root / f"{stem}_wrist.png", wrist_img)
        except Exception as e:
            logging.warning("Failed to write images for %s: %s", stem, e)

    # JSON summary (human-friendly)
    payload = {
        "prompt": str(prompt),
        "state_len": int(state_vec.shape[0]),
        "state_dtype": str(state_vec.dtype),
        "image_shape": list(img.shape),
        "wrist_image_shape": list(wrist_img.shape),
        "image_dtype": str(img.dtype),
        "wrist_image_dtype": str(wrist_img.dtype),
        "replan_steps": int(replan_steps),
        "action_chunk_shape": list(action_chunk.shape),
        "chosen_actions_shape": list(action_chunk[:replan_steps].shape),
    }
    try:
        (root / f"{stem}_io.json").write_text(json.dumps(payload, indent=2))
    except Exception as e:
        logging.warning("Failed to write IO json for %s: %s", stem, e)

    # Full arrays (.npy) for exact reproducibility
    try:
        np.save(root / f"{stem}_state.npy", state_vec)
        np.save(root / f"{stem}_actions.npy", action_chunk)
        if debug_last_layer_input is not None:
            np.save(root / f"{stem}_last_layer_input.npy", debug_last_layer_input)
        if debug_last_layer_output is not None:
            np.save(root / f"{stem}_last_layer_output.npy", debug_last_layer_output)
    except Exception as e:
        logging.warning("Failed to write arrays for %s: %s", stem, e)


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info("Task suite: %s", args.task_suite_name)

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError("Unknown task suite: %s" % args.task_suite_name)

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info("\nTask: %s", task_description)

            env.reset()
            action_plan = collections.deque()

            obs = env.set_init_state(initial_states[episode_idx])

            t = 0
            replay_images = []

            logging.info("Starting episode %d...", task_episodes + 1)
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: match baseline waiting for objects to settle
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Preprocess images EXACTLY like baseline (rotate + resize_with_pad + uint8)
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # For replay video
                    replay_images.append(img)

                    if not action_plan:
                        # Build element EXACTLY like baseline
                        state_vec = np.concatenate(
                            (
                                obs["robot0_eef_pos"],
                                _quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            )
                        )
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": state_vec,
                            "prompt": str(task_description),
                        }

                        # Inference
                        result = client.infer(element)
                        # Expected: result["actions"] is (H, 7) or similar
                        action_chunk = np.asarray(result["actions"])

                        # Optional debug fields (only if server provided them)
                        debug_input = None
                        debug_output = None
                        if args.log_debug_last_layer:
                            if "last_layer_input" in result:
                                debug_input = np.asarray(result["last_layer_input"])
                            if "last_layer_output" in result:
                                debug_output = np.asarray(result["last_layer_output"])

                        # Sanity check identical to baseline
                        if len(action_chunk) < args.replan_steps:
                            raise RuntimeError(
                                "We want to replan every %d steps, but policy only predicts %d steps."
                                % (args.replan_steps, len(action_chunk))
                            )

                        # Save IO bundle (does NOT change behavior)
                        if args.log_dir is not None:
                            stem = "task%02d_ep%03d_t%04d" % (task_id, episode_idx, t)
                            _save_io_bundle(
                                root=pathlib.Path(args.log_dir),
                                stem=stem,
                                prompt=task_description,
                                state_vec=state_vec,
                                img=img,
                                wrist_img=wrist_img,
                                replan_steps=args.replan_steps,
                                action_chunk=action_chunk,
                                debug_last_layer_input=debug_input,
                                debug_last_layer_output=debug_output,
                                save_images=bool(args.log_images),
                            )

                        # Queue first K actions (unchanged)
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()

                    # Step env (unchanged)
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error("Caught exception: %s", e)
                    break

            task_episodes += 1
            total_episodes += 1

            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            try:
                imageio.mimwrite(
                    pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )
            except Exception as e:
                logging.warning("Failed to write video for episode: %s", e)

            logging.info("Success: %s", done)
            logging.info("# episodes completed so far: %d", total_episodes)
            logging.info("# successes: %d (%.1f%%)", total_successes, (total_successes / max(total_episodes, 1) * 100.0))

        logging.info("Current task success rate: %f", float(task_successes) / float(max(task_episodes, 1)))
        logging.info("Current total success rate: %f", float(total_successes) / float(max(total_episodes, 1)))

    logging.info("Total success rate: %f", float(total_successes) / float(max(total_episodes, 1)))
    logging.info("Total episodes: %d", total_episodes)


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
