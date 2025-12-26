import collections
import dataclasses
import logging
import math
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
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 20  # Number of rollouts per task

    # Camera perturbation: offsets for the "agentview" camera
    camera_pitch_deg: float = 0.0  # tilt toward the table (positive tilts up less steeply)
    camera_yaw_deg: float = 45.0  # move camera around table normal (rotate location)
    camera_fovy_deg: Optional[float] = 80.0  # widen field of view to capture the whole table

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos_camshift"  # Path to save videos
    save_video_every: int = 1  # save video every N episodes

    seed: int = 7  # Random Seed (for reproducibility)


def _rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix (world-from-camera) to MuJoCo quaternion (w, x, y, z)."""
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        S = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * S
        qx = (m21 - m12) / S
        qy = (m02 - m20) / S
        qz = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        qw = (m21 - m12) / S
        qx = 0.25 * S
        qy = (m01 + m10) / S
        qz = (m02 + m20) / S
    elif m11 > m22:
        S = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        qw = (m02 - m20) / S
        qx = (m01 + m10) / S
        qy = 0.25 * S
        qz = (m12 + m21) / S
    else:
        S = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        qw = (m10 - m01) / S
        qx = (m02 + m20) / S
        qy = (m12 + m21) / S
        qz = 0.25 * S
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    return q / (np.linalg.norm(q) + 1e-9)


def _quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion q (w, x, y, z)."""
    w, x, y, z = q
    qvec = np.array([x, y, z], dtype=np.float64)
    uv = np.cross(qvec, v)
    uuv = np.cross(qvec, uv)
    return v + 2.0 * (w * uv + uuv)


def _apply_camera_shift(env, pitch_deg: float, yaw_deg: float = 0.0, fovy_deg: Optional[float] = None) -> None:
    """
    Apply pitch/yaw rotations to the MuJoCo camera named 'agentview', keeping the original base orientation.

    We do this after env construction and after each reset, since some environments
    can rebuild the underlying MuJoCo model on reset.
    """
    sim = None
    if hasattr(env, "sim"):
        sim = env.sim
    elif hasattr(env, "env") and hasattr(env.env, "sim"):
        sim = env.env.sim

    if sim is None:
        logging.warning("[camshift] Could not locate sim on env; skipping camera shift.")
        return

    model = getattr(sim, "model", None)
    if model is None:
        logging.warning("[camshift] sim has no 'model' attribute; skipping camera shift.")
        return

    try:
        cam_id = model.camera_name2id("agentview")
    except Exception as e:
        logging.warning(f"[camshift] Could not find camera 'agentview': {e}")
        return

    # NOTE: positive `camera_pitch_deg` tilts the camera downward; use positive to tilt up if desired.
    # In practice, the previous sign flipped the pitch; here we align sign so +pitch looks up less steeply.
    pitch_theta = math.radians(pitch_deg)
    pitch_quat = np.array(
        [math.cos(pitch_theta / 2.0), math.sin(pitch_theta / 2.0), 0.0, 0.0],
        dtype=np.float64,
    )
    yaw_theta = math.radians(yaw_deg)

    # Grab (or cache) the original camera orientation/position/target so repeated calls don't accumulate rotation.
    base_cache_attr = "_camshift_base_quat"
    model_cache_attr = "_camshift_model_id"
    pos_cache_attr = "_camshift_base_pos"
    target_cache_attr = "_camshift_base_target"
    base_quat = getattr(env, base_cache_attr, None)
    cached_model_id = getattr(env, model_cache_attr, None)
    base_pos = getattr(env, pos_cache_attr, None)
    base_target = getattr(env, target_cache_attr, None)
    if base_quat is None or base_pos is None or base_target is None or cached_model_id != id(model):
        base_quat = np.array(model.cam_quat[cam_id], dtype=np.float64)
        base_pos = np.array(model.cam_pos[cam_id], dtype=np.float64)
        # Assume camera looks along its -z axis in local frame to define a default target.
        forward = _quat_apply(base_quat, np.array([0.0, 0.0, -1.0], dtype=np.float64))
        base_target = base_pos + forward
        try:
            setattr(env, base_cache_attr, base_quat)
            setattr(env, pos_cache_attr, base_pos)
            setattr(env, target_cache_attr, base_target)
            setattr(env, model_cache_attr, id(model))
        except Exception:
            pass

    # Move camera around table normal: rotate the base position about z by yaw_deg.
    cos_y, sin_y = math.cos(yaw_theta), math.sin(yaw_theta)
    Rz = np.array([[cos_y, -sin_y, 0.0], [sin_y, cos_y, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    new_pos = Rz @ base_pos

    # Build a look-at orientation so the camera points to the base target from its new position, then apply extra pitch.
    target = base_target
    forward_vec = target - new_pos
    norm_f = np.linalg.norm(forward_vec)
    if norm_f < 1e-6:
        forward_vec = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        norm_f = 1.0
    f = forward_vec / norm_f
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(np.dot(f, up)) > 0.99:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    s = np.cross(f, up)
    s /= np.linalg.norm(s) + 1e-9
    u = np.cross(s, f)
    R = np.stack(
        [
            [s[0], u[0], -f[0]],
            [s[1], u[1], -f[1]],
            [s[2], u[2], -f[2]],
        ],
        axis=0,
    )
    lookat_quat = _rotmat_to_quat(R)

    # Apply additional pitch around the camera x-axis in camera frame.
    lw, lx, ly, lz = lookat_quat
    pw, px, py, pz = pitch_quat
    composed = np.array(
        [
            lw * pw - lx * px - ly * py - lz * pz,
            lw * px + lx * pw + ly * pz - lz * py,
            lw * py - lx * pz + ly * pw + lz * px,
            lw * pz + lx * py - ly * px + lz * pw,
        ],
        dtype=np.float64,
    )
    composed /= np.linalg.norm(composed) + 1e-9

    model.cam_pos[cam_id] = new_pos
    model.cam_quat[cam_id] = composed
    if fovy_deg is not None:
        try:
            model.cam_fovy[cam_id] = fovy_deg
        except Exception as e:
            logging.warning(f"[camshift] Failed to set fovy: {e}")
    try:
        sim.forward()
    except Exception:
        # Not fatal; the new orientation will still be picked up on the next sim step
        pass

    logging.info(
        f"[camshift] Set 'agentview' yaw_pos={yaw_deg} deg, pitch={pitch_deg} deg, fovy={fovy_deg}, "
        f"base_pos={base_pos}, new_pos={new_pos}, base_quat={base_quat}, new_quat={composed}"
    )


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")
    logging.info(
        f"[camshift] camera_pitch_deg = {args.camera_pitch_deg}, "
        f"camera_yaw_deg = {args.camera_yaw_deg}, camera_fovy_deg = {args.camera_fovy_deg}"
    )

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
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(
            task,
            LIBERO_ENV_RESOLUTION,
            args.seed,
            args.camera_pitch_deg,
            args.camera_yaw_deg,
            args.camera_fovy_deg,
        )

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment and re-apply camera shift
            env.reset()
            _apply_camera_shift(env, args.camera_pitch_deg, args.camera_yaw_deg, args.camera_fovy_deg)

            action_plan = collections.deque()

            # Set initial states (and re-apply camera shift to be safe)
            obs = env.set_init_state(initial_states[episode_idx])
            _apply_camera_shift(env, args.camera_pitch_deg, args.camera_yaw_deg, args.camera_fovy_deg)

            # Setup
            t = 0
            replay_images = []

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
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

                        # Query model to get action
                        action_chunk = client.infer(element)["actions"]
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode every N episodes
            if replay_images and (episode_idx % args.save_video_every == 0):
                suffix = "success" if done else "failure"
                task_segment = task_description.replace(" ", "_")
                out_path = (
                    pathlib.Path(args.video_out_path)
                    / f"rollout_{task_segment}_ep{episode_idx:03d}_{suffix}.mp4"
                )
                imageio.mimwrite(
                    out_path,
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )
                logging.info(f"[camshift] Saved video: {out_path}")

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(
                f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)"
            )

        # Log final results for this task
        logging.info(
            f"Current task success rate: {float(task_successes) / float(task_episodes)}"
        )
        logging.info(
            f"Current total success rate: {float(total_successes) / float(total_episodes)}"
        )

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


def _get_libero_env(
    task,
    resolution,
    seed,
    camera_pitch_deg: float,
    camera_yaw_deg: float,
    camera_fovy_deg: Optional[float],
):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    _apply_camera_shift(env, camera_pitch_deg, camera_yaw_deg, camera_fovy_deg)
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
