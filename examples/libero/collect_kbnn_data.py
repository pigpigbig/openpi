"""
Collect successful LIBERO episodes with the original camera angle for KBNN training.

Runs a pi0.5 policy via websocket, gathers observations/actions for a set of envs,
and saves only successful episodes to disk as compressed .npz files.
"""

import collections
import dataclasses
import logging
import math
import pathlib
from typing import List, Optional

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
    env_ids: Optional[List[int]] = None  # explicit env ids to collect (overrides train_envs)
    successes_per_env: int = 200
    num_steps_wait: int = 10
    max_steps: int = 400  # safety cap per episode (incl. wait steps)

    # Preprocessing
    resize_size: int = 224

    # Camshift target view (match main_camshift.py)
    camera_pitch_deg: float = 0.0
    camera_yaw_deg: float = 45.0
    camera_fovy_deg: Optional[float] = 80.0

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


def _rotmat_to_quat(R: np.ndarray) -> np.ndarray:
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


def _apply_camera_shift_for_render(model, cam_id, base_pos, pitch_deg, yaw_deg, fovy_deg, height, width, sim):
    """Temporarily apply camshift (match main_camshift.py) to render a frame, then restore."""
    orig_pos = np.array(model.cam_pos[cam_id], copy=True)
    orig_quat = np.array(model.cam_quat[cam_id], copy=True)
    orig_fovy = float(model.cam_fovy[cam_id])

    pitch_theta = math.radians(pitch_deg)
    yaw_theta = math.radians(yaw_deg)

    base_quat = orig_quat
    base_target = base_pos + _quat_apply(base_quat, np.array([0.0, 0.0, -1.0], dtype=np.float64))

    # rotate position about z by yaw (same as main_camshift)
    cos_y, sin_y = math.cos(yaw_theta), math.sin(yaw_theta)
    Rz = np.array([[cos_y, -sin_y, 0.0], [sin_y, cos_y, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    new_pos = Rz @ base_pos

    # Build a look-at orientation so the camera points to the base target from its new position.
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
    R = np.stack([[s[0], u[0], -f[0]], [s[1], u[1], -f[1]], [s[2], u[2], -f[2]]], axis=0)
    lookat_quat = _rotmat_to_quat(R)

    # apply additional pitch about camera x-axis
    pw, px, py, pz = math.cos(pitch_theta / 2.0), math.sin(pitch_theta / 2.0), 0.0, 0.0
    lw, lx, ly, lz = lookat_quat
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
        model.cam_fovy[cam_id] = fovy_deg

    sim.forward()
    frame = sim.render(height=height, width=width, camera_name="agentview")

    # restore
    model.cam_pos[cam_id] = orig_pos
    model.cam_quat[cam_id] = orig_quat
    model.cam_fovy[cam_id] = orig_fovy
    sim.forward()
    return frame


def _quat2axisangle(quat):
    """Convert quaternion (x, y, z, w) to axis-angle (3,)."""
    quat = np.array(quat, dtype=np.float64)
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(max(1.0 - quat[3] * quat[3], 0.0))
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float64)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def collect(args: Args) -> None:
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}, n_tasks={num_tasks_in_suite}")

    out_root = pathlib.Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Train envs: explicit list overrides first-N behavior
    if args.env_ids:
        env_ids = [eid for eid in args.env_ids if 0 <= eid < num_tasks_in_suite]
        if not env_ids:
            raise ValueError(f"No valid env_ids provided (0..{num_tasks_in_suite - 1}).")
    else:
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
            camshift_frames = []
            wrist_frames = []
            states = []
            actions = []

            model = env.sim.model if hasattr(env, "sim") else env.env.sim.model
            cam_id = model.camera_name2id("agentview")
            base_pos = np.array(model.cam_pos[cam_id], dtype=np.float64)

            while t < args.max_steps:
                # let objects settle
                if t < args.num_steps_wait:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                # preprocess images (match main.py: flip 180 degrees)
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
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
                            _quat2axisangle(obs["robot0_eef_quat"]),
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

                # camshift render (target view) without affecting policy input
                camshift_img = _apply_camera_shift_for_render(
                    model,
                    cam_id,
                    base_pos,
                    args.camera_pitch_deg,
                    args.camera_yaw_deg,
                    args.camera_fovy_deg,
                    args.resize_size,
                    args.resize_size,
                    env.sim if hasattr(env, "sim") else env.env.sim,
                )
                camshift_img = image_tools.convert_to_uint8(camshift_img)
                camshift_frames.append(camshift_img)

                if done:
                    break

            if done:
                successes += 1
                total_success += 1
                ep_path = env_dir / f"ep_{successes:04d}.npz"
                np.savez_compressed(
                    ep_path,
                    images=np.stack(frames, axis=0),  # default camera (policy input)
                    camshift_images=np.stack(camshift_frames, axis=0),  # target camshift view
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
