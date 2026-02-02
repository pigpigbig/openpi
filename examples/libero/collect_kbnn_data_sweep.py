"""
Collect successful LIBERO episodes for a single env across a sweep of camera yaw angles.

Runs a pi0.5 policy via websocket (original pi05 server), gathers observations/actions,
and saves only successful episodes to disk as compressed .npz files.
"""

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
import tyro

LIBERO_ENV_RESOLUTION = 256
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]


@dataclasses.dataclass
class Args:
    # Model server (original pi05)
    host: str = "0.0.0.0"
    port: int = 8000

    # Data collection targets
    task_suite_name: str = "libero_10"
    env_id: int = 0
    successes_per_angle: int = 200
    num_steps_wait: int = 10
    max_steps: int = 400

    # Preprocessing
    resize_size: int = 224

    # Camshift sweep
    camera_pitch_deg: float = 0.0
    yaw_start: float = 0.0
    yaw_end: float = 45.0
    yaw_step: float = 5.0
    camera_fovy_deg: Optional[float] = 80.0

    # Output
    output_dir: str = "data/libero/kbnn_dataset_sweep"
    save_videos: bool = False
    video_every: int = 50

    # Misc
    seed: int = 7
    log_level: str = "INFO"


def _get_libero_env(task, resolution, seed):
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


def _apply_camera_shift_for_render(model, cam_id, base_pos, pitch_deg, yaw_deg, fovy_deg, height, width, sim):
    orig_pos = np.array(model.cam_pos[cam_id], copy=True)
    orig_quat = np.array(model.cam_quat[cam_id], copy=True)
    orig_fovy = float(model.cam_fovy[cam_id])

    pitch_theta = math.radians(pitch_deg)
    yaw_theta = math.radians(yaw_deg)

    cos_y, sin_y = math.cos(yaw_theta), math.sin(yaw_theta)
    Rz = np.array([[cos_y, -sin_y, 0.0], [sin_y, cos_y, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    new_pos = Rz @ base_pos

    forward = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    target = new_pos + forward
    f = target - new_pos
    f /= np.linalg.norm(f) + 1e-9
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(np.dot(f, up)) > 0.99:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    s = np.cross(f, up)
    s /= np.linalg.norm(s) + 1e-9
    u = np.cross(s, f)
    R = np.stack([[s[0], u[0], -f[0]], [s[1], u[1], -f[1]], [s[2], u[2], -f[2]]], axis=0)
    look_quat = _rotmat_to_quat(R)

    pw, px, py, pz = math.cos(pitch_theta / 2.0), math.sin(pitch_theta / 2.0), 0.0, 0.0
    lw, lx, ly, lz = look_quat
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

    model.cam_pos[cam_id] = orig_pos
    model.cam_quat[cam_id] = orig_quat
    model.cam_fovy[cam_id] = orig_fovy
    sim.forward()
    return frame


def _quat2axisangle(quat):
    quat = np.array(quat, dtype=np.float64)
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(max(1.0 - quat[3] * quat[3], 0.0))
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float64)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _yaw_dir_name(yaw: float) -> str:
    return f"yaw_{yaw:.1f}".replace(".", "p")


def collect(args: Args) -> None:
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    task_suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    if args.env_id < 0 or args.env_id >= task_suite.n_tasks:
        raise ValueError(f"env_id must be in [0, {task_suite.n_tasks - 1}]")

    out_root = pathlib.Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    yaw_angles = np.arange(args.yaw_start, args.yaw_end + 1e-9, args.yaw_step, dtype=float)
    if yaw_angles.size == 0:
        raise ValueError("No yaw angles to sweep; check yaw_start/yaw_end/yaw_step.")

    task = task_suite.get_task(args.env_id)
    init_states = task_suite.get_task_init_states(args.env_id)

    for yaw in yaw_angles:
        yaw = float(yaw)
        successes = 0
        attempts = 0
        yaw_dir = out_root / _yaw_dir_name(yaw) / f"env_{args.env_id:02d}"
        yaw_dir.mkdir(parents=True, exist_ok=True)
        logging.info("[env %d] yaw=%.1f collecting %d successes", args.env_id, yaw, args.successes_per_angle)

        while successes < args.successes_per_angle:
            attempts += 1
            env = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed + attempts)
            obs = env.reset()
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

                camshift_img = _apply_camera_shift_for_render(
                    model,
                    cam_id,
                    base_pos,
                    args.camera_pitch_deg,
                    yaw,
                    args.camera_fovy_deg,
                    args.resize_size,
                    args.resize_size,
                    env.sim if hasattr(env, "sim") else env.env.sim,
                )
                camshift_frames.append(image_tools.convert_to_uint8(camshift_img))

                if done:
                    break

            if done:
                successes += 1
                ep_path = yaw_dir / f"ep_{successes:04d}.npz"
                np.savez_compressed(
                    ep_path,
                    images=np.stack(frames, axis=0),
                    camshift_images=np.stack(camshift_frames, axis=0),
                    wrist_images=np.stack(wrist_frames, axis=0),
                    states=np.stack(states, axis=0),
                    actions=np.stack(actions, axis=0),
                    prompt=str(task.language),
                    env_id=args.env_id,
                    camera_yaw_deg=yaw,
                )

                if args.save_videos and successes % max(1, args.video_every) == 0:
                    video_path = ep_path.with_suffix(".mp4")
                    imageio.mimwrite(video_path, [np.asarray(x) for x in frames], fps=10)

                logging.info(
                    "[env %d] yaw=%.1f success %d/%d (attempt %d, steps %d)",
                    args.env_id,
                    yaw,
                    successes,
                    args.successes_per_angle,
                    attempts,
                    t,
                )
            else:
                logging.info("[env %d] yaw=%.1f attempt %d failed", args.env_id, yaw, attempts)


if __name__ == "__main__":
    tyro.cli(collect)
