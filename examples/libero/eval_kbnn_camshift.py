import collections
import dataclasses
import logging
import math
import pathlib
from typing import Optional

import imageio
import numpy as np
import torch
import tqdm
import tyro
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from safetensors.torch import load_file as load_safetensors

from KBNN2 import KBNN
from openpi.models import pi0_config
from openpi.models_pytorch import pi0_pytorch
from openpi_client import image_tools

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    # Torch pi05 checkpoint (from convert_jax_model_to_pytorch)
    checkpoint_dir: str = "/media/data-ssd-2/qiaoan_ckpt/pi05_libero_pytorch"

    # KBNN checkpoint produced by scripts/train_kbnn.py
    kbnn_checkpoint: str = "kbnn_checkpoint.pt"

    # Which task suite to evaluate
    task_suite_name: str = "libero_10"

    # Env ids to evaluate (e.g. held-out [8, 9] if train envs were [0..7])
    env_ids: list[int] = dataclasses.field(default_factory=lambda: [8, 9])
    num_trials_per_task: int = 20
    num_steps_wait: int = 10

    # Preprocessing (should match training / main_camshift.py)
    resize_size: int = 224

    # Camshift: match main_camshift.py (camera location rotated around table)
    camera_pitch_deg: float = 0.0
    camera_yaw_deg: float = 45.0
    camera_fovy_deg: Optional[float] = 80.0

    # Policy behavior
    replan_steps: int = 1  # we run 1-step actions (KBNN predicts one action per obs)
    use_kbnn: bool = True  # if False, use pi05's native action_out_proj baseline

    # Video
    video_out_path: str = "data/libero/videos_kbnn_camshift"
    save_video_every: int = 1
    video_prefix: Optional[str] = None  # default: "kbnn" if use_kbnn else "baseline"

    # Misc
    seed: int = 7
    device: str = "cuda"


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
    w, x, y, z = q
    qvec = np.array([x, y, z], dtype=np.float64)
    uv = np.cross(qvec, v)
    uuv = np.cross(qvec, uv)
    return v + 2.0 * (w * uv + uuv)


def _apply_camera_shift(env, pitch_deg: float, yaw_deg: float = 0.0, fovy_deg: Optional[float] = None) -> None:
    sim = None
    if hasattr(env, "sim"):
        sim = env.sim
    elif hasattr(env, "env") and hasattr(env.env, "sim"):
        sim = env.env.sim
    if sim is None:
        return
    model = getattr(sim, "model", None)
    if model is None:
        return
    try:
        cam_id = model.camera_name2id("agentview")
    except Exception:
        return

    pitch_theta = math.radians(pitch_deg)
    yaw_theta = math.radians(yaw_deg)

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
        forward = _quat_apply(base_quat, np.array([0.0, 0.0, -1.0], dtype=np.float64))
        base_target = base_pos + forward
        try:
            setattr(env, base_cache_attr, base_quat)
            setattr(env, pos_cache_attr, base_pos)
            setattr(env, target_cache_attr, base_target)
            setattr(env, model_cache_attr, id(model))
        except Exception:
            pass

    cos_y, sin_y = math.cos(yaw_theta), math.sin(yaw_theta)
    Rz = np.array([[cos_y, -sin_y, 0.0], [sin_y, cos_y, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    new_pos = Rz @ base_pos

    # Look-at orientation to keep pointing at base_target from the new position.
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
        try:
            model.cam_fovy[cam_id] = fovy_deg
        except Exception:
            pass
    try:
        sim.forward()
    except Exception:
        pass


def _get_libero_env(task, resolution, seed, camera_pitch_deg, camera_yaw_deg, camera_fovy_deg):
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    _apply_camera_shift(env, camera_pitch_deg, camera_yaw_deg, camera_fovy_deg)
    return env


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = quat.copy()
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _preprocess_obs(obs: dict, resize_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Match main_camshift.py preprocessing: rotate 180 deg and resize-with-pad.
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, resize_size, resize_size))
    wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, resize_size, resize_size))
    state = np.concatenate(
        (
            obs["robot0_eef_pos"],
            _quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"][:2],
        )
    ).astype(np.float32)
    return img, wrist_img, state


def _load_pi05_torch(checkpoint_dir: str, device: str) -> pi0_pytorch.PI0Pytorch:
    cfg = pi0_config.Pi0Config(pi05=True)
    model = pi0_pytorch.PI0Pytorch(cfg)
    state_dict = load_safetensors(str(pathlib.Path(checkpoint_dir) / "model.safetensors"), device=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _load_kbnn(kbnn_checkpoint: str, device: str) -> tuple[KBNN, list[int]]:
    ckpt = torch.load(kbnn_checkpoint, map_location="cpu")
    kbnn_geom = ckpt["kbnn_geometry"]
    geom_with_bias = ckpt.get("geometry_with_bias", None)
    kbnn = KBNN(kbnn_geom, cov_mode=ckpt.get("cov_mode", "diag"), device=device, dtype=torch.float32)
    with torch.no_grad():
        for i, w in enumerate(ckpt["mws"]):
            kbnn.mws[i].copy_(w.to(device=device, dtype=kbnn.mws[i].dtype))
    return kbnn, geom_with_bias


@torch.no_grad()
def _infer_action_pi05_or_kbnn(
    model: pi0_pytorch.PI0Pytorch,
    kbnn: Optional[KBNN],
    img: np.ndarray,
    wrist_img: np.ndarray,
    state: np.ndarray,
    *,
    use_kbnn: bool,
) -> np.ndarray:
    # Build a single-step batch for encode_for_actions: [B=1, T=1, ...]
    imgs = img[None, None, ...]
    wrists = wrist_img[None, None, ...]
    states = state[None, None, ...]

    feats = model.encode_for_actions(imgs, wrists, states)  # (1, 1024)

    if use_kbnn:
        assert kbnn is not None
        action32 = kbnn.forward_deterministic(feats)[0].detach().cpu().numpy()
    else:
        # Baseline: use model's native linear projection (no KBNN).
        action32 = model.action_out_proj(feats.to(model.action_out_proj.weight.dtype))[0].float().cpu().numpy()

    # LIBERO expects 7-d action.
    return action32[:7].astype(np.float32)


def eval_kbnn_camshift(args: Args) -> None:
    logging.basicConfig(level=logging.INFO)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    video_prefix = args.video_prefix or ("kbnn" if args.use_kbnn else "baseline")

    model = _load_pi05_torch(args.checkpoint_dir, device=device)
    kbnn, _geom = _load_kbnn(args.kbnn_checkpoint, device=device)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    logging.info(f"Task suite: {args.task_suite_name}, n_tasks={task_suite.n_tasks}, env_ids={args.env_ids}")
    logging.info(f"Using KBNN: {args.use_kbnn}")

    # max_steps depends on suite; mirror main_camshift.py defaults
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
        max_steps = 400

    total_episodes = 0
    total_successes = 0

    for env_id in tqdm.tqdm(args.env_ids):
        task = task_suite.get_task(env_id)
        init_states = task_suite.get_task_init_states(env_id)
        task_desc = str(task.language)

        task_successes = 0
        task_episodes = 0

        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            env = _get_libero_env(
                task,
                LIBERO_ENV_RESOLUTION,
                args.seed + episode_idx,
                args.camera_pitch_deg,
                args.camera_yaw_deg,
                args.camera_fovy_deg,
            )
            env.reset()
            _apply_camera_shift(env, args.camera_pitch_deg, args.camera_yaw_deg, args.camera_fovy_deg)
            obs = env.set_init_state(init_states[episode_idx % len(init_states)])
            _apply_camera_shift(env, args.camera_pitch_deg, args.camera_yaw_deg, args.camera_fovy_deg)

            action_plan = collections.deque()
            replay_images = []

            t = 0
            done = False
            while t < max_steps + args.num_steps_wait:
                if t < args.num_steps_wait:
                    obs, _reward, done, _info = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                img, wrist_img, state = _preprocess_obs(obs, args.resize_size)
                replay_images.append(img)

                if not action_plan:
                    action = _infer_action_pi05_or_kbnn(
                        model, kbnn, img, wrist_img, state, use_kbnn=args.use_kbnn
                    )
                    action_plan.extend([action] * max(1, args.replan_steps))

                action = action_plan.popleft()
                obs, _reward, done, _info = env.step(action.tolist())
                t += 1
                if done:
                    break

            task_episodes += 1
            total_episodes += 1
            if done:
                task_successes += 1
                total_successes += 1

            if replay_images and (episode_idx % args.save_video_every == 0):
                suffix = "success" if done else "failure"
                out_path = (
                    pathlib.Path(args.video_out_path)
                    / f"{video_prefix}_env{env_id:02d}_ep{episode_idx:03d}_{suffix}.mp4"
                )
                imageio.mimwrite(out_path, [np.asarray(x) for x in replay_images], fps=10)
                logging.info(f"Saved video: {out_path}")

            logging.info(
                f"[env {env_id}] ep {episode_idx} done={done} "
                f"task_sr={task_successes}/{task_episodes} total_sr={total_successes}/{total_episodes}"
            )

        logging.info(f"[env {env_id}] final success rate: {task_successes / max(1, task_episodes):.3f} ({task_desc})")

    logging.info(f"TOTAL success rate: {total_successes / max(1, total_episodes):.3f} ({total_successes}/{total_episodes})")


if __name__ == "__main__":
    tyro.cli(eval_kbnn_camshift)
