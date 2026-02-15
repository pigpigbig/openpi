import dataclasses
import logging
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Optional

import tyro


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclasses.dataclass
class Args:
    # Policy server settings
    port: int = 7777
    host: str = "127.0.0.1"
    policy_config: str = "pi05_libero"
    policy_dir: str = ""
    pytorch_device: str = "cuda"
    norm_stats_assets_dir: Optional[str] = None

    # KBNN checkpoint sweep
    kbnn_checkpoint_dir: str = "data/libero/action_expert_io_env06"
    step_start: int = 65
    step_end: int = 295
    step_stride: int = 5
    kbnn_scale: float = 1.0
    disable_kbnn: bool = False

    # LIBERO eval settings (uses main_camshift_env with zero camshift)
    libero_python: str = "examples/libero/.venv/bin/python"
    libero_script: str = "examples/libero/main_camshift_env.py"
    task_suite_name: str = "libero_10"
    num_trials_per_task: int = 50
    env_ids: Optional[List[int]] = None
    num_steps_wait: int = 10
    resize_size: int = 224
    replan_steps: int = 5
    seed: int = 7
    save_videos: bool = True
    save_video_every: int = 5
    video_out_base: str = "data/libero/videos_kbnn_sweep"
    camera_pitch_deg: float = 0.0
    camera_yaw_deg: float = 0.0
    camera_fovy_deg: Optional[float] = None
    camera_radius_scale: float = 1.0

    # Misc
    wait_timeout_s: float = 30.0


def _wait_for_server(host: str, port: int, timeout_s: float) -> bool:
    url = f"http://{host}:{port}/healthz"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, urllib.error.HTTPError):
            time.sleep(0.5)
    return False


def _start_server(args: Args, checkpoint_path: Path) -> subprocess.Popen:
    serve_script = REPO_ROOT / "scripts" / "serve_rotation_policy.py"
    cmd = [
        sys.executable,
        str(serve_script),
        "--port",
        str(args.port),
        "--policy.config",
        args.policy_config,
        "--policy.dir",
        args.policy_dir,
        "--kbnn-checkpoint",
        str(checkpoint_path),
        "--kbnn-scale",
        str(args.kbnn_scale),
        "--pytorch-device",
        args.pytorch_device,
    ]
    if args.norm_stats_assets_dir:
        cmd.extend(["--norm-stats-assets-dir", args.norm_stats_assets_dir])
    if args.disable_kbnn:
        cmd.append("--disable-kbnn")
    return subprocess.Popen(cmd)


def _stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def _eval_command(args: Args, video_out_path: str) -> list[str]:
    cmd = [
        str(REPO_ROOT / args.libero_python),
        str(REPO_ROOT / args.libero_script),
        "--args.host",
        args.host,
        "--args.port",
        str(args.port),
        "--args.task-suite-name",
        args.task_suite_name,
        "--args.num-trials-per-task",
        str(args.num_trials_per_task),
        "--args.num-steps-wait",
        str(args.num_steps_wait),
        "--args.resize-size",
        str(args.resize_size),
        "--args.replan-steps",
        str(args.replan_steps),
        "--args.video-out-path",
        video_out_path,
        "--args.save-video-every",
        str(args.save_video_every),
        "--args.camera-pitch-deg",
        str(args.camera_pitch_deg),
        "--args.camera-yaw-deg",
        str(args.camera_yaw_deg),
        "--args.camera-radius-scale",
        str(args.camera_radius_scale),
        "--args.seed",
        str(args.seed),
    ]
    if args.env_ids:
        cmd.extend(["--args.env-ids", *[str(v) for v in args.env_ids]])
    if args.camera_fovy_deg is None:
        cmd.extend(["--args.camera-fovy-deg", "None"])
    else:
        cmd.extend(["--args.camera-fovy-deg", str(args.camera_fovy_deg)])
    if args.save_videos:
        cmd.append("--args.save-videos")
    else:
        cmd.append("--args.no-save-videos")
    return cmd


def main() -> None:
    args = tyro.cli(Args)
    if not args.policy_dir:
        raise ValueError("--policy-dir is required.")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s", force=True)
    ckpt_dir = Path(args.kbnn_checkpoint_dir)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint dir not found: {ckpt_dir}")

    steps = list(range(args.step_start, args.step_end + 1, args.step_stride))
    if not steps:
        raise ValueError("No steps to run; check step_start/step_end/step_stride.")

    for step in steps:
        checkpoint = ckpt_dir / f"kbnn_weights_step_{step}.pt"
        if not checkpoint.exists():
            logging.warning("[kbnn_sweep] missing checkpoint: %s", checkpoint)
            continue

        logging.info("[kbnn_sweep] Starting step=%d checkpoint=%s", step, checkpoint)
        server_proc = _start_server(args, checkpoint)
        try:
            if not _wait_for_server(args.host, args.port, args.wait_timeout_s):
                logging.error("[kbnn_sweep] server did not become ready for step=%d", step)
                continue

            video_out = str(Path(args.video_out_base) / f"step_{step}")
            eval_cmd = _eval_command(args, video_out)
            env = os.environ.copy()
            extra_path = str(REPO_ROOT / "third_party" / "libero")
            env["PYTHONPATH"] = f"{env.get('PYTHONPATH', '')}:{extra_path}"
            logging.info("[kbnn_sweep] Running eval: %s", " ".join(eval_cmd))
            subprocess.run(eval_cmd, check=True, env=env)
        finally:
            _stop_server(server_proc)


if __name__ == "__main__":
    main()
