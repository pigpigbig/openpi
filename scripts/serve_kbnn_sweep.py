import dataclasses
import logging
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Optional

import tyro


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO_ROOT / "examples" / "libero"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))
import main_camshift_env as _eval_libero  # noqa: E402


@dataclasses.dataclass
class Args:
    # Policy server settings
    port: int = 8000
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
    task_suite_name: str = "libero_10"
    num_trials_per_task: int = 50
    env_ids: Optional[List[int]] = None
    num_steps_wait: int = 10
    resize_size: int = 224
    replan_steps: int = 5
    seed: int = 7
    save_videos: bool = False
    save_video_every: int = 1
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


def _build_eval_args(args: Args, video_out_path: str) -> _eval_libero.Args:
    return _eval_libero.Args(
        host=args.host,
        port=args.port,
        resize_size=args.resize_size,
        replan_steps=args.replan_steps,
        task_suite_name=args.task_suite_name,
        num_steps_wait=args.num_steps_wait,
        num_trials_per_task=args.num_trials_per_task,
        env_ids=args.env_ids,
        camera_pitch_deg=args.camera_pitch_deg,
        camera_yaw_deg=args.camera_yaw_deg,
        camera_fovy_deg=args.camera_fovy_deg,
        camera_radius_scale=args.camera_radius_scale,
        video_out_path=video_out_path,
        save_video_every=args.save_video_every,
        save_videos=args.save_videos,
        seed=args.seed,
    )


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
            eval_args = _build_eval_args(args, video_out)
            summary = _eval_libero.eval_libero(eval_args)
            env_rates = {k: round(v, 3) for k, v in summary["env_success_rates"].items()}
            logging.info(
                "[kbnn_sweep] step=%d total_success=%.3f total_episodes=%d env_rates=%s",
                step,
                summary["total_success_rate"],
                summary["total_episodes"],
                env_rates,
            )
        finally:
            _stop_server(server_proc)


if __name__ == "__main__":
    main()
