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
ALL_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"]


@dataclasses.dataclass
class Args:
    # Policy server
    port: int = 8000
    host: str = "127.0.0.1"
    policy_config: str = "pi05_libero"
    policy_dir: str = ""
    wait_timeout_s: float = 45.0

    # LIBERO eval process (py3.8 env)
    libero_python: str = "examples/libero/.venv/bin/python"
    libero_script: str = "examples/libero/main_camshift_env.py"
    suites: Optional[List[str]] = None
    num_trials_per_task: int = 50
    num_steps_wait: int = 10
    resize_size: int = 224
    replan_steps: int = 5
    seed: int = 7
    env_ids: Optional[List[int]] = None

    # Zero camshift (baseline view)
    camera_pitch_deg: float = 0.0
    camera_yaw_deg: float = 0.0
    camera_fovy_deg: Optional[float] = None
    camera_radius_scale: float = 1.0

    # Output
    save_videos: bool = False
    save_video_every: int = 1
    output_root: str = "data/libero/pi05_all_suites_29999"
    results_filename: str = "summary.txt"
    server_log_filename: str = "server.log"


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


def _start_server(args: Args, log_path: Path) -> subprocess.Popen:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "serve_policy.py"),
        "--env",
        "LIBERO",
        "--port",
        str(args.port),
        "policy:checkpoint",
        "--policy.config",
        args.policy_config,
        "--policy.dir",
        args.policy_dir,
    ]
    log_f = log_path.open("w", encoding="utf-8")
    return subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)


def _stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def _build_eval_cmd(args: Args, suite: str, video_out_path: Path) -> list[str]:
    cmd = [
        str(REPO_ROOT / args.libero_python),
        str(REPO_ROOT / args.libero_script),
        "--args.host",
        args.host,
        "--args.port",
        str(args.port),
        "--args.task-suite-name",
        suite,
        "--args.num-trials-per-task",
        str(args.num_trials_per_task),
        "--args.num-steps-wait",
        str(args.num_steps_wait),
        "--args.resize-size",
        str(args.resize_size),
        "--args.replan-steps",
        str(args.replan_steps),
        "--args.video-out-path",
        str(video_out_path),
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


def _parse_eval_line(line: str, summary: dict) -> bool:
    text = line.strip()
    if not text:
        return False
    tokens = (
        "Success:",
        "# episodes completed so far:",
        "# successes:",
        "Current task success rate:",
        "Current total success rate:",
        "Total success rate:",
        "Total episodes:",
    )
    should_log = any(token in text for token in tokens)

    if "Total success rate:" in text:
        try:
            summary["total_success_rate"] = float(text.split("Total success rate:")[1].strip())
        except ValueError:
            pass
    if "Total episodes:" in text:
        try:
            summary["total_episodes"] = int(text.split("Total episodes:")[1].strip())
        except ValueError:
            pass
    return should_log


def _run_suite(args: Args, suite: str, video_out_path: Path) -> dict:
    cmd = _build_eval_cmd(args, suite, video_out_path)
    env = os.environ.copy()
    extra_path = str(REPO_ROOT / "third_party" / "libero")
    env["PYTHONPATH"] = f"{env.get('PYTHONPATH', '')}:{extra_path}"

    summary: dict = {}
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    output_lines: list[str] = []
    if proc.stdout:
        for line in proc.stdout:
            output_lines.append(line)
            if _parse_eval_line(line, summary):
                logging.info("[all_libero][%s] %s", suite, line.strip())
    ret = proc.wait()
    if ret != 0:
        raise subprocess.CalledProcessError(ret, cmd, output="".join(output_lines))
    return summary


def _append_line(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(text + "\n")


def main() -> None:
    args = tyro.cli(Args)
    if not args.policy_dir:
        raise ValueError("--policy-dir is required.")

    suites = args.suites if args.suites else ALL_SUITES
    if not suites:
        raise ValueError("No suites to run.")

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    results_file = out_root / args.results_filename
    server_log = out_root / args.server_log_filename

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s", force=True)
    header = (
        f"[all_libero] run_start policy_dir={args.policy_dir} suites={suites} "
        f"trials={args.num_trials_per_task} env_ids={args.env_ids}"
    )
    logging.info(header)
    _append_line(results_file, header)

    server_proc = _start_server(args, server_log)
    try:
        if not _wait_for_server(args.host, args.port, args.wait_timeout_s):
            raise RuntimeError(
                f"Server did not become ready on {args.host}:{args.port}. Check log: {server_log}"
            )

        suite_results: list[tuple[str, float, int]] = []
        for suite in suites:
            suite_video_out = out_root / suite
            suite_video_out.mkdir(parents=True, exist_ok=True)
            summary = _run_suite(args, suite, suite_video_out)
            success = float(summary.get("total_success_rate", 0.0))
            episodes = int(summary.get("total_episodes", 0))
            suite_results.append((suite, success, episodes))
            line = f"[all_libero] suite={suite} total_success={success:.3f} total_episodes={episodes}"
            logging.info(line)
            _append_line(results_file, line)

        total_eps = sum(x[2] for x in suite_results)
        weighted = sum(x[1] * x[2] for x in suite_results) / total_eps if total_eps > 0 else 0.0
        line = f"[all_libero] overall_weighted_success={weighted:.3f} overall_episodes={total_eps}"
        logging.info(line)
        _append_line(results_file, line)
    finally:
        _stop_server(server_proc)


if __name__ == "__main__":
    main()
