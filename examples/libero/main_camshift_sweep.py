import dataclasses
import logging
import pathlib
from typing import Optional

import numpy as np
import tyro

from main_camshift import Args as BaseArgs
from main_camshift import eval_libero


@dataclasses.dataclass
class Args(BaseArgs):
    """Sweep camera yaw from yaw_start to yaw_end in yaw_step increments."""

    yaw_start: float = 0.0
    yaw_end: float = 45.0
    yaw_step: float = 5.0
    video_out_base: str = "data/libero/videos_camshift_sweep"
    camera_pitch_deg: float = 0.0
    camera_fovy_deg: Optional[float] = 80.0


def main() -> None:
    args = tyro.cli(Args)
    angles = np.arange(args.yaw_start, args.yaw_end + 1e-9, args.yaw_step, dtype=float)
    if angles.size == 0:
        raise ValueError("No yaw angles to sweep; check yaw_start/yaw_end/yaw_step.")

    for yaw in angles:
        yaw = float(yaw)
        logging.info("[camshift_sweep] Running yaw=%.1f deg", yaw)
        args.camera_yaw_deg = yaw
        args.video_out_path = str(pathlib.Path(args.video_out_base) / f"yaw_{int(round(yaw))}")
        eval_libero(args)


if __name__ == "__main__":
    main()
