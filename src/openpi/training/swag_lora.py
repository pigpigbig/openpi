from __future__ import annotations

import dataclasses
import json
import pickle
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import flax.nnx as nnx
import jax
import numpy as np

import openpi.shared.nnx_utils as nnx_utils


def lora_param_filter() -> nnx.filterlib.Filter:
    """Match LoRA parameters only."""
    return nnx.All(nnx.Param, nnx_utils.PathRegex(".*lora.*"))


def freeze_non_lora_params_filter() -> nnx.filterlib.Filter:
    """Freeze every parameter except LoRA parameters."""
    return nnx.All(nnx.Param, nnx.Not(nnx_utils.PathRegex(".*lora.*")))


def _path_to_str(path: nnx.filterlib.PathParts) -> str:
    return "/".join(str(part) for part in path)


def extract_lora_params(state: nnx.State) -> dict[str, np.ndarray]:
    """Extract LoRA params as a host-side mapping from path string to numpy array."""
    lora_state = state.filter(lora_param_filter())
    return {
        _path_to_str(path): np.asarray(jax.device_get(value.value), dtype=np.float32)
        for path, value in lora_state.flat_state().items()
    }


@dataclasses.dataclass
class SWAGLoRACollector:
    """Track SWAG statistics over LoRA parameters only.

    This matches the paper's core idea: keep a posterior only over the trainable LoRA
    weights while the frozen base model remains fixed.
    """

    max_num_models: int = 10
    var_clamp: float = 1e-30
    n_models: int = 0
    last_collect_step: int = -1
    param_order: list[str] = dataclasses.field(default_factory=list)
    param_shapes: dict[str, tuple[int, ...]] = dataclasses.field(default_factory=dict)
    mean: dict[str, np.ndarray] = dataclasses.field(default_factory=dict)
    sq_mean: dict[str, np.ndarray] = dataclasses.field(default_factory=dict)
    deviations: dict[str, list[np.ndarray]] = dataclasses.field(default_factory=dict)

    def collect_from_state(self, state: nnx.State, step: int) -> None:
        self.collect(extract_lora_params(state), step)

    def collect(self, params: dict[str, np.ndarray], step: int) -> None:
        if not params:
            raise ValueError("No LoRA parameters were found to collect into SWAG.")

        if not self.param_order:
            self.param_order = sorted(params.keys())
            self.param_shapes = {name: tuple(params[name].shape) for name in self.param_order}
            self.mean = {name: np.zeros(params[name].size, dtype=np.float32) for name in self.param_order}
            self.sq_mean = {name: np.zeros(params[name].size, dtype=np.float32) for name in self.param_order}
            self.deviations = {name: [] for name in self.param_order}

        missing = sorted(set(self.param_order) - set(params))
        extra = sorted(set(params) - set(self.param_order))
        if missing or extra:
            raise ValueError(f"LoRA parameter mismatch while collecting SWAG. Missing={missing}, extra={extra}")

        for name in self.param_order:
            flat = np.asarray(params[name], dtype=np.float32).reshape(-1)
            n = float(self.n_models)
            new_mean = self.mean[name] * (n / (n + 1.0)) + flat / (n + 1.0)
            new_sq_mean = self.sq_mean[name] * (n / (n + 1.0)) + np.square(flat) / (n + 1.0)
            dev = flat - new_mean

            self.mean[name] = new_mean
            self.sq_mean[name] = new_sq_mean
            self.deviations[name].append(dev.astype(np.float32, copy=True))
            if len(self.deviations[name]) > self.max_num_models:
                self.deviations[name] = self.deviations[name][-self.max_num_models :]

        self.n_models += 1
        self.last_collect_step = int(step)

    def summary(self) -> dict[str, Any]:
        return {
            "n_models": self.n_models,
            "last_collect_step": self.last_collect_step,
            "num_lora_tensors": len(self.param_order),
            "max_num_models": self.max_num_models,
            "var_clamp": self.var_clamp,
            "param_order": self.param_order,
            "param_shapes": {k: list(v) for k, v in self.param_shapes.items()},
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "max_num_models": self.max_num_models,
            "var_clamp": self.var_clamp,
            "n_models": self.n_models,
            "last_collect_step": self.last_collect_step,
            "param_order": list(self.param_order),
            "param_shapes": {k: tuple(v) for k, v in self.param_shapes.items()},
            "mean": self.mean,
            "sq_mean": self.sq_mean,
            "deviations": self.deviations,
        }

    @classmethod
    def from_state_dict(cls, state_dict: dict[str, Any]) -> "SWAGLoRACollector":
        collector = cls(
            max_num_models=int(state_dict["max_num_models"]),
            var_clamp=float(state_dict["var_clamp"]),
        )
        collector.n_models = int(state_dict["n_models"])
        collector.last_collect_step = int(state_dict["last_collect_step"])
        collector.param_order = list(state_dict["param_order"])
        collector.param_shapes = {k: tuple(v) for k, v in state_dict["param_shapes"].items()}
        collector.mean = {k: np.asarray(v, dtype=np.float32) for k, v in state_dict["mean"].items()}
        collector.sq_mean = {k: np.asarray(v, dtype=np.float32) for k, v in state_dict["sq_mean"].items()}
        collector.deviations = {
            k: [np.asarray(dev, dtype=np.float32) for dev in v] for k, v in state_dict["deviations"].items()
        }
        return collector

    def sample(self, rng: np.random.Generator, *, scale: float = 1.0, use_cov_mat: bool = True) -> dict[str, np.ndarray]:
        if self.n_models == 0:
            raise ValueError("Cannot sample from SWAG before any models have been collected.")

        scale_sqrt = np.sqrt(scale * 0.5)
        sampled: dict[str, np.ndarray] = {}
        for name in self.param_order:
            mean = self.mean[name]
            var = np.clip(self.sq_mean[name] - np.square(mean), self.var_clamp, None)
            diag_sample = rng.standard_normal(mean.shape, dtype=np.float32) * np.sqrt(var, dtype=np.float32)

            cov_sample = np.zeros_like(diag_sample)
            if use_cov_mat and len(self.deviations[name]) > 1:
                D = np.stack(self.deviations[name], axis=0)
                z = rng.standard_normal(D.shape[0], dtype=np.float32)
                cov_sample = (D.T @ z) / np.sqrt(max(D.shape[0] - 1, 1))

            flat = mean + scale_sqrt * (diag_sample + cov_sample)
            sampled[name] = flat.reshape(self.param_shapes[name])
        return sampled


def save_collector(collector: SWAGLoRACollector, output_dir: str | Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "swag_lora_state.pkl"
    summary_path = output_dir / "swag_lora_summary.json"
    with state_path.open("wb") as f:
        pickle.dump(collector.state_dict(), f)
    summary_path.write_text(json.dumps(collector.summary(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return state_path, summary_path


def load_collector(output_dir: str | Path) -> SWAGLoRACollector | None:
    output_dir = Path(output_dir)
    state_path = output_dir / "swag_lora_state.pkl"
    if not state_path.exists():
        return None
    with state_path.open("rb") as f:
        state_dict = pickle.load(f)
    return SWAGLoRACollector.from_state_dict(state_dict)
