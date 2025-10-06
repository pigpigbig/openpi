# patch_stack_and_run_norm_stats.py
"""
Patches:
  1) torch.stack to accept HF Datasets Column / PyArrow arrays
  2) torch.utils.data.DataLoader to force num_workers=0
Then runs scripts/compute_norm_stats.py safely.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch

# --- Patch torch.stack -------------------------------------------------------
_orig_stack = torch.stack

def _stack_patch(seq, *args, **kwargs):
    try:
        # If it's a HF Datasets Column or PyArrow Array, turn into a Python list
        if hasattr(seq, "to_pylist"):
            seq = seq.to_pylist()
        else:
            try:
                import pyarrow as pa
                if isinstance(seq, pa.Array):
                    seq = seq.to_pylist()
            except Exception:
                pass

        if not isinstance(seq, (list, tuple)):
            seq = list(seq)

        seq = [t if torch.is_tensor(t) else torch.tensor(t) for t in seq]
    except Exception:
        pass
    return _orig_stack(seq, *args, **kwargs)

torch.stack = _stack_patch
# -----------------------------------------------------------------------------


# --- Force single-worker DataLoader ------------------------------------------
from torch.utils.data import dataloader as _dl_mod
_OrigDataLoader = _dl_mod.DataLoader

def _PatchedDataLoader(*args, **kwargs):
    kwargs["num_workers"] = 0        # force single worker
    kwargs.setdefault("pin_memory", False)
    return _OrigDataLoader(*args, **kwargs)

_dl_mod.DataLoader = _PatchedDataLoader
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    # Make sure multiprocessing start method is set before DataLoader spins up workers.
    import multiprocessing as mp
    try:
        mp.set_start_method("fork", force=True)  # preferred on Linux
    except RuntimeError:
        # If already set or unsupported, ignore.
        pass

    import tyro
    from scripts.compute_norm_stats import main
    tyro.cli(main)
