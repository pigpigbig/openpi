# patch_stack_and_run_norm_stats.py
"""
Patches torch.stack to accept HF Datasets Column / PyArrow arrays,
then safely runs scripts/compute_norm_stats.py under a __main__ guard
so PyTorch DataLoader multiprocessing doesn't crash.
"""

import torch

# --- Patch torch.stack -------------------------------------------------------
_orig_stack = torch.stack

def _stack_patch(seq, *args, **kwargs):
    # Accept HF datasets Column / PyArrow arrays and plain lists
    try:
        # If it's a datasets Column, convert to Python list
        if hasattr(seq, "to_pylist"):
            seq = seq.to_pylist()
        else:
            # If it's a PyArrow Array, convert to Python list
            try:
                import pyarrow as pa
                if isinstance(seq, pa.Array):
                    seq = seq.to_pylist()
            except Exception:
                pass

        # Ensure it's an iterable and elements are tensors
        if not isinstance(seq, (list, tuple)):
            seq = list(seq)
        seq = [t if torch.is_tensor(t) else torch.tensor(t) for t in seq]
    except Exception:
        # Fall back to original behavior on any unexpected type
        pass
    return _orig_stack(seq, *args, **kwargs)

# Patch torch.stack for this process (and any worker that imports this module)
torch.stack = _stack_patch
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    # Avoid spawn/import recursion with DataLoader workers.
    import multiprocessing as mp
    try:
        mp.set_start_method("fork", force=True)  # preferred on Linux
    except RuntimeError:
        # If already set (or on non-Linux), try spawn (safe fallback).
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

    import tyro
    from scripts.compute_norm_stats import main
    tyro.cli(main)
