# patch_stack_and_run_norm_stats.py
import torch

# Save original
_orig_stack = torch.stack

def _stack_patch(seq, *args, **kwargs):
    # Accept HF datasets Column / Arrow arrays and plain lists
    try:
        # If it's a datasets Column, convert to Python list
        if hasattr(seq, "to_pylist"):
            seq = seq.to_pylist()
        # If it's a PyArrow Array, convert to Python list
        try:
            import pyarrow as pa
            if isinstance(seq, pa.Array):
                seq = seq.to_pylist()
        except Exception:
            pass
        # Ensure each element is a torch tensor
        seq = [t if torch.is_tensor(t) else torch.tensor(t) for t in seq]
    except Exception:
        # Fall back to original behavior on any unexpected type
        pass
    return _orig_stack(seq, *args, **kwargs)

# Patch torch.stack for this process
torch.stack = _stack_patch

# Now run the actual script with its CLI
from scripts.compute_norm_stats import main
import tyro
tyro.cli(main)
