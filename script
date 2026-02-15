import torch
import numpy as np
from KBNN2 import KBNN
import math
from typing import List, Union, Optional
import glob
import os

def _ease(alpha: torch.Tensor, kind: str = "cosine", sharpness: float = 10.0) -> torch.Tensor:
    """
    alpha: values in [0, 1]
    kind: "linear", "cosine", or "sigmoid"
    """
    if kind == "linear":
        return alpha
    elif kind == "cosine":
        # Smooth at endpoints (zero slope at 0 and 1)
        return 0.5 - 0.5 * torch.cos(math.pi * alpha)
    elif kind == "sigmoid":
        # Smooth-ish ramp; sharpness controls how gradual the middle transition is
        # Maps alpha in [0,1] to ~[0,1]
        x = (alpha - 0.5) * sharpness
        y = torch.sigmoid(x)
        # Normalize so exactly 0 -> 0 and 1 -> 1 (approximately; improves endpoints)
        y0 = torch.sigmoid(torch.tensor(-0.5 * sharpness, device=alpha.device, dtype=alpha.dtype))
        y1 = torch.sigmoid(torch.tensor( 0.5 * sharpness, device=alpha.device, dtype=alpha.dtype))
        return (y - y0) / (y1 - y0)
    else:
        raise ValueError(f"Unknown ease kind: {kind}")

def apply_yaw(actions: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    Apply yaw rotation about +z to translation (x,y) and axis-angle rotation vector (rx,ry).
    Gripper unchanged.

    actions: (..., 7)
    theta:   (...) or scalar tensor; must be broadcastable to actions[..., 0]
    """
    if actions.shape[-1] != 7:
        raise ValueError(f"Expected last dim = 7, got {actions.shape}")

    # Ensure theta has a trailing dim for broadcasting with (..., 1)
    # but we’ll just rely on PyTorch broadcasting in arithmetic.
    c = torch.cos(theta)
    s = torch.sin(theta)

    out = actions.clone()

    # position
    x = actions[..., 0]
    y = actions[..., 1]
    out[..., 0] = c * x - s * y
    out[..., 1] = s * x + c * y
    # z unchanged
    out[..., 2] = actions[..., 2]

    # axis-angle rotation vector
    rx = actions[..., 3]
    ry = actions[..., 4]
    out[..., 3] = c * rx - s * ry
    out[..., 4] = s * rx + c * ry
    # rz unchanged
    out[..., 5] = actions[..., 5]

    # gripper unchanged
    out[..., 6] = actions[..., 6]

    return out

def gradual_yaw(actions: torch.Tensor,
                final_yaw_rad: float = math.pi / 2,
                schedule: str = "cosine",
                sharpness: float = 10.0) -> torch.Tensor:
    """
    Gradually increase yaw from 0 to final_yaw_rad across the N rows of actions.

    actions: (N, 7)
    schedule: "linear", "cosine", or "sigmoid"
    """
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f"Expected shape (N, 7), got {actions.shape}")

    N = actions.shape[0]
    if N == 1:
        theta = torch.zeros(1, device=actions.device, dtype=actions.dtype)
    else:
        alpha = torch.linspace(0.0, 1.0, N, device=actions.device, dtype=actions.dtype)
        alpha = _ease(alpha, kind=schedule, sharpness=sharpness)
        theta = alpha * final_yaw_rad

    return apply_yaw(actions, theta)


def stepwise_yaw(actions: torch.Tensor,
                 num_steps: int,
                 total_yaw_rad: float = math.pi / 2,
                 strict_equal_bins: bool = True) -> torch.Tensor:
    """
    Stepwise (piecewise-constant) yaw drift across the N samples.

    - Splits N samples into `num_steps` equal-sized contiguous bins.
    - Uses yaw angles equally spaced from 0 to total_yaw_rad across those steps:
        theta_i = i * total_yaw_rad/(num_steps-1),  i=0..num_steps-1
      (If num_steps == 1, we apply theta = total_yaw_rad to all samples.)

    actions: (N, 7)
    returns: (N, 7)
    """
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f"Expected actions shape (N, 7), got {actions.shape}")
    if num_steps <= 0:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")

    N = actions.shape[0]
    if strict_equal_bins and (N % num_steps != 0):
        raise ValueError(
            f"N={N} is not divisible by num_steps={num_steps}, "
            f"but strict_equal_bins=True requires equal samples per step."
        )

    step_size = N // num_steps  # valid when divisible; if not strict, last remainder ignored below

    device = actions.device
    dtype = actions.dtype

    if num_steps == 1:
        # One regime: apply the final yaw everywhere.
        theta_vals = torch.tensor([total_yaw_rad], device=device, dtype=dtype)
    else:
        theta_vals = torch.linspace(
            0.0, float(total_yaw_rad),
            steps=num_steps,
            device=device, dtype=dtype
        )

    theta = theta_vals.repeat_interleave(step_size)  # length = step_size*num_steps

    if theta.numel() < N:
        # Only possible when strict_equal_bins=False and N not divisible.
        # We keep the last angle for the remainder so every sample is covered.
        remainder = N - theta.numel()
        theta = torch.cat([theta, theta_vals[-1].expand(remainder)], dim=0)

    return apply_yaw(actions, theta)

def yaw_R7_neg_thetas(
    thetas: Union[torch.Tensor, List[float]],
) -> List[torch.Tensor]:
    """
    Given a 1D list/tensor of angles theta_i (radians),
    return a Python list where the i-th element is the 7x7 matrix R(-theta_i).

    The 7x7 matrix is block-diagonal:
      - top-left 3x3: yaw rotation about z by (-theta_i) for (x,y,z)
      - middle 3x3:   same yaw rotation for (rx,ry,rz) axis-angle vector components
      - bottom-right: 1 for gripper
    """
    theta = torch.as_tensor(thetas)
    if theta.ndim != 1:
        theta = theta.reshape(-1)

    angle = -theta  # we want R(-theta_i)
    c = torch.cos(angle)
    s = torch.sin(angle)

    K = theta.numel()
    R = torch.zeros((K, 7, 7), dtype=theta.dtype, device=theta.device)

    # Fill translation block (indices 0..2)
    R[:, 0, 0] = c
    R[:, 0, 1] = -s
    R[:, 1, 0] = s
    R[:, 1, 1] = c
    R[:, 2, 2] = 1.0

    # Fill rotation-vector block (indices 3..5)
    R[:, 3, 3] = c
    R[:, 3, 4] = -s
    R[:, 4, 3] = s
    R[:, 4, 4] = c
    R[:, 5, 5] = 1.0

    # Gripper unchanged
    R[:, 6, 6] = 1.0

    return [R[i] for i in range(K)]


def make_stepwise_R7_neg_list(
    num_steps: int,
    total_yaw_rad: float = math.pi / 2,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> List[torch.Tensor]:
    """
    Convenience helper for your stepwise drift setting:
    theta_i are equally spaced from 0 to total_yaw_rad across num_steps.

    Returns a list of length num_steps:
      [R(-theta_0), R(-theta_1), ..., R(-theta_{num_steps-1})]
    """
    if num_steps <= 0:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")

    if num_steps == 1:
        thetas = torch.tensor([total_yaw_rad], device=device, dtype=dtype)
    else:
        thetas = torch.linspace(0.0, float(total_yaw_rad), steps=num_steps, device=device, dtype=dtype)

    return yaw_R7_neg_thetas(thetas)

num_datasets = 100
# data = np.load("./env_06/ep_0000.npz")
# actions = torch.tensor(data["actions"])
# print(actions.shape)
# # print(actions[0,1,:])
data_dir = "./env_06/"   # change this
all_actions = []

for i in range(num_datasets):
    filename = os.path.join(data_dir, f"ep_{i:04d}.npz")
    
    if not os.path.exists(filename):
        continue  # skip missing files
    
    data = np.load(filename)
    actions = data["actions"]              # shape (K, 10, 7)
    
    actions_slice = torch.from_numpy(actions[:, 0, :])  # shape (K, 7)
    all_actions.append(actions_slice)

# Concatenate into shape (L, 7)
actions_tensor = torch.cat(all_actions, dim=0)

print(actions_tensor.shape)



N = 300
# ref_outputs = actions_tensor[:N, :]
# generate ref_outputs by setting each element to be uniformly random in [-1, 1]
ref_outputs = torch.rand((N, 7)) * 2 - 1



print(ref_outputs[1,:])
ref_outputs_original = ref_outputs.clone()


num_steps = 5
modified_outputs = stepwise_yaw(ref_outputs, num_steps=num_steps, total_yaw_rad=math.pi/2, strict_equal_bins=False)
R7 = make_stepwise_R7_neg_list(num_steps=num_steps, total_yaw_rad=math.pi/2, device=ref_outputs.device, dtype=ref_outputs.dtype)

target = modified_outputs - ref_outputs

# normalize ref_outputs and target to have zero mean and unit variance
ref_mean = torch.mean(ref_outputs, dim=0, keepdim=True)
ref_std = torch.std(ref_outputs, dim=0, keepdim=True)
ref_outputs = (ref_outputs - ref_mean) / ref_std
# only normalize entries with non-zero std
target_mean = torch.mean(target, dim=0, keepdim=True)
target_std = torch.std(target, dim=0, keepdim=True)
nonzero_std_mask = target_std > 1e-6
target[:, nonzero_std_mask[0]] = (target[:, nonzero_std_mask[0]] - target_mean[:, nonzero_std_mask[0]]) / target_std[:, nonzero_std_mask[0]]

# implement a standard neural network to learn the mapping from data_points to modified_outputs - ref_outputs, without kbnn
class SimpleNN(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 50)
        self.fc2 = torch.nn.Linear(50, 40)
        self.fc3 = torch.nn.Linear(40, output_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

nn_model = SimpleNN(7, 7)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(nn_model.parameters(), lr=1e-3)
n_epochs = 4000

for sample in range(N):
    if sample > N/num_steps and sample % 5 == 0:
        train_size = int(N/num_steps)
        train_data = ref_outputs[sample + 5 - train_size:sample]
        train_outputs = target[sample + 5 - train_size:sample]
        for epoch in range(n_epochs):
            nn_model.train()
            optimizer.zero_grad()
            outputs = nn_model(train_data)
            loss = criterion(outputs, train_outputs)
            loss.backward()
            optimizer.step()
            if (epoch) % 800 == 0:
                print(f"Sample {sample}, Epoch [{epoch + 1}/{n_epochs}], Loss: {loss.item():.6g}")
                nn_model.eval()
                with torch.no_grad():
                    val_data = ref_outputs[sample:sample + 5]
                    val_outputs = target[sample:sample + 5]
                    val_preds = nn_model(val_data)
                    val_loss = criterion(val_preds, val_outputs)
                    print(val_outputs[0])
                    print(val_preds[0])
                    print(f"Validation Loss: {val_loss.item():.6g}")

kbnn_input_dim = 7

network_geometry = [kbnn_input_dim, 7, 7, 7]



kbnn = KBNN(network_geometry, init_scale=0.1, init_cov=0.01, zero_init=True, dtype=torch.float32, device='cpu')
running_loss = []

for i in range(N):
    x = ref_outputs[i]
    y = target[i]
    kbnn_input = x
    cache = kbnn.forward(kbnn_input)
    # detect if there is any nan in cache["mus"][-1]
    if torch.isnan(cache["mus"][-1]).any():
        kbnn.mws = last_mws
        kbnn.sws = last_sws
    else:
        last_mws = kbnn.mws
        last_sws = kbnn.sws
    # loss = (cache["mus"][-1] - y).T @ (cache["mus"][-1] - y)
    loss = torch.nn.functional.mse_loss(cache["mus"][-1], y)
    print(cache["mus"][-1])
    print(y)
    print(f"Loss at step {i}: {loss.item()}")
    # compute running average loss for the last 20 steps
    running_loss.append(loss.item())
    if len(running_loss) > 10:
        running_loss.pop(0)
    print(f"Running average loss: {np.mean(running_loss)}")
    kbnn.backward(y)
    # print the mse loss as if the output is all zeros
    zero_loss = torch.nn.functional.mse_loss(y, torch.zeros_like(y))
    print(f"Zero output loss at step {i}: {zero_loss.item()}")
    # for every i, save kbnn.mws and kbnn.sws to a file named "kbnn_weights_step_{i}.pt"
    torch.save({"mws": kbnn.mws, "sws": kbnn.sws, "ref_mean": ref_mean, "ref_std": ref_std, "target_mean": target_mean, "target_std": target_std, "rotation": R7[int(i/(N/num_steps))]}, f"kbnn_weights_step_{i}.pt")
    for j in range(20):
        cache = kbnn.forward(kbnn_input)
        kbnn.backward(y)