import torch
import numpy as np
from KBNN2 import KBNN
import math

def yaw_plus_90_mat(actions: torch.Tensor) -> torch.Tensor:
    """
    Same as yaw_plus_90, but using an explicit rotation matrix.
    """
    if actions.shape[-1] != 7:
        raise ValueError(f"Expected last dim = 7, got {actions.shape}")

    R = torch.tensor(
        [[0.0, -1.0, 0.0],
         [1.0,  0.0, 0.0],
         [0.0,  0.0, 1.0]],
        dtype=actions.dtype,
        device=actions.device,
    )

    out = actions.clone()
    out[..., 0:3] = actions[..., 0:3] @ R.T
    out[..., 3:6] = actions[..., 3:6] @ R.T
    # out[..., 6] unchanged
    return out


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

# generate a random 10240 by 7 matrix in torch
ref_weight = torch.randn(10240, 7)

N = 100
# generate N data points, each with 10240 features
data_points = torch.randn(N, 10240)

# generate reference outputs using the reference weight
ref_outputs = data_points @ ref_weight  # shape (N, 7)

# normalize ref_outputs to zero mean and unit variance
# ref_outputs = (ref_outputs - ref_outputs.mean(dim=0)) / ref_outputs.std(dim=0)

num_steps = 5
# modified_outputs = yaw_plus_90_mat(ref_outputs)
# modified_outputs = gradual_yaw(ref_outputs, final_yaw_rad=math.pi/500, schedule="cosine", sharpness=1.0)
modified_outputs = stepwise_yaw(ref_outputs, num_steps=5, total_yaw_rad=math.pi/50, strict_equal_bins=False)

# to_input_matrix = torch.randn(kbnn_input_dim, 10240)
to_input_matrix = ref_weight.T

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

# # split data into training and validation sets
# train_size = int(0.9 * N)
# train_data = to_input_matrix @ data_points[:train_size].T
# train_data = train_data.T
# train_outputs = modified_outputs[:train_size] - ref_outputs[:train_size]
# for epoch in range(n_epochs):
#     nn_model.train()
#     optimizer.zero_grad()
#     outputs = nn_model(train_data)
#     loss = criterion(outputs, train_outputs)
#     loss.backward()
#     optimizer.step()
#     if (epoch) % 10 == 0:
#         print(f"Epoch [{epoch + 1}/{n_epochs}], Loss: {loss.item():.6g}")
#         nn_model.eval()
#         with torch.no_grad():
#             val_data = to_input_matrix @ data_points[train_size:].T
#             val_data = val_data.T
#             val_outputs = modified_outputs[train_size:] - ref_outputs[train_size:]
#             val_preds = nn_model(val_data)
#             val_loss = criterion(val_preds, val_outputs)
#             print(val_outputs[0])
#             print(val_preds[0])
#             print(f"Validation Loss: {val_loss.item():.6g}")

for sample in range(N):
    if sample > N/num_steps and sample % 5 == 0:
        train_size = int(N/num_steps)
        train_data = to_input_matrix @ data_points[sample + 5 - train_size:sample].T
        train_data = train_data.T
        train_outputs = modified_outputs[sample + 5 - train_size:sample] - ref_outputs[sample + 5 - train_size:sample]
        for epoch in range(n_epochs):
            nn_model.train()
            optimizer.zero_grad()
            outputs = nn_model(train_data)
            loss = criterion(outputs, train_outputs)
            loss.backward()
            optimizer.step()
            if (epoch) % 400 == 0:
                print(f"Sample {sample}, Epoch [{epoch + 1}/{n_epochs}], Loss: {loss.item():.6g}")
                nn_model.eval()
                with torch.no_grad():
                    val_data = to_input_matrix @ data_points[sample:sample + 5].T
                    val_data = val_data.T
                    val_outputs = modified_outputs[sample:sample + 5] - ref_outputs[sample:sample + 5]
                    val_preds = nn_model(val_data)
                    val_loss = criterion(val_preds, val_outputs)
                    print(val_outputs[0])
                    print(val_preds[0])
                    print(f"Validation Loss: {val_loss.item():.6g}")

kbnn_input_dim = 7

network_geometry = [kbnn_input_dim, 50, 40, 7]



kbnn = KBNN(network_geometry, init_scale=0.1, init_cov=0.01, zero_init=True, dtype=torch.float32, device='cpu')
running_loss = []

for i in range(N):
    if i > N/num_steps:
        x = data_points[i]
        y = modified_outputs[i] - ref_outputs[i]
        kbnn_input = to_input_matrix @ x
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
        if len(running_loss) > 20:
            running_loss.pop(0)
        print(f"Running average loss: {np.mean(running_loss)}")
        kbnn.backward(y)
        # print the mse loss as if the output is all zeros
        zero_loss = torch.nn.functional.mse_loss(y, torch.zeros_like(y))
        print(f"Zero output loss at step {i}: {zero_loss.item()}")
        
        for j in range(200):
            cache = kbnn.forward(kbnn_input)
            kbnn.backward(y)