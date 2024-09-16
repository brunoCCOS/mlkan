from train import train_PINN
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ParamPPPPPeters
alpha = 0.01  # thermal diffusivity
num_points = 1000

# Generate random points in the domain
x = np.random.uniform(0, 1, num_points)
t = np.random.uniform(0, 1, num_points)
X, T = np.meshgrid(x, t)
X_flat = X.flatten()
T_flat = T.flatten()

# Exact solution (for testing purposes), e.g., u(x,t) = sin(pi x) * exp(-pi^2 alpha t)
u_exact = np.sin(np.pi * X_flat) * np.exp(-np.pi**2 * alpha * T_flat)

# Add noise
noise = 0.01 * np.random.randn(u_exact.size)
u_noisy = u_exact + noise

# Convert to tensors
x_tensor = torch.tensor(X_flat, dtype=torch.float32).reshape(-1, 1)
t_tensor = torch.tensor(T_flat, dtype=torch.float32).reshape(-1, 1)
u_tensor = torch.tensor(u_noisy, dtype=torch.float32).reshape(-1, 1)

train_PINN(x_tensor,t_tensor,u_tensor,alpha = alpha)
