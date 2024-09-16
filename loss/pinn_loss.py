import torch
import torch.nn as nn


def loss_function(model, x, t, u, alpha):

    # Enable automatic differentiation
    x.requires_grad = True
    t.requires_grad = True

    u_pred = model(x, t)
    u_t = torch.autograd.grad(u_pred, t, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
    u_x = torch.autograd.grad(u_pred, x, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

    # Heat equation residual
    f = u_t - alpha * u_xx

    # Data loss
    mse_data = nn.MSELoss()(u_pred, u)

    # Physics loss
    mse_phys = nn.MSELoss()(f, torch.zeros_like(f))

    loss = mse_data + mse_phys
    return loss
