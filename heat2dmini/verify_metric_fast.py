import torch
import torch.nn as nn
import numpy as np
import os
import sys
import argparse
from datetime import datetime
import time

# Add root to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from func.logging_utils import compute_metrics
from func.sampling_utils import generate_sobol_points
from Heat2D.src.Heat2D_PINN import train_modelPINN
from Heat2D.src.physics import HeatEquation2D

# Mock args
class Args:
    arch = '120,100,80,60,40,20'
    act = 'GELU'
    epochs = 2000
    lbfgs_iter = 500
    bc_weight = 50.0
    seed = 123
    n_collocation = 40

args = Args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

# Re-define model classes for fast verification (same as in Heat2D_adaptive_mini.py)
class AdaptiveActivation(nn.Module):
    def __init__(self, activation_fn, n_layers):
        super().__init__()
        self.activation = activation_fn()
        self.a = nn.Parameter(torch.full((n_layers,), 1.1))
    def forward(self, x, layer_idx):
        return self.activation(self.a[layer_idx] * x)

class AdaptiveFCN(nn.Module):
    def __init__(self, layers, activation_fn=nn.GELU):
        super().__init__()
        self.fcs = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.fcs.append(nn.Linear(layers[i], layers[i+1]))
        self.adaptive_act = AdaptiveActivation(activation_fn, len(layers) - 1)
    def forward(self, x):
        for i, layer in enumerate(self.fcs[:-1]):
            x = self.adaptive_act(layer(x), i)
        return self.fcs[-1](x)

def soluzione_analitica(x, y, Lx=1.0, Ly=1.0, Nx=50):
    x_orig = (x + 1) * Lx / 2.0
    y_orig = (y + 1) * Ly / 2.0
    x_flat = x_orig.reshape(-1, 1); y_flat = y_orig.reshape(-1, 1)
    n_vals = torch.arange(1, Nx + 1, 2, device=x.device, dtype=x.dtype)
    pi = torch.tensor(torch.pi, device=x.device, dtype=x.dtype)
    lam = n_vals * pi / Ly; An = 4.0 / (n_vals * pi)
    lx = lam * x_flat
    terms = An * (torch.sinh(lx) / torch.sinh(lam * Lx)) * torch.sin(lam * y_flat)
    T_flat = terms.sum(dim=-1, keepdim=True)
    return T_flat.reshape(x.shape)

x_grid = torch.linspace(-1, 1, 50, device=device)
y_grid = torch.linspace(-1, 1, 50, device=device)
X, Y = torch.meshgrid(x_grid, y_grid, indexing='xy')
T_grid = soluzione_analitica(X, Y, 1.0, 1.0, Nx=50)
xy_grid_flat = torch.stack([X.flatten(), Y.flatten()], dim=1)

num_internal = args.n_collocation * args.n_collocation
xy_master_grid = generate_sobol_points(num_internal, 2.0, 2.0, device=device) - 1.0
mask = (xy_master_grid[:,0] > -0.98) & (xy_master_grid[:,0] < 0.98) & \
       (xy_master_grid[:,1] > -0.98) & (xy_master_grid[:,1] < 0.98)
xy_master_grid = xy_master_grid[mask]

num_b_side = 100
pts_bc = torch.linspace(-0.99, 0.99, num_b_side, device=device).reshape(-1, 1)
bc_left = torch.cat([-torch.ones(num_b_side, 1, device=device), pts_bc], dim=1)
bc_right = torch.cat([torch.ones(num_b_side, 1, device=device), pts_bc], dim=1)
bc_bottom = torch.cat([pts_bc, -torch.ones(num_b_side, 1, device=device)], dim=1)
bc_top = torch.cat([pts_bc, torch.ones(num_b_side, 1, device=device)], dim=1)
xy_master_boundary = torch.cat([bc_left, bc_right, bc_bottom, bc_top], dim=0)
T_master_boundary = soluzione_analitica(xy_master_boundary[:,0], xy_master_boundary[:,1]).reshape(-1, 1)

layers = [2, 120, 100, 80, 60, 40, 20, 1]
model = AdaptiveFCN(layers=layers, activation_fn=nn.GELU).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
heat_physics = HeatEquation2D()

history = train_modelPINN(
    model=model,
    optimizer=optimizer,
    data_internal=(torch.empty(0, 2, device=device), torch.empty(0, 1, device=device)),
    data_boundary=(xy_master_boundary, T_master_boundary),
    validation_grid=(xy_grid_flat, T_grid, X, Y),
    physics_problem=heat_physics,
    epochs=args.epochs,
    plots_dir='temp_plots',
    final_dir='temp_results',
    show_plots_interactively=False,
    collocation_points=xy_master_grid,
    lr_strategy='plateau',
    loss_weights={'bc': args.bc_weight, 'physics': 1.0, 'data': 0.0},
    dynamic_weighting=True,
    update_weights_every=100,
    warmup_epochs=0,
    max_total_lbfgs=args.lbfgs_iter
)

l2_err, _ = compute_metrics(model, xy_grid_flat, T_grid)
print(l2_err)
