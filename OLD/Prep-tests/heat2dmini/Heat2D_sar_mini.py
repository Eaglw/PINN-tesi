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
from func.logging_utils import compute_metrics, update_results_csv
from func.sampling_utils import generate_internal_points, generate_grid_points, filter_and_refill
from Heat2D.src.Heat2D_PINN import train_modelPINN
from Heat2D.src.physics import HeatEquation2D

# --- 1. SETUP & ARGUMENTS ---
parser = argparse.ArgumentParser(description='SAR PINN Experiment')
parser.add_argument('--arch', type=str, default='120,100,80,60,40,20', help='Hidden layers')
parser.add_argument('--act', type=str, default='GELU', help='Activation')
parser.add_argument('--epochs', type=int, default=2000, help='Adam epochs')
parser.add_argument('--lbfgs_iter', type=int, default=500, help='L-BFGS iterations')
parser.add_argument('--bc_weight', type=float, default=50.0, help='Initial BC weight')
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--sar_every', type=int, default=500, help='Add points every N epochs')
parser.add_argument('--sar_points', type=int, default=100, help='Number of points to add per step')
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

# --- 2. MODEL DEFINITION ---
class FCN(nn.Module):
    def __init__(self, layers, activation_fn=nn.GELU):
        super().__init__()
        self.activation = activation_fn()
        self.fcs = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.fcs.append(nn.Linear(layers[i], layers[i+1]))
    def forward(self, x):
        for layer in self.fcs[:-1]:
            x = self.activation(layer(x))
        return self.fcs[-1](x)

def get_act_fn(name):
    if name == 'Tanh': return nn.Tanh
    if name == 'SiLU': return nn.SiLU
    if name == 'GELU': return nn.GELU
    return nn.GELU

# --- 3. PROBLEM DEFINITION ---
Lx, Ly = 1.0, 1.0
Nx_dom, Ny_dom = 50, 50
x_grid = torch.linspace(0, Lx, Nx_dom, device=device)
y_grid = torch.linspace(0, Ly, Ny_dom, device=device)
X, Y = torch.meshgrid(x_grid, y_grid, indexing='xy')
xy_grid_flat = torch.stack([X.flatten(), Y.flatten()], dim=1)

def soluzione_analitica(x, y, Lx=1.0, Ly=1.0, Nx=50):
    x_flat = x.reshape(-1, 1); y_flat = y.reshape(-1, 1)
    n_vals = torch.arange(1, Nx + 1, 2, device=x.device, dtype=x.dtype)
    pi = torch.tensor(torch.pi, device=x.device, dtype=x.dtype)
    lam = n_vals * pi / Ly; An = 4.0 / (n_vals * pi)
    lx = lam * x_flat
    terms = An * (torch.sinh(lx) / torch.sinh(lam * Lx)) * torch.sin(lam * y_flat)
    T_flat = terms.sum(dim=-1, keepdim=True)
    return T_flat.reshape(x.shape)

T_grid = soluzione_analitica(X, Y, Lx, Ly, Nx=50)

# --- 4. SAR LOGIC ---
def get_residual(model, xy):
    xy_p = xy.clone().requires_grad_(True)
    T = model(xy_p)
    grads = torch.autograd.grad(T, xy_p, torch.ones_like(T), create_graph=True)[0]
    dT_dx, dT_dy = grads[:, 0], grads[:, 1]
    grads2_x = torch.autograd.grad(dT_dx, xy_p, torch.ones_like(dT_dx), create_graph=True)[0]
    d2T_dx2 = grads2_x[:, 0]
    grads2_y = torch.autograd.grad(dT_dy, xy_p, torch.ones_like(dT_dy), create_graph=True)[0]
    d2T_dy2 = grads2_y[:, 1]
    res = d2T_dx2 + d2T_dy2
    return (res**2).detach()

# --- 5. TRAINING LOOP WITH SAR ---
layers = [2] + [int(x) for x in args.arch.split(',')] + [1]
model = FCN(layers=layers, activation_fn=get_act_fn(args.act)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
heat_physics = HeatEquation2D()

margin = 2e-2
xy_physics = generate_grid_points(40, 40, Lx, Ly, margin=margin, device=device)

# BC Points
num_b_side = 50
pts_bc = torch.linspace(0.02, 0.98, num_b_side, device=device).reshape(-1, 1)
bc_left = torch.cat([torch.zeros(num_b_side, 1, device=device), pts_bc], dim=1)
bc_right = torch.cat([torch.ones(num_b_side, 1, device=device), pts_bc], dim=1)
bc_bottom = torch.cat([pts_bc, torch.zeros(num_b_side, 1, device=device)], dim=1)
bc_top = torch.cat([pts_bc, torch.ones(num_b_side, 1, device=device)], dim=1)
xy_bc = torch.cat([bc_left, bc_right, bc_bottom, bc_top], dim=0)
T_bc = torch.cat([torch.zeros(num_b_side, 1, device=device), torch.ones(num_b_side, 1, device=device), torch.zeros(num_b_side, 1, device=device), torch.zeros(num_b_side, 1, device=device)], dim=0)

start_time = time.time()
print(f"Starting Training with SAR (every {args.sar_every} epochs)...")

# We run train_modelPINN in segments or just implement a custom loop
# To keep using train_modelPINN features, we can do it in steps
current_epochs = 0
while current_epochs < args.epochs:
    step_epochs = min(args.sar_every, args.epochs - current_epochs)
    
    from Heat2D.src.Heat2D_PINN import train_modelPINN
    train_modelPINN(
        model=model, optimizer=optimizer,
        data_internal=(torch.empty(0,2,device=device), torch.empty(0,1,device=device)),
        data_boundary=(xy_bc, T_bc),
        validation_grid=(xy_grid_flat, T_grid, X, Y),
        physics_problem=heat_physics,
        epochs=step_epochs,
        collocation_points=xy_physics,
        loss_weights={'bc': args.bc_weight, 'physics': 1.0, 'data': 0.0},
        dynamic_weighting=True,
        update_weights_every=100,
        warmup_epochs=0,
        max_total_lbfgs=0 # No L-BFGS until the end
    )
    current_epochs += step_epochs
    
    if current_epochs < args.epochs:
        print(f"Applying SAR at epoch {current_epochs}...")
        # Sample many random points to find high residual areas
        n_candidates = 10000
        xy_candidates = generate_internal_points(n_candidates, Lx, Ly, margin=margin, device=device)
        res = get_residual(model, xy_candidates)
        
        # Pick top sar_points
        _, idx = torch.topk(res, args.sar_points)
        new_points = xy_candidates[idx]
        xy_physics = torch.cat([xy_physics, new_points], dim=0)
        print(f"New collocation count: {xy_physics.shape[0]}")

# Final L-BFGS
print("Final L-BFGS refinement...")
train_modelPINN(
    model=model, optimizer=optimizer,
    data_internal=(torch.empty(0,2,device=device), torch.empty(0,1,device=device)),
    data_boundary=(xy_bc, T_bc),
    validation_grid=(xy_grid_flat, T_grid, X, Y),
    physics_problem=heat_physics,
    epochs=0,
    collocation_points=xy_physics,
    loss_weights={'bc': args.bc_weight, 'physics': 1.0, 'data': 0.0},
    dynamic_weighting=True,
    update_weights_every=100,
    warmup_epochs=0,
    max_total_lbfgs=args.lbfgs_iter
)

duration = time.time() - start_time
l2_err, max_err = compute_metrics(model, xy_grid_flat, T_grid)

# Log results
results_csv = "heat2dmini/mini_results.csv"
log_data = {
    'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    'Max_Relative_Error_Peak': max_err,
    'Architecture': str(layers),
    'Activation_Func': f"{args.act}_SAR",
    'Epochs': args.epochs,
    'Run_Type': 'PINN_SAR',
    'Optimizer': f"Adam+LBFGS({args.lbfgs_iter})",
    'Learning_Rate': "1e-3 (plateau)",
    'L2_Relative_Error': l2_err,
    'Seed': args.seed,
    'Duration_Sec': duration
}

file_exists = os.path.exists(results_csv)
with open(results_csv, 'a') as f:
    import csv
    writer = csv.DictWriter(f, fieldnames=log_data.keys())
    if not file_exists: writer.writeheader()
    writer.writerow(log_data)

print(f"\nSAR Experiment Finished!")
print(f"L2 Relative Error: {l2_err:.6f}")
