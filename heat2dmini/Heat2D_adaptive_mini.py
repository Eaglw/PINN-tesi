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
from func.sampling_utils import generate_internal_points, generate_grid_points, filter_and_refill, generate_sobol_points, generate_halton_points
from Heat2D.src.Heat2D_PINN import train_modelPINN
from Heat2D.src.physics import HeatEquation2D

# --- 1. SETUP & ARGUMENTS ---
parser = argparse.ArgumentParser(description='Adaptive Activation PINN Experiment')
parser.add_argument('--arch', type=str, default='120,100,80,60,40,20', help='Hidden layers')
parser.add_argument('--act', type=str, default='GELU', help='Base activation')
parser.add_argument('--epochs', type=int, default=2000, help='Adam epochs')
parser.add_argument('--lbfgs_iter', type=int, default=500, help='L-BFGS iterations')
parser.add_argument('--bc_weight', type=float, default=50.0, help='Initial BC weight')
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--n_collocation', type=int, default=40, help='Collocation points')
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

# --- 2. ADAPTIVE MODEL DEFINITION ---
class AdaptiveActivation(nn.Module):
    def __init__(self, activation_fn, n_layers):
        super().__init__()
        self.activation = activation_fn()
        # Learnable parameter 'a' for each layer: f(x) = activation(a * x)
        self.a = nn.Parameter(torch.ones(n_layers))

    def forward(self, x, layer_idx):
        return self.activation(self.a[layer_idx] * x)

class AdaptiveFCN(nn.Module):
    def __init__(self, layers, activation_fn=nn.GELU):
        super().__init__()
        self.fcs = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.fcs.append(nn.Linear(layers[i], layers[i+1]))
        
        # We need n-1 activations for n-1 hidden transitions
        self.adaptive_act = AdaptiveActivation(activation_fn, len(layers) - 1)

    def forward(self, x):
        for i, layer in enumerate(self.fcs[:-1]):
            x = self.adaptive_act(layer(x), i)
        return self.fcs[-1](x)

def get_act_fn(name):
    if name == 'Tanh': return nn.Tanh
    if name == 'SiLU': return nn.SiLU
    if name == 'GELU': return nn.GELU
    return nn.GELU

# --- 3. PROBLEM DEFINITION ---
def soluzione_analitica(x, y, Lx=1.0, Ly=1.0, Nx=50):
    # Map [-1, 1] back to [0, Lx] for analytical formula
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

Lx_val, Ly_val = 1.0, 1.0
# Evaluation grid in [-1, 1]
x_grid = torch.linspace(-1, 1, 50, device=device)
y_grid = torch.linspace(-1, 1, 50, device=device)
X, Y = torch.meshgrid(x_grid, y_grid, indexing='xy')
T_grid = soluzione_analitica(X, Y, Lx_val, Ly_val, Nx=50)
xy_grid_flat = torch.stack([X.flatten(), Y.flatten()], dim=1)

# --- 4. DATA PREPARATION ---
# Use [-1, 1] domain for all points
margin = 0.02
num_internal = args.n_collocation * args.n_collocation
xy_master_grid = generate_sobol_points(num_internal, 2.0, 2.0, device=device) - 1.0
# Filter with margin in [-1, 1]
mask = (xy_master_grid[:,0] > -1+margin) & (xy_master_grid[:,0] < 1-margin) & \
       (xy_master_grid[:,1] > -1+margin) & (xy_master_grid[:,1] < 1-margin)
xy_master_grid = xy_master_grid[mask]

num_b_side = 100
pts_bc = torch.linspace(-0.99, 0.99, num_b_side, device=device).reshape(-1, 1)
bc_left = torch.cat([-torch.ones(num_b_side, 1, device=device), pts_bc], dim=1)
bc_right = torch.cat([torch.ones(num_b_side, 1, device=device), pts_bc], dim=1)
bc_bottom = torch.cat([pts_bc, -torch.ones(num_b_side, 1, device=device)], dim=1)
bc_top = torch.cat([pts_bc, torch.ones(num_b_side, 1, device=device)], dim=1)
xy_master_boundary = torch.cat([bc_left, bc_right, bc_bottom, bc_top], dim=0)

# Analytic solution for boundary mapping
T_master_boundary = soluzione_analitica(xy_master_boundary[:,0], xy_master_boundary[:,1], Lx_val, Ly_val).reshape(-1, 1)

pinn_data_internal = (torch.empty(0, 2, device=device), torch.empty(0, 1, device=device))
pinn_data_boundary = (xy_master_boundary, T_master_boundary)

# --- 5. TRAINING ---
layers = [2] + [int(x) for x in args.arch.split(',')] + [1]
model = AdaptiveFCN(layers=layers, activation_fn=get_act_fn(args.act)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
heat_physics = HeatEquation2D()

exp_name = f"ADAPTIVE_{args.arch}_{args.act}"
base_dir = "heat2dmini/mini_experiments"
os.makedirs(base_dir, exist_ok=True)
exp_dir = os.path.join(base_dir, exp_name)

start_time = time.time()
history = train_modelPINN(
    model=model,
    optimizer=optimizer,
    data_internal=pinn_data_internal,
    data_boundary=pinn_data_boundary,
    validation_grid=(xy_grid_flat, T_grid, X, Y),
    physics_problem=heat_physics,
    epochs=args.epochs,
    plots_dir=os.path.join(exp_dir, 'plots'),
    final_dir=exp_dir,
    show_plots_interactively=False,
    collocation_points=xy_master_grid,
    lr_strategy='cosine',
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
    'Architecture': f"ADAPTIVE_{layers}",
    'Activation_Func': args.act,
    'Epochs': args.epochs,
    'Run_Type': 'PINN_AdaptiveAct',
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

print(f"\nAdaptive Experiment Finished!")
print(f"L2 Relative Error: {l2_err:.6f}")
print(f"Adaptive parameters: {model.adaptive_act.a.detach().cpu().numpy()}")
