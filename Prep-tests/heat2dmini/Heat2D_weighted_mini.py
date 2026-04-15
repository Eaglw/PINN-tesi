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
parser = argparse.ArgumentParser(description='Mini PINN Experiment for Autoresearch')
parser.add_argument('--arch', type=str, default='100,100,100,100', help='Hidden layers as comma-separated list')
parser.add_argument('--act', type=str, default='Tanh', choices=['Tanh', 'SiLU', 'GELU'], help='Activation function')
parser.add_argument('--epochs', type=int, default=2000, help='Adam epochs')
parser.add_argument('--lbfgs_iter', type=int, default=500, help='L-BFGS max iterations')
parser.add_argument('--run_type', type=str, default='PINN_PurePhys', choices=['PINN_PurePhys', 'PINN_DataPhys'], help='Run type')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--weight_mode', type=str, default='dynamic', choices=['dynamic', 'static'], help='Weighting mode')
parser.add_argument('--update_weights_every', type=int, default=100, help='Frequency of weight updates')
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--n_collocation', type=int, default=40, help='Number of collocation points per dimension')
parser.add_argument('--bc_weight', type=float, default=1.0, help='Initial weight for BC loss')
parser.add_argument('--sampling', type=str, default='grid', choices=['grid', 'sobol', 'halton'], help='Sampling strategy')
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

# --- 2. MODEL DEFINITION ---
class FCN(nn.Module):
    def __init__(self, layers, activation_fn=nn.Tanh):
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
    return nn.Tanh

# --- 3. PROBLEM DEFINITION ---
def soluzione_analitica(x, y, Lx=1.0, Ly=1.0, Nx=50):
    x_flat = x.reshape(-1, 1); y_flat = y.reshape(-1, 1)
    n_vals = torch.arange(1, Nx + 1, 2, device=x.device, dtype=x.dtype)
    pi = torch.tensor(torch.pi, device=x.device, dtype=x.dtype)
    lam = n_vals * pi / Ly; An = 4.0 / (n_vals * pi)
    lx = lam * x_flat
    terms = An * (torch.sinh(lx) / torch.sinh(lam * Lx)) * torch.sin(lam * y_flat)
    T_flat = terms.sum(dim=-1, keepdim=True)
    return T_flat.reshape(x.shape)

Lx, Ly = 1.0, 1.0
Nx_dom, Ny_dom = 50, 50
x_grid = torch.linspace(0, Lx, Nx_dom, device=device)
y_grid = torch.linspace(0, Ly, Ny_dom, device=device)
X, Y = torch.meshgrid(x_grid, y_grid, indexing='xy')
T_grid = soluzione_analitica(X, Y, Lx, Ly, Nx=50)
xy_grid_flat = torch.stack([X.flatten(), Y.flatten()], dim=1)

# --- 4. DATA PREPARATION ---
margin = 2e-2
Nx_grid_master, Ny_grid_master = args.n_collocation, args.n_collocation
num_collocation = Nx_grid_master * Ny_grid_master

from func.sampling_utils import generate_sobol_points, generate_halton_points

if args.sampling == 'sobol':
    xy_master_grid = generate_sobol_points(num_collocation, Lx, Ly, margin=margin, device=device)
elif args.sampling == 'halton':
    xy_master_grid = generate_halton_points(num_collocation, Lx, Ly, margin=margin, device=device)
else:
    xy_master_grid = generate_grid_points(Nx_grid_master, Ny_grid_master, Lx, Ly, margin=margin, device=device)

num_b_side = 50
pts_bc = torch.linspace(0.02, 0.98, num_b_side, device=device).reshape(-1, 1)
bc_left = torch.cat([torch.zeros(num_b_side, 1, device=device), pts_bc], dim=1)
bc_right = torch.cat([torch.ones(num_b_side, 1, device=device), pts_bc], dim=1)
bc_bottom = torch.cat([pts_bc, torch.zeros(num_b_side, 1, device=device)], dim=1)
bc_top = torch.cat([pts_bc, torch.ones(num_b_side, 1, device=device)], dim=1)
xy_master_boundary = torch.cat([bc_left, bc_right, bc_bottom, bc_top], dim=0)
T_master_boundary = torch.cat([torch.zeros(num_b_side, 1, device=device), torch.ones(num_b_side, 1, device=device), torch.zeros(num_b_side, 1, device=device), torch.zeros(num_b_side, 1, device=device)], dim=0)

if args.run_type == 'PINN_DataPhys':
    num_subset = 1000
    generator_fn = lambda n: generate_internal_points(n, Lx, Ly, margin=1e-5, device=device)
    xy_pinn_data = filter_and_refill(xy_master_grid, generator_fn, num_subset, d_min=1e-4)
    T_pinn_data = soluzione_analitica(xy_pinn_data[:, 0:1], xy_pinn_data[:, 1:2], Lx, Ly, Nx=50)
    pinn_data_internal = (xy_pinn_data, T_pinn_data)
else:
    pinn_data_internal = (torch.empty(0, 2, device=device), torch.empty(0, 1, device=device))

pinn_data_boundary = (xy_master_boundary, T_master_boundary)

# --- 5. TRAINING ---
layers = [2] + [int(x) for x in args.arch.split(',')] + [1]
model = FCN(layers=layers, activation_fn=get_act_fn(args.act)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
heat_physics = HeatEquation2D()

exp_name = f"MINI_{args.run_type}_{args.arch}_{args.act}"
base_dir = "heat2dmini/mini_experiments"
os.makedirs(base_dir, exist_ok=True)
exp_dir = os.path.join(base_dir, exp_name)

# Use the existing one but with smaller epochs and no interactive plots
import Heat2D.src.Heat2D_PINN as Heat2D_PINN

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
    lr_strategy='plateau',
    loss_weights={'bc': args.bc_weight, 'physics': 1.0, 'data': 1.0 if args.run_type == 'PINN_DataPhys' else 0.0},
    dynamic_weighting=(args.weight_mode == 'dynamic'),
    update_weights_every=args.update_weights_every,
    warmup_epochs=0,
    max_total_lbfgs=args.lbfgs_iter
)
duration = time.time() - start_time

l2_err, max_err = compute_metrics(model, xy_grid_flat, T_grid)

# Log to a specific mini_results.csv
results_csv = "heat2dmini/mini_results.csv"
log_data = {
    'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    'Max_Relative_Error_Peak': max_err,
    'Architecture': str(layers),
    'Activation_Func': args.act,
    'Epochs': args.epochs,
    'Run_Type': args.run_type,
    'Optimizer': f"Adam+LBFGS({args.lbfgs_iter})",
    'Learning_Rate': f"{args.lr} (plateau)",
    'L2_Relative_Error': l2_err,
    'Seed': args.seed,
    'Duration_Sec': duration
}

# Simple CSV update
file_exists = os.path.exists(results_csv)
with open(results_csv, 'a') as f:
    import csv
    writer = csv.DictWriter(f, fieldnames=log_data.keys())
    if not file_exists: writer.writeheader()
    writer.writerow(log_data)

print(f"\nExperiment Finished!")
print(f"L2 Relative Error: {l2_err:.6f}")
print(f"Duration: {duration:.2f} seconds")
