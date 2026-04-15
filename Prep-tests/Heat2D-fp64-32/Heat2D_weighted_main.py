import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time
from datetime import datetime
import shutil

# Import utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from func.logging_utils import compute_metrics, update_results_csv
from func.sampling_utils import generate_internal_points, generate_grid_points, check_overlaps
from src.Heat2D_hybrid import train_hybrid_logic
from src.physics import HeatEquation2D

def soluzione_analitica(x, y, Lx=1.0, Ly=1.0, Nx=50):
    original_shape = x.shape
    x_flat = x.reshape(-1, 1)
    y_flat = y.reshape(-1, 1)
    n_vals = torch.arange(1, Nx + 1, 2, device=x.device, dtype=x.dtype)
    pi = torch.tensor(torch.pi, device=x.device, dtype=x.dtype)
    lam = n_vals * pi / Ly
    An = 4.0 / (n_vals * pi)
    lx = lam * x_flat
    terms = An * (torch.sinh(lx) / torch.sinh(lam * Lx)) * torch.sin(lam * y_flat)
    T_flat = terms.sum(dim=-1, keepdim=True)
    return T_flat.reshape(original_shape)

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

def format_layers_name(layers):
    if len(layers) > 3:
        hidden = layers[1:-1]
        if all(x == hidden[0] for x in hidden):
            return f"{layers[0]}_{hidden[0]}x{len(hidden)}_{layers[-1]}"
    return "_".join(map(str, layers))

# --- CONFIGURAZIONE ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

goal = [0, 1, 2, 3] 
layers_options = [[2, 50, 50, 50, 50, 1], [2, 80, 80, 80, 80, 80, 80, 1], [2, 100, 100, 100, 100, 100, 100, 100, 100, 1] ]
epochs = 40000
activation_options = [nn.Tanh, nn.SiLU, nn.GELU]
lr_strategy = 'plateau'
weight_mode = 'dynamic'

base_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiments_weighted_hybrid')
results_csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.csv')

# --- DATA GENERATION (FP64 Master) ---
Lx, Ly = 1.0, 1.0
Nx_dom, Ny_dom = 50, 50
x_grid = torch.linspace(0, Lx, Nx_dom, device=device, dtype=torch.float64)
y_grid = torch.linspace(0, Ly, Ny_dom, device=device, dtype=torch.float64)
X, Y = torch.meshgrid(x_grid, y_grid, indexing='xy')
T_grid = soluzione_analitica(X, Y, Lx, Ly, Nx=50)
xy_grid_flat = torch.stack([X.flatten(), Y.flatten()], dim=1)

torch.manual_seed(123)
xy_master_grid = generate_grid_points(40, 40, Lx, Ly, margin=2e-2, device=device)
xy_master_random = generate_internal_points(1600, Lx, Ly, margin=2e-2, device=device)

num_b_side = 50
pts_bc = torch.linspace(0.02, 0.98, num_b_side, device=device, dtype=torch.float64).reshape(-1, 1)
bc_left = torch.cat([torch.zeros_like(pts_bc), pts_bc], dim=1)
bc_right = torch.cat([torch.ones_like(pts_bc)*Lx, pts_bc], dim=1)
bc_bottom = torch.cat([pts_bc, torch.zeros_like(pts_bc)], dim=1)
bc_top = torch.cat([pts_bc, torch.ones_like(pts_bc)*Ly], dim=1)
xy_master_boundary = torch.cat([bc_left, bc_right, bc_bottom, bc_top], dim=0)
T_master_boundary = torch.cat([torch.zeros(num_b_side, 1, device=device), torch.ones(num_b_side, 1, device=device), 
                                torch.zeros(num_b_side, 1, device=device), torch.zeros(num_b_side, 1, device=device)], dim=0)

T_master_grid = soluzione_analitica(xy_master_grid[:, 0:1], xy_master_grid[:, 1:2], Lx, Ly)
T_master_random = soluzione_analitica(xy_master_random[:, 0:1], xy_master_random[:, 1:2], Lx, Ly)

validation_grid_tuple = (xy_grid_flat, T_grid, X, Y)
heat_physics = HeatEquation2D()

print(f"Starting WEIGHTED HYBRID Grid Search (Adam@32 -> L-BFGS@64)")

for layers_config in layers_options:
    for act_fn in activation_options:
        config_name = f"W_Hybrid_L{format_layers_name(layers_config)}_E{epochs}_{act_fn.__name__}"
        config_dir = os.path.join(base_output_dir, config_name)
        os.makedirs(config_dir, exist_ok=True)
        
        print(f"\n========== CONFIGURATION: {config_name} ==========")

        # --- 0. NN Random ---
        if 0 in goal:
            model = FCN(layers=layers_config, activation_fn=act_fn).to(device)
            train_data = (torch.cat([xy_master_random, xy_master_boundary], 0), torch.cat([T_master_random, T_master_boundary], 0))
            start = time.time()
            train_hybrid_logic(model, train_data, (None, None), validation_grid_tuple, epochs=epochs, 
                               collocation_points=None, lr_strategy=lr_strategy, final_dir=config_dir,
                               loss_weights={'data': 1.0, 'bc': 0.0, 'physics': 0.0}, case_name="0_NN_Random", 
                               dynamic_weighting=False)
            l2, _ = compute_metrics(model, xy_grid_flat, T_grid)
            update_results_csv(results_csv_path, {'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Architecture': str(layers_config), 'Activation_Func': act_fn.__name__, 'Run_Type': 'NN_Rand_W_Hybrid', 'Time': time.time()-start, 'L2_Error': l2})

        # --- 1. NN Grid ---
        if 1 in goal:
            model = FCN(layers=layers_config, activation_fn=act_fn).to(device)
            train_data = (torch.cat([xy_master_grid, xy_master_boundary], 0), torch.cat([T_master_grid, T_master_boundary], 0))
            start = time.time()
            train_hybrid_logic(model, train_data, (None, None), validation_grid_tuple, epochs=epochs, 
                               collocation_points=None, lr_strategy=lr_strategy, final_dir=config_dir,
                               loss_weights={'data': 1.0, 'bc': 0.0, 'physics': 0.0}, case_name="1_NN_Grid",
                               dynamic_weighting=False)
            l2, _ = compute_metrics(model, xy_grid_flat, T_grid)
            update_results_csv(results_csv_path, {'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Architecture': str(layers_config), 'Activation_Func': act_fn.__name__, 'Run_Type': 'NN_Grid_W_Hybrid', 'Time': time.time()-start, 'L2_Error': l2})

        # --- 2. PINN Data+Phys ---
        if 2 in goal:
            model = FCN(layers=layers_config, activation_fn=act_fn).to(device)
            xy_pinn_data = xy_master_random[:1000]
            T_pinn_data = soluzione_analitica(xy_pinn_data[:, 0:1], xy_pinn_data[:, 1:2], Lx, Ly)
            start = time.time()
            train_hybrid_logic(model, (xy_pinn_data, T_pinn_data), (xy_master_boundary, T_master_boundary), validation_grid_tuple, 
                               epochs=epochs, collocation_points=xy_master_grid, physics_problem=heat_physics, lr_strategy=lr_strategy, final_dir=config_dir,
                               loss_weights={'data': 1.0, 'bc': 1.0, 'physics': 1.0}, case_name="2_PINN_DataPhys",
                               dynamic_weighting=True, update_weights_every=500)
            l2, _ = compute_metrics(model, xy_grid_flat, T_grid)
            update_results_csv(results_csv_path, {'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Architecture': str(layers_config), 'Activation_Func': act_fn.__name__, 'Run_Type': 'PINN_DP_W_Hybrid', 'Time': time.time()-start, 'L2_Error': l2})

        # --- 3. PINN Pure Phys ---
        if 3 in goal:
            model = FCN(layers=layers_config, activation_fn=act_fn).to(device)
            start = time.time()
            train_hybrid_logic(model, (xy_master_random[:1000], T_master_random[:1000]), (xy_master_boundary, T_master_boundary), validation_grid_tuple, 
                               epochs=epochs, collocation_points=xy_master_grid, physics_problem=heat_physics, lr_strategy=lr_strategy, final_dir=config_dir,
                               loss_weights={'data': 0.0, 'bc': 1.0, 'physics': 1.0}, case_name="3_PINN_PurePhys",
                               dynamic_weighting=True, update_weights_every=500)
            l2, _ = compute_metrics(model, xy_grid_flat, T_grid)
            update_results_csv(results_csv_path, {'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Architecture': str(layers_config), 'Activation_Func': act_fn.__name__, 'Run_Type': 'PINN_PP_W_Hybrid', 'Time': time.time()-start, 'L2_Error': l2})

print("\nWeighted Hybrid Grid Search completed.")
