"""
Experiment: Heat2D_final_verification.py
Description: Final Verification of the Model
"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from tqdm import tqdm

# Import shared functions
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from Heat2D.src.Heat2D_PINN import train_modelPINN
from Heat2D.src.physics import HeatEquation2D
from func.graphic_func import plot_error_map_comparison, plot_loss_comparison

# Device and Precision
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)
print(f"Using device: {device}")

# --- Configuration ---
epochs = 30000 # Full training
Lx, Ly = 1.0, 1.0
Nx_fourier = 50
layers_config = [2, 50, 50, 50, 50, 1]
results_dir = os.path.join(os.path.dirname(__file__), 'Results', 'final_verification')
os.makedirs(results_dir, exist_ok=True)
plots_dir = os.path.join(results_dir, 'plots')
os.makedirs(plots_dir, exist_ok=True)

# --- Definitions ---
def soluzione_analitica(x, y, Lx=1.0, Ly=1.0, Nx=50):
    T = torch.zeros_like(x)
    const_pi = torch.tensor(np.pi, device=x.device)
    for n in range(1, Nx + 1, 2):
        lambda_n = n * const_pi / Ly
        An = 4 / (n * const_pi)
        term = An * (torch.sinh(lambda_n * x) / torch.sinh(lambda_n * Lx)) * torch.sin(lambda_n * y)
        T += term
    return T

class FCN(nn.Module):
    def __init__(self, layers, activation_fn=nn.Tanh):
        super().__init__()
        self.activation = activation_fn()
        self.fcs = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.fcs.append(nn.Linear(layers[i], layers[i+1]))
    def forward(self, x):
        for i, layer in enumerate(self.fcs):
            x = layer(x)
            if i < len(self.fcs) - 1:
                x = self.activation(x)
        return x

# --- Data Preparation ---
Nx_dom, Ny_dom = 50, 50
x_grid = torch.linspace(0, Lx, Nx_dom, device=device)
y_grid = torch.linspace(0, Ly, Ny_dom, device=device)
X, Y = torch.meshgrid(x_grid, y_grid, indexing='xy')
xy_grid_flat = torch.stack([X.flatten(), Y.flatten()], dim=1)
T_grid = soluzione_analitica(X, Y, Lx, Ly, Nx=Nx_fourier)
validation_grid_tuple = (xy_grid_flat, T_grid, X, Y)

torch.manual_seed(123)

num_data_internal = 1000
num_data_boundary = 50

# Internal
x_int = torch.rand(num_data_internal, 1, device=device) * Lx
y_int = torch.rand(num_data_internal, 1, device=device) * Ly
xy_internal = torch.cat([x_int, y_int], dim=1)
T_internal = soluzione_analitica(x_int, y_int, Lx, Ly, Nx=Nx_fourier)

# Boundary
x_b_left = torch.zeros(num_data_boundary, 1, device=device)
y_b_left = torch.rand(num_data_boundary, 1, device=device) * Ly
x_b_right = torch.ones(num_data_boundary, 1, device=device) * Lx
y_b_right = torch.rand(num_data_boundary, 1, device=device) * Ly
x_b_bottom = torch.rand(num_data_boundary, 1, device=device) * Lx
y_b_bottom = torch.zeros(num_data_boundary, 1, device=device)
x_b_top = torch.rand(num_data_boundary, 1, device=device) * Lx
y_b_top = torch.ones(num_data_boundary, 1, device=device) * Ly

x_b_all = torch.cat([x_b_left, x_b_right, x_b_bottom, x_b_top], dim=0)
y_b_all = torch.cat([y_b_left, y_b_right, y_b_bottom, y_b_top], dim=0)
xy_boundary = torch.cat([x_b_all, y_b_all], dim=1)
T_boundary = soluzione_analitica(x_b_all, y_b_all, Lx, Ly, Nx=Nx_fourier)

data_internal = (xy_internal, T_internal)
data_boundary = (xy_boundary, T_boundary)

# --- Experiments ---

weights = {'data': 1.0, 'bc': 1.0, 'physics': 0.05}
heat_physics = HeatEquation2D()

# 1. Standard (Baseline) Run - 30k epochs
print(f"\n--- Running Baseline PINN (50x50 Collocation, {epochs} epochs) ---")
model_baseline = FCN(layers=layers_config).to(device)
optimizer_baseline = torch.optim.Adam(model_baseline.parameters(), lr=1e-3)

train_modelPINN(
    model=model_baseline,
    optimizer=optimizer_baseline,
    data_internal=data_internal,
    data_boundary=data_boundary,
    validation_grid=validation_grid_tuple,
    physics_problem=heat_physics,
    epochs=epochs,
    plots_dir=os.path.join(plots_dir, 'baseline'),
    final_dir=os.path.join(results_dir, 'baseline'),
    show_plots_interactively=False,
    loss_weights=weights,
    n_collocation=50
)

# 2. Optimized Run - 30k epochs
print(f"\n--- Running Optimized PINN (100x100 Collocation, {epochs} epochs) ---")
model_optim = FCN(layers=layers_config).to(device)
optimizer_optim = torch.optim.Adam(model_optim.parameters(), lr=1e-3)

train_modelPINN(
    model=model_optim,
    optimizer=optimizer_optim,
    data_internal=data_internal,
    data_boundary=data_boundary,
    validation_grid=validation_grid_tuple,
    physics_problem=heat_physics,
    epochs=epochs,
    plots_dir=os.path.join(plots_dir, 'optimized'),
    final_dir=os.path.join(results_dir, 'optimized'),
    show_plots_interactively=False,
    loss_weights=weights,
    n_collocation=100
)

# --- Final Comparisons ---
print("\n--- Generating Final Comparisons ---")

# 1. Error Maps
model_baseline.eval()
model_optim.eval()

with torch.no_grad():
    pred_baseline = model_baseline(xy_grid_flat).reshape(Nx_dom, Ny_dom)
    pred_optim = model_optim(xy_grid_flat).reshape(Nx_dom, Ny_dom)

plot_error_map_comparison(
    X, Y, T_grid,
    [pred_baseline, pred_optim],
    ['Baseline (50x50)', 'Optimized (100x100)'] ,
    save_path=os.path.join(results_dir, 'Final_Comparison_ErrorMap.png')
)

print(f"Final Verification complete. Results saved in {results_dir}")
