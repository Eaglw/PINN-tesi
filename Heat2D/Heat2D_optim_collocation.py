import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from tqdm import tqdm

# Import shared functions
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Heat2D.Heat2D_PINN import train_modelPINN
from Heat2D.physics import HeatEquation2D
from func.graphic_func import plot_error_map_comparison

# Device and Precision
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)
print(f"Using device: {device}")

# --- Configuration ---
epochs = 5000
Lx, Ly = 1.0, 1.0
Nx_fourier = 50
layers_config = [2, 50, 50, 50, 50, 1]
results_dir = os.path.join(os.path.dirname(__file__), 'Results', 'optim_collocation')
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

weights = {'data': 1.0, 'bc': 1.0, 'physics': 0.05} # Baseline weights
heat_physics = HeatEquation2D()

# Experiment 1: Standard Resolution (50x50)
print("\n--- Running Experiment 1: Standard Collocation (50x50) ---")
model_std = FCN(layers=layers_config).to(device)
optimizer_std = torch.optim.Adam(model_std.parameters(), lr=1e-3)

train_modelPINN(
    model=model_std,
    optimizer=optimizer_std,
    data_internal=data_internal,
    data_boundary=data_boundary,
    validation_grid=validation_grid_tuple,
    physics_problem=heat_physics,
    epochs=epochs,
    plots_dir=os.path.join(plots_dir, 'std_res'),
    final_dir=os.path.join(results_dir, 'std_res'),
    show_plots_interactively=False,
    loss_weights=weights,
    n_collocation=50
)

# Experiment 2: High Resolution (100x100)
print("\n--- Running Experiment 2: High Collocation (100x100) ---")
model_high = FCN(layers=layers_config).to(device)
optimizer_high = torch.optim.Adam(model_high.parameters(), lr=1e-3)

train_modelPINN(
    model=model_high,
    optimizer=optimizer_high,
    data_internal=data_internal,
    data_boundary=data_boundary,
    validation_grid=validation_grid_tuple,
    physics_problem=heat_physics,
    epochs=epochs,
    plots_dir=os.path.join(plots_dir, 'high_res'),
    final_dir=os.path.join(results_dir, 'high_res'),
    show_plots_interactively=False,
    loss_weights=weights,
    n_collocation=100
)

# --- Comparison ---
print("\n--- Generating Comparison Maps ---")
model_std.eval()
model_high.eval()

with torch.no_grad():
    pred_std = model_std(xy_grid_flat).reshape(Nx_dom, Ny_dom)
    pred_high = model_high(xy_grid_flat).reshape(Nx_dom, Ny_dom)

plot_error_map_comparison(
    X, Y, T_grid,
    [pred_std, pred_high],
    ['Standard (50x50)', 'High Res (100x100)'],
    save_path=os.path.join(results_dir, 'Comparison_ErrorMap_Collocation.png')
)

print(f"Collocation Experiment complete. Results in {results_dir}")
