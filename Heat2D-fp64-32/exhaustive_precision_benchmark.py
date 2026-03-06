import torch
import torch.nn as nn
import numpy as np
import os
import time
import pandas as pd
from tqdm import tqdm
from src.precision_utils import PrecisionConfig
from src.Heat2D_PINN import train_modelPINN_precision
from src.physics import HeatEquation2D
from visualize_precision_benchmark import visualize_results

def set_seed(seed=123):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
    def __init__(self, layers, activation_fn=nn.GELU):
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

def run_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Aumentiamo significativamente il carico per saturare la GPU
    num_internal = 10000 
    layers = [2, 256, 256, 256, 256, 1]
    epochs = 2000 # Riduciamo le epoche perché ogni epoca ora è molto più pesante
    
    print(f"Benchmark starting on {device}")
    print(f"Config: Network {layers}, Points: {num_internal}, Epochs: {epochs}")
    
    Lx, Ly = 1.0, 1.0
    Nx_fourier = 50
    Nx_dom, Ny_dom = 100, 100 # Griglia di validazione più densa
    
    # Gold Standard (FP64)
    torch.set_default_dtype(torch.float64)
    set_seed(123)
    x_grid = torch.linspace(0, Lx, Nx_dom, device=device)
    y_grid = torch.linspace(0, Ly, Ny_dom, device=device)
    X, Y = torch.meshgrid(x_grid, y_grid, indexing='xy')
    T_grid = soluzione_analitica(X, Y, Lx, Ly, Nx=Nx_fourier)
    xy_grid_flat = torch.stack([X.flatten(), Y.flatten()], dim=1)
    
    xy_int = torch.rand((num_internal, 2), device=device)
    T_int = soluzione_analitica(xy_int[:, 0:1], xy_int[:, 1:2], Lx, Ly, Nx=Nx_fourier)
    
    num_bc = 200
    pts_bc = torch.linspace(0, Ly, num_bc, device=device).reshape(-1, 1)
    bc_left = torch.cat([torch.zeros_like(pts_bc), pts_bc], dim=1)
    bc_right = torch.cat([torch.ones_like(pts_bc)*Lx, pts_bc], dim=1)
    bc_bottom = torch.cat([pts_bc, torch.zeros_like(pts_bc)], dim=1)
    bc_top = torch.cat([pts_bc, torch.ones_like(pts_bc)*Ly], dim=1)
    xy_bc = torch.cat([bc_left, bc_right, bc_bottom, bc_top], dim=0)
    T_bc = torch.cat([torch.zeros(num_bc, 1, device=device), torch.ones(num_bc, 1, device=device), 
                      torch.zeros(num_bc, 1, device=device), torch.zeros(num_bc, 1, device=device)], dim=0)
    
    results = []
    
    print("\n--- Running GOLD STANDARD (Full FP64) ---")
    gold_config = PrecisionConfig(nn_opt=torch.float64, data=torch.float64, physics=torch.float64, bc=torch.float64)
    model = FCN(layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    start_time = time.time()
    train_modelPINN_precision(
        model, optimizer, (xy_int, T_int), (xy_bc, T_bc), (xy_grid_flat, T_grid, X, Y),
        epochs=epochs, physics_problem=HeatEquation2D(),
        plots_dir=None, final_dir=None,
        show_plots_interactively=False,
        precision_config=gold_config,
        collocation_points=xy_int
    )
    gold_time = time.time() - start_time
    
    model.eval()
    model.to(torch.float64)
    with torch.no_grad():
        T_pred_gold = model(xy_grid_flat.to(torch.float64)).reshape(Nx_dom, Ny_dom)
    gold_mae = torch.mean(torch.abs(T_pred_gold - T_grid)).item()
    print(f"Gold MAE: {gold_mae:.2e}, Time: {gold_time:.2f}s")
    
    results.append({
        'mask': 15, 'config': str(gold_config), 'MAE_Analytic': gold_mae, 'MAE_Gold': 0.0, 'Time': gold_time, 'Speedup': 1.0
    })
    
    parts = ['nn_opt', 'data', 'physics', 'bc']
    for mask in tqdm(range(15), desc="Benchmarking"):
        config = PrecisionConfig.from_bitmask(mask, parts)
        torch.set_default_dtype(config.nn_opt)
        set_seed(123)
        model = FCN(layers).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        start_time = time.time()
        try:
            train_modelPINN_precision(
                model, optimizer, (xy_int, T_int), (xy_bc, T_bc), (xy_grid_flat, T_grid, X, Y),
                epochs=epochs, physics_problem=HeatEquation2D(),
                plots_dir=None, final_dir=None,
                show_plots_interactively=False,
                precision_config=config,
                collocation_points=xy_int
            )
            run_time = time.time() - start_time
            model.eval()
            model.to(torch.float64)
            with torch.no_grad():
                T_pred = model(xy_grid_flat.to(torch.float64)).reshape(Nx_dom, Ny_dom)
            mae_analytic = torch.mean(torch.abs(T_pred - T_grid)).item()
            mae_gold = torch.mean(torch.abs(T_pred - T_pred_gold)).item()
            results.append({
                'mask': mask, 'config': str(config), 'MAE_Analytic': mae_analytic, 'MAE_Gold': mae_gold, 'Time': run_time, 'Speedup': gold_time / run_time if run_time > 0 else 0
            })
        except Exception as e:
            print(f"Error in mask {mask}: {e}")
            results.append({
                'mask': mask, 'config': str(config), 'MAE_Analytic': np.nan, 'MAE_Gold': np.nan, 'Time': np.nan, 'Speedup': np.nan, 'Error': str(e)
            })

    df = pd.DataFrame(results)
    for i, part in enumerate(parts):
        df[part] = df['mask'].apply(lambda m: "FP64" if (m >> i) & 1 else "FP32")
    csv_path = 'Heat2D-fp64-32/precision_benchmark_results.csv'
    df.to_csv(csv_path, index=False)
    
    print("\nBenchmark complete. Generating visualization...")
    visualize_results(csv_path=csv_path, output_dir='Heat2D-fp64-32/benchmark_plots')

if __name__ == "__main__":
    run_benchmark()
