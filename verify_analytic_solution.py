import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())
from Heat2D.Heat2D_main import soluzione_analitica
from Heat2D.src.physics import HeatEquation2D

def verify_analytic_residual():
    print("Verifying Laplace residual of the analytical solution...")
    
    Lx, Ly = 1.0, 1.0
    Nx_fourier = 50
    Nx_dom, Ny_dom = 100, 100 # Higher density for verification
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    x_grid = torch.linspace(0.1, 0.9, Nx_dom, device=device) # Stay away from boundaries to avoid singularities
    y_grid = torch.linspace(0.1, 0.9, Ny_dom, device=device)
    X, Y = torch.meshgrid(x_grid, y_grid, indexing='xy')
    xy = torch.stack([X.flatten(), Y.flatten()], dim=1).requires_grad_(True)
    
    # We can't pass the function directly to the physics class because it expects a model.
    # We'll compute the residual manually following the same logic.
    
    T = soluzione_analitica(xy[:, 0:1], xy[:, 1:2], Lx, Ly, Nx=Nx_fourier)
    
    # 1st derivatives
    grads = torch.autograd.grad(T, xy, torch.ones_like(T), create_graph=True)[0]
    dT_dx = grads[:, 0]
    dT_dy = grads[:, 1]
    
    # 2nd derivatives
    grads2_x = torch.autograd.grad(dT_dx, xy, torch.ones_like(dT_dx), create_graph=True)[0]
    d2T_dx2 = grads2_x[:, 0]
    
    grads2_y = torch.autograd.grad(dT_dy, xy, torch.ones_like(dT_dy), create_graph=True)[0]
    d2T_dy2 = grads2_y[:, 1]
    
    res = d2T_dx2 + d2T_dy2
    
    mean_res = torch.mean(res).item()
    std_res = torch.std(res).item()
    mean_sq_res = torch.mean(res**2).item()
    
    print(f"Fourier Terms: {Nx_fourier}")
    print(f"Mean Residual: {mean_res:.2e}")
    print(f"Std Residual: {std_res:.2e}")
    print(f"Mean Squared Residual: {mean_sq_res:.2e}")
    
    # Check at a single point to see the values
    idx = Nx_dom * Ny_dom // 2
    print(f"Sample at {xy[idx].detach().cpu().numpy()}:")
    print(f"  T: {T[idx].item():.6f}")
    print(f"  d2T/dx2: {d2T_dx2[idx].item():.6e}")
    print(f"  d2T/dy2: {d2T_dy2[idx].item():.6e}")
    print(f"  sum: {res[idx].item():.6e}")

    if mean_sq_res > 1e-4:
        print("\nWARNING: Analytical solution has a significant PDE residual.")
        print("Increasing Fourier terms to 200...")
        T_200 = soluzione_analitica(xy[:, 0:1], xy[:, 1:2], Lx, Ly, Nx=200)
        grads_200 = torch.autograd.grad(T_200, xy, torch.ones_like(T_200), create_graph=True)[0]
        grads2_x_200 = torch.autograd.grad(grads_200[:, 0], xy, torch.ones_like(grads_200[:, 0]), create_graph=True)[0]
        grads2_y_200 = torch.autograd.grad(grads_200[:, 1], xy, torch.ones_like(grads_200[:, 1]), create_graph=True)[0]
        res_200 = grads2_x_200[:, 0] + grads2_y_200[:, 1]
        print(f"Mean Squared Residual (Nx=200): {torch.mean(res_200**2).item():.2e}")

if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    verify_analytic_residual()
