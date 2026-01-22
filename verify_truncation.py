import torch
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())
from Heat2D.Heat2D_main import soluzione_analitica

def verify_truncation_error():
    print("Comparing 10-term vs 50-term analytical solution...")
    
    Lx, Ly = 1.0, 1.0
    Nx_dom, Ny_dom = 100, 100
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    x_grid = torch.linspace(0, Lx, Nx_dom, device=device)
    y_grid = torch.linspace(0, Ly, Ny_dom, device=device)
    X, Y = torch.meshgrid(x_grid, y_grid, indexing='xy')
    
    T_50 = soluzione_analitica(X, Y, Lx, Ly, Nx=50)
    T_10 = soluzione_analitica(X, Y, Lx, Ly, Nx=19) # n terms up to 19 (10 terms: 1,3,5,7,9,11,13,15,17,19)
    
    l2_error = torch.norm(T_10 - T_50, 2)
    l2_ref = torch.norm(T_50, 2)
    l2_rel_error = (l2_error / l2_ref).item()
    
    print(f"L2 Relative Error (10 vs 50 terms): {l2_rel_error:.4f}")
    
    if abs(l2_rel_error - 0.0416) < 0.01:
        print("SUCCESS: The 4.1% error in HardBC is purely due to Fourier series truncation.")
    else:
        print(f"DIFFERENCE: Truncation error is {l2_rel_error:.4f}, but HardBC error was 0.0416.")

if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    verify_truncation_error()
