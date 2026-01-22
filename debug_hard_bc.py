import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())
from Heat2D.src.Heat2D_PINN_hardBC import HardBCWrapper

def debug_hard_bc():
    print("Debugging HardBCWrapper output...")
    
    Lx, Ly = 1.0, 1.0
    
    # Mock model
    class ZeroModel(nn.Module):
        def forward(self, x):
            return torch.zeros((x.shape[0], 1), dtype=torch.float64)
            
    model = ZeroModel()
    wrapper = HardBCWrapper(model, Lx, Ly, n_terms=10)
    
    # Test points
    # Point at center (0.5, 0.5)
    xy = torch.tensor([[0.5, 0.5], [1.0, 0.5]], dtype=torch.float64)
    
    T_out = wrapper(xy)
    
    print("Input points:")
    print(xy.numpy())
    print("Output T:")
    print(T_out.detach().numpy())
    
    # Check T_boundary manually at (1.0, 0.5)
    # T(1, 0.5) should be ~1
    print(f"T at (1.0, 0.5) should be near 1.0. Actual: {T_out[1].item():.6f}")
    
    if T_out[1].item() < 0.1:
        print("FAIL: HardBCWrapper is producing near-zero values where it shouldn't.")
    else:
        print("SUCCESS: HardBCWrapper is producing non-zero values.")

if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    debug_hard_bc()