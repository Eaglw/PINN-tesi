import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())
from Heat2D.src.physics import HeatEquation2D

def test_laplace_autograd():
    print("Testing Laplace Autograd with T(x, y) = x^2 + y^2")
    print("Expected Residual: d2T/dx2 + d2T/dy2 = 2 + 2 = 4")
    
    # Mock model that implements T(x, y) = x^2 + y^2
    class QuadraticModel(nn.Module):
        def forward(self, xy):
            x = xy[:, 0:1]
            y = xy[:, 1:2]
            return x**2 + y**2
    
    model = QuadraticModel()
    physics = HeatEquation2D()
    
    # Test points
    xy = torch.tensor([[0.5, 0.5], [0.1, 0.9], [0.8, 0.2]], dtype=torch.float64, requires_grad=True)
    
    # We want to check the raw residual before the mean square
    # Modifying the residual check slightly for this test
    T = model(xy)
    grads = torch.autograd.grad(T, xy, torch.ones_like(T), create_graph=True)[0]
    dT_dx = grads[:, 0]
    dT_dy = grads[:, 1]
    
    print(f"dT/dx: {dT_dx.detach().numpy()} (Expected: 2*x = [1.0, 0.2, 1.6])")
    print(f"dT/dy: {dT_dy.detach().numpy()} (Expected: 2*y = [1.0, 1.8, 0.4])")
    
    grads2_x = torch.autograd.grad(dT_dx, xy, torch.ones_like(dT_dx), create_graph=True)[0]
    d2T_dx2 = grads2_x[:, 0]
    
    grads2_y = torch.autograd.grad(dT_dy, xy, torch.ones_like(dT_dy), create_graph=True)[0]
    d2T_dy2 = grads2_y[:, 1]
    
    res = d2T_dx2 + d2T_dy2
    print(f"Residuals: {res.detach().numpy()} (Expected: [4.0, 4.0, 4.0])")
    
    mean_sq_res = torch.mean((res - 0)**2) # The physics class returns mean(res**2) for Laplace
    # But for our test function x^2 + y^2, the Laplace is 4, so residual for Laplace is 4.
    # The HeatEquation2D.residual returns mean((d2T/dx2 + d2T/dy2)**2)
    
    actual_phys_res = physics.residual(model, xy)
    print(f"Physics Class Residual Output: {actual_phys_res.item()} (Expected: 16.0)")
    
    assert torch.allclose(res, torch.tensor([4.0, 4.0, 4.0], dtype=torch.float64))
    assert torch.allclose(actual_phys_res, torch.tensor(16.0, dtype=torch.float64))
    print("SUCCESS: Autograd logic for Laplace residual is correct.")

if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    test_laplace_autograd()
