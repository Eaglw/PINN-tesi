import torch
import numpy as np

class InversePoissonPhysics:
    """
    Handles physics for the Heat2D Inverse Poisson problem.
    Equation: -k * (T_xx + T_yy) = Q(x,y)
    Domain: [0,1]^2
    BCs: T = 0 on all boundaries.
    Q(x,y) = Q0 * sin(pi*x) * sin(pi*y)
    """
    def __init__(self, k_param, Q0=1.0):
        self.k_param = k_param
        self.Q0 = Q0

    def residual(self, model, x_phys):
        if not x_phys.requires_grad:
            x_phys.requires_grad_(True)
            
        T = model(x_phys)
        
        grads = torch.autograd.grad(T, x_phys, torch.ones_like(T), create_graph=True)[0]
        dT_dx = grads[:, 0]
        dT_dy = grads[:, 1]
        
        grads2_x = torch.autograd.grad(dT_dx, x_phys, torch.ones_like(dT_dx), create_graph=True, allow_unused=True)[0]
        d2T_dx2 = grads2_x[:, 0]
        
        grads2_y = torch.autograd.grad(dT_dy, x_phys, torch.ones_like(dT_dy), create_graph=True, allow_unused=True)[0]
        d2T_dy2 = grads2_y[:, 1]
        
        laplacian = d2T_dx2 + d2T_dy2
        
        # Source term Q(x,y) = Q0 * sin(pi*x) * sin(pi*y)
        pi = np.pi
        Q = self.Q0 * torch.sin(pi * x_phys[:, 0]) * torch.sin(pi * x_phys[:, 1])
        
        # Residual = k * laplacian + Q = 0  => -k * laplacian = Q
        res = self.k_param * laplacian + Q
        return torch.mean(res**2)

def compute_analytical_poisson(x, y, k_true=1.0, Q0=1.0):
    """
    Analytical solution for:
    -k * Laplacian(T) = Q0 * sin(pi*x) * sin(pi*y)
    with T=0 on boundaries.
    Solution: T(x,y) = (Q0 / (k * 2 * pi^2)) * sin(pi*x) * sin(pi*y)
    """
    pi = np.pi
    denom = k_true * 2.0 * (pi**2)
    return (Q0 / denom) * torch.sin(pi * x) * torch.sin(pi * y)

def generate_poisson_data(n_points, noise_level=0.0, k_true=1.0, Q0=1.0):
    # Find factors nx, ny such that nx * ny = n_points and nx, ny are close
    nx = int(np.sqrt(n_points))
    while n_points % nx != 0:
        nx -= 1
    ny = n_points // nx
    
    # Create regular grid excluding boundaries
    # Using linspace with offset to avoid 0 and 1
    x = torch.linspace(0, 1, nx + 2)[1:-1]
    y = torch.linspace(0, 1, ny + 2)[1:-1]
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    x_data = X.flatten().reshape(-1, 1)
    y_data = Y.flatten().reshape(-1, 1)
    
    T_exact = compute_analytical_poisson(x_data, y_data, k_true, Q0)
    
    if noise_level > 0:
        noise = torch.randn_like(T_exact) * noise_level * T_exact.max().item()
        T_data = T_exact + noise
    else:
        T_data = T_exact
        
    xy_data = torch.cat([x_data, y_data], dim=1)
    return xy_data, T_data
