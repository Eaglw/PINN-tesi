import torch
import numpy as np

class InversePoissonPhysics:
    """
    Handles physics and data generation for the Heat2D Inverse Poisson problem.
    Equation: k * laplacian(T) + Q = 0
    Domain: [0,1]^2
    BCs: T(1,y)=1, others 0.
    Q = 1.0 (constant).
    """
    def __init__(self, k_param, Q_val=1.0):
        self.k_param = k_param
        self.Q_val = Q_val

    def residual(self, model, x_phys):
        """
        Computes the physics residual: k * (T_xx + T_yy) + Q = 0
        """
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
        
        # Residual = k * laplacian + Q
        res = self.k_param * laplacian + self.Q_val
        return torch.mean(res**2)

def compute_analytical_poisson(x, y, k_true=1.0, n_terms=50):
    """
    Computes analytical solution for:
    k * Laplacian(T) + 1 = 0
    BCs: T(1,y)=1, others 0.
    
    Solution T = T_hom + (1/k) * T_part
    """
    # Ensure inputs are torch tensors
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y)
        
    # T_hom: Laplace equation with T(1,y)=1
    # T_hom = sum(A_n * sinh(n*pi*x) * sin(n*pi*y))
    # A_n = 4 / (n*pi*sinh(n*pi))
    # Term = 4/(n*pi) * [sinh(n*pi*x)/sinh(n*pi)] * sin(n*pi*y)
    
    T_hom = torch.zeros_like(x)
    pi = np.pi
    
    for n in range(1, n_terms * 2, 2): # Odd terms only
        # Numerically stable calculation of sinh(a)/sinh(b)
        # For large b, approx exp(a-b).
        # We calculate log ratio and exp it.
        # sinh(z) ~ exp(z)/2. 
        # ratio ~ exp(n*pi*x - n*pi) = exp(n*pi*(x-1))
        
        # Exact ratio using exp to avoid overflow of sinh(b)
        # sinh(a)/sinh(b) = (e^a - e^-a) / (e^b - e^-b)
        # = e^(a-b) * (1 - e^-2a) / (1 - e^-2b)
        
        arg_x = n * pi * x
        arg_max = n * pi
        
        # We use a simplified stable approach:
        # If n*pi > 20, use exp approximation.
        if n * pi > 20:
             ratio = torch.exp(n * pi * (x - 1))
        else:
             ratio = torch.sinh(n * pi * x) / np.sinh(n * pi)
             
        coeff = 4.0 / (n * pi)
        term = coeff * ratio * torch.sin(n * pi * y)
        T_hom += term
        
    # T_part: Poisson equation Laplacian(T) = -1 with zero BCs
    # T_part = sum(C_nm * sin(n*pi*x) * sin(m*pi*y))
    # This corresponds to k=1 case. For other k, scale by 1/k.
    T_part = torch.zeros_like(x)
    
    # We can use fewer terms for T_part as it converges faster (1/n^3)
    for n in range(1, n_terms, 2):
        for m in range(1, n_terms, 2):
            denom = n * m * (n**2 + m**2)
            coeff = 16.0 / (pi**4 * denom)
            term = coeff * torch.sin(n * pi * x) * torch.sin(m * pi * y)
            T_part += term
            
    # Combine: T = T_hom + (1/k)*T_part
    # Note: T_part solves Laplacian(T) = -1.
    # We want k*Laplacian(T_total) = -1 => Laplacian(T_total) = -1/k.
    # Laplacian(T_hom) = 0.
    # Laplacian(1/k * T_part) = 1/k * (-1) = -1/k. Correct.
    
    return T_hom + (1.0 / k_true) * T_part

def generate_poisson_data(n_points, noise_level=0.0, k_true=1.0):
    """
    Generates synthetic data from the analytical solution.
    """
    x = torch.rand(n_points, 1)
    y = torch.rand(n_points, 1)
    
    T_exact = compute_analytical_poisson(x, y, k_true)
    
    if noise_level > 0:
        noise = torch.randn_like(T_exact) * noise_level * T_exact.max().item()
        T_data = T_exact + noise
    else:
        T_data = T_exact
        
    xy_data = torch.cat([x, y], dim=1)
    return xy_data, T_data