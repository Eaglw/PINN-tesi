import torch
import numpy as np

def analytical_solution_source(x, y, alpha=1.0):
    """
    Computes the analytical solution for the Poisson equation:
    alpha * (d2T/dx2 + d2T/dy2) = -Q
    with Q = 2 * pi^2 * alpha * sin(pi*x) * sin(pi*y)
    
    The solution is T(x,y) = sin(pi*x) * sin(pi*y)
    
    Note: The 'alpha' parameter scales the source term Q to ensure T remains the same 
    regardless of alpha. This allows us to fix T_observed and estimate alpha
    from the physics mismatch.
    """
    return torch.sin(np.pi * x) * torch.sin(np.pi * y)

def source_term(x, y, alpha=1.0):
    """
    Returns the source term Q for the equation alpha * laplacian(T) + Q = 0
    Q = 2 * pi^2 * alpha * sin(pi*x) * sin(pi*y)
    """
    return 2 * (np.pi**2) * alpha * torch.sin(np.pi * x) * torch.sin(np.pi * y)

def generate_inverse_data(n_points, noise_level=0.0, alpha_true=1.0, domain_limits=(1.0, 1.0)):
    """
    Generates synthetic observation data for the inverse problem.
    
    Args:
        n_points (int): Number of points to sample.
        noise_level (float): Standard deviation of Gaussian noise to add (as a fraction of max value).
        alpha_true (float): The true thermal diffusivity used to generate the physics.
        domain_limits (tuple): (Lx, Ly) dimensions of the domain.
        
    Returns:
        tuple: (xy_data, T_data)
            - xy_data: torch.Tensor of shape (n_points, 2)
            - T_data: torch.Tensor of shape (n_points, 1)
    """
    Lx, Ly = domain_limits
    
    # Generate random points in the domain
    x = torch.rand(n_points, 1) * Lx
    y = torch.rand(n_points, 1) * Ly
    
    # Calculate true temperature
    T_exact = analytical_solution_source(x, y, alpha=alpha_true)
    
    # Add noise
    if noise_level > 0:
        noise = torch.randn_like(T_exact) * noise_level * T_exact.max().item()
        T_data = T_exact + noise
    else:
        T_data = T_exact
        
    xy_data = torch.cat([x, y], dim=1)
    
    return xy_data, T_data

class InverseHeatPhysics:
    """
    Implements the physics loss for the inverse Heat2D problem.
    The equation is: alpha * (d2T/dx2 + d2T/dy2) + Q = 0
    where Q is the source term.
    """
    def __init__(self, alpha_param):
        self.alpha_param = alpha_param
        
    def __call__(self, model, xy_p):
        """
        Computes the physics residual: alpha * Laplacian(T) + Q
        """
        if not xy_p.requires_grad:
            xy_p.requires_grad_(True)
            
        T = model(xy_p)
        
        # Gradients
        grads = torch.autograd.grad(T, xy_p, torch.ones_like(T), create_graph=True)[0]
        dT_dx = grads[:, 0]
        dT_dy = grads[:, 1]
        
        grads2_x = torch.autograd.grad(dT_dx, xy_p, torch.ones_like(dT_dx), create_graph=True, allow_unused=True)[0]
        d2T_dx2 = grads2_x[:, 0] if grads2_x is not None else torch.zeros_like(dT_dx)
        
        grads2_y = torch.autograd.grad(dT_dy, xy_p, torch.ones_like(dT_dy), create_graph=True, allow_unused=True)[0]
        d2T_dy2 = grads2_y[:, 1] if grads2_y is not None else torch.zeros_like(dT_dy)
        
        laplacian = d2T_dx2 + d2T_dy2
        
        # Source term Q (calculated with true alpha=1.0 in our synthetic case)
        # Note: In a real case, Q would be a known function or data.
        Q = source_term(xy_p[:, 0], xy_p[:, 1], alpha=1.0) 
        
        # Residual: alpha * Laplacian + Q
        # We want this to be zero when alpha == alpha_true (which is 1.0)
        res = self.alpha_param * laplacian + Q
        
        return torch.mean(res**2)
