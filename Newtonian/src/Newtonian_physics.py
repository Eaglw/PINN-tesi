import torch
import torch.nn as nn

class NewtonianPhysics:
    """
    2D Steady Navier-Stokes Implementation using Stream Function (psi):
    - u = d(psi)/dy
    - v = -d(psi)/dx
    - Continuity (u_x + v_y = 0) is automatically satisfied.
    """
    def __init__(self, mu=1.0, rho=1.0):
        self.mu = mu
        self.rho = rho

    def get_velocity(self, model: nn.Module, xy: torch.Tensor):
        """
        Calculates [u, v, p] from model outputs [psi, p]
        """
        if not xy.requires_grad:
            xy.requires_grad_(True)
            
        out = model(xy)
        psi = out[:, 0:1]
        p = out[:, 1:2]
        
        # u = d(psi)/dy, v = -d(psi)/dx
        grads_psi = torch.autograd.grad(psi, xy, torch.ones_like(psi), create_graph=True)[0]
        u = grads_psi[:, 1:2]
        v = -grads_psi[:, 0:1]
        
        return u, v, p

    def residual(self, model: nn.Module, xy: torch.Tensor) -> torch.Tensor:
        """
        Calculates the residual of the Momentum equations using psi.
        """
        u, v, p = self.get_velocity(model, xy)
        
        # First order derivatives of u, v for convective terms and gradients of p
        du_dxy = torch.autograd.grad(u, xy, torch.ones_like(u), create_graph=True)[0]
        u_x, u_y = du_dxy[:, 0:1], du_dxy[:, 1:2]
        
        dv_dxy = torch.autograd.grad(v, xy, torch.ones_like(v), create_graph=True)[0]
        v_x, v_y = dv_dxy[:, 0:1], dv_dxy[:, 1:2]
        
        dp_dxy = torch.autograd.grad(p, xy, torch.ones_like(p), create_graph=True)[0]
        p_x, p_y = dp_dxy[:, 0:1], dp_dxy[:, 1:2]
        
        # Second order derivatives of u, v for viscous terms
        u_xx = torch.autograd.grad(u_x, xy, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
        u_yy = torch.autograd.grad(u_y, xy, torch.ones_like(u_y), create_graph=True)[0][:, 1:2]
        
        v_xx = torch.autograd.grad(v_x, xy, torch.ones_like(v_x), create_graph=True)[0][:, 0:1]
        v_yy = torch.autograd.grad(v_y, xy, torch.ones_like(v_y), create_graph=True)[0][:, 1:2]
        
        # Momentum Equations (Steady 2D)
        # MX: rho*(u*u_x + v*u_y) + p_x - mu*(u_xx + u_yy) = 0
        res_mx = self.rho * (u * u_x + v * u_y) + p_x - self.mu * (u_xx + u_yy)
        
        # MY: rho*(u*v_x + v*v_y) + p_y - mu*(v_xx + v_yy) = 0
        res_my = self.rho * (u * v_x + v * v_y) + p_y - self.mu * (v_xx + v_yy)
        
        # Total residual loss (Continuity is implicitly satisfied)
        return torch.mean(res_mx**2) + torch.mean(res_my**2)

    def boundary_loss(self, model: nn.Module, x_bc: torch.Tensor, y_bc: torch.Tensor) -> torch.Tensor:
        """
        Compute MSE loss at the boundary points for [u, v, p].
        Target y_bc: (N, 3) -> [u_exact, v_exact, p_exact]
        """
        u, v, p = self.get_velocity(model, x_bc)
        pred = torch.cat([u, v, p], dim=1)
        return nn.MSELoss()(pred, y_bc)

def generate_boundaries(Lx, Ly, u_max, p_exact, P_grid, Nx_dom, Ny_dom, device):
    """
    Generates boundary conditions points and target values [u, v, p] for a 2D Poiseuille flow.
    """
    num_b_y = Ny_dom 
    pts_bc = torch.linspace(0, Ly, num_b_y, device=device).reshape(-1, 1)
    
    u_parabolic = 4 * u_max * (pts_bc * (Ly - pts_bc)) / (Ly**2)
    v_zero = torch.zeros_like(pts_bc)
    
    # Left (x=0) - Inlet
    bc_left = torch.cat([torch.zeros(num_b_y, 1, device=device), pts_bc], dim=1)
    p_in = p_exact.flatten()[0]
    bc_left_val = torch.cat([u_parabolic, v_zero, torch.ones_like(pts_bc) * p_in], dim=1)
    
    # Right (x=Lx) - Outlet
    bc_right = torch.cat([torch.ones(num_b_y, 1, device=device) * Lx, pts_bc], dim=1)
    p_out = p_exact.flatten()[-1]
    bc_right_val = torch.cat([u_parabolic, v_zero, torch.ones_like(pts_bc) * p_out], dim=1)
    
    # Walls (Top/Bottom) - No-slip
    num_b_x = Nx_dom
    pts_x = torch.linspace(0, Lx, num_b_x, device=device).reshape(-1, 1)
    
    # Bottom (y=0)
    bc_bottom = torch.cat([pts_x, torch.zeros(num_b_x, 1, device=device)], dim=1)
    p_bottom = P_grid[0, :] 
    bc_bottom_val = torch.cat([torch.zeros_like(pts_x), torch.zeros_like(pts_x), p_bottom.reshape(-1, 1)], dim=1)
    
    # Top (y=Ly)
    bc_top = torch.cat([pts_x, torch.ones(num_b_x, 1, device=device) * Ly], dim=1)
    p_top = P_grid[-1, :]
    bc_top_val = torch.cat([torch.zeros_like(pts_x), torch.zeros_like(pts_x), p_top.reshape(-1, 1)], dim=1)
    
    xy_boundary = torch.cat([bc_left, bc_right, bc_bottom, bc_top], dim=0)
    uvp_boundary = torch.cat([bc_left_val, bc_right_val, bc_bottom_val, bc_top_val], dim=0)
    
    return xy_boundary, uvp_boundary