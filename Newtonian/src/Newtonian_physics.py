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