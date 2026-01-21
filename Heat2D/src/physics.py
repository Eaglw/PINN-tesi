import torch
import torch.nn as nn

class PhysicsProblem:
    """
    Base class for physical problems solved via PINNs.
    """
    
    def residual(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the PDE residual.
        
        Args:
            model: The neural network model.
            x: Collocation points (coordinates).
            
        Returns:
            The scalar residual (e.g., mean squared residual).
        """
        raise NotImplementedError("Subclasses must implement residual method")

    def boundary_loss(self, model: nn.Module, x_bc: torch.Tensor, y_bc: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss on the boundary.
        
        Args:
            model: The neural network model.
            x_bc: Boundary points coordinates.
            y_bc: Target values at boundary points.
            
        Returns:
            The scalar boundary loss.
        """
        raise NotImplementedError("Subclasses must implement boundary_loss method")

class HeatEquation2D(PhysicsProblem):
    """
    2D Heat Equation (Laplace) Implementation: d2T/dx2 + d2T/dy2 = 0
    """
    
    def residual(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """
        Calcola il residuo dell'equazione di Laplace 2D: d2T/dx2 + d2T/dy2 = 0
        """
        # Assicuriamoci che x richieda gradienti se non già impostato
        if not x.requires_grad:
            x.requires_grad_(True)
            
        T = model(x)
        
        # Calcolo gradienti primi
        grads = torch.autograd.grad(T, x, torch.ones_like(T), create_graph=True)[0]
        dT_dx = grads[:, 0]
        dT_dy = grads[:, 1]
        
        # Calcolo gradienti secondi
        grads2_x = torch.autograd.grad(dT_dx, x, torch.ones_like(dT_dx), create_graph=True, allow_unused=True)[0]
        d2T_dx2 = grads2_x[:, 0] if grads2_x is not None else torch.zeros_like(dT_dx)
        
        grads2_y = torch.autograd.grad(dT_dy, x, torch.ones_like(dT_dy), create_graph=True, allow_unused=True)[0]
        d2T_dy2 = grads2_y[:, 1] if grads2_y is not None else torch.zeros_like(dT_dy)
        
        # Residuo PDE
        res = d2T_dx2 + d2T_dy2
        return torch.mean(res**2)

    def boundary_loss(self, model: nn.Module, x_bc: torch.Tensor, y_bc: torch.Tensor) -> torch.Tensor:
        """
        Compute MSE loss at the boundary points.
        """
        pred = model(x_bc)
        return nn.MSELoss()(pred, y_bc)