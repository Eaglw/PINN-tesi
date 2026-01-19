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