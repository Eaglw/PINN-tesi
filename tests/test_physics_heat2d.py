import torch
import torch.nn as nn
from Heat2D.physics import PhysicsProblem

# Simple linear model for testing residuals
# T = 1.0 * x + 2.0 * y
class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.tensor([[1.0, 2.0]]))
        self.bias = nn.Parameter(torch.tensor([0.0]))
        
    def forward(self, x):
        return x @ self.weights.t() + self.bias

def test_heat_equation_2d_residual():
    # Placeholder import until class is implemented
    from Heat2D.physics import HeatEquation2D
    
    heat_eq = HeatEquation2D()
    model = LinearModel()
    
    # Test point (0.5, 0.5)
    x = torch.tensor([[0.5, 0.5]], requires_grad=True)
    
    # For a linear model T = x + 2y, d2T/dx2 = 0 and d2T/dy2 = 0
    # So the residual should be 0
    res = heat_eq.residual(model, x)
    
    assert torch.isclose(res, torch.tensor(0.0))

def test_heat_equation_2d_boundary_loss():
    from Heat2D.physics import HeatEquation2D
    
    heat_eq = HeatEquation2D()
    model = LinearModel()
    
    # Boundary points and targets
    x_bc = torch.tensor([[0.0, 0.5], [1.0, 0.5]])
    # Target values according to T = x + 2y:
    # (0, 0.5) -> 0 + 2*0.5 = 1.0
    # (1, 0.5) -> 1 + 2*0.5 = 2.0
    y_bc = torch.tensor([[1.0], [2.0]])
    
    loss = heat_eq.boundary_loss(model, x_bc, y_bc)
    
    assert torch.isclose(loss, torch.tensor(0.0))

def test_physics_problem_base_raises():
    class IncompletePhysics(PhysicsProblem):
        pass
    
    physics = IncompletePhysics()
    model = LinearModel()
    x = torch.tensor([[0.5, 0.5]])
    
    import pytest
    with pytest.raises(NotImplementedError):
        physics.residual(model, x)
    
    with pytest.raises(NotImplementedError):
        physics.boundary_loss(model, x, x)

def test_heat_equation_2d_requires_grad_auto():
    from Heat2D.physics import HeatEquation2D
    heat_eq = HeatEquation2D()
    model = LinearModel()
    # x without requires_grad
    x = torch.tensor([[0.5, 0.5]])
    res = heat_eq.residual(model, x)
    assert x.requires_grad
    assert torch.isclose(res, torch.tensor(0.0))
