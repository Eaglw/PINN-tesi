import torch
import torch.nn as nn
from Heat2D.physics import PhysicsProblem

# Dummy physics for testing
class DummyPhysics(PhysicsProblem):
    def residual(self, model, x):
        return torch.tensor(0.1) # constant residual
    def boundary_loss(self, model, x_bc, y_bc):
        return torch.tensor(0.2)

# Dummy model
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 1)
    def forward(self, x):
        return self.fc(x)

def test_train_modelPINN_modular():
    from Heat2D.Heat2D_PINN import train_modelPINN
    
    # Very short run for testing
    model = SimpleNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    physics = DummyPhysics()
    
    data_internal = (torch.tensor([[0.5, 0.5]]), torch.tensor([[1.0]]))
    data_boundary = (torch.tensor([[0.0, 0.5]]), torch.tensor([[1.0]]))
    
    # Mock grid for validation
    X, Y = torch.meshgrid(torch.linspace(0, 1, 2), torch.linspace(0, 1, 2), indexing='xy')
    validation_grid = (torch.stack([X.flatten(), Y.flatten()], dim=1), torch.zeros_like(X), X, Y)
    
    # This should now accept a physics_problem argument
    # We set epochs to 1 for a quick test
    train_modelPINN(
        model,
        optimizer,
        data_internal,
        data_boundary,
        validation_grid,
        epochs=1,
        physics_problem=physics,
        show_plots_interactively=False
    )
