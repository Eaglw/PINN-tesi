import torch
import torch.nn as nn
from src.precision_utils import PrecisionConfig
from src.Heat2D_PINN import train_modelPINN_precision
from src.physics import HeatEquation2D

def test_pinn_precision():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 5 toggleable parts: NN, Data, Physics, BC, Optimizer
    # Example: NN=FP32, others=FP64
    config = PrecisionConfig(nn=torch.float32, data=torch.float64, physics=torch.float64, bc=torch.float64, optimizer=torch.float64)
    print(f"Testing with: {config}")
    
    layers = [2, 10, 1]
    model = nn.Sequential(
        nn.Linear(2, 10),
        nn.Tanh(),
        nn.Linear(10, 1)
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Dummy data
    xy_int = torch.rand(10, 2).to(device)
    T_int = torch.rand(10, 1).to(device)
    xy_bc = torch.rand(10, 2).to(device)
    T_bc = torch.rand(10, 1).to(device)
    
    xy_grid = torch.rand(100, 2).to(device)
    T_grid = torch.rand(100, 1).to(device)
    X, Y = torch.meshgrid(torch.linspace(0, 1, 10), torch.linspace(0, 1, 10), indexing='xy')
    
    data_internal = (xy_int, T_int)
    data_boundary = (xy_bc, T_bc)
    validation_grid = (xy_grid, T_grid, X, Y)
    
    physics_problem = HeatEquation2D()
    
    print("Starting 1 epoch training...")
    history = train_modelPINN_precision(
        model, optimizer, data_internal, data_boundary, validation_grid,
        epochs=1, physics_problem=physics_problem,
        plots_dir='test_plots', final_dir='test_results',
        show_plots_interactively=False,
        precision_config=config,
        collocation_points=xy_int
    )
    print("Training successful!")
    
if __name__ == "__main__":
    test_pinn_precision()
