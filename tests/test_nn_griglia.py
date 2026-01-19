import torch
import os
from Heat2D.Heat2D_NN_griglia import train_modelNN_griglia
from Heat2D.Heat2D_main import FCN

def test_nn_griglia_integration():
    # Setup dummy data
    model = FCN(layers=[2, 10, 1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    xy_train = torch.rand(10, 2)
    T_train = torch.rand(10, 1)
    training_data = (xy_train, T_train)
    
    # Validation grid
    X, Y = torch.meshgrid(torch.linspace(0, 1, 5), torch.linspace(0, 1, 5), indexing='xy')
    xy_grid = torch.stack([X.flatten(), Y.flatten()], dim=1)
    validation_grid = (xy_grid, torch.zeros_like(X.flatten().unsqueeze(1)), X, Y)
    
    # Run training for 1 epoch
    history = train_modelNN_griglia(
        model, optimizer, training_data, validation_grid, 
        epochs=1, show_plots_interactively=False,
        plots_dir='tests/plots', final_dir='tests/results'
    )
    
    assert history is not None
    assert len(history.epochs) == 1
