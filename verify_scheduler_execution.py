import os
import torch
import torch.nn as nn
from Heat2D.src.Heat2D_NN import train_modelNN
from func.logging_utils import update_results_csv
from datetime import datetime

# Setup
device = torch.device("cpu") # CPU for test
layers = [2, 10, 1]
epochs = 20 # Enough to trigger a step if we set step size small? 
# Wait, step size is hardcoded as 0.25*epochs inside the functions.
# 20 * 0.25 = 5. So decay happens at epoch 5, 10, 15.

class FCN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.fcs = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.fcs.append(nn.Linear(layers[i], layers[i+1]))
    def forward(self, x):
        for layer in self.fcs[:-1]:
            x = torch.tanh(layer(x))
        x = self.fcs[-1](x)
        return x
    def loss_fn(self, pred, target):
        return nn.MSELoss()(pred, target)

def run_test():
    print("Running Verification Script...")
    
    # Dummy Data
    xy_train = torch.rand((10, 2), device=device)
    T_train = torch.rand((10, 1), device=device)
    
    # Dummy Validation
    X = torch.rand((5, 5))
    Y = torch.rand((5, 5))
    xy_grid = torch.stack([X.flatten(), Y.flatten()], dim=1).to(device)
    T_exact = torch.rand((5, 5), device=device)
    val_grid = (xy_grid, T_exact, X, Y)
    
    # Run Fixed
    print("Testing Fixed Strategy...")
    model_fixed = FCN(layers).to(device)
    opt_fixed = torch.optim.Adam(model_fixed.parameters(), lr=1e-3)
    train_modelNN(model_fixed, opt_fixed, (xy_train, T_train), val_grid, epochs=epochs, plots_dir='test_plots', final_dir='test_results', show_plots_interactively=False, lr_strategy='fixed')
    
    # Run Step Decay
    print("Testing Step Decay Strategy...")
    model_decay = FCN(layers).to(device)
    opt_decay = torch.optim.Adam(model_decay.parameters(), lr=1e-3)
    # Check if scheduler is effectively applied by checking final LR?
    # The training function doesn't return the scheduler.
    # We can inspect the optimizer param groups after training?
    # Yes, passed by reference.
    train_modelNN(model_decay, opt_decay, (xy_train, T_train), val_grid, epochs=epochs, plots_dir='test_plots', final_dir='test_results', show_plots_interactively=False, lr_strategy='step_decay')
    
    final_lr = opt_decay.param_groups[0]['lr']
    print(f"Final LR for Step Decay (should be < 1e-3): {final_lr}")
    
    # Expected: 1e-3 * (0.5)^4 (since 20 epochs, step=5, at 5,10,15,20? No, StepLR steps at epoch indices.
    # If step_size=5. Epochs: 0,1,2,3,4 (step), 5...
    # It steps at 5, 10, 15. (3 times). 
    # 1e-3 * 0.5^3 = 1.25e-4?
    # Or maybe 4 times if it includes 20? 
    # Let's just check if it's different from 1e-3.
    
    if final_lr < 1e-3:
        print("PASS: Learning rate decayed.")
    else:
        print("FAIL: Learning rate did not decay.")
        exit(1)

    print("Verification Complete.")

if __name__ == "__main__":
    run_test()
