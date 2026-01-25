import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from tqdm import tqdm
import time

# Add root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from func.graphic_func import save_gif_PIL, plot2D_comparison
from func.logging_utils import compute_metrics
from func.history_tracker import TrainingHistory
from Heat2D.src.inverse_physics import generate_inverse_data, InverseHeatPhysics, analytical_solution_source

# Precision and device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

# Model Definition
class FCN(nn.Module):
    def __init__(self, layers, activation_fn=nn.Tanh):
        super().__init__()
        self.activation = activation_fn()
        self.fcs = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.fcs.append(nn.Linear(layers[i], layers[i+1]))
    def forward(self, x):
        for i, layer in enumerate(self.fcs):
            x = layer(x)
            if i < len(self.fcs) - 1:
                x = self.activation(x)
        return x

def plot_alpha_convergence(alpha_history, true_alpha, save_path):
    plt.figure(figsize=(8, 5))
    plt.plot(alpha_history, label='Estimated Alpha', color='blue')
    plt.axhline(y=true_alpha, color='red', linestyle='--', label='True Alpha')
    plt.xlabel('Epoch')
    plt.ylabel('Alpha Value')
    plt.title('Alpha Parameter Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.close()

def update_inverse_results_csv(file_path, data_dict):
    import csv
    file_exists = os.path.exists(file_path)
    fieldnames = [
        'Experiment_ID', 'Architecture', 'Optimizer', 'Epochs', 'Activation', 
        'Scheduler', 'Loss_Function', 'Execution_Time(s)', 'Final_Total_Loss',
        'Final_Physics_Loss', 'Final_Data_Loss', 'Final_BC_Loss', 'MAE', 'L2_Error',
        'True_Alpha', 'Estimated_Alpha', 'Alpha_Relative_Error_Percent',
        'Noise_Level', 'Data_Points'
    ]
    
    with open(file_path, mode='a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(data_dict)

def run_inverse_experiment(
    layers, 
    epochs, 
    n_points, 
    noise_level, 
    lr_weights=1e-3, 
    lr_alpha=1e-2,
    alpha_init=0.5,
    true_alpha=1.0,
    exp_name="Inverse_Exp",
    use_lbfgs=True
):
    # Setup folders
    exp_dir = os.path.join("Heat2D/experiments_inverse", exp_name)
    plots_dir = os.path.join(exp_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Generate Data
    xy_data, T_data = generate_inverse_data(n_points, noise_level, alpha_true=true_alpha)
    xy_data, T_data = xy_data.to(device), T_data.to(device)
    
    # Validation grid for plotting
    x_val = torch.linspace(0, 1, 50)
    y_val = torch.linspace(0, 1, 50)
    X, Y = torch.meshgrid(x_val, y_val, indexing='ij')
    xy_grid = torch.stack([X.flatten(), Y.flatten()], dim=1).to(device)
    T_exact_grid = analytical_solution_source(X, Y, alpha=true_alpha).to(device)
    
    # 2. Initialize Model and Parameter
    model = FCN(layers).to(device)
    alpha_train = nn.Parameter(torch.tensor([alpha_init], device=device, requires_grad=True))
    
    # 3. Optimizer
    optimizer = torch.optim.Adam([
        {'params': model.parameters(), 'lr': lr_weights},
        {'params': [alpha_train], 'lr': lr_alpha}
    ])
    
    # 4. Physics
    physics_fn = InverseHeatPhysics(alpha_train)
    
    # 5. Training Loop (Adam)
    history = TrainingHistory()
    alpha_history = []
    plot_files = []
    
    start_time = time.time()
    pbar = tqdm(range(epochs), desc=f"Training {exp_name} (Adam)")
    
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()
        
        # Loss calculation
        T_pred = model(xy_data)
        loss_data = torch.mean((T_pred - T_data)**2)
        
        xy_phys = torch.rand((1000, 2), device=device)
        loss_phys = physics_fn(model, xy_phys)
        
        total_loss = loss_data + loss_phys
        
        total_loss.backward()
        optimizer.step()
        
        alpha_history.append(alpha_train.item())
        history.update(epoch, {
            'total_loss': total_loss.item(),
            'data_loss': loss_data.item(),
            'pde_loss': loss_phys.item()
        })
        
        if (epoch + 1) % 500 == 0:
            pbar.set_postfix({'Loss': f"{total_loss.item():.2e}", 'Alpha': f"{alpha_train.item():.4f}"})
            
            model.eval()
            with torch.no_grad():
                T_pred_grid = model(xy_grid).reshape(X.shape)
            
            plot_path = os.path.join(plots_dir, f'epoch_{epoch+1}.png')
            plot2D_comparison(X.cpu(), Y.cpu(), T_exact_grid.cpu(), T_pred_grid.cpu(), epoch+1, plot_path, physics_points=xy_data.cpu())
            plot_files.append(plot_path)

    # L-BFGS Refinement
    if use_lbfgs:
        print("\nInizio fase di raffinamento con L-BFGS...")
        model.train()
        
        # Fixed points for L-BFGS to avoid graph issues
        xy_phys_lbfgs = torch.rand((1000, 2), device=device, requires_grad=True)
        
        optimizer_lbfgs = torch.optim.LBFGS(
            list(model.parameters()) + [alpha_train], 
            lr=1.0, 
            max_iter=500, 
            history_size=50,
            line_search_fn="strong_wolfe"
        )
        
        def closure():
            optimizer_lbfgs.zero_grad()
            T_pred = model(xy_data)
            loss_data = torch.mean((T_pred - T_data)**2)
            
            # Use fixed points
            loss_phys = physics_fn(model, xy_phys_lbfgs)
            
            total_loss = loss_data + loss_phys
            total_loss.backward()
            return total_loss
            
        optimizer_lbfgs.step(closure)
        
        # Final update
        model.eval()
        xy_phys_final = torch.rand((1000, 2), device=device, requires_grad=True)
        # We need gradients for physics_fn, so we don't use no_grad for it
        # but we don't call backward either.
        loss_phys_final = physics_fn(model, xy_phys_final).item()
        
        with torch.no_grad():
            T_pred = model(xy_data)
            loss_data_final = torch.mean((T_pred - T_data)**2).item()
            
            history.update(epochs + 1, {
                'total_loss': loss_data_final + loss_phys_final,
                'data_loss': loss_data_final,
                'pde_loss': loss_phys_final
            })
            alpha_history.append(alpha_train.item())

    execution_time = time.time() - start_time
    
    # 6. Finalize and Log
    model.eval()
    with torch.no_grad():
        T_final_grid = model(xy_grid).reshape(X.shape)
        final_alpha = alpha_train.item()
    
    # Save Alpha Convergence Plot
    plot_alpha_convergence(alpha_history, true_alpha, os.path.join(exp_dir, "alpha_convergence.png"))
    
    # Save Loss History - Explicitly set show_plot=False
    history.plot_losses(save_path=os.path.join(exp_dir, "loss_history.png"), experiment_name=exp_name, show_plot=False)
    
    if plot_files:
        save_gif_PIL(os.path.join(exp_dir, "training_evolution.gif"), plot_files, fps=5, loop=0, delete_files=True)
        
    # Compute Metrics
    l2_rel, max_peak = compute_metrics(model, xy_grid, T_exact_grid)
    alpha_error = abs(final_alpha - true_alpha) / true_alpha * 100
    
    # Update CSV
    results_data = {
        'Experiment_ID': exp_name,
        'Architecture': f"{layers}",
        'Optimizer': 'Adam+LBFGS' if use_lbfgs else 'Adam',
        'Epochs': epochs,
        'Activation': 'Tanh',
        'Scheduler': 'None',
        'Loss_Function': 'MSE',
        'Execution_Time(s)': execution_time,
        'Final_Total_Loss': history.losses['total_loss'][-1],
        'Final_Physics_Loss': history.losses['pde_loss'][-1],
        'Final_Data_Loss': history.losses['data_loss'][-1],
        'Final_BC_Loss': 0.0,
        'MAE': max_peak, 
        'L2_Error': l2_rel,
        'True_Alpha': true_alpha,
        'Estimated_Alpha': final_alpha,
        'Alpha_Relative_Error_Percent': alpha_error,
        'Noise_Level': noise_level,
        'Data_Points': n_points
    }
    
    update_inverse_results_csv("Heat2D/results_inverse.csv", results_data)
    
    print(f"Final Alpha: {final_alpha:.4f} (True: {true_alpha}, Error: {alpha_error:.2f}%)")
    plt.close('all') # Safety measure to close any lingering figures

if __name__ == "__main__":

    # Robust grid search

    layers_options = [[2, 50, 50, 50, 50, 1]]

    noise_options = [0.0, 0.02, 0.05]

    points_options = [100, 500, 1000]

    

    for noise in noise_options:

        for pts in points_options:

            exp_name = f"Inverse_N{int(noise*100)}_P{pts}"

            print(f"\n--- Running Experiment: {exp_name} ---")

            run_inverse_experiment(

                layers=layers_options[0],

                epochs=10000, 

                n_points=pts,

                noise_level=noise,

                exp_name=exp_name,

                use_lbfgs=True

            )




