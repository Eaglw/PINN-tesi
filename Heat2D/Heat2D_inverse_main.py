import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from tqdm import tqdm
import time
import csv

# Add root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from func.graphic_func import save_gif_PIL, plot2D_comparison
from func.logging_utils import compute_metrics
from func.history_tracker import TrainingHistory
from Heat2D.src.inverse_physics import generate_poisson_data, InversePoissonPhysics, compute_analytical_poisson

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

def plot_k_convergence(k_history, true_k, save_path):
    plt.figure(figsize=(8, 5))
    plt.plot(k_history, label='Estimated k', color='blue')
    plt.axhline(y=true_k, color='red', linestyle='--', label='True k')
    plt.xlabel('Epoch')
    plt.ylabel('Conductivity k')
    plt.title('Parameter Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.close()

def update_inverse_results_csv(file_path, data_dict):
    file_exists = os.path.exists(file_path)
    # Correct order from spec/plan
    fieldnames = [
        'Experiment_ID', 'Architecture', 'Optimizer', 'Epochs', 'Activation',
        'Scheduler', 'Loss_Function', 'Noise_Level', 'Data_Points', 'Balanced_Loss', 'Execution_Time(s)',
        'Final_Total_Loss', 'Final_Physics_Loss', 'Final_Data_Loss', 'Final_BC_Loss',
        'MAE', 'L2_Error', 'True_K', 'Estimated_K', 'Rel_Error_K'
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
    activation_fn=nn.Tanh,
    lr_weights=1e-3, 
    lr_k=1e-2,
    k_init=0.5,
    true_k=1.0,
    exp_name="Inverse_Exp",
    use_lbfgs=True,
    balanced_loss=False
):
    # Setup folders
    exp_dir = os.path.join("Heat2D/experiments_inverse", exp_name)
    plots_dir = os.path.join(exp_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Generate Data (Ground Truth)
    # Generate points in domain for data loss
    xy_data, T_data = generate_poisson_data(n_points, noise_level, k_true=true_k)
    xy_data, T_data = xy_data.to(device), T_data.to(device)
    
    # 2. Validation grid for plotting
    x_val = torch.linspace(0, 1, 50)
    y_val = torch.linspace(0, 1, 50)
    X, Y = torch.meshgrid(x_val, y_val, indexing='ij')
    xy_grid = torch.stack([X.flatten(), Y.flatten()], dim=1).to(device)
    T_exact_grid = compute_analytical_poisson(X, Y, k_true=true_k).to(device)
    
    # 3. Initialize Model and Parameter
    model = FCN(layers, activation_fn=activation_fn).to(device)
    k_train = nn.Parameter(torch.tensor([k_init], device=device, requires_grad=True))
    
    # 4. Optimizer
    optimizer = torch.optim.Adam([
        {'params': model.parameters(), 'lr': lr_weights},
        {'params': [k_train], 'lr': lr_k}
    ])
    
    # Physics helper
    physics_fn = InversePoissonPhysics(k_train, Q0=1.0)
    
    # 5. Training Loop (Adam)
    history = TrainingHistory()
    k_history = []
    plot_files = []
    
    start_time = time.time()
    pbar = tqdm(range(epochs), desc=f"Training {exp_name} (Adam)")
    
    # Fixed physics points for consistency/stability
    xy_phys = torch.rand((2000, 2), device=device, requires_grad=True)

    # Generate BC points (All T=0) - Equidistanti senza angoli
    n_bc = 200
    # Usiamo linspace[1:-1] per escludere 0.0 e 1.0 (gli angoli)
    pts = torch.linspace(0, 1, n_bc, device=device)[1:-1].reshape(-1, 1)
    num_pts = pts.shape[0]
    
    # All boundaries -> T=0
    bc_right = torch.cat([torch.ones(num_pts, 1, device=device), pts], dim=1)
    bc_left = torch.cat([torch.zeros(num_pts, 1, device=device), pts], dim=1)
    bc_top = torch.cat([pts, torch.ones(num_pts, 1, device=device)], dim=1)
    bc_bottom = torch.cat([pts, torch.zeros(num_pts, 1, device=device)], dim=1)
    
    all_bc_x = torch.cat([bc_right, bc_left, bc_top, bc_bottom], dim=0)
    all_bc_val = torch.zeros((all_bc_x.shape[0], 1), device=device) # T=0 everywhere on BC
    
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()
        
        # Data Loss
        T_pred_data = model(xy_data)
        if balanced_loss:
            # Explicitly normalized on number of points
            loss_data = torch.sum((T_pred_data - T_data)**2) / n_points
        else:
            # Standard MSE
            loss_data = torch.mean((T_pred_data - T_data)**2)
        
        # BC Loss
        T_pred_bc = model(all_bc_x)
        loss_bc = torch.mean((T_pred_bc - all_bc_val)**2)
        
        # Physics Loss
        loss_phys = physics_fn.residual(model, xy_phys)
        
        total_loss = loss_data + loss_bc + loss_phys
        
        total_loss.backward()
        optimizer.step()
        
        k_history.append(k_train.item())
        current_lr = optimizer.param_groups[0]['lr']
        history.update(epoch, {
            'total_loss': total_loss.item(),
            'data_loss': loss_data.item(),
            'bc_loss': loss_bc.item(),
            'pde_loss': loss_phys.item()
        }, lr=current_lr)
        
        if (epoch + 1) % 500 == 0:
            pbar.set_postfix({'Loss': f"{total_loss.item():.2e}", 'k': f"{k_train.item():.4f}"})
            
            model.eval()
            with torch.no_grad():
                T_pred_grid = model(xy_grid).reshape(X.shape)
            
            plot_path = os.path.join(plots_dir, f'epoch_{epoch+1}.png')
            # Visualizing the experimental data points on the prediction map
            plot2D_comparison(X.cpu(), Y.cpu(), T_exact_grid.cpu(), T_pred_grid.cpu(), epoch+1, plot_path, physics_points=xy_data.cpu())
            plot_files.append(plot_path)

    # L-BFGS Refinement
    if use_lbfgs:
        print("\nStarting L-BFGS refinement...")
        model.train()
        
        optimizer_lbfgs = torch.optim.LBFGS(
            list(model.parameters()) + [k_train], 
            lr=1.0, 
            max_iter=1000, 
            history_size=50,
            line_search_fn="strong_wolfe"
        )
        
        lbfgs_iter = [0]
        def closure():
            optimizer_lbfgs.zero_grad()
            T_pred_data = model(xy_data)
            if balanced_loss:
                loss_data = torch.sum((T_pred_data - T_data)**2) / n_points
            else:
                loss_data = torch.mean((T_pred_data - T_data)**2)
            
            T_pred_bc = model(all_bc_x)
            loss_bc = torch.mean((T_pred_bc - all_bc_val)**2)
            
            loss_phys = physics_fn.residual(model, xy_phys)
            
            total_loss = loss_data + loss_bc + loss_phys
            total_loss.backward()
            
            if lbfgs_iter[0] % 10 == 0:
                history.update(epochs + lbfgs_iter[0], {
                    'total_loss': total_loss.item(),
                    'data_loss': loss_data.item(),
                    'bc_loss': loss_bc.item(),
                    'pde_loss': loss_phys.item()
                }, lr=1.0)
            lbfgs_iter[0] += 1
            return total_loss
            
        optimizer_lbfgs.step(closure)
        
        # Final update
        model.eval()
        with torch.no_grad():
             T_pred_data = model(xy_data)
             if balanced_loss:
                loss_data_final = (torch.sum((T_pred_data - T_data)**2) / n_points).item()
             else:
                loss_data_final = torch.mean((T_pred_data - T_data)**2).item()
             T_pred_bc = model(all_bc_x)
             loss_bc_final = torch.mean((T_pred_bc - all_bc_val)**2).item()
        
        # Physics needs grad
        loss_phys_final = physics_fn.residual(model, xy_phys).item()
        
        total_loss_final = loss_data_final + loss_bc_final + loss_phys_final
        
        history.update(epochs + lbfgs_iter[0], {
            'total_loss': total_loss_final,
            'data_loss': loss_data_final,
            'bc_loss': loss_bc_final,
            'pde_loss': loss_phys_final
        }, lr=1.0)
        k_history.append(k_train.item())

    execution_time = time.time() - start_time
    
    # 6. Finalize and Log
    model.eval()
    with torch.no_grad():
        T_final_grid = model(xy_grid).reshape(X.shape)
        final_k = k_train.item()
    
    plot_k_convergence(k_history, true_k, os.path.join(exp_dir, "k_convergence.png"))
    history.plot_losses(adam_epochs=epochs if use_lbfgs else None, 
                        save_path=os.path.join(exp_dir, "loss_history.png"), 
                        experiment_name=exp_name, 
                        show_plot=False)
    
    if plot_files:
        save_gif_PIL(os.path.join(exp_dir, "training_evolution.gif"), plot_files, fps=5, loop=0, delete_files=True)
        
    # Compute Metrics
    l2_rel, max_peak = compute_metrics(model, xy_grid, T_exact_grid)
    k_error = abs(final_k - true_k) / true_k * 100
    
    # Update CSV
    results_data = {
        'Experiment_ID': exp_name,
        'Architecture': f"{layers}",
        'Optimizer': 'Adam+LBFGS' if use_lbfgs else 'Adam',
        'Epochs': epochs,
        'Activation': activation_fn.__name__,
        'Scheduler': 'None',
        'Loss_Function': 'MSE',
        'Noise_Level': noise_level,
        'Data_Points': n_points,
        'Balanced_Loss': balanced_loss,
        'Execution_Time(s)': execution_time,
        'Final_Total_Loss': history.losses['total_loss'][-1],
        'Final_Physics_Loss': history.losses['pde_loss'][-1],
        'Final_Data_Loss': history.losses['data_loss'][-1],
        'Final_BC_Loss': history.losses['bc_loss'][-1],
        'MAE': max_peak, 
        'L2_Error': l2_rel,
        'True_K': true_k,
        'Estimated_K': final_k,
        'Rel_Error_K': k_error
    }
    
    update_inverse_results_csv("Heat2D/results_inverse.csv", results_data)
    
    print(f"Final k: {final_k:.4f} (True: {true_k}, Error: {k_error:.2f}%)")
    plt.close('all')

if __name__ == "__main__":

    # Grid Search
    layers_options = [[2, 50, 50, 50, 50, 1]]
    noise_options = [0.0, 0.01, 0.02, 0.05, 0.10]
    points_options = [50, 100, 500]
    activation_options = [nn.Tanh, nn.GELU, nn.SiLU]
    balanced_options = [False, True]

    for act_fn in activation_options:
        for noise in noise_options:
            for pts in points_options:
                for balanced in balanced_options:
                    act_name = act_fn.__name__
                    bal_str = "Bal" if balanced else "Std"
                    exp_name = f"Poisson_Inverse_{act_name}_N{int(noise*100)}_P{pts}_{bal_str}"
                    print(f"\n--- Running Experiment: {exp_name} ---")

                    run_inverse_experiment(
                        layers=layers_options[0],
                        epochs=5000, 
                        n_points=pts,
                        noise_level=noise,
                        activation_fn=act_fn,
                        exp_name=exp_name,
                        use_lbfgs=True,
                        balanced_loss=balanced
                    )

            






