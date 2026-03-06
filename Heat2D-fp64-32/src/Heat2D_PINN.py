import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from tqdm import tqdm

# Import function for GIF and loss comparison
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from func.graphic_func import save_gif_PIL, plot2D_comparison, plot2D_final_result
from func.history_tracker import TrainingHistory, compute_pinn_loss

# Configurazione dispositivo e precisione
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

try:
    from .precision_utils import PrecisionConfig, compute_data_loss, compute_bc_loss, compute_physics_loss, cast_to
except ImportError:
    from precision_utils import PrecisionConfig, compute_data_loss, compute_bc_loss, compute_physics_loss, cast_to

def train_modelPINN_precision(
    model, optimizer, data_internal, data_boundary, validation_grid,
    epochs=20000, physics_problem=None, plots_dir='plots', final_dir='Heat2D/Results',
    show_plots_interactively=True, loss_weights=None,
    warmup_epochs=None, collocation_points=None,
    lr_strategy='fixed', precision_config: PrecisionConfig = None
):
    """
    Training PINN con precisione configurata.
    """
    if precision_config is None:
        precision_config = PrecisionConfig() # Default FP64
        
    xy_int, T_int = data_internal
    xy_bc, T_bc = data_boundary
    xy_grid, T_exact_grid, X, Y = validation_grid
    Nx_dom, Ny_dom = X.shape

    if plots_dir: os.makedirs(plots_dir, exist_ok=True)
    if final_dir: os.makedirs(final_dir, exist_ok=True)
    plot_files = []
    
    # IMPORTANTE: Il modello viene messo subito nella precisione del training (NN + Optimizer)
    model.to(precision_config.nn_opt)
    
    pbar = tqdm(range(epochs), desc=f"Training {precision_config}")
    loss_history = TrainingHistory()
    
    if loss_weights is None: loss_weights = {'data': 1.0, 'bc': 1.0, 'physics': 1.0}
    lambda_data = loss_weights.get('data', 1.0)
    lambda_bc = loss_weights.get('bc', 1.0)
    lambda_physics = loss_weights.get('physics', 1.0)
    
    scheduler = None
    if lr_strategy == 'step_decay':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(epochs * 0.25), gamma=0.5)

    xy_physics = collocation_points.clone() if collocation_points is not None else torch.empty(0)

    for epoch in pbar:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        
        # Le funzioni compute_*_loss gestiscono internamente il casting degli input
        l_data = torch.tensor(0.0, device=xy_int.device, dtype=precision_config.nn_opt)
        if lambda_data > 0:
            l_data = compute_data_loss(model, xy_int, T_int, precision_config)
        
        l_bc = torch.tensor(0.0, device=xy_bc.device, dtype=precision_config.nn_opt)
        if lambda_bc > 0:
            l_bc = compute_bc_loss(model, xy_bc, T_bc, precision_config, physics_problem)
            
        l_phys = torch.tensor(0.0, device=xy_physics.device, dtype=precision_config.nn_opt)
        if lambda_physics > 0:
            l_phys = compute_physics_loss(model, xy_physics, precision_config, physics_problem)

        total_loss = lambda_data * l_data + lambda_bc * l_bc + lambda_physics * l_phys

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if scheduler: scheduler.step()

        # Logging
        current_lr = optimizer.param_groups[0]['lr']
        loss_dict = {
            'total_loss': total_loss.item(),
            'data_loss': l_data.item(),
            'bc_loss': l_bc.item(),
            'pde_loss': l_phys.item()
        }
        loss_history.update(epoch, loss_dict, lr=current_lr)

        if (epoch + 1) % 500 == 0:
            pbar.set_postfix({'Loss': f"{total_loss.item():.2e}", 'LR': f"{current_lr:.1e}"})
            if plots_dir:
                model.eval()
                model.to(torch.float64) # Plot sempre in FP64
                with torch.no_grad():
                    T_pred_grid = model(xy_grid.to(torch.float64)).reshape(Nx_dom, Ny_dom)
                plot_path = os.path.join(plots_dir, f'epoch_{epoch+1}.png')
                plot2D_comparison(X, Y, T_exact_grid, T_pred_grid, epoch+1, plot_path)
                plot_files.append(plot_path)
                model.to(precision_config.nn_opt) # Torna in precisione training

    # Final Result
    model.eval()
    model.to(torch.float64)
    with torch.no_grad():
        T_final = model(xy_grid.to(torch.float64)).reshape(Nx_dom, Ny_dom)
    
    if final_dir:
        final_path = os.path.join(final_dir, 'PINNfinal_result_precision.png')
        plot2D_final_result(X, Y, T_exact_grid, T_final, epochs, save_path=final_path)
    
    return loss_history
