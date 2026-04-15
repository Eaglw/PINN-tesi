import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from tqdm import tqdm

# Import function for GIF
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from func.graphic_func import save_gif_PIL, plot2D_comparison, plot2D_final_result
from func.history_tracker import TrainingHistory # Added import

"""# Configurazione dispositivo e precisione
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)
"""
try:
    from .precision_utils import PrecisionConfig, compute_data_loss, cast_to
except ImportError:
    from precision_utils import PrecisionConfig, compute_data_loss, cast_to

def train_modelNN_precision(
    model, optimizer, training_data, validation_grid,
    epochs=20000, plots_dir='plots', final_dir='Heat2D/Results',
    show_plots_interactively=True, lr_strategy='fixed',
    precision_config: PrecisionConfig = None
):
    """
    Esegue il training della NN con precisione configurabile.
    """
    if precision_config is None:
        precision_config = PrecisionConfig()
        
    xy_train, T_train = training_data
    xy_grid, T_exact_grid, X, Y = validation_grid
    Nx_dom, Ny_dom = X.shape
    
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)
    
    pbar = tqdm(range(epochs), desc=f"Training NN Precision ({precision_config})")
    loss_history = TrainingHistory()
    
    scheduler = None
    if lr_strategy == 'step_decay':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(epochs * 0.25), gamma=0.5)

    for epoch in pbar:
        model.train()
        model.to(precision_config.optimizer)
        optimizer.zero_grad(set_to_none=True)
        
        # Data loss
        loss = compute_data_loss(model, xy_train, T_train, precision_config)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if scheduler: scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        loss_history.update(epoch, {'total_loss': loss.item()}, lr=current_lr)

        if (epoch + 1) % 500 == 0:
            pbar.set_postfix({'Loss': f"{loss.item():.2e}", 'LR': f"{current_lr:.1e}"})
            model.eval()
            model.to(torch.float64)
            with torch.no_grad():
                T_pred_grid = model(xy_grid.to(torch.float64)).reshape(Nx_dom, Ny_dom)
            plot_path = os.path.join(plots_dir, f'epoch_{epoch+1}.png')
            plot2D_comparison(X, Y, T_exact_grid, T_pred_grid, epoch+1, plot_path)

    model.eval()
    model.to(torch.float64)
    with torch.no_grad():
        T_final = model(xy_grid.to(torch.float64)).reshape(Nx_dom, Ny_dom)
    
    final_path = os.path.join(final_dir, 'NNfinal_result_precision.png')
    plot2D_final_result(X, Y, T_exact_grid, T_final, epochs, save_path=final_path, data_points=xy_train)
    
    return loss_history

# --- TRAINING FUNCTION ---
def train_modelNN(
    model,
    optimizer,
    training_data,
    validation_grid,
    epochs=20000,
    plots_dir='plots',
    final_dir='Heat2D/Results',
    show_plots_interactively=True,
    lr_strategy='fixed'
):
    """
    Esegue il training della NN.
    
    Args:
        model: Istanza del modello FCN.
        optimizer: Istanza dell'ottimizzatore.
        training_data: Tupla (xy_train, T_train).
        validation_grid: Tupla (xy_grid, T_exact_grid, X, Y).
                         X e Y servono per i plot e contengono la shape della griglia.
        show_plots_interactively: Booleano per controllare la visualizzazione interattiva dei plot.
        lr_strategy: Strategia di learning rate ('fixed' o 'step_decay').
    """
    
    # Unpack dei dati
    xy_train, T_train = training_data
    xy_grid, T_exact_grid, X, Y = validation_grid
    
    # Ricavo dimensioni griglia per reshape
    Nx_dom, Ny_dom = X.shape
    
    # Directory Output
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)
    
    plot_files = []
    
    # Training Loop
    pbar = tqdm(range(epochs), desc=f"Training NN ({lr_strategy})")
    loss_history = TrainingHistory() # Changed to TrainingHistory
    
    # Scheduler per il Learning Rate
    scheduler = None
    if lr_strategy == 'step_decay':
        # Decadimento lr ogni 25% delle epoche con gamma=0.5
        step_size = int(epochs * 0.25)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)
    elif lr_strategy == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=600, min_lr=1e-6, cooldown=3000)

    for epoch in pbar:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        
        pred = model(xy_train)
        loss = model.loss_fn(pred, T_train)
        
        loss.backward()
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Step dello scheduler
        if scheduler:
            if lr_strategy == 'plateau':
                scheduler.step(loss.item())
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        loss_history.update(epoch, {'total_loss': loss.item()}, lr=current_lr) # Changed to update method

        # Monitoraggio e Plotting periodico
        if (epoch + 1) % 500 == 0:
            pbar.set_postfix({'Loss': f"{loss.item():.2e}", 'LR': f"{current_lr:.1e}"})
            model.eval()
            with torch.no_grad():
                T_pred_grid = model(xy_grid).reshape(Nx_dom, Ny_dom)
                
            plot_path = os.path.join(plots_dir, f'epoch_{epoch+1}.png')
            plot2D_comparison(X, Y, T_exact_grid, T_pred_grid, epoch+1, plot_path)
            plot_files.append(plot_path)

    # Plot Finale Interattivo
    print("Training completato. Generazione plot finale...")
    model.eval()
    with torch.no_grad():
        T_final = model(xy_grid).reshape(Nx_dom, Ny_dom)
    
    # Salvataggio ultimo plot (Results)
    final_path = os.path.join(final_dir, 'NNfinal_result.png')
    plot2D_final_result(X, Y, T_exact_grid, T_final, epochs, save_path=final_path, data_points=xy_train)
    
    # Generazione GIF
    print(f"Creazione GIF con {len(plot_files)} frames...")
    if plot_files:
        gif_path = os.path.join(final_dir, 'NNtraining_evolution.gif')
        save_gif_PIL(gif_path, plot_files, fps=3, loop=1, delete_files=True)
    
    # Plot Loss History
    loss_history.plot_losses(
        save_path=os.path.join(final_dir, 'NNloss_history.png'), 
        experiment_name="Heat2D NN", 
        show_plot=show_plots_interactively,
        skip_epochs=50
    ) # Updated plot_losses call
    
    return loss_history