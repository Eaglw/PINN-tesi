"""
Experiment Goal: 1_NN_Grid
Description: NN Grid. Config: L2_80x6_1_E20000_GELU
"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from tqdm import tqdm

# Import function for GIF
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from func.graphic_func import save_gif_PIL, plot2D_comparison
from func.history_tracker import TrainingHistory

def train_modelNN_griglia(
    model,
    optimizer,
    training_data,
    validation_grid,
    epochs=20000,
    plots_dir='plots',
    final_dir='Heat2D/Results',
    show_plots_interactively=True
):
    """
    Esegue il training della NN utilizzando dati su griglia.
    
    Args:
        model: Istanza del modello FCN.
        optimizer: Istanza dell'ottimizzatore.
        training_data: Tupla (xy_train, T_train).
        validation_grid: Tupla (xy_grid, T_exact_grid, X, Y).
        show_plots_interactively: Booleano per controllare la visualizzazione interattiva dei plot.
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
    pbar = tqdm(range(epochs), desc="Training NN (Grid)")
    loss_history = TrainingHistory()
    
    # Scheduler per il Learning Rate (stesso di Heat2D_NN)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6000, gamma=0.4)

    for epoch in pbar:
        model.train()
        optimizer.zero_grad()
        
        pred = model(xy_train)
        loss = model.loss_fn(pred, T_train)
        
        loss.backward()
        optimizer.step()
        
        # Step dello scheduler
        scheduler.step()
        
        loss_history.update(epoch, {'total_loss': loss.item()})
        
        # Monitoraggio e Plotting periodico
        if (epoch + 1) % 500 == 0:
            current_lr = scheduler.get_last_lr()[0]
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
    final_path = os.path.join(final_dir, 'NN_Grid_final_result.png')
    plot2D_comparison(X, Y, T_exact_grid, T_final, epochs, save_path=final_path)
    
    # Generazione GIF
    print(f"Creazione GIF con {len(plot_files)} frames...")
    if plot_files:
        gif_path = os.path.join(final_dir, 'NN_Grid_training_evolution.gif')
        save_gif_PIL(gif_path, plot_files, fps=3, loop=1, delete_files=True)
    
    # Plot Loss History
    loss_history.plot_losses(save_path=os.path.join(final_dir, 'NN_Grid_loss_history.png'), experiment_name="Heat2D NN (Grid)", show_plot=show_plots_interactively)
    
    # Return history for comparison
    return loss_history
