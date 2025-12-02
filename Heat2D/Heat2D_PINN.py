import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from tqdm import tqdm

# Import function for GIF and loss comparison
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from func.graphic_func import save_gif_PIL, plot2D_comparison
from func.history_tracker import TrainingHistory, compute_pinn_loss
# Configurazione dispositivo e precisione
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

def heat2d_physics_loss(model, xy_p):
    """
    Calcola il residuo dell'equazione di Laplace 2D: d2T/dx2 + d2T/dy2 = 0
    """
    # xy_p è (N, 2). Richiediamo gradienti.
    T = model(xy_p)
    
    # Calcolo gradienti primi
    grads = torch.autograd.grad(T, xy_p, torch.ones_like(T), create_graph=True)[0]
    dT_dx = grads[:, 0]
    dT_dy = grads[:, 1]
    
    # Calcolo gradienti secondi
    # Nota: autograd.grad restituisce una tupla, prendiamo [0]
    grads2_x = torch.autograd.grad(dT_dx, xy_p, torch.ones_like(dT_dx), create_graph=True)[0]
    d2T_dx2 = grads2_x[:, 0]
    
    grads2_y = torch.autograd.grad(dT_dy, xy_p, torch.ones_like(dT_dy), create_graph=True)[0]
    d2T_dy2 = grads2_y[:, 1]
    
    # Residuo PDE
    res = d2T_dx2 + d2T_dy2
    return torch.mean(res**2)

def train_modelPINN(
    model,
    optimizer,
    training_data,
    validation_grid,
    epochs=20000,
    plots_dir='plots',
    final_dir='Heat2D/Results'
):
    """
    Esegue il training della PINN.
    
    Args:
        model: Istanza del modello FCN.
        optimizer: Istanza dell'ottimizzatore.
        training_data: Tupla (xy_train, T_train).
        validation_grid: Tupla (xy_grid, T_exact_grid, X, Y).
                         X e Y servono per i plot e contengono la shape della griglia.
    """
    
    # Unpack dei dati
    xy_train, T_train = training_data
    xy_grid, T_exact_grid, X, Y = validation_grid
    
    # Ricavo dimensioni griglia per reshape e limiti dominio
    Nx_dom, Ny_dom = X.shape
    Lx = X.max().item()
    Ly = Y.max().item()
    
    # Directory Output
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)
    
    plot_files = []
    
    # Generazione punti di collocazione fissi (o dinamici nel loop)
    # Per ora usiamo 2000 punti casuali nel dominio
    num_physics = 2000
    xy_physics = torch.rand(num_physics, 2, device=device)
    xy_physics[:, 0] = xy_physics[:, 0] * Lx
    xy_physics[:, 1] = xy_physics[:, 1] * Ly
    xy_physics.requires_grad_(True)
    
    # Training Loop
    pbar = tqdm(range(epochs), desc="Training PINN")
    loss_history = TrainingHistory()

    for epoch in pbar:
        
        model.train()
        optimizer.zero_grad()
        
        # Calcolo loss usando la funzione generica
        loss, loss_dict = compute_pinn_loss(
            model, 
            xy_train, 
            T_train, 
            physics_loss_fn=heat2d_physics_loss, 
            x_physics=xy_physics
        )
        
        loss.backward()
        optimizer.step()
        
        # Aggiornamento history
        loss_history.update(epoch, loss_dict)
        
        # Monitoraggio e Plotting periodico
        if (epoch + 1) % 500 == 0:
            pbar.set_postfix({'Loss': f"{loss.item():.2e}"})
            
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
    final_path = os.path.join(final_dir, 'PINNfinal_result.png')
    plot2D_comparison(X, Y, T_exact_grid, T_final, epochs, save_path=final_path)
    
    # Generazione GIF
    print(f"Creazione GIF con {len(plot_files)} frames...")
    if plot_files:
        gif_path = os.path.join(final_dir, 'PINNtraining_evolution.gif')
        save_gif_PIL(gif_path, plot_files, fps=3, loop=1, delete_files=True)
    
    # Plot Loss History
    loss_history.plot_losses(save_path=os.path.join(final_dir, 'PINNloss_history.png'), experiment_name="Heat2D PINN")
    plt.show()