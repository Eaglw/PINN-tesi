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

# ---  DEFINIZIONE DELLA LOSS FISICA ---
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
    final_dir='Heat2D/Results',
    show_plots_interactively=True
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
    
    # Generazione punti di collocazione: griglia regolare
    Nx_phys, Ny_phys = 50, 50 # Numero di punti per dimensione
    x_phys_line = torch.linspace(0, Lx, Nx_phys, device=device)
    y_phys_line = torch.linspace(0, Ly, Ny_phys, device=device)
    X_phys, Y_phys = torch.meshgrid(x_phys_line, y_phys_line, indexing='xy')
    xy_physics = torch.stack([X_phys.flatten(), Y_phys.flatten()], dim=1)
    xy_physics.requires_grad_(True)
    
    # Training Loop
    pbar = tqdm(range(epochs), desc="Training PINN")
    loss_history = TrainingHistory()
    
    # Pesi per bilanciare le componenti della loss
    # Ridurre il peso della fisica aiuta spesso la convergenza iniziale
    lambda_data = 1.0
    lambda_physics = 0.1 
    
    # Scheduler per il Learning Rate
    # Decadimento lr ogni 6000 epoche con gamma=0.4
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6000, gamma=0.4)

    for epoch in pbar:
        
        model.train()
        optimizer.zero_grad()
        
        # Calcolo loss usando la funzione generica con i pesi
        loss, loss_dict = compute_pinn_loss(
            model, 
            xy_train, 
            T_train, 
            physics_loss_fn=heat2d_physics_loss, 
            x_physics=xy_physics,
            lambda_data=lambda_data,
            lambda_physics=lambda_physics
        )
        
        loss.backward()
        optimizer.step()
        
        # Step dello scheduler
        scheduler.step()
        
        # Aggiornamento history
        loss_history.update(epoch, loss_dict)
        
        # Monitoraggio e Plotting periodico
        if (epoch + 1) % 500 == 0:
            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({'Loss': f"{loss.item():.2e}", 'LR': f"{current_lr:.1e}"})
            
            model.eval()
            with torch.no_grad():
                T_pred_grid = model(xy_grid).reshape(Nx_dom, Ny_dom)
                
            plot_path = os.path.join(plots_dir, f'epoch_{epoch+1}.png')
            plot2D_comparison(X, Y, T_exact_grid, T_pred_grid, epoch+1, plot_path, physics_points=xy_physics)
            plot_files.append(plot_path)

    # Plot Finale Interattivo
    print("Training completato. Generazione plot finale...")
    model.eval()
    with torch.no_grad():
        T_final = model(xy_grid).reshape(Nx_dom, Ny_dom)
    
    # Salvataggio ultimo plot (Results)
    final_path = os.path.join(final_dir, 'PINNfinal_result.png')
    plot2D_comparison(X, Y, T_exact_grid, T_final, epochs, save_path=final_path, physics_points=xy_physics)
    
    # Generazione GIF
    print(f"Creazione GIF con {len(plot_files)} frames...")
    if plot_files:
        gif_path = os.path.join(final_dir, 'PINNtraining_evolution.gif')
        save_gif_PIL(gif_path, plot_files, fps=3, loop=1, delete_files=True)
    
    # Plot Loss History
    loss_history.plot_losses(save_path=os.path.join(final_dir, 'PINNloss_history.png'), experiment_name="Heat2D PINN", show_plot=show_plots_interactively)
    if show_plots_interactively:
        plt.show()
    else:
        plt.close("all")