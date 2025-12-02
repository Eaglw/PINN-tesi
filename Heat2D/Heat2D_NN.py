import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from tqdm import tqdm

# Import function for GIF
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from func.graphic_func import save_gif_PIL, plot2D_comparison

# Configurazione dispositivo e precisione
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

# --- TRAINING FUNCTION ---
def train_modelNN(
    model,
    optimizer,
    training_data,
    validation_grid,
    epochs=20000,
    plots_dir='plots',
    final_dir='Heat2D/Results'
):
    """
    Esegue il training della NN.
    
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
    
    # Ricavo dimensioni griglia per reshape
    Nx_dom, Ny_dom = X.shape
    
    # Directory Output
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)
    
    plot_files = []
    
    # Training Loop
    pbar = tqdm(range(epochs), desc="Training")
    loss_history = []

    for epoch in pbar:
        model.train()
        optimizer.zero_grad()
        
        pred = model(xy_train)
        loss = model.loss_fn(pred, T_train)
        
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
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
    final_path = os.path.join(final_dir, 'NNfinal_result.png')
    plot2D_comparison(X, Y, T_exact_grid, T_final, epochs, save_path=final_path)
    
    # Generazione GIF
    print(f"Creazione GIF con {len(plot_files)} frames...")
    if plot_files:
        gif_path = os.path.join(final_dir, 'NNtraining_evolution.gif')
        save_gif_PIL(gif_path, plot_files, fps=3, loop=1, delete_files=True)
    
    # Plot Loss History
    plt.figure(figsize=(6,4))
    plt.semilogy(loss_history)
    plt.title("Loss History")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True, which="both", ls="-")
    plt.show()