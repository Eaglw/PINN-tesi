import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from tqdm import tqdm

# Import function for GIF
# Adjust path to import graphic_func from the 'func' directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from func.graphic_func import save_gif_PIL, plot2D_comparison


# Configurazione dispositivo e precisione
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

# --- 1. DEFINIZIONE DEL PROBLEMA E SOLUZIONE ANALITICA ---
def soluzione_analitica(x, y, Lx=1.0, Ly=1.0, Nx=50):
    """
    Calcola la soluzione analitica per l'equazione di Laplace (stato stazionario calore).
    T=0 su y=0, y=Ly, x=0; T=1 su x=Lx.
    """
    T = torch.zeros_like(x)
    const_pi = torch.tensor(np.pi, device=x.device)
    
    for n in range(1, Nx + 1, 2):
        lambda_n = n * const_pi / Ly
        An = 4 / (n * const_pi)
        term = An * (torch.sinh(lambda_n * x) / torch.sinh(lambda_n * Lx)) * torch.sin(lambda_n * y)
        T += term
    return T

# --- 2. DEFINIZIONE DELLA RETE NEURALE ---
class FCN(nn.Module):
    """Rete Neurale a Connessioni Complete (Fully Connected Network)"""
    def __init__(self, layers, activation_fn=nn.Tanh):
        super().__init__()
        self.activation = activation_fn()
        self.fcs = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.fcs.append(nn.Linear(layers[i], layers[i+1]))

    def forward(self, x):
        for i, layer in enumerate(self.fcs):
            x = layer(x)
            if i < len(self.fcs) - 1: # Apply activation to all but the last layer
                x = self.activation(x)
        return x
    
    def loss_fn(self, pred, target):
        return nn.MSELoss()(pred, target)

# --- 3. TRAINING FUNCTION ---
def train_model(
    epochs=20000,
    lr=1e-3,
    layers=[2, 32, 32, 32, 1],
    Lx=1.0,
    Ly=1.0,
    Nx_dom=100,
    Ny_dom=100,
    num_train=500,
    plots_dir='plots',
    final_dir='Heat2D/Results'
):
    """
    Esegue il training della PINN per il problema 2D Heat.
    Incapsula la logica precedentemente nel main block.
    """
    
    # Directory Output
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)
    
    plot_files = []

    # Generazione Dati Training (Random Sampling)
    x_train = torch.rand(num_train, 1, device=device) * Lx
    y_train = torch.rand(num_train, 1, device=device) * Ly
    xy_train = torch.cat([x_train, y_train], dim=1)
    T_train = soluzione_analitica(x_train, y_train, Lx, Ly)

    # Generazione Griglia Validazione
    x_grid = torch.linspace(0, Lx, Nx_dom, device=device)
    y_grid = torch.linspace(0, Ly, Ny_dom, device=device)
    X, Y = torch.meshgrid(x_grid, y_grid, indexing='xy')
    xy_grid = torch.stack([X.flatten(), Y.flatten()], dim=1)
    T_exact_grid = soluzione_analitica(X, Y, Lx, Ly)

    # Inizializzazione Modello
    model = FCN(layers=layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
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
    final_path = os.path.join(final_dir, 'final_result.png')
    plot2D_comparison(X, Y, T_exact_grid, T_final, epochs, save_path=final_path)
    
    # Generazione GIF
    print(f"Creazione GIF con {len(plot_files)} frames...")
    if plot_files:
        gif_path = os.path.join(final_dir, 'training_evolution.gif')
        save_gif_PIL(gif_path, plot_files, fps=3, loop=1, delete_files=True)
    
    # Plot Loss History
    plt.figure(figsize=(6,4))
    plt.semilogy(loss_history)
    plt.title("Loss History")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True, which="both", ls="-")
    plt.show()