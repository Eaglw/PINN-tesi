import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from tqdm import tqdm

# Import function for GIF
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from func.graphic_func import save_gif_PIL

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
    def __init__(self, layers=[2, 64, 64, 64, 64, 64, 1]):
        super().__init__()
        self.activation = nn.Tanh()
        self.loss_fn = nn.MSELoss()
        
        # Creazione dinamica dei layer
        module_list = []
        for i in range(len(layers) - 1):
            module_list.append(nn.Linear(layers[i], layers[i+1]))
        self.layers = nn.ModuleList(module_list)
        
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1: # Attivazione su tutti tranne l'ultimo
                x = self.activation(x)
        return x

# --- 3. FUNZIONI UTILITY PER PLOTTING ---
def plot_comparison(X, Y, T_true, T_pred, epoch, save_path):
    """Genera grafici side-by-side: Predizione, Errore Assoluto, Errore Relativo."""
    
    # Calcolo Errori
    abs_error = torch.abs(T_pred - T_true)
    
    # Errore Relativo (Gestione della divisione per zero)
    # Calcoliamo l'errore relativo solo dove T_true è significativo (> 0.01)
    mask = torch.abs(T_true) > 0.01
    rel_error = torch.zeros_like(T_true)
    rel_error[mask] = (abs_error[mask] / torch.abs(T_true[mask])) * 100
    
    # Setup plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    X_np, Y_np = X.cpu().numpy(), Y.cpu().numpy()
    
    # 1. Soluzione Predetta
    ax = axes[0]
    c1 = ax.contourf(X_np, Y_np, T_pred.detach().cpu().numpy(), levels=50, cmap='inferno')
    plt.colorbar(c1, ax=ax, label='Temp')
    ax.set_title(f'Predizione NN (Epoch {epoch})')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # 2. Errore Assoluto (Più robusto)
    ax = axes[1]
    c2 = ax.contourf(X_np, Y_np, abs_error.detach().cpu().numpy(), levels=50, cmap='magma')
    plt.colorbar(c2, ax=ax, label='Errore Assoluto')
    ax.set_title('Errore Assoluto |T_pred - T_true|')
    ax.set_xlabel('x')

    # 3. Errore Relativo (Mascherato)
    ax = axes[2]
    # Usiamo vmin/vmax per evitare saturazione da outlier
    c3 = ax.contourf(X_np, Y_np, rel_error.detach().cpu().numpy(), levels=50, cmap='jet', vmin=0, vmax=10) 
    cbar = plt.colorbar(c3, ax=ax, label='% Errore')
    # Coloriamo di grigio le zone escluse (dove T_true ~ 0)
    ax.set_facecolor('lightgray') 
    ax.set_title('Errore Relativo % (dove T_true > 0.01)')
    ax.set_xlabel('x')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

# --- 4. MAIN SCRIPT ---
if __name__ == "__main__":
    # Parametri
    Lx, Ly = 1.0, 1.0
    Nx_dom, Ny_dom = 100, 100
    epochs = 20000  # Ridotto per test rapido, aumenta a piacere
    
    # Directory Output
    plots_dir = 'plots'
    os.makedirs(plots_dir, exist_ok=True)
    final_dir = '2DHeat/Results'
    os.makedirs(final_dir, exist_ok=True)
    
    plot_files = []

    # Generazione Dati Training (Random Sampling)
    num_train = 500
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
    model = FCN(layers=[2, 32, 32, 32, 1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
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
            plot_comparison(X, Y, T_exact_grid, T_pred_grid, epoch+1, plot_path)
            plot_files.append(plot_path)

    # Plot Finale Interattivo
    print("Training completato. Generazione plot finale...")
    model.eval()
    with torch.no_grad():
        T_final = model(xy_grid).reshape(Nx_dom, Ny_dom)
    
    # Salvataggio ultimo plot (Results)
    final_path = os.path.join(final_dir, 'final_result.png')
    plot_comparison(X, Y, T_exact_grid, T_final, epochs, save_path=final_path)
    
    # Generazione GIF
    print(f"Creazione GIF con {len(plot_files)} frames...")
    if plot_files:
        gif_path = os.path.join(final_dir, 'training_evolution.gif')
        save_gif_PIL(gif_path, plot_files, fps=3, loop=1,delete_files=True)
    
    # Plot Loss History
    plt.figure(figsize=(6,4))
    plt.semilogy(loss_history)
    plt.title("Loss History")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True, which="both", ls="-")
    plt.show()
