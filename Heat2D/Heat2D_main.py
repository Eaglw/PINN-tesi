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
print(f"Using device: {device} with default dtype: {torch.get_default_dtype()}")



# Flag per controllare la visualizzazione interattiva dei plot
show_plots_interactively = False # Imposta su False per eseguire lo script senza blocchi
"""
Seleziona quali casi eseguire inserendo nell'array goal il corrispettivo numero
0. NN classica
1. PINN con dati e fisica
2. Solo fisica e BC
3. Problema inverso
4. PINN che confronta l'andamento di diversi optimizer e activation function
"""
goal = [0,1]
# Directory Output
base_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(base_dir, 'Results')
os.makedirs(results_dir, exist_ok=True)
final_dir = results_dir
plots_dir = os.path.join(results_dir, 'plots')
os.makedirs(plots_dir, exist_ok=True)
# Parametri 
epochs = 30000
Lx, Ly = 1.0, 1.0
Nx_fourier = 50  # termini serie
# --- Setup comune Training ---
# Definizione layer modello: [Input, Hidden..., Output]
# Corrisponde a: Input=2, Hidden=32 (3 layer), Output=1
layers_config = [2, 32, 32, 32, 1]
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
    def __init__(self, layers, activation_fn=nn.GELU):
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

# Griglia dominio per visualizzazione (Validazione)
Nx_dom, Ny_dom = 100, 100
x_grid = torch.linspace(0, Lx, Nx_dom, device=device)
y_grid = torch.linspace(0, Ly, Ny_dom, device=device)
X, Y = torch.meshgrid(x_grid, y_grid, indexing='xy')

# Calcolo soluzione esatta su griglia
T_grid = soluzione_analitica(X, Y, Lx, Ly, Nx=Nx_fourier)

# Preparazione dati griglia per training (appiattimento)
xy_grid_flat = torch.stack([X.flatten(), Y.flatten()], dim=1)
T_grid_flat = T_grid # T_grid serve anche flattened per il confronto, ma lo gestisce plot2D

# Estrazione dati randomici ma uniformi (Training Data)
num_data = 300  # cambia a piacere
torch.manual_seed(1)
x_data = torch.rand(num_data, 1, device=device) * Lx
y_data = torch.rand(num_data, 1, device=device) * Ly

# Calcolo target training
T_data = soluzione_analitica(x_data, y_data, Lx, Ly, Nx=Nx_fourier)

# Concatenazione input training
xy_train = torch.cat([x_data, y_data], dim=1)
plt.figure(figsize=(8,6))
cp = plt.contourf(X.cpu().numpy(), Y.cpu().numpy(), T_grid.cpu().numpy(), 50, cmap='inferno')
plt.colorbar(cp)
plt.scatter(x_data.cpu().numpy(), y_data.cpu().numpy(), c='cyan', s=21, edgecolor='k', label='Dati estratti')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('Soluzione analitica e punti dati estratti')
plt.legend()
plt.savefig(os.path.join(results_dir, 'analytic_sol.png'))
if show_plots_interactively:
    plt.show()
else:
    plt.close("all") # Chiude la figura per evitare che rimanga in memoria
# Setup Tuple Dati per train_model
training_data_tuple = (xy_train, T_data)
# Passiamo T_grid (la griglia 2D completa) perché plot2D_comparison se l'aspetta
validation_grid_tuple = (xy_grid_flat, T_grid, X, Y) 
if 0 in goal:
    print("0. NN classica")
    from Heat2D_NN import train_modelNN
    # Inizializzazione Modello e Optimizer per questo caso
    model_0 = FCN(layers=layers_config).to(device)
    optimizer_0 = torch.optim.Adam(model_0.parameters(), lr=1e-3)
    
    train_modelNN(
        model=model_0,
        optimizer=optimizer_0,
        training_data=training_data_tuple,
        validation_grid=validation_grid_tuple,
        epochs=epochs,
        plots_dir=plots_dir,
        final_dir=final_dir,
        show_plots_interactively=show_plots_interactively 
    )

if 1 in goal:
    print("1. PINN con dati e fisica")
    from Heat2D_PINN import train_modelPINN
    # Inizializzazione Modello e Optimizer per questo caso
    model_0 = FCN(layers=layers_config).to(device)
    optimizer_0 = torch.optim.Adam(model_0.parameters(), lr=1e-3)
    
    train_modelPINN(
        model=model_0,
        optimizer=optimizer_0,
        training_data=training_data_tuple,
        validation_grid=validation_grid_tuple,
        epochs=epochs,
        plots_dir=plots_dir,
        final_dir=final_dir,
        show_plots_interactively=show_plots_interactively 
    )
if 2 in goal:
    print("2. Problema inverso")
    pass

if 3 in goal:
    print("3. Analisi ottimizzatori e funzioni di attivazione")
    pass
