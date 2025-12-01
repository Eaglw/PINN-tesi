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

# Import training logic
try:
    from Heat2D_NN import train_model
except ImportError:
    # Fallback if running from root as module
    from .Heat2D_NN import train_model

# Configurazione dispositivo e precisione
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)


step=4000 #step di training condivisi tra i try per comparare 
"""
Seleziona quali casi eseguire inserendo nell'array goal il corrispettivo numero
0. NN classica e PINN con dati e fisica
1. Solo fisica e BC
2. Problema inverso
3. PINN che confronta l'andamento di diversi optimizer e activation function
"""
goal = [0]


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
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS, activation_fn=nn.Tanh):
        super().__init__()
        self.activation = activation_fn()
        self.fcs = nn.Linear(N_INPUT, N_HIDDEN)
        self.fch = nn.ModuleList([nn.Linear(N_HIDDEN, N_HIDDEN) for _ in range(N_LAYERS - 1)])
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)
        
    def forward(self, x):
        x = self.fcs(x)
        x = self.activation(x)
        for layer in self.fch:
            x = layer(x)
            x = self.activation(x)
        x = self.fce(x)
        return x
    
# --- intro mio --- 

# Directory Output
plots_dir = 'plots'
os.makedirs(plots_dir, exist_ok=True)
final_dir = 'Heat2D/Results'
os.makedirs(final_dir, exist_ok=True)
 
# Parametri 
epochs = 5000
Lx, Ly = 1.0, 1.0
Nx_fourier = 50  # termini serie

# Griglia dominio per visualizzazione
Nx_dom, Ny_dom = 100, 100
x_grid = torch.linspace(0, Lx, Nx_dom, dtype=torch.float64)
y_grid = torch.linspace(0, Ly, Ny_dom, dtype=torch.float64)
X, Y = torch.meshgrid(x_grid, y_grid, indexing='xy')

T_grid = soluzione_analitica(X, Y, Lx, Ly, Nx=Nx_fourier)

# Estrazione dati randomici ma uniformi
num_data = 200  # cambia a piacere
torch.manual_seed(0)

x_data = torch.rand(num_data, dtype=torch.float64) * Lx
y_data = torch.rand(num_data, dtype=torch.float64) * Ly
T_data = soluzione_analitica(x_data, y_data, Lx, Ly, Nx=Nx_fourier)

# Plot
results_dir = 'Heat2D/Results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

plt.figure(figsize=(8,6))
cp = plt.contourf(X.numpy(), Y.numpy(), T_grid.numpy(), 50, cmap='inferno')
plt.colorbar(cp)
plt.scatter(x_data.numpy(), y_data.numpy(), c='cyan', s=21, edgecolor='k', label='Dati estratti')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('Soluzione analitica e punti dati estratti')
plt.legend()
plt.savefig(os.path.join(results_dir, 'analytic_sol.png'))
plt.show()


if 0 in goal:
    print("0. NN classica e PINN con dati e fisica")
    train_model()
if 1 in goal:
    print("1. Solo fisica e BC")
    train_model()
if 2 in goal:
    print("2. Problema inverso")
    train_model()
if 3 in goal:
    print("3. Analisi ottimizzatori e funzioni di attivazione")
    train_model()
