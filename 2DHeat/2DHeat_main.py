import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

def soluzione_analitica(x, y, Lx, Ly, Nx=50):
    """
    Soluzione analitica della temperatura in una lastra rettangolare 2D
    con Dirichlet: T=0 su y=0, y=Ly, x=0; T=1 su x=Lx.
    x, y possono essere tensori di eguali dimensioni.
    """
    T = torch.zeros_like(x, dtype=torch.float64)
    for n in range(1, Nx + 1, 2):  # Itera solo sui numeri dispari
        lambda_n = n * torch.pi / Ly
        An = 4 / (n * torch.pi)
        T += An * torch.sinh(lambda_n * x) / torch.sinh(torch.tensor(lambda_n * Lx, dtype=torch.float64)) * torch.sin(lambda_n * y)
    return T

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
    

# Parametri dominio
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
results_dir = '2DHeat/Results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

plt.figure(figsize=(8,6))
cp = plt.contourf(X.numpy(), Y.numpy(), T_grid.numpy(), 50, cmap='inferno')
plt.colorbar(cp)
plt.scatter(x_data.numpy(), y_data.numpy(), c='cyan', s=21, edgecolor='k', label='Dati estratti')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('Soluzione analitica e punti dati estratti (PyTorch)')
plt.legend()
plt.savefig(os.path.join(results_dir, 'soluzione_analitica.png'))
plt.show()