from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from func.graphic_func import save_gif_PIL, plot_result

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
plt.title('Soluzione analitica e punti dati estratti')
plt.legend()
plt.savefig(os.path.join(results_dir, 'analytic_sol.png'))
plt.show()


# Training di NN normale
torch.manual_seed(42)
model = FCN(2, 1, 32, 4).to(torch.float64)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# Prepara i dati per il training
xy_data = torch.stack([x_data, y_data], dim=1)
T_data_reshaped = T_data.unsqueeze(1) # Aggiunge una dimensione per il target

# Prepara la griglia per la visualizzazione
xy_grid = torch.stack([X.flatten(), Y.flatten()], dim=1)

epochs = 5000
pbar = tqdm(range(epochs), desc='Training NN')

# Directory per i plot dell'errore durante il training
error_plot_dir = os.path.join(results_dir, 'training_error')
if not os.path.exists(error_plot_dir):
    os.makedirs(error_plot_dir)

for epoch in pbar:
    model.train()
    # Forward pass
    T_pred = model(xy_data)
    loss = loss_fn(T_pred, T_data_reshaped)
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 100 == 0:
        pbar.set_postfix({'Loss': f'{loss.item():.4e}'})
        
        # Plotting dell'errore relativo
        model.eval()
        with torch.no_grad():
            T_pred_grid = model(xy_grid).reshape(Nx_dom, Ny_dom)
        
        # Calcolo errore relativo percentuale
        relative_error = 100 * torch.abs(T_pred_grid - T_grid) / (torch.abs(T_grid) + 1e-8)
        
        plt.figure(figsize=(8, 6))
        # vmin=0, vmax=100 mappa l'intervallo [0, 100] (0%-100%) sulla colormap
        cp = plt.contourf(X.numpy(), Y.numpy(), relative_error.numpy(), 50, cmap='coolwarm')#, vmin=0, vmax=100)
        cbar = plt.colorbar(cp, label='Errore Relativo Percentuale')
        cbar.set_ticks(np.linspace(0, 100, 11))
        cbar.set_ticklabels([f'{i:.0f}%' for i in np.linspace(0, 100, 11)])
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.title(f'Errore Relativo Percentuale - Epoch {epoch + 1}')
        plt.savefig(os.path.join(error_plot_dir, f'relative_error_epoch_{epoch+1}.png'))
        plt.close() # Chiude la figura per non mostrarla a schermo

# Plot finale dopo il training
model.eval()
with torch.no_grad():
    T_pred_final = model(xy_grid).reshape(Nx_dom, Ny_dom)

# Plot della soluzione appresa
plt.figure(figsize=(10, 7))
cp = plt.contourf(X.numpy(), Y.numpy(), T_pred_final.numpy(), 50, cmap='inferno')
plt.colorbar(cp, label='Temperatura')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('Soluzione Appresa dalla Rete Neurale')
plt.savefig(os.path.join(results_dir, 'learned_solution.png'))
plt.show()

# Plot dell'errore relativo finale
relative_error_final = 100*torch.abs(T_pred_final - T_grid) / (torch.abs(T_grid) + 1e-8)
plt.figure(figsize=(10, 7))
cp = plt.contourf(X.numpy(), Y.numpy(), relative_error_final.numpy(), 100, cmap='coolwarm', vmin=0, vmax=100)
cbar = plt.colorbar(cp, label='Errore Relativo Percentuale')
cbar.set_ticks(np.linspace(0, 100, 11))
cbar.set_ticklabels([f'{i:.0f}%' for i in np.linspace(0, 100, 11)])
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('Errore Relativo Percentuale Finale vs Soluzione Analitica')
plt.savefig(os.path.join(results_dir, 'final_relative_error.png'))
plt.show()


