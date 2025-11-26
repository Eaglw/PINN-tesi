import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from tqdm import tqdm

# Import function for GIF
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from func.graphic_func import save_gif_PIL, DHeat_plot_comparison

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
    
# --- 3. FUNZIONI UTILITY PER PLOTTING ---
# La funzione plot_comparison è stata spostata in func.graphic_func come 2DHeat_plot_comparison




# --- intro mio --- 

# Directory Output
plots_dir = 'plots'
os.makedirs(plots_dir, exist_ok=True)
final_dir = '2DHeat/Results'
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



# Directory per i plot dell'errore durante il training
error_plot_dir = os.path.join(results_dir, 'training_error')
if not os.path.exists(error_plot_dir):
    os.makedirs(error_plot_dir)

pbar = tqdm(range(epochs), desc='Training NN')
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

final_path = os.path.join(final_dir, 'final_result.png')
DHeat_plot_comparison(X, Y, T_grid, T_pred_final, epochs, save_path=final_path)





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
            DHeat_plot_comparison(X, Y, T_exact_grid, T_pred_grid, epoch+1, plot_path)
            plot_files.append(plot_path)

    # Plot Finale Interattivo
    print("Training completato. Generazione plot finale...")
    model.eval()
    with torch.no_grad():
        T_final = model(xy_grid).reshape(Nx_dom, Ny_dom)
    
    # Salvataggio ultimo plot (Results)
    final_path = os.path.join(final_dir, 'final_result.png')
    DHeat_plot_comparison(X, Y, T_exact_grid, T_final, epochs, save_path=final_path)
    
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
