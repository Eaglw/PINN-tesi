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
goal = [0,5]
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
# Aumentiamo la capacità della rete: 4 hidden layers da 50 neuroni
layers_config = [2, 50, 50, 50, 50, 1]
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

# Griglia dominio per visualizzazione (Validazione)
Nx_dom, Ny_dom = 50, 50
x_grid = torch.linspace(0, Lx, Nx_dom, device=device)
y_grid = torch.linspace(0, Ly, Ny_dom, device=device)
X, Y = torch.meshgrid(x_grid, y_grid, indexing='xy')

# Calcolo soluzione esatta su griglia
T_grid = soluzione_analitica(X, Y, Lx, Ly, Nx=Nx_fourier)

# Preparazione dati griglia per training (appiattimento)
xy_grid_flat = torch.stack([X.flatten(), Y.flatten()], dim=1)
T_grid_flat = T_grid # T_grid serve anche flattened per il confronto, ma lo gestisce plot2D

# Estrazione dati randomici ma uniformi (Training Data)
num_data_internal = 1000  # Punti interni
num_data_boundary = 50    # Punti per ogni lato del bordo

torch.manual_seed(123)

# 1. Punti Interni
x_int = torch.rand(num_data_internal, 1, device=device) * Lx
y_int = torch.rand(num_data_internal, 1, device=device) * Ly

# 2. Punti al Bordo (Boundary Conditions)
# Lato Sinistro (x=0)
x_b_left = torch.zeros(num_data_boundary, 1, device=device)
y_b_left = torch.rand(num_data_boundary, 1, device=device) * Ly

# Lato Destro (x=Lx)
x_b_right = torch.ones(num_data_boundary, 1, device=device) * Lx
y_b_right = torch.rand(num_data_boundary, 1, device=device) * Ly

# Lato Inferiore (y=0)
x_b_bottom = torch.rand(num_data_boundary, 1, device=device) * Lx
y_b_bottom = torch.zeros(num_data_boundary, 1, device=device)

# Lato Superiore (y=Ly)
x_b_top = torch.rand(num_data_boundary, 1, device=device) * Lx
y_b_top = torch.ones(num_data_boundary, 1, device=device) * Ly

# Concatenazione di tutti i punti del Bordo
x_b_all = torch.cat([x_b_left, x_b_right, x_b_bottom, x_b_top], dim=0)
y_b_all = torch.cat([y_b_left, y_b_right, y_b_bottom, y_b_top], dim=0)

# -- Dati per NN classica (tutto insieme) --
x_data = torch.cat([x_int, x_b_all], dim=0)
y_data = torch.cat([y_int, y_b_all], dim=0)
T_data = soluzione_analitica(x_data, y_data, Lx, Ly, Nx=Nx_fourier)
xy_train = torch.cat([x_data, y_data], dim=1)

# -- Dati Separati per PINN --
xy_internal = torch.cat([x_int, y_int], dim=1)
T_internal = soluzione_analitica(x_int, y_int, Lx, Ly, Nx=Nx_fourier)

xy_boundary = torch.cat([x_b_all, y_b_all], dim=1)
T_boundary = soluzione_analitica(x_b_all, y_b_all, Lx, Ly, Nx=Nx_fourier)


plt.figure(figsize=(8,6))
cp = plt.contourf(X.cpu().numpy(), Y.cpu().numpy(), T_grid.cpu().numpy(), 50, cmap='inferno')
plt.colorbar(cp)
plt.scatter(x_data.cpu().numpy(), y_data.cpu().numpy(), c='cyan', s=10, edgecolor='k', linewidth=0.5, label='Dati Training (Interni + Bordi)')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('Soluzione analitica e punti dati estratti')
plt.legend()
plt.savefig(os.path.join(results_dir, 'analytic_sol.png'))
if show_plots_interactively:
    plt.show()
else:
    plt.close("all") 
# Setup Tuple Dati per train_model
training_data_NN = (xy_train, T_data)

# Passiamo T_grid (la griglia 2D completa) perché plot2D_comparison se l'aspetta
validation_grid_tuple = (xy_grid_flat, T_grid, X, Y) 

# Dizionari per salvare risultati per il confronto
histories = {}
final_models = {}

if 0 in goal:
    print("0. NN classica")
    from Heat2D_NN import train_modelNN
    # Inizializzazione Modello e Optimizer per questo caso
    model_0 = FCN(layers=layers_config).to(device)
    optimizer_0 = torch.optim.Adam(model_0.parameters(), lr=1e-3)
    
    history_0 = train_modelNN(
        model=model_0,
        optimizer=optimizer_0,
        training_data=training_data_NN,
        validation_grid=validation_grid_tuple,
        epochs=epochs,
        plots_dir=plots_dir,
        final_dir=final_dir,
        show_plots_interactively=show_plots_interactively 
    )
    histories['NN Random'] = history_0
    final_models['NN Random'] = model_0

if 1 in goal:
    print("1. PINN con dati e fisica")
    from Heat2D.Heat2D_PINN import train_modelPINN
    from Heat2D.physics import HeatEquation2D
    
    # Inizializzazione Fisica Modulare
    heat_physics = HeatEquation2D()
    
    # Inizializzazione Modello e Optimizer per questo caso
    model_1 = FCN(layers=layers_config).to(device)
    optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=1e-3)
    
    # Passiamo i dati separati alla PINN
    data_internal = (xy_internal, T_internal)
    data_boundary = (xy_boundary, T_boundary)

    train_modelPINN(
        model=model_1,
        optimizer=optimizer_1,
        data_internal=data_internal,
        data_boundary=data_boundary,
        validation_grid=validation_grid_tuple,
        physics_problem=heat_physics,
        epochs=epochs,
        plots_dir=plots_dir,
        final_dir=final_dir,
        show_plots_interactively=show_plots_interactively 
    )
    # PINN currently doesn't return history in the same clean way for comparison, 
    # but we can add it later if needed.

if 2 in goal:
    print("2. Problema inverso")

if 5 in goal:
    print("5. NN classica su griglia")
    from Heat2D.Heat2D_NN_griglia import train_modelNN_griglia
    
    # Generazione dati su griglia per il training
    # Cerchiamo di avere un numero di punti simile al caso random (1000 interni + 200 bordo = 1200)
    # sqrt(1200) ~ 34.6. Usiamo 35x35 = 1225 punti.
    Nx_train, Ny_train = 35, 35
    x_train_line = torch.linspace(0, Lx, Nx_train, device=device)
    y_train_line = torch.linspace(0, Ly, Ny_train, device=device)
    X_train, Y_train = torch.meshgrid(x_train_line, y_train_line, indexing='xy')
    
    # Appiattimento
    xy_train_grid = torch.stack([X_train.flatten(), Y_train.flatten()], dim=1)
    
    # Calcolo target analitico
    T_train_grid = soluzione_analitica(X_train.flatten().unsqueeze(1), Y_train.flatten().unsqueeze(1), Lx, Ly, Nx=Nx_fourier)
    
    training_data_grid = (xy_train_grid, T_train_grid)
    
    # Inizializzazione Modello e Optimizer
    model_5 = FCN(layers=layers_config).to(device)
    optimizer_5 = torch.optim.Adam(model_5.parameters(), lr=1e-3)
    
    history_5 = train_modelNN_griglia(
        model=model_5,
        optimizer=optimizer_5,
        training_data=training_data_grid,
        validation_grid=validation_grid_tuple,
        epochs=epochs,
        plots_dir=plots_dir,
        final_dir=final_dir,
        show_plots_interactively=show_plots_interactively
    )
    histories['NN Grid'] = history_5
    final_models['NN Grid'] = model_5

# --- COMPARISON LOGIC ---
if 0 in goal and 5 in goal:
    print("\n--- Generating Comparison: Random vs Grid ---")
    from func.graphic_func import plot_loss_comparison, plot_error_map_comparison
    
    # 1. Loss Comparison
    plot_loss_comparison(
        [histories['NN Random'], histories['NN Grid']],
        ['NN Random', 'NN Grid'],
        save_path=os.path.join(results_dir, 'Comparison_Loss_Random_vs_Grid.png')
    )
    
    # 2. Error Map Comparison
    model_random = final_models['NN Random']
    model_grid = final_models['NN Grid']
    
    model_random.eval()
    model_grid.eval()
    
    with torch.no_grad():
        # Recalculate predictions on validation grid
        pred_random = model_random(xy_grid_flat).reshape(Nx_dom, Ny_dom)
        pred_grid = model_grid(xy_grid_flat).reshape(Nx_dom, Ny_dom)
        
    plot_error_map_comparison(
        X, Y, T_grid,
        [pred_random, pred_grid],
        ['NN Random', 'NN Grid'],
        save_path=os.path.join(results_dir, 'Comparison_ErrorMap_Random_vs_Grid.png')
    )
    print("Comparison plots saved in Results/.")
    

