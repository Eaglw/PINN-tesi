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
from func.history_tracker import TrainingHistory, compute_pinn_loss
# Import architecture and training function from Heat2D_NN
from Heat2D_NN import train_modelNN
from Heat2D_main import FCN

# Configurazione dispositivo e precisione
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)
print(f"Using device: {device} with default dtype: {torch.get_default_dtype()}")

show_plots_interactively = False
base_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(base_dir, 'Results/inverse')
os.makedirs(results_dir, exist_ok=True)
final_dir = results_dir
plots_dir = os.path.join(results_dir, 'plots')
os.makedirs(plots_dir, exist_ok=True)

# --------------------------------------------------------
# 0. Definizioni Funzioni Ausiliarie (Analitica e Fisica)
# --------------------------------------------------------

def soluzione_analitica_k_var(x, y, Lx=1.0, Ly=1.0, Nx=50, beta=1):
    """
    Calcola la soluzione analitica per conduzione stazionaria con
    conducibilità variabile esponenzialmente: k(x) = k0 * exp(beta * x).
    """
    T = torch.zeros_like(x)
    const_pi = torch.tensor(np.pi, device=x.device)
    
    for n in range(1, Nx + 1, 2):  # Solo n dispari per l'onda quadra
        lambda_n = n * const_pi / Ly
        An = 4 / (n * const_pi)  # Coefficienti Fourier dell'onda quadra (T=1)
        
        # Nuovi autovalori per la parte in x dovuti al k variabile
        delta = torch.sqrt(beta**2 + 4 * lambda_n**2)
        mu_n = delta / 2.0
        
        # Numeratore
        num = torch.exp(-beta * x / 2.0) * torch.sinh(mu_n * x)
        # Denominatore
        den = torch.exp(torch.tensor(-beta * Lx / 2.0, device=x.device)) * torch.sinh(mu_n * Lx)
        
        y_part = torch.sin(lambda_n * y)
        term = An * (num / den) * y_part
        T += term
        
    return T

def heat2d_variable_k_physics_loss(model, xy_p, beta):
    """
    Calcola il residuo dell'equazione del calore stazionario 2D con
    conducibilità k(x) ~ exp(beta * x).
    PDE: d2T/dx2 + d2T/dy2 + beta * dT/dx = 0
    """
    T = model(xy_p)
    
    grads = torch.autograd.grad(T, xy_p, torch.ones_like(T), create_graph=True)[0]
    dT_dx = grads[:, 0]
    dT_dy = grads[:, 1]
    
    grads2_x = torch.autograd.grad(dT_dx, xy_p, torch.ones_like(dT_dx), create_graph=True)[0]
    d2T_dx2 = grads2_x[:, 0]
    
    grads2_y = torch.autograd.grad(dT_dy, xy_p, torch.ones_like(dT_dy), create_graph=True)[0]
    d2T_dy2 = grads2_y[:, 1]
    
    res = d2T_dx2 + d2T_dy2 + beta * dT_dx
    return torch.mean(res**2)

# --------------------------------------------------------
# 1. Configurazione Parametri e Dati
# --------------------------------------------------------

# Parametri Generali
Lx, Ly = 1.0, 1.0
Nx_dom, Ny_dom = 50, 50
layers_config = [2, 50, 50, 50, 50, 1]

# Parametro "Reale" (Hidden) da scoprire
beta_true = 0.35

# 1. Generazione griglia 2D (Ground Truth)
x_grid = torch.linspace(0, Lx, Nx_dom, device=device)
y_grid = torch.linspace(0, Ly, Ny_dom, device=device)
X, Y = torch.meshgrid(x_grid, y_grid, indexing='xy')

# 2. Calcolo Target (Soluzione esatta con beta_true)
print(f"Generazione dati sintetici con beta_true = {beta_true}...")
T_grid = soluzione_analitica_k_var(X, Y, Lx, Ly, Nx=50, beta=beta_true)

# 3. Preparazione Dati Training (Flattening)
x_data = X.flatten().unsqueeze(1)
y_data = Y.flatten().unsqueeze(1)
T_data = T_grid.flatten().unsqueeze(1)
xy_train = torch.cat([x_data, y_data], dim=1)

print(f"Dataset di training: {xy_train.shape} punti")

# 4. Preparazione Punti Fisica (Collocation Points)
# Usiamo una griglia più densa o casuale per la fisica, o la stessa.
# Qui creo nuovi punti casuali nel dominio per variare rispetto ai nodi fissi
N_phys = 5000
# Creiamo il tensore senza gradienti inizialmente
xy_physics = torch.rand((N_phys, 2), device=device)
# Scaliamo alle dimensioni del dominio (operazione in-place sicura qui perché requires_grad=False)
xy_physics[:, 0] = xy_physics[:, 0] * Lx
xy_physics[:, 1] = xy_physics[:, 1] * Ly
# Ora attiviamo i gradienti per la fisica
xy_physics.requires_grad_(True)

# Estrazione Boundary Points (per loss BC, anche se inclusi nei dati)
# Filtriamo i punti sui bordi dai dati generati
mask_bc = (xy_train[:, 0] == 0) | (xy_train[:, 0] == Lx) | \
          (xy_train[:, 1] == 0) | (xy_train[:, 1] == Ly)
xy_bc = xy_train[mask_bc]
T_bc = T_data[mask_bc]

# --------------------------------------------------------
# 2. Fase 1: Pre-training su Dati (Regressione Pura)
# --------------------------------------------------------
print("\n=== FASE 1: Warmup NN sui dati (1000 epoche) ===")
print("Obiettivo: Inizializzare la rete su valori sensati prima di applicare la fisica.")

model_0 = FCN(layers=layers_config).to(device)
optimizer_0 = torch.optim.Adam(model_0.parameters(), lr=1e-3)

# Setup Tuple per funzione di training esistente
# Utilizziamo la griglia regolare (50x50) per il pre-training
xy_grid_flat = torch.stack([X.flatten(), Y.flatten()], dim=1)
T_grid_flat_data = T_grid.flatten().unsqueeze(1) # Target per la griglia piatta
training_data_NN = (xy_grid_flat, T_grid_flat_data)
validation_grid_tuple = (xy_grid_flat, T_grid, X, Y)

# Eseguiamo il pre-training (Warmup breve)
epochs_pretrain = 1000
train_modelNN(
    model=model_0,
    optimizer=optimizer_0,
    training_data=training_data_NN,
    validation_grid=validation_grid_tuple,
    epochs=epochs_pretrain,
    plots_dir=os.path.join(plots_dir, 'pretrain'),
    final_dir=final_dir,
    show_plots_interactively=show_plots_interactively 
)

# --------------------------------------------------------
# 3. Fase 2: Training Inverso (Recupero Beta)
# --------------------------------------------------------
print("\n=== FASE 2: Joint Training (Dati + Fisica + Beta) ===")
print("NOTA: Inizia il training vero e proprio. La fisica guiderà la rete verso")
print("      le derivate corrette e contemporaneamente stimerà Beta.")

# DIAGNOSTICA INIZIALE
print("\n--- DIAGNOSTICA DERIVATE ---")
loss_true = heat2d_variable_k_physics_loss(model_0, xy_physics, beta=beta_true)
print(f"Physics Loss con Beta VERO ({beta_true}): {loss_true.item():.6e}")
print("Nota: Un valore alto qui è normale post-warmup. Scenderà durante il Joint Training.")
print("----------------------------\n")


# Inizializzazione parametro learnable (guess iniziale sbagliato)
beta_guess = 0.0
beta_train = nn.Parameter(torch.tensor(beta_guess, dtype=torch.float64, device=device, requires_grad=True))

print(f"Beta vero: {beta_true}")
print(f"Beta iniziale (trainable): {beta_train.item()}")

# Optimizer: Rete e Parametro insieme
optimizer_full = torch.optim.Adam([
    {'params': model_0.parameters(), 'lr': 1e-4}, 
    {'params': [beta_train], 'lr': 1e-2}          
])

epochs_inverse = 20000
pbar = tqdm(range(epochs_inverse), desc="Inverse Training")
loss_history = TrainingHistory()
beta_history = []

# Pesi della Loss: Bilanciamo Dati e Fisica
lambda_data = 100.0   
lambda_bc = 0.0 # Incluso nei dati
lambda_physics = 1.0 

plot_files = []

for epoch in pbar:
    model_0.train() # Training attivo
    optimizer_full.zero_grad()
    
    # Creiamo una funzione di loss fisica parziale che usa il beta corrente
    def current_physics_loss(m, x):
        return heat2d_variable_k_physics_loss(m, x, beta=beta_train)
    
    # Calcolo Loss Completa 
    loss, loss_dict = compute_pinn_loss(
        model_0, 
        x_data=xy_train, 
        y_data=T_data,
        x_bc=xy_bc,
        y_bc=T_bc,
        physics_loss_fn=current_physics_loss, 
        x_physics=xy_physics,
        lambda_data=lambda_data,
        lambda_bc=lambda_bc,
        lambda_physics=lambda_physics
    )
    
    loss.backward()
    optimizer_full.step()
    
    # Salvataggio storia
    loss_history.update(epoch, loss_dict)
    beta_history.append(beta_train.item())
    
    # Logging
    if (epoch + 1) % 100 == 0:
        pbar.set_postfix({
            'PhysLoss': f"{loss_dict.get('physics_loss', 0):.2e}", 
            'DataLoss': f"{loss_dict.get('data_loss', 0):.2e}",
            'Beta': f"{beta_train.item():.4f}", 
            'ErrBeta%': f"{abs(beta_train.item() - beta_true)/beta_true*100:.1f}%"
        })

    # Plotting ogni 1000 epoche
    if (epoch + 1) % 1000 == 0:
        model_0.eval()
        with torch.no_grad():
            T_pred_grid = model_0(xy_grid_flat).reshape(Nx_dom, Ny_dom)
        
        plot_path = os.path.join(plots_dir, f'inverse_epoch_{epoch+1}.png')
        plot2D_comparison(X, Y, T_grid, T_pred_grid, epoch+1, plot_path, physics_points=xy_physics)
        plot_files.append(plot_path)

# --------------------------------------------------------
# 4. Output Finale e Analisi
# --------------------------------------------------------

print(f"\nTraining Completato.")
print(f"Beta Vero: {beta_true}")
print(f"Beta Stimato: {beta_train.item():.5f}")
err_perc = abs(beta_train.item() - beta_true)/beta_true * 100
print(f"Errore Percentuale: {err_perc:.2f}%")

# Plot andamento Beta
plt.figure(figsize=(10, 6))
plt.plot(beta_history, label='Estimated Beta', linewidth=2)
plt.axhline(y=beta_true, color='r', linestyle='--', label='True Beta')
plt.title(f'Parameter Estimation: Beta (Final Err: {err_perc:.2f}%)')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(final_dir, 'beta_history.png'))
if show_plots_interactively:
    plt.show()
else:
    plt.close()

# GIF Evoluzione Inversa
if plot_files:
    gif_path = os.path.join(final_dir, 'Inverse_Training_Evolution.gif')
    save_gif_PIL(gif_path, plot_files, fps=5, loop=1)

# Loss History
loss_history.plot_losses(save_path=os.path.join(final_dir, 'Inverse_Loss_History.png'), experiment_name="Inverse Heat2D")
