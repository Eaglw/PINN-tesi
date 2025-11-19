# Questo script esegue un'analisi incrociata di ottimizzatori e funzioni di attivazione
# per il problema del reattore CSTR in configurazione "forward" (k è noto).
# I risultati, inclusi i grafici delle loss e delle predizioni finali,
# vengono salvati automaticamente nella cartella 'plots/CSTR/'.

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import sys

# Assicura che i moduli nella cartella 'func' siano importabili
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from func.history_tracker import TrainingHistory
from func.graphic_func import plot_result


# --- CONFIGURAZIONE DEGLI ESPERIMENTI ---
experiments_to_run = [
    {'name': 'Adam_Tanh_10k', 'optimizer': 'Adam', 'activation': 'Tanh', 'learning_rate': 1e-3, 'epochs': 10000},
    {'name': 'Adam_GELU_10k', 'optimizer': 'Adam', 'activation': 'GELU', 'learning_rate': 1e-3, 'epochs': 10000},
    {'name': 'LBFGS_Tanh_10k', 'optimizer': 'LBFGS', 'activation': 'Tanh', 'learning_rate': 1.0, 'epochs': 10000}, # Per LBFGS, le "epoche" sono gestite da max_iter
]

# --- PARAMETRI E DEFINIZIONI DEL MODELLO ---

x_physics = torch.linspace(0, 5, 30).view(-1, 1).requires_grad_(True)

# Mappatura delle funzioni di attivazione
activation_functions = {
    'Tanh': nn.Tanh,
    'GELU': nn.GELU,
    'SiLU': nn.SiLU,
    'ReLU': nn.ReLU
}

class FCN(nn.Module):
    """Rete Neurale a Connessioni Complete (Fully Connected Network)"""
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS, activation_fn=nn.Tanh):
        super().__init__()
        self.fcs = nn.Sequential(nn.Linear(N_INPUT, N_HIDDEN), activation_fn())
        self.fch = nn.Sequential(*[
            nn.Sequential(nn.Linear(N_HIDDEN, N_HIDDEN), activation_fn()) for _ in range(N_LAYERS - 1)
        ])
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)
        
    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x





# --- CICLO PRINCIPALE DEGLI ESPERIMENTI ---
for experiment in experiments_to_run:
    exp_name = experiment['name']
    optimizer_name = experiment['optimizer']
    activation_name = experiment['activation']
    lr = experiment['learning_rate']
    epochs = experiment['epochs']
    
    print(f"\n--- Avvio Esperimento: {exp_name} ---")

    # Inizializza l'oggetto per tracciare le loss
    history = TrainingHistory()
    
    # Seleziona la funzione di attivazione e inizializza il modello
    activation_fn = activation_functions.get(activation_name, nn.Tanh)
    pinn = FCN(1, 1, 32, 5, activation_fn=activation_fn)
    
    # Inizializza l'ottimizzatore
    if optimizer_name == 'LBFGS':
        # LBFGS è un caso speciale, epochs=1 ma l'ottimizzazione avviene negli 'max_iter' interni
        optimizer = torch.optim.LBFGS(pinn.parameters(), lr=lr, max_iter=500, history_size=50, line_search_fn="strong_wolfe")
        loop_range = range(epochs)
    else: # Adam
        optimizer = torch.optim.Adam(pinn.parameters(), lr=lr)
        loop_range = tqdm(range(epochs), desc=f"Training {exp_name}")

    # --- CICLO DI TRAINING ---
    for i in loop_range:
        
        def closure():
            optimizer.zero_grad() 
            
            # 1. Data loss
            y_pred_data = pinn(x_data)
            loss_data = torch.mean((y_pred_data - y_data)**2)
            
            # 2. Physics loss (PDE)
            y_pinn_physics = pinn(x_physics)
            dy_pinn = torch.autograd.grad(y_pinn_physics, x_physics, torch.ones_like(y_pinn_physics), create_graph=True)[0]
            loss_pde = torch.mean(((V/F) * dy_pinn + y_pinn_physics - cAin + (V*k/F) * y_pinn_physics)**2)

            # 3. Initial Condition loss
            x_ic = torch.tensor([0.0]).view(-1, 1)
            y_ic_pred = pinn(x_ic)
            loss_ic = torch.mean((y_ic_pred - cA0)**2)
            
            # Loss totale
            loss = loss_data + loss_pde + loss_ic
            loss.backward()

            # Aggiorna l'history (ogni 10 step per LBFGS, ogni 100 per Adam)
            log_freq = 10 if optimizer_name == 'LBFGS' else 100
            if i % log_freq == 0:
                history.update(i, {
                    'total_loss': loss, 'data_loss': loss_data,
                    'pde_loss': loss_pde, 'ic_loss': loss_ic
                })
            return loss

        # Esegui lo step di ottimizzazione
        optimizer.step(closure)

    # --- SALVATAGGIO RISULTATI A FINE ESPERIMENTO ---
    print("Training completato. Salvataggio dei risultati...")

    # 1. Grafico andamento delle loss
    loss_plot_path = f"plots/CSTR/{exp_name}_loss_trends.png"
    history.plot_losses(save_path=loss_plot_path, experiment_name=exp_name)
    
    # 2. Grafico del risultato finale
    pinn.eval()
    with torch.no_grad():
        y_pred_full = pinn(x)
    
    result_plot_path = f"plots/CSTR/{exp_name}_final_prediction.png"
    
    # Usa la funzione di plot standardizzata e poi salva
    plot_result(i, x, y, x_data, y_data, y_pred_full, xp=x_physics.detach())
    plt.title(f'Risultato Finale - {exp_name}', y=0.85) # Aggiunge titolo e lo sposta
    os.makedirs(os.path.dirname(result_plot_path), exist_ok=True)
    plt.savefig(result_plot_path, bbox_inches='tight')
    plt.close()
    print(f"Grafico del risultato salvato in: {result_plot_path}")

print("\n--- Tutti gli esperimenti sono stati completati. ---")
