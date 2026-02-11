# Questo script esegue un'analisi incrociata di ottimizzatori e funzioni di attivazione
# per il problema del reattore CSTR in configurazione "forward" (k è noto).
# I risultati, inclusi i grafici delle loss e delle predizioni finali,
# vengono salvati automaticamente nella cartella 'plots/CSTR/'.

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import sys

# Assicura che i moduli nella cartella 'func' siano importabili
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from func.history_tracker import TrainingHistory, compute_pinn_loss
from func.graphic_func import plot_result

# Punti per la loss sulla fisica (collocation points)
x_physics = torch.linspace(0, 5, 50).view(-1, 1).requires_grad_(True)

# --- DEFINIZIONE DELLA RETE ---

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

# --- FUNZIONI DI TRAINING REFACTORING ---

def cstr_physics_loss(model, x_physics):
    """Calcola il residuo della PDE per il CSTR."""
    # Nota: V, F, k, cAin devono essere definiti nel contesto globale o passati
    # Qui assumiamo siano globali come nel codice originale
    y_pinn = model(x_physics)
    dy_pinn = torch.autograd.grad(y_pinn, x_physics, torch.ones_like(y_pinn), create_graph=True)[0]
    
    # Equazione differenziale: (V/F)*dCa/dt + Ca - Cain + (V*k/F)*Ca = 0
    residual = (V/F) * dy_pinn + y_pinn - cAin + (V*k/F) * y_pinn
    return torch.mean(residual**2)

def cstr_ic_loss(model):
    """Calcola la loss sulla condizione iniziale."""
    # Nota: cA0 deve essere globale
    x_ic = torch.tensor([0.0]).view(-1, 1)
    if next(model.parameters()).is_cuda:
         x_ic = x_ic.cuda()
    y_ic_pred = model(x_ic)
    return torch.mean((y_ic_pred - cA0)**2)

def compute_loss(pinn, x_data, y_data, x_physics):
    """
    Wrapper per compute_pinn_loss specifico per questo problema.
    Mantiene la firma originale per compatibilità con le funzioni di training esistenti.
    """
    return compute_pinn_loss(
        model=pinn, 
        x_data=x_data, 
        y_data=y_data, 
        physics_loss_fn=cstr_physics_loss, 
        x_physics=x_physics, 
        ic_loss_fn=cstr_ic_loss
    )

def train_adam(pinn, optimizer, history, epochs, x_data, y_data, x_physics, pbar_desc, use_patience=False, patience_epochs=500):
    """Ciclo di training per l'ottimizzatore Adam con early stopping basato sulla pazienza."""
    last_epoch = 0
    
    # Variabili per la logica di 'patience'
    best_loss = float('inf')
    patience_counter = 0

    with tqdm(total=epochs, desc=pbar_desc) as pbar:
        for i in range(epochs):
            last_epoch = i
            optimizer.zero_grad()
            
            loss, loss_dict = compute_loss(pinn, x_data, y_data, x_physics)
            
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'loss': loss.item()})
            pbar.update(1)

            if i % 100 == 0:
                history.update(i, {k: v.detach() for k, v in loss_dict.items()})

            # Logica di Early Stopping basata sulla 'pazienza'
            if use_patience:
                current_loss_val = loss.item()
                if current_loss_val < best_loss:
                    best_loss = current_loss_val
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience_epochs:
                    print(f"\nEarly stopping all'epoca {i} per mancanza di miglioramento nelle ultime {patience_epochs} epoche.")
                    break # Interrompe il training
    return last_epoch

def train_lbfgs(pinn, optimizer, history, x_data, y_data, x_physics, epoch_offset=0):
    """Ciclo di training per l'ottimizzatore LBFGS."""
    print(f"Training con LBFGS (max_iter={optimizer.param_groups[0]['max_iter']})...")
    
    lbfgs_iter_wrapper = [0]
    def closure():
        optimizer.zero_grad()
        loss, loss_dict = compute_loss(pinn, x_data, y_data, x_physics)
        loss.backward()

        current_iter = lbfgs_iter_wrapper[0]
        history.update(epoch_offset + current_iter, {k: v.detach() for k, v in loss_dict.items()})
        
        if current_iter % 100 == 0:
            print(f'LBFGS iter {current_iter}, Loss: {loss.item()}')
        
        lbfgs_iter_wrapper[0] += 1
        return loss

    optimizer.step(closure)

# --- CONFIGURAZIONE DEGLI ESPERIMENTI ---
experiments_to_run = [
    {'name': 'Adam_Tanh_10k', 'optimizer': 'Adam', 'activation': 'Tanh', 'learning_rate': 1e-3, 'epochs': 10000},
    {'name': 'Adam_GELU_10k', 'optimizer': 'Adam', 'activation': 'GELU', 'learning_rate': 1e-3, 'epochs': 10000},
    {'name': 'LBFGS_Tanh_1500iter', 'optimizer': 'LBFGS', 'activation': 'Tanh', 'learning_rate': 1.0, 'max_iter': 1500},
    {'name': 'LBFGS_GELU_1500iter', 'optimizer': 'LBFGS', 'activation': 'GELU', 'learning_rate': 1.0, 'max_iter': 1500},
    {'name': 'Adam_then_LBFGS_Tanh', 'optimizer': 'Adam_then_LBFGS', 'activation': 'Tanh', 'learning_rate': 1e-3, 'epochs': 20000, 'max_iter_lbfgs': 1500},
    {'name': 'Adam_then_LBFGS_GELU', 'optimizer': 'Adam_then_LBFGS', 'activation': 'GELU', 'learning_rate': 1e-3, 'epochs': 20000, 'max_iter_lbfgs': 1500},
]

# --- CICLO PRINCIPALE DEGLI ESPERIMENTI ---
for experiment in experiments_to_run:
    exp_name = experiment['name']
    optimizer_name = experiment['optimizer']
    activation_name = experiment['activation']
    lr = experiment['learning_rate']
    
    print(f"\n--- Avvio Esperimento: {exp_name} ---")

    history = TrainingHistory()
    activation_fn = activation_functions.get(activation_name, nn.Tanh)
    pinn = FCN(1, 1, 32, 5, activation_fn=activation_fn)
    last_adam_epoch = 0
    # --- LOGICA DI TRAINING ---
    if optimizer_name == 'Adam_then_LBFGS':
        print("Fase 1: Training con Adam...")
        adam_optimizer = torch.optim.Adam(pinn.parameters(), lr=lr)
        adam_epochs = experiment.get('epochs', 20000)
        
        last_adam_epoch = train_adam(
            pinn, adam_optimizer, history, adam_epochs, x_data, y_data, x_physics,
            pbar_desc="Training Adam", use_patience=True
        )
        
        print("\nFase 2: Fine-tuning con LBFGS...")
        max_iter = experiment.get('max_iter_lbfgs', 1500)
        lbfgs_optimizer = torch.optim.LBFGS(
            pinn.parameters(), lr=1.0, max_iter=max_iter, 
            history_size=50, line_search_fn="strong_wolfe"
        )
        train_lbfgs(
            pinn, lbfgs_optimizer, history, x_data, y_data, x_physics,
            epoch_offset=last_adam_epoch
        )

    elif optimizer_name == 'LBFGS':
        max_iter = experiment.get('max_iter', 1500)
        lbfgs_optimizer = torch.optim.LBFGS(
            pinn.parameters(), lr=lr, max_iter=max_iter, 
            history_size=50, line_search_fn="strong_wolfe"
        )
        train_lbfgs(pinn, lbfgs_optimizer, history, x_data, y_data, x_physics)

    else: # Adam
        adam_optimizer = torch.optim.Adam(pinn.parameters(), lr=lr)
        adam_epochs = experiment.get('epochs', 10000)
        train_adam(
            pinn, adam_optimizer, history, adam_epochs, x_data, y_data, x_physics,
            pbar_desc=f"Training {exp_name}"
        )

    # --- SALVATAGGIO RISULTATI A FINE ESPERIMENTO ---
    print("Training completato. Salvataggio dei risultati...")
    
    plot_dir = f"IrreversibleCSTR/Results/{exp_name}"
    os.makedirs(plot_dir, exist_ok=True)
    
    loss_plot_path = os.path.join(plot_dir, "loss_trends.png")
    # Usiamo adam_epochs per dividere il grafico solo se abbiamo avuto entrambe le fasi
    history.plot_losses(adam_epochs=last_adam_epoch if last_adam_epoch > 0 else None, 
                        save_path=loss_plot_path, 
                        experiment_name=exp_name)
   

    pinn.eval()
    with torch.no_grad():
        y_pred_full = pinn(x)
    
    result_plot_path = os.path.join(plot_dir, "final_prediction.png")
    plot_result(experiment.get('epochs', 0), x, y, x_data, y_data, y_pred_full, xp=x_physics.detach())
    plt.title(f'Risultato Finale - {exp_name}', y=0.85)
    plt.savefig(result_plot_path, bbox_inches='tight')
    plt.close()
    print(f"Grafici salvati in: {plot_dir}")

print("\n--- Tutti gli esperimenti sono stati completati. ---")
