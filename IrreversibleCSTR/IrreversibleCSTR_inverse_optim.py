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

x_physics = torch.linspace(0, 5, 30).view(-1, 1).requires_grad_(True)

# --- CONFIGURAZIONE DEGLI ESPERIMENTI ---
experiments_to_run = [
    #{'name': 'Adam_Tanh_10k', 'optimizer': 'Adam', 'activation': 'Tanh', 'learning_rate': 1e-3, 'epochs': 10000},
    #{'name': 'Adam_GELU_10k', 'optimizer': 'Adam', 'activation': 'GELU', 'learning_rate': 1e-3, 'epochs': 10000},
    {'name': 'LBFGS_Tanh_1500iter', 'optimizer': 'LBFGS', 'activation': 'Tanh', 'learning_rate': 1.0, 'epochs': 1},
    {'name': 'LBFGS_GELU_1500iter', 'optimizer': 'LBFGS', 'activation': 'GELU', 'learning_rate': 1.0, 'epochs': 1},
    {'name': 'Adam_then_LBFGS_Tanh', 'optimizer': 'Adam_then_LBFGS', 'activation': 'Tanh', 'learning_rate': 1e-3, 'epochs': 20000},
    {'name': 'Adam_then_LBFGS_GELU', 'optimizer': 'Adam_then_LBFGS', 'activation': 'GELU', 'learning_rate': 1e-3, 'epochs': 20000},
]

# --- DEFINIZIONI DEL MODELLO ---

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

    history = TrainingHistory()
    activation_fn = activation_functions.get(activation_name, nn.Tanh)
    pinn = FCN(1, 1, 32, 5, activation_fn=activation_fn)
    
    # --- LOGICA DI TRAINING ---
    
    if optimizer_name == 'Adam_then_LBFGS':
        print("Fase 1: Training con Adam...")
        optimizer = torch.optim.Adam(pinn.parameters(), lr=lr)
        loss_history_for_switch = []
        plateau_window = 1000
        plateau_threshold = 0.01

        adam_epochs = epochs
        last_epoch = 0

        with tqdm(total=adam_epochs, desc=f"Training Adam") as pbar:
            for i in range(adam_epochs):
                last_epoch = i
                optimizer.zero_grad()
                
                y_pred_data = pinn(x_data)
                loss_data = torch.mean((y_pred_data - y_data)**2)
                y_pinn_physics = pinn(x_physics)
                dy_pinn = torch.autograd.grad(y_pinn_physics, x_physics, torch.ones_like(y_pinn_physics), create_graph=True)[0]
                loss_pde = torch.mean(((V/F) * dy_pinn + y_pinn_physics - cAin + (V*k/F) * y_pinn_physics)**2)
                x_ic = torch.tensor([0.0]).view(-1, 1)
                y_ic_pred = pinn(x_ic)
                loss_ic = torch.mean((y_ic_pred - cA0)**2)
                loss = loss_data + loss_pde + loss_ic
                
                loss.backward()
                optimizer.step()
                pbar.set_postfix({'loss': loss.item()})
                pbar.update(1)

                if i % 100 == 0:
                    history.update(i, {'total_loss': loss, 'data_loss': loss_data, 'pde_loss': loss_pde, 'ic_loss': loss_ic})

                current_loss_val = loss.item()
                loss_history_for_switch.append(current_loss_val)
                if i > plateau_window:
                    past_loss = loss_history_for_switch[i - plateau_window]
                    relative_improvement = (past_loss - current_loss_val) / past_loss if past_loss > 0 else 0
                    if relative_improvement < plateau_threshold:
                        print(f"\nLoss plateaued at epoch {i}. Improvement in last {plateau_window} epochs: {relative_improvement*100:.4f}%")
                        break
        
        print("\nFase 2: Fine-tuning con LBFGS...")
        optimizer = torch.optim.LBFGS(pinn.parameters(), lr=1.0, max_iter=1500, history_size=50, line_search_fn="strong_wolfe")
        
        lbfgs_iter_wrapper = [0]
        def lbfgs_closure():
            optimizer.zero_grad()
            y_pred_data = pinn(x_data)
            loss_data = torch.mean((y_pred_data - y_data)**2)
            y_pinn_physics = pinn(x_physics)
            dy_pinn = torch.autograd.grad(y_pinn_physics, x_physics, torch.ones_like(y_pinn_physics), create_graph=True)[0]
            loss_pde = torch.mean(((V/F) * dy_pinn + y_pinn_physics - cAin + (V*k/F) * y_pinn_physics)**2)
            x_ic = torch.tensor([0.0]).view(-1, 1)
            y_ic_pred = pinn(x_ic)
            loss_ic = torch.mean((y_ic_pred - cA0)**2)
            loss = loss_data + loss_pde + loss_ic
            loss.backward()

            # Log at each iteration
            history.update(last_epoch + lbfgs_iter_wrapper[0], {
                'total_loss': loss.detach(), 
                'data_loss': loss_data.detach(), 
                'pde_loss': loss_pde.detach(), 
                'ic_loss': loss_ic.detach()
            })
            lbfgs_iter_wrapper[0] += 1
            if lbfgs_iter_wrapper[0] % 100 == 0:
                print(f'LBFGS iter {lbfgs_iter_wrapper[0]}, Loss: {loss.item()}')
            
            return loss

        optimizer.step(lbfgs_closure)

    else:
        if optimizer_name == 'LBFGS':
            optimizer = torch.optim.LBFGS(pinn.parameters(), lr=lr, max_iter=1500, history_size=50, line_search_fn="strong_wolfe")
            print(f"Training con LBFGS (max_iter=1500)...")
            
            lbfgs_iter_wrapper = [0]
            def lbfgs_closure():
                optimizer.zero_grad()
                y_pred_data = pinn(x_data)
                loss_data = torch.mean((y_pred_data - y_data)**2)
                y_pinn_physics = pinn(x_physics)
                dy_pinn = torch.autograd.grad(y_pinn_physics, x_physics, torch.ones_like(y_pinn_physics), create_graph=True)[0]
                loss_pde = torch.mean(((V/F) * dy_pinn + y_pinn_physics - cAin + (V*k/F) * y_pinn_physics)**2)
                x_ic = torch.tensor([0.0]).view(-1, 1)
                y_ic_pred = pinn(x_ic)
                loss_ic = torch.mean((y_ic_pred - cA0)**2)
                loss = loss_data + loss_pde + loss_ic
                loss.backward()

                # Log at each iteration
                history.update(lbfgs_iter_wrapper[0], {
                    'total_loss': loss.detach(), 
                    'data_loss': loss_data.detach(), 
                    'pde_loss': loss_pde.detach(), 
                    'ic_loss': loss_ic.detach()
                })
                lbfgs_iter_wrapper[0] += 1
                if lbfgs_iter_wrapper[0] % 100 == 0:
                    print(f'LBFGS iter {lbfgs_iter_wrapper[0]}, Loss: {loss.item()}')
                
                return loss
            
            optimizer.step(lbfgs_closure)

        else: # Adam
            optimizer = torch.optim.Adam(pinn.parameters(), lr=lr)
            
            def closure(current_epoch=0, log_freq=100):
                optimizer.zero_grad()
                y_pred_data = pinn(x_data)
                loss_data = torch.mean((y_pred_data - y_data)**2)
                y_pinn_physics = pinn(x_physics)
                dy_pinn = torch.autograd.grad(y_pinn_physics, x_physics, torch.ones_like(y_pinn_physics), create_graph=True)[0]
                loss_pde = torch.mean(((V/F) * dy_pinn + y_pinn_physics - cAin + (V*k/F) * y_pinn_physics)**2)
                x_ic = torch.tensor([0.0]).view(-1, 1)
                y_ic_pred = pinn(x_ic)
                loss_ic = torch.mean((y_ic_pred - cA0)**2)
                loss = loss_data + loss_pde + loss_ic
                loss.backward()
                if current_epoch % log_freq == 0:
                    history.update(current_epoch, {'total_loss': loss, 'data_loss': loss_data, 'pde_loss': loss_pde, 'ic_loss': loss_ic})
                return loss

            with tqdm(total=epochs, desc=f"Training {exp_name}") as pbar:
                for i in range(epochs):
                    loss = closure(current_epoch=i)
                    optimizer.step()
                    pbar.set_postfix({'loss': loss.item()})
                    pbar.update(1)

    # --- SALVATAGGIO RISULTATI A FINE ESPERIMENTO ---
    print("Training completato. Salvataggio dei risultati...")
    
    loss_plot_path = f"plots/CSTR/{exp_name}_loss_trends.png"
    history.plot_losses(save_path=loss_plot_path, experiment_name=exp_name)
    
    pinn.eval()
    with torch.no_grad():
        y_pred_full = pinn(x)
    
    result_plot_path = f"plots/CSTR/{exp_name}_final_prediction.png"
    
    plot_result(epochs, x, y, x_data, y_data, y_pred_full, xp=x_physics.detach())
    plt.title(f'Risultato Finale - {exp_name}', y=0.85)
    os.makedirs(os.path.dirname(result_plot_path), exist_ok=True)
    plt.savefig(result_plot_path, bbox_inches='tight')
    plt.close()
    print(f"Grafico del risultato salvato in: {result_plot_path}")

print("\n--- Tutti gli esperimenti sono stati completati. ---")
