import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import os
import sys
import random

# Importa moduli condivisi
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from func.graphic_func import save_gif_PIL
from func.history_tracker import TrainingHistory

print("--- Esecuzione CSTR Coupled (Mass + Energy Balance) ---")

# Configurazione dispositivo e precisione sono ereditate da IrreversibleCSTR_main.py
# come richiesto dall'utente: forza CPU e float64.
# `device` e `torch.get_default_dtype()` sono già impostati dal main.
_dtype = torch.get_default_dtype()
_device = device # Usa la variabile 'device' già definita nel contesto del main

print(f"Coupled Script using inherited device: {_device} with inherited dtype: {_dtype}")

def set_seed(seed):
    """Fissa il seed per garantire riproducibilità."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Per operazioni deterministiche (può rallentare)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed impostato a: {seed}")

# Imposta il seed (modifica qui per cambiare l'inizializzazione)
set_seed(42)

# 1. Definizione Parametri Fisici
# F, V, cAin, k, cA0 sono ereditati dal main
tau = V / F
k0 = 1.5e3     
E_R = 2500.0   
dH = -20.0     
rho_Cp = 1.0   
UA_V = 0.5     
Tin = 300.0    
Tcool = 300.0  
cAin_coupled = cAin 

# Fattori di Normalizzazione (Cruciali per il training multi-fisica)
C_REF = 10.0   # Scala tipica concentrazione
T_REF = 350.0  # Scala tipica temperatura

# 2. Generazione Dati "Ground Truth" Numerici
def cstr_ode_system(t, y):
    C_A = y[0]
    T = y[1]
    k_val = k0 * np.exp(-E_R / T)
    dC = (cAin_coupled - C_A) / tau - k_val * C_A
    alpha = (-dH / rho_Cp)
    beta  = (UA_V / rho_Cp)
    dT = (Tin - T) / tau + alpha * k_val * C_A + beta * (Tcool - T)
    return np.array([dC, dT])

def solve_system_rk4(t_steps):
    y0 = np.array([cA0, 300.0]) 
    res = [y0]
    dt = t_steps[1] - t_steps[0]
    y = y0
    for i in range(len(t_steps)-1):
        k1 = cstr_ode_system(t_steps[i], y)
        k2 = cstr_ode_system(t_steps[i] + 0.5*dt, y + 0.5*dt*k1)
        k3 = cstr_ode_system(t_steps[i] + 0.5*dt, y + 0.5*dt*k2)
        k4 = cstr_ode_system(t_steps[i] + dt, y + dt*k3)
        y = y + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        res.append(y)
    return np.array(res)

# Dominio temporale per il training (quello che vogliamo alla fine)
t_domain = np.linspace(0, 2, 200) 

# Risoluzione fine per integrazione stabile (evita overflow/instabilità numerica)
# Usiamo molti più passi per garantire che il dt sia piccolo abbastanza
t_fine = np.linspace(0, 2, 20000) 
data_sol_fine = solve_system_rk4(t_fine)

# Sottocampionamento per ottenere i 200 punti corrispondenti a t_domain
# Prendiamo un indice ogni (len(t_fine)-1) / (len(t_domain)-1) -> ogni 100 passi circa
indices = np.linspace(0, len(t_fine)-1, len(t_domain), dtype=int)
data_sol = data_sol_fine[indices]

# Soluzione completa per plots
C_true = torch.tensor(data_sol[:, 0], dtype=_dtype).view(-1, 1).to(_device)
T_true = torch.tensor(data_sol[:, 1], dtype=_dtype).view(-1, 1).to(_device)
t_tens = torch.tensor(t_domain, dtype=_dtype).view(-1, 1).to(_device)

# Dati di Training (Sparsi)
idx_train = np.linspace(0, len(t_domain)-1, 20, dtype=int) 
t_data = t_tens[idx_train]
C_data_target = C_true[idx_train] / C_REF # Normalizzato per il training
T_data_target = T_true[idx_train] / T_REF # Normalizzato per il training

# Punti per la Fisica
t_physics = torch.linspace(0, 2, 100, dtype=_dtype).view(-1, 1).to(_device).requires_grad_(True)

# 3. Reti Neurali
# Le reti predicono valori normalizzati [~0, ~1]
net_C = FCN(1, 1, 32, 3).to(_device)
net_T = FCN(1, 1, 32, 3).to(_device)

params = list(net_C.parameters()) + list(net_T.parameters())
optimizer_coupled = torch.optim.Adam(params, lr=1e-3)

history = TrainingHistory()

# Funzione Plotting (Denormalizza per la visualizzazione)
def plot_coupled_result(i, t, C_true, T_true, t_data, C_data_norm, T_data_norm, C_pred_norm, T_pred_norm, t_phys=None):
    plt.figure(figsize=(12, 5))
    
    # Denormalizzazione predizioni e dati
    C_pred = C_pred_norm.detach().cpu() * C_REF
    T_pred = T_pred_norm.detach().cpu() * T_REF
    C_data = C_data_norm.cpu() * C_REF
    T_data = T_data_norm.cpu() * T_REF
    
    # Subplot 1: Concentrazione
    plt.subplot(1, 2, 1)
    plt.plot(t.cpu(), C_true.cpu(), 'grey', linewidth=2, alpha=0.8, label="Exact Solution")
    plt.plot(t.cpu(), C_pred, 'tab:blue', linewidth=4, alpha=0.8, label="NN Prediction")
    plt.scatter(t_data.cpu(), C_data, s=60, color="white", alpha=0.4, label='Training Data')
    if t_phys is not None:
         # Plot punti fisica sulla riga 0 (o minimo)
         plt.scatter(t_phys.detach().cpu(), torch.zeros_like(t_phys.detach().cpu()), 
                     s=20, color="tab:green", alpha=0.2, label='Phys Pts')
    plt.xlabel('Time')
    plt.ylabel('Concentration ($C_A$)')
    plt.title(f'Concentration (Step {i+1})')
    plt.legend(frameon=False)
    
    # Subplot 2: Temperatura
    plt.subplot(1, 2, 2)
    plt.plot(t.cpu(), T_true.cpu(), 'grey', linewidth=2, alpha=0.8, label="Exact Solution")
    plt.plot(t.cpu(), T_pred, 'tab:red', linewidth=4, alpha=0.8, label="NN Prediction")
    plt.scatter(t_data.cpu(), T_data, s=60, color="white", alpha=0.4, label='Training Data')
    plt.xlabel('Time')
    plt.ylabel('Temperature (K)')
    plt.title(f'Temperature (Step {i+1})')
    plt.legend(frameon=False)

    plt.tight_layout()

# 4. Training Loop
files = []
os.makedirs("IrreversibleCSTR/Results/plots", exist_ok=True)

print("Inizio training Coupled PINN (Normalized)...")
for i in tqdm(range(step), desc="Coupled Training"):
    optimizer_coupled.zero_grad()
    
    # --- 1. Data Loss (su output normalizzati) ---
    C_out_data = net_C(t_data)
    T_out_data = net_T(t_data)
    
    loss_d_C = torch.mean((C_out_data - C_data_target)**2)
    loss_d_T = torch.mean((T_out_data - T_data_target)**2)

    # --- 2. Physics Loss (Denormalizzazione dentro la fisica) ---
    C_out_phys = net_C(t_physics)
    T_out_phys = net_T(t_physics)
    
    # Variabili fisiche reali
    C_phys = C_out_phys * C_REF
    T_phys = T_out_phys * T_REF
    
    # Derivate automatiche (Attenzione: d(Out)/dt, quindi dC/dt = d(Out)/dt * C_REF)
    dC_out_dt = torch.autograd.grad(C_out_phys, t_physics, torch.ones_like(C_out_phys), create_graph=True)[0]
    dT_out_dt = torch.autograd.grad(T_out_phys, t_physics, torch.ones_like(T_out_phys), create_graph=True)[0]
    
    dC_dt = dC_out_dt * C_REF
    dT_dt = dT_out_dt * T_REF
    
    # Equazioni Fisiche
    k_phys = k0 * torch.exp(-E_R / T_phys)
    alpha = (-dH / rho_Cp)
    beta  = (UA_V / rho_Cp)
    
    # Residui (Dimensionali)
    res_mass = dC_dt - ( (cAin_coupled - C_phys)/tau - k_phys * C_phys )
    res_energy = dT_dt - ( (Tin - T_phys)/tau + alpha * k_phys * C_phys + beta * (Tcool - T_phys) )
    
    # Normalizzazione Residui (Importante per bilanciare gradienti)
    # Dividiamo i residui per le scale di riferimento (o scale temporali) per renderli adimensionali/O(1)
    loss_p_mass = torch.mean((res_mass / C_REF)**2)
    loss_p_energy = torch.mean((res_energy / T_REF)**2)
    
    # Loss Totale (Pesi bilanciati poiché tutto è normalizzato)
    # STRATEGIA WARM-UP: 
    # Spegniamo la fisica all'inizio per permettere alla rete di portarsi su valori 
    # di temperatura fisici (evitando T negativi che fanno esplodere l'Arrhenius)
    if i < 1000:
        lambda_phys = 0.0
    else:
        lambda_phys = 0.1
        if i == 1000:
            print("\n[INFO] Warm-up completato. Attivazione Loss Fisica.")

    loss = loss_d_C + loss_d_T + lambda_phys * (loss_p_mass + loss_p_energy)
    
    loss.backward()
    optimizer_coupled.step()
    
    # Update History
    loss_dict = {
        'total_loss': loss,
        'data_C': loss_d_C,
        'data_T': loss_d_T,
        'phys_M': loss_p_mass,
        'phys_E': loss_p_energy
    }
    history.update(i, loss_dict, lr=optimizer_coupled.param_groups[0]['lr'])
    
    # Plotting animazione
    if (i+1) % 200 == 0:
        net_C.eval()
        net_T.eval()
        with torch.no_grad():
            C_curr = net_C(t_tens)
            T_curr = net_T(t_tens)
        
        plot_coupled_result(i, t_tens, C_true, T_true, t_data, C_data_target, T_data_target, C_curr, T_curr, t_physics)
        
        file_path = f"IrreversibleCSTR/Results/plots/CSTR_Coupled_{i+1:08d}.png"
        plt.savefig(file_path, bbox_inches='tight', facecolor="white")
        files.append(file_path)
        plt.close("all")
        
        net_C.train()
        net_T.train()

# 5. Salvataggio GIF e Loss
save_gif_PIL("IrreversibleCSTR/Results/CSTR_Coupled.gif", files, fps=15, loop=0, delete_files=True)

# Plot Losses (Forziamo un limite y se necessario, ma log scale dovrebbe andare)
history.plot_losses(save_path="IrreversibleCSTR/Results/CSTR_Coupled_Loss.png", experiment_name="Coupled PINN (Normalized)", show_plot=False)

# Plot statico finale
net_C.eval()
net_T.eval()
with torch.no_grad():
    C_final = net_C(t_tens)
    T_final = net_T(t_tens)
plot_coupled_result(step-1, t_tens, C_true, T_true, t_data, C_data_target, T_data_target, C_final, T_final)
plt.savefig("IrreversibleCSTR/Results/CSTR_Coupled_Final.png", bbox_inches='tight')
print("Training completato. Risultati salvati in IrreversibleCSTR/Results/")