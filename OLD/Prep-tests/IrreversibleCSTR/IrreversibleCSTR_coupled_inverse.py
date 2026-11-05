import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import os
import sys

# Importa moduli condivisi
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from func.graphic_func import save_gif_PIL
from func.history_tracker import TrainingHistory

print("--- Esecuzione CSTR Coupled Inverse Problem(Mass + Energy Balance) ---")

# Configurazione dispositivo e precisione sono ereditate da IrreversibleCSTR_main.py
# come richiesto dall'utente: forza CPU e float64.
# `device` e `torch.get_default_dtype()` sono già impostati dal main.
_dtype = torch.get_default_dtype()
_device = device # Usa la variabile 'device' già definita nel contesto del main

print(f"Coupled Script using inherited device: {_device} with inherited dtype: {_dtype}")

# 1. Definizione Parametri Fisici
# F, V, cAin, k, cA0 sono ereditati dal main
tau = V / F
k0 = 1.5e3 #solo per dati fisici, da indovinare
k0_1st=1e4
E_R = 2500.0   
dH = -20.0     
rho_Cp = 1.0 #solo per dati fisici, da indovinare  
rho_Cp_1st=10
UA_V = 0.5     
Tin = 300.0    
Tcool = 300.0  
cAin_coupled = cAin 

# Fattori di Normalizzazione (Cruciali per il training multi-fisica)
C_REF = 10.0   # Scala tipica concentrazione
T_REF = 350.0  # Scala tipica temperatura

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

# Tensori Ground Truth
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
