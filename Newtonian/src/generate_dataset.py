import torch
import numpy as np
import pandas as pd
import os
import sys

# Aggiunge il root del progetto al path per importare le utility
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from func.sampling_utils import generate_grid_points, generate_sobol_points

def generate_poiseuille_dataset(
    L=1.0, 
    H=0.1, 
    mu=1e-3, 
    u_max=1.0, 
    nx=50, 
    ny=20, 
    sampling_type='grid',
    noise_type='percentage', 
    noise_value=0.01,
    p_ref=1.0,
    save_dir='Newtonian/data',
    filename='poiseuille_data'
):
    """
    Genera un dataset sintetico per il flusso di Poiseuille stazionario in un canale.
    """
    print(f"Generating Poiseuille dataset (L={L}, H={H}, u_max={u_max})...")
    
    # 1. Generazione Punti
    if sampling_type == 'grid':
        xy = generate_grid_points(nx, ny, L, H, margin=0.0)
    else:
        xy = generate_sobol_points(nx * ny, L, H, margin=0.0)
    
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    
    # 2. Calcolo Soluzione Analitica
    # Profilo di velocità u(y) = 4 * u_max * y * (H - y) / H^2
    u_base = 4 * u_max * (y * (H - y)) / (H**2)
    v_base = torch.zeros_like(u_base)
    
    # Gradiente di pressione: mu * d2u/dy2 = dp/dx
    # d2u/dy2 = - 8 * u_max / H^2
    dp_dx = - (8 * mu * u_max) / (H**2)
    p_base = p_ref + dp_dx * x
    
    # 3. Aggiunta Rumore
    u_noisy = u_base.clone()
    v_noisy = v_base.clone()
    p_noisy = p_base.clone()
    
    if noise_value > 0:
        if noise_type == 'percentage':
            std_u = noise_value * u_max
            std_p = noise_value * torch.abs(p_base.max() - p_base.min()) if p_base.max() != p_base.min() else noise_value * p_ref
        else: # absolute
            std_u = noise_value
            std_p = noise_value
        
        u_noisy += torch.randn_like(u_base) * std_u
        v_noisy += torch.randn_like(v_base) * std_u # noise anche in verticale per realismo
        p_noisy += torch.randn_like(p_base) * std_p
        
    # 4. Salvataggio in formato .pt (PyTorch)
    dataset = {
        'coords': xy,
        'u': u_noisy,
        'v': v_noisy,
        'p': p_noisy,
        'u_exact': u_base,
        'p_exact': p_base,
        'params': {
            'L': L, 'H': H, 'mu': mu, 'u_max': u_max, 
            'noise_type': noise_type, 'noise_value': noise_value
        }
    }
    
    os.makedirs(save_dir, exist_ok=True)
    pt_path = os.path.join(save_dir, f"{filename}.pt")
    torch.save(dataset, pt_path)
    print(f"Dataset salvato in PyTorch: {pt_path}")
    
    # 5. Salvataggio in formato .csv (Pandas)
    data_df = pd.DataFrame({
        'x': xy[:, 0].numpy(),
        'y': xy[:, 1].numpy(),
        'u': u_noisy.flatten().numpy(),
        'v': v_noisy.flatten().numpy(),
        'p': p_noisy.flatten().numpy(),
        'u_exact': u_base.flatten().numpy(),
        'p_exact': p_base.flatten().numpy()
    })
    
    csv_path = os.path.join(save_dir, f"{filename}.csv")
    data_df.to_csv(csv_path, index=False)
    print(f"Dataset salvato in CSV: {csv_path}")
    
    return dataset

if __name__ == "__main__":
    # Esempio di generazione: Accuratezza 99% (Rumore 1%)
    generate_poiseuille_dataset(
        L=1.0, 
        H=0.2, 
        mu=0.01, 
        u_max=1.0, 
        nx=60, 
        ny=30, 
        noise_type='percentage', 
        noise_value=0.01,
        filename='poiseuille_noisy'
    )
    
    # Generazione Clean per riferimento
    generate_poiseuille_dataset(
        L=1.0, 
        H=0.2, 
        mu=0.01, 
        u_max=1.0, 
        nx=60, 
        ny=30, 
        noise_value=0.0,
        filename='poiseuille_clean'
    )
