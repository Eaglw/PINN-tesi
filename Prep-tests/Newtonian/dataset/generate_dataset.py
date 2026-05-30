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
    u_max=None, # Se None, calcolato da deltaP
    p_in=1.0,
    p_out=0.0,
    nx=50, 
    ny=20, 
    sampling_type='grid',
    noise_type='percentage', 
    noise_value=0.01,
    save_dir=os.path.dirname(os.path.abspath(__file__)),
    filename='poiseuille_data'
):
    """
    Genera un dataset sintetico per il flusso di Poiseuille stazionario in un canale.
    Impone un gradiente di pressione uniforme da p_in a p_out.
    """
    print(f"Generating Poiseuille dataset (L={L}, H={H}, p: {p_in}->{p_out})...")
    
    # 1. Calcolo Gradiente e Coerenza Fisica
    dp_dx = (p_out - p_in) / L
    
    if u_max is None:
        # u_max derivato dalla fisica: u_max = |dp/dx| * H^2 / (8 * mu)
        u_max = (abs(dp_dx) * H**2) / (8 * mu)
        print(f"Computed consistent u_max: {u_max:.4f}")
    else:
        # Se u_max è fornito, il gradiente calcolato potrebbe non essere coerente con mu e H
        # ma rispettiamo la richiesta dell'utente sulla pressione.
        print(f"Using provided u_max: {u_max:.4f}. Note: physics might be inconsistent if mu/H don't match.")

    # 2. Generazione Punti
    if sampling_type == 'grid':
        xy = generate_grid_points(nx, ny, L, H, margin=0.0)
    else:
        xy = generate_sobol_points(nx * ny, L, H, margin=0.0)
    
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    
    # 3. Calcolo Soluzione Analitica
    # Profilo di velocità u(y) = 4 * u_max * y * (H - y) / H^2
    u_base = 4 * u_max * (y * (H - y)) / (H**2)
    v_base = torch.zeros_like(u_base)
    
    # Pressione lineare: p(x) = p_in + dp_dx * x
    p_base = p_in + dp_dx * x
    
    # Stream Function psi(y) = (4*u_max/H^2) * (H*y^2/2 - y^3/3)
    psi_base = (4.0 * u_max / H**2) * (H * y**2 / 2.0 - y**3 / 3.0)
    
    # 4. Aggiunta Rumore
    u_noisy = u_base.clone()
    v_noisy = v_base.clone()
    p_noisy = p_base.clone()
    psi_noisy = psi_base.clone()
    
    if noise_value > 0:
        if noise_type == 'percentage':
            std_u = noise_value * u_max
            std_p = noise_value * abs(p_in - p_out) if p_in != p_out else noise_value * p_in
            std_psi = noise_value * (2.0/3.0 * u_max * H) # Basato sul valore max di psi
        else: # absolute
            std_u = noise_value
            std_p = noise_value
            std_psi = noise_value
        
        u_noisy += torch.randn_like(u_base) * std_u
        v_noisy += torch.randn_like(v_base) * std_u 
        p_noisy += torch.randn_like(p_base) * std_p
        psi_noisy += torch.randn_like(psi_base) * std_psi
        
    # 4. Salvataggio in formato .pt (PyTorch)
    dataset = {
        'coords': xy,
        'u': u_noisy,
        'v': v_noisy,
        'p': p_noisy,
        'psi': psi_noisy,
        'u_exact': u_base,
        'p_exact': p_base,
        'psi_exact': psi_base,
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
        'psi': psi_noisy.flatten().numpy(),
        'u_exact': u_base.flatten().numpy(),
        'p_exact': p_base.flatten().numpy(),
        'psi_exact': psi_base.flatten().numpy()
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
        nx=60, 
        ny=30, 
        noise_value=0.0,
        filename='poiseuille_clean'
    )
