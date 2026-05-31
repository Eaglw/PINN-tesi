import torch
import numpy as np
import pandas as pd
import os
import sys

# Aggiunge il root del progetto al path per importare le utility
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from func.sampling_utils import generate_grid_points, generate_sobol_points

def generate_oldroyd_b_dataset(
    L=1.0, 
    H=0.2, 
    mu_s=0.005, 
    mu_p=0.005,
    lam=1.0, # tempo di rilassamento
    u_max=None, # Se None, calcolato da deltaP
    p_in=1.0,
    p_out=0.0,
    nx=60, 
    ny=30, 
    sampling_type='grid',
    noise_type='percentage', 
    noise_value=0.0,
    save_dir=os.path.dirname(os.path.abspath(__file__)),
    filename='oldroydb_data',
    seed=42
):
    """
    Genera un dataset sintetico per il flusso di Poiseuille stazionario in un canale
    per un fluido Oldroyd-B.
    """
    mu_tot = mu_s + mu_p
    print(f"Generating Oldroyd-B dataset (L={L}, H={H}, mu_s={mu_s}, mu_p={mu_p}, lam={lam}, p: {p_in}->{p_out})...")
    
    # 1. Calcolo Gradiente e Coerenza Fisica
    dp_dx = (p_out - p_in) / L
    
    if u_max is None:
        # u_max derivato dalla fisica: u_max = |dp/dx| * H^2 / (8 * mu_tot)
        u_max = (abs(dp_dx) * H**2) / (8 * mu_tot)
        print(f"Computed consistent u_max: {u_max:.4f}")
    else:
        print(f"Using provided u_max: {u_max:.4f}. Note: physics might be inconsistent if mu/H don't match.")

    # 2. Generazione Punti
    if sampling_type == 'grid':
        xy = generate_grid_points(nx, ny, L, H, margin=0.0, device='cpu')
    else:
        xy = generate_sobol_points(nx * ny, L, H, margin=0.0, device='cpu')
    
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
    
    # Gradienti di velocità: gamma_dot = du/dy = 4 * u_max * (H - 2y) / H^2
    gamma_dot = (4.0 * u_max / H**2) * (H - 2 * y)
    
    # Sforzi Polimerici per Oldroyd-B in Poiseuille
    tau_xy_base = mu_p * gamma_dot
    tau_xx_base = 2.0 * lam * mu_p * (gamma_dot ** 2)
    tau_yy_base = torch.zeros_like(u_base)
    
    # 4. Aggiunta Rumore
    u_noisy = u_base.clone()
    v_noisy = v_base.clone()
    p_noisy = p_base.clone()
    psi_noisy = psi_base.clone()
    tau_xy_noisy = tau_xy_base.clone()
    tau_xx_noisy = tau_xx_base.clone()
    tau_yy_noisy = tau_yy_base.clone()
    
    if noise_value > 0:
        # Seed per riproducibilità del rumore
        if seed is not None:
            torch.manual_seed(seed)
        if noise_type == 'percentage':
            std_u = noise_value * u_max
            # std_v separato: per Poiseuille v=0, ma se si estende a casi con v≠0
            # serve la deviazione standard corretta sulla componente v
            std_v = noise_value * u_max  # scala su u_max perché v_base=0
            std_p = noise_value * abs(p_in - p_out) if p_in != p_out else noise_value * p_in
            std_psi = noise_value * (2.0/3.0 * u_max * H) 
            std_tau_xy = noise_value * torch.max(torch.abs(tau_xy_base))
            std_tau_xx = noise_value * torch.max(torch.abs(tau_xx_base))
        else: # absolute
            std_u = noise_value
            std_v = noise_value
            std_p = noise_value
            std_psi = noise_value
            std_tau_xy = noise_value
            std_tau_xx = noise_value
        
        u_noisy += torch.randn_like(u_base) * std_u
        v_noisy += torch.randn_like(v_base) * std_v
        p_noisy += torch.randn_like(p_base) * std_p
        psi_noisy += torch.randn_like(psi_base) * std_psi
        tau_xy_noisy += torch.randn_like(tau_xy_base) * std_tau_xy
        tau_xx_noisy += torch.randn_like(tau_xx_base) * std_tau_xx
        tau_yy_noisy += torch.randn_like(tau_yy_base) * std_tau_xx * 0.1 # Piccolo rumore su componente nulla
        
    # 5. Salvataggio in formato .pt (PyTorch)
    dataset = {
        'coords': xy,
        'u': u_noisy, 'v': v_noisy, 'p': p_noisy, 'psi': psi_noisy,
        'tau_xx': tau_xx_noisy, 'tau_xy': tau_xy_noisy, 'tau_yy': tau_yy_noisy,
        'u_exact': u_base, 'p_exact': p_base, 'psi_exact': psi_base,
        'tau_xx_exact': tau_xx_base, 'tau_xy_exact': tau_xy_base, 'tau_yy_exact': tau_yy_base,
        'params': {
            'L': L, 'H': H, 'mu_s': mu_s, 'mu_p': mu_p, 'lam': lam, 'u_max': u_max, 
            'noise_type': noise_type, 'noise_value': noise_value
        }
    }
    
    os.makedirs(save_dir, exist_ok=True)
    pt_path = os.path.join(save_dir, f"{filename}.pt")
    torch.save(dataset, pt_path)
    print(f"Dataset salvato in PyTorch: {pt_path}")
    
    # 6. Salvataggio in formato .csv (Pandas)
    data_df = pd.DataFrame({
        'x': xy[:, 0].numpy(),
        'y': xy[:, 1].numpy(),
        'u': u_noisy.flatten().numpy(),
        'v': v_noisy.flatten().numpy(),
        'p': p_noisy.flatten().numpy(),
        'psi': psi_noisy.flatten().numpy(),
        'tau_xx': tau_xx_noisy.flatten().numpy(),
        'tau_xy': tau_xy_noisy.flatten().numpy(),
        'tau_yy': tau_yy_noisy.flatten().numpy(),
        'u_exact': u_base.flatten().numpy(),
        'p_exact': p_base.flatten().numpy(),
        'psi_exact': psi_base.flatten().numpy(),
        'tau_xx_exact': tau_xx_base.flatten().numpy(),
        'tau_xy_exact': tau_xy_base.flatten().numpy(),
        'tau_yy_exact': tau_yy_base.flatten().numpy(),
    })
    
    csv_path = os.path.join(save_dir, f"{filename}.csv")
    data_df.to_csv(csv_path, index=False)
    print(f"Dataset salvato in CSV: {csv_path}")
    
    return dataset

if __name__ == "__main__":
    # Generazione Clean per riferimento
    generate_oldroyd_b_dataset(
        L=1.0, H=0.2, mu_s=0.005, mu_p=0.005, lam=1.0, 
        nx=60, ny=30, noise_value=0.0, filename='oldroydb_clean'
    )
