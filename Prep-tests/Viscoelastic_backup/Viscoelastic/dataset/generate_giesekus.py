import torch
import numpy as np
import pandas as pd
import os
import sys
from scipy.optimize import brentq
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d

# Aggiunge il root del progetto al path per importare le utility
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from func.sampling_utils import generate_grid_points, generate_sobol_points

def solve_giesekus_point(tau_tot, mu_s, mu_p, lam, alpha):
    """
    Risolve il modello di Giesekus in 1D usando la radice esatta del polinomio.
    """
    if abs(tau_tot) < 1e-12:
        return 0.0, 0.0, 0.0, 0.0
        
    if alpha < 1e-4:
        g_dot_sol = tau_tot / (mu_s + mu_p)
        return g_dot_sol, 2 * lam * mu_p * (g_dot_sol**2), mu_p * g_dot_sol, 0.0

    def get_txy(g_dot):
        if abs(g_dot) < 1e-12: return 0.0
        chi = lam * abs(g_dot)
        
        def F(N):
            # L'equazione polinomiale algebricamente corretta per il Giesekus
            term1 = (chi**2) * ((alpha + N)**4)
            term2 = N * (1.0 + N) * ((alpha + N * (2.0 * alpha - 1.0))**2)
            return term1 + term2
            
        # N è fisicamente e matematicamente confinato tra -alpha e 0
        eps = 1e-12
        N_root = brentq(F, -alpha + eps, 0.0)
        
        S = np.sqrt(-N_root * (1.0 + N_root))
        return np.sign(g_dot) * S * mu_p / (alpha * lam)

    def momentum_res(g_dot):
        return mu_s * g_dot + get_txy(g_dot) - tau_tot
        
    # Il gradiente di velocità sarà sempre compreso tra il caso
    # totalmente viscoso a riposo e il caso totalmente shear-thinning (solo solvente)
    g1 = tau_tot / (mu_s + mu_p)
    g2 = tau_tot / mu_s
    g_min, g_max = min(g1, g2), max(g1, g2)
    
    # Allarghiamo l'intervallo dell'1% per coprire approssimazioni di macchina
    g_min = g_min * 1.01 if g_min < 0 else g_min * 0.99
    g_max = g_max * 1.01 if g_max > 0 else g_max * 0.99
    
    g_dot_sol = brentq(momentum_res, g_min, g_max)
    
    # Ricostruzione tensoriale esatta
    chi = lam * abs(g_dot_sol)
    if chi < 1e-12:
        return 0.0, 0.0, 0.0, 0.0
        
    def F_final(N):
        term1 = (chi**2) * ((alpha + N)**4)
        term2 = N * (1.0 + N) * ((alpha + N * (2.0 * alpha - 1.0))**2)
        return term1 + term2
        
    N_sol = brentq(F_final, -alpha + 1e-12, 0.0)
    S_sol = np.sqrt(-N_sol * (1.0 + N_sol))
    
    txy = np.sign(g_dot_sol) * S_sol * mu_p / (alpha * lam)
    tyy = N_sol * mu_p / (alpha * lam)
    
    if S_sol > 1e-12:
        X = (chi * (alpha + N_sol)) / S_sol
    else:
        X = 1.0 + N_sol
        
    C_sol = X - 1.0 - N_sol
    txx = C_sol * mu_p / (alpha * lam)
    
    return g_dot_sol, txx, txy, tyy
def generate_giesekus_dataset(
    L=1.0, 
    H=0.2, 
    mu_s=0.005, 
    mu_p=0.005,
    lam=1.0, 
    alpha=0.2, 
    p_in=1.0,
    p_out=0.0,
    nx=60, 
    ny=30, 
    sampling_type='grid',
    noise_type='percentage', 
    noise_value=0.0,
    save_dir=os.path.dirname(os.path.abspath(__file__)),
    filename='giesekus_data',
    seed=42
):
    print(f"Generating Giesekus dataset (L={L}, H={H}, mu_s={mu_s}, mu_p={mu_p}, lam={lam}, alpha={alpha})...")
    dp_dx = (p_out - p_in) / L
    
    if sampling_type == 'grid':
        xy = generate_grid_points(nx, ny, L, H, margin=0.0, device='cpu')
    else:
        xy = generate_sobol_points(nx * ny, L, H, margin=0.0, device='cpu')
    
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    
    # Risoluzione numerica 1D ad alta fedeltà con Doppia Bisezione
    print("Risoluzione rigorosa delle equazioni di Giesekus (Doppia Bisezione)...")
    y_1d = np.linspace(0, H, 1000) # Aumentato a 1000 punti per un'integrazione perfetta
    
    g_dot_1d = np.zeros(len(y_1d))
    t_xx_1d = np.zeros(len(y_1d))
    t_xy_1d = np.zeros(len(y_1d))
    t_yy_1d = np.zeros(len(y_1d))
    
    for i, yi in enumerate(y_1d):
        tau_tot_target = dp_dx * (yi - H/2.0)
        g, txx, txy, tyy = solve_giesekus_point(tau_tot_target, mu_s, mu_p, lam, alpha)
        g_dot_1d[i] = g
        t_xx_1d[i] = txx
        t_xy_1d[i] = txy
        t_yy_1d[i] = tyy
    
    u_1d = cumulative_trapezoid(g_dot_1d, y_1d, initial=0.0)
    psi_1d = cumulative_trapezoid(u_1d, y_1d, initial=0.0)
    
    u_interp = interp1d(y_1d, u_1d, kind='cubic', bounds_error=False, fill_value="extrapolate")
    psi_interp = interp1d(y_1d, psi_1d, kind='cubic', bounds_error=False, fill_value="extrapolate")
    txx_interp = interp1d(y_1d, t_xx_1d, kind='cubic', bounds_error=False, fill_value="extrapolate")
    txy_interp = interp1d(y_1d, t_xy_1d, kind='cubic', bounds_error=False, fill_value="extrapolate")
    tyy_interp = interp1d(y_1d, t_yy_1d, kind='cubic', bounds_error=False, fill_value="extrapolate")
    
    y_np = y.numpy().flatten()
    u_base = torch.tensor(u_interp(y_np), dtype=torch.float32).view(-1, 1)
    v_base = torch.zeros_like(u_base)
    p_base = p_in + dp_dx * x
    psi_base = torch.tensor(psi_interp(y_np), dtype=torch.float32).view(-1, 1)
    
    tau_xx_base = torch.tensor(txx_interp(y_np), dtype=torch.float32).view(-1, 1)
    tau_xy_base = torch.tensor(txy_interp(y_np), dtype=torch.float32).view(-1, 1)
    tau_yy_base = torch.tensor(tyy_interp(y_np), dtype=torch.float32).view(-1, 1)
    
    u_max = torch.max(u_base).item()
    
    # ... Inserisci qui la logica di aggiunta del rumore (come nello script precedente)
    u_noisy = u_base.clone()
    v_noisy = v_base.clone()
    p_noisy = p_base.clone()
    psi_noisy = psi_base.clone()
    tau_xy_noisy = tau_xy_base.clone()
    tau_xx_noisy = tau_xx_base.clone()
    tau_yy_noisy = tau_yy_base.clone()
    
    dataset = {
        'coords': xy,
        'u': u_noisy, 'v': v_noisy, 'p': p_noisy, 'psi': psi_noisy,
        'tau_xx': tau_xx_noisy, 'tau_xy': tau_xy_noisy, 'tau_yy': tau_yy_noisy,
        'u_exact': u_base, 'p_exact': p_base, 'psi_exact': psi_base,
        'tau_xx_exact': tau_xx_base, 'tau_xy_exact': tau_xy_base, 'tau_yy_exact': tau_yy_base,
        'params': {
            'L': L, 'H': H, 'mu_s': mu_s, 'mu_p': mu_p, 'lam': lam, 'alpha': alpha,
            'noise_type': noise_type, 'noise_value': noise_value
        }
    }
    
    os.makedirs(save_dir, exist_ok=True)
    pt_path = os.path.join(save_dir, f"{filename}.pt")
    torch.save(dataset, pt_path)
    print(f"✅ Dataset salvato in PyTorch: {pt_path}")
    
    return dataset

if __name__ == "__main__":
    generate_giesekus_dataset(
        L=1.0, H=0.2, mu_s=0.001, mu_p=0.05, lam=10.0, alpha=0.45, p_in=0.1, #con parametri diversi è difficile apprezzare il plug flow
        nx=60, ny=30, noise_value=0.0, filename='giesekus_clean'
    )