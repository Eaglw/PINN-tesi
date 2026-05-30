import torch
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def verify_dataset(file_path):
    print("\n" + "="*50)
    print(f"VERIFICA DATASET VISCOELASTICO: {os.path.basename(file_path)}")
    print("="*50)
    
    if not os.path.exists(file_path):
        print(f"[ERRORE] File {file_path} non trovato.")
        return

    if file_path.endswith('.csv'):
        try:
            from load_comsol import load_comsol_csv
        except ImportError:
            import sys
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from load_comsol import load_comsol_csv
            
        COMSOL_PARAMS = {
            'mu_s': 0.005,   # Viscosità solvente [Pa·s]
            'mu_p': 0.005,   # Viscosità polimerica [Pa·s]
            'lam': 0.1,      # Tempo di rilassamento [s]
            'eps': 0.0,      # Parametro PTT
            'alpha': 0.0,    # Parametro Giesekus
            'rho': 1.0,      # Densità [kg/m³]
        }
        data = load_comsol_csv(file_path, COMSOL_PARAMS, device='cpu')
    else:
        data = torch.load(file_path, weights_only=False) if file_path.endswith('.pt') else None

    if data is None:
        print("[ERRORE] Caricamento fallito o formato non supportato.")
        return

    params = data.get('params', {})
    if 'scales' in data:
        L = data['scales']['H'] * (data['scales']['L'] / data['scales']['H']) # Dimensional length L
        H = data['scales']['H']
        u_max = data['scales']['U_ref']
    else:
        L, H = params.get('L', 1.0), params.get('H', 0.2)
        u_max = params.get('u_max', 0.0)
        
    mu_s, mu_p = params.get('mu_s', 0.0), params.get('mu_p', 0.0)
    lam = params.get('lam', 0.0)
    mu_tot = mu_s + mu_p

    print(f"\n--- PARAMETRI FISICI RILEVATI ---")
    print(f"L (Lunghezza): {L} m")
    print(f"H (Altezza):   {H} m")
    print(f"mu_s (Solvente): {mu_s} Pa·s")
    print(f"mu_p (Polimero): {mu_p} Pa·s")
    print(f"lambda (Rilassamento): {lam} s")
    print(f"u_max (Velocità max): {u_max:.4f} m/s")
    print(f"Viscosità totale (mu_tot): {mu_tot:.4f} Pa·s")

    print(f"\n--- EQUAZIONI DI GENERAZIONE (Oldroyd-B Poiseuille) ---")
    print(f"1. Velocità:   u(y) = 4 * u_max * y * (H - y) / H^2")
    print(f"2. Gradiente:  gamma_dot = du/dy = (4 * u_max / H^2) * (H - 2y)")
    print(f"3. Sforzo XY:  tau_xy = mu_p * gamma_dot")
    print(f"4. Sforzo XX:  tau_xx = 2 * lambda * mu_p * (gamma_dot^2)")
    print(f"5. Sforzo YY:  tau_yy = 0")
    print(f"6. Pressione:  dp/dx = - (8 * mu_tot * u_max) / H^2")

    dp_dx_teorico = -(8 * mu_tot * u_max) / (H**2)
    print(f"\nGradiente di pressione teorico (dp/dx): {dp_dx_teorico:.4f} Pa/m")

    xy = data['coords']
    u = data['u']
    p = data['p']
    tau_xx = data.get('tau_xx', None)
    tau_xy = data.get('tau_xy', None)
    
    n_points = xy.shape[0]
    x_coords = xy[:, 0].numpy()
    y_coords = xy[:, 1].numpy()
    x_unique = np.unique(x_coords)
    y_unique = np.unique(y_coords)
    nx, ny = len(x_unique), len(y_unique)

    # Estrazione profilo 1D al centro del canale (x* = L / 2H)
    x_mid = 0.5 * (L / H)
    dists = np.abs(x_coords - x_mid)
    min_dist = np.min(dists)
    tol = max(min_dist + 1e-5, 0.02 * (L / H))
    slice_idx = np.where(np.abs(x_coords - x_mid) < tol)[0]
    
    slice_y = y_coords[slice_idx]
    slice_u = u[slice_idx].view(-1).numpy()
    slice_txx = tau_xx[slice_idx].view(-1).numpy() if tau_xx is not None else None
    slice_txy = tau_xy[slice_idx].view(-1).numpy() if tau_xy is not None else None
    
    # Ordina per altezza y
    sort_idx = np.argsort(slice_y)
    slice_y = slice_y[sort_idx]
    slice_u = slice_u[sort_idx]
    if slice_txx is not None: slice_txx = slice_txx[sort_idx]
    if slice_txy is not None: slice_txy = slice_txy[sort_idx]

    # Plotting
    try:
        fig, axs = plt.subplots(2, 3, figsize=(18, 10))
        
        # Punti densi per la soluzione analitica
        y_dense = np.linspace(0, 1, 200)
        beta_val = mu_s / mu_tot if mu_tot > 0 else 0.0
        Wi_val = lam * u_max / H
        
        if not file_path.endswith('.csv'):
            X_grid = xy[:, 0].reshape(ny, nx).numpy()
            Y_grid = xy[:, 1].reshape(ny, nx).numpy()
            U_grid = u.reshape(ny, nx).numpy()
            P_grid = p.reshape(ny, nx).numpy()
            
            # 1. Mappa Velocità U
            im1 = axs[0, 0].pcolormesh(X_grid, Y_grid, U_grid, shading='auto', cmap='viridis')
            plt.colorbar(im1, ax=axs[0, 0])
            axs[0, 0].set_title("Velocità u(x,y)")
            
            # 2. Mappa Pressione P
            im2 = axs[0, 1].pcolormesh(X_grid, Y_grid, P_grid, shading='auto', cmap='plasma')
            plt.colorbar(im2, ax=axs[0, 1])
            axs[0, 1].set_title("Pressione p(x,y)")
    
            if tau_xx is not None and tau_xy is not None:
                TXX_grid = tau_xx.reshape(ny, nx).numpy()
                TXY_grid = tau_xy.reshape(ny, nx).numpy()
                
                # 3. Mappa Tau_xx
                im3 = axs[1, 0].pcolormesh(X_grid, Y_grid, TXX_grid, shading='auto', cmap='inferno')
                plt.colorbar(im3, ax=axs[1, 0])
                axs[1, 0].set_title("Sforzo Normale Polimerico tau_xx")
                
                # 4. Mappa Tau_xy
                im4 = axs[1, 1].pcolormesh(X_grid, Y_grid, TXY_grid, shading='auto', cmap='coolwarm')
                plt.colorbar(im4, ax=axs[1, 1])
                axs[1, 1].set_title("Sforzo di Taglio Polimerico tau_xy")
        else:
            import matplotlib.tri as tri
            triang = tri.Triangulation(x_coords, y_coords)
            
            # 1. Mappa Velocità U
            im1 = axs[0, 0].tripcolor(triang, u.view(-1).numpy(), cmap='viridis')
            plt.colorbar(im1, ax=axs[0, 0])
            axs[0, 0].set_title("Velocità u(x,y) [Adimensionale]")
            
            # 2. Mappa Pressione P
            im2 = axs[0, 1].tripcolor(triang, p.view(-1).numpy(), cmap='plasma')
            plt.colorbar(im2, ax=axs[0, 1])
            axs[0, 1].set_title("Pressione p(x,y) [Adimensionale]")
    
            if tau_xx is not None and tau_xy is not None:
                # 3. Mappa Tau_xx
                im3 = axs[1, 0].tripcolor(triang, tau_xx.view(-1).numpy(), cmap='inferno')
                plt.colorbar(im3, ax=axs[1, 0])
                axs[1, 0].set_title("Sforzo Normale Polimerico tau_xx [Adimensionale]")
                
                # 4. Mappa Tau_xy
                im4 = axs[1, 1].tripcolor(triang, tau_xy.view(-1).numpy(), cmap='coolwarm')
                plt.colorbar(im4, ax=axs[1, 1])
                axs[1, 1].set_title("Sforzo di Taglio Polimerico tau_xy [Adimensionale]")

        # 5. Profilo 1D Velocità (Riga 0, Colonna 2)
        axs[0, 2].plot(slice_u, slice_y, 'ro', label='Dataset / COMSOL', markersize=4, alpha=0.6)
        u_theory = 4 * y_dense * (1.0 - y_dense)
        axs[0, 2].plot(u_theory, y_dense, 'k--', label='Teorico Poiseuille', linewidth=2)
        axs[0, 2].set_title(f"Profilo Velocità u(y) a x*={x_mid:.2f}")
        axs[0, 2].set_xlabel("u*")
        axs[0, 2].set_ylabel("y*")
        axs[0, 2].grid(True, linestyle='--', alpha=0.5)
        axs[0, 2].legend()

        # 6. Profili 1D Sforzi (Riga 1, Colonna 2)
        if slice_txy is not None:
            axs[1, 2].plot(slice_txy, slice_y, 'bo', label='tau_xy (Dataset)', markersize=4, alpha=0.6)
            txy_theory = 4.0 * (1.0 - beta_val) * (1.0 - 2.0 * y_dense)
            axs[1, 2].plot(txy_theory, y_dense, 'b--', label='tau_xy (Teorico)', linewidth=2)
            
        if slice_txx is not None:
            axs[1, 2].plot(slice_txx, slice_y, 'go', label='tau_xx (Dataset)', markersize=4, alpha=0.6)
            txx_theory = 32.0 * Wi_val * (1.0 - beta_val) * ((1.0 - 2.0 * y_dense) ** 2)
            axs[1, 2].plot(txx_theory, y_dense, 'g--', label='tau_xx (Teorico)', linewidth=2)
            
        axs[1, 2].set_title(f"Profili Sforzi a x*={x_mid:.2f}")
        axs[1, 2].set_xlabel("Sforzi*")
        axs[1, 2].set_ylabel("y*")
        axs[1, 2].grid(True, linestyle='--', alpha=0.5)
        axs[1, 2].legend()

        plt.tight_layout()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        plot_path = os.path.join(script_dir, "../../plots/visco_verify.png")
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path)
        print(f"\n[OK] Plot di verifica salvato in: {os.path.abspath(plot_path)}")
        
    except Exception as e:
        print(f"[ERRORE] Errore durante il plotting: {e}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    verify_dataset(os.path.join(current_dir, "../../COMSOL/Oldroyd.csv"))
