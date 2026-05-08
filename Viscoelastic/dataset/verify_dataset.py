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
        print(f"❌ Errore: File {file_path} non trovato.")
        return

    data = torch.load(file_path, weights_only=False) if file_path.endswith('.pt') else None
    if data is None:
        print("❌ Caricamento fallito o formato non supportato.")
        return

    params = data.get('params', {})
    L, H = params.get('L', 1.0), params.get('H', 0.2)
    mu_s, mu_p = params.get('mu_s', 0.0), params.get('mu_p', 0.0)
    lam = params.get('lam', 0.0)
    u_max = params.get('u_max', 0.0)
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
    x_unique = np.unique(xy[:, 0].numpy())
    y_unique = np.unique(xy[:, 1].numpy())
    nx, ny = len(x_unique), len(y_unique)

    # Plotting
    try:
        X_grid = xy[:, 0].reshape(ny, nx).numpy()
        Y_grid = xy[:, 1].reshape(ny, nx).numpy()
        U_grid = u.reshape(ny, nx).numpy()
        P_grid = p.reshape(ny, nx).numpy()
        
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
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

        plt.tight_layout()
        plot_path = os.path.join(os.path.dirname(file_path), "../../plots/visco_verify.png")
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path)
        print(f"\n✅ Plot di verifica salvato in: {plot_path}")
        
    except Exception as e:
        print(f"❌ Errore durante il plotting: {e}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    verify_dataset(os.path.join(current_dir, "oldroydb_clean.pt"))
