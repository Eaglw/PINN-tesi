import torch
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def verify_giesekus_dataset(file_path):
    print("\n" + "="*60)
    print(f"VERIFICA DATASET GIESEKUS: {os.path.basename(file_path)}")
    print("="*60)
    
    if not os.path.exists(file_path):
        print(f"❌ Errore: File {file_path} non trovato.")
        return

    data = torch.load(file_path, weights_only=False) if file_path.endswith('.pt') else None
    if data is None:
        print("❌ Caricamento fallito o formato non supportato.")
        return

    # Estrazione Parametri
    params = data.get('params', {})
    L, H = params.get('L', 1.0), params.get('H', 0.2)
    mu_s, mu_p = params.get('mu_s', 0.0), params.get('mu_p', 0.0)
    lam = params.get('lam', 0.0)
    alpha = params.get('alpha', 0.0)
    mu_tot = mu_s + mu_p

    # Estrazione tensori
    xy = data['coords']
    u = data['u']
    p = data['p']
    tau_xx = data.get('tau_xx', None)
    tau_xy = data.get('tau_xy', None)
    tau_yy = data.get('tau_yy', None)
    
    n_points = xy.shape[0]
    x_unique = np.unique(xy[:, 0].numpy())
    y_unique = np.unique(xy[:, 1].numpy())
    nx, ny = len(x_unique), len(y_unique)

    # Plotting e Calcoli Analitici
    try:
        X_grid = xy[:, 0].reshape(ny, nx).numpy()
        Y_grid = xy[:, 1].reshape(ny, nx).numpy()
        U_grid = u.reshape(ny, nx).numpy()
        P_grid = p.reshape(ny, nx).numpy()
        
        # 1. Calcolo rigoroso del dp/dx direttamente dai dati di pressione generati
        dp = np.mean(P_grid[:, -1]) - np.mean(P_grid[:, 0])
        dx = x_unique[-1] - x_unique[0]
        dp_dx_effettivo = dp / dx if dx != 0 else 0.0
        
        # 2. Calcolo Parabola Newtoniana Esatta (Analitica)
        # u(y) = (-dp_dx / 2*mu) * y * (H - y)
        u_newton = (-dp_dx_effettivo / (2 * mu_tot)) * y_unique * (H - y_unique)
        
        # 3. Estrazione Profilo Giesekus al centro del canale
        mid_x_idx = nx // 2
        u_giesekus = U_grid[:, mid_x_idx]
        
        u_max_n = np.max(u_newton)
        u_max_g = np.max(u_giesekus)

        # Creazione Figure 2x4
        fig, axs = plt.subplots(2, 4, figsize=(22, 10))
        
        # --- RIGA 1 ---
        # 1. Mappa Velocità U
        im1 = axs[0, 0].pcolormesh(X_grid, Y_grid, U_grid, shading='auto', cmap='viridis')
        plt.colorbar(im1, ax=axs[0, 0])
        axs[0, 0].set_title("Mappa Velocità u(x,y)")
        
        # 2. Mappa Pressione P
        im2 = axs[0, 1].pcolormesh(X_grid, Y_grid, P_grid, shading='auto', cmap='plasma')
        plt.colorbar(im2, ax=axs[0, 1])
        axs[0, 1].set_title("Mappa Pressione p(x,y)")

        # 3. Profilo 1D Assoluto
        axs[0, 2].plot(u_newton, y_unique, 'k--', label='Newtoniano', linewidth=2)
        axs[0, 2].plot(u_giesekus, y_unique, 'b-', label='Giesekus', linewidth=2.5)
        axs[0, 2].set_title(f"Profilo u(y) - Valori Assoluti\nx = {x_unique[mid_x_idx]:.2f} m")
        axs[0, 2].set_xlabel("u [m/s]")
        axs[0, 2].set_ylabel("y [m]")
        axs[0, 2].grid(True, linestyle='--', alpha=0.7)
        axs[0, 2].legend()

        # 4. Profilo 1D Normalizzato (Per vedere lo Shear-Thinning)
        axs[0, 3].plot(u_newton / u_max_n, y_unique, 'k--', label='Newtoniano', linewidth=2)
        axs[0, 3].plot(u_giesekus / u_max_g, y_unique, 'r-', label='Giesekus', linewidth=3)
        axs[0, 3].fill_betweenx(y_unique, u_newton / u_max_n, u_giesekus / u_max_g, color='red', alpha=0.1)
        axs[0, 3].set_title(r"Profilo Normalizzato ($u / u_{max}$)" + "\nL'area rossa mostra l'appiattimento")
        axs[0, 3].set_xlabel(r"$u / u_{max}$")
        axs[0, 3].grid(True, linestyle='--', alpha=0.7)
        axs[0, 3].legend()

        # --- RIGA 2 ---
        if tau_xx is not None and tau_xy is not None and tau_yy is not None:
            TXX_grid = tau_xx.reshape(ny, nx).numpy()
            TXY_grid = tau_xy.reshape(ny, nx).numpy()
            TYY_grid = tau_yy.reshape(ny, nx).numpy()
            
            # 5. Mappa Tau_xx
            im5 = axs[1, 0].pcolormesh(X_grid, Y_grid, TXX_grid, shading='auto', cmap='inferno')
            plt.colorbar(im5, ax=axs[1, 0])
            axs[1, 0].set_title(r"Sforzo Normale $\tau_{xx}$")
            
            # 6. Mappa Tau_xy
            im6 = axs[1, 1].pcolormesh(X_grid, Y_grid, TXY_grid, shading='auto', cmap='coolwarm')
            plt.colorbar(im6, ax=axs[1, 1])
            axs[1, 1].set_title(r"Sforzo di Taglio $\tau_{xy}$")

            # 7. Mappa Tau_yy
            im7 = axs[1, 2].pcolormesh(X_grid, Y_grid, TYY_grid, shading='auto', cmap='magma')
            plt.colorbar(im7, ax=axs[1, 2])
            axs[1, 2].set_title(r"Sforzo Normale Secondario $\tau_{yy}$")

        # 8. Riquadro con metriche di confronto
        axs[1, 3].axis('off')
        info_text = (
            "--- METRICHE DI CONFRONTO ---\n\n"
            f"Parametri Reologici:\n"
            f"  - alpha (Mobilità): {alpha}\n"
            f"  - lambda (Rilassamento): {lam} s\n"
            f"  - mu_s (Solvente): {mu_s} Pa·s\n"
            f"  - mu_p (Polimero): {mu_p} Pa·s\n\n"
            f"Dinamica del Flusso:\n"
            f"  - dp/dx calcolato: {dp_dx_effettivo:.2f} Pa/m\n"
            f"  - u_max (Newton): {u_max_n:.4f} m/s\n"
            f"  - u_max (Giesekus): {u_max_g:.4f} m/s\n\n"
            f"L'effetto lubrificante (Shear-Thinning)\n"
            f"ha aumentato la velocità di picco del:\n"
            f"  + {((u_max_g - u_max_n)/u_max_n)*100:.1f} %"
        )
        axs[1, 3].text(0.05, 0.5, info_text, fontsize=12, va='center', family='monospace',
                       bbox=dict(facecolor='#f8f9fa', edgecolor='#dee2e6', boxstyle='round,pad=1'))

        plt.tight_layout()
        plot_path = os.path.join(os.path.dirname(file_path), "giesekus_verify_norm.png")
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, dpi=300)
        print(f"\n✅ Plot di verifica dettagliato salvato in: {plot_path}")
        print(info_text)
        
    except Exception as e:
        print(f"❌ Errore durante il plotting: {e}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Sostituisci con il file del tuo test "estremo" per vedere bene il plug-flow
    verify_giesekus_dataset(os.path.join(current_dir, "giesekus_clean.pt"))