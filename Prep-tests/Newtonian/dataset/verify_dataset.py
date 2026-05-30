import torch
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def verify_dataset(file_path):
    print(f"Loading dataset from {file_path} for verification...")
    
    if file_path.endswith('.pt'):
        data = torch.load(file_path)
        xy = data['coords']
        u = data['u']
        v = data['v']
        p = data['p']
        u_exact = data.get('u_exact', None)
        params = data.get('params', {})
    else:
        df = pd.read_csv(file_path)
        xy = torch.tensor(df[['x', 'y']].values)
        u = torch.tensor(df['u'].values).reshape(-1, 1)
        v = torch.tensor(df['v'].values).reshape(-1, 1)
        p = torch.tensor(df['p'].values).reshape(-1, 1)
        u_exact = torch.tensor(df['u_exact'].values).reshape(-1, 1) if 'u_exact' in df.columns else None
        params = {}

    n_points = xy.shape[0]
    print(f"Total points in dataset: {n_points}")

    # Identifica le dimensioni della griglia per i plot 2D
    x_unique = np.unique(xy[:, 0].numpy())
    y_unique = np.unique(xy[:, 1].numpy())
    nx, ny = len(x_unique), len(y_unique)
    
    # Reshape dei dati per il plotting 2D (se possibile)
    try:
        # indexing='xy' in meshgrid significa che l'array piatto ha ny righe di nx colonne
        X_grid = xy[:, 0].reshape(ny, nx).numpy()
        Y_grid = xy[:, 1].reshape(ny, nx).numpy()
        U_grid = u.reshape(ny, nx).numpy()
        P_grid = p.reshape(ny, nx).numpy()
        is_grid = True
    except Exception as e:
        is_grid = False
        print(f"Dataset sampling is not a regular grid or reshape failed: {e}")

    # Plotting
    if is_grid:
        fig, axs = plt.subplots(2, 2, figsize=(16, 8), gridspec_kw={'height_ratios': [1, 2]})
        
        # 2D Velocity U
        im1 = axs[0, 0].pcolormesh(X_grid, Y_grid, U_grid, shading='auto', cmap='viridis')
        fig.colorbar(im1, ax=axs[0, 0], label='u (m/s)')
        axs[0, 0].set_title(f"2D Velocity Field (u) - {n_points} points")
        axs[0, 0].set_aspect('equal')

        # 2D Pressure P
        im2 = axs[0, 1].pcolormesh(X_grid, Y_grid, P_grid, shading='auto', cmap='plasma')
        fig.colorbar(im2, ax=axs[0, 1], label='p (Pa)')
        axs[0, 1].set_title("2D Pressure Field (p)")
        axs[0, 1].set_aspect('equal')

        # Profiles (Cutlines)
        # 1. Velocità Profilo (a x_medio)
        x_mid_val = x_unique[nx // 2]
        mask = np.abs(xy[:, 0].numpy() - x_mid_val) < 1e-5
        axs[1, 0].scatter(u[mask].numpy(), xy[mask, 1].numpy(), label='Noisy Data', color='red', s=10)
        if u_exact is not None:
            axs[1, 0].plot(u_exact[mask].numpy(), xy[mask, 1].numpy(), label='Exact', color='black')
        axs[1, 0].set_title(f"Velocity Profile at x={x_mid_val:.2f}")
        axs[1, 0].grid(True)
        axs[1, 0].legend()

        # 2. Pressione lungo x (a y_medio)
        y_mid_val = y_unique[ny // 2]
        mask_p = np.abs(xy[:, 1].numpy() - y_mid_val) < 1e-5
        axs[1, 1].scatter(xy[mask_p, 0].numpy(), p[mask_p].numpy(), label='Noisy P', color='blue', s=10)
        axs[1, 1].set_title(f"Pressure Drop at y={y_mid_val:.2f}")
        axs[1, 1].grid(True)
    else:
        # Fallback se non è una griglia
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        # ... (codice precedente per cutline se serve) ...

    plt.tight_layout()
    plot_name = os.path.basename(file_path).split('.')[0] + "_verification.png"
    save_path = os.path.join(os.path.dirname(file_path), "../../plots", plot_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Plot di verifica salvato in: {save_path}")
    # plt.show() # Rimosso per non bloccare lo script

if __name__ == "__main__":
    # La cartella del dataset è quella dove si trova lo script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Lista di file da verificare (usa percorsi relativi alla posizione dello script o assoluti)
    files_to_verify = [
        os.path.join(current_dir, "poiseuille_noisy.pt"),
        os.path.join(current_dir, "poiseuille_clean.pt")
    ]
    
    for pt_file in files_to_verify:
        if os.path.exists(pt_file):
            verify_dataset(pt_file)
        else:
            print(f"File {pt_file} non trovato. Corri prima generate_dataset.py")
