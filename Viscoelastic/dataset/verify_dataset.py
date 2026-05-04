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
        p = data['p']
        tau_xx = data.get('tau_xx', None)
        tau_xy = data.get('tau_xy', None)
        u_exact = data.get('u_exact', None)
        tau_xx_exact = data.get('tau_xx_exact', None)
        tau_xy_exact = data.get('tau_xy_exact', None)
    else:
        df = pd.read_csv(file_path)
        xy = torch.tensor(df[['x', 'y']].values)
        u = torch.tensor(df['u'].values).reshape(-1, 1)
        p = torch.tensor(df['p'].values).reshape(-1, 1)
        tau_xx = torch.tensor(df['tau_xx'].values).reshape(-1, 1) if 'tau_xx' in df.columns else None
        tau_xy = torch.tensor(df['tau_xy'].values).reshape(-1, 1) if 'tau_xy' in df.columns else None
        u_exact = torch.tensor(df['u_exact'].values).reshape(-1, 1) if 'u_exact' in df.columns else None
        tau_xx_exact = torch.tensor(df['tau_xx_exact'].values).reshape(-1, 1) if 'tau_xx_exact' in df.columns else None
        tau_xy_exact = torch.tensor(df['tau_xy_exact'].values).reshape(-1, 1) if 'tau_xy_exact' in df.columns else None

    n_points = xy.shape[0]
    print(f"Total points in dataset: {n_points}")

    # Identifica le dimensioni della griglia per i plot 2D
    x_unique = np.unique(xy[:, 0].numpy())
    y_unique = np.unique(xy[:, 1].numpy())
    nx, ny = len(x_unique), len(y_unique)
    
    # Reshape dei dati per il plotting 2D
    try:
        X_grid = xy[:, 0].reshape(ny, nx).numpy()
        Y_grid = xy[:, 1].reshape(ny, nx).numpy()
        U_grid = u.reshape(ny, nx).numpy()
        P_grid = p.reshape(ny, nx).numpy()
        if tau_xx is not None:
            TauXX_grid = tau_xx.reshape(ny, nx).numpy()
            TauXY_grid = tau_xy.reshape(ny, nx).numpy()
        is_grid = True
    except Exception as e:
        is_grid = False
        print(f"Dataset sampling is not a regular grid or reshape failed: {e}")

    # Plotting
    if is_grid:
        fig, axs = plt.subplots(3, 2, figsize=(16, 12), gridspec_kw={'height_ratios': [1, 1, 1.5]})
        
        # 2D Velocity U
        im1 = axs[0, 0].pcolormesh(X_grid, Y_grid, U_grid, shading='auto', cmap='viridis')
        fig.colorbar(im1, ax=axs[0, 0], label='u (m/s)')
        axs[0, 0].set_title(f"Velocity Field (u) - {n_points} points")
        axs[0, 0].set_aspect('equal')

        # 2D Pressure P
        im2 = axs[0, 1].pcolormesh(X_grid, Y_grid, P_grid, shading='auto', cmap='plasma')
        fig.colorbar(im2, ax=axs[0, 1], label='p (Pa)')
        axs[0, 1].set_title("Pressure Field (p)")
        axs[0, 1].set_aspect('equal')
        
        if tau_xx is not None:
            # 2D Tau XX
            im3 = axs[1, 0].pcolormesh(X_grid, Y_grid, TauXX_grid, shading='auto', cmap='inferno')
            fig.colorbar(im3, ax=axs[1, 0], label='tau_xx')
            axs[1, 0].set_title("Polymeric Stress (tau_xx)")
            axs[1, 0].set_aspect('equal')

            # 2D Tau XY
            im4 = axs[1, 1].pcolormesh(X_grid, Y_grid, TauXY_grid, shading='auto', cmap='coolwarm')
            fig.colorbar(im4, ax=axs[1, 1], label='tau_xy')
            axs[1, 1].set_title("Polymeric Shear Stress (tau_xy)")
            axs[1, 1].set_aspect('equal')
        else:
            axs[1, 0].set_visible(False)
            axs[1, 1].set_visible(False)

        # Profiles (Cutlines)
        x_mid_val = x_unique[nx // 2]
        mask = np.abs(xy[:, 0].numpy() - x_mid_val) < 1e-5
        
        # Velocity Profile
        axs[2, 0].scatter(u[mask].numpy(), xy[mask, 1].numpy(), label='Noisy u', color='red', s=10)
        if u_exact is not None:
            axs[2, 0].plot(u_exact[mask].numpy(), xy[mask, 1].numpy(), label='Exact u', color='black')
        axs[2, 0].set_title(f"Velocity Profile at x={x_mid_val:.2f}")
        axs[2, 0].grid(True)
        axs[2, 0].legend()

        # Tau XX Profile
        if tau_xx is not None:
            axs[2, 1].scatter(tau_xx[mask].numpy(), xy[mask, 1].numpy(), label='Noisy tau_xx', color='purple', s=10)
            if tau_xx_exact is not None:
                axs[2, 1].plot(tau_xx_exact[mask].numpy(), xy[mask, 1].numpy(), label='Exact tau_xx', color='black')
            axs[2, 1].set_title(f"Tau_xx Profile at x={x_mid_val:.2f}")
            axs[2, 1].grid(True)
            axs[2, 1].legend()
        else:
            y_mid_val = y_unique[ny // 2]
            mask_p = np.abs(xy[:, 1].numpy() - y_mid_val) < 1e-5
            axs[2, 1].scatter(xy[mask_p, 0].numpy(), p[mask_p].numpy(), label='Noisy P', color='blue', s=10)
            axs[2, 1].set_title(f"Pressure Drop at y={y_mid_val:.2f}")
            axs[2, 1].grid(True)

    plt.tight_layout()
    plot_name = os.path.basename(file_path).split('.')[0] + "_verification.png"
    save_path = os.path.join(os.path.dirname(file_path), "../../plots", plot_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Plot di verifica salvato in: {save_path}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    files_to_verify = [
        os.path.join(current_dir, "oldroydb_noisy.pt"),
        os.path.join(current_dir, "oldroydb_clean.pt")
    ]
    
    for pt_file in files_to_verify:
        if os.path.exists(pt_file):
            verify_dataset(pt_file)
        else:
            print(f"File {pt_file} non trovato. Corri prima generate_dataset.py")
