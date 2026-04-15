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
        u_exact = data.get('u_exact', None)
        params = data.get('params', {})
    else:
        df = pd.read_csv(file_path)
        xy = torch.tensor(df[['x', 'y']].values)
        u = torch.tensor(df['u'].values).reshape(-1, 1)
        p = torch.tensor(df['p'].values).reshape(-1, 1)
        u_exact = torch.tensor(df['u_exact'].values).reshape(-1, 1) if 'u_exact' in df.columns else None
        params = {}

    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    
    # 1. Velocità Profilo (a x_medio)
    x_unique = torch.unique(xy[:, 0])
    x_mid = x_unique[len(x_unique)//2]
    mask = torch.abs(xy[:, 0] - x_mid) < 1e-5
    
    y_plot = xy[mask, 1]
    u_plot = u[mask]
    
    # Ordina per y
    idx = torch.argsort(y_plot)
    y_plot = y_plot[idx]
    u_plot = u_plot[idx]
    
    axs[0].scatter(u_plot.numpy(), y_plot.numpy(), label='Noisy Data', color='red', alpha=0.6)
    if u_exact is not None:
        u_ex_plot = u_exact[mask][idx]
        axs[0].plot(u_ex_plot.numpy(), y_plot.numpy(), label='Exact Profile', color='black', linewidth=2)
    
    axs[0].set_title(f"Velocity Profile at x={x_mid:.2f}")
    axs[0].set_xlabel("u (m/s)")
    axs[0].set_ylabel("y (m)")
    axs[0].legend()
    axs[0].grid(True)

    # 2. Pressione lungo x (a y_medio)
    y_unique = torch.unique(xy[:, 1])
    y_mid = y_unique[len(y_unique)//2]
    mask_p = torch.abs(xy[:, 1] - y_mid) < 1e-5
    
    x_p_plot = xy[mask_p, 0]
    p_plot = p[mask_p]
    
    # Ordina per x
    idx_p = torch.argsort(x_p_plot)
    x_p_plot = x_p_plot[idx_p]
    p_plot = p_plot[idx_p]
    
    axs[1].scatter(x_p_plot.numpy(), p_plot.numpy(), label='Noisy Pressure', color='blue', alpha=0.6)
    axs[1].set_title(f"Pressure drop at y={y_mid:.2f}")
    axs[1].set_xlabel("x (m)")
    axs[1].set_ylabel("Pressure (Pa)")
    axs[1].grid(True)

    plt.tight_layout()
    plot_name = os.path.basename(file_path).split('.')[0] + "_verification.png"
    save_path = os.path.join(os.path.dirname(file_path), "../../plots", plot_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Plot di verifica salvato in: {save_path}")
    # plt.show() # Rimosso per non bloccare lo script

if __name__ == "__main__":
    # Verifica il file generato
    pt_file = "Newtonian/data/poiseuille_noisy.pt"
    if os.path.exists(pt_file):
        verify_dataset(pt_file)
    else:
        print(f"File {pt_file} non trovato. Corri prima generate_dataset.py")
