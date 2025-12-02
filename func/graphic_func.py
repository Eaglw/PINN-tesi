import os
from PIL import Image
import matplotlib.pyplot as plt
import torch
import numpy as np

def save_gif_PIL(outfile, files, fps=5, loop=0, delete_files=False):
    "Helper function for saving GIFs, modificata per rimuovere i file"
    imgs = [Image.open(file) for file in files]
    imgs[0].save(fp=outfile, format='GIF', append_images=imgs[1:], save_all=True, duration=int(1000/fps), loop=loop)
    if delete_files:
        for file in files:
            try:
                os.remove(file)
            except Exception as e:
                print(f"Warning: unable to delete file {file}. Error: {e}")

def plot_result(i,x,y,x_data,y_data,yh,xp=None):
    "Pretty plot training results"
    plt.figure(figsize=(8,4))
    plt.plot(x,y, color="grey", linewidth=2, alpha=0.8, label="Exact solution")
    plt.plot(x,yh, color="tab:blue", linewidth=4, alpha=0.8, label="Neural network prediction")
    plt.scatter(x_data, y_data, s=60, color="tab:orange", alpha=0.4, label='Training data')
    if xp is not None:
        plt.scatter(xp, -0*torch.ones_like(xp), s=60, color="tab:green", alpha=0.4, 
                    label='Physics loss training locations')
    l = plt.legend(loc=(1.01,0.34), frameon=False, fontsize="large")
    plt.setp(l.get_texts(), color="k")
    #plt.xlim(-0.05, 1.05)
    #plt.ylim(-1.1, 1.1)
    plt.text(1.065,0.7,"Training step: %i"%(i+1),fontsize="xx-large",color="k")
    plt.axis("off")

def plot2D_comparison(X, Y, T_true, T_pred, epoch, save_path, physics_points=None):
    """Genera grafici side-by-side: Predizione, Errore Assoluto, Errore Relativo.
    Rinominata da plot_comparison per uso generale.
    Aggiunge la visualizzazione dei punti di collocazione della fisica se forniti."""
    
    # Calcolo Errori
    abs_error = torch.abs(T_pred - T_true)
    
    # Errore Relativo (Gestione della divisione per zero)
    # Calcoliamo l'errore relativo solo dove T_true è significativo (> 0.01)
    mask = torch.abs(T_true) > 0.01
    rel_error = torch.zeros_like(T_true)
    rel_error[mask] = (abs_error[mask] / torch.abs(T_true[mask])) * 100
    
    # Setup plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    X_np, Y_np = X.cpu().numpy(), Y.cpu().numpy()
    
    # 1. Soluzione Predetta
    ax = axes[0]
    c1 = ax.contourf(X_np, Y_np, T_pred.detach().cpu().numpy(), levels=50, cmap='inferno')
    plt.colorbar(c1, ax=ax, label='Temp')
    ax.set_title(f'Predizione (Epoch {epoch})')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    # Aggiungi i punti della fisica
    if physics_points is not None:
        xy_physics_np = physics_points.detach().cpu().numpy()
        ax.scatter(xy_physics_np[:, 0], xy_physics_np[:, 1], s=5, color='cyan', alpha=0.6, label='Punti Fisica')
        ax.legend(loc='upper right')

    # 2. Errore Assoluto (Più robusto)
    ax = axes[1]
    c2 = ax.contourf(X_np, Y_np, abs_error.detach().cpu().numpy(), levels=50, cmap='magma')
    plt.colorbar(c2, ax=ax, label='Errore Assoluto')
    ax.set_title('Errore Assoluto |T_pred - T_true|')
    ax.set_xlabel('x')

    # 3. Errore Relativo (Mascherato)
    ax = axes[2]
    # Usiamo vmin/vmax per evitare saturazione da outlier
    c3 = ax.contourf(X_np, Y_np, rel_error.detach().cpu().numpy(), levels=50, cmap='jet', vmin=0, vmax=10) 
    cbar = plt.colorbar(c3, ax=ax, label='% Errore')
    # Coloriamo di grigio le zone escluse (dove T_true ~ 0)
    ax.set_facecolor('lightgray') 
    ax.set_title('Errore Relativo % (dove T_true > 0.01)')
    ax.set_xlabel('x')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True) # Ensure directory exists
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
