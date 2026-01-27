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
    plt.scatter(x_data, y_data, s=60, color="white", alpha=0.4, label='Training data')
    if xp is not None:
        plt.scatter(xp, -0*torch.ones_like(xp), s=60, color="white", alpha=0.4, 
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
    
    # Errore Relativo Standard (diviso per valore locale) con masking
    rel_error = torch.zeros_like(T_true)
    mask = torch.abs(T_true) > 0.01
    if mask.sum() > 0:
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
        ax.scatter(xy_physics_np[:, 0], xy_physics_np[:, 1], s=5, color='white', alpha=0.6, label='Punti Fisica')
        ax.legend(loc='upper right')

    # 2. Errore Assoluto (Più robusto)
    ax = axes[1]
    c2 = ax.contourf(X_np, Y_np, abs_error.detach().cpu().numpy(), levels=50, cmap='magma')
    plt.colorbar(c2, ax=ax, label='Errore Assoluto')
    ax.set_title('Errore Assoluto |T_pred - T_true|')
    ax.set_xlabel('x')

    # 3. Errore Relativo (Normalizzato su max)
    ax = axes[2]
    # Usiamo vmin/vmax per evitare saturazione da outlier
    c3 = ax.contourf(X_np, Y_np, rel_error.detach().cpu().numpy(), levels=50, cmap='jet', vmin=0, vmax=10) 
    cbar = plt.colorbar(c3, ax=ax, label='% Errore (Norm. su max)')
    ax.set_title('Errore Relativo % (|err| / max|T_true|)')
    ax.set_xlabel('x')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True) # Ensure directory exists
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot2D_final_result(X, Y, T_true, T_pred, epoch, save_path, data_points=None, physics_points=None):
    """
    Generates a 2-column plot:
    Left: Solution u(x,y) with overlaid training points (Data & Physics).
    Right: Relative Error Map %.
    """
    # Calculate Relative Error Standard (diviso per valore locale) con masking
    abs_error = torch.abs(T_pred - T_true)
    rel_error = torch.zeros_like(T_true)
    mask = torch.abs(T_true) > 0.01
    if mask.sum() > 0:
        rel_error[mask] = (abs_error[mask] / torch.abs(T_true[mask])) * 100
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    X_np, Y_np = X.cpu().numpy(), Y.cpu().numpy()
    
    # 1. Solution + Points
    ax = axes[0]
    c1 = ax.contourf(X_np, Y_np, T_pred.detach().cpu().numpy(), levels=50, cmap='inferno')
    plt.colorbar(c1, ax=ax, label='Temp')
    ax.set_title(f'Prediction (Epoch {epoch})')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    # Overlay Points
    if physics_points is not None:
        xy_phys = physics_points.detach().cpu().numpy()
        ax.scatter(xy_phys[:, 0], xy_phys[:, 1], s=10, c='white', marker='x', alpha=0.6, label='Physics Points')
        
    if data_points is not None:
        xy_data = data_points.detach().cpu().numpy()
        ax.scatter(xy_data[:, 0], xy_data[:, 1], s=15, c='white', marker='o', alpha=0.8, label='Data Points')
        
    if physics_points is not None or data_points is not None:
        ax.legend(loc='upper right', framealpha=0.9)

    # 2. Relative Error
    ax = axes[1]
    c2 = ax.contourf(X_np, Y_np, rel_error.detach().cpu().numpy(), levels=50, cmap='jet', vmin=0, vmax=10)
    plt.colorbar(c2, ax=ax, label='% Error (Norm. su max)')
    ax.set_title('Relative Error % (|err| / max|T_true|)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot2D_unified_comparison(X, Y, T_true, model_results, hyperparams, save_path=None):
    """
    Generates a 2x2 grid of relative error maps.
    
    Args:
        X, Y: Meshgrid tensors.
        T_true: Analytical solution tensor.
        model_results: List of dictionaries [{'T_pred': tensor, 'label': str}, ...] (exactly 4 expected).
        hyperparams: Dictionary {'arch': str, 'epochs': int, 'act': str}.
        save_path: Path to save the plot.
    """
    if len(model_results) != 4:
        print("Warning: plot2D_unified_comparison expects exactly 4 model results for a 2x2 grid.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    X_np, Y_np = X.cpu().numpy(), Y.cpu().numpy()
    
    # Max for normalization
    T_max = torch.max(torch.abs(T_true)).item()
    
    arch = hyperparams.get('arch', 'N/A')
    epochs = hyperparams.get('epochs', 'N/A')
    act = hyperparams.get('act', 'N/A')
    
    fig.suptitle(f"Comparison: {arch} | Epochs: {epochs} | Activation: {act}", fontsize=18, fontweight='bold')

    for i, res in enumerate(model_results):
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        
        T_pred = res['T_pred']
        label = res['label']
        
        abs_error = torch.abs(T_pred - T_true)
        
        # Errore Relativo Standard con masking
        rel_error = torch.zeros_like(T_true)
        mask = torch.abs(T_true) > 0.01
        if mask.sum() > 0:
            rel_error[mask] = (abs_error[mask] / torch.abs(T_true[mask])) * 100
            
        rel_error_np = rel_error.detach().cpu().numpy()
        
        # Plot with individual colorbar
        c = ax.contourf(X_np, Y_np, rel_error_np, levels=50, cmap='jet', vmin=0, vmax=10)
        cbar = plt.colorbar(c, ax=ax)
        cbar.set_label('% Error (Norm. su max)', rotation=270, labelpad=15)
        
        ax.set_facecolor('lightgray')
        ax.set_title(label, fontsize=14)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_error_map_comparison(X, Y, T_true, T_preds, labels, save_path=None):
    """
    Plots side-by-side relative error maps for multiple models.
    """
    num_models = len(T_preds)
    fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 5))
    if num_models == 1:
        axes = [axes]
        
    X_np, Y_np = X.cpu().numpy(), Y.cpu().numpy()
    
    # Max for normalization
    T_max = torch.max(torch.abs(T_true)).item()
    
    for i, (T_pred, label) in enumerate(zip(T_preds, labels)):
        ax = axes[i]
        
        abs_error = torch.abs(T_pred - T_true)
        
        # Errore Relativo Standard con masking
        rel_error = torch.zeros_like(T_true)
        mask = torch.abs(T_true) > 0.01
        if mask.sum() > 0:
            rel_error[mask] = (abs_error[mask] / torch.abs(T_true[mask])) * 100
            
        rel_error_np = rel_error.detach().cpu().numpy()
        
        # Use vmin/vmax to handle outliers in relative error
        c = ax.contourf(X_np, Y_np, rel_error_np, levels=50, cmap='jet', vmin=0, vmax=10)
        plt.colorbar(c, ax=ax, label='% Error (Norm. su max)')
        ax.set_facecolor('lightgray') # Color excluded regions
        ax.set_title(f'{label} - Rel Error %')
        ax.set_xlabel('x')
        if i == 0:
            ax.set_ylabel('y')
            
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_loss_comparison(histories, labels, save_path=None, title="Loss Comparison"):
    """
    Plots overlapping loss curves from multiple training histories.
    """
    plt.figure(figsize=(10, 6))
    
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    
    for i, (history, label) in enumerate(zip(histories, labels)):
        color = colors[i % len(colors)]
        if 'total_loss' in history.losses:
            # Clean None values
            values = history.losses['total_loss']
            clean_values = [v if v is not None else np.nan for v in values]
            plt.plot(history.epochs, clean_values, label=f"{label} (Total)", color=color, linewidth=2)
    
    plt.yscale('log')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()