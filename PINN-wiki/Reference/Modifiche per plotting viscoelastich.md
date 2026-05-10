# Walkthrough: Migliorie Visualizzazione Viscoelastic

## Obiettivo
Revisione e miglioramento completo della pipeline di visualizzazione e logging dei risultati per il caso Viscoelastic Oldroyd-B.

---

## File Modificati

### 1. [graphic_func.py](file:///c:/Users/eaglw/Documents/PINN%20tesi/func/graphic_func.py) — Nuove funzioni + Fix

#### Nuove Funzioni
- **`plot2D_viscoelastic_final()`** — Genera un plot a griglia 5×3 con Predizione | Exact | Errore Relativo per ognuno dei 5 campi fisici (u, p, τ_xx, τ_xy, τ_yy). Colormap differenziata: inferno per velocità, viridis per pressione, plasma per stress. Color limits condivisi tra Pred e Exact per confronto diretto.
- **`plot2D_viscoelastic_comparison()`** — Error map multi-campo tra i goal (PurePhys, Phys+Data, SoloData). Griglia n_fields × n_models per confrontare dove ogni goal eccelle o fallisce.
- **`_compute_rel_error()`** — Helper condiviso per il calcolo dell'errore relativo con masking.

#### Fix: vmax Adattivo
> Sostituito `vmax=10` (hardcoded) con `vmax=max(np.percentile(95°), 1.0)` in **tutte** le funzioni di error map. Questo risolve il problema dell'error map completamente satura (tutta rossa) nei casi con errore >> 10%.

```diff:graphic_func.py
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

def plot2D_comparison(X, Y, T_true, T_pred, epoch, save_path, physics_points=None, val_label='Value'):
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
    
    # Setup plot: 3 rows, 1 col for stacked visualization (optimal for channels)
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    X_np, Y_np = X.detach().cpu().numpy(), Y.detach().cpu().numpy()
    
    # 1. Soluzione Predetta
    ax = axes[0]
    c1 = ax.contourf(X_np, Y_np, T_pred.detach().cpu().numpy(), levels=50, cmap='inferno')
    plt.colorbar(c1, ax=ax, label=val_label)
    ax.set_title(f'Predizione (Epoch {epoch})')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal', adjustable='box')
    
    # Aggiungi i punti della fisica
    if physics_points is not None and len(physics_points) > 0:
        xy_physics_np = physics_points.detach().cpu().numpy()
        ax.scatter(xy_physics_np[:, 0], xy_physics_np[:, 1], s=5, color='white', marker='x', alpha=0.6, label='Punti Fisica')
        ax.legend(loc='upper right')

    # 2. Errore Assoluto (Più robusto)
    ax = axes[1]
    c2 = ax.contourf(X_np, Y_np, abs_error.detach().cpu().numpy(), levels=50, cmap='magma')
    plt.colorbar(c2, ax=ax, label='Errore Assoluto')
    ax.set_title('Errore Assoluto |T_pred - T_true|')
    ax.set_xlabel('x')
    ax.set_aspect('equal', adjustable='box')

    # 3. Errore Relativo (Locale)
    ax = axes[2]
    # Usiamo vmin/vmax per evitare saturazione da outlier
    c3 = ax.contourf(X_np, Y_np, rel_error.detach().cpu().numpy(), levels=50, cmap='jet', vmin=0, vmax=10) 
    cbar = plt.colorbar(c3, ax=ax, label='% Errore Relativo (|err|/|T_true|)')
    ax.set_title('Errore Relativo % (|err| / |T_true|)')
    ax.set_xlabel('x')
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True) # Ensure directory exists
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot2D_final_result(X, Y, T_true, T_pred, epoch, save_path, internal_points=None, boundary_points=None, physics_points=None, val_label='Value'):
    """
    Generates a 2-column plot:
    Left: Solution u(x,y) with overlaid training points (Internal, Boundary & Physics).
    Right: Relative Error Map %.
    """
    # Calculate Relative Error Standard (diviso per valore locale) con masking
    abs_error = torch.abs(T_pred - T_true)
    rel_error = torch.zeros_like(T_true)
    mask = torch.abs(T_true) > 0.01
    if mask.sum() > 0:
        rel_error[mask] = (abs_error[mask] / torch.abs(T_true[mask])) * 100
    
    # Setup plot: 2 rows, 1 col for stacked visualization (optimal for channels)
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    X_np, Y_np = X.detach().cpu().numpy(), Y.detach().cpu().numpy()
    
    # 1. Solution + Points
    ax = axes[0]
    c1 = ax.contourf(X_np, Y_np, T_pred.detach().cpu().numpy(), levels=50, cmap='inferno')
    plt.colorbar(c1, ax=ax, label=val_label)
    ax.set_title(f'Prediction (Epoch {epoch})')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal', adjustable='box')
    
    # Overlay Points
    if physics_points is not None and len(physics_points) > 0:
        xy_phys = physics_points.detach().cpu().numpy()
        ax.scatter(xy_phys[:, 0], xy_phys[:, 1], s=10, c='white', marker='x', alpha=0.5, label='Physics Points')
        
    if internal_points is not None and len(internal_points) > 0:
        xy_int = internal_points.detach().cpu().numpy()
        ax.scatter(xy_int[:, 0], xy_int[:, 1], s=15, c='cyan', marker='o', alpha=0.8, edgecolor='k', label='Internal Points')
        
    if boundary_points is not None and len(boundary_points) > 0:
        xy_bc = boundary_points.detach().cpu().numpy()
        ax.scatter(xy_bc[:, 0], xy_bc[:, 1], s=20, c='red', marker='s', alpha=0.8, edgecolor='k', label='Boundary Points')
        
    if physics_points is not None or internal_points is not None or boundary_points is not None:
        ax.legend(loc='upper right', framealpha=0.9, fontsize='small')

    # 2. Relative Error
    ax = axes[1]
    c2 = ax.contourf(X_np, Y_np, rel_error.detach().cpu().numpy(), levels=50, cmap='jet', vmin=0, vmax=10)
    plt.colorbar(c2, ax=ax, label='% Relative Error (|err|/|T_true|)')
    ax.set_title('Relative Error % (|err| / |T_true|)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot2D_unified_comparison(X, Y, T_true, model_results, hyperparams, save_path=None):
    """
    Generates a dynamic grid of relative error maps.
    
    Args:
        X, Y: Meshgrid tensors.
        T_true: Analytical solution tensor.
        model_results: List of dictionaries [{'T_pred': tensor, 'label': str}, ...].
            Supporta qualsiasi numero di risultati (1, 2, 3, 4, ...).
            Il layout della griglia si adatta automaticamente.
        hyperparams: Dictionary {'arch': str, 'epochs': int, 'act': str}.
        save_path: Path to save the plot.
    """
    n = len(model_results)
    if n == 0:
        print("Warning: plot2D_unified_comparison called with 0 model results. Skipping.")
        return

    # Griglia dinamica: max 2 colonne, righe calcolate di conseguenza
    cols = min(n, 2)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 6 * rows), squeeze=False)
    X_np, Y_np = X.detach().cpu().numpy(), Y.detach().cpu().numpy()
    
    arch = hyperparams.get('arch', 'N/A')
    epochs = hyperparams.get('epochs', 'N/A')
    act = hyperparams.get('act', 'N/A')
    
    fig.suptitle(f"Comparison: {arch} | Epochs: {epochs} | Activation: {act}", fontsize=18, fontweight='bold')

    for i, res in enumerate(model_results):
        row = i // cols
        col = i % cols
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
        cbar.set_label('% Relative Error (|err|/|T_true|)', rotation=270, labelpad=15)
        
        ax.set_facecolor('lightgray')
        ax.set_title(label, fontsize=14)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal', adjustable='box')

    # Nasconde gli assi vuoti se il numero di risultati non riempie la griglia
    for i in range(n, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].set_visible(False)

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
    # Stacked vertically for channel geometries
    fig, axes = plt.subplots(num_models, 1, figsize=(12, 5 * num_models))
    if num_models == 1:
        axes = [axes]
        
    X_np, Y_np = X.detach().cpu().numpy(), Y.detach().cpu().numpy()
    
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
        plt.colorbar(c, ax=ax, label='% Relative Error (|err|/|T_true|)')
        ax.set_facecolor('lightgray') # Color excluded regions
        ax.set_title(f'{label} - Rel Error %')
        ax.set_xlabel('x')
        if i == 0:
            ax.set_ylabel('y')
        ax.set_aspect('equal', adjustable='box')
            
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
===
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

def plot2D_comparison(X, Y, T_true, T_pred, epoch, save_path, physics_points=None, val_label='Value'):
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
    
    # Setup plot: 3 rows, 1 col for stacked visualization (optimal for channels)
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    X_np, Y_np = X.detach().cpu().numpy(), Y.detach().cpu().numpy()
    
    # 1. Soluzione Predetta
    ax = axes[0]
    c1 = ax.contourf(X_np, Y_np, T_pred.detach().cpu().numpy(), levels=50, cmap='inferno')
    plt.colorbar(c1, ax=ax, label=val_label)
    ax.set_title(f'Predizione (Epoch {epoch})')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal', adjustable='box')
    
    # Aggiungi i punti della fisica
    if physics_points is not None and len(physics_points) > 0:
        xy_physics_np = physics_points.detach().cpu().numpy()
        ax.scatter(xy_physics_np[:, 0], xy_physics_np[:, 1], s=5, color='white', marker='x', alpha=0.6, label='Punti Fisica')
        ax.legend(loc='upper right')

    # 2. Errore Assoluto (Più robusto)
    ax = axes[1]
    c2 = ax.contourf(X_np, Y_np, abs_error.detach().cpu().numpy(), levels=50, cmap='magma')
    plt.colorbar(c2, ax=ax, label='Errore Assoluto')
    ax.set_title('Errore Assoluto |T_pred - T_true|')
    ax.set_xlabel('x')
    ax.set_aspect('equal', adjustable='box')

    # 3. Errore Relativo (Locale)
    ax = axes[2]
    # Usiamo vmin/vmax per evitare saturazione da outlier
    rel_error_np = rel_error.detach().cpu().numpy()
    vmax_adaptive = max(np.percentile(rel_error_np, 95), 1.0)  # Almeno 1% per evitare scale degeneri
    c3 = ax.contourf(X_np, Y_np, rel_error_np, levels=50, cmap='jet', vmin=0, vmax=vmax_adaptive) 
    cbar = plt.colorbar(c3, ax=ax, label='% Errore Relativo (|err|/|T_true|)')
    ax.set_title('Errore Relativo % (|err| / |T_true|)')
    ax.set_xlabel('x')
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True) # Ensure directory exists
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot2D_final_result(X, Y, T_true, T_pred, epoch, save_path, internal_points=None, boundary_points=None, physics_points=None, val_label='Value'):
    """
    Generates a 2-column plot:
    Left: Solution u(x,y) with overlaid training points (Internal, Boundary & Physics).
    Right: Relative Error Map %.
    """
    # Calculate Relative Error Standard (diviso per valore locale) con masking
    abs_error = torch.abs(T_pred - T_true)
    rel_error = torch.zeros_like(T_true)
    mask = torch.abs(T_true) > 0.01
    if mask.sum() > 0:
        rel_error[mask] = (abs_error[mask] / torch.abs(T_true[mask])) * 100
    
    # Setup plot: 2 rows, 1 col for stacked visualization (optimal for channels)
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    X_np, Y_np = X.detach().cpu().numpy(), Y.detach().cpu().numpy()
    
    # 1. Solution + Points
    ax = axes[0]
    c1 = ax.contourf(X_np, Y_np, T_pred.detach().cpu().numpy(), levels=50, cmap='inferno')
    plt.colorbar(c1, ax=ax, label=val_label)
    ax.set_title(f'Prediction (Epoch {epoch})')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal', adjustable='box')
    
    # Overlay Points
    if physics_points is not None and len(physics_points) > 0:
        xy_phys = physics_points.detach().cpu().numpy()
        ax.scatter(xy_phys[:, 0], xy_phys[:, 1], s=10, c='white', marker='x', alpha=0.5, label='Physics Points')
        
    if internal_points is not None and len(internal_points) > 0:
        xy_int = internal_points.detach().cpu().numpy()
        ax.scatter(xy_int[:, 0], xy_int[:, 1], s=15, c='cyan', marker='o', alpha=0.8, edgecolor='k', label='Internal Points')
        
    if boundary_points is not None and len(boundary_points) > 0:
        xy_bc = boundary_points.detach().cpu().numpy()
        ax.scatter(xy_bc[:, 0], xy_bc[:, 1], s=20, c='red', marker='s', alpha=0.8, edgecolor='k', label='Boundary Points')
        
    if physics_points is not None or internal_points is not None or boundary_points is not None:
        ax.legend(loc='upper right', framealpha=0.9, fontsize='small')

    # 2. Relative Error
    ax = axes[1]
    rel_error_np_final = rel_error.detach().cpu().numpy()
    vmax_adaptive = max(np.percentile(rel_error_np_final, 95), 1.0)
    c2 = ax.contourf(X_np, Y_np, rel_error_np_final, levels=50, cmap='jet', vmin=0, vmax=vmax_adaptive)
    plt.colorbar(c2, ax=ax, label='% Relative Error (|err|/|T_true|)')
    ax.set_title('Relative Error % (|err| / |T_true|)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot2D_unified_comparison(X, Y, T_true, model_results, hyperparams, save_path=None):
    """
    Generates a dynamic grid of relative error maps.
    
    Args:
        X, Y: Meshgrid tensors.
        T_true: Analytical solution tensor.
        model_results: List of dictionaries [{'T_pred': tensor, 'label': str}, ...].
            Supporta qualsiasi numero di risultati (1, 2, 3, 4, ...).
            Il layout della griglia si adatta automaticamente.
        hyperparams: Dictionary {'arch': str, 'epochs': int, 'act': str}.
        save_path: Path to save the plot.
    """
    n = len(model_results)
    if n == 0:
        print("Warning: plot2D_unified_comparison called with 0 model results. Skipping.")
        return

    # Griglia dinamica: max 2 colonne, righe calcolate di conseguenza
    cols = min(n, 2)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 6 * rows), squeeze=False)
    X_np, Y_np = X.detach().cpu().numpy(), Y.detach().cpu().numpy()
    
    arch = hyperparams.get('arch', 'N/A')
    epochs = hyperparams.get('epochs', 'N/A')
    act = hyperparams.get('act', 'N/A')
    
    fig.suptitle(f"Comparison: {arch} | Epochs: {epochs} | Activation: {act}", fontsize=18, fontweight='bold')

    for i, res in enumerate(model_results):
        row = i // cols
        col = i % cols
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
        
        # Plot with individual colorbar — vmax adattivo
        vmax_adaptive = max(np.percentile(rel_error_np, 95), 1.0)
        c = ax.contourf(X_np, Y_np, rel_error_np, levels=50, cmap='jet', vmin=0, vmax=vmax_adaptive)
        cbar = plt.colorbar(c, ax=ax)
        cbar.set_label('% Relative Error (|err|/|T_true|)', rotation=270, labelpad=15)
        
        ax.set_facecolor('lightgray')
        ax.set_title(label, fontsize=14)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal', adjustable='box')

    # Nasconde gli assi vuoti se il numero di risultati non riempie la griglia
    for i in range(n, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].set_visible(False)

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
    # Stacked vertically for channel geometries
    fig, axes = plt.subplots(num_models, 1, figsize=(12, 5 * num_models))
    if num_models == 1:
        axes = [axes]
        
    X_np, Y_np = X.detach().cpu().numpy(), Y.detach().cpu().numpy()
    
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
        vmax_adaptive = max(np.percentile(rel_error_np, 95), 1.0)
        c = ax.contourf(X_np, Y_np, rel_error_np, levels=50, cmap='jet', vmin=0, vmax=vmax_adaptive)
        plt.colorbar(c, ax=ax, label='% Relative Error (|err|/|T_true|)')
        ax.set_facecolor('lightgray') # Color excluded regions
        ax.set_title(f'{label} - Rel Error %')
        ax.set_xlabel('x')
        if i == 0:
            ax.set_ylabel('y')
        ax.set_aspect('equal', adjustable='box')
            
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

def _compute_rel_error(pred, exact):
    """Calcola errore relativo percentuale con masking per valori piccoli."""
    abs_error = torch.abs(pred - exact)
    rel_error = torch.zeros_like(exact)
    mask = torch.abs(exact) > 0.01
    if mask.sum() > 0:
        rel_error[mask] = (abs_error[mask] / torch.abs(exact[mask])) * 100
    return rel_error

def plot2D_viscoelastic_final(X, Y, fields_pred, fields_exact, epoch, save_path,
                              internal_points=None, boundary_points=None,
                              physics_points=None):
    """
    Plot multi-campo finale per il caso viscoelastico.
    Genera una griglia con: Predizione | Soluzione Esatta | Errore Relativo
    per ogni campo fisico (u, p, τ_xx, τ_xy, τ_yy).
    
    Args:
        X, Y: Meshgrid tensors (CPU).
        fields_pred: Dict {'u': tensor, 'p': tensor, 'tau_xx': ..., 'tau_xy': ..., 'tau_yy': ...}
        fields_exact: Dict con le stesse chiavi.
        epoch: Numero di epoche totali.
        save_path: Path per salvare la figura.
    """
    field_names = ['u', 'p', 'tau_xx', 'tau_xy', 'tau_yy']
    field_labels = ['u (Velocity)', 'p (Pressure)', 'τ_xx', 'τ_xy', 'τ_yy']
    cmaps_field = ['inferno', 'viridis', 'plasma', 'plasma', 'plasma']
    
    n_fields = len(field_names)
    fig, axes = plt.subplots(n_fields, 3, figsize=(18, 4 * n_fields))
    X_np, Y_np = X.detach().cpu().numpy(), Y.detach().cpu().numpy()
    
    fig.suptitle(f'Viscoelastic PINN — Final Results (Epoch {epoch})', fontsize=18, fontweight='bold', y=0.995)
    
    for i, (fname, flabel, cmap) in enumerate(zip(field_names, field_labels, cmaps_field)):
        pred = fields_pred.get(fname)
        exact = fields_exact.get(fname)
        
        if pred is None or exact is None:
            for j in range(3):
                axes[i, j].set_visible(False)
            continue
        
        pred_np = pred.detach().cpu().numpy()
        exact_np = exact.detach().cpu().numpy()
        
        # Shared color limits tra pred e exact
        vmin_shared = min(pred_np.min(), exact_np.min())
        vmax_shared = max(pred_np.max(), exact_np.max())
        
        # Col 0: Predizione
        ax = axes[i, 0]
        c = ax.contourf(X_np, Y_np, pred_np, levels=50, cmap=cmap, vmin=vmin_shared, vmax=vmax_shared)
        plt.colorbar(c, ax=ax, label=flabel)
        ax.set_title(f'{flabel} — Prediction')
        ax.set_ylabel('y')
        ax.set_aspect('equal', adjustable='box')
        
        # Overlay punti solo sulla prima riga
        if i == 0:
            if physics_points is not None and len(physics_points) > 0:
                xy_p = physics_points.detach().cpu().numpy()
                ax.scatter(xy_p[:, 0], xy_p[:, 1], s=5, c='white', marker='x', alpha=0.4, label='Physics')
            if internal_points is not None and len(internal_points) > 0:
                xy_i = internal_points.detach().cpu().numpy()
                ax.scatter(xy_i[:, 0], xy_i[:, 1], s=10, c='cyan', marker='o', alpha=0.6, edgecolor='k', linewidth=0.3, label='Data')
            if boundary_points is not None and len(boundary_points) > 0:
                xy_b = boundary_points.detach().cpu().numpy()
                ax.scatter(xy_b[:, 0], xy_b[:, 1], s=15, c='red', marker='s', alpha=0.7, edgecolor='k', linewidth=0.3, label='BC')
            ax.legend(loc='upper right', fontsize='x-small', framealpha=0.8)
        
        # Col 1: Soluzione Esatta
        ax = axes[i, 1]
        c = ax.contourf(X_np, Y_np, exact_np, levels=50, cmap=cmap, vmin=vmin_shared, vmax=vmax_shared)
        plt.colorbar(c, ax=ax, label=flabel)
        ax.set_title(f'{flabel} — Exact')
        ax.set_aspect('equal', adjustable='box')
        
        # Col 2: Errore Relativo
        ax = axes[i, 2]
        rel_err = _compute_rel_error(pred.cpu(), exact.cpu())
        rel_err_np = rel_err.numpy()
        vmax_err = max(np.percentile(rel_err_np, 95), 1.0)
        c = ax.contourf(X_np, Y_np, rel_err_np, levels=50, cmap='jet', vmin=0, vmax=vmax_err)
        plt.colorbar(c, ax=ax, label='% Relative Error')
        ax.set_title(f'{flabel} — Rel. Error %')
        ax.set_aspect('equal', adjustable='box')
        
        # Label x solo sull'ultima riga
        if i == n_fields - 1:
            for j in range(3):
                axes[i, j].set_xlabel('x')
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot2D_viscoelastic_comparison(X, Y, fields_exact, model_results_multi, hyperparams, save_path=None):
    """
    Comparison multi-campo tra diversi goal (PurePhys, Phys+Data, SoloData)
    per tutti i campi fisici.
    
    Args:
        X, Y: Meshgrid tensors.
        fields_exact: Dict {'u': tensor, 'p': tensor, 'tau_xx': ..., ...}
        model_results_multi: List of dicts:
            [{'label': 'PurePhys', 'fields': {'u': tensor, 'p': tensor, ...}}, ...]
        hyperparams: Dict per il suptitle.
        save_path: Path per salvare.
    """
    field_names = ['u', 'p', 'tau_xx', 'tau_xy', 'tau_yy']
    field_labels = ['u (Velocity)', 'p (Pressure)', 'τ_xx', 'τ_xy', 'τ_yy']
    
    n_models = len(model_results_multi)
    n_fields = len(field_names)
    
    fig, axes = plt.subplots(n_fields, n_models, figsize=(6 * n_models, 3.5 * n_fields), squeeze=False)
    X_np, Y_np = X.detach().cpu().numpy(), Y.detach().cpu().numpy()
    
    arch = hyperparams.get('arch', 'N/A')
    epochs = hyperparams.get('epochs', 'N/A')
    act = hyperparams.get('act', 'N/A')
    fig.suptitle(f'Relative Error Comparison | {arch} | E={epochs} | {act}', fontsize=16, fontweight='bold')
    
    for row, (fname, flabel) in enumerate(zip(field_names, field_labels)):
        exact = fields_exact.get(fname)
        if exact is None:
            for col in range(n_models):
                axes[row, col].set_visible(False)
            continue
        
        for col, mres in enumerate(model_results_multi):
            ax = axes[row, col]
            pred = mres['fields'].get(fname)
            label = mres['label']
            
            if pred is None:
                ax.set_visible(False)
                continue
            
            rel_err = _compute_rel_error(pred.cpu(), exact.cpu())
            rel_err_np = rel_err.numpy()
            vmax_err = max(np.percentile(rel_err_np, 95), 1.0)
            
            c = ax.contourf(X_np, Y_np, rel_err_np, levels=50, cmap='jet', vmin=0, vmax=vmax_err)
            cbar = plt.colorbar(c, ax=ax)
            cbar.set_label('%', rotation=0, labelpad=10)
            
            if row == 0:
                ax.set_title(f'{label}', fontsize=13, fontweight='bold')
            
            ax.set_ylabel(f'{flabel}' if col == 0 else '')
            ax.set_aspect('equal', adjustable='box')
            
            if row == n_fields - 1:
                ax.set_xlabel('x')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
```

---

### 2. [logging_utils.py](file:///c:/Users/eaglw/Documents/PINN%20tesi/func/logging_utils.py) — Metriche Multi-Campo

#### Nuova Funzione
- **`compute_viscoelastic_metrics()`** — Calcola L2 Relative Error e Max Relative Error per ogni campo fisico separatamente (u, p, τ_xx, τ_xy, τ_yy). Ritorna un dict di tuple `{fname: (l2, max)}`.

#### CSV Esteso
- Aggiunte colonne: `L2_u`, `Max_u`, `L2_p`, `Max_p`, `L2_tau_xx`, `Max_tau_xx`, `L2_tau_xy`, `Max_tau_xy`, `L2_tau_yy`, `Max_tau_yy`
- Retrocompatibilità: le colonne legacy `L2_Relative_Error` e `Max_Relative_Error_Peak` sono mantenute.

```diff:logging_utils.py
import os
import csv
import torch
import numpy as np

def compute_metrics(model, xy_grid_flat, T_grid_true):
    """
    Computes L2 Relative Error and Max Relative Error Peak.
    
    IMPORTANTE: Questa funzione assume che model(x) restituisca un singolo output
    scalare per punto (shape (N, 1) o (N,)). Per modelli multi-output come
    ViscoelasticCombinedModel (che produce [psi, p, tau_xx, tau_xy, tau_yy]),
    è NECESSARIO wrappare il modello con VelocityInferenceWrapper prima di
    chiamare questa funzione, altrimenti le dimensioni non matchano.
    
    Args:
        model: Trained PyTorch model (single-output o wrapped).
        xy_grid_flat: Tensor of shape (N, 2) containing grid points.
        T_grid_true: Tensor of shape (Nx, Ny) or (N,) containing analytical solution.
    
    Returns:
        l2_rel_error (float): Global L2 relative error norm (ratio).
        max_rel_error_peak (float): Maximum pointwise relative error (percentage).
    """
    model.eval()
    with torch.no_grad():
        # Ensure input has the same dtype as the model weights
        dtype = next(model.parameters()).dtype
        T_pred = model(xy_grid_flat.to(dtype))
        
    # Ensure shapes match (flatten both) and use analytical solution's dtype for metrics
    T_pred_flat = T_pred.view(-1).to(T_grid_true.dtype)
    T_true_flat = T_grid_true.view(-1)
    
    # L2 Relative Error
    # ||u_pred - u_true||_2 / ||u_true||_2
    l2_error = torch.norm(T_pred_flat - T_true_flat, 2)
    l2_ref = torch.norm(T_true_flat, 2)
    
    # Handle division by zero for L2
    if l2_ref > 1e-10:
        l2_rel_error = (l2_error / l2_ref).item()
    else:
        l2_rel_error = 0.0 # Should unlikely happen for Heat Eq solution 
    
    # Max Relative Error Peak
    # Using the same mask logic as in graphic_func.py to avoid division by small numbers
    abs_error = torch.abs(T_pred_flat - T_true_flat)
    mask = torch.abs(T_true_flat) > 0.01
    
    rel_error = torch.zeros_like(T_true_flat)
    
    # Check if mask has any valid values to avoid empty tensor operations
    if mask.sum() > 0:
        # Calculate percentage error
        rel_error[mask] = (abs_error[mask] / torch.abs(T_true_flat[mask])) * 100
        max_rel_error_peak = torch.max(rel_error).item()
    else:
        max_rel_error_peak = 0.0 
        
    return l2_rel_error, max_rel_error_peak

def update_results_csv(file_path, data_dict):
    """
    Appends a row of results to the CSV file.
    
    Args:
        file_path: Path to the CSV file.
        data_dict: Dictionary containing the data to log. 
                   Keys must match the specified columns.
    """
    fieldnames = [
        'Timestamp', 'Max_Relative_Error_Peak', 'Architecture', 'Activation_Func', 'Epochs', 'Run_Type',
        'Optimizer', 'Learning_Rate', 'Loss_Total', 'Loss_Physics', 
        'Loss_Boundary', 'Loss_Data', 'L2_Relative_Error', 'Seed', 'n_points', 'Loss_Weight'
    ]
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    file_exists = os.path.exists(file_path)
    
    try:
        with open(file_path, mode='a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            # Write the row
            writer.writerow(data_dict)
            
    except Exception as e:
        print(f"Error updating CSV log: {e}")

def extract_hyperparams_from_path(path):
    """
    Extracts hyperparameters from a directory path following the naming convention:
    'L<arch>_E<epochs>_<activation>'
    
    Args:
        path: Path string.
        
    Returns:
        tuple: (architecture, epochs, activation)
    """
    parts = os.path.normpath(path).split(os.sep)
    target = None
    # Look for the segment that follows our convention
    for p in reversed(parts):
        if p.startswith('L') and '_E' in p:
            target = p
            break
            
    if not target:
        return "N/A", "N/A", "N/A"
        
    try:
        # Example: L2_50x4_1_E20000_GELU
        # Find index of _E
        idx_e = target.find('_E')
        arch = target[1:idx_e] # 2_50x4_1
        
        rest = target[idx_e+2:] # 20000_GELU
        if '_' in rest:
            split_rest = rest.split('_')
            epochs = split_rest[0]
            activation = split_rest[1]
        else:
            epochs = rest
            activation = "N/A"
            
        return arch, epochs, activation
    except Exception:
        return "Error", "Error", "Error"
===
import os
import csv
import torch
import numpy as np

def compute_metrics(model, xy_grid_flat, T_grid_true):
    """
    Computes L2 Relative Error and Max Relative Error Peak.
    
    IMPORTANTE: Questa funzione assume che model(x) restituisca un singolo output
    scalare per punto (shape (N, 1) o (N,)). Per modelli multi-output come
    ViscoelasticCombinedModel (che produce [psi, p, tau_xx, tau_xy, tau_yy]),
    è NECESSARIO wrappare il modello con VelocityInferenceWrapper prima di
    chiamare questa funzione, altrimenti le dimensioni non matchano.
    
    Args:
        model: Trained PyTorch model (single-output o wrapped).
        xy_grid_flat: Tensor of shape (N, 2) containing grid points.
        T_grid_true: Tensor of shape (Nx, Ny) or (N,) containing analytical solution.
    
    Returns:
        l2_rel_error (float): Global L2 relative error norm (ratio).
        max_rel_error_peak (float): Maximum pointwise relative error (percentage).
    """
    model.eval()
    with torch.no_grad():
        # Ensure input has the same dtype as the model weights
        dtype = next(model.parameters()).dtype
        T_pred = model(xy_grid_flat.to(dtype))
        
    # Ensure shapes match (flatten both) and use analytical solution's dtype for metrics
    T_pred_flat = T_pred.view(-1).to(T_grid_true.dtype)
    T_true_flat = T_grid_true.view(-1)
    
    # L2 Relative Error
    # ||u_pred - u_true||_2 / ||u_true||_2
    l2_error = torch.norm(T_pred_flat - T_true_flat, 2)
    l2_ref = torch.norm(T_true_flat, 2)
    
    # Handle division by zero for L2
    if l2_ref > 1e-10:
        l2_rel_error = (l2_error / l2_ref).item()
    else:
        l2_rel_error = 0.0 # Should unlikely happen for Heat Eq solution 
    
    # Max Relative Error Peak
    # Using the same mask logic as in graphic_func.py to avoid division by small numbers
    abs_error = torch.abs(T_pred_flat - T_true_flat)
    mask = torch.abs(T_true_flat) > 0.01
    
    rel_error = torch.zeros_like(T_true_flat)
    
    # Check if mask has any valid values to avoid empty tensor operations
    if mask.sum() > 0:
        # Calculate percentage error
        rel_error[mask] = (abs_error[mask] / torch.abs(T_true_flat[mask])) * 100
        max_rel_error_peak = torch.max(rel_error).item()
    else:
        max_rel_error_peak = 0.0 
        
    return l2_rel_error, max_rel_error_peak

def compute_viscoelastic_metrics(model, physics_problem, xy_grid_flat, fields_exact_flat, Ny_dom, Nx_dom):
    """
    Calcola L2 Relative Error e Max Relative Error per ogni campo fisico
    del modello viscoelastico: u, p, tau_xx, tau_xy, tau_yy.
    
    Args:
        model: ViscoelasticCombinedModel trainato.
        physics_problem: ViscoelasticPhysics instance (per ricavare u da psi).
        xy_grid_flat: Tensor (N, 2) con i punti della griglia.
        fields_exact_flat: Dict con tensori (Ny, Nx) per ogni campo:
            {'u': ..., 'p': ..., 'tau_xx': ..., 'tau_xy': ..., 'tau_yy': ...}
        Ny_dom, Nx_dom: Dimensioni della griglia.
        
    Returns:
        Dict con coppie (l2_rel, max_rel) per ogni campo:
            {'u': (l2, max), 'p': (l2, max), 'tau_xx': (l2, max), ...}
    """
    model.eval()
    dtype = next(model.parameters()).dtype
    x_input = xy_grid_flat.clone().to(dtype).requires_grad_(True)
    
    with torch.set_grad_enabled(True):
        u_pred, v_pred, p_pred, tau_pred = physics_problem.get_velocity(model, x_input)
        out = model(x_input)
        tau_xx_pred = out[:, 2:3]
        tau_xy_pred = out[:, 3:4]
        tau_yy_pred = out[:, 4:5]
    
    preds = {
        'u': u_pred.detach().cpu().view(-1),
        'p': p_pred.detach().cpu().view(-1),
        'tau_xx': tau_xx_pred.detach().cpu().view(-1),
        'tau_xy': tau_xy_pred.detach().cpu().view(-1),
        'tau_yy': tau_yy_pred.detach().cpu().view(-1),
    }
    
    metrics = {}
    for fname, pred_flat in preds.items():
        exact_grid = fields_exact_flat.get(fname)
        if exact_grid is None:
            metrics[fname] = (0.0, 0.0)
            continue
        
        true_flat = exact_grid.view(-1).to(pred_flat.dtype)
        
        # L2 Relative Error
        l2_error = torch.norm(pred_flat - true_flat, 2)
        l2_ref = torch.norm(true_flat, 2)
        l2_rel = (l2_error / l2_ref).item() if l2_ref > 1e-10 else 0.0
        
        # Max Relative Error
        abs_error = torch.abs(pred_flat - true_flat)
        mask = torch.abs(true_flat) > 0.01
        rel_error = torch.zeros_like(true_flat)
        if mask.sum() > 0:
            rel_error[mask] = (abs_error[mask] / torch.abs(true_flat[mask])) * 100
            max_rel = torch.max(rel_error).item()
        else:
            max_rel = 0.0
        
        metrics[fname] = (l2_rel, max_rel)
    
    return metrics

def update_results_csv(file_path, data_dict):
    """
    Appends a row of results to the CSV file.
    
    Args:
        file_path: Path to the CSV file.
        data_dict: Dictionary containing the data to log. 
                   Keys must match the specified columns.
    """
    fieldnames = [
        'Timestamp', 'Max_Relative_Error_Peak', 'Architecture', 'Activation_Func', 'Epochs', 'Run_Type',
        'Optimizer', 'Learning_Rate', 'Loss_Total', 'Loss_Physics', 
        'Loss_Boundary', 'Loss_Data', 'L2_Relative_Error', 'Max_Relative_Error_Peak',
        'L2_u', 'Max_u', 'L2_p', 'Max_p',
        'L2_tau_xx', 'Max_tau_xx', 'L2_tau_xy', 'Max_tau_xy', 'L2_tau_yy', 'Max_tau_yy',
        'Seed', 'n_points', 'Loss_Weight'
    ]
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    file_exists = os.path.exists(file_path)
    
    try:
        with open(file_path, mode='a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            # Write the row
            writer.writerow(data_dict)
            
    except Exception as e:
        print(f"Error updating CSV log: {e}")

def extract_hyperparams_from_path(path):
    """
    Extracts hyperparameters from a directory path following the naming convention:
    'L<arch>_E<epochs>_<activation>'
    
    Args:
        path: Path string.
        
    Returns:
        tuple: (architecture, epochs, activation)
    """
    parts = os.path.normpath(path).split(os.sep)
    target = None
    # Look for the segment that follows our convention
    for p in reversed(parts):
        if p.startswith('L') and '_E' in p:
            target = p
            break
            
    if not target:
        return "N/A", "N/A", "N/A"
        
    try:
        # Example: L2_50x4_1_E20000_GELU
        # Find index of _E
        idx_e = target.find('_E')
        arch = target[1:idx_e] # 2_50x4_1
        
        rest = target[idx_e+2:] # 20000_GELU
        if '_' in rest:
            split_rest = rest.split('_')
            epochs = split_rest[0]
            activation = split_rest[1]
        else:
            epochs = rest
            activation = "N/A"
            
        return arch, epochs, activation
    except Exception:
        return "Error", "Error", "Error"
```

---

### 3. [history_tracker.py](file:///c:/Users/eaglw/Documents/PINN%20tesi/func/history_tracker.py) — Loss History Improvements

#### Nuovi parametri di `plot_losses()`:
| Parametro | Tipo | Descrizione |
|-----------|------|-------------|
| `phase_markers` | `List[Dict]` | Linee verticali per il cambio di fase nel Staged Training |
| `smoothing_alpha` | `Float` | EMA smoothing (0 = off, 0.95 = forte) sovrapposto alle curve originali |
| `active_loss_keys` | `Set[str]` | Filtra le loss con peso 0 dalla visualizzazione (es. `data_loss` in PurePhys) |

```diff:history_tracker.py
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import numpy as np

class TrainingHistory:
    """
    Una classe per registrare e visualizzare l'andamento delle loss durante il training.
    """
    def __init__(self):
        self.epochs = []
        self.losses = {} # Dizionario di liste: {'total_loss': [v1, v2...], 'pde_loss': [None, ..., v100...]}
        self.lr_history = []

    def extend(self, other):
        """
        Concatena un'altra TrainingHistory a questa.
        Gestisce l'offset delle epoche per rendere la sequenza continua.
        """
        if not other.epochs:
            return
            
        last_epoch = self.epochs[-1] if self.epochs else -1
        # Offset delle epoche per farle seguire l'ultima registrata
        self.epochs.extend([e + last_epoch + 1 for e in other.epochs])
        self.lr_history.extend(other.lr_history)
        
        # Sincronizza le chiavi di tutte le loss
        all_keys = set(self.losses.keys()).union(set(other.losses.keys()))
        current_len_before = len(self.epochs) - len(other.epochs)
        
        for name in all_keys:
            if name not in self.losses:
                # Se la chiave è nuova per 'self', riempiamo il passato con None
                self.losses[name] = [None] * current_len_before
            
            if name in other.losses:
                self.losses[name].extend(other.losses[name])
            else:
                # Se la chiave manca in 'other', riempiamo la nuova sezione con None
                self.losses[name].extend([None] * len(other.epochs))

    def update(self, epoch, loss_dict, lr=None):
        """
        Registra i valori delle loss per un dato 'epoch'.
        """
        self.epochs.append(epoch)
        
        # Gestione Learning Rate: assicuriamoci che sia un float o None
        if lr is not None:
            lr = lr.item() if hasattr(lr, "item") else lr
        elif 'lr' in loss_dict:
            lr = loss_dict['lr']
            lr = lr.item() if hasattr(lr, 'item') else lr
        self.lr_history.append(lr)
        
        # 1. Identifica tutte le chiavi di loss viste finora
        current_keys = set(loss_dict.keys())
        known_keys = set(self.losses.keys())
        all_keys = current_keys.union(known_keys)
        
        for name in all_keys:
            if name not in self.losses:
                self.losses[name] = [None] * (len(self.epochs) - 1)
            
            if name in loss_dict:
                val = loss_dict[name]
                val = val.item() if hasattr(val, 'item') else val
            else:
                val = None
            
            self.losses[name].append(val)

    def plot_losses(self, warmup_epoch=0, adam_epochs=None, save_path=None, experiment_name="", show_plot=True, skip_epochs=0):
        """
        Genera un grafico con l'andamento di tutte le loss registrate.
        
        Arguments:
            skip_epochs: Numero di epoche iniziali da non visualizzare nel grafico.
        """
        has_lbfgs = adam_epochs is not None and any(e >= adam_epochs for e in self.epochs)
        
        if has_lbfgs:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={'width_ratios': [3, 1]})
            fig.subplots_adjust(wspace=0.3)
        else:
            plt.figure(figsize=(8, 4))
            ax1 = plt.gca()
            ax2 = None

        def plot_on_ax(ax, epoch_range_indices, title_suffix=""):
            # Filtro per skip_epochs
            epoch_range_indices = [i for i in epoch_range_indices if self.epochs[i] >= skip_epochs]
            if not epoch_range_indices: return

            for name, values in self.losses.items():
                if name.startswith('grad_') or name.startswith('weight_'): continue
                
                r_epochs = [self.epochs[i] for i in epoch_range_indices]
                r_values = [values[i] if values[i] is not None else np.nan for i in epoch_range_indices]
                
                if all(np.isnan(r_values)): continue

                # Total loss pesata (linea spessa), componenti pure (linee sottili)
                if name == "total_loss":
                    label = f"{name} (weighted)"
                    linewidth = 2.5
                    alpha = 1.0
                else:
                    label = f"{name} (pure)"
                    linewidth = 1.2
                    alpha = 0.8

                ax.plot(r_epochs, r_values, linewidth=linewidth, label=label, alpha=alpha)
            
            # Disegno linee verticali per i cambi di Learning Rate
            if len(self.lr_history) > 0:
                first_lr_vline = True
                for i in range(1, len(epoch_range_indices)):
                    idx_curr = epoch_range_indices[i]
                    idx_prev = epoch_range_indices[i-1]
                    
                    if idx_curr >= len(self.lr_history) or idx_prev >= len(self.lr_history):
                        continue
                        
                    lr_curr = self.lr_history[idx_curr]
                    lr_prev = self.lr_history[idx_prev]
                    
                    # Confronto robusto per cambi di LR (escludendo i None)
                    if lr_curr is not None and lr_prev is not None and not np.isclose(lr_curr, lr_prev, rtol=1e-8, atol=1e-12):
                        label = "LR Change" if first_lr_vline else None
                        # Linea più visibile: nera tratteggiata con alpha maggiore
                        ax.axvline(self.epochs[idx_curr], color="black", linestyle="--", alpha=0.3, linewidth=1, label=label)
                        first_lr_vline = False

            ax.set_title(f'Loss {title_suffix}')
            ax.set_xlabel('Epoch/Iter')
            ax.set_ylabel('Loss')
            ax.set_yscale('log')
            ax.grid(True, which="both", ls="--", alpha=0.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        if has_lbfgs:
            adam_indices = [i for i, e in enumerate(self.epochs) if e < adam_epochs]
            lbfgs_indices = [i for i, e in enumerate(self.epochs) if e >= adam_epochs]
            
            if adam_indices:
                plot_on_ax(ax1, adam_indices, "(Adam Phase)")
                if warmup_epoch != 0 and warmup_epoch >= skip_epochs:
                    ax1.axvline(warmup_epoch, color="r", linestyle="--", label="End Warmup")
                ax1.legend(loc='upper right', frameon=False, fontsize="x-small")

            if lbfgs_indices:
                lbfgs_plot_indices = [adam_indices[-1]] + lbfgs_indices if adam_indices else lbfgs_indices
                plot_on_ax(ax2, lbfgs_plot_indices, "(L-BFGS Refinement)")
                ax2.set_xlabel('Iter')
        else:
            plot_on_ax(ax1, range(len(self.epochs)), f"- {experiment_name}")
            if warmup_epoch != 0 and warmup_epoch >= skip_epochs:
                ax1.axvline(warmup_epoch, color="r", linestyle="--", label="End Warmup")
            ax1.legend(loc='upper right', frameon=False, fontsize="small")

        if save_path:
            dir_name = os.path.dirname(save_path)
            if dir_name: os.makedirs(dir_name, exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
        
        if show_plot: plt.show()
        plt.close()

    def plot_gradients(self, save_path=None, experiment_name="", show_plot=True):
        grad_keys = [k for k in self.losses.keys() if k.startswith('grad_')]
        if not grad_keys: return

        plt.figure(figsize=(8, 4))
        for name in grad_keys:
            values = self.losses[name]
            clean_values = [v if v is not None else np.nan for v in values]
            valid_indices = [i for i, v in enumerate(clean_values) if not np.isnan(v)]
            if valid_indices:
                plt.plot([self.epochs[i] for i in valid_indices], [clean_values[i] for i in valid_indices], label=name, marker='o', markersize=2)
        
        plt.title(f'Gradient Norms - {experiment_name}')
        plt.yscale('log')
        plt.grid(True, ls="--", alpha=0.5)
        plt.legend(loc='upper right', frameon=False)
        if save_path:
            dir_name = os.path.dirname(save_path)
            if dir_name: os.makedirs(dir_name, exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
        if show_plot: plt.show()
        plt.close()

    def plot_weights(self, save_path=None, experiment_name="", show_plot=True):
        weight_keys = [k for k in self.losses.keys() if k.startswith('weight_')]
        if not weight_keys: return

        plt.figure(figsize=(8, 4))
        for name in weight_keys:
            values = self.losses[name]
            clean_values = [v if v is not None else np.nan for v in values]
            valid_indices = [i for i, v in enumerate(clean_values) if not np.isnan(v)]
            if valid_indices:
                plt.plot([self.epochs[i] for i in valid_indices], [clean_values[i] for i in valid_indices], label=name.replace('weight_', 'lambda_'), linewidth=2)
        
        plt.title(f'Evolution of Loss Weights - {experiment_name}')
        plt.yscale('log')
        plt.grid(True, ls="--", alpha=0.5)
        plt.legend(loc='best', frameon=False)
        if save_path:
            dir_name = os.path.dirname(save_path)
            if dir_name: os.makedirs(dir_name, exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
        if show_plot: plt.show()
        plt.close()

def compute_pinn_loss(model, x_data, y_data, x_bc=None, y_bc=None, physics_loss_fn=None, x_physics=None, ic_loss_fn=None, physics_problem=None, lambda_data=1.0, lambda_bc=1.0, lambda_physics=1.0, **kwargs):
    """
    Computes the components of the PINN loss.
    COMPONENTS IN 'loss_dict' ARE PURE RESIDUALS (UNWEIGHTED).
    'total_loss' IS WEIGHTED.
    """
    loss_dict = {}
    total_loss = 0.0
    mse_loss = nn.MSELoss()
    
    if x_data is not None and y_data is not None and x_data.numel() > 0:
        y_pred = model(x_data)
        data_loss = mse_loss(y_pred, y_data)
        loss_dict['data_loss'] = data_loss
        total_loss += lambda_data * data_loss

    if physics_problem is not None and x_bc is not None and y_bc is not None and x_bc.numel() > 0:
        bc_loss_val = physics_problem.boundary_loss(model, x_bc, y_bc)
        loss_dict['bc_loss'] = bc_loss_val
        total_loss += lambda_bc * bc_loss_val
    elif x_bc is not None and y_bc is not None and x_bc.numel() > 0:
        bc_loss_val = mse_loss(model(x_bc), y_bc)
        loss_dict['bc_loss'] = bc_loss_val
        total_loss += lambda_bc * bc_loss_val
    
    if physics_problem is not None and x_physics is not None:
        pde_loss = physics_problem.residual(model, x_physics)
        loss_dict['pde_loss'] = pde_loss
        total_loss += lambda_physics * pde_loss
    elif physics_loss_fn is not None:
        if x_physics is not None:
            if not x_physics.requires_grad: x_physics.requires_grad_(True)
            pde_loss = physics_loss_fn(model, x_physics, **kwargs)
        else:
            pde_loss = physics_loss_fn(model, **kwargs)
        loss_dict['pde_loss'] = pde_loss
        total_loss += lambda_physics * pde_loss
        
    if ic_loss_fn is not None:
        ic_loss = ic_loss_fn(model, **kwargs)
        loss_dict['ic_loss'] = ic_loss
        total_loss += ic_loss
        
    loss_dict['total_loss'] = total_loss
    return total_loss, loss_dict
===
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import numpy as np

class TrainingHistory:
    """
    Una classe per registrare e visualizzare l'andamento delle loss durante il training.
    """
    def __init__(self):
        self.epochs = []
        self.losses = {} # Dizionario di liste: {'total_loss': [v1, v2...], 'pde_loss': [None, ..., v100...]}
        self.lr_history = []

    def extend(self, other):
        """
        Concatena un'altra TrainingHistory a questa.
        Gestisce l'offset delle epoche per rendere la sequenza continua.
        """
        if not other.epochs:
            return
            
        last_epoch = self.epochs[-1] if self.epochs else -1
        # Offset delle epoche per farle seguire l'ultima registrata
        self.epochs.extend([e + last_epoch + 1 for e in other.epochs])
        self.lr_history.extend(other.lr_history)
        
        # Sincronizza le chiavi di tutte le loss
        all_keys = set(self.losses.keys()).union(set(other.losses.keys()))
        current_len_before = len(self.epochs) - len(other.epochs)
        
        for name in all_keys:
            if name not in self.losses:
                # Se la chiave è nuova per 'self', riempiamo il passato con None
                self.losses[name] = [None] * current_len_before
            
            if name in other.losses:
                self.losses[name].extend(other.losses[name])
            else:
                # Se la chiave manca in 'other', riempiamo la nuova sezione con None
                self.losses[name].extend([None] * len(other.epochs))

    def update(self, epoch, loss_dict, lr=None):
        """
        Registra i valori delle loss per un dato 'epoch'.
        """
        self.epochs.append(epoch)
        
        # Gestione Learning Rate: assicuriamoci che sia un float o None
        if lr is not None:
            lr = lr.item() if hasattr(lr, "item") else lr
        elif 'lr' in loss_dict:
            lr = loss_dict['lr']
            lr = lr.item() if hasattr(lr, 'item') else lr
        self.lr_history.append(lr)
        
        # 1. Identifica tutte le chiavi di loss viste finora
        current_keys = set(loss_dict.keys())
        known_keys = set(self.losses.keys())
        all_keys = current_keys.union(known_keys)
        
        for name in all_keys:
            if name not in self.losses:
                self.losses[name] = [None] * (len(self.epochs) - 1)
            
            if name in loss_dict:
                val = loss_dict[name]
                val = val.item() if hasattr(val, 'item') else val
            else:
                val = None
            
            self.losses[name].append(val)

    def plot_losses(self, warmup_epoch=0, adam_epochs=None, save_path=None, experiment_name="", show_plot=True, skip_epochs=0, phase_markers=None, smoothing_alpha=0.0, active_loss_keys=None):
        """
        Genera un grafico con l'andamento di tutte le loss registrate.
        
        Arguments:
            skip_epochs: Numero di epoche iniziali da non visualizzare nel grafico.
            phase_markers: Lista di dict [{'epoch': N, 'label': 'Fase 2', 'color': 'purple'}]
                per disegnare linee verticali ai cambi di fase (es. Staged Training).
            smoothing_alpha: Float tra 0 e 1. Se > 0, sovrappone una curva EMA smoothed
                alle loss per rendere il trend leggibile. 0 = nessuno smoothing.
            active_loss_keys: Set di chiavi loss che hanno peso > 0. Se fornito,
                le loss non presenti vengono escluse dalla visualizzazione.
        """
        has_lbfgs = adam_epochs is not None and any(e >= adam_epochs for e in self.epochs)
        
        if has_lbfgs:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={'width_ratios': [3, 1]})
            fig.subplots_adjust(wspace=0.3)
        else:
            plt.figure(figsize=(8, 4))
            ax1 = plt.gca()
            ax2 = None

        def plot_on_ax(ax, epoch_range_indices, title_suffix=""):
            # Filtro per skip_epochs
            epoch_range_indices = [i for i in epoch_range_indices if self.epochs[i] >= skip_epochs]
            if not epoch_range_indices: return

            for name, values in self.losses.items():
                if name.startswith('grad_') or name.startswith('weight_'): continue
                
                # Filtra loss con peso 0 (es. data_loss in PurePhys)
                if active_loss_keys is not None and name != 'total_loss':
                    # Mappa il nome della loss alla chiave nel set
                    loss_key_map = {'data_loss': 'data', 'bc_loss': 'bc', 'pde_loss': 'physics'}
                    mapped_key = loss_key_map.get(name, name)
                    if mapped_key not in active_loss_keys:
                        continue
                
                r_epochs = [self.epochs[i] for i in epoch_range_indices]
                r_values = [values[i] if values[i] is not None else np.nan for i in epoch_range_indices]
                
                if all(np.isnan(r_values)): continue

                # Total loss pesata (linea spessa), componenti pure (linee sottili)
                if name == "total_loss":
                    label = f"{name} (weighted)"
                    linewidth = 2.5
                    alpha = 1.0
                else:
                    label = f"{name} (pure)"
                    linewidth = 1.2
                    alpha = 0.8

                line, = ax.plot(r_epochs, r_values, linewidth=linewidth, label=label, alpha=alpha)
                
                # Smoothing EMA overlay
                if smoothing_alpha > 0 and len(r_values) > 10:
                    ema = []
                    current = None
                    for v in r_values:
                        if np.isnan(v):
                            ema.append(np.nan)
                        elif current is None:
                            current = v
                            ema.append(v)
                        else:
                            current = smoothing_alpha * current + (1 - smoothing_alpha) * v
                            ema.append(current)
                    ax.plot(r_epochs, ema, linewidth=linewidth + 0.5, alpha=0.5, 
                            color=line.get_color(), linestyle='--')
            
            # Disegno linee verticali per i cambi di Learning Rate
            if len(self.lr_history) > 0:
                first_lr_vline = True
                for i in range(1, len(epoch_range_indices)):
                    idx_curr = epoch_range_indices[i]
                    idx_prev = epoch_range_indices[i-1]
                    
                    if idx_curr >= len(self.lr_history) or idx_prev >= len(self.lr_history):
                        continue
                        
                    lr_curr = self.lr_history[idx_curr]
                    lr_prev = self.lr_history[idx_prev]
                    
                    # Confronto robusto per cambi di LR (escludendo i None)
                    if lr_curr is not None and lr_prev is not None and not np.isclose(lr_curr, lr_prev, rtol=1e-8, atol=1e-12):
                        label = "LR Change" if first_lr_vline else None
                        # Linea più visibile: nera tratteggiata con alpha maggiore
                        ax.axvline(self.epochs[idx_curr], color="black", linestyle="--", alpha=0.3, linewidth=1, label=label)
                        first_lr_vline = False

            ax.set_title(f'Loss {title_suffix}')
            ax.set_xlabel('Epoch/Iter')
            ax.set_ylabel('Loss')
            ax.set_yscale('log')
            ax.grid(True, which="both", ls="--", alpha=0.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Phase markers (Staged Training)
            if phase_markers:
                for pm in phase_markers:
                    pm_epoch = pm.get('epoch', 0)
                    pm_label = pm.get('label', 'Phase Change')
                    pm_color = pm.get('color', 'purple')
                    # Solo se il marker è nel range visualizzato
                    displayed_epochs = [self.epochs[i] for i in epoch_range_indices]
                    if displayed_epochs and min(displayed_epochs) <= pm_epoch <= max(displayed_epochs):
                        ax.axvline(pm_epoch, color=pm_color, linestyle='-.', linewidth=1.5, alpha=0.7, label=pm_label)

        if has_lbfgs:
            adam_indices = [i for i, e in enumerate(self.epochs) if e < adam_epochs]
            lbfgs_indices = [i for i, e in enumerate(self.epochs) if e >= adam_epochs]
            
            if adam_indices:
                plot_on_ax(ax1, adam_indices, "(Adam Phase)")
                if warmup_epoch != 0 and warmup_epoch >= skip_epochs:
                    ax1.axvline(warmup_epoch, color="r", linestyle="--", label="End Warmup")
                ax1.legend(loc='upper right', frameon=False, fontsize="x-small")

            if lbfgs_indices:
                lbfgs_plot_indices = [adam_indices[-1]] + lbfgs_indices if adam_indices else lbfgs_indices
                plot_on_ax(ax2, lbfgs_plot_indices, "(L-BFGS Refinement)")
                ax2.set_xlabel('Iter')
        else:
            plot_on_ax(ax1, range(len(self.epochs)), f"- {experiment_name}")
            if warmup_epoch != 0 and warmup_epoch >= skip_epochs:
                ax1.axvline(warmup_epoch, color="r", linestyle="--", label="End Warmup")
            ax1.legend(loc='upper right', frameon=False, fontsize="small")

        if save_path:
            dir_name = os.path.dirname(save_path)
            if dir_name: os.makedirs(dir_name, exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
        
        if show_plot: plt.show()
        plt.close()

    def plot_gradients(self, save_path=None, experiment_name="", show_plot=True):
        grad_keys = [k for k in self.losses.keys() if k.startswith('grad_')]
        if not grad_keys: return

        plt.figure(figsize=(8, 4))
        for name in grad_keys:
            values = self.losses[name]
            clean_values = [v if v is not None else np.nan for v in values]
            valid_indices = [i for i, v in enumerate(clean_values) if not np.isnan(v)]
            if valid_indices:
                plt.plot([self.epochs[i] for i in valid_indices], [clean_values[i] for i in valid_indices], label=name, marker='o', markersize=2)
        
        plt.title(f'Gradient Norms - {experiment_name}')
        plt.yscale('log')
        plt.grid(True, ls="--", alpha=0.5)
        plt.legend(loc='upper right', frameon=False)
        if save_path:
            dir_name = os.path.dirname(save_path)
            if dir_name: os.makedirs(dir_name, exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
        if show_plot: plt.show()
        plt.close()

    def plot_weights(self, save_path=None, experiment_name="", show_plot=True):
        weight_keys = [k for k in self.losses.keys() if k.startswith('weight_')]
        if not weight_keys: return

        plt.figure(figsize=(8, 4))
        for name in weight_keys:
            values = self.losses[name]
            clean_values = [v if v is not None else np.nan for v in values]
            valid_indices = [i for i, v in enumerate(clean_values) if not np.isnan(v)]
            if valid_indices:
                plt.plot([self.epochs[i] for i in valid_indices], [clean_values[i] for i in valid_indices], label=name.replace('weight_', 'lambda_'), linewidth=2)
        
        plt.title(f'Evolution of Loss Weights - {experiment_name}')
        plt.yscale('log')
        plt.grid(True, ls="--", alpha=0.5)
        plt.legend(loc='best', frameon=False)
        if save_path:
            dir_name = os.path.dirname(save_path)
            if dir_name: os.makedirs(dir_name, exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
        if show_plot: plt.show()
        plt.close()

def compute_pinn_loss(model, x_data, y_data, x_bc=None, y_bc=None, physics_loss_fn=None, x_physics=None, ic_loss_fn=None, physics_problem=None, lambda_data=1.0, lambda_bc=1.0, lambda_physics=1.0, **kwargs):
    """
    Computes the components of the PINN loss.
    COMPONENTS IN 'loss_dict' ARE PURE RESIDUALS (UNWEIGHTED).
    'total_loss' IS WEIGHTED.
    """
    loss_dict = {}
    total_loss = 0.0
    mse_loss = nn.MSELoss()
    
    if x_data is not None and y_data is not None and x_data.numel() > 0:
        y_pred = model(x_data)
        data_loss = mse_loss(y_pred, y_data)
        loss_dict['data_loss'] = data_loss
        total_loss += lambda_data * data_loss

    if physics_problem is not None and x_bc is not None and y_bc is not None and x_bc.numel() > 0:
        bc_loss_val = physics_problem.boundary_loss(model, x_bc, y_bc)
        loss_dict['bc_loss'] = bc_loss_val
        total_loss += lambda_bc * bc_loss_val
    elif x_bc is not None and y_bc is not None and x_bc.numel() > 0:
        bc_loss_val = mse_loss(model(x_bc), y_bc)
        loss_dict['bc_loss'] = bc_loss_val
        total_loss += lambda_bc * bc_loss_val
    
    if physics_problem is not None and x_physics is not None:
        pde_loss = physics_problem.residual(model, x_physics)
        loss_dict['pde_loss'] = pde_loss
        total_loss += lambda_physics * pde_loss
    elif physics_loss_fn is not None:
        if x_physics is not None:
            if not x_physics.requires_grad: x_physics.requires_grad_(True)
            pde_loss = physics_loss_fn(model, x_physics, **kwargs)
        else:
            pde_loss = physics_loss_fn(model, **kwargs)
        loss_dict['pde_loss'] = pde_loss
        total_loss += lambda_physics * pde_loss
        
    if ic_loss_fn is not None:
        ic_loss = ic_loss_fn(model, **kwargs)
        loss_dict['ic_loss'] = ic_loss
        total_loss += ic_loss
        
    loss_dict['total_loss'] = total_loss
    return total_loss, loss_dict
```

---

### 4. [Viscoelastic_PINN.py](file:///c:/Users/eaglw/Documents/PINN%20tesi/Viscoelastic/src/Viscoelastic_PINN.py) — Integrazione Training

- **Plot multi-campo finale**: dopo il training, genera `PINN_viscoelastic_fields.png` con tutti e 5 i campi
- **Phase markers**: costruisce automaticamente i marker per Warmup e Fase 2 (Tau) nel Staged Training
- **Active loss keys**: calcola automaticamente quali loss hanno peso > 0 e le passa al plotter
- **EMA smoothing**: attivato di default con α=0.95

```diff:Viscoelastic_PINN.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from tqdm import tqdm

# Import function for GIF and loss comparison
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from func.graphic_func import save_gif_PIL, plot2D_comparison, plot2D_final_result
from func.history_tracker import TrainingHistory, compute_pinn_loss

# Configurazione dispositivo e precisione
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Rimosso set_default_dtype globale per non interferire con Adam @ FP32

# ---  DEFINIZIONE DELLA RETE NEURALE E WRAPPER ---
class FCN(nn.Module):
    """Rete Neurale a Connessioni Complete (Fully Connected Network)"""
    def __init__(self, layers, activation_fn=nn.Tanh):
        super().__init__()
        self.activation = activation_fn()
        self.fcs = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.fcs.append(nn.Linear(layers[i], layers[i+1]))
    def forward(self, x):
        for layer in self.fcs[:-1]:   # tutti tranne l'ultimo
            x = self.activation(layer(x))
        return self.fcs[-1](x) 
    def loss_fn(self, pred, target):
        return nn.MSELoss()(pred, target)

def get_activation_name(activation_class):
    return activation_class.__name__

def format_layers_name(layers):
    if len(layers) > 3:
        hidden = layers[1:-1]
        if all(x == hidden[0] for x in hidden):
            return f"{layers[0]}_{hidden[0]}x{len(hidden)}_{layers[-1]}"
    return "_".join(map(str, layers))

class ViscoelasticCombinedModel(nn.Module):
    """
    Wrapper to unify separate networks for Training (e.g. model_psi, model_p, model_tau).
    Mimics a single model with multiple outputs [psi, p, tau_xx, tau_xy, tau_yy].
    """
    def __init__(self, model_psi, model_p, model_tau):
        super().__init__()
        self.model_psi = model_psi
        self.model_p = model_p
        self.model_tau = model_tau
    def forward(self, x):
        psi = self.model_psi(x)
        p = self.model_p(x)
        tau = self.model_tau(x)
        return torch.cat([psi, p, tau], dim=1)

class VelocityInferenceWrapper(nn.Module):
    """
    Wrapper to extract Velocity (u) from a Combined Model or Single Model.
    Used for metrics and validation plots.
    """
    def __init__(self, model, phys_problem):
        super().__init__()
        self.model = model
        self.phys_problem = phys_problem
    def forward(self, x):
        with torch.set_grad_enabled(True):
            if not x.requires_grad: x.requires_grad_(True)
            u, _, _, _ = self.phys_problem.get_velocity(self.model, x)
        return u.detach()
    def eval(self):
        super().eval()
        self.model.eval()
        return self

def set_model_trainable(model_combined, active_components=['psi', 'p', 'tau']):
    """
    Congela o sblocca le sottoreti del modello combinato.
    active_components: lista di stringhe ('psi', 'p', 'tau')
    """
    # Prima congeliamo tutto
    for p in model_combined.parameters():
        p.requires_grad = False
    
    # Sblocchiamo solo i componenti richiesti
    if 'psi' in active_components:
        for p in model_combined.model_psi.parameters(): p.requires_grad = True
    if 'p' in active_components:
        for p in model_combined.model_p.parameters(): p.requires_grad = True
    if 'tau' in active_components:
        for p in model_combined.model_tau.parameters(): p.requires_grad = True
        
    print(f"  [Trainable status] Psi: {'psi' in active_components}, P: {'p' in active_components}, Tau: {'tau' in active_components}")



def train_ViscoelasticPINN(
    model, optimizer, data_internal, data_boundary, validation_grid,
    epochs=20000, physics_problem=None, plots_dir='plots', final_dir='Viscoelastic/Results',
    show_plots_interactively=True, log_gradients_every=0, loss_weights=None,
    warmup_epochs=None, n_collocation=(50, 50), collocation_points=None,
    lr_strategy='fixed', dynamic_weighting=False, update_weights_every=100,
    max_total_lbfgs=100, resample_every=0, resample_fn=None,
    experiment_name="PINN Training", val_label="Value",
    grad_clip_norm=5.0, stress_exact_grids=None,
    staged_training=False, base_lr=1e-3
):
    """
    Esegue il training della PINN viscoelastica.
    
    Args:
        staged_training: (Bool) Se True, il training Adam è diviso in due fasi:
            - Fase 1 (prima metà): allena solo psi+p (cinematica), tau congelato.
            - Fase 2 (seconda metà): allena solo tau (costitutivo), psi+p congelati.
            Infine L-BFGS raffina con tutto sbloccato.
        base_lr: (Float) Learning rate di base per Adam (usato per ricreare
            l'ottimizzatore al cambio di fase nel staged training).
        resample_every: (Int) Se > 0, ricampiona i punti di collocazione ogni N epoche.
        resample_fn: (Callable) Funzione che restituisce nuovi punti di collocazione.
        grad_clip_norm: (Float) Norma massima per il gradient clipping.
        stress_exact_grids: (Dict) Se fornito, contiene le soluzioni analitiche degli stress.
    
    NOTA sulla data_loss:
        La data_loss confronta direttamente l'output raw del modello [psi, p, tau_xx, tau_xy, tau_yy].
        Il fit su psi è più vincolante del fit su u perché fissa la costante di integrazione.
        La boundary_loss invece opera su [u, v, p] derivati dalla stream function.
    """
    # --- SETUP STAGED TRAINING ---
    half_epochs = epochs // 2
    if staged_training:
        print(f"\n  [Staged Training] Fase 1: Cinematica (psi+p) per {half_epochs} epoche")
        set_model_trainable(model, ['psi', 'p'])
        phase_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(phase_params, lr=base_lr)
    else:
        set_model_trainable(model, ['psi', 'p', 'tau'])
    xy_int, T_int = data_internal
    xy_bc, T_bc = data_boundary
    xy_grid, T_exact_grid, X, Y = validation_grid
    # Spostiamo tutto su CPU per il plotting con matplotlib
    X, Y = X.cpu(), Y.cpu()
    T_exact_grid = T_exact_grid.cpu()
    Ny_dom, Nx_dom = X.shape
    Lx, Ly = X.max().item(), Y.max().item()

    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)
    plot_files = []
    
    pbar = tqdm(range(epochs), desc=f"Training PINN (Adam) ({lr_strategy})", mininterval=2.0)
    loss_history = TrainingHistory()
    
    if loss_weights is None: loss_weights = {'data': 1.0, 'bc': 1.0, 'physics': 1.0}
    lambda_data, lambda_bc, target_lambda_physics = loss_weights.get('data', 1.0), loss_weights.get('bc', 1.0), loss_weights.get('physics', 1.0)
    if warmup_epochs is None: warmup_epochs = epochs // 3
    
    scheduler = None
    if lr_strategy == 'step_decay':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(epochs * 0.25), gamma=0.5)
    elif lr_strategy == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=600, min_lr=1e-6, cooldown=3000)

    if collocation_points is not None:
        # Casting esplicito a dtype/device del modello per evitare mismatch
        # se i punti sono stati creati prima di un eventuale switch FP64
        _dtype = next(model.parameters()).dtype
        _device = next(model.parameters()).device
        xy_physics = collocation_points.clone().to(dtype=_dtype, device=_device)
        if not xy_physics.requires_grad: xy_physics.requires_grad_(True)
    else:
        xy_physics = torch.rand((n_collocation[0]*n_collocation[1], 2), device=device)
        xy_physics[:, 0], xy_physics[:, 1] = xy_physics[:, 0] * Lx, xy_physics[:, 1] * Ly
        xy_physics.requires_grad_(True)

    alpha_dynamic = 0.9
    for epoch in pbar:
        # --- STAGED TRAINING: Cambio fase a metà epoche ---
        if staged_training and epoch == half_epochs:
            print(f"\n  [Staged Training] Fase 2: Costitutivo (tau) per {epochs - half_epochs} epoche")
            set_model_trainable(model, ['tau'])
            phase_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(phase_params, lr=base_lr)
            # Ricreiamo lo scheduler per la nuova fase
            scheduler = None
            if lr_strategy == 'step_decay':
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int((epochs - half_epochs) * 0.25), gamma=0.5)
            elif lr_strategy == 'plateau':
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=600, min_lr=1e-6, cooldown=3000)

        # Periodic Resampling
        if resample_every > 0 and resample_fn is not None and epoch > 0 and epoch % resample_every == 0:
            _dtype = next(model.parameters()).dtype
            _device = next(model.parameters()).device
            xy_physics = resample_fn().clone().detach().to(device=_device, dtype=_dtype)
            xy_physics.requires_grad_(True)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        # Gestione Warmup con solo dati
        if epoch < warmup_epochs:
            current_physics_problem, lambda_physics, phase_desc = None, 0.0, "Warmup"
        else:
            current_physics_problem, lambda_physics, phase_desc = physics_problem, target_lambda_physics, "Physics"

        # Calcolo loss
        # Se siamo in warmup, passiamo x_physics=None per saltare il calcolo dei residui PDE
        # ma manteniamo physics_problem per il corretto calcolo delle BC (psi -> u,v)
        loss, loss_dict = compute_pinn_loss(
            model, 
            x_data=xy_int, 
            y_data=T_int,
            x_bc=xy_bc,
            y_bc=T_bc,
            physics_loss_fn=None, 
            physics_problem=physics_problem,
            x_physics=xy_physics if epoch >= warmup_epochs else None,
            lambda_data=lambda_data,
            lambda_bc=lambda_bc,
            lambda_physics=lambda_physics
        )        
        # Dynamic Weighting (Learning Rate Annealing style)
        if dynamic_weighting and epoch >= warmup_epochs and (epoch + 1) % update_weights_every == 0:
            # Calcoliamo i gradienti per le BC (riferimento standard)
            pure_bc = physics_problem.boundary_loss(model, xy_bc, T_bc) if physics_problem else nn.MSELoss()(model(xy_bc), T_bc)
            grads_bc = torch.autograd.grad(pure_bc, model.parameters(), retain_graph=True, allow_unused=True)
            max_norm_bc = max([g.norm(2) for g in grads_bc if g is not None]).item() if any(g is not None for g in grads_bc) else 0.0
            
            # Applichiamo l'aggiornamento solo se il riferimento (BC) è attivo (>0)
            # Se lambda_bc è 0, non possiamo usarlo come ancora per bilanciare gli altri.
            if lambda_bc > 0:
                if lambda_physics > 0:
                    pure_phys = physics_problem.residual(model, xy_physics)
                    grads_ph = torch.autograd.grad(pure_phys, model.parameters(), retain_graph=True, allow_unused=True)
                    m_n_ph = max([g.norm(2) for g in grads_ph if g is not None]).item() if any(g is not None for g in grads_ph) else 0.0
                    if m_n_ph > 1e-12: 
                        ratio = min(max_norm_bc / m_n_ph, 100.0)
                        target_lambda_physics = alpha_dynamic * target_lambda_physics + (1-alpha_dynamic) * ratio * lambda_bc

                if lambda_data > 0:
                    pure_data = nn.MSELoss()(model(xy_int), T_int)
                    grads_dt = torch.autograd.grad(pure_data, model.parameters(), retain_graph=True, allow_unused=True)
                    m_n_dt = max([g.norm(2) for g in grads_dt if g is not None]).item() if any(g is not None for g in grads_dt) else 0.0
                    if m_n_dt > 1e-12: 
                        ratio_d = min(max_norm_bc / m_n_dt, 100.0)
                        lambda_data = alpha_dynamic * lambda_data + (1-alpha_dynamic) * ratio_d * lambda_bc
        
        # Logging context
        current_lr = optimizer.param_groups[0]['lr']
        history_entry = loss_dict.copy()
        history_entry.update({'weight_data': lambda_data, 'weight_bc': lambda_bc, 'weight_phys': lambda_physics})

        if log_gradients_every > 0 and (epoch + 1) % log_gradients_every == 0:
            grad_norms = {}
            for name, l_val in loss_dict.items():
                if name == 'total_loss': continue
                # Per i gradienti usiamo la componente pesata effettiva nel calcolo della loss totale
                w = lambda_data if name == 'data_loss' else (lambda_bc if name == 'bc_loss' else (lambda_physics if name == 'pde_loss' else 1.0))
                grads = torch.autograd.grad(l_val * w, model.parameters(), retain_graph=True, allow_unused=True)
                grad_norms[f'grad_{name}'] = sum(g.data.norm(2).item()**2 for g in grads if g is not None)**0.5
            history_entry.update(grad_norms)

        loss_history.update(epoch, history_entry, lr=current_lr)
        loss.backward()
        
        # Gradient Clipping — con 3 reti e 5 equazioni la norma è strutturalmente alta
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        
        optimizer.step()
        
        if lr_strategy == 'step_decay': scheduler.step()
        elif lr_strategy == 'plateau':
            # Use unweighted loss for stable scheduling, monitoring only active components (weight > 0)
            active_losses = [
                loss_dict[k] for k in ['data_loss', 'bc_loss', 'pde_loss']
                if loss_dict.get(k) is not None and isinstance(loss_dict[k], torch.Tensor)
            ]
            monitored_loss = sum(active_losses) if active_losses else torch.tensor(0.0)
            scheduler.step(monitored_loss.item())
            # Monitoraggio e Plotting periodico
        if (epoch + 1) % 500 == 0:
            pbar.set_postfix({
                'Phase': phase_desc,
                'Loss': f"{loss.item():.2e}", 
                'BC_L': f"{loss_dict.get('bc_loss', 0):.2e}",
                'LR': f"{current_lr:.1e}"
            })            
            model.eval()
            # Per calcolare u = psi_y serve attivare i gradienti, usiamo set_grad_enabled(True) 
            with torch.set_grad_enabled(True): 
                xy_grid_val = xy_grid.clone().detach().requires_grad_(True)
                # Ricaviamo u dal problema fisico (Stream Function)
                if hasattr(physics_problem, 'get_velocity'):
                    u_pred, _, _, _ = physics_problem.get_velocity(model, xy_grid_val)
                    T_pred_grid = u_pred.detach().cpu().reshape(Ny_dom, Nx_dom)
                    
                    # Estraggo anche gli stress
                    out = model(xy_grid_val)
                    tau_xx_pred = out[:, 2].detach().cpu().reshape(Ny_dom, Nx_dom)
                    tau_xy_pred = out[:, 3].detach().cpu().reshape(Ny_dom, Nx_dom)
                    tau_yy_pred = out[:, 4].detach().cpu().reshape(Ny_dom, Nx_dom)
                else:
                    # Fallback per 3-output o altri casi
                    T_pred_grid = model(xy_grid_val)[:, 0].detach().cpu().reshape(Ny_dom, Nx_dom)
                del xy_grid_val
                
            plot_path = os.path.join(plots_dir, f'epoch_{epoch+1}.png')
            plot2D_comparison(X, Y, T_exact_grid, T_pred_grid, epoch+1, plot_path, physics_points=xy_physics, val_label=val_label)
            plot_files.append(plot_path)
            
            if hasattr(physics_problem, 'get_velocity'):
                # Usa ground truth se disponibile, altrimenti mostra solo la predizione
                if stress_exact_grids is not None:
                    tau_xx_exact_g = stress_exact_grids.get('tau_xx', torch.zeros_like(T_exact_grid))
                    tau_xy_exact_g = stress_exact_grids.get('tau_xy', torch.zeros_like(T_exact_grid))
                    tau_yy_exact_g = stress_exact_grids.get('tau_yy', torch.zeros_like(T_exact_grid))
                else:
                    tau_xx_exact_g = torch.zeros_like(T_exact_grid)
                    tau_xy_exact_g = torch.zeros_like(T_exact_grid)
                    tau_yy_exact_g = torch.zeros_like(T_exact_grid)
                plot2D_comparison(X, Y, tau_xx_exact_g, tau_xx_pred, epoch+1, os.path.join(plots_dir, f'tau_xx_{epoch+1}.png'), physics_points=xy_physics, val_label='tau_xx')
                plot2D_comparison(X, Y, tau_xy_exact_g, tau_xy_pred, epoch+1, os.path.join(plots_dir, f'tau_xy_{epoch+1}.png'), physics_points=xy_physics, val_label='tau_xy')
                plot2D_comparison(X, Y, tau_yy_exact_g, tau_yy_pred, epoch+1, os.path.join(plots_dir, f'tau_yy_{epoch+1}.png'), physics_points=xy_physics, val_label='tau_yy')

    # --- SBLOCCO TOTALE + PRECISION SWITCH PER L-BFGS ---
    if staged_training:
        print(f"\n  [Staged Training] Fase 3: Raffinamento L-BFGS (tutto sbloccato)")
        set_model_trainable(model, ['psi', 'p', 'tau'])
    
    # Prima di iniziare L-BFGS, passiamo a FP64 (Float64) per la massima precisione scientifica
    print("\n--- Switching to FP64 for L-BFGS Refinement ---")
    torch.set_default_dtype(torch.float64)
    torch.backends.cuda.matmul.allow_tf32 = False # Disabilitato per FP64
    model.double()
    xy_int      = xy_int.double()
    T_int       = T_int.double()
    xy_bc       = xy_bc.double()
    T_bc        = T_bc.double()
    xy_physics  = xy_physics.detach().double().requires_grad_(True)
    xy_grid     = xy_grid.double()
    T_exact_grid = T_exact_grid.double()
    X, Y = X.double(), Y.double()
    
    # Verifica
    assert all(p.dtype == torch.float64 for p in model.parameters()), \
        "Errore: parametri del modello non tutti in float64 dopo .double()"

    lbfgs_iter = [0]
    pbar_lbfgs = tqdm(total=max_total_lbfgs, desc="Training PINN (L-BFGS)", mininterval=2.0)
    
    for current_lr in [1.0, 0.5]:
        start_iter_call = lbfgs_iter[0]
        remaining_evals = max_total_lbfgs - start_iter_call
        if remaining_evals <= 0:
            break
            
        optimizer_lbfgs = torch.optim.LBFGS(
            model.parameters(), 
            lr=current_lr, 
            max_iter=remaining_evals, 
            # max_eval deve essere maggiore di max_iter perché la strong Wolfe line search
            # richiede 2-5 valutazioni per iterazione. Senza margine, L-BFGS si ferma
            # molto prima del budget previsto.
            max_eval=remaining_evals * 5, 
            tolerance_grad=1e-7, 
            tolerance_change=1e-9,
            history_size=300,
            line_search_fn="strong_wolfe"
        )
        
        # Closure factory per evitare late binding dell'ottimizzatore nel loop
        def make_closure(opt_ref):
            def closure():
                opt_ref.zero_grad()
                loss, loss_dict = compute_pinn_loss(
                    model, 
                    x_data=xy_int, 
                    y_data=T_int,
                    x_bc=xy_bc,
                    y_bc=T_bc,
                    physics_loss_fn=None, 
                    physics_problem=physics_problem,
                    x_physics=xy_physics,
                    lambda_data=lambda_data,
                    lambda_bc=lambda_bc,
                    lambda_physics=target_lambda_physics
                )
                loss.backward()
                if lbfgs_iter[0] % 10 == 0: 
                    history_entry = loss_dict.copy()
                    history_entry.update({'weight_data': lambda_data, 'weight_bc': lambda_bc, 'weight_phys': target_lambda_physics})
                    loss_history.update(epochs + lbfgs_iter[0], history_entry, lr=current_lr)
                
                lbfgs_iter[0] += 1
                pbar_lbfgs.update(1)
                pbar_lbfgs.set_postfix({'Loss': f"{loss.item():.2e}"})
                return loss
            return closure
            
        optimizer_lbfgs.step(make_closure(optimizer_lbfgs))
        
        # Se abbiamo raggiunto il limite massimo, usciamo
        if lbfgs_iter[0] >= max_total_lbfgs:
            break
        
        if current_lr == 1.0:
            print(f"\nL-BFGS interrotto a {lbfgs_iter[0]} chiamate (LR=1.0). Riprovo con LR=0.5 per le restanti {max_total_lbfgs - lbfgs_iter[0]}...")
    
    pbar_lbfgs.close()
    
    # Final loss check after L-BFGS
    final_loss, final_loss_dict = compute_pinn_loss(
            model, 
            x_data=xy_int, 
            y_data=T_int,
            x_bc=xy_bc,
            y_bc=T_bc,
            physics_loss_fn=None, 
            physics_problem=physics_problem,
            x_physics=xy_physics,
            lambda_data=lambda_data,
            lambda_bc=lambda_bc,
            lambda_physics=target_lambda_physics
    )
    final_entry = final_loss_dict.copy()
    final_entry.update({'weight_data': lambda_data, 'weight_bc': lambda_bc, 'weight_phys': target_lambda_physics})
    loss_history.update(epochs + lbfgs_iter[0], final_entry, lr=current_lr)
    print(f"Loss finale dopo L-BFGS (iter {lbfgs_iter[0]}): {final_loss.item():.2e}")

    # Plot Finale Interattivo
    print("Training completato. Generazione plot finale...")
    model.eval()
    with torch.set_grad_enabled(True): 
        xy_grid_val = xy_grid.clone().detach().requires_grad_(True)
        if hasattr(physics_problem, 'get_velocity'):
            u_p, _, _, _ = physics_problem.get_velocity(model, xy_grid_val)
            T_final = u_p.detach().cpu().reshape(Ny_dom, Nx_dom)
        else:
            T_final = model(xy_grid_val)[:, 0].detach().cpu().reshape(Ny_dom, Nx_dom)
        del xy_grid_val
    lambda_data_viz, lambda_bc_viz = loss_weights.get('data', 1.0), loss_weights.get('bc', 1.0)
    internal_pts = xy_int if lambda_data_viz > 0 else None
    boundary_pts = xy_bc if lambda_bc_viz > 0 else None

    final_path = os.path.join(final_dir, 'PINNfinal_result.png')
    plot2D_final_result(X, Y, T_exact_grid, T_final, epochs, save_path=final_path, internal_points=internal_pts, boundary_points=boundary_pts, physics_points=xy_physics, val_label=val_label)
    
    # Generazione GIF
    print(f"Creazione GIF con {len(plot_files)} frames...")
    if plot_files:
        gif_path = os.path.join(final_dir, 'PINNtraining_evolution.gif')
        save_gif_PIL(gif_path, plot_files, fps=3, loop=1, delete_files=True)
    
    # Plot Loss History con split tra Adam e L-BFGS
    loss_history.plot_losses(
        warmup_epoch=warmup_epochs, 
        adam_epochs=epochs,
        save_path=os.path.join(final_dir, 'PINNloss_history.png'), 
        experiment_name=experiment_name, 
        show_plot=show_plots_interactively,
        skip_epochs=50
    )
    
    # Plot Gradient History if available
    loss_history.plot_gradients(save_path=os.path.join(final_dir, 'PINN_gradients.png'), experiment_name=f"{experiment_name} Gradients", show_plot=show_plots_interactively)
    
    # Plot Weight History if available
    loss_history.plot_weights(save_path=os.path.join(final_dir, 'PINN_weights.png'), experiment_name=f"{experiment_name} Weights", show_plot=show_plots_interactively)

    if show_plots_interactively:
        plt.show()
    else:
        plt.close("all")

    # RIPRISTINO PRECISIONE PER EVENTUALI CHIAMATE SUCCESSIVE
    torch.set_default_dtype(torch.float32)
    return loss_history
===
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from tqdm import tqdm

# Import function for GIF and loss comparison
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from func.graphic_func import save_gif_PIL, plot2D_comparison, plot2D_final_result, plot2D_viscoelastic_final
from func.history_tracker import TrainingHistory, compute_pinn_loss

# Configurazione dispositivo e precisione
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Rimosso set_default_dtype globale per non interferire con Adam @ FP32

# ---  DEFINIZIONE DELLA RETE NEURALE E WRAPPER ---
class FCN(nn.Module):
    """Rete Neurale a Connessioni Complete (Fully Connected Network)"""
    def __init__(self, layers, activation_fn=nn.Tanh):
        super().__init__()
        self.activation = activation_fn()
        self.fcs = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.fcs.append(nn.Linear(layers[i], layers[i+1]))
    def forward(self, x):
        for layer in self.fcs[:-1]:   # tutti tranne l'ultimo
            x = self.activation(layer(x))
        return self.fcs[-1](x) 
    def loss_fn(self, pred, target):
        return nn.MSELoss()(pred, target)

def get_activation_name(activation_class):
    return activation_class.__name__

def format_layers_name(layers):
    if len(layers) > 3:
        hidden = layers[1:-1]
        if all(x == hidden[0] for x in hidden):
            return f"{layers[0]}_{hidden[0]}x{len(hidden)}_{layers[-1]}"
    return "_".join(map(str, layers))

class ViscoelasticCombinedModel(nn.Module):
    """
    Wrapper to unify separate networks for Training (e.g. model_psi, model_p, model_tau).
    Mimics a single model with multiple outputs [psi, p, tau_xx, tau_xy, tau_yy].
    """
    def __init__(self, model_psi, model_p, model_tau):
        super().__init__()
        self.model_psi = model_psi
        self.model_p = model_p
        self.model_tau = model_tau
    def forward(self, x):
        psi = self.model_psi(x)
        p = self.model_p(x)
        tau = self.model_tau(x)
        return torch.cat([psi, p, tau], dim=1)

class VelocityInferenceWrapper(nn.Module):
    """
    Wrapper to extract Velocity (u) from a Combined Model or Single Model.
    Used for metrics and validation plots.
    """
    def __init__(self, model, phys_problem):
        super().__init__()
        self.model = model
        self.phys_problem = phys_problem
    def forward(self, x):
        with torch.set_grad_enabled(True):
            if not x.requires_grad: x.requires_grad_(True)
            u, _, _, _ = self.phys_problem.get_velocity(self.model, x)
        return u.detach()
    def eval(self):
        super().eval()
        self.model.eval()
        return self

def set_model_trainable(model_combined, active_components=['psi', 'p', 'tau']):
    """
    Congela o sblocca le sottoreti del modello combinato.
    active_components: lista di stringhe ('psi', 'p', 'tau')
    """
    # Prima congeliamo tutto
    for p in model_combined.parameters():
        p.requires_grad = False
    
    # Sblocchiamo solo i componenti richiesti
    if 'psi' in active_components:
        for p in model_combined.model_psi.parameters(): p.requires_grad = True
    if 'p' in active_components:
        for p in model_combined.model_p.parameters(): p.requires_grad = True
    if 'tau' in active_components:
        for p in model_combined.model_tau.parameters(): p.requires_grad = True
        
    print(f"  [Trainable status] Psi: {'psi' in active_components}, P: {'p' in active_components}, Tau: {'tau' in active_components}")



def train_ViscoelasticPINN(
    model, optimizer, data_internal, data_boundary, validation_grid,
    epochs=20000, physics_problem=None, plots_dir='plots', final_dir='Viscoelastic/Results',
    show_plots_interactively=True, log_gradients_every=0, loss_weights=None,
    warmup_epochs=None, n_collocation=(50, 50), collocation_points=None,
    lr_strategy='fixed', dynamic_weighting=False, update_weights_every=100,
    max_total_lbfgs=100, resample_every=0, resample_fn=None,
    experiment_name="PINN Training", val_label="Value",
    grad_clip_norm=5.0, stress_exact_grids=None,
    staged_training=False, base_lr=1e-3
):
    """
    Esegue il training della PINN viscoelastica.
    
    Args:
        staged_training: (Bool) Se True, il training Adam è diviso in due fasi:
            - Fase 1 (prima metà): allena solo psi+p (cinematica), tau congelato.
            - Fase 2 (seconda metà): allena solo tau (costitutivo), psi+p congelati.
            Infine L-BFGS raffina con tutto sbloccato.
        base_lr: (Float) Learning rate di base per Adam (usato per ricreare
            l'ottimizzatore al cambio di fase nel staged training).
        resample_every: (Int) Se > 0, ricampiona i punti di collocazione ogni N epoche.
        resample_fn: (Callable) Funzione che restituisce nuovi punti di collocazione.
        grad_clip_norm: (Float) Norma massima per il gradient clipping.
        stress_exact_grids: (Dict) Se fornito, contiene le soluzioni analitiche degli stress.
    
    NOTA sulla data_loss:
        La data_loss confronta direttamente l'output raw del modello [psi, p, tau_xx, tau_xy, tau_yy].
        Il fit su psi è più vincolante del fit su u perché fissa la costante di integrazione.
        La boundary_loss invece opera su [u, v, p] derivati dalla stream function.
    """
    # --- SETUP STAGED TRAINING ---
    half_epochs = epochs // 2
    if staged_training:
        print(f"\n  [Staged Training] Fase 1: Cinematica (psi+p) per {half_epochs} epoche")
        set_model_trainable(model, ['psi', 'p'])
        phase_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(phase_params, lr=base_lr)
    else:
        set_model_trainable(model, ['psi', 'p', 'tau'])
    xy_int, T_int = data_internal
    xy_bc, T_bc = data_boundary
    xy_grid, T_exact_grid, X, Y = validation_grid
    # Spostiamo tutto su CPU per il plotting con matplotlib
    X, Y = X.cpu(), Y.cpu()
    T_exact_grid = T_exact_grid.cpu()
    Ny_dom, Nx_dom = X.shape
    Lx, Ly = X.max().item(), Y.max().item()

    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)
    plot_files = []
    
    pbar = tqdm(range(epochs), desc=f"Training PINN (Adam) ({lr_strategy})", mininterval=2.0)
    loss_history = TrainingHistory()
    
    if loss_weights is None: loss_weights = {'data': 1.0, 'bc': 1.0, 'physics': 1.0}
    lambda_data, lambda_bc, target_lambda_physics = loss_weights.get('data', 1.0), loss_weights.get('bc', 1.0), loss_weights.get('physics', 1.0)
    if warmup_epochs is None: warmup_epochs = epochs // 3
    
    scheduler = None
    if lr_strategy == 'step_decay':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(epochs * 0.25), gamma=0.5)
    elif lr_strategy == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=600, min_lr=1e-6, cooldown=3000)

    if collocation_points is not None:
        # Casting esplicito a dtype/device del modello per evitare mismatch
        # se i punti sono stati creati prima di un eventuale switch FP64
        _dtype = next(model.parameters()).dtype
        _device = next(model.parameters()).device
        xy_physics = collocation_points.clone().to(dtype=_dtype, device=_device)
        if not xy_physics.requires_grad: xy_physics.requires_grad_(True)
    else:
        xy_physics = torch.rand((n_collocation[0]*n_collocation[1], 2), device=device)
        xy_physics[:, 0], xy_physics[:, 1] = xy_physics[:, 0] * Lx, xy_physics[:, 1] * Ly
        xy_physics.requires_grad_(True)

    alpha_dynamic = 0.9
    for epoch in pbar:
        # --- STAGED TRAINING: Cambio fase a metà epoche ---
        if staged_training and epoch == half_epochs:
            print(f"\n  [Staged Training] Fase 2: Costitutivo (tau) per {epochs - half_epochs} epoche")
            set_model_trainable(model, ['tau'])
            phase_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(phase_params, lr=base_lr)
            # Ricreiamo lo scheduler per la nuova fase
            scheduler = None
            if lr_strategy == 'step_decay':
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int((epochs - half_epochs) * 0.25), gamma=0.5)
            elif lr_strategy == 'plateau':
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=600, min_lr=1e-6, cooldown=3000)

        # Periodic Resampling
        if resample_every > 0 and resample_fn is not None and epoch > 0 and epoch % resample_every == 0:
            _dtype = next(model.parameters()).dtype
            _device = next(model.parameters()).device
            xy_physics = resample_fn().clone().detach().to(device=_device, dtype=_dtype)
            xy_physics.requires_grad_(True)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        # Gestione Warmup con solo dati
        if epoch < warmup_epochs:
            current_physics_problem, lambda_physics, phase_desc = None, 0.0, "Warmup"
        else:
            current_physics_problem, lambda_physics, phase_desc = physics_problem, target_lambda_physics, "Physics"

        # Calcolo loss
        # Se siamo in warmup, passiamo x_physics=None per saltare il calcolo dei residui PDE
        # ma manteniamo physics_problem per il corretto calcolo delle BC (psi -> u,v)
        loss, loss_dict = compute_pinn_loss(
            model, 
            x_data=xy_int, 
            y_data=T_int,
            x_bc=xy_bc,
            y_bc=T_bc,
            physics_loss_fn=None, 
            physics_problem=physics_problem,
            x_physics=xy_physics if epoch >= warmup_epochs else None,
            lambda_data=lambda_data,
            lambda_bc=lambda_bc,
            lambda_physics=lambda_physics
        )        
        # Dynamic Weighting (Learning Rate Annealing style)
        if dynamic_weighting and epoch >= warmup_epochs and (epoch + 1) % update_weights_every == 0:
            # Calcoliamo i gradienti per le BC (riferimento standard)
            pure_bc = physics_problem.boundary_loss(model, xy_bc, T_bc) if physics_problem else nn.MSELoss()(model(xy_bc), T_bc)
            grads_bc = torch.autograd.grad(pure_bc, model.parameters(), retain_graph=True, allow_unused=True)
            max_norm_bc = max([g.norm(2) for g in grads_bc if g is not None]).item() if any(g is not None for g in grads_bc) else 0.0
            
            # Applichiamo l'aggiornamento solo se il riferimento (BC) è attivo (>0)
            # Se lambda_bc è 0, non possiamo usarlo come ancora per bilanciare gli altri.
            if lambda_bc > 0:
                if lambda_physics > 0:
                    pure_phys = physics_problem.residual(model, xy_physics)
                    grads_ph = torch.autograd.grad(pure_phys, model.parameters(), retain_graph=True, allow_unused=True)
                    m_n_ph = max([g.norm(2) for g in grads_ph if g is not None]).item() if any(g is not None for g in grads_ph) else 0.0
                    if m_n_ph > 1e-12: 
                        ratio = min(max_norm_bc / m_n_ph, 100.0)
                        target_lambda_physics = alpha_dynamic * target_lambda_physics + (1-alpha_dynamic) * ratio * lambda_bc

                if lambda_data > 0:
                    pure_data = nn.MSELoss()(model(xy_int), T_int)
                    grads_dt = torch.autograd.grad(pure_data, model.parameters(), retain_graph=True, allow_unused=True)
                    m_n_dt = max([g.norm(2) for g in grads_dt if g is not None]).item() if any(g is not None for g in grads_dt) else 0.0
                    if m_n_dt > 1e-12: 
                        ratio_d = min(max_norm_bc / m_n_dt, 100.0)
                        lambda_data = alpha_dynamic * lambda_data + (1-alpha_dynamic) * ratio_d * lambda_bc
        
        # Logging context
        current_lr = optimizer.param_groups[0]['lr']
        history_entry = loss_dict.copy()
        history_entry.update({'weight_data': lambda_data, 'weight_bc': lambda_bc, 'weight_phys': lambda_physics})

        if log_gradients_every > 0 and (epoch + 1) % log_gradients_every == 0:
            grad_norms = {}
            for name, l_val in loss_dict.items():
                if name == 'total_loss': continue
                # Per i gradienti usiamo la componente pesata effettiva nel calcolo della loss totale
                w = lambda_data if name == 'data_loss' else (lambda_bc if name == 'bc_loss' else (lambda_physics if name == 'pde_loss' else 1.0))
                grads = torch.autograd.grad(l_val * w, model.parameters(), retain_graph=True, allow_unused=True)
                grad_norms[f'grad_{name}'] = sum(g.data.norm(2).item()**2 for g in grads if g is not None)**0.5
            history_entry.update(grad_norms)

        loss_history.update(epoch, history_entry, lr=current_lr)
        loss.backward()
        
        # Gradient Clipping — con 3 reti e 5 equazioni la norma è strutturalmente alta
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        
        optimizer.step()
        
        if lr_strategy == 'step_decay': scheduler.step()
        elif lr_strategy == 'plateau':
            # Use unweighted loss for stable scheduling, monitoring only active components (weight > 0)
            active_losses = [
                loss_dict[k] for k in ['data_loss', 'bc_loss', 'pde_loss']
                if loss_dict.get(k) is not None and isinstance(loss_dict[k], torch.Tensor)
            ]
            monitored_loss = sum(active_losses) if active_losses else torch.tensor(0.0)
            scheduler.step(monitored_loss.item())
            # Monitoraggio e Plotting periodico
        if (epoch + 1) % 500 == 0:
            pbar.set_postfix({
                'Phase': phase_desc,
                'Loss': f"{loss.item():.2e}", 
                'BC_L': f"{loss_dict.get('bc_loss', 0):.2e}",
                'LR': f"{current_lr:.1e}"
            })            
            model.eval()
            # Per calcolare u = psi_y serve attivare i gradienti, usiamo set_grad_enabled(True) 
            with torch.set_grad_enabled(True): 
                xy_grid_val = xy_grid.clone().detach().requires_grad_(True)
                # Ricaviamo u dal problema fisico (Stream Function)
                if hasattr(physics_problem, 'get_velocity'):
                    u_pred, _, _, _ = physics_problem.get_velocity(model, xy_grid_val)
                    T_pred_grid = u_pred.detach().cpu().reshape(Ny_dom, Nx_dom)
                    
                    # Estraggo anche gli stress
                    out = model(xy_grid_val)
                    tau_xx_pred = out[:, 2].detach().cpu().reshape(Ny_dom, Nx_dom)
                    tau_xy_pred = out[:, 3].detach().cpu().reshape(Ny_dom, Nx_dom)
                    tau_yy_pred = out[:, 4].detach().cpu().reshape(Ny_dom, Nx_dom)
                else:
                    # Fallback per 3-output o altri casi
                    T_pred_grid = model(xy_grid_val)[:, 0].detach().cpu().reshape(Ny_dom, Nx_dom)
                del xy_grid_val
                
            plot_path = os.path.join(plots_dir, f'epoch_{epoch+1}.png')
            plot2D_comparison(X, Y, T_exact_grid, T_pred_grid, epoch+1, plot_path, physics_points=xy_physics, val_label=val_label)
            plot_files.append(plot_path)
            
            if hasattr(physics_problem, 'get_velocity'):
                # Usa ground truth se disponibile, altrimenti mostra solo la predizione
                if stress_exact_grids is not None:
                    tau_xx_exact_g = stress_exact_grids.get('tau_xx', torch.zeros_like(T_exact_grid))
                    tau_xy_exact_g = stress_exact_grids.get('tau_xy', torch.zeros_like(T_exact_grid))
                    tau_yy_exact_g = stress_exact_grids.get('tau_yy', torch.zeros_like(T_exact_grid))
                else:
                    tau_xx_exact_g = torch.zeros_like(T_exact_grid)
                    tau_xy_exact_g = torch.zeros_like(T_exact_grid)
                    tau_yy_exact_g = torch.zeros_like(T_exact_grid)
                plot2D_comparison(X, Y, tau_xx_exact_g, tau_xx_pred, epoch+1, os.path.join(plots_dir, f'tau_xx_{epoch+1}.png'), physics_points=xy_physics, val_label='tau_xx')
                plot2D_comparison(X, Y, tau_xy_exact_g, tau_xy_pred, epoch+1, os.path.join(plots_dir, f'tau_xy_{epoch+1}.png'), physics_points=xy_physics, val_label='tau_xy')
                plot2D_comparison(X, Y, tau_yy_exact_g, tau_yy_pred, epoch+1, os.path.join(plots_dir, f'tau_yy_{epoch+1}.png'), physics_points=xy_physics, val_label='tau_yy')

    # --- SBLOCCO TOTALE + PRECISION SWITCH PER L-BFGS ---
    if staged_training:
        print(f"\n  [Staged Training] Fase 3: Raffinamento L-BFGS (tutto sbloccato)")
        set_model_trainable(model, ['psi', 'p', 'tau'])
    
    # Prima di iniziare L-BFGS, passiamo a FP64 (Float64) per la massima precisione scientifica
    print("\n--- Switching to FP64 for L-BFGS Refinement ---")
    torch.set_default_dtype(torch.float64)
    torch.backends.cuda.matmul.allow_tf32 = False # Disabilitato per FP64
    model.double()
    xy_int      = xy_int.double()
    T_int       = T_int.double()
    xy_bc       = xy_bc.double()
    T_bc        = T_bc.double()
    xy_physics  = xy_physics.detach().double().requires_grad_(True)
    xy_grid     = xy_grid.double()
    T_exact_grid = T_exact_grid.double()
    X, Y = X.double(), Y.double()
    
    # Verifica
    assert all(p.dtype == torch.float64 for p in model.parameters()), \
        "Errore: parametri del modello non tutti in float64 dopo .double()"

    lbfgs_iter = [0]
    pbar_lbfgs = tqdm(total=max_total_lbfgs, desc="Training PINN (L-BFGS)", mininterval=2.0)
    
    for current_lr in [1.0, 0.5]:
        start_iter_call = lbfgs_iter[0]
        remaining_evals = max_total_lbfgs - start_iter_call
        if remaining_evals <= 0:
            break
            
        optimizer_lbfgs = torch.optim.LBFGS(
            model.parameters(), 
            lr=current_lr, 
            max_iter=remaining_evals, 
            # max_eval deve essere maggiore di max_iter perché la strong Wolfe line search
            # richiede 2-5 valutazioni per iterazione. Senza margine, L-BFGS si ferma
            # molto prima del budget previsto.
            max_eval=remaining_evals * 5, 
            tolerance_grad=1e-7, 
            tolerance_change=1e-9,
            history_size=300,
            line_search_fn="strong_wolfe"
        )
        
        # Closure factory per evitare late binding dell'ottimizzatore nel loop
        def make_closure(opt_ref):
            def closure():
                opt_ref.zero_grad()
                loss, loss_dict = compute_pinn_loss(
                    model, 
                    x_data=xy_int, 
                    y_data=T_int,
                    x_bc=xy_bc,
                    y_bc=T_bc,
                    physics_loss_fn=None, 
                    physics_problem=physics_problem,
                    x_physics=xy_physics,
                    lambda_data=lambda_data,
                    lambda_bc=lambda_bc,
                    lambda_physics=target_lambda_physics
                )
                loss.backward()
                if lbfgs_iter[0] % 10 == 0: 
                    history_entry = loss_dict.copy()
                    history_entry.update({'weight_data': lambda_data, 'weight_bc': lambda_bc, 'weight_phys': target_lambda_physics})
                    loss_history.update(epochs + lbfgs_iter[0], history_entry, lr=current_lr)
                
                lbfgs_iter[0] += 1
                pbar_lbfgs.update(1)
                pbar_lbfgs.set_postfix({'Loss': f"{loss.item():.2e}"})
                return loss
            return closure
            
        optimizer_lbfgs.step(make_closure(optimizer_lbfgs))
        
        # Se abbiamo raggiunto il limite massimo, usciamo
        if lbfgs_iter[0] >= max_total_lbfgs:
            break
        
        if current_lr == 1.0:
            print(f"\nL-BFGS interrotto a {lbfgs_iter[0]} chiamate (LR=1.0). Riprovo con LR=0.5 per le restanti {max_total_lbfgs - lbfgs_iter[0]}...")
    
    pbar_lbfgs.close()
    
    # Final loss check after L-BFGS
    final_loss, final_loss_dict = compute_pinn_loss(
            model, 
            x_data=xy_int, 
            y_data=T_int,
            x_bc=xy_bc,
            y_bc=T_bc,
            physics_loss_fn=None, 
            physics_problem=physics_problem,
            x_physics=xy_physics,
            lambda_data=lambda_data,
            lambda_bc=lambda_bc,
            lambda_physics=target_lambda_physics
    )
    final_entry = final_loss_dict.copy()
    final_entry.update({'weight_data': lambda_data, 'weight_bc': lambda_bc, 'weight_phys': target_lambda_physics})
    loss_history.update(epochs + lbfgs_iter[0], final_entry, lr=current_lr)
    print(f"Loss finale dopo L-BFGS (iter {lbfgs_iter[0]}): {final_loss.item():.2e}")

    # Plot Finale Interattivo
    print("Training completato. Generazione plot finale...")
    model.eval()
    with torch.set_grad_enabled(True): 
        xy_grid_val = xy_grid.clone().detach().requires_grad_(True)
        if hasattr(physics_problem, 'get_velocity'):
            u_p, _, _, _ = physics_problem.get_velocity(model, xy_grid_val)
            T_final = u_p.detach().cpu().reshape(Ny_dom, Nx_dom)
        else:
            T_final = model(xy_grid_val)[:, 0].detach().cpu().reshape(Ny_dom, Nx_dom)
        del xy_grid_val
    lambda_data_viz, lambda_bc_viz = loss_weights.get('data', 1.0), loss_weights.get('bc', 1.0)
    internal_pts = xy_int if lambda_data_viz > 0 else None
    boundary_pts = xy_bc if lambda_bc_viz > 0 else None

    final_path = os.path.join(final_dir, 'PINNfinal_result.png')
    plot2D_final_result(X, Y, T_exact_grid, T_final, epochs, save_path=final_path, internal_points=internal_pts, boundary_points=boundary_pts, physics_points=xy_physics, val_label=val_label)
    
    # Plot Finale Multi-Campo Viscoelastic (u, p, tau_xx, tau_xy, tau_yy)
    if hasattr(physics_problem, 'get_velocity') and stress_exact_grids is not None:
        print("Generazione plot multi-campo viscoelastico...")
        with torch.set_grad_enabled(True):
            xy_grid_val = xy_grid.clone().detach().requires_grad_(True)
            u_final, v_final, p_final, _ = physics_problem.get_velocity(model, xy_grid_val)
            out_final = model(xy_grid_val)
            
            fields_pred = {
                'u': u_final.detach().cpu().reshape(Ny_dom, Nx_dom),
                'p': p_final.detach().cpu().reshape(Ny_dom, Nx_dom),
                'tau_xx': out_final[:, 2].detach().cpu().reshape(Ny_dom, Nx_dom),
                'tau_xy': out_final[:, 3].detach().cpu().reshape(Ny_dom, Nx_dom),
                'tau_yy': out_final[:, 4].detach().cpu().reshape(Ny_dom, Nx_dom),
            }
            del xy_grid_val
        
        # Prepara exact grids (possono essere su CUDA)
        p_exact_grid = T_exact_grid  # Per ora, ma potremo passare separatamente
        fields_exact = {
            'u': T_exact_grid.cpu(),
            'p': stress_exact_grids.get('p', torch.zeros_like(T_exact_grid)).cpu(),
            'tau_xx': stress_exact_grids.get('tau_xx', torch.zeros_like(T_exact_grid)).cpu(),
            'tau_xy': stress_exact_grids.get('tau_xy', torch.zeros_like(T_exact_grid)).cpu(),
            'tau_yy': stress_exact_grids.get('tau_yy', torch.zeros_like(T_exact_grid)).cpu(),
        }
        
        visco_final_path = os.path.join(final_dir, 'PINN_viscoelastic_fields.png')
        plot2D_viscoelastic_final(
            X, Y, fields_pred, fields_exact, epochs,
            save_path=visco_final_path,
            internal_points=internal_pts,
            boundary_points=boundary_pts,
            physics_points=xy_physics
        )
    
    # Generazione GIF
    print(f"Creazione GIF con {len(plot_files)} frames...")
    if plot_files:
        gif_path = os.path.join(final_dir, 'PINNtraining_evolution.gif')
        save_gif_PIL(gif_path, plot_files, fps=3, loop=1, delete_files=True)
    
    # Costruzione dei phase markers per Staged Training
    _phase_markers = None
    if staged_training:
        _phase_markers = [
            {'epoch': half_epochs, 'label': 'Fase 2 (Tau)', 'color': 'purple'},
        ]
        if warmup_epochs > 0:
            _phase_markers.insert(0, {'epoch': warmup_epochs, 'label': 'End Warmup', 'color': 'red'})
    
    # Determina quali loss sono attive (peso > 0)
    _active_keys = set()
    if loss_weights.get('data', 0) > 0: _active_keys.add('data')
    if loss_weights.get('bc', 0) > 0: _active_keys.add('bc')
    if loss_weights.get('physics', 0) > 0: _active_keys.add('physics')
    
    # Plot Loss History con split tra Adam e L-BFGS
    loss_history.plot_losses(
        warmup_epoch=warmup_epochs, 
        adam_epochs=epochs,
        save_path=os.path.join(final_dir, 'PINNloss_history.png'), 
        experiment_name=experiment_name, 
        show_plot=show_plots_interactively,
        skip_epochs=50,
        phase_markers=_phase_markers,
        smoothing_alpha=0.95,
        active_loss_keys=_active_keys if _active_keys else None
    )
    
    # Plot Gradient History if available
    loss_history.plot_gradients(save_path=os.path.join(final_dir, 'PINN_gradients.png'), experiment_name=f"{experiment_name} Gradients", show_plot=show_plots_interactively)
    
    # Plot Weight History if available
    loss_history.plot_weights(save_path=os.path.join(final_dir, 'PINN_weights.png'), experiment_name=f"{experiment_name} Weights", show_plot=show_plots_interactively)

    if show_plots_interactively:
        plt.show()
    else:
        plt.close("all")

    # RIPRISTINO PRECISIONE PER EVENTUALI CHIAMATE SUCCESSIVE
    torch.set_default_dtype(torch.float32)
    return loss_history
```

---

### 5. [Viscoelastic_main.py](file:///c:/Users/eaglw/Documents/PINN%20tesi/Viscoelastic/Viscoelastic_main.py) — Integrazione Main

- **Import aggiornati** con le nuove funzioni
- **`stress_exact_grids`** ora include anche `p` (pressione)
- **Metriche**: sostituito `compute_metrics` con `compute_viscoelastic_metrics`
- **Comparison multi-campo**: genera `Comparison_Viscoelastic_AllFields.png` con error maps per tutti i campi × tutti i goal
- **Fix `shutil.rmtree`**: rimosso il `finally` che cancellava la cartella plots (conteneva i plot degli stress)
- **Error handling**: il `finally` è stato sostituito con `except` con traceback per non perdere silenziosamente errori

```diff:Viscoelastic_main.py
import torch
import torch.nn as nn
import os
import sys
import shutil
import itertools
from datetime import datetime

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Import funzioni esterne
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from func.logging_utils import compute_metrics, update_results_csv
from func.sampling_utils import generate_internal_points, generate_grid_points
from func.graphic_func import plot2D_unified_comparison, plot_loss_comparison

# Import locali Viscoelastic
from Viscoelastic.src.Viscoelastic_PINN import train_ViscoelasticPINN, FCN, ViscoelasticCombinedModel, VelocityInferenceWrapper, get_activation_name, format_layers_name
from Viscoelastic.src.Viscoelastic_physics import ViscoelasticPhysics, generate_boundaries

torch.backends.cuda.matmul.allow_tf32 = True  
torch.backends.cudnn.benchmark = True           
torch.backends.cudnn.deterministic = False      

def setup_experiment_folder(parent_dir, goal_folder, description):
    exp_dir = os.path.join(parent_dir, goal_folder)
    plots_dir = os.path.join(exp_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    return exp_dir, plots_dir

# --- SETUP DISPOSITIVO ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)
print(f"Using device: {device} with default dtype: {torch.get_default_dtype()}")

show_plots_interactively = False 

# Cases to run: 0 (Pure Phys), 1 (Phys+Data), 2 (Solo Data)
goals_to_run = [0, 1, 2]

# --- CONFIGURATION FLAGS ---
STAGED_TRAINING = True 

# --- HYPERPARAMETERS GRID SEARCH SETUP ---
layers_options = [[2, 120, 100, 80, 60, 40, 20, 1]] 
epochs_options = [80]
activation_options = [nn.SiLU]
lr_strategies = ['plateau']
weighting_options = ['dynamic']

STATIC_WEIGHTS = {'bc': 1.0, 'physics': 20.0, 'data': 100.0}
STATIC_WEIGHT_STR = "BC=1-PHYS=20-DATA=100"
DYNAMIC_WEIGHT_STR = "Dynamic-Annealing"

base_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiments_weighted')
results_csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.csv')

# --- CARICAMENTO DATASET ---
dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset', 'oldroydb_clean.pt')

if not os.path.exists(dataset_path):
    print(f"❌ Dataset non trovato in: {dataset_path}")
    sys.exit(1)

dataset = torch.load(dataset_path, map_location=device, weights_only=False)
for key in ['coords', 'u', 'v', 'p', 'psi', 'tau_xx', 'tau_xy', 'tau_yy', 'u_exact', 'p_exact', 'psi_exact', 'tau_xx_exact', 'tau_xy_exact', 'tau_yy_exact']:
    if key in dataset:
        dataset[key] = dataset[key].to(torch.float32)
params = dataset['params']

Lx, Ly, mu_s, mu_p, lam, u_max = params['L'], params['H'], params.get('mu_s', 0.005), params.get('mu_p', 0.005), params.get('lam', 1.0), params['u_max']
print(f"Dataset caricato: L={Lx}, H={Ly}, mu_s={mu_s}, mu_p={mu_p}, lam={lam}, u_max={u_max}")

xy_grid_flat = dataset['coords']
u_exact = dataset['u_exact']
p_exact = dataset['p_exact']
psi_exact = dataset['psi_exact']
v_exact = torch.zeros_like(u_exact)
tau_xx_exact = dataset.get('tau_xx_exact', torch.zeros_like(u_exact))
tau_xy_exact = dataset.get('tau_xy_exact', torch.zeros_like(u_exact))
tau_yy_exact = dataset.get('tau_yy_exact', torch.zeros_like(u_exact))

x_sorted = torch.unique(xy_grid_flat[:, 0], sorted=True)
y_sorted = torch.unique(xy_grid_flat[:, 1], sorted=True)
Nx_dom, Ny_dom = len(x_sorted), len(y_sorted)

X = xy_grid_flat[:, 0].reshape(Ny_dom, Nx_dom)
Y = xy_grid_flat[:, 1].reshape(Ny_dom, Nx_dom)
U_grid = u_exact.reshape(Ny_dom, Nx_dom)
P_grid = p_exact.reshape(Ny_dom, Nx_dom)
TAU_XX_grid = tau_xx_exact.reshape(Ny_dom, Nx_dom)
TAU_XY_grid = tau_xy_exact.reshape(Ny_dom, Nx_dom)
TAU_YY_grid = tau_yy_exact.reshape(Ny_dom, Nx_dom)
validation_grid_u = (xy_grid_flat, U_grid, X, Y)
stress_exact_grids = {'tau_xx': TAU_XX_grid, 'tau_xy': TAU_XY_grid, 'tau_yy': TAU_YY_grid}

margin=2e-2
Nx_grid_master, Ny_grid_master = 40, 40
xy_master_grid = generate_grid_points(Nx_grid_master, Ny_grid_master, Lx, Ly, margin=margin, device=device)

# Controlla che tutti i punti rispettino il dominio
assert xy_master_grid[:, 0].min() >= 0 and xy_master_grid[:, 0].max() <= Lx
assert xy_master_grid[:, 1].min() >= 0 and xy_master_grid[:, 1].max() <= Ly

# --- BOUNDARY CONDITIONS (u, v, p) ---
xy_master_boundary, uvp_master_boundary = generate_boundaries(Lx, Ly, u_max, p_exact, P_grid, Nx_dom, Ny_dom, device)

num_subset = 1000
torch.manual_seed(42)
idx = torch.randperm(xy_grid_flat.shape[0])[:num_subset]
xy_pinn_data = xy_grid_flat[idx]
psip_pinn_data = torch.cat([psi_exact[idx], p_exact[idx], tau_xx_exact[idx], tau_xy_exact[idx], tau_yy_exact[idx]], dim=1) 

pinn_data_internal = (xy_pinn_data, psip_pinn_data)
pinn_data_boundary = (xy_master_boundary, uvp_master_boundary)

# --- GRID SEARCH EXECUTION ---
configs = list(itertools.product(layers_options, epochs_options, activation_options, lr_strategies, weighting_options))
print(f"Starting Weighted Grid Search over {len(configs)} configurations...")

def get_last(hist, key): 
    return hist.losses[key][-1] if (key in hist.losses and hist.losses[key]) else 0

for layers_config, epochs, act_fn, lr_strat, weight_mode in configs:
    torch.set_default_dtype(torch.float32)
    layers_str = format_layers_name(layers_config)
    act_str = get_activation_name(act_fn)
    config_name = f"L{layers_str}_E{epochs}_{act_str}_{lr_strat}_{weight_mode}"
    
    config_dir = os.path.join(base_output_dir, config_name)
    os.makedirs(config_dir, exist_ok=True)
    
    print(f"\n=== Running Configuration: {config_name} ===")
    
    histories, final_models = {}, {}
    base_lr = 1e-3
    if lr_strat == 'step_decay':
        lr_log_str = f"[{base_lr} -> {base_lr * (0.5**4)}]"
    elif lr_strat == 'plateau':
        lr_log_str = "[plateau min:1e-6]"
    else:
        lr_log_str = str(base_lr)

    is_dynamic = (weight_mode == 'dynamic')
    current_weight_str = DYNAMIC_WEIGHT_STR if is_dynamic else STATIC_WEIGHT_STR

    phys_problem = ViscoelasticPhysics(mu_s=mu_s, mu_p=mu_p, lam=lam)

    for goal in goals_to_run:
        # Mapping dei Goal: 0=PurePhys, 1=Phys+Data, 2=SoloData
        if goal == 0:
            label = "PurePhys"
            current_w = {'bc': 1.0, 'physics': 1.0, 'data': 0.0}
        elif goal == 1:
            label = "Phys+Data"
            current_w = {'bc': 1.0, 'physics': 1.0, 'data': 1.0}
        else: # goal == 2
            label = "SoloData"
            current_w = {'bc': 0.0, 'physics': 0.0, 'data': 1.0}

        prefix = f"{goal}_{label}"
        print(f"  > {label} ({config_name})")
        
        exp_dir, plots_dir = setup_experiment_folder(config_dir, prefix, f"{label} {weight_mode}")
        
        torch.manual_seed(123)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(123)
            
        torch.set_default_dtype(torch.float32)
        pinn_data_internal_fresh = (xy_pinn_data.float(), psip_pinn_data.float())
        pinn_data_boundary_fresh = (xy_master_boundary.float(), uvp_master_boundary.float())
        
        # Forziamo l'ultimo layer
        layers_psi = layers_config[:-1] + [1]
        layers_p = layers_config[:-1] + [1]
        layers_tau = layers_config[:-1] + [3] # tau ha 3 componenti (xx, xy, yy)
        
        model_psi = FCN(layers=layers_psi, activation_fn=act_fn).to(device).to(torch.float32)
        model_p = FCN(layers=layers_p, activation_fn=act_fn).to(device).to(torch.float32)
        model_tau = FCN(layers=layers_tau, activation_fn=act_fn).to(device).to(torch.float32)
        model_combined = ViscoelasticCombinedModel(model_psi, model_p, model_tau)

        # Passiamo una lista unica di parametri all'ottimizzatore
        optimizer_params = list(model_combined.parameters())
        optimizer = torch.optim.Adam(optimizer_params, lr=base_lr)
        
        # Se siamo nel Goal 2 (SoloData), il dynamic weighting non ha senso (una sola componente)
        # Lo disabilitiamo localmente per questa run
        run_is_dynamic = is_dynamic if goal != 2 else False

        # Se non siamo in modalità dinamica, applichiamo i pesi statici (tranne che per SoloData)
        effective_w = dict(current_w)
        if not run_is_dynamic and goal != 2:
            effective_w['bc'] *= STATIC_WEIGHTS['bc']
            effective_w['physics'] *= STATIC_WEIGHTS['physics']
            effective_w['data'] *= STATIC_WEIGHTS['data']

        warmup = 0 if goal == 2 else epochs // 5

        try:
            use_staged = STAGED_TRAINING and goal != 2
            history = train_ViscoelasticPINN(
                model=model_combined, optimizer=optimizer,
                data_internal=pinn_data_internal_fresh, data_boundary=pinn_data_boundary_fresh,
                validation_grid=validation_grid_u, physics_problem=phys_problem,
                epochs=epochs, plots_dir=plots_dir, final_dir=exp_dir,
                show_plots_interactively=show_plots_interactively,
                log_gradients_every=500, collocation_points=xy_master_grid,
                lr_strategy=lr_strat, loss_weights=effective_w, dynamic_weighting=run_is_dynamic,
                update_weights_every=100, warmup_epochs=warmup,
                experiment_name=f"Viscoelastic {label}", val_label="u (Velocity)",
                stress_exact_grids=stress_exact_grids,
                staged_training=use_staged, base_lr=base_lr
            )
            
            # NOTA: compute_metrics richiede il wrapper VelocityInferenceWrapper
            # perché il modello combinato produce 5 output [psi,p,tau_xx,tau_xy,tau_yy]
            # mentre le metriche si calcolano solo sulla velocità u.
            metrics_wrapper = VelocityInferenceWrapper(model_combined, phys_problem)
            l2_err, max_err = compute_metrics(metrics_wrapper, xy_grid_flat, U_grid)
            
            log_data = {
                'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Architecture': str(layers_config),
                'Activation_Func': act_str, 'Epochs': epochs, 'Run_Type': label,
                'Optimizer': 'Adam', 'Learning_Rate': lr_log_str, 
                'Loss_Total': get_last(history, 'total_loss'), 'Loss_Physics': get_last(history, 'pde_loss'),
                'Loss_Boundary': get_last(history, 'bc_loss'), 'Loss_Data': get_last(history, 'data_loss'),
                'L2_Relative_Error': l2_err, 'Max_Relative_Error_Peak': max_err,
                'Seed': 123, 'n_points': xy_pinn_data.shape[0] if goal in [1, 2] else 0,
                'Loss_Weight': current_weight_str
            }
            update_results_csv(results_csv_path, log_data)
            histories[label] = history
            final_models[label] = model_combined
        finally:
            if os.path.exists(plots_dir):
                shutil.rmtree(plots_dir)

    print(f"  > Generating Comparisons for {config_name}...")
    results_dir = os.path.join(config_dir, 'comparisons')
    os.makedirs(results_dir, exist_ok=True)
    
    model_results = []
    for label, model in final_models.items():
        model.eval()
        with torch.set_grad_enabled(True):
            x_input = xy_grid_flat.clone().to(next(model.parameters()).dtype).requires_grad_(True)
            u_p, _, _, _ = phys_problem.get_velocity(model, x_input)
            pred = u_p.detach().cpu().to(torch.float32).reshape(Ny_dom, Nx_dom)
        model_results.append({'T_pred': pred, 'label': label})
    
    if model_results:
        hparams = {'arch': layers_str, 'epochs': str(epochs), 'act': act_str, 'lr_strategy': lr_strat, 'weight': current_weight_str}
        # X, Y, U_grid possono essere su CUDA; i plot li richiedono su CPU
        plot2D_unified_comparison(X.cpu(), Y.cpu(), U_grid.cpu(), model_results, hparams, save_path=os.path.join(results_dir, 'Comparison_Unified_ErrorMaps.png'))
    
    if len(histories) > 1:
        labels_list = list(histories.keys())
        hist_list = [histories[l] for l in labels_list]
        plot_loss_comparison(hist_list, labels_list, save_path=os.path.join(results_dir, 'Comparison_Loss_All_Goals.png'))

print("\nWeighted Grid Search configurations completed.")
===
import torch
import torch.nn as nn
import os
import sys
import shutil
import itertools
from datetime import datetime

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Import funzioni esterne
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from func.logging_utils import compute_metrics, compute_viscoelastic_metrics, update_results_csv
from func.sampling_utils import generate_internal_points, generate_grid_points
from func.graphic_func import plot2D_unified_comparison, plot_loss_comparison, plot2D_viscoelastic_comparison

# Import locali Viscoelastic
from Viscoelastic.src.Viscoelastic_PINN import train_ViscoelasticPINN, FCN, ViscoelasticCombinedModel, VelocityInferenceWrapper, get_activation_name, format_layers_name
from Viscoelastic.src.Viscoelastic_physics import ViscoelasticPhysics, generate_boundaries

torch.backends.cuda.matmul.allow_tf32 = True  
torch.backends.cudnn.benchmark = True           
torch.backends.cudnn.deterministic = False      

def setup_experiment_folder(parent_dir, goal_folder, description):
    exp_dir = os.path.join(parent_dir, goal_folder)
    plots_dir = os.path.join(exp_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    return exp_dir, plots_dir

# --- SETUP DISPOSITIVO ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)
print(f"Using device: {device} with default dtype: {torch.get_default_dtype()}")

show_plots_interactively = False 

# Cases to run: 0 (Pure Phys), 1 (Phys+Data), 2 (Solo Data)
goals_to_run = [0, 1, 2]

# --- CONFIGURATION FLAGS ---
STAGED_TRAINING = True 

# --- HYPERPARAMETERS GRID SEARCH SETUP ---
layers_options = [[2, 120, 100, 80, 60, 40, 20, 1]] 
epochs_options = [80]
activation_options = [nn.SiLU]
lr_strategies = ['plateau']
weighting_options = ['dynamic']

STATIC_WEIGHTS = {'bc': 1.0, 'physics': 20.0, 'data': 100.0}
STATIC_WEIGHT_STR = "BC=1-PHYS=20-DATA=100"
DYNAMIC_WEIGHT_STR = "Dynamic-Annealing"

base_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiments_weighted')
results_csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.csv')

# --- CARICAMENTO DATASET ---
dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset', 'oldroydb_clean.pt')

if not os.path.exists(dataset_path):
    print(f"❌ Dataset non trovato in: {dataset_path}")
    sys.exit(1)

dataset = torch.load(dataset_path, map_location=device, weights_only=False)
for key in ['coords', 'u', 'v', 'p', 'psi', 'tau_xx', 'tau_xy', 'tau_yy', 'u_exact', 'p_exact', 'psi_exact', 'tau_xx_exact', 'tau_xy_exact', 'tau_yy_exact']:
    if key in dataset:
        dataset[key] = dataset[key].to(torch.float32)
params = dataset['params']

Lx, Ly, mu_s, mu_p, lam, u_max = params['L'], params['H'], params.get('mu_s', 0.005), params.get('mu_p', 0.005), params.get('lam', 1.0), params['u_max']
print(f"Dataset caricato: L={Lx}, H={Ly}, mu_s={mu_s}, mu_p={mu_p}, lam={lam}, u_max={u_max}")

xy_grid_flat = dataset['coords']
u_exact = dataset['u_exact']
p_exact = dataset['p_exact']
psi_exact = dataset['psi_exact']
v_exact = torch.zeros_like(u_exact)
tau_xx_exact = dataset.get('tau_xx_exact', torch.zeros_like(u_exact))
tau_xy_exact = dataset.get('tau_xy_exact', torch.zeros_like(u_exact))
tau_yy_exact = dataset.get('tau_yy_exact', torch.zeros_like(u_exact))

x_sorted = torch.unique(xy_grid_flat[:, 0], sorted=True)
y_sorted = torch.unique(xy_grid_flat[:, 1], sorted=True)
Nx_dom, Ny_dom = len(x_sorted), len(y_sorted)

X = xy_grid_flat[:, 0].reshape(Ny_dom, Nx_dom)
Y = xy_grid_flat[:, 1].reshape(Ny_dom, Nx_dom)
U_grid = u_exact.reshape(Ny_dom, Nx_dom)
P_grid = p_exact.reshape(Ny_dom, Nx_dom)
TAU_XX_grid = tau_xx_exact.reshape(Ny_dom, Nx_dom)
TAU_XY_grid = tau_xy_exact.reshape(Ny_dom, Nx_dom)
TAU_YY_grid = tau_yy_exact.reshape(Ny_dom, Nx_dom)
validation_grid_u = (xy_grid_flat, U_grid, X, Y)
stress_exact_grids = {'p': P_grid, 'tau_xx': TAU_XX_grid, 'tau_xy': TAU_XY_grid, 'tau_yy': TAU_YY_grid}

margin=2e-2
Nx_grid_master, Ny_grid_master = 40, 40
xy_master_grid = generate_grid_points(Nx_grid_master, Ny_grid_master, Lx, Ly, margin=margin, device=device)

# Controlla che tutti i punti rispettino il dominio
assert xy_master_grid[:, 0].min() >= 0 and xy_master_grid[:, 0].max() <= Lx
assert xy_master_grid[:, 1].min() >= 0 and xy_master_grid[:, 1].max() <= Ly

# --- BOUNDARY CONDITIONS (u, v, p) ---
xy_master_boundary, uvp_master_boundary = generate_boundaries(Lx, Ly, u_max, p_exact, P_grid, Nx_dom, Ny_dom, device)

num_subset = 1000
torch.manual_seed(42)
idx = torch.randperm(xy_grid_flat.shape[0])[:num_subset]
xy_pinn_data = xy_grid_flat[idx]
psip_pinn_data = torch.cat([psi_exact[idx], p_exact[idx], tau_xx_exact[idx], tau_xy_exact[idx], tau_yy_exact[idx]], dim=1) 

pinn_data_internal = (xy_pinn_data, psip_pinn_data)
pinn_data_boundary = (xy_master_boundary, uvp_master_boundary)

# --- GRID SEARCH EXECUTION ---
configs = list(itertools.product(layers_options, epochs_options, activation_options, lr_strategies, weighting_options))
print(f"Starting Weighted Grid Search over {len(configs)} configurations...")

def get_last(hist, key): 
    return hist.losses[key][-1] if (key in hist.losses and hist.losses[key]) else 0

for layers_config, epochs, act_fn, lr_strat, weight_mode in configs:
    torch.set_default_dtype(torch.float32)
    layers_str = format_layers_name(layers_config)
    act_str = get_activation_name(act_fn)
    config_name = f"L{layers_str}_E{epochs}_{act_str}_{lr_strat}_{weight_mode}"
    
    config_dir = os.path.join(base_output_dir, config_name)
    os.makedirs(config_dir, exist_ok=True)
    
    print(f"\n=== Running Configuration: {config_name} ===")
    
    histories, final_models = {}, {}
    base_lr = 1e-3
    if lr_strat == 'step_decay':
        lr_log_str = f"[{base_lr} -> {base_lr * (0.5**4)}]"
    elif lr_strat == 'plateau':
        lr_log_str = "[plateau min:1e-6]"
    else:
        lr_log_str = str(base_lr)

    is_dynamic = (weight_mode == 'dynamic')
    current_weight_str = DYNAMIC_WEIGHT_STR if is_dynamic else STATIC_WEIGHT_STR

    phys_problem = ViscoelasticPhysics(mu_s=mu_s, mu_p=mu_p, lam=lam)

    for goal in goals_to_run:
        # Mapping dei Goal: 0=PurePhys, 1=Phys+Data, 2=SoloData
        if goal == 0:
            label = "PurePhys"
            current_w = {'bc': 1.0, 'physics': 1.0, 'data': 0.0}
        elif goal == 1:
            label = "Phys+Data"
            current_w = {'bc': 1.0, 'physics': 1.0, 'data': 1.0}
        else: # goal == 2
            label = "SoloData"
            current_w = {'bc': 0.0, 'physics': 0.0, 'data': 1.0}

        prefix = f"{goal}_{label}"
        print(f"  > {label} ({config_name})")
        
        exp_dir, plots_dir = setup_experiment_folder(config_dir, prefix, f"{label} {weight_mode}")
        
        torch.manual_seed(123)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(123)
            
        torch.set_default_dtype(torch.float32)
        pinn_data_internal_fresh = (xy_pinn_data.float(), psip_pinn_data.float())
        pinn_data_boundary_fresh = (xy_master_boundary.float(), uvp_master_boundary.float())
        
        # Forziamo l'ultimo layer
        layers_psi = layers_config[:-1] + [1]
        layers_p = layers_config[:-1] + [1]
        layers_tau = layers_config[:-1] + [3] # tau ha 3 componenti (xx, xy, yy)
        
        model_psi = FCN(layers=layers_psi, activation_fn=act_fn).to(device).to(torch.float32)
        model_p = FCN(layers=layers_p, activation_fn=act_fn).to(device).to(torch.float32)
        model_tau = FCN(layers=layers_tau, activation_fn=act_fn).to(device).to(torch.float32)
        model_combined = ViscoelasticCombinedModel(model_psi, model_p, model_tau)

        # Passiamo una lista unica di parametri all'ottimizzatore
        optimizer_params = list(model_combined.parameters())
        optimizer = torch.optim.Adam(optimizer_params, lr=base_lr)
        
        # Se siamo nel Goal 2 (SoloData), il dynamic weighting non ha senso (una sola componente)
        # Lo disabilitiamo localmente per questa run
        run_is_dynamic = is_dynamic if goal != 2 else False

        # Se non siamo in modalità dinamica, applichiamo i pesi statici (tranne che per SoloData)
        effective_w = dict(current_w)
        if not run_is_dynamic and goal != 2:
            effective_w['bc'] *= STATIC_WEIGHTS['bc']
            effective_w['physics'] *= STATIC_WEIGHTS['physics']
            effective_w['data'] *= STATIC_WEIGHTS['data']

        warmup = 0 if goal == 2 else epochs // 5

        try:
            use_staged = STAGED_TRAINING and goal != 2
            history = train_ViscoelasticPINN(
                model=model_combined, optimizer=optimizer,
                data_internal=pinn_data_internal_fresh, data_boundary=pinn_data_boundary_fresh,
                validation_grid=validation_grid_u, physics_problem=phys_problem,
                epochs=epochs, plots_dir=plots_dir, final_dir=exp_dir,
                show_plots_interactively=show_plots_interactively,
                log_gradients_every=500, collocation_points=xy_master_grid,
                lr_strategy=lr_strat, loss_weights=effective_w, dynamic_weighting=run_is_dynamic,
                update_weights_every=100, warmup_epochs=warmup,
                experiment_name=f"Viscoelastic {label}", val_label="u (Velocity)",
                stress_exact_grids=stress_exact_grids,
                staged_training=use_staged, base_lr=base_lr
            )
            
            # Metriche multi-campo per il caso viscoelastico
            fields_exact_for_metrics = {
                'u': U_grid, 'p': P_grid,
                'tau_xx': TAU_XX_grid, 'tau_xy': TAU_XY_grid, 'tau_yy': TAU_YY_grid
            }
            visco_metrics = compute_viscoelastic_metrics(
                model_combined, phys_problem, xy_grid_flat, fields_exact_for_metrics, Ny_dom, Nx_dom
            )
            
            # Metriche legacy (u only) per retrocompatibilità
            l2_err = visco_metrics['u'][0]
            max_err = visco_metrics['u'][1]
            
            log_data = {
                'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Architecture': str(layers_config),
                'Activation_Func': act_str, 'Epochs': epochs, 'Run_Type': label,
                'Optimizer': 'Adam', 'Learning_Rate': lr_log_str, 
                'Loss_Total': get_last(history, 'total_loss'), 'Loss_Physics': get_last(history, 'pde_loss'),
                'Loss_Boundary': get_last(history, 'bc_loss'), 'Loss_Data': get_last(history, 'data_loss'),
                'L2_Relative_Error': l2_err, 'Max_Relative_Error_Peak': max_err,
                'L2_u': visco_metrics['u'][0], 'Max_u': visco_metrics['u'][1],
                'L2_p': visco_metrics['p'][0], 'Max_p': visco_metrics['p'][1],
                'L2_tau_xx': visco_metrics['tau_xx'][0], 'Max_tau_xx': visco_metrics['tau_xx'][1],
                'L2_tau_xy': visco_metrics['tau_xy'][0], 'Max_tau_xy': visco_metrics['tau_xy'][1],
                'L2_tau_yy': visco_metrics['tau_yy'][0], 'Max_tau_yy': visco_metrics['tau_yy'][1],
                'Seed': 123, 'n_points': xy_pinn_data.shape[0] if goal in [1, 2] else 0,
                'Loss_Weight': current_weight_str
            }
            update_results_csv(results_csv_path, log_data)
            histories[label] = history
            final_models[label] = model_combined
        except Exception as e:
            print(f"  ❌ Errore nel training {label}: {e}")
            import traceback
            traceback.print_exc()

    print(f"  > Generating Comparisons for {config_name}...")
    results_dir = os.path.join(config_dir, 'comparisons')
    os.makedirs(results_dir, exist_ok=True)
    
    model_results = []
    model_results_multi = []  # Per comparison multi-campo
    for label, model in final_models.items():
        model.eval()
        with torch.set_grad_enabled(True):
            x_input = xy_grid_flat.clone().to(next(model.parameters()).dtype).requires_grad_(True)
            u_p, _, p_p, _ = phys_problem.get_velocity(model, x_input)
            out = model(x_input)
            pred_u = u_p.detach().cpu().to(torch.float32).reshape(Ny_dom, Nx_dom)
            pred_p = p_p.detach().cpu().to(torch.float32).reshape(Ny_dom, Nx_dom)
            pred_txx = out[:, 2].detach().cpu().to(torch.float32).reshape(Ny_dom, Nx_dom)
            pred_txy = out[:, 3].detach().cpu().to(torch.float32).reshape(Ny_dom, Nx_dom)
            pred_tyy = out[:, 4].detach().cpu().to(torch.float32).reshape(Ny_dom, Nx_dom)
        model_results.append({'T_pred': pred_u, 'label': label})
        model_results_multi.append({
            'label': label,
            'fields': {'u': pred_u, 'p': pred_p, 'tau_xx': pred_txx, 'tau_xy': pred_txy, 'tau_yy': pred_tyy}
        })
    
    if model_results:
        hparams = {'arch': layers_str, 'epochs': str(epochs), 'act': act_str, 'lr_strategy': lr_strat, 'weight': current_weight_str}
        # X, Y, U_grid possono essere su CUDA; i plot li richiedono su CPU
        plot2D_unified_comparison(X.cpu(), Y.cpu(), U_grid.cpu(), model_results, hparams, save_path=os.path.join(results_dir, 'Comparison_Unified_ErrorMaps.png'))
    
    # Comparison multi-campo per tutti i campi fisici
    if model_results_multi:
        fields_exact_cpu = {
            'u': U_grid.cpu(), 'p': P_grid.cpu(),
            'tau_xx': TAU_XX_grid.cpu(), 'tau_xy': TAU_XY_grid.cpu(), 'tau_yy': TAU_YY_grid.cpu()
        }
        plot2D_viscoelastic_comparison(
            X.cpu(), Y.cpu(), fields_exact_cpu, model_results_multi, hparams,
            save_path=os.path.join(results_dir, 'Comparison_Viscoelastic_AllFields.png')
        )
    
    if len(histories) > 1:
        labels_list = list(histories.keys())
        hist_list = [histories[l] for l in labels_list]
        plot_loss_comparison(hist_list, labels_list, save_path=os.path.join(results_dir, 'Comparison_Loss_All_Goals.png'))

print("\nWeighted Grid Search configurations completed.")
```

---

## Riepilogo Bug Risolti

| # | Problema | Soluzione |
|---|----------|-----------|
| 1 | Plot finale mostra solo u | Nuovo `plot2D_viscoelastic_final` con tutti i 5 campi |
| 2 | Metriche CSV solo su u | `compute_viscoelastic_metrics` con colonne per campo |
| 3 | Plot stress cancellati dal finally | Rimosso `shutil.rmtree(plots_dir)` |
| 4 | Error map satura a 10% | vmax adattivo con percentile 95° |
| 5 | data_loss in PurePhys | Filtro `active_loss_keys` |
| 6 | Nessun marker staged training | `phase_markers` nel loss plot |
| 7 | Spike LR illeggibili | EMA smoothing sovrapposto |
| 8 | Comparison solo su u | `plot2D_viscoelastic_comparison` multi-campo |

## Verification
- ✅ Syntax check passato su tutti i 5 file modificati
- ⏳ Run di test non ancora eseguita (richiede GPU e ~30min per E=8000)

## Output Attesi Dopo le Modifiche

Per ogni experiment folder, i nuovi file generati saranno:
```
experiments_weighted/<config>/
├── 0_PurePhys/
│   ├── PINNfinal_result.png        (invariato, solo u)
│   ├── PINN_viscoelastic_fields.png ← NUOVO: 5 campi × 3 colonne
│   ├── PINNloss_history.png        ← MIGLIORATO: marker, smoothing, filtro
│   ├── PINN_gradients.png
│   └── PINN_weights.png
├── 1_Phys+Data/
│   └── ... (stessa struttura)
├── 2_SoloData/
│   └── ... (stessa struttura)
└── comparisons/
    ├── Comparison_Unified_ErrorMaps.png    (invariato, solo u)
    ├── Comparison_Viscoelastic_AllFields.png ← NUOVO: error map 5×3
    └── Comparison_Loss_All_Goals.png
```
