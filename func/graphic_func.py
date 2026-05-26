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

def plot2D_comparison(X, Y, T_true, T_pred, epoch, save_path, physics_points=None, val_label='Value', show_points=False):
    """Genera grafici side-by-side: Predizione, Errore Assoluto, Errore Relativo.
    Rinominata da plot_comparison per uso generale.
    Aggiunge la visualizzazione dei punti di collocazione della fisica se forniti."""
    
    # Assicurati che siano su CPU per le operazioni matplotlib
    T_true = T_true.detach().cpu()
    T_pred = T_pred.detach().cpu()
    
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
    c1 = ax.contourf(X_np, Y_np, T_pred.detach().cpu().numpy(), levels=50, cmap='inferno', antialiased=True)
    plt.colorbar(c1, ax=ax, label=val_label)
    ax.set_title(f'Predizione (Epoch {epoch})')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal', adjustable='box')
    
    # Aggiungi i punti della fisica solo se richiesto esplicitamente (per evitare sovrapposizioni)
    if show_points and physics_points is not None and len(physics_points) > 0:
        xy_physics_np = physics_points.detach().cpu().numpy()
        # Se ci sono troppi punti, facciamo un subsampling per il plot
        if len(xy_physics_np) > 2000:
            idx = np.random.choice(len(xy_physics_np), 2000, replace=False)
            xy_physics_np = xy_physics_np[idx]
        ax.scatter(xy_physics_np[:, 0], xy_physics_np[:, 1], s=1, facecolor='white', edgecolor='none', marker='o', alpha=0.2, label='Punti Fisica')
        ax.legend(loc='upper right', fontsize='x-small', framealpha=0.5)

    # 2. Errore Assoluto (Più robusto)
    ax = axes[1]
    c2 = ax.contourf(X_np, Y_np, abs_error.detach().cpu().numpy(), levels=50, cmap='magma', antialiased=True)
    plt.colorbar(c2, ax=ax, label='Errore Assoluto')
    ax.set_title('Errore Assoluto |T_pred - T_true|')
    ax.set_xlabel('x')
    ax.set_aspect('equal', adjustable='box')

    # 3. Errore Relativo (Locale)
    ax = axes[2]
    # Usiamo vmin/vmax per evitare saturazione da outlier
    rel_error_np = rel_error.detach().cpu().numpy()
    # vmax fisso al 10%
    vmax_adaptive = 10.0 
    c3 = ax.contourf(X_np, Y_np, rel_error_np, levels=50, cmap='jet', vmin=0, vmax=vmax_adaptive, antialiased=True) 
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
    # Assicurati che siano su CPU
    T_true = T_true.detach().cpu()
    T_pred = T_pred.detach().cpu()
    
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
    c1 = ax.contourf(X_np, Y_np, T_pred.detach().cpu().numpy(), levels=50, cmap='inferno', antialiased=True)
    plt.colorbar(c1, ax=ax, label=val_label)
    ax.set_title(f'Prediction (Epoch {epoch})')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal', adjustable='box')
    
    # Overlay Points (with reduced sizes and no dark edges to avoid 'black lines' artifacts)
    if physics_points is not None and len(physics_points) > 0:
        xy_phys = physics_points.detach().cpu().numpy()
        if len(xy_phys) > 2000:
            xy_phys = xy_phys[np.random.choice(len(xy_phys), 2000, replace=False)]
        ax.scatter(xy_phys[:, 0], xy_phys[:, 1], s=1, facecolor='white', edgecolor='none', marker='o', alpha=0.2, label='Physics Points')
        
    if internal_points is not None and len(internal_points) > 0:
        xy_int = internal_points.detach().cpu().numpy()
        if len(xy_int) > 3000:
            xy_int = xy_int[np.random.choice(len(xy_int), 3000, replace=False)]
        ax.scatter(xy_int[:, 0], xy_int[:, 1], s=8, c='cyan', marker='o', alpha=0.6, edgecolor='none', label='Internal Points')
        
    if boundary_points is not None and len(boundary_points) > 0:
        xy_bc = boundary_points.detach().cpu().numpy()
        ax.scatter(xy_bc[:, 0], xy_bc[:, 1], s=12, c='red', marker='s', alpha=0.7, edgecolor='none', label='Boundary Points')
        
    if physics_points is not None or internal_points is not None or boundary_points is not None:
        ax.legend(loc='upper right', framealpha=0.9, fontsize='small')

    # 2. Relative Error
    ax = axes[1]
    rel_error_np_final = rel_error.detach().cpu().numpy()
    # vmax fisso al 10%
    vmax_adaptive = 10.0
    c2 = ax.contourf(X_np, Y_np, rel_error_np_final, levels=50, cmap='jet', vmin=0, vmax=vmax_adaptive, antialiased=True)
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
        
        T_pred = res['T_pred'].detach().cpu()
        T_true = T_true.detach().cpu()
        label = res['label']
        
        abs_error = torch.abs(T_pred - T_true)
        
        # Errore Relativo Standard con masking
        rel_error = torch.zeros_like(T_true)
        mask = torch.abs(T_true) > 0.01
        if mask.sum() > 0:
            rel_error[mask] = (abs_error[mask] / torch.abs(T_true[mask])) * 100
            
        rel_error_np = rel_error.detach().cpu().numpy()
        
        # vmax fisso al 10%
        vmax_adaptive = 10.0
        c = ax.contourf(X_np, Y_np, rel_error_np, levels=50, cmap='jet', vmin=0, vmax=vmax_adaptive, antialiased=True)
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
        
        T_pred = T_pred.detach().cpu()
        T_true = T_true.detach().cpu()
        
        abs_error = torch.abs(T_pred - T_true)
        
        # Errore Relativo Standard con masking
        rel_error = torch.zeros_like(T_true)
        mask = torch.abs(T_true) > 0.01
        if mask.sum() > 0:
            rel_error[mask] = (abs_error[mask] / torch.abs(T_true[mask])) * 100
            
        rel_error_np = rel_error.detach().cpu().numpy()
        
        # vmax fisso al 10%
        vmax_adaptive = 10.0
        c = ax.contourf(X_np, Y_np, rel_error_np, levels=50, cmap='jet', vmin=0, vmax=vmax_adaptive, antialiased=True)
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
    # Cast to same dtype (L-BFGS usa float64, ma i plot vogliono float32) e spostamento su CPU
    pred = pred.detach().cpu().float()
    exact = exact.detach().cpu().float()
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
    
    fig.suptitle(f'Viscoelastic VE — Final Results (Epoch {epoch})', fontsize=18, fontweight='bold', y=0.995)
    
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
        c = ax.contourf(X_np, Y_np, pred_np, levels=50, cmap=cmap, vmin=vmin_shared, vmax=vmax_shared, antialiased=True)
        plt.colorbar(c, ax=ax, label=flabel)
        ax.set_title(f'{flabel} — Prediction')
        ax.set_ylabel('y')
        ax.set_aspect('equal', adjustable='box')
        
        # Overlay punti solo sulla prima riga (con limiti e senza bordi neri)
        if i == 0:
            if physics_points is not None and len(physics_points) > 0:
                xy_p = physics_points.detach().cpu().numpy()
                if len(xy_p) > 2000:
                    xy_p = xy_p[np.random.choice(len(xy_p), 2000, replace=False)]
                ax.scatter(xy_p[:, 0], xy_p[:, 1], s=1, facecolor='white', edgecolor='none', marker='o', alpha=0.2, label='Physics')
            if internal_points is not None and len(internal_points) > 0:
                xy_i = internal_points.detach().cpu().numpy()
                if len(xy_i) > 3000:
                    xy_i = xy_i[np.random.choice(len(xy_i), 3000, replace=False)]
                ax.scatter(xy_i[:, 0], xy_i[:, 1], s=6, c='cyan', marker='o', alpha=0.5, edgecolor='none', label='Data')
            if boundary_points is not None and len(boundary_points) > 0:
                xy_b = boundary_points.detach().cpu().numpy()
                ax.scatter(xy_b[:, 0], xy_b[:, 1], s=10, c='red', marker='s', alpha=0.6, edgecolor='none', label='BC')
            ax.legend(loc='upper right', fontsize='x-small', framealpha=0.6)
        
        # Col 1: Soluzione Esatta
        ax = axes[i, 1]
        c = ax.contourf(X_np, Y_np, exact_np, levels=50, cmap=cmap, vmin=vmin_shared, vmax=vmax_shared, antialiased=True)
        plt.colorbar(c, ax=ax, label=flabel)
        ax.set_title(f'{flabel} — Exact')
        ax.set_aspect('equal', adjustable='box')
        
        # Col 2: Errore Relativo
        ax = axes[i, 2]
        rel_err = _compute_rel_error(pred.cpu(), exact.cpu())
        rel_err_np = rel_err.numpy()
        # vmax fisso al 10%
        vmax_err = 10.0
        c = ax.contourf(X_np, Y_np, rel_err_np, levels=50, cmap='jet', vmin=0, vmax=vmax_err, antialiased=True)
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
            # vmax fisso al 10%
            vmax_err = 10.0
            
            c = ax.contourf(X_np, Y_np, rel_err_np, levels=50, cmap='jet', vmin=0, vmax=vmax_err, antialiased=True)
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