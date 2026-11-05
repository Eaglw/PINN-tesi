import os
from PIL import Image
import matplotlib.pyplot as plt
import torch
import numpy as np
import matplotlib.tri as tri

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


# =============================================================================
#  UNSTRUCTURED MESH PLOTTING (tricontourf-based)
# =============================================================================

def _to_numpy_1d(arr):
    """Convert a tensor or array to a flat numpy array (CPU)."""
    if hasattr(arr, 'detach'):
        arr = arr.detach().cpu().numpy()
    return np.asarray(arr).ravel()


def plot2D_comparison(triang, field_exact_1d, field_pred_1d, epoch, save_path,
                      val_label='Value'):
    """Genera grafici: Predizione, Errore Assoluto, Errore Relativo su mesh non strutturata.
    Usa tricontourf su un oggetto Triangulation.

    Args:
        triang: matplotlib.tri.Triangulation object.
        field_exact_1d: 1-D tensor/array with exact field values at mesh nodes.
        field_pred_1d: 1-D tensor/array with predicted field values at mesh nodes.
        epoch: Current epoch number (used in title).
        save_path: Path to save the figure (None → plt.show()).
        val_label: Label for the colorbar of the solution plot.
    """
    pred_np = _to_numpy_1d(field_pred_1d)
    exact_np = _to_numpy_1d(field_exact_1d)

    abs_error = np.abs(pred_np - exact_np)

    # Relative error with dynamic masking
    rel_error = np.zeros_like(exact_np)
    max_val = np.max(np.abs(exact_np))
    threshold = max(0.05 * max_val, 1e-8)
    mask = np.abs(exact_np) > threshold
    if mask.sum() > 0:
        rel_error[mask] = (abs_error[mask] / np.abs(exact_np[mask])) * 100

    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

    # 1. Predicted solution
    ax = axes[0]
    c1 = ax.tricontourf(triang, pred_np, levels=50, cmap='inferno')
    plt.colorbar(c1, ax=ax, label=val_label)
    ax.set_title(f'Predizione (Epoch {epoch})')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal', adjustable='box')

    # 2. Absolute error
    ax = axes[1]
    c2 = ax.tricontourf(triang, abs_error, levels=50, cmap='magma')
    plt.colorbar(c2, ax=ax, label='Errore Assoluto')
    ax.set_title('Errore Assoluto |pred - exact|')
    ax.set_xlabel('x')
    ax.set_aspect('equal', adjustable='box')

    # 3. Relative error
    ax = axes[2]
    vmax_adaptive = 10.0
    c3 = ax.tricontourf(triang, rel_error, levels=50, cmap='jet', vmin=0, vmax=vmax_adaptive)
    plt.colorbar(c3, ax=ax, label='% Errore Relativo (|err|/|exact|)')
    ax.set_title('Errore Relativo % (|err| / |exact|)')
    ax.set_xlabel('x')
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot2D_final_result(triang, field_exact_1d, field_pred_1d, epoch, save_path,
                        internal_points=None, boundary_points=None, val_label='Value'):
    """Plot finale su mesh non strutturata: Soluzione + punti sovrapposti, Errore Relativo.
    Usa tricontourf su un oggetto Triangulation.

    Args:
        triang: matplotlib.tri.Triangulation object.
        field_exact_1d: 1-D tensor/array with exact field values.
        field_pred_1d: 1-D tensor/array with predicted field values.
        epoch: Current epoch number.
        save_path: Path to save the figure (None → plt.show()).
        internal_points: Optional (N,2) tensor/array of internal data points.
        boundary_points: Optional (N,2) tensor/array of boundary points.
        val_label: Label for the colorbar.
    """
    pred_np = _to_numpy_1d(field_pred_1d)
    exact_np = _to_numpy_1d(field_exact_1d)

    # Relative error with masking
    abs_error = np.abs(pred_np - exact_np)
    rel_error = np.zeros_like(exact_np)
    max_val = np.max(np.abs(exact_np))
    threshold = max(0.05 * max_val, 1e-8)
    mask = np.abs(exact_np) > threshold
    if mask.sum() > 0:
        rel_error[mask] = (abs_error[mask] / np.abs(exact_np[mask])) * 100

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # 1. Solution + Points
    ax = axes[0]
    c1 = ax.tricontourf(triang, pred_np, levels=50, cmap='inferno')
    plt.colorbar(c1, ax=ax, label=val_label)
    ax.set_title(f'Prediction (Epoch {epoch})')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal', adjustable='box')

    # Overlay Points
    if internal_points is not None:
        xy_int = internal_points.detach().cpu().numpy() if hasattr(internal_points, 'detach') else np.asarray(internal_points)
        s_int = max(2.0, 3000.0 / len(xy_int)) if len(xy_int) > 1000 else 4.0
        alpha_int = 0.8
        ax.scatter(xy_int[:, 0], xy_int[:, 1], s=s_int, c='cyan', marker='o',
                   alpha=alpha_int, edgecolor='none', label='Internal Points')

    if boundary_points is not None:
        xy_bc = boundary_points.detach().cpu().numpy() if hasattr(boundary_points, 'detach') else np.asarray(boundary_points)
        s_bc = max(3.5, 1500.0 / len(xy_bc)) if len(xy_bc) > 300 else 7.0
        alpha_bc = 0.9
        ax.scatter(xy_bc[:, 0], xy_bc[:, 1], s=s_bc, c='red', marker='s',
                   alpha=alpha_bc, edgecolor='none', label='Boundary Points')

    if internal_points is not None or boundary_points is not None:
        ax.legend(loc='upper right', framealpha=0.9, fontsize='small')

    # 2. Relative Error
    ax = axes[1]
    vmax_adaptive = 10.0
    c2 = ax.tricontourf(triang, rel_error, levels=50, cmap='jet', vmin=0, vmax=vmax_adaptive)
    plt.colorbar(c2, ax=ax, label='% Relative Error (|err|/|exact|)')
    ax.set_title('Relative Error % (|err| / |exact|)')
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


def plot2D_viscoelastic_final(triang, fields_pred, fields_exact, epoch, save_path,
                              physics_points=None, internal_points=None, boundary_points=None):
    """Plot multi-campo finale per il caso viscoelastico su mesh non strutturata.
    Genera una griglia n_fields × 3: Predizione | Soluzione Esatta | Errore Relativo.
    Usa tricontourf su un oggetto Triangulation.

    Args:
        triang: matplotlib.tri.Triangulation object.
        fields_pred: Dict {'u': 1d_tensor, 'p': 1d_tensor, 'tau_xx': ..., 'tau_xy': ..., 'tau_yy': ...}
        fields_exact: Dict with the same keys.
        epoch: Current epoch number.
        save_path: Path to save the figure (None → plt.show()).
    """
    field_names = ['u', 'p', 'tau_xx', 'tau_xy', 'tau_yy']
    field_labels = ['u (Velocity)', 'p (Pressure)', 'τ_xx', 'τ_xy', 'τ_yy']
    cmaps_field = ['inferno', 'viridis', 'plasma', 'plasma', 'plasma']

    n_fields = len(field_names)
    fig, axes = plt.subplots(n_fields, 3, figsize=(18, 4 * n_fields))

    fig.suptitle(f'Viscoelastic VE — Final Results (Epoch {epoch})', fontsize=18, fontweight='bold', y=0.995)

    for i, (fname, flabel, cmap) in enumerate(zip(field_names, field_labels, cmaps_field)):
        pred_raw = fields_pred.get(fname)
        exact_raw = fields_exact.get(fname)

        if pred_raw is None or exact_raw is None:
            for j in range(3):
                axes[i, j].set_visible(False)
            continue

        pred_np = _to_numpy_1d(pred_raw)
        exact_np = _to_numpy_1d(exact_raw)

        # Shared color limits
        vmin_shared = min(pred_np.min(), exact_np.min())
        vmax_shared = max(pred_np.max(), exact_np.max())

        # Col 0: Prediction
        ax = axes[i, 0]
        c = ax.tricontourf(triang, pred_np, levels=50, cmap=cmap, vmin=vmin_shared, vmax=vmax_shared)
        plt.colorbar(c, ax=ax, label=flabel)
        ax.set_title(f'{flabel} — Prediction')
        ax.set_ylabel('y')
        ax.set_aspect('equal', adjustable='box')

        # Overlay points only on the first row
        if i == 0:
            pass # Removed points to keep the plot clean as requested

        # Col 1: Exact solution
        ax = axes[i, 1]
        c = ax.tricontourf(triang, exact_np, levels=50, cmap=cmap, vmin=vmin_shared, vmax=vmax_shared)
        plt.colorbar(c, ax=ax, label=flabel)
        ax.set_title(f'{flabel} — Exact')
        ax.set_aspect('equal', adjustable='box')

        # Col 2: Relative error
        ax = axes[i, 2]
        abs_err = np.abs(pred_np - exact_np)
        rel_err = np.zeros_like(exact_np)
        max_val = np.max(np.abs(exact_np))
        thr = max(0.05 * max_val, 1e-8)
        m = np.abs(exact_np) > thr
        if m.sum() > 0:
            rel_err[m] = (abs_err[m] / np.abs(exact_np[m])) * 100

        vmax_err = 10.0
        c = ax.tricontourf(triang, rel_err, levels=50, cmap='jet', vmin=0, vmax=vmax_err)
        plt.colorbar(c, ax=ax, label='% Relative Error')
        ax.set_title(f'{flabel} — Rel. Error %')
        ax.set_aspect('equal', adjustable='box')

        # x-label only on last row
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


def plot2D_unified_comparison(triang, T_true_1d, model_results, hyperparams, save_path=None):
    """Genera una griglia dinamica di mappe di errore relativo su mesh non strutturata.
    Usa tricontourf su un oggetto Triangulation.

    Args:
        triang: matplotlib.tri.Triangulation object.
        T_true_1d: 1-D tensor/array with exact solution values at mesh nodes.
        model_results: List of dicts [{'T_pred': 1d_tensor, 'label': str}, ...].
        hyperparams: Dict {'arch': str, 'epochs': int, 'act': str}.
        save_path: Path to save the figure (None → plt.show()).
    """
    n = len(model_results)
    if n == 0:
        print("Warning: plot2D_unified_comparison called with 0 model results. Skipping.")
        return

    exact_np = _to_numpy_1d(T_true_1d)

    cols = min(n, 2)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 6 * rows), squeeze=False)

    arch = hyperparams.get('arch', 'N/A')
    epochs = hyperparams.get('epochs', 'N/A')
    act = hyperparams.get('act', 'N/A')

    fig.suptitle(f"Comparison: {arch} | Epochs: {epochs} | Activation: {act}", fontsize=18, fontweight='bold')

    for i, res in enumerate(model_results):
        row = i // cols
        col = i % cols
        ax = axes[row, col]

        pred_np = _to_numpy_1d(res['T_pred'])
        label = res['label']

        abs_error = np.abs(pred_np - exact_np)

        # Relative error with masking
        rel_error = np.zeros_like(exact_np)
        m = np.abs(exact_np) > 0.01
        if m.sum() > 0:
            rel_error[m] = (abs_error[m] / np.abs(exact_np[m])) * 100

        vmax_adaptive = 10.0
        c = ax.tricontourf(triang, rel_error, levels=50, cmap='jet', vmin=0, vmax=vmax_adaptive)
        cbar = plt.colorbar(c, ax=ax)
        cbar.set_label('% Relative Error (|err|/|exact|)', rotation=270, labelpad=15)

        ax.set_facecolor('lightgray')
        ax.set_title(label, fontsize=14)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal', adjustable='box')

    # Hide empty axes
    for i in range(n, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot2D_viscoelastic_comparison(triang, fields_exact, model_results_multi,
                                   hyperparams, save_path=None):
    """Comparison multi-campo tra diversi modelli su mesh non strutturata.
    Usa tricontourf su un oggetto Triangulation.

    Args:
        triang: matplotlib.tri.Triangulation object.
        fields_exact: Dict {'u': 1d_tensor, 'p': 1d_tensor, 'tau_xx': ..., ...}
        model_results_multi: List of dicts:
            [{'label': 'Model A', 'fields': {'u': 1d_tensor, 'p': ..., ...}}, ...]
        hyperparams: Dict for the suptitle.
        save_path: Path to save the figure (None → plt.show()).
    """
    field_names = ['u', 'p', 'tau_xx', 'tau_xy', 'tau_yy']
    field_labels = ['u (Velocity)', 'p (Pressure)', 'τ_xx', 'τ_xy', 'τ_yy']

    n_models = len(model_results_multi)
    n_fields = len(field_names)

    fig, axes = plt.subplots(n_fields, n_models, figsize=(6 * n_models, 3.5 * n_fields), squeeze=False)

    arch = hyperparams.get('arch', 'N/A')
    epochs = hyperparams.get('epochs', 'N/A')
    act = hyperparams.get('act', 'N/A')
    fig.suptitle(f'Relative Error Comparison | {arch} | E={epochs} | {act}', fontsize=16, fontweight='bold')

    for row, (fname, flabel) in enumerate(zip(field_names, field_labels)):
        exact_raw = fields_exact.get(fname)
        if exact_raw is None:
            for col in range(n_models):
                axes[row, col].set_visible(False)
            continue

        exact_np = _to_numpy_1d(exact_raw)

        for col, mres in enumerate(model_results_multi):
            ax = axes[row, col]
            pred_raw = mres['fields'].get(fname)
            label = mres['label']

            if pred_raw is None:
                ax.set_visible(False)
                continue

            pred_np = _to_numpy_1d(pred_raw)

            # Relative error
            abs_err = np.abs(pred_np - exact_np)
            rel_err = np.zeros_like(exact_np)
            max_val = np.max(np.abs(exact_np))
            thr = max(0.05 * max_val, 1e-8)
            m = np.abs(exact_np) > thr
            if m.sum() > 0:
                rel_err[m] = (abs_err[m] / np.abs(exact_np[m])) * 100

            vmax_err = 10.0
            c = ax.tricontourf(triang, rel_err, levels=50, cmap='jet', vmin=0, vmax=vmax_err)
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


def generate_epoch_diagnostic_plot(model, physics_problem, xy_grid, T_exact_grid, triang, epoch, plots_dir, plot_every, val_label, plot_files):
    """Genera e salva il plot diagnostico dell'epoca corrente, ripulendo la cache CUDA."""
    model.eval()
    with torch.set_grad_enabled(True): 
        xy_grid_val = xy_grid.clone().detach().requires_grad_(True)
        if hasattr(physics_problem, 'get_velocity'):
            u_pred, _, _, _ = physics_problem.get_velocity(model, xy_grid_val)
            T_pred_grid = u_pred.detach().cpu().view(-1)
        else:
            u_pred = model(xy_grid_val)[:, 0].detach().cpu()
            T_pred_grid = u_pred.view(-1)
        del xy_grid_val
        
    plot_path = os.path.join(plots_dir, f'epoch_{epoch+1}.png')
    plot2D_comparison(triang, T_exact_grid.cpu().view(-1), T_pred_grid, epoch+1, plot_path, val_label=val_label)
    plot_files.append(plot_path)
    
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate_final_training_plots(final_dir, plots_dir, triang, T_exact_grid, T_final, p_final, tau_final, stress_exact_grids, plot_files, epochs, val_label, internal_pts, boundary_pts, physics_pts=None):
    """Genera i plot finali (singolo e multi-campo), crea la GIF ed elimina la cartella temporanea."""
    import shutil
    # --- PLOT COMPARATIVO PRINCIPALE ---
    final_path = os.path.join(final_dir, 'VEfinal_result.png')
    plot2D_final_result(triang, T_exact_grid.view(-1), T_final, epochs, save_path=final_path, 
                        internal_points=internal_pts, boundary_points=boundary_pts, val_label=val_label)
    
    # --- PLOT MULTI-CAMPO VISCOELASTICO ---
    if stress_exact_grids is not None:
        fields_pred = {
            'u': T_final, 
            'p': p_final.detach().cpu().view(-1),
            'tau_xx': tau_final[:, 0].detach().cpu().view(-1),
            'tau_xy': tau_final[:, 1].detach().cpu().view(-1),
            'tau_yy': tau_final[:, 2].detach().cpu().view(-1),
        }
        fields_exact = {
            'u': T_exact_grid.view(-1),
            'p': stress_exact_grids.get('p', torch.zeros_like(T_exact_grid)).cpu().view(-1),
            'tau_xx': stress_exact_grids.get('tau_xx', torch.zeros_like(T_exact_grid)).cpu().view(-1),
            'tau_xy': stress_exact_grids.get('tau_xy', torch.zeros_like(T_exact_grid)).cpu().view(-1),
            'tau_yy': stress_exact_grids.get('tau_yy', torch.zeros_like(T_exact_grid)).cpu().view(-1),
        }
        visco_final_path = os.path.join(final_dir, 'VE_viscoelastic_fields.png')
        plot2D_viscoelastic_final(triang, fields_pred, fields_exact, epochs, save_path=visco_final_path,
                                  physics_points=physics_pts, internal_points=internal_pts, boundary_points=boundary_pts)
    
    # --- GIF E PULIZIA ---
    if plot_files:
        gif_path = os.path.join(final_dir, 'VEtraining_evolution.gif')
        save_gif_PIL(gif_path, plot_files, fps=3, loop=1, delete_files=True)
    
    shutil.rmtree(plots_dir, ignore_errors=True)