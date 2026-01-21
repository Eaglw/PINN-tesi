"""
Experiment Goal: 3_PINN_PurePhys
Description: PINN PurePhys. Config: L2_50x4_1_E50000_Tanh
"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from tqdm import tqdm

# Import function for GIF and loss comparison
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from func.graphic_func import save_gif_PIL, plot2D_comparison
from func.history_tracker import TrainingHistory, compute_pinn_loss


# Configurazione dispositivo e precisione
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

# ---  DEFINIZIONE DELLA LOSS FISICA ---
def heat2d_physics_loss(model, xy_p):
    """
    Calcola il residuo dell'equazione di Laplace 2D: d2T/dx2 + d2T/dy2 = 0
    """
    # xy_p è (N, 2). Richiediamo gradienti.
    T = model(xy_p)
    
    # Calcolo gradienti primi
    grads = torch.autograd.grad(T, xy_p, torch.ones_like(T), create_graph=True)[0]
    dT_dx = grads[:, 0]
    dT_dy = grads[:, 1]
    
    # Calcolo gradienti secondi
    # Nota: autograd.grad restituisce una tupla, prendiamo [0]
    grads2_x = torch.autograd.grad(dT_dx, xy_p, torch.ones_like(dT_dx), create_graph=True, allow_unused=True)[0]
    d2T_dx2 = grads2_x[:, 0] if grads2_x is not None else torch.zeros_like(dT_dx)
    
    grads2_y = torch.autograd.grad(dT_dy, xy_p, torch.ones_like(dT_dy), create_graph=True, allow_unused=True)[0]
    d2T_dy2 = grads2_y[:, 1] if grads2_y is not None else torch.zeros_like(dT_dy)
    
    # Residuo PDE
    res = d2T_dx2 + d2T_dy2
    return torch.mean(res**2)

def train_modelPINN(
    model,
    optimizer,
    data_internal,
    data_boundary,
    validation_grid,
    epochs=20000,
    physics_problem=None,
    plots_dir='plots',
    final_dir='Heat2D/Results',
    show_plots_interactively=True,
    log_gradients_every=0,
    loss_weights=None,
    warmup_epochs=None,
    n_collocation=(50, 50),
    collocation_points=None
):
    """
    Esegue il training della PINN.
    
    Args:
        model: Istanza del modello FCN.
        optimizer: Istanza dell'ottimizzatore.
        data_internal: Tupla (xy_int, T_int).
        data_boundary: Tupla (xy_bc, T_bc).
        validation_grid: Tupla (xy_grid, T_exact_grid, X, Y).
        physics_problem: Istanza di PhysicsProblem (opzionale).
        log_gradients_every: Se > 0, calcola e logga le norme dei gradienti ogni N epoche.
        loss_weights: Dizionario con i pesi delle loss (keys: 'data', 'bc', 'physics').
        warmup_epochs: Numero di epoche di warmup (solo dati).
        n_collocation: Numero di punti di collocazione per dimensione (int o tuple (Nx, Ny)).
        collocation_points: (Opzionale) Tensor (N, 2) con i punti di collocazione espliciti.
                            Se fornito, ignora n_collocation.
    """
    
    # Unpack dei dati
    xy_int, T_int = data_internal
    xy_bc, T_bc = data_boundary
    xy_grid, T_exact_grid, X, Y = validation_grid
    
    # Ricavo dimensioni griglia per reshape e limiti dominio
    Nx_dom, Ny_dom = X.shape
    Lx = X.max().item()
    Ly = Y.max().item()
    
    # Directory Output
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)
    
    plot_files = []
    
    # Generazione punti di collocazione (STRETTAMENTE INTERNI)
    if collocation_points is not None:
        xy_physics = collocation_points.clone()
        if not xy_physics.requires_grad:
            xy_physics.requires_grad_(True)
    else:
        if isinstance(n_collocation, int):
            Nx_phys, Ny_phys = n_collocation, n_collocation
        else:
            Nx_phys, Ny_phys = n_collocation
            
        # Usiamo +2 e slicing [1:-1] per escludere 0 e 1
        x_phys_line = torch.linspace(0, Lx, Nx_phys + 2, device=device)[1:-1]
        y_phys_line = torch.linspace(0, Ly, Ny_phys + 2, device=device)[1:-1]
        X_phys, Y_phys = torch.meshgrid(x_phys_line, y_phys_line, indexing='xy')
        xy_physics = torch.stack([X_phys.flatten(), Y_phys.flatten()], dim=1)
        xy_physics.requires_grad_(True)
    
    # Training Loop (Adam)
    pbar = tqdm(range(epochs), desc="Training PINN (Adam)")
    loss_history = TrainingHistory()
    
    # Configurazione Pesi Loss
    if loss_weights is None:
        loss_weights = {'data': 1.0, 'bc': 1.0, 'physics': 0.05}
    
    lambda_data = loss_weights.get('data', 1.0)
    lambda_bc = loss_weights.get('bc', 1.0)
    target_lambda_physics = loss_weights.get('physics', 0.05)
    
    # Configurazione Warmup
    if warmup_epochs is None:
        warmup_epochs = epochs // 3
    
    # Scheduler per il Learning Rate
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6000, gamma=0.4)

    for epoch in pbar:
        
        model.train()
        optimizer.zero_grad()
        
        # Gestione Warmup e Fisica
        if epoch < warmup_epochs:
            # Fase 1: Solo Dati (Interni + BC). Niente calcolo gradienti fisici.
            current_physics_fn = None
            current_physics_problem = None
            lambda_physics = 0.0
            phase_desc = "Warmup (Data Only)"
        else:
            # Fase 2: Dati + Fisica
            current_physics_fn = heat2d_physics_loss if physics_problem is None else None
            current_physics_problem = physics_problem
            lambda_physics = target_lambda_physics
            phase_desc = "Physics Refinement"

        # Calcolo loss
        loss, loss_dict = compute_pinn_loss(
            model, 
            x_data=xy_int, 
            y_data=T_int,
            x_bc=xy_bc,
            y_bc=T_bc,
            physics_loss_fn=current_physics_fn, 
            physics_problem=current_physics_problem,
            x_physics=xy_physics,
            lambda_data=lambda_data,
            lambda_bc=lambda_bc,
            lambda_physics=lambda_physics
        )
        
        # Gradient Logging Logic
        if log_gradients_every > 0 and (epoch + 1) % log_gradients_every == 0:
            grad_norms = {}
            for name, loss_tensor in loss_dict.items():
                if name == 'total_loss': continue
                if isinstance(loss_tensor, torch.Tensor):
                    # Get the weight used
                    weight = 1.0
                    if name == 'data_loss': weight = lambda_data
                    elif name == 'bc_loss': weight = lambda_bc
                    elif name == 'pde_loss': weight = lambda_physics
                    
                    if weight > 0:
                        # Retain graph needed because we do multiple backward calls (via grad)
                        # and then the final backward.
                        grads = torch.autograd.grad(loss_tensor * weight, model.parameters(), retain_graph=True, allow_unused=True)
                        
                        # Total L2 norm of all params
                        total_norm = 0.0
                        for g in grads:
                            if g is not None:
                                total_norm += g.data.norm(2).item()**2
                        total_norm = total_norm ** 0.5
                        grad_norms[f'grad_{name}'] = total_norm
            
            loss_history.update(epoch, grad_norms)

        loss.backward()
        optimizer.step()
        
        # Step dello scheduler
        scheduler.step()
        
        # Aggiornamento history
        loss_history.update(epoch, loss_dict)
        
        # Monitoraggio e Plotting periodico
        if (epoch + 1) % 500 == 0:
            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({
                'Phase': phase_desc,
                'Loss': f"{loss.item():.2e}", 
                'BC_L': f"{loss_dict.get('bc_loss', 0):.2e}",
                'LR': f"{current_lr:.1e}"
            })
            
            model.eval()
            with torch.no_grad():
                T_pred_grid = model(xy_grid).reshape(Nx_dom, Ny_dom)
                
            plot_path = os.path.join(plots_dir, f'epoch_{epoch+1}.png')
            plot2D_comparison(X, Y, T_exact_grid, T_pred_grid, epoch+1, plot_path, physics_points=xy_physics)
            plot_files.append(plot_path)

    # --- L-BFGS Optimization Phase ---
    print("\nInizio fase di raffinamento con L-BFGS...")
    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(), 
        lr=1.0, 
        max_iter=2000, 
        max_eval=2000, 
        tolerance_grad=1e-7, 
        tolerance_change=1e-9,
        history_size=100,
        line_search_fn="strong_wolfe"
    )

    # Closure function richiesta da L-BFGS
    def closure():
        optimizer_lbfgs.zero_grad()
        loss, loss_dict = compute_pinn_loss(
            model, 
            x_data=xy_int, 
            y_data=T_int,
            x_bc=xy_bc,
            y_bc=T_bc,
            physics_loss_fn=heat2d_physics_loss if physics_problem is None else None, 
            physics_problem=physics_problem,
            x_physics=xy_physics,
            lambda_data=lambda_data,
            lambda_bc=lambda_bc,
            lambda_physics=target_lambda_physics
        )
        loss.backward()
        return loss

    optimizer_lbfgs.step(closure)
    
    # Calcolo loss finale dopo L-BFGS per aggiornare history
    final_loss, final_loss_dict = compute_pinn_loss(
            model, 
            x_data=xy_int, 
            y_data=T_int,
            x_bc=xy_bc,
            y_bc=T_bc,
            physics_loss_fn=heat2d_physics_loss if physics_problem is None else None, 
            physics_problem=physics_problem,
            x_physics=xy_physics,
            lambda_data=lambda_data,
            lambda_bc=lambda_bc,
            lambda_physics=target_lambda_physics
    )
    loss_history.update(epochs + 1, final_loss_dict) 
    print(f"Loss finale dopo L-BFGS: {final_loss.item():.2e}")

    # Plot Finale Interattivo
    print("Training completato. Generazione plot finale...")
    model.eval()
    with torch.no_grad():
        T_final = model(xy_grid).reshape(Nx_dom, Ny_dom)
    
    # Salvataggio ultimo plot (Results)
    final_path = os.path.join(final_dir, 'PINNfinal_result.png')
    plot2D_comparison(X, Y, T_exact_grid, T_final, epochs, save_path=final_path, physics_points=xy_physics)
    
    # Generazione GIF
    print(f"Creazione GIF con {len(plot_files)} frames...")
    if plot_files:
        gif_path = os.path.join(final_dir, 'PINNtraining_evolution.gif')
        save_gif_PIL(gif_path, plot_files, fps=3, loop=1, delete_files=True)
    
    # Plot Loss History con linea verticale per fine warmup
    loss_history.plot_losses(last_adam_epoch=warmup_epochs, save_path=os.path.join(final_dir, 'PINNloss_history.png'), experiment_name="Heat2D PINN", show_plot=show_plots_interactively)
    
    # Plot Gradient History if available
    loss_history.plot_gradients(save_path=os.path.join(final_dir, 'PINN_gradients.png'), experiment_name="Heat2D PINN Gradients", show_plot=show_plots_interactively)
    
    if show_plots_interactively:
        plt.show()
    else:
        plt.close("all")

    return loss_history