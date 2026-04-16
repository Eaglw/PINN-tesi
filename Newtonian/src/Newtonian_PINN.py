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

# ---  DEFINIZIONE DELLA LOSS FISICA ---
def heat2d_physics_loss(model, xy_p):
    """
    Calcola il residuo dell'equazione di Laplace 2D: d2T/dx2 + d2T/dy2 = 0
    """
    T = model(xy_p)
    grads = torch.autograd.grad(T, xy_p, torch.ones_like(T), create_graph=True)[0]
    dT_dx, dT_dy = grads[:, 0], grads[:, 1]
    grads2_x = torch.autograd.grad(dT_dx, xy_p, torch.ones_like(dT_dx), create_graph=True, allow_unused=True)[0]
    d2T_dx2 = grads2_x[:, 0] if grads2_x is not None else torch.zeros_like(dT_dx)
    grads2_y = torch.autograd.grad(dT_dy, xy_p, torch.ones_like(dT_dy), create_graph=True, allow_unused=True)[0]
    d2T_dy2 = grads2_y[:, 1] if grads2_y is not None else torch.zeros_like(dT_dy)
    res = d2T_dx2 + d2T_dy2
    return torch.mean(res**2)

def train_modelPINN(
    model, optimizer, data_internal, data_boundary, validation_grid,
    epochs=20000, physics_problem=None, plots_dir='plots', final_dir='Heat2D/Results',
    show_plots_interactively=True, log_gradients_every=0, loss_weights=None,
    warmup_epochs=None, n_collocation=(50, 50), collocation_points=None,
    lr_strategy='fixed', dynamic_weighting=False, update_weights_every=100,
    max_total_lbfgs=100, resample_every=0, resample_fn=None,
    experiment_name="PINN Training", val_label="Value"
):
    """
    Esegue il training della PINN.
    
    Args:
        ... (altri args rimangono invariati)
        resample_every: (Int) Se > 0, ricampiona i punti di collocazione ogni N epoche.
        resample_fn: (Callable) Funzione che restituisce nuovi punti di collocazione.
    """
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
    
    pbar = tqdm(range(epochs), desc=f"Training PINN (Adam) ({lr_strategy})")
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
        xy_physics = collocation_points.clone()
        if not xy_physics.requires_grad: xy_physics.requires_grad_(True)
    else:
        xy_physics = torch.rand((n_collocation[0]*n_collocation[1], 2), device=device)
        xy_physics[:, 0], xy_physics[:, 1] = xy_physics[:, 0] * Lx, xy_physics[:, 1] * Ly
        xy_physics.requires_grad_(True)

    alpha_dynamic = 0.9
    for epoch in pbar:
        # Periodic Resampling
        if resample_every > 0 and resample_fn is not None and epoch > 0 and epoch % resample_every == 0:
            xy_physics = resample_fn().clone().detach()
            xy_physics.requires_grad_(True)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        # Gestione Warmup con solo dati
        if epoch < warmup_epochs:
            current_physics_fn, current_physics_problem, lambda_physics, phase_desc = None, None, 0.0, "Warmup"
        else:
            current_physics_fn, current_physics_problem, lambda_physics, phase_desc = (heat2d_physics_loss if physics_problem is None else None), physics_problem, target_lambda_physics, "Physics"

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
        # Dynamic Weighting
        if dynamic_weighting and epoch >= warmup_epochs and (epoch + 1) % update_weights_every == 0:
            pure_bc = physics_problem.boundary_loss(model, xy_bc, T_bc) if physics_problem else nn.MSELoss()(model(xy_bc), T_bc)
            grads_bc = torch.autograd.grad(pure_bc, model.parameters(), retain_graph=True, allow_unused=True)
            max_norm_bc = max([g.norm(2) for g in grads_bc if g is not None]).item() if any(g is not None for g in grads_bc) else 0.0
            
            if lambda_physics > 0:
                pure_phys = physics_problem.residual(model, xy_physics) if physics_problem else heat2d_physics_loss(model, xy_physics)
                grads_ph = torch.autograd.grad(pure_phys, model.parameters(), retain_graph=True, allow_unused=True)
                m_n_ph = max([g.norm(2) for g in grads_ph if g is not None]).item() if any(g is not None for g in grads_ph) else 0.0
                if m_n_ph > 1e-12: target_lambda_physics = alpha_dynamic * target_lambda_physics + (1-alpha_dynamic) * (max_norm_bc/m_n_ph)*lambda_bc

            if lambda_data > 0:
                pure_data = nn.MSELoss()(model(xy_int), T_int)
                grads_dt = torch.autograd.grad(pure_data, model.parameters(), retain_graph=True, allow_unused=True)
                m_n_dt = max([g.norm(2) for g in grads_dt if g is not None]).item() if any(g is not None for g in grads_dt) else 0.0
                if m_n_dt > 1e-12: lambda_data = alpha_dynamic * lambda_data + (1-alpha_dynamic) * (max_norm_bc/m_n_dt)*lambda_bc
        
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
        
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        if lr_strategy == 'step_decay': scheduler.step()
        elif lr_strategy == 'plateau':
            # Use unweighted loss for stable scheduling, monitoring only active components (weight > 0)
            monitored_loss = 0.0
            if lambda_data > 0: monitored_loss += loss_dict.get('data_loss', 0.0)
            if lambda_bc > 0: monitored_loss += loss_dict.get('bc_loss', 0.0)
            if lambda_physics > 0: monitored_loss += loss_dict.get('pde_loss', 0.0)
            scheduler.step(monitored_loss)
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
                if not xy_grid.requires_grad: xy_grid.requires_grad_(True)
                # Ricaviamo u dal problema fisico (Stream Function)
                if hasattr(physics_problem, 'get_velocity'):
                    u_pred, _, _ = physics_problem.get_velocity(model, xy_grid)
                    T_pred_grid = u_pred.detach().cpu().reshape(Ny_dom, Nx_dom)
                else:
                    # Fallback per 3-output o altri casi
                    T_pred_grid = model(xy_grid)[:, 0].detach().cpu().reshape(Ny_dom, Nx_dom)
            plot_path = os.path.join(plots_dir, f'epoch_{epoch+1}.png')
            plot2D_comparison(X, Y, T_exact_grid, T_pred_grid, epoch+1, plot_path, physics_points=xy_physics, val_label=val_label)
            plot_files.append(plot_path)

    # --- STAGED PRECISION SWITCH ---
    # Prima di iniziare L-BFGS, passiamo a FP64 (Float64) per la massima precisione scientifica
    print("\n--- Switching to FP64 for L-BFGS Refinement ---")
    torch.set_default_dtype(torch.float64)
    torch.backends.cuda.matmul.allow_tf32 = False # Disabilitato per FP64
    model.double()
    xy_int, T_int = xy_int.double(), T_int.double()
    xy_bc, T_bc = xy_bc.double(), T_bc.double()
    xy_physics = xy_physics.double()
    xy_grid, T_exact_grid = xy_grid.double(), T_exact_grid.double()
    X, Y = X.double(), Y.double()

    lbfgs_iter = [0]
    pbar_lbfgs = tqdm(total=max_total_lbfgs, desc="Training PINN (L-BFGS)")
    
    for current_lr in [1.0, 0.5]:
        start_iter_call = lbfgs_iter[0]
        remaining_evals = max_total_lbfgs - start_iter_call
        if remaining_evals <= 0:
            break
            
        optimizer_lbfgs = torch.optim.LBFGS(
            model.parameters(), 
            lr=current_lr, 
            max_iter=remaining_evals, 
            max_eval=remaining_evals, 
            tolerance_grad=1e-7, 
            tolerance_change=1e-9,
            history_size=300,
            line_search_fn="strong_wolfe"
        )
        
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
            if lbfgs_iter[0] % 10 == 0: 
                history_entry = loss_dict.copy()
                history_entry.update({'weight_data': lambda_data, 'weight_bc': lambda_bc, 'weight_phys': target_lambda_physics})
                loss_history.update(epochs + lbfgs_iter[0], history_entry, lr=current_lr)
            
            lbfgs_iter[0] += 1
            pbar_lbfgs.update(1)
            pbar_lbfgs.set_postfix({'Loss': f"{loss.item():.2e}"})
            return loss
            
        optimizer_lbfgs.step(closure)
        
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
            physics_loss_fn=heat2d_physics_loss if physics_problem is None else None, 
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
        if not xy_grid.requires_grad: xy_grid.requires_grad_(True)
        if hasattr(physics_problem, 'get_velocity'):
            u_p, _, _ = physics_problem.get_velocity(model, xy_grid)
            T_final = u_p.detach().cpu().reshape(Ny_dom, Nx_dom)
        else:
            T_final = model(xy_grid)[:, 0].detach().cpu().reshape(Ny_dom, Nx_dom)
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