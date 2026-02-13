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
    collocation_points=None,
    lr_strategy='fixed',
    dynamic_weighting=False,
    update_weights_every=100
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
        lr_strategy: Strategia di learning rate ('fixed' o 'step_decay').
        dynamic_weighting: (Bool) Se True, attiva il Learning Rate Annealing per i pesi.
        update_weights_every: (Int) Frequenza aggiornamento pesi dinamici (epoche).
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
    
    # Store dimensions for resampling
    if isinstance(n_collocation, int):
        Nx_phys, Ny_phys = n_collocation, n_collocation
    else:
        Nx_phys, Ny_phys = n_collocation
    
    # Training Loop (Adam)
    pbar = tqdm(range(epochs), desc=f"Training PINN (Adam) ({lr_strategy})")
    loss_history = TrainingHistory()
    
    # Configurazione Pesi Loss
    if loss_weights is None:
        loss_weights = {'data': 1.0, 'bc': 1.0, 'physics': 1.0}
    
    lambda_data = loss_weights.get('data', 1.0)
    lambda_bc = loss_weights.get('bc', 1.0)
    target_lambda_physics = loss_weights.get('physics', 1.0)
    
    # Configurazione Warmup
    if warmup_epochs is None:
        warmup_epochs = epochs // 3
    
    # Scheduler per il Learning Rate
    scheduler = None
    if lr_strategy == 'step_decay':
        step_size = int(epochs * 0.25)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)
    elif lr_strategy == 'plateau': # O sostituisci nel tuo if
    # factor=0.5: Dimezza il LR quando si blocca (come facevi prima)
    # patience=1000: Aspetta 1000 epoche senza miglioramenti prima di tagliare.
    #   Dato che il tuo grafico è molto rumoroso, serve una pazienza alta per non
    #   scattare per sbaglio su un picco di rumore.
    # verbose=True: Ti stampa a video quando cambia il LR (fondamentale per debug)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=500,
            min_lr=1e-6
        )
    # Pre-generate Collocation Points (Fixed across epochs)
    if collocation_points is not None:
        xy_physics = collocation_points.clone()
        if not xy_physics.requires_grad:
            xy_physics.requires_grad_(True)
    else:
        # Random points in [0, Lx] x [0, Ly], fixed for the whole training
        xy_physics = torch.rand((Nx_phys * Ny_phys, 2), device=device)
        xy_physics[:, 0] = xy_physics[:, 0] * Lx
        xy_physics[:, 1] = xy_physics[:, 1] * Ly
        xy_physics.requires_grad_(True)

    # Dynamic Weighting Variables
    alpha_dynamic = 0.9
    
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
        
        # --- Dynamic Weighting Logic (LR Annealing) ---
        if dynamic_weighting and epoch >= warmup_epochs and (epoch + 1) % update_weights_every == 0:
            # We use the MAX gradient norm heuristic (Wang et al.)
            # Target: Gradient norms of Physics and Data should match Gradient norm of BC
            
            # 1. BC Gradient (Reference)
            max_norm_bc = 0.0
            if 'bc_loss' in loss_dict:
                grads_bc = torch.autograd.grad(loss_dict['bc_loss'], model.parameters(), retain_graph=True, allow_unused=True)
                norms_bc = [g.norm(2) for g in grads_bc if g is not None]
                if norms_bc:
                    max_norm_bc = max(norms_bc).item()
            
            # 2. Physics Gradient
            if 'pde_loss' in loss_dict and lambda_physics > 0:
                grads_phys = torch.autograd.grad(loss_dict['pde_loss'], model.parameters(), retain_graph=True, allow_unused=True)
                norms_phys = [g.norm(2) for g in grads_phys if g is not None]
                if norms_phys:
                    max_norm_phys = max(norms_phys).item()
                    if max_norm_phys > 1e-12:
                        # Ratio of gradients * current lambda_bc
                        new_lambda_phys = (max_norm_bc / max_norm_phys) * lambda_bc
                        # Update with moving average
                        target_lambda_physics = alpha_dynamic * target_lambda_physics + (1 - alpha_dynamic) * new_lambda_phys

            # 3. Data Gradient
            if 'data_loss' in loss_dict and lambda_data > 0:
                 grads_data = torch.autograd.grad(loss_dict['data_loss'], model.parameters(), retain_graph=True, allow_unused=True)
                 norms_data = [g.norm(2) for g in grads_data if g is not None]
                 if norms_data:
                     max_norm_data = max(norms_data).item()
                     if max_norm_data > 1e-12:
                         new_lambda_data = (max_norm_bc / max_norm_data) * lambda_bc
                         lambda_data = alpha_dynamic * lambda_data + (1 - alpha_dynamic) * new_lambda_data

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
        # AGGIUNGI QUESTO: Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Step dello scheduler
        if lr_strategy == 'step_decay':
            scheduler.step()
        elif lr_strategy == 'plateau':
            scheduler.step(loss)
            
        # Aggiornamento history
        loss_dict.update({
            'weight_data': lambda_data,
            'weight_bc': lambda_bc,
            'weight_phys': lambda_physics
        })
        loss_history.update(epoch, loss_dict)
        
        # Monitoraggio e Plotting periodico
        if (epoch + 1) % 500 == 0:
            current_lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
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
        max_iter=5000, 
        max_eval=5000, 
        tolerance_grad=1e-7, 
        tolerance_change=1e-9,
        history_size=100,
        line_search_fn="strong_wolfe"
    )

    lbfgs_iter = [0]
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
        
        # Log every 10 iterations to keep the history manageable but detailed
        if lbfgs_iter[0] % 10 == 0:
            loss_history.update(epochs + lbfgs_iter[0], loss_dict)
        
        lbfgs_iter[0] += 1
        return loss

    optimizer_lbfgs.step(closure)
    
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
    loss_history.update(epochs + lbfgs_iter[0], final_loss_dict) 
    print(f"Loss finale dopo L-BFGS (iter {lbfgs_iter[0]}): {final_loss.item():.2e}")

    # Plot Finale Interattivo
    print("Training completato. Generazione plot finale...")
    model.eval()
    with torch.no_grad():
        T_final = model(xy_grid).reshape(Nx_dom, Ny_dom)
    
    # Concatenate data points for visualization based on weights
    # If lambda_data is 0 (Pure Physics), we only show boundary points if lambda_bc > 0
    lambda_data_viz = loss_weights.get('data', 1.0)
    lambda_bc_viz = loss_weights.get('bc', 1.0)
    viz_data_points = []
    if lambda_data_viz > 0:
        viz_data_points.append(xy_int)
    if lambda_bc_viz > 0:
        viz_data_points.append(xy_bc)
    
    xy_data_points = torch.cat(viz_data_points, dim=0) if viz_data_points else None

    # Salvataggio ultimo plot (Results)
    final_path = os.path.join(final_dir, 'PINNfinal_result.png')
    plot2D_final_result(X, Y, T_exact_grid, T_final, epochs, save_path=final_path, data_points=xy_data_points, physics_points=xy_physics)
    
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
        experiment_name="Heat2D PINN", 
        show_plot=show_plots_interactively
    )
    
    # Plot Gradient History if available
    loss_history.plot_gradients(save_path=os.path.join(final_dir, 'PINN_gradients.png'), experiment_name="Heat2D PINN Gradients", show_plot=show_plots_interactively)
    
    # Plot Weight History if available
    loss_history.plot_weights(save_path=os.path.join(final_dir, 'PINN_weights.png'), experiment_name="Heat2D PINN Weights", show_plot=show_plots_interactively)

    if show_plots_interactively:
        plt.show()
    else:
        plt.close("all")

    return loss_history