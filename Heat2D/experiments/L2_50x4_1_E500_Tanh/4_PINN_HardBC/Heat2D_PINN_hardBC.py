"""
Experiment Goal: 4_PINN_HardBC
Description: PINN HardBC. Config: L2_50x4_1_E500_Tanh
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
from func.graphic_func import save_gif_PIL, plot2D_comparison, plot2D_final_result
from func.history_tracker import TrainingHistory, compute_pinn_loss


# Configurazione dispositivo e precisione
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

class HardBCWrapper(nn.Module):
    """
    Wrapper che applica Hard Boundary Conditions (BC) all'output della rete.
    Impone T=0 su x=0, y=0, y=Ly.
    Impone T~1 su x=Lx (approssimazione tramite serie di Fourier).
    
    Ansatz:
    T(x,y) = T_boundary(x,y) + x * y * (Ly - y) * NN(x,y)
    
    Dove T_boundary è costruito per essere 0 su x=0, y=0, y=Ly e ~1 su x=Lx.
    Usiamo la serie: Sum_{n odd} (4/(n*pi)) * (sinh(n*pi*x/Ly)/sinh(n*pi*Lx/Ly)) * sin(n*pi*y/Ly)
    che è la soluzione analitica troncata, soddisfacente esattamente le BC sui 3 lati e approssimante 1 sul quarto.
    """
    def __init__(self, model, Lx, Ly, n_terms=10):
        super().__init__()
        self.model = model
        self.Lx = Lx
        self.Ly = Ly
        self.n_terms = n_terms
        
    def forward(self, xy):
        x = xy[:, 0:1]
        y = xy[:, 1:2]
        
        # 1. Calcolo Termine di Bordo (Analitico Troncato)
        T_boundary = torch.zeros_like(x)
        const_pi = np.pi
        
        for n in range(1, self.n_terms * 2, 2): # n dispari: 1, 3, 5...
            lambda_n = n * const_pi / self.Ly
            An = 4 / (n * const_pi)
            # Termine sinh(lambda * x) / sinh(lambda * Lx)
            # Per stabilità numerica con x grandi, usiamo exp
            # Ma qui x è limitato, sinh va bene se Lx non è enorme.
            
            # Nota: sinh(a)/sinh(b) ~ exp(a-b). 
            # Se lambda_n * Lx è grande, sinh overflow. 
            # Implementazione safe:
            arg_x = lambda_n * x
            arg_L = torch.tensor(lambda_n * self.Lx, device=x.device, dtype=x.dtype)
            
            # term_x = torch.sinh(arg_x) / torch.sinh(arg_L) 
            term_x = torch.sinh(arg_x) / torch.sinh(arg_L)
            
            term = An * term_x * torch.sin(lambda_n * y)
            T_boundary = T_boundary + term
            
        # 2. Calcolo Termine Correttivo (NN con maschera)
        # Maschera che si annulla su x=0, y=0, y=Ly.
        # Su x=Lx vale Lx * y * (Ly-y). 
        # La NN imparerà a correggere l'approssimazione di Fourier su x=Lx (e nell'interno).
        mask = x * y * (self.Ly - y)
        
        T_nn = self.model(xy)
        
        return T_boundary + mask * T_nn

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
    Esegue il training della PINN con Hard BC.
    """
    
    # Unpack dei dati
    xy_int, T_int = data_internal
    xy_bc, T_bc = data_boundary
    xy_grid, T_exact_grid, X, Y = validation_grid
    
    # Ricavo dimensioni griglia per reshape e limiti dominio
    Nx_dom, Ny_dom = X.shape
    Lx = X.max().item()
    Ly = Y.max().item()
    
    # --- WRAP MODEL FOR HARD BC ---
    # Avvolgiamo il modello originale per imporre le BC.
    # L'optimizer continuerà ad aggiornare i pesi di 'model' (passato per riferimento).
    wrapped_model = HardBCWrapper(model, Lx, Ly, n_terms=10).to(device)
    
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
    pbar = tqdm(range(epochs), desc="Training PINN HardBC")
    loss_history = TrainingHistory()
    
    # Configurazione Pesi Loss
    # FORZIAMO BC LOSS A 0 PERCHE' SONO HARDCODED
    if loss_weights is None:
        loss_weights = {'data': 1.0, 'bc': 0.0, 'physics': 1.0}
    else:
        loss_weights['bc'] = 0.0 # Override
    
    lambda_data = loss_weights.get('data', 1.0)
    lambda_bc = 0.0
    target_lambda_physics = loss_weights.get('physics', 1.0)
    
    # Configurazione Warmup
    if warmup_epochs is None:
        warmup_epochs = epochs // 3
    
    # Scheduler per il Learning Rate
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6000, gamma=0.4)

    for epoch in pbar:
        
        # NOTA: model.train() agisce sui pesi interni. wrapped_model propaga la chiamata.
        wrapped_model.train()
        optimizer.zero_grad()
        
        # Gestione Warmup e Fisica
        if epoch < warmup_epochs:
            # Fase 1: Solo Dati Interni (se presenti). Niente BC Loss (è hard).
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

        # Calcolo loss usando WRAPPED_MODEL
        # Passiamo x_bc=None, y_bc=None per evitare calcolo inutile, tanto lambda_bc=0
        loss, loss_dict = compute_pinn_loss(
            wrapped_model, 
            x_data=xy_int, 
            y_data=T_int,
            x_bc=None, # Disabilitiamo BC loss esplicitamente
            y_bc=None,
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
                        grads = torch.autograd.grad(loss_tensor * weight, model.parameters(), retain_graph=True, allow_unused=True)
                        total_norm = 0.0
                        for g in grads:
                            if g is not None:
                                total_norm += g.data.norm(2).item()**2
                        total_norm = total_norm ** 0.5
                        grad_norms[f'grad_{name}'] = total_norm
            loss_history.update(epoch, grad_norms)

        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_history.update(epoch, loss_dict)
        
        # Monitoraggio e Plotting periodico
        if (epoch + 1) % 500 == 0:
            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({
                'Phase': phase_desc,
                'Loss': f"{loss.item():.2e}", 
                'LR': f"{current_lr:.1e}"
            })
            
            wrapped_model.eval()
            with torch.no_grad():
                T_pred_grid = wrapped_model(xy_grid).reshape(Nx_dom, Ny_dom)
                
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

    def closure():
        optimizer_lbfgs.zero_grad()
        loss, loss_dict = compute_pinn_loss(
            wrapped_model, 
            x_data=xy_int, 
            y_data=T_int,
            x_bc=None,
            y_bc=None,
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
    
    final_loss, final_loss_dict = compute_pinn_loss(
            wrapped_model, 
            x_data=xy_int, 
            y_data=T_int,
            x_bc=None,
            y_bc=None,
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
    wrapped_model.eval()
    with torch.no_grad():
        T_final = wrapped_model(xy_grid).reshape(Nx_dom, Ny_dom)
    
    # Concatenate data points for visualization based on weights
    # If lambda_data is 0 (Pure Physics), we only show boundary points if lambda_bc > 0
    lambda_data_viz = loss_weights.get('data', 1.0)
    lambda_bc_viz = loss_weights.get('bc', 0.0) # In HardBC, bc weight is 0
    viz_data_points = []
    if lambda_data_viz > 0:
        viz_data_points.append(xy_int)
    if lambda_bc_viz > 0:
        viz_data_points.append(xy_bc)
    
    xy_data_points = torch.cat(viz_data_points, dim=0) if viz_data_points else None

    final_path = os.path.join(final_dir, 'PINNfinal_result.png')
    plot2D_final_result(X, Y, T_exact_grid, T_final, epochs, save_path=final_path, data_points=xy_data_points, physics_points=xy_physics)
    
    print(f"Creazione GIF con {len(plot_files)} frames...")
    if plot_files:
        gif_path = os.path.join(final_dir, 'PINNtraining_evolution.gif')
        save_gif_PIL(gif_path, plot_files, fps=3, loop=1, delete_files=True)
    
    loss_history.plot_losses(last_adam_epoch=warmup_epochs, save_path=os.path.join(final_dir, 'PINNloss_history.png'), experiment_name="Heat2D PINN HardBC", show_plot=show_plots_interactively)
    loss_history.plot_gradients(save_path=os.path.join(final_dir, 'PINN_gradients.png'), experiment_name="Heat2D PINN HardBC Gradients", show_plot=show_plots_interactively)
    
    if show_plots_interactively:
        plt.show()
    else:
        plt.close("all")

    return wrapped_model, loss_history