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