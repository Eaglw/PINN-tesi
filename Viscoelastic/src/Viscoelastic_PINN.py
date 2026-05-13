import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from dataclasses import dataclass, field
from tqdm import tqdm

# Import function for GIF and loss comparison
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from func.graphic_func import save_gif_PIL, plot2D_comparison, plot2D_final_result, plot2D_viscoelastic_final
from func.history_tracker import TrainingHistory, compute_pinn_loss

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


@dataclass
class TrainingConfig:
    """Configurazione centralizzata per il training della PINN viscoelastica."""
    # --- Optimizer ---
    epochs: int = 1000
    base_lr: float = 1e-3
    adam_eps: float = 1e-7
    lr_strategy: str = 'cosine'
    # --- Staged Training ---
    staged_training: bool = True
    # --- Precision ---
    precision_mode: str = 'staged'
    # --- L-BFGS ---
    max_lbfgs_iters: int = 100
    # --- Gradient ---
    grad_clip_norm: float = 5.0
    # --- Mini-Batching ---
    minibatch_internal: int = 1024
    minibatch_boundary: int = 256
    # --- Loss Weighting ---
    dynamic_weighting: bool = True
    update_weights_every: int = 100
    loss_weights: dict = field(default_factory=lambda: {'data': 1.0, 'bc': 1.0, 'physics': 1.0})
    mode: str = 'standard'  # 'standard' | 'semi_inverse'
    variance_weights: dict = None
    # --- Logging & Plotting ---
    log_gradients_every: int = 500
    plot_every: int = 500
    experiment_name: str = "PINN Training"
    val_label: str = "Value"


def _sample_minibatch(xy, targets, batch_size, device):
    """Campiona un mini-batch casuale da un dataset."""
    if batch_size is None or batch_size >= xy.shape[0]:
        return xy, targets
    idx = torch.randperm(xy.shape[0], device=device)[:batch_size]
    return xy[idx], targets[idx]


def _get_scheduler(optimizer, strategy, total_steps):
    """Crea lo scheduler LR in base alla strategia scelta."""
    if strategy == 'step_decay':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(total_steps * 0.25), gamma=0.5)
    elif strategy == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=600, min_lr=1e-6, cooldown=3000)
    elif strategy == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    return None


def initialize_last_layer_zero(model):
    """
    Inizializza l'ultimo layer della rete a zero (pesi e bias).
    Utile per far partire la rete con output nullo (es. stress).
    """
    last_layer = list(model.fcs)[-1]
    nn.init.zeros_(last_layer.weight)
    nn.init.zeros_(last_layer.bias)
    print(f"  [Init] Ultimo layer di {model.__class__.__name__} inizializzato a zero.")

def train_ViscoelasticPINN(
    model, config, data_internal, data_boundary,
    validation_grid, physics_problem, collocation_points,
    plots_dir, final_dir, stress_exact_grids=None
):
    """
    Esegue il training della PINN viscoelastica.
    """
    # --- Inizializzazione Stress a Zero ---
    # Questo evita che il rumore iniziale di Tau disturbi la cinematica nella Fase 1
    initialize_last_layer_zero(model.model_tau)
    # --- Unpack config ---
    cfg = config
    epochs = cfg.epochs
    base_lr = cfg.base_lr
    lr_strategy = cfg.lr_strategy
    staged_training = cfg.staged_training
    mode = cfg.mode
    variance_weights = cfg.variance_weights
    
    # --- SETUP STAGED TRAINING ---
    half_epochs = epochs // 2
    if staged_training:
        print(f"\n  [Staged Training] Fase 1: Cinematica (psi+p) per {half_epochs} epoche. (Tau esplicitamente congelato)")
        set_model_trainable(model, ['psi', 'p'])
        for param in model.model_tau.parameters():
            param.requires_grad_(False)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=base_lr, eps=cfg.adam_eps)
    else:
        set_model_trainable(model, ['psi', 'p', 'tau'])
        trainable_params = list(model.parameters())
        optimizer = torch.optim.Adam(trainable_params, lr=base_lr, eps=cfg.adam_eps)

    xy_int, T_int = data_internal
    xy_bc, T_bc = data_boundary
    xy_grid, T_exact_grid, X, Y = validation_grid
    # Spostiamo tutto su CPU per il plotting con matplotlib
    X, Y = X.cpu(), Y.cpu()
    T_exact_grid = T_exact_grid.cpu()
    Ny_dom, Nx_dom = X.shape

    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)
    plot_files = []
    
    pbar = tqdm(range(epochs), desc=f"Training PINN (Adam) ({lr_strategy})", mininterval=2.0)
    loss_history = TrainingHistory()
    
    loss_weights = cfg.loss_weights
    lambda_data = loss_weights.get('data', 1.0)
    lambda_bc = loss_weights.get('bc', 1.0)
    target_lambda_physics = loss_weights.get('physics', 1.0)
    
    # Determina quali loss sono attive per il plotting
    _active_keys = set()
    if loss_weights.get('data', 0) > 0: _active_keys.add('data')
    if loss_weights.get('bc', 0) > 0: _active_keys.add('bc')
    if loss_weights.get('physics', 0) > 0: _active_keys.add('physics')
    
    phase1_steps = half_epochs if staged_training else epochs
    scheduler = _get_scheduler(optimizer, lr_strategy, phase1_steps)

    # Cache dtype/device — non cambiano durante la fase Adam
    _dtype = next(model.parameters()).dtype
    _device = next(model.parameters()).device
    xy_physics = collocation_points.clone().to(dtype=_dtype, device=_device)
    if not xy_physics.requires_grad: xy_physics.requires_grad_(True)

    alpha_dynamic = 0.9
    for epoch in pbar:
        # --- STAGED TRAINING: Cambio fase a metà epoche ---
        if staged_training and epoch == half_epochs:
            print(f"\n  [Staged Training] Fase 2: Costitutivo (tau) + Cinematica (psi) per {epochs - half_epochs} epoche")
            set_model_trainable(model, ['tau', 'psi'])
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(trainable_params, lr=base_lr, eps=cfg.adam_eps)
            scheduler = _get_scheduler(optimizer, lr_strategy, epochs - half_epochs)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        lambda_physics = target_lambda_physics

        # Campionamento dati interni
        if lambda_data > 0:
            xy_int_batch, T_int_batch = _sample_minibatch(xy_int, T_int, cfg.minibatch_internal, _device)
        else:
            xy_int_batch, T_int_batch = xy_int, T_int

        # Campionamento punti fisica
        if lambda_physics > 0:
            if lambda_data > 0:
                # Goal 1: Fisica sugli stessi punti dei dati
                xy_phys_batch = xy_int_batch.clone().to(dtype=_dtype, device=_device).requires_grad_(True)
            else:
                # Goal 0: Fisica su collocation points separati
                xy_phys_batch, _ = _sample_minibatch(collocation_points, collocation_points, cfg.minibatch_internal, _device)
                xy_phys_batch = xy_phys_batch.clone().to(dtype=_dtype, device=_device).requires_grad_(True)
        else:
            xy_phys_batch = None

        # Campionamento boundary
        xy_bc_batch, T_bc_batch = _sample_minibatch(xy_bc, T_bc, cfg.minibatch_boundary, _device)

        # Calcolo loss
        loss, loss_dict = compute_pinn_loss(
            model, 
            x_data=xy_int_batch, 
            y_data=T_int_batch,
            x_bc=xy_bc_batch,
            y_bc=T_bc_batch,
            physics_problem=physics_problem,
            x_physics=xy_phys_batch,
            lambda_data=lambda_data,
            lambda_bc=lambda_bc,
            lambda_physics=lambda_physics,
            mode=mode,
            variance_weights=variance_weights
        )

        # Controllo NaN
        if torch.isnan(loss):
            print(f"\n!!! [NaN] Rilevato NaN all'epoca {epoch} !!!")
            for k, v in loss_dict.items():
                print(f"  - {k}: {v.item() if hasattr(v, 'item') else v}")
            sys.exit(1)
            
        # Dynamic Weighting (Learning Rate Annealing style)
        if cfg.dynamic_weighting and (epoch + 1) % cfg.update_weights_every == 0:
            pure_bc = physics_problem.boundary_loss(model, xy_bc, T_bc)
            grads_bc = torch.autograd.grad(pure_bc, trainable_params, retain_graph=True, allow_unused=True)
            max_norm_bc = max([g.norm(2) for g in grads_bc if g is not None]).item() if any(g is not None for g in grads_bc) else 0.0
            
            if lambda_bc > 0:
                if lambda_physics > 0 and xy_phys_batch is not None:
                    pure_phys = physics_problem.residual(model, xy_phys_batch)
                    grads_ph = torch.autograd.grad(pure_phys, trainable_params, retain_graph=True, allow_unused=True)
                    m_n_ph = max([g.norm(2) for g in grads_ph if g is not None]).item() if any(g is not None for g in grads_ph) else 0.0
                    if m_n_ph > 1e-12: 
                        ratio = min(max_norm_bc / m_n_ph, 100.0)
                        target_lambda_physics = alpha_dynamic * target_lambda_physics + (1-alpha_dynamic) * ratio * lambda_bc

                if lambda_data > 0:
                    if mode == 'semi_inverse' and physics_problem is not None:
                        u_p, v_p, _, _ = physics_problem.get_velocity(model, xy_int)
                        s_u = max(variance_weights.get('u', 1.0), 1e-8) if variance_weights else 1.0
                        s_v = max(variance_weights.get('v', 1.0), 1e-8) if variance_weights else 1.0
                        pure_data = 0.5 * (nn.MSELoss()(u_p, T_int[:, 0:1])/s_u + nn.MSELoss()(v_p, T_int[:, 1:2])/s_v)
                    else:
                        pure_data = nn.MSELoss()(model(xy_int), T_int)
                        
                    grads_dt = torch.autograd.grad(pure_data, trainable_params, retain_graph=True, allow_unused=True)
                    m_n_dt = max([g.norm(2) for g in grads_dt if g is not None]).item() if any(g is not None for g in grads_dt) else 0.0
                    if m_n_dt > 1e-12: 
                        ratio_d = min(max_norm_bc / m_n_dt, 100.0)
                        lambda_data = alpha_dynamic * lambda_data + (1-alpha_dynamic) * ratio_d * lambda_bc
        
        # Logging
        current_lr = optimizer.param_groups[0]['lr']
        history_entry = loss_dict.copy()
        history_entry.update({'weight_data': lambda_data, 'weight_bc': lambda_bc, 'weight_phys': lambda_physics})

        if cfg.log_gradients_every > 0 and (epoch + 1) % cfg.log_gradients_every == 0:
            grad_norms = {}
            for name, l_val in loss_dict.items():
                if name == 'total_loss': continue
                w = lambda_data if name == 'data_loss' else (lambda_bc if name == 'bc_loss' else (lambda_physics if name == 'pde_loss' else 1.0))
                grads = torch.autograd.grad(l_val * w, trainable_params, retain_graph=True, allow_unused=True)
                grad_norms[f'grad_{name}'] = sum(g.data.norm(2).item()**2 for g in grads if g is not None)**0.5
            history_entry.update(grad_norms)

        loss_history.update(epoch, history_entry, lr=current_lr)
        loss.backward()
        
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip_norm)
        
        optimizer.step()
        
        if lr_strategy in ['step_decay', 'cosine']: scheduler.step()
        elif lr_strategy == 'plateau':
            active_losses = [
                loss_dict[k] for k in ['data_loss', 'bc_loss', 'pde_loss']
                if loss_dict.get(k) is not None and isinstance(loss_dict[k], torch.Tensor)
            ]
            monitored_loss = sum(active_losses) if active_losses else torch.tensor(0.0)
            scheduler.step(monitored_loss.item())

        # Monitoraggio e Plotting periodico
        if (epoch + 1) % 100 == 0:
            pbar.set_postfix({
                'Loss': f"{loss.item():.2e}", 
                'BC_L': f"{loss_dict.get('bc_loss', 0):.2e}",
                'LR': f"{current_lr:.1e}"
            })            
            model.eval()
            if (epoch + 1) % cfg.plot_every == 0:
                with torch.set_grad_enabled(True): 
                    xy_grid_val = xy_grid.clone().detach().requires_grad_(True)
                    if hasattr(physics_problem, 'get_velocity'):
                        u_pred, _, _, _ = physics_problem.get_velocity(model, xy_grid_val)
                        T_pred_grid = u_pred.detach().cpu().reshape(Ny_dom, Nx_dom)
                        out = model(xy_grid_val)
                        stress_preds = {
                            'tau_xx': out[:, 2].detach().cpu().reshape(Ny_dom, Nx_dom),
                            'tau_xy': out[:, 3].detach().cpu().reshape(Ny_dom, Nx_dom),
                            'tau_yy': out[:, 4].detach().cpu().reshape(Ny_dom, Nx_dom),
                        }
                    else:
                        T_pred_grid = model(xy_grid_val)[:, 0].detach().cpu().reshape(Ny_dom, Nx_dom)
                        stress_preds = {}
                    del xy_grid_val
                    
                plot_path = os.path.join(plots_dir, f'epoch_{epoch+1}.png')
                plot2D_comparison(X, Y, T_exact_grid, T_pred_grid, epoch+1, plot_path, physics_points=None, val_label=cfg.val_label, show_points=False)
                plot_files.append(plot_path)
                
                # Plot stress fields in loop
                for sname, spred in stress_preds.items():
                    exact_g = stress_exact_grids.get(sname, torch.zeros_like(T_exact_grid)).cpu() if stress_exact_grids else torch.zeros_like(T_exact_grid)
                    plot2D_comparison(X, Y, exact_g, spred, epoch+1, os.path.join(plots_dir, f'{sname}_{epoch+1}.png'), physics_points=None, val_label=sname, show_points=False)
                
                loss_history.plot_losses(
                    save_path=os.path.join(final_dir, 'PINN_loss_history.png'),
                    experiment_name=cfg.experiment_name,
                    smoothing_alpha=0.95,
                    active_loss_keys=_active_keys
                )


    # --- SBLOCCO TOTALE + PRECISION SWITCH PER L-BFGS ---
    if staged_training:
        print(f"\n  [Staged Training] Fase 3: Raffinamento L-BFGS (tutto sbloccato)")
        set_model_trainable(model, ['psi', 'p', 'tau'])
    
    # Ripristino Full-Batch per L-BFGS
    if lambda_data > 0 and lambda_physics > 0:
        xy_physics_full = xy_int.clone()
    elif lambda_data == 0 and lambda_physics > 0 and collocation_points is not None:
        xy_physics_full = collocation_points.clone()
    else:
        xy_physics_full = None

    # Gestione precisione per L-BFGS
    if cfg.precision_mode == 'staged':
        print("\n--- Switching to FP64 for L-BFGS Refinement (Staged Mode) ---")
        torch.set_default_dtype(torch.float64)
        torch.backends.cuda.matmul.allow_tf32 = False
        model.double()
        xy_int      = xy_int.double()
        T_int       = T_int.double()
        xy_bc       = xy_bc.double()
        T_bc        = T_bc.double()
        if xy_physics_full is not None:
            xy_physics_full = xy_physics_full.detach().double().requires_grad_(True)
        xy_grid     = xy_grid.double()
        T_exact_grid = T_exact_grid.double()
        X, Y = X.double(), Y.double()
    elif cfg.precision_mode == 'full_64':
        print("\n--- Continuing with FP64 for L-BFGS (Full 64 Mode) ---")
        if xy_physics_full is not None:
            xy_physics_full = xy_physics_full.detach().requires_grad_(True)
    else: # full_32
        print("\n--- Continuing with FP32 for L-BFGS (Full 32 Mode) ---")
        if xy_physics_full is not None:
            xy_physics_full = xy_physics_full.detach().requires_grad_(True)
    
    # Verifica precisione finale
    target_dtype = torch.float64 if cfg.precision_mode in ['staged', 'full_64'] else torch.float32
    assert all(p.dtype == target_dtype for p in model.parameters()), \
        f"Errore: parametri del modello non tutti in {target_dtype} prima di L-BFGS"

    max_total_lbfgs = cfg.max_lbfgs_iters
    lbfgs_iter = [0]
    pbar_lbfgs = tqdm(total=max_total_lbfgs, desc="Training PINN (L-BFGS)", mininterval=2.0)
    
    # Kwargs condivisi per compute_pinn_loss (evita duplicazione tra closure e final check)
    loss_kwargs = {
        'x_data': xy_int, 'y_data': T_int,
        'x_bc': xy_bc, 'y_bc': T_bc,
        'physics_problem': physics_problem,
        'x_physics': xy_physics_full,
        'lambda_data': lambda_data,
        'lambda_bc': lambda_bc,
        'lambda_physics': target_lambda_physics,
        'mode': mode,
        'variance_weights': variance_weights
    }
    
    for current_lr in [1.0, 0.5]:
        remaining_evals = max_total_lbfgs - lbfgs_iter[0]
        if remaining_evals <= 0:
            break
            
        optimizer_lbfgs = torch.optim.LBFGS(
            model.parameters(), 
            lr=current_lr, 
            max_iter=remaining_evals, 
            max_eval=remaining_evals * 5, 
            tolerance_grad=1e-7, 
            tolerance_change=1e-9,
            history_size=300,
            line_search_fn="strong_wolfe"
        )
        
        def make_closure(opt_ref):
            def closure():
                opt_ref.zero_grad()
                loss, loss_dict = compute_pinn_loss(model, **loss_kwargs)
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
        
        if lbfgs_iter[0] >= max_total_lbfgs:
            break
        
        if current_lr == 1.0:
            print(f"\nL-BFGS interrotto a {lbfgs_iter[0]} chiamate (LR=1.0). Riprovo con LR=0.5 per le restanti {max_total_lbfgs - lbfgs_iter[0]}...")
    
    pbar_lbfgs.close()
    
    # Final loss check
    final_loss, final_loss_dict = compute_pinn_loss(model, **loss_kwargs)
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
    internal_pts = xy_int if lambda_data > 0 else None
    boundary_pts = xy_bc if lambda_bc > 0 else None

    final_path = os.path.join(final_dir, 'PINNfinal_result.png')
    plot2D_final_result(X, Y, T_exact_grid, T_final, epochs, save_path=final_path, internal_points=internal_pts, boundary_points=boundary_pts, physics_points=xy_physics, val_label=cfg.val_label)
    
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
    
    # Phase markers per Staged Training
    _phase_markers = None
    if staged_training:
        _phase_markers = [
            {'epoch': half_epochs, 'label': 'Fase 2 (Tau)', 'color': 'purple'},
        ]
    
    # Plot Loss History con split tra Adam e L-BFGS
    loss_history.plot_losses(
        adam_epochs=epochs,
        save_path=os.path.join(final_dir, 'PINN_loss_history.png'), 
        experiment_name=cfg.experiment_name, 
        skip_epochs=50,
        phase_markers=_phase_markers,
        smoothing_alpha=0.95,
        active_loss_keys=_active_keys if _active_keys else None
    )
    
    loss_history.plot_gradients(save_path=os.path.join(final_dir, 'PINN_gradients.png'), experiment_name=f"{cfg.experiment_name} Gradients")
    loss_history.plot_weights(save_path=os.path.join(final_dir, 'PINN_weights.png'), experiment_name=f"{cfg.experiment_name} Weights")

    plt.close("all")

    # RIPRISTINO PRECISIONE PER EVENTUALI CHIAMATE SUCCESSIVE
    final_dtype = torch.float32 if cfg.precision_mode != 'full_64' else torch.float64
    torch.set_default_dtype(final_dtype)
    return loss_history