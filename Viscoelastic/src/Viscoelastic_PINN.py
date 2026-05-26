import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import shutil
from dataclasses import dataclass, field
from tqdm import tqdm

# Import function for GIF and loss comparison
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from func.graphic_func import save_gif_PIL, plot2D_comparison, plot2D_final_result, plot2D_viscoelastic_final
from func.history_tracker import TrainingHistory, compute_pinn_loss
from Viscoelastic.src.losses import compute_chunked_gradients
from func.hardware_utils import IS_1050TI, SUPPORTS_COMPILE

# --- DEFINIZIONE DELLA RETE NEURALE E WRAPPER ---
class FCN(nn.Module):
    """Rete Neurale a Connessioni Complete (Fully Connected Network)"""
    def __init__(self, layers, activation_fn=nn.Tanh):
        super().__init__()
        self.activation = activation_fn()
        self.fcs = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.fcs.append(nn.Linear(layers[i], layers[i+1]))
    def forward(self, x):
        for layer in self.fcs[:-1]:
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
    for p in model_combined.parameters():
        p.requires_grad = False
    
    if 'psi' in active_components:
        for p in model_combined.model_psi.parameters(): p.requires_grad = True
    if 'p' in active_components:
        for p in model_combined.model_p.parameters(): p.requires_grad = True
    if 'tau' in active_components:
        for p in model_combined.model_tau.parameters(): p.requires_grad = True
        
    print(f"  [Trainable status] Psi: {'psi' in active_components}, P: {'p' in active_components}, Tau: {'tau' in active_components}")

@dataclass
class TrainingConfig:
    epochs: int = 1000
    base_lr: float = 1e-3
    adam_eps: float = 1e-7
    lr_strategy: str = 'cosine'
    staged_training: bool = True
    warmup_ratio: float = 0.1
    precision_mode: str = 'staged'
    max_lbfgs_iters: int = 100
    use_compile: bool = field(default_factory=lambda: SUPPORTS_COMPILE)
    grad_clip_norm: float = 5.0
    param_clip_norm: float = 1.0
    param_lr_factor: float = 0.1
    minibatch_internal: int = 1024
    minibatch_boundary: int = 256
    dynamic_weighting: bool = True
    update_weights_every: int = 100
    loss_weights: dict = field(default_factory=lambda: {'data': 1.0, 'bc': 1.0, 'physics': 1.0})
    mode: str = 'standard'
    variance_weights: dict = None
    log_gradients_every: int = 500
    plot_every: int = 500
    experiment_name: str = "VE Training"
    val_label: str = "Value"

def _sample_minibatch(xy, targets, batch_size, device):
    if batch_size is None or batch_size >= xy.shape[0]:
        return xy, targets
    idx = torch.randperm(xy.shape[0], device=device)[:batch_size]
    if isinstance(targets, tuple):
        return xy[idx], tuple(t[idx] for t in targets)
    return xy[idx], targets[idx]

def _get_scheduler(optimizer, strategy, total_steps):
    if strategy == 'step_decay':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(total_steps * 0.25), gamma=0.5)
    elif strategy == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=600, min_lr=1e-6, cooldown=3000)
    elif strategy == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    return None

def initialize_last_layer_zero(model):
    last_layer = list(model.fcs)[-1]
    nn.init.zeros_(last_layer.weight)
    nn.init.zeros_(last_layer.bias)
    print(f"  [Init] Ultimo layer di {model.__class__.__name__} inizializzato a zero.")

def setup_inverse_parameters(physics_problem):
    """
    Scansiona i parametri fisici prima del training. 
    Restituisce una lista di tensori che richiedono il clamp a ogni step,
    eliminando la necessità di fare if/isinstance nei loop.
    """
    params_to_clamp = []
    if getattr(physics_problem, 'inverse_mode', False):
        for p_name in ['mu_s', 'mu_p', 'lam']:
            p_val = getattr(physics_problem, p_name)
            if isinstance(p_val, torch.Tensor) and p_val.requires_grad:
                params_to_clamp.append(p_val)
    return params_to_clamp

def clamp_physical_parameters_(params_list, min_val=1e-6):
    """Applica il clamp in-place alla lista pre-compilata di parametri."""
    if not params_list:
        return
    with torch.no_grad():
        for p in params_list:
            p.clamp_(min=min_val)

def train_ViscoelasticPINN(
    model, config, data_internal, data_boundary,
    validation_grid, physics_problem, collocation_points,
    plots_dir, final_dir, stress_exact_grids=None
):
    """
    Training ottimizzato per PINN Viscoelastiche.
    """
    cfg = config
    _dtype = next(model.parameters()).dtype
    _device = next(model.parameters()).device

    epochs = cfg.epochs
    half_epochs = epochs // 2
    staged_training = cfg.staged_training
    
    # --- Estrazione Liste Parametri (NO IF NEI LOOP) ---
    params_to_clamp = setup_inverse_parameters(physics_problem)
    
    # Pre-computazione dei pesi
    lambda_data = cfg.loss_weights.get('data', 1.0)
    lambda_bc = cfg.loss_weights.get('bc', 1.0)
    target_lambda_physics = cfg.loss_weights.get('physics', 1.0)
    base_pde_weights = physics_problem.pde_weights.copy()
    
    # Unpack dei dati
    xy_int, T_int = data_internal
    xy_bc, dir_bc, neu_bc, norm_bc = data_boundary
    T_bc_tuple = (dir_bc, neu_bc, norm_bc)
    xy_grid, T_exact_grid, X, Y = validation_grid
    
    X, Y = X.cpu(), Y.cpu()
    T_exact_grid = T_exact_grid.cpu()
    Ny_dom, Nx_dom = X.shape
    
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)
    plot_files = []
    loss_history = TrainingHistory()
    
    # --- FASE ADAM ---
    pbar = tqdm(range(epochs), desc=f"Training VE (Adam) ({cfg.lr_strategy})", mininterval=2.0)
    
    # Inizializzazione Staged/Non-Staged
    if staged_training:
        print(f"\n  [Staged Training] Fase 1: Cinematica e Reologia (psi+tau).")
        set_model_trainable(model, ['psi', 'tau'])
        physics_problem.pde_weights = {'momentum': 0.0, 'constitutive': base_pde_weights.get('constitutive', 1.0)}
        current_active_bcs = ['u', 'v', 'txx', 'txy', 'tyy']
    else:
        set_model_trainable(model, ['psi', 'p', 'tau'])
        current_active_bcs = None

    trainable_params = [p for p in model.parameters() if p.requires_grad] + \
                       [p for p in physics_problem.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=cfg.base_lr, eps=cfg.adam_eps)
    scheduler = _get_scheduler(optimizer, cfg.lr_strategy, half_epochs if staged_training else epochs)

    # --- Accelerazione: torch.compile (preferito) ---
    _model_base = model  # Riferimento al modello non compilato (necessario per L-BFGS)
    use_compiler = cfg.use_compile and _device.type == 'cuda'

    if use_compiler:
        print("\n  [Compiler] Compilazione JIT con torch.compile (prima epoca lenta)...")
        model = torch.compile(model, mode="reduce-overhead")

    # Identificazione ultimo layer per Dynamic Weighting (salva memoria)
    _last_layer_trainable = []
    for net in [model.model_psi, model.model_p, model.model_tau]:
        if hasattr(net, 'fcs') and len(net.fcs) > 0:
            _last_layer_trainable.extend([p for p in net.fcs[-1].parameters() if p.requires_grad])
    if not _last_layer_trainable:
        _last_layer_trainable = trainable_params

    alpha_dynamic = 0.9

    for epoch in pbar:
        # Gestione Staged Training: Switch Fase 2
        if staged_training and epoch == half_epochs:
            print(f"\n  [Staged Training] Fase 2: Dinamica (psi+p). Navier-Stokes ON.")
            set_model_trainable(model, ['psi', 'p'])
            physics_problem.pde_weights = base_pde_weights
            current_active_bcs = ['u', 'v', 'p']
            
            # Ricostruzione optimizer pulita
            trainable_params = [p for p in model.parameters() if p.requires_grad] + \
                               [p for p in physics_problem.parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(trainable_params, lr=cfg.base_lr, eps=cfg.adam_eps)
            scheduler = _get_scheduler(optimizer, cfg.lr_strategy, epochs - half_epochs)
            params_to_clamp = setup_inverse_parameters(physics_problem) # Aggiorna i clamp
            
            # Ricostruzione _last_layer_trainable pulita (per evitare errori di gradienti disattivati)
            _last_layer_trainable = []
            for net in [model.model_psi, model.model_p, model.model_tau]:
                if hasattr(net, 'fcs') and len(net.fcs) > 0:
                    _last_layer_trainable.extend([p for p in net.fcs[-1].parameters() if p.requires_grad])
            if not _last_layer_trainable:
                _last_layer_trainable = trainable_params

        model.train()
        optimizer.zero_grad(set_to_none=True)
        
        # Campionamento Mini-Batch
        xy_int_batch, T_int_batch = _sample_minibatch(xy_int, T_int, cfg.minibatch_internal, _device)
        xy_bc_batch, T_bc_tuple_batch = _sample_minibatch(xy_bc, T_bc_tuple, cfg.minibatch_boundary, _device)
        
        if target_lambda_physics > 0:
            if lambda_data > 0:
                xy_phys_batch = xy_int_batch.clone().requires_grad_(True)
            else:
                xy_phys_batch = _sample_minibatch(collocation_points, collocation_points, cfg.minibatch_internal, _device)[0].clone().requires_grad_(True)
        else:
            xy_phys_batch = None

        # CLAMP EFFICIENTE (Zero overhead Python)
        clamp_physical_parameters_(params_to_clamp)

        # Forward + Loss
        loss, loss_dict = compute_pinn_loss(
            model, x_data=xy_int_batch, y_data=T_int_batch,
            x_bc=xy_bc_batch, y_bc=T_bc_tuple_batch,
            physics_problem=physics_problem, x_physics=xy_phys_batch,
            lambda_data=lambda_data, lambda_bc=lambda_bc, lambda_physics=target_lambda_physics,
            mode=cfg.mode, variance_weights=cfg.variance_weights, active_bcs=current_active_bcs
        )

        # Dynamic Weighting logic (eseguito PRIMA di loss.backward per non liberare il grafo)
        if cfg.dynamic_weighting and (epoch + 1) % cfg.update_weights_every == 0:
            if lambda_bc > 0 and 'bc_loss' in loss_dict and isinstance(loss_dict['bc_loss'], torch.Tensor) and loss_dict['bc_loss'].requires_grad:
                grads_bc = torch.autograd.grad(loss_dict['bc_loss'], _last_layer_trainable, retain_graph=True, allow_unused=True)
                max_norm_bc = max([g.norm(2) for g in grads_bc if g is not None]).item() if any(g is not None for g in grads_bc) else 0.0
                
                if target_lambda_physics > 0 and 'pde_loss' in loss_dict and isinstance(loss_dict['pde_loss'], torch.Tensor) and loss_dict['pde_loss'].requires_grad:
                    grads_ph = torch.autograd.grad(loss_dict['pde_loss'], _last_layer_trainable, retain_graph=True, allow_unused=True)
                    m_n_ph = max([g.norm(2) for g in grads_ph if g is not None]).item() if any(g is not None for g in grads_ph) else 0.0
                    if m_n_ph > 1e-12: 
                        ratio = min(max_norm_bc / m_n_ph, 100.0)
                        target_lambda_physics = alpha_dynamic * target_lambda_physics + (1-alpha_dynamic) * ratio * lambda_bc

                if lambda_data > 0 and 'data_loss' in loss_dict and isinstance(loss_dict['data_loss'], torch.Tensor) and loss_dict['data_loss'].requires_grad:
                    grads_dt = torch.autograd.grad(loss_dict['data_loss'], _last_layer_trainable, retain_graph=True, allow_unused=True)
                    m_n_dt = max([g.norm(2) for g in grads_dt if g is not None]).item() if any(g is not None for g in grads_dt) else 0.0
                    if m_n_dt > 1e-12: 
                        ratio_d = min(max_norm_bc / m_n_dt, 100.0)
                        lambda_data = alpha_dynamic * lambda_data + (1-alpha_dynamic) * ratio_d * lambda_bc

        # Backward + Optimizer step
        loss.backward(inputs=trainable_params)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip_norm)
        phys_params_clip = [p for p in physics_problem.parameters() if p.requires_grad and p.grad is not None]
        if phys_params_clip:
            torch.nn.utils.clip_grad_norm_(phys_params_clip, max_norm=cfg.param_clip_norm)
        optimizer.step()
        
        if cfg.lr_strategy in ['step_decay', 'cosine']: 
            scheduler.step()

        # Logging
        current_lr = optimizer.param_groups[0]['lr']
        history_entry = loss_dict.copy()
        history_entry.update({'weight_data': lambda_data, 'weight_bc': lambda_bc, 'weight_phys': target_lambda_physics})
        
        if getattr(physics_problem, 'inverse_mode', False):
            history_entry.update({
                'param_etas': physics_problem.mu_s.item(),
                'param_etap': physics_problem.mu_p.item(),
                'param_lam': physics_problem.lam.item()
            })
        loss_history.update(epoch, history_entry, lr=current_lr)

        if (epoch + 1) % 100 == 0:
            pbar.set_postfix({'Loss': f"{loss.item():.2e}", 'LR': f"{optimizer.param_groups[0]['lr']:.1e}"})
            
            # --- Plot intermedi ---
            if (epoch + 1) % cfg.plot_every == 0:
                model.eval()
                with torch.set_grad_enabled(True): 
                    xy_grid_val = xy_grid.clone().detach().requires_grad_(True)
                    if hasattr(physics_problem, 'get_velocity'):
                        u_pred, _, _, _ = physics_problem.get_velocity(model, xy_grid_val)
                        T_pred_grid = u_pred.detach().cpu().reshape(Ny_dom, Nx_dom)
                    else:
                        T_pred_grid = model(xy_grid_val)[:, 0].detach().cpu().reshape(Ny_dom, Nx_dom)
                    del xy_grid_val
                    
                plot_path = os.path.join(plots_dir, f'epoch_{epoch+1}.png')
                plot2D_comparison(X, Y, T_exact_grid, T_pred_grid, epoch+1, plot_path, physics_points=None, val_label=cfg.val_label, show_points=False)
                plot_files.append(plot_path)
            
    pbar.close()

    # --- FASE L-BFGS (Full Batch, Precision Switch) ---
    # Ripristino modello non compilato: L-BFGS usa line-search Python puro,
    # incompatibile con il compilatore JIT di torch.compile.
    model = _model_base
    print("\n  [Staged Training] Fase 3: Raffinamento L-BFGS (Tutto sbloccato)")
    set_model_trainable(model, ['psi', 'p', 'tau'])
    physics_problem.pde_weights = base_pde_weights
    
    # Riaggiorniamo la lista dei parametri da clippare in L-BFGS
    if getattr(physics_problem, 'inverse_mode', False):
        physics_problem.mu_s.requires_grad_(True)
        physics_problem.mu_p.requires_grad_(True)
        physics_problem.lam.requires_grad_(True)
    params_to_clamp_lbfgs = setup_inverse_parameters(physics_problem)

    # Precision Switch (Gestito elegantemente)
    if cfg.precision_mode == 'staged':
        torch.set_default_dtype(torch.float64)
        model.double()
        physics_problem.double()
        xy_int, T_int, xy_bc = xy_int.double(), T_int.double(), xy_bc.double()
        T_bc_tuple = tuple(t.double() for t in T_bc_tuple)
        if target_lambda_physics > 0:
            xy_physics_full = xy_int.clone().requires_grad_(True) if lambda_data > 0 else collocation_points.double().requires_grad_(True)
        else:
            xy_physics_full = None
    else:
        if target_lambda_physics > 0:
            xy_physics_full = xy_int.clone().requires_grad_(True) if lambda_data > 0 else collocation_points.clone().requires_grad_(True)
        else:
            xy_physics_full = None

    lbfgs_params = list(model.parameters()) + [p for p in physics_problem.parameters() if p.requires_grad]
    optimizer_lbfgs = torch.optim.LBFGS(
        lbfgs_params, lr=1.0, max_iter=cfg.max_lbfgs_iters,
        tolerance_grad=1e-7, tolerance_change=1e-9,
        history_size=50 if IS_1050TI else 300, line_search_fn="strong_wolfe"
    )

    lbfgs_iter = [0]
    pbar_lbfgs = tqdm(total=cfg.max_lbfgs_iters, desc="Training VE (L-BFGS)", mininterval=2.0)
    
    # Chunking configuration to prevent CUDA OOM on GTX 1050 Ti while preserving exact mathematical precision
    chunk_size = 500 if IS_1050TI else None
    
    def closure():
        optimizer_lbfgs.zero_grad()
        clamp_physical_parameters_(params_to_clamp_lbfgs) # Niente if qui!
        
        if chunk_size is None:
            # Original full-batch behavior (runs on standard/unrestricted GPUs)
            loss, loss_dict = compute_pinn_loss(
                model, x_data=xy_int, y_data=T_int,
                x_bc=xy_bc, y_bc=T_bc_tuple,
                physics_problem=physics_problem, x_physics=xy_physics_full,
                lambda_data=lambda_data, lambda_bc=lambda_bc, lambda_physics=target_lambda_physics,
                mode=cfg.mode, variance_weights=cfg.variance_weights, active_bcs=None
            )
            loss.backward()
        else:
            # Chunked gradient accumulation behavior (strictly equivalent but memory-friendly)
            loss, loss_dict = compute_chunked_gradients(
                model, physics_problem, xy_int, T_int, xy_bc, T_bc_tuple, xy_physics_full, 
                cfg.mode, cfg.variance_weights, lambda_data, lambda_bc, target_lambda_physics, chunk_size
            )
        
        if lbfgs_iter[0] % 10 == 0: 
            history_entry = loss_dict.copy()
            history_entry.update({'weight_data': lambda_data, 'weight_bc': lambda_bc, 'weight_phys': target_lambda_physics})
            if getattr(physics_problem, 'inverse_mode', False):
                history_entry.update({
                    'param_etas': physics_problem.mu_s.item(),
                    'param_etap': physics_problem.mu_p.item(),
                    'param_lam': physics_problem.lam.item()
                })
            loss_history.update(epochs + lbfgs_iter[0], history_entry, lr=1.0)
            
        lbfgs_iter[0] += 1
        pbar_lbfgs.update(1)
        pbar_lbfgs.set_postfix({'Loss': f"{loss.item():.2e}"})
        return loss

    print("Esecuzione L-BFGS...")
    optimizer_lbfgs.step(closure)
    pbar_lbfgs.close()

    # Ripristino Precisione
    if cfg.precision_mode == 'staged':
        torch.set_default_dtype(_dtype)
        model.to(_dtype)
        physics_problem.to(_dtype)
        xy_int = xy_int.to(_dtype)
        T_int = T_int.to(_dtype)
        xy_bc = xy_bc.to(_dtype)
        T_bc_tuple = tuple(t.to(_dtype) for t in T_bc_tuple)
        if xy_physics_full is not None:
            xy_physics_full = xy_physics_full.to(_dtype)
    else:
        torch.set_default_dtype(torch.float32)
    
    # --- PLOT FINALI E GIF ---
    print("Training completato. Generazione plot finali e GIF...")
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

    final_path = os.path.join(final_dir, 'VEfinal_result.png')
    plot2D_final_result(X, Y, T_exact_grid, T_final, epochs, save_path=final_path, internal_points=internal_pts, boundary_points=boundary_pts, physics_points=xy_physics_full, val_label=cfg.val_label)
    
    if hasattr(physics_problem, 'get_velocity') and stress_exact_grids is not None:
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
        
        visco_final_path = os.path.join(final_dir, 'VE_viscoelastic_fields.png')
        plot2D_viscoelastic_final(
            X, Y, fields_pred, fields_exact, epochs,
            save_path=visco_final_path, internal_points=internal_pts, boundary_points=boundary_pts, physics_points=xy_physics_full
        )
    
    if plot_files:
        gif_path = os.path.join(final_dir, 'VEtraining_evolution.gif')
        save_gif_PIL(gif_path, plot_files, fps=3, loop=1, delete_files=True)
    
    shutil.rmtree(plots_dir, ignore_errors=True)
    
    _phase_markers = [{'epoch': half_epochs, 'label': 'Fase 2 (Dinamica)', 'color': 'purple'}] if staged_training else None
    
    _active_keys = set()
    if lambda_data > 0: _active_keys.add('data')
    if lambda_bc > 0: _active_keys.add('bc')
    if target_lambda_physics > 0: _active_keys.add('physics')
    
    loss_history.plot_losses(
        adam_epochs=epochs,
        save_path=os.path.join(final_dir, 'VE_loss_history.png'), 
        experiment_name=cfg.experiment_name, 
        skip_epochs=50,
        phase_markers=_phase_markers,
        smoothing_alpha=0.95,
        active_loss_keys=_active_keys if _active_keys else None
    )
    plt.close("all")

    return loss_history
