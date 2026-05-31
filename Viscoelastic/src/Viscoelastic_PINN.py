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
from func.graphic_func import generate_epoch_diagnostic_plot, generate_final_training_plots
from func.history_tracker import TrainingHistory
from func.hardware_utils import IS_1050TI

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

def set_physics_trainable(physics_problem, active_params=['mu_s', 'mu_p', 'lam', 'eps', 'alpha']):
    if not getattr(physics_problem, 'inverse_mode', False):
        return
    for p_name in ['mu_s', 'mu_p', 'lam', 'eps', 'alpha']:
        p_val = getattr(physics_problem, p_name)
        if isinstance(p_val, torch.Tensor) and p_val.is_leaf:
            p_val.requires_grad_(p_name in active_params)
    print(f"  [Physics Trainable] {active_params}")

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


def _run_adam_phase(model, physics_problem, cfg, data_internal, data_boundary, validation_grid, collocation_points, loss_history, plots_dir):
    """Esegue la prima e la seconda fase di training tramite ottimizzatore Adam."""
    _device = next(model.parameters()).device
    epochs = cfg.epochs
    half_epochs = epochs // 2
    staged_training = cfg.staged_training
    lambda_data = cfg.loss_weights.get('data', 1.0)
    lambda_bc = cfg.loss_weights.get('bc', 1.0)
    target_lambda_physics = cfg.loss_weights.get('physics', 1.0)
    base_pde_weights = physics_problem.pde_weights.copy()
    
    xy_int, obs_int = data_internal
    xy_bc, dir_bc, neu_bc, norm_bc = data_boundary
    bc_targets = (dir_bc, neu_bc, norm_bc)
    xy_grid, T_exact_grid, triang = validation_grid
    plot_files = []
    
    pbar = tqdm(range(epochs), desc=f"Training VE (Adam) ({cfg.lr_strategy})", mininterval=2.0)
    
    warmup_ratio = getattr(cfg, 'warmup_ratio', 0.1)
    warmup_epochs_1 = int(epochs * warmup_ratio) if staged_training else 0
    warmup_epochs_2 = half_epochs + int(epochs * warmup_ratio) if staged_training else 0
    
    def _rebuild_optimizer(steps_remaining):
        net_params = [p for p in model.parameters() if p.requires_grad]
        phys_params = [p for p in physics_problem.parameters() if p.requires_grad]
        
        param_groups = [
            {'params': net_params, 'lr': cfg.base_lr},
            {'params': phys_params, 'lr': cfg.base_lr * getattr(cfg, 'param_lr_factor', 0.1)}
        ]
        
        opt = torch.optim.Adam(param_groups, eps=cfg.adam_eps)
        sch = _get_scheduler(opt, cfg.lr_strategy, steps_remaining if steps_remaining > 0 else 1)
        
        last_layer = []
        for net in [model.model_psi, model.model_p, model.model_tau]:
            if hasattr(net, 'fcs') and len(net.fcs) > 0:
                last_layer.extend([p for p in net.fcs[-1].parameters() if p.requires_grad])
        if not last_layer:
            last_layer = net_params + phys_params
        
        return opt, sch, last_layer, net_params + phys_params
    
    if staged_training:
        print(f"\n  [Staged Training] Fase 1 (Warmup): Cinematica e Reologia (psi+tau).")
        set_model_trainable(model, ['psi', 'tau']) #qui spengo rete pressione
        physics_problem.pde_weights = {'momentum': 0.0, 'constitutive': base_pde_weights.get('constitutive', 1.0)} 
        current_active_bcs = ['u', 'v', 'txx', 'txy', 'tyy'] #non ho bc su p
        set_physics_trainable(physics_problem, []) # Tutto bloccato nel warmup
    else:
        set_model_trainable(model, ['psi', 'p', 'tau'])
        current_active_bcs = None #non filtra nulla, quindi tutto
        set_physics_trainable(physics_problem, ['mu_s', 'mu_p', 'lam', 'eps', 'alpha'])

    optimizer, scheduler, _last_layer_trainable, trainable_params = _rebuild_optimizer(warmup_epochs_1 if staged_training else epochs)

    alpha_dynamic = 0.9 #coefficiente di smoothing per aggiornare i pesi dinamici di loss bc e dati

    for epoch in pbar:
        if staged_training and epoch == warmup_epochs_1:
            print(f"\n  [Staged Training] Fine Warmup 1. Sblocco parametri costitutivi.")
            set_physics_trainable(physics_problem, ['mu_p', 'lam', 'eps', 'alpha'])
            optimizer, scheduler, _last_layer_trainable, trainable_params = _rebuild_optimizer(half_epochs - warmup_epochs_1)
            
        if staged_training and epoch == half_epochs:
            print(f"\n  [Staged Training] Fase 2 (Warmup): Dinamica (psi+p). Navier-Stokes ON.")
            set_model_trainable(model, ['psi', 'p'])
            physics_problem.pde_weights = {'momentum': base_pde_weights.get('momentum', 10.0), 'constitutive': 0.0}
            current_active_bcs = ['u', 'v', 'p']
            set_physics_trainable(physics_problem, []) # Tutto bloccato nel warmup
            optimizer, scheduler, _last_layer_trainable, trainable_params = _rebuild_optimizer(warmup_epochs_2 - half_epochs)

        if staged_training and epoch == warmup_epochs_2:
            print(f"\n  [Staged Training] Fine Warmup 2. Sblocco parametro momentum.")
            set_physics_trainable(physics_problem, ['mu_s'])
            optimizer, scheduler, _last_layer_trainable, trainable_params = _rebuild_optimizer(epochs - warmup_epochs_2)

        model.train()
        optimizer.zero_grad(set_to_none=True) #azzera i gradienti accumulati, ripartendo proprio da zero
        
        xy_int_batch, obs_int_batch = _sample_minibatch(xy_int, obs_int, cfg.minibatch_internal, _device) #prende un batch di dati
        xy_bc_batch, bc_targets_batch = _sample_minibatch(xy_bc, bc_targets, cfg.minibatch_boundary, _device)
        
        if target_lambda_physics > 0: #attiva punti in memoria solo se la fisica viene calcolata
            if lambda_data > 0:
                xy_phys_batch = xy_int_batch.clone().requires_grad_(True) #riusa i punti dei dati se vengono usati
            else:
                xy_phys_batch = _sample_minibatch(collocation_points, collocation_points, cfg.minibatch_internal, _device)[0].clone().requires_grad_(True)
        else:
            xy_phys_batch = None

        loss, loss_dict = compute_pinn_loss(
            model, x_data=xy_int_batch, y_data=obs_int_batch,
            x_bc=xy_bc_batch, y_bc=bc_targets_batch,
            physics_problem=physics_problem, x_physics=xy_phys_batch,
            lambda_data=lambda_data, lambda_bc=lambda_bc, lambda_physics=target_lambda_physics,
            mode=cfg.mode, variance_weights=cfg.variance_weights, active_bcs=current_active_bcs,
            force_data_loss=((epoch + 1) % 100 == 0)
        )

        if cfg.dynamic_weighting and (epoch + 1) % cfg.update_weights_every == 0:
            #calcolo norma massima del gradiente delle BC
            if lambda_bc > 0 and 'bc_loss' in loss_dict and isinstance(loss_dict['bc_loss'], torch.Tensor) and loss_dict['bc_loss'].requires_grad:
                grads_bc = torch.autograd.grad(loss_dict['bc_loss'], _last_layer_trainable, retain_graph=True, allow_unused=True)
                max_norm_bc = max([g.norm(2) for g in grads_bc if g is not None]).item() if any(g is not None for g in grads_bc) else 0.0
                #scalo il peso delle PDE
                if target_lambda_physics > 0 and 'pde_loss' in loss_dict and isinstance(loss_dict['pde_loss'], torch.Tensor) and loss_dict['pde_loss'].requires_grad:
                    grads_ph = torch.autograd.grad(loss_dict['pde_loss'], _last_layer_trainable, retain_graph=True, allow_unused=True)
                    m_n_ph = max([g.norm(2) for g in grads_ph if g is not None]).item() if any(g is not None for g in grads_ph) else 0.0
                    if m_n_ph > 1e-12: 
                        ratio = min(max_norm_bc / m_n_ph, 100.0)
                        target_lambda_physics = alpha_dynamic * target_lambda_physics + (1-alpha_dynamic) * ratio * lambda_bc
                #scalo il peso dei dati interni
                if lambda_data > 0 and 'data_loss' in loss_dict and isinstance(loss_dict['data_loss'], torch.Tensor) and loss_dict['data_loss'].requires_grad:
                    grads_dt = torch.autograd.grad(loss_dict['data_loss'], _last_layer_trainable, retain_graph=True, allow_unused=True)
                    m_n_dt = max([g.norm(2) for g in grads_dt if g is not None]).item() if any(g is not None for g in grads_dt) else 0.0
                    if m_n_dt > 1e-12: 
                        ratio_d = min(max_norm_bc / m_n_dt, 100.0)
                        lambda_data = alpha_dynamic * lambda_data + (1-alpha_dynamic) * ratio_d * lambda_bc

        loss.backward(inputs=trainable_params)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip_norm) #clippa i gradienti con clip a 5
        phys_params_clip = [p for p in physics_problem.parameters() if p.requires_grad and p.grad is not None]
        if phys_params_clip:
            torch.nn.utils.clip_grad_norm_(phys_params_clip, max_norm=cfg.param_clip_norm) #clippa i parametri con clip a 1 più aggressivo
        optimizer.step()
        
        if cfg.lr_strategy in ['step_decay', 'cosine']: 
            scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        history_entry = loss_dict.copy()
        history_entry.update({'weight_data': lambda_data, 'weight_bc': lambda_bc, 'weight_phys': target_lambda_physics})
        
        if getattr(physics_problem, 'inverse_mode', False):
            eff = physics_problem.get_logged_parameters()
            history_entry.update({
                'param_etas': eff['mu_s'],
                'param_etap': eff['mu_p'],
                'param_lam': eff['lam'],
                'param_epsilon': eff['eps'],
                'param_alpha': eff['alpha']
            })
        loss_history.update(epoch, history_entry, lr=current_lr)

        if (epoch + 1) % 100 == 0:
            pbar.set_postfix({'Loss': f"{loss.item():.2e}", 'LR': f"{optimizer.param_groups[0]['lr']:.1e}"})
            # plotting
            if (epoch + 1) % cfg.plot_every == 0:
                generate_epoch_diagnostic_plot(
                    model, physics_problem, xy_grid, T_exact_grid, triang,
                    epoch, plots_dir, cfg.plot_every, cfg.val_label, plot_files
                )
            
    pbar.close()
    return loss_history, plot_files, lambda_data, lambda_bc, target_lambda_physics, base_pde_weights

def _run_lbfgs_phase(model, physics_problem, cfg, data_internal, data_boundary, collocation_points, loss_history, lambda_data, lambda_bc, target_lambda_physics, base_pde_weights):
    """Esegue il raffinamento finale in FP64 usando L-BFGS."""
    _dtype = next(model.parameters()).dtype
    xy_int, obs_int = data_internal
    xy_bc, dir_bc, neu_bc, norm_bc = data_boundary
    bc_targets = (dir_bc, neu_bc, norm_bc)
    
    print("\n  [Staged Training] Fase 3: Raffinamento L-BFGS (Tutto sbloccato)")
    set_model_trainable(model, ['psi', 'p', 'tau']) #sblocchiamo tutto
    physics_problem.pde_weights = base_pde_weights
    
    if getattr(physics_problem, 'inverse_mode', False):
        physics_problem.mu_s.requires_grad_(True)
        physics_problem.mu_p.requires_grad_(True)
        physics_problem.lam.requires_grad_(True)
        physics_problem.eps.requires_grad_(True)
        physics_problem.alpha.requires_grad_(True)

    if cfg.precision_mode == 'staged': #se abbiamo staged e si parte da float32 poi casta tutto a 64
        torch.set_default_dtype(torch.float64)
        model.double()
        physics_problem.double()
        xy_int, obs_int, xy_bc = xy_int.double(), obs_int.double(), xy_bc.double()
        bc_targets = tuple(t.double() for t in bc_targets)
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
        tolerance_grad=1e-9, tolerance_change=1e-12,
        history_size=50 if IS_1050TI else 300, line_search_fn="strong_wolfe" #ottimizzazione parametri per lowvram
    )

    lbfgs_iter = [0]
    pbar_lbfgs = tqdm(total=cfg.max_lbfgs_iters, desc="Training VE (L-BFGS)", mininterval=2.0)
    
    # Imposta un chunk_size di default prudenziale (es. 2000) per evitare OOM su qualsiasi GPU
    chunk_size = 500 if IS_1050TI else 2000
    
    def closure():
        optimizer_lbfgs.zero_grad()
        
        # Calcoliamo data_loss solo ogni 50 iterazioni (quando la scriviamo nella history)
        do_force_data = (lbfgs_iter[0] % 50 == 0)
        
        loss, loss_dict = compute_pinn_loss(
            model, x_data=xy_int, y_data=obs_int,
            x_bc=xy_bc, y_bc=bc_targets,
            physics_problem=physics_problem, x_physics=xy_physics_full,
            lambda_data=lambda_data, lambda_bc=lambda_bc, lambda_physics=target_lambda_physics,
            mode=cfg.mode, variance_weights=cfg.variance_weights, active_bcs=None,
            force_data_loss=do_force_data,
            chunk_size=chunk_size
        )
        if chunk_size is None:
            loss.backward()
        
        if lbfgs_iter[0] % 50 == 0: #logging della loss modificato a 50 iterazioni
            history_entry = loss_dict.copy()
            history_entry.update({'weight_data': lambda_data, 'weight_bc': lambda_bc, 'weight_phys': target_lambda_physics})
            if getattr(physics_problem, 'inverse_mode', False):
                eff = physics_problem.get_logged_parameters()
                history_entry.update({
                    'param_etas': eff['mu_s'],
                    'param_etap': eff['mu_p'],
                    'param_lam': eff['lam'],
                    'param_epsilon': eff['eps'],
                    'param_alpha': eff['alpha']
                })
            loss_history.update(cfg.epochs + lbfgs_iter[0], history_entry, lr=1.0) #per continuare su stesso grafico di adam
            
        lbfgs_iter[0] += 1
        pbar_lbfgs.update(1)
        pbar_lbfgs.set_postfix({'Loss': f"{loss.item():.2e}"})
        return loss

    print("Esecuzione L-BFGS...")
    optimizer_lbfgs.step(closure)
    pbar_lbfgs.close()

    if cfg.precision_mode == 'staged': #reset della precisione precedente in base alla config
        torch.set_default_dtype(_dtype)
        model.to(_dtype)
        physics_problem.to(_dtype)
    else:
        torch.set_default_dtype(torch.float32)

    return loss_history

def _generate_training_artifacts(model, physics_problem, validation_grid, stress_exact_grids,
                                 data_internal, data_boundary, collocation_points, 
                                 plots_dir, final_dir, loss_history, plot_files, 
                                 lambda_data, lambda_bc, target_lambda_physics, cfg):
    """Gestisce l'inferenza finale, il plotting, le GIF e il salvataggio dei log."""
    print("Training completato. Generazione plot finali e GIF...")
    xy_grid, T_exact_grid, triang = validation_grid
    T_exact_grid = T_exact_grid.cpu()
    
    xy_int, _ = data_internal
    xy_bc, _, _, _ = data_boundary

    model.eval()
    
    # --- INFERENZA SINGOLA OTTIMIZZATA ---
    with torch.set_grad_enabled(True): 
        xy_grid_val = xy_grid.clone().detach().requires_grad_(True)
        
        # Calcoliamo tutto subito una volta sola
        if hasattr(physics_problem, 'get_velocity'):
            u_final, v_final, p_final, _ = physics_problem.get_velocity(model, xy_grid_val)
            out_final = model(xy_grid_val)
            T_final = u_final.detach().cpu().view(-1)
        else:
            out_final = model(xy_grid_val)
            u_pred = out_final[:, 0].detach().cpu()
            T_final = u_pred.view(-1)
            
        del xy_grid_val # Puliamo immediatamente la VRAM
        
    # --- GESTIONE PUNTI DI PLOT ---
    internal_pts = xy_int if lambda_data > 0 else None
    boundary_pts = xy_bc if lambda_bc > 0 else None
    if target_lambda_physics > 0:
        xy_physics_full = xy_int if lambda_data > 0 else collocation_points
    else:
        xy_physics_full = None

    # --- GENERAZIONE DEI PLOT FINALI E EVOLUZIONE ---
    generate_final_training_plots(
        final_dir, plots_dir, triang, T_exact_grid, T_final, p_final, out_final,
        stress_exact_grids, plot_files, cfg.epochs, cfg.val_label,
        internal_pts, boundary_pts, xy_physics_full
    )
    
    # --- STORICO LOSS ---
    half_epochs = cfg.epochs // 2
    _phase_markers = [{'epoch': half_epochs, 'label': 'Fase 2 (Dinamica)', 'color': 'purple'}] if cfg.staged_training else None
    
    _active_keys = set()
    if lambda_data > 0: _active_keys.add('data')
    if lambda_bc > 0: _active_keys.add('bc')
    if target_lambda_physics > 0: _active_keys.add('physics')
    
    loss_history.plot_losses(
        adam_epochs=cfg.epochs,
        save_path=os.path.join(final_dir, 'VE_loss_history.png'), 
        experiment_name=cfg.experiment_name, 
        skip_epochs=50,
        phase_markers=_phase_markers,
        smoothing_alpha=0.95,
        active_loss_keys=_active_keys if _active_keys else None
    )
    
    # --- STORICO GRADIENTI ---
    loss_history.plot_gradients(
        save_path=os.path.join(final_dir, 'VE_gradients.png'), 
        experiment_name=f"{cfg.experiment_name} Gradients", 
        show_plot=False
    )
    
    plt.close("all")

def train_ViscoelasticPINN(
    model, config, data_internal, data_boundary,
    validation_grid, physics_problem, collocation_points,
    plots_dir, final_dir, stress_exact_grids=None
):
    """
    Pipeline principale di addestramento per PINN Viscoelastiche.
    Si divide in 3 fasi:
      1. Adam (Warmup: Cinematica e Reologia)
      2. Adam (Dinamica accoppiata)
      3. L-BFGS (Raffinamento in precisione FP64)
    """
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)
    loss_history = TrainingHistory()
    
    # 1. Fase Adam (Eventuale Staged Training gestito internamente)
    loss_history, plot_files, lambda_data, lambda_bc, target_lambda_physics, base_pde_weights = _run_adam_phase(
        model, physics_problem, config, 
        data_internal, data_boundary, validation_grid, collocation_points, 
        loss_history, plots_dir
    )

    # 2. Fase L-BFGS (Raffinamento di precisione)
    loss_history = _run_lbfgs_phase(
        model, physics_problem, config, 
        data_internal, data_boundary, collocation_points, 
        loss_history, lambda_data, lambda_bc, target_lambda_physics, base_pde_weights
    )

    # 3. Inferenza e Generazione Artefatti Finali
    _generate_training_artifacts(
        model, physics_problem, validation_grid, stress_exact_grids,
        data_internal, data_boundary, collocation_points,
        plots_dir, final_dir, loss_history, plot_files,
        lambda_data, lambda_bc, target_lambda_physics, config
    )

    return loss_history


def compute_pinn_loss(model, x_data, y_data, x_bc=None, y_bc=None, x_physics=None, physics_problem=None, lambda_data=1.0, lambda_bc=1.0, lambda_physics=1.0, mode='standard', variance_weights=None, force_data_loss=False, chunk_size=None, **kwargs):
    """
    Computes the components of the PINN loss.
    COMPONENTS IN 'loss_dict' ARE PURE RESIDUALS (UNWEIGHTED).
    'total_loss' IS WEIGHTED.
    Requires a physics_problem instance; uses semi_inverse or comsol_full mode.
    If chunk_size is specified, performs gradient accumulation internally chunk-by-chunk.
    """
    # ═══════════════════════════════════════════════════
    # Gestione chunking (accumulo dei gradienti a blocchi)
    # ═══════════════════════════════════════════════════
    if chunk_size is not None:
        total_loss_val = 0.0
        loss_dict_accum = {'data_loss': 0.0, 'bc_loss': 0.0, 'pde_loss': 0.0, 'total_loss': 0.0}
        cur_device = next(model.parameters()).device
        cur_dtype = next(model.parameters()).dtype

        # 1. Chunking Data Loss
        if x_data is not None and y_data is not None and x_data.numel() > 0:
            N_data = x_data.shape[0]
            for i in range(0, N_data, chunk_size):
                x_c = x_data[i : i + chunk_size]
                y_c = y_data[i : i + chunk_size]
                c_loss, c_dict = compute_pinn_loss(
                    model, x_data=x_c, y_data=y_c, x_bc=None, y_bc=None, x_physics=None,
                    physics_problem=physics_problem, lambda_data=lambda_data, lambda_bc=0.0, lambda_physics=0.0,
                    mode=mode, variance_weights=variance_weights, force_data_loss=force_data_loss,
                    chunk_size=None, **kwargs
                )
                c_loss_scaled = c_loss * (x_c.shape[0] / N_data)
                
                c_data_val = c_dict.get('data_loss', 0.0)
                c_data_val_item = c_data_val.item() if hasattr(c_data_val, 'item') else c_data_val
                loss_dict_accum['data_loss'] += c_data_val_item * (x_c.shape[0] / N_data)
                
                if c_loss_scaled.requires_grad:
                    c_loss_scaled.backward()

            total_loss_val += lambda_data * loss_dict_accum['data_loss']

        # 2. Boundary Loss (un-chunked)
        if physics_problem is not None and x_bc is not None and y_bc is not None and x_bc.numel() > 0:
            bc_loss_val, bc_dict = compute_pinn_loss(
                model, x_data=None, y_data=None, x_bc=x_bc, y_bc=y_bc, x_physics=None,
                physics_problem=physics_problem, lambda_data=0.0, lambda_bc=lambda_bc, lambda_physics=0.0,
                mode=mode, variance_weights=variance_weights, force_data_loss=False,
                chunk_size=None, **kwargs
            )
            bc_val = bc_dict.get('bc_loss', 0.0)
            loss_dict_accum['bc_loss'] = bc_val.item() if hasattr(bc_val, 'item') else bc_val
            
            if bc_loss_val.requires_grad:
                bc_loss_val.backward()
            
            total_loss_val += bc_loss_val.item() if hasattr(bc_loss_val, 'item') else bc_loss_val

        # 3. Chunking PDE Loss
        if physics_problem is not None and x_physics is not None and x_physics.numel() > 0:
            N_phys = x_physics.shape[0]
            for i in range(0, N_phys, chunk_size):
                x_c = x_physics[i : i + chunk_size]
                c_loss, c_dict = compute_pinn_loss(
                    model, x_data=None, y_data=None, x_bc=None, y_bc=None, x_physics=x_c,
                    physics_problem=physics_problem, lambda_data=0.0, lambda_bc=0.0, lambda_physics=lambda_physics,
                    mode=mode, variance_weights=variance_weights, force_data_loss=False,
                    chunk_size=None, **kwargs
                )
                c_loss_scaled = c_loss * (x_c.shape[0] / N_phys)
                
                c_pde_val = c_dict.get('pde_loss', 0.0)
                c_pde_val_item = c_pde_val.item() if hasattr(c_pde_val, 'item') else c_pde_val
                loss_dict_accum['pde_loss'] += c_pde_val_item * (x_c.shape[0] / N_phys)
                
                if c_loss_scaled.requires_grad:
                    c_loss_scaled.backward()
            
            total_loss_val += lambda_physics * loss_dict_accum['pde_loss']

        loss_dict_accum['total_loss'] = total_loss_val
        return torch.tensor(total_loss_val, device=cur_device, dtype=cur_dtype), loss_dict_accum

    # ═══════════════════════════════════════════════════
    # Calcolo Loss Standard (senza chunking)
    # ═══════════════════════════════════════════════════
    loss_dict = {}
    total_loss = 0.0
    mse_loss = nn.MSELoss()
    
    # Normalizzazione per Goal 1 (ViscoelasticNet)
    scale_u = 1.0
    scale_v = 1.0
    if mode == 'semi_inverse' and variance_weights is not None:
        scale_u = variance_weights.get('u', 1.0)
        scale_v = variance_weights.get('v', 1.0)
    
    # Determina se lambda_data è zero (evitando sync se è un tensore)
    lambda_data_is_zero = kwargs.get('lambda_data_is_zero', None)
    if lambda_data_is_zero is None:
        if isinstance(lambda_data, torch.Tensor):
            lambda_data_is_zero = False
        else:
            lambda_data_is_zero = (lambda_data == 0.0)

    if x_data is not None and y_data is not None and x_data.numel() > 0:
        if lambda_data_is_zero:
            if not force_data_loss:
                # Se la loss dei dati ha peso zero e non è esplicitamente richiesto il calcolo per diagnostica,
                # restituiamo 0.0 per evitare il calcolo costoso e l'accumulo di VRAM dovuto ai grafi di autograd.
                data_loss = torch.tensor(0.0, device=x_data.device, dtype=x_data.dtype)
            else:
                # Congeliamo temporaneamente i parametri della rete per evitare la creazione di grafi per i pesi
                saved_requires_grad = [p.requires_grad for p in model.parameters()]
                for p in model.parameters():
                    p.requires_grad = False
                
                try:
                    u_pred, v_pred, p_pred, tau_pred = physics_problem.get_velocity(model, x_data)
                    u_obs = y_data[:, 0:1]
                    v_obs = y_data[:, 1:2]
                    loss_u = mse_loss(u_pred, u_obs) / scale_u
                    loss_v = mse_loss(v_pred, v_obs) / scale_v
                    data_loss = 0.5 * (loss_u + loss_v)
                    if mode == 'comsol_full' and y_data.shape[1] >= 6:
                        out = model(x_data)
                        data_loss = data_loss + mse_loss(out[:, 1:2], y_data[:, 2:3])  # p
                        data_loss = data_loss + mse_loss(out[:, 2:5], y_data[:, 3:6])  # tau
                finally:
                    # Ripristiniamo lo stato dei parametri
                    for p, req in zip(model.parameters(), saved_requires_grad):
                        p.requires_grad = req
        else:
            # y_data contiene [u_obs, v_obs] (semi_inverse) o [u,v,p,txx,txy,tyy] (comsol_full)
            u_pred, v_pred, p_pred, tau_pred = physics_problem.get_velocity(model, x_data)
            u_obs = y_data[:, 0:1]
            v_obs = y_data[:, 1:2]
            
            loss_u = mse_loss(u_pred, u_obs) / scale_u
            loss_v = mse_loss(v_pred, v_obs) / scale_v
            
            data_loss = 0.5 * (loss_u + loss_v)
            
            # In comsol_full, confrontiamo anche p e tau direttamente
            if mode == 'comsol_full' and y_data.shape[1] >= 6:
                out = model(x_data)
                scale_p = variance_weights.get('p', 1.0) if variance_weights is not None else 1.0
                scale_txx = variance_weights.get('txx', 1.0) if variance_weights is not None else 1.0
                scale_txy = variance_weights.get('txy', 1.0) if variance_weights is not None else 1.0
                scale_tyy = variance_weights.get('tyy', 1.0) if variance_weights is not None else 1.0
                loss_p = mse_loss(out[:, 1:2], y_data[:, 2:3]) / scale_p
                loss_txx = mse_loss(out[:, 2:3], y_data[:, 3:4]) / scale_txx
                loss_txy = mse_loss(out[:, 3:4], y_data[:, 4:5]) / scale_txy
                loss_tyy = mse_loss(out[:, 4:5], y_data[:, 5:6]) / scale_tyy
                data_loss = data_loss + (loss_p + loss_txx + loss_txy + loss_tyy) / 4.0
            
        loss_dict['data_loss'] = data_loss
        total_loss += lambda_data * data_loss

    if physics_problem is not None and x_bc is not None and y_bc is not None and x_bc.numel() > 0:
        # Passiamo variance_weights per normalizzare u, v, p, tau individualmente sia in semi_inverse che in PurePhys
        v_weights = variance_weights
        active_bcs = kwargs.get('active_bcs', None)
        bc_loss_val = physics_problem.boundary_loss(model, x_bc, y_bc, variance_weights=v_weights, active_bcs=active_bcs)
        loss_dict['bc_loss'] = bc_loss_val
        total_loss += lambda_bc * bc_loss_val
    
    if physics_problem is not None and x_physics is not None:
        pde_loss = physics_problem.residual(model, x_physics, variance_weights=variance_weights)
        loss_dict['pde_loss'] = pde_loss
        total_loss += lambda_physics * pde_loss
        
    loss_dict['total_loss'] = total_loss
    return total_loss, loss_dict


def compute_viscoelastic_metrics(model, physics_problem, xy_grid_flat, fields_exact_flat, Ny_dom=None, Nx_dom=None):
    """
    Calcola L2 Relative Error e Max Relative Error per ogni campo fisico
    del modello viscoelastico: u, p, tau_xx, tau_xy, tau_yy.
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
        
        true_flat = exact_grid.view(-1).cpu().to(pred_flat.dtype)
        
        # L2 Relative Error
        l2_error = torch.norm(pred_flat - true_flat, 2)
        l2_ref = torch.norm(true_flat, 2)
        l2_rel = (l2_error / l2_ref).item() if l2_ref > 1e-10 else 0.0
        
        # Max Relative Error
        abs_error = torch.abs(pred_flat - true_flat)
        max_val = torch.max(torch.abs(true_flat)).item()
        threshold = max(0.05 * max_val, 1e-8)
        mask = torch.abs(true_flat) > threshold
        rel_error = torch.zeros_like(true_flat)
        if mask.sum() > 0:
            rel_error[mask] = (abs_error[mask] / torch.abs(true_flat[mask])) * 100
            max_rel = torch.max(rel_error).item()
        else:
            max_rel = 0.0
        
        metrics[fname] = (l2_rel, max_rel)
    
    return metrics
