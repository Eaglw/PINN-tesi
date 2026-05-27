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

def setup_inverse_parameters(physics_problem): #roba di ottimizzazione che non ho capito troppo bene
    """
    Scansiona i parametri fisici prima del training. 
    Restituisce una lista di tensori che richiedono il clamp a ogni step,
    eliminando la necessità di fare if/isinstance nei loop.
    """
    params_to_clamp = []
    if getattr(physics_problem, 'inverse_mode', False):
        for p_name in ['mu_s', 'mu_p', 'lam', 'eps', 'alpha']:
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

def _run_adam_phase(model, physics_problem, cfg, data_internal, data_boundary, validation_grid, collocation_points, loss_history, plots_dir):
    """Esegue la prima e la seconda fase di training tramite ottimizzatore Adam."""
    _device = next(model.parameters()).device
    epochs = cfg.epochs
    half_epochs = epochs // 2
    staged_training = cfg.staged_training
    
    params_to_clamp = setup_inverse_parameters(physics_problem)
    lambda_data = cfg.loss_weights.get('data', 1.0)
    lambda_bc = cfg.loss_weights.get('bc', 1.0)
    target_lambda_physics = cfg.loss_weights.get('physics', 1.0)
    base_pde_weights = physics_problem.pde_weights.copy()
    
    xy_int, obs_int = data_internal
    xy_bc, dir_bc, neu_bc, norm_bc = data_boundary
    bc_targets = (dir_bc, neu_bc, norm_bc)
    xy_grid, T_exact_grid, X, Y = validation_grid
    
    Ny_dom, Nx_dom = X.shape
    plot_files = []
    
    pbar = tqdm(range(epochs), desc=f"Training VE (Adam) ({cfg.lr_strategy})", mininterval=2.0)
    
    if staged_training:
        print(f"\n  [Staged Training] Fase 1: Cinematica e Reologia (psi+tau).")
        set_model_trainable(model, ['psi', 'tau']) #qui spengo rete pressione
        physics_problem.pde_weights = {'momentum': 0.0, 'constitutive': base_pde_weights.get('constitutive', 1.0)} #calcolo anche loss su momentum ma a zero, motivi di efficienza JIT
        current_active_bcs = ['u', 'v', 'txx', 'txy', 'tyy'] #non ho bc su p
    else:
        set_model_trainable(model, ['psi', 'p', 'tau'])
        current_active_bcs = None #non filtra nulla, quindi tutto

    trainable_params = [p for p in model.parameters() if p.requires_grad] + \
                       [p for p in physics_problem.parameters() if p.requires_grad] #assegnamo solo i parametri che hanno requiresgrad
    optimizer = torch.optim.Adam(trainable_params, lr=cfg.base_lr, eps=cfg.adam_eps)
    scheduler = _get_scheduler(optimizer, cfg.lr_strategy, half_epochs if staged_training else epochs)


    _last_layer_trainable = [] #bastano i gradienti dell'ultimo layer per una previsione accurata di tuta la rete, ma meno pesante
    for net in [model.model_psi, model.model_p, model.model_tau]:
        if hasattr(net, 'fcs') and len(net.fcs) > 0:
            _last_layer_trainable.extend([p for p in net.fcs[-1].parameters() if p.requires_grad])
    if not _last_layer_trainable:
        _last_layer_trainable = trainable_params

    alpha_dynamic = 0.9 #coefficiente di smoothing per aggiornare i pesi dinamici di loss bc e dati

    for epoch in pbar:
        if staged_training and epoch == half_epochs:
            print(f"\n  [Staged Training] Fase 2: Dinamica (psi+p). Navier-Stokes ON.")
            set_model_trainable(model, ['psi', 'p'])
            physics_problem.pde_weights = base_pde_weights
            current_active_bcs = ['u', 'v', 'p']
            
            trainable_params = [p for p in model.parameters() if p.requires_grad] + \
                               [p for p in physics_problem.parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(trainable_params, lr=cfg.base_lr, eps=cfg.adam_eps) #va rifatto l'optimizer con set di parametri nuovo
            scheduler = _get_scheduler(optimizer, cfg.lr_strategy, epochs - half_epochs)
            params_to_clamp = setup_inverse_parameters(physics_problem)
            
            _last_layer_trainable = []
            for net in [model.model_psi, model.model_p, model.model_tau]:
                if hasattr(net, 'fcs') and len(net.fcs) > 0:
                    _last_layer_trainable.extend([p for p in net.fcs[-1].parameters() if p.requires_grad])
            if not _last_layer_trainable:
                _last_layer_trainable = trainable_params

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

        clamp_physical_parameters_(params_to_clamp) #parametri fisici mai negativi

        loss, loss_dict = compute_pinn_loss(
            model, x_data=xy_int_batch, y_data=obs_int_batch,
            x_bc=xy_bc_batch, y_bc=bc_targets_batch,
            physics_problem=physics_problem, x_physics=xy_phys_batch,
            lambda_data=lambda_data, lambda_bc=lambda_bc, lambda_physics=target_lambda_physics,
            mode=cfg.mode, variance_weights=cfg.variance_weights, active_bcs=current_active_bcs
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
            history_entry.update({
                'param_etas': physics_problem.mu_s.item(),
                'param_etap': physics_problem.mu_p.item(),
                'param_lam': physics_problem.lam.item(),
                'param_epsilon': physics_problem.eps.item(),
                'param_alpha': physics_problem.alpha.item()
            })
        loss_history.update(epoch, history_entry, lr=current_lr)

        if (epoch + 1) % 100 == 0:
            pbar.set_postfix({'Loss': f"{loss.item():.2e}", 'LR': f"{optimizer.param_groups[0]['lr']:.1e}"})
            #plotting 
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
    
    if getattr(physics_problem, 'inverse_mode', False): #check che i parametri siano trainabili
        physics_problem.mu_s.requires_grad_(True)
        physics_problem.mu_p.requires_grad_(True)
        physics_problem.lam.requires_grad_(True)
        physics_problem.eps.requires_grad_(True)
        physics_problem.alpha.requires_grad_(True)
    params_to_clamp_lbfgs = setup_inverse_parameters(physics_problem)

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
    chunk_size = 500 if IS_1050TI else None
    
    def closure():
        optimizer_lbfgs.zero_grad()
        clamp_physical_parameters_(params_to_clamp_lbfgs)
        
        if chunk_size is None: #se lowvram splitta il calcolo dei gradienti
            loss, loss_dict = compute_pinn_loss(
                model, x_data=xy_int, y_data=obs_int,
                x_bc=xy_bc, y_bc=bc_targets,
                physics_problem=physics_problem, x_physics=xy_physics_full,
                lambda_data=lambda_data, lambda_bc=lambda_bc, lambda_physics=target_lambda_physics,
                mode=cfg.mode, variance_weights=cfg.variance_weights, active_bcs=None
            )
            loss.backward()
        else:
            loss, loss_dict = compute_chunked_gradients(
                model, physics_problem, xy_int, obs_int, xy_bc, bc_targets, xy_physics_full, 
                cfg.mode, cfg.variance_weights, lambda_data, lambda_bc, target_lambda_physics, chunk_size
            )
        
        if lbfgs_iter[0] % 20 == 0: #logging della loss
            history_entry = loss_dict.copy()
            history_entry.update({'weight_data': lambda_data, 'weight_bc': lambda_bc, 'weight_phys': target_lambda_physics})
            if getattr(physics_problem, 'inverse_mode', False):
                history_entry.update({
                    'param_etas': physics_problem.mu_s.item(),
                    'param_etap': physics_problem.mu_p.item(),
                    'param_lam': physics_problem.lam.item(),
                    'param_epsilon': physics_problem.eps.item(),
                    'param_alpha': physics_problem.alpha.item()
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
    xy_grid, T_exact_grid, X, Y = validation_grid
    X, Y = X.cpu(), Y.cpu()
    T_exact_grid = T_exact_grid.cpu()
    Ny_dom, Nx_dom = X.shape
    
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
            T_final = u_final.detach().cpu().reshape(Ny_dom, Nx_dom)
        else:
            out_final = model(xy_grid_val)
            T_final = out_final[:, 0].detach().cpu().reshape(Ny_dom, Nx_dom)
            
        del xy_grid_val # Puliamo immediatamente la VRAM
        
    # --- GESTIONE PUNTI DI PLOT ---
    internal_pts = xy_int if lambda_data > 0 else None
    boundary_pts = xy_bc if lambda_bc > 0 else None
    if target_lambda_physics > 0:
        xy_physics_full = xy_int if lambda_data > 0 else collocation_points
    else:
        xy_physics_full = None

    # --- PLOT COMPARATIVO PRINCIPALE ---
    final_path = os.path.join(final_dir, 'VEfinal_result.png')
    plot2D_final_result(X, Y, T_exact_grid, T_final, cfg.epochs, save_path=final_path, 
                        internal_points=internal_pts, boundary_points=boundary_pts, 
                        physics_points=xy_physics_full, val_label=cfg.val_label)
    
    # --- PLOT MULTI-CAMPO VISCOELASTICO ---
    if hasattr(physics_problem, 'get_velocity') and stress_exact_grids is not None:
        # Riutilizziamo le variabili già estratte nel blocco precedente!
        fields_pred = {
            'u': T_final, # T_final è già la nostra u
            'p': p_final.detach().cpu().reshape(Ny_dom, Nx_dom),
            'tau_xx': out_final[:, 2].detach().cpu().reshape(Ny_dom, Nx_dom),
            'tau_xy': out_final[:, 3].detach().cpu().reshape(Ny_dom, Nx_dom),
            'tau_yy': out_final[:, 4].detach().cpu().reshape(Ny_dom, Nx_dom),
        }
        
        fields_exact = {
            'u': T_exact_grid,
            'p': stress_exact_grids.get('p', torch.zeros_like(T_exact_grid)).cpu(),
            'tau_xx': stress_exact_grids.get('tau_xx', torch.zeros_like(T_exact_grid)).cpu(),
            'tau_xy': stress_exact_grids.get('tau_xy', torch.zeros_like(T_exact_grid)).cpu(),
            'tau_yy': stress_exact_grids.get('tau_yy', torch.zeros_like(T_exact_grid)).cpu(),
        }
        
        visco_final_path = os.path.join(final_dir, 'VE_viscoelastic_fields.png')
        plot2D_viscoelastic_final(X, Y, fields_pred, fields_exact, cfg.epochs,
                                  save_path=visco_final_path, internal_points=internal_pts, 
                                  boundary_points=boundary_pts, physics_points=xy_physics_full)
    
    # --- GIF E PULIZIA ---
    if plot_files:
        gif_path = os.path.join(final_dir, 'VEtraining_evolution.gif')
        save_gif_PIL(gif_path, plot_files, fps=3, loop=1, delete_files=True)
    
    shutil.rmtree(plots_dir, ignore_errors=True)
    
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
