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
    group_weights: dict = field(default_factory=lambda: {'Inlet': 1.0, 'Walls': 1.0, 'Outlet': 1.0})
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
        param_groups = [{'params': net_params, 'lr': cfg.base_lr}, {'params': phys_params, 'lr': cfg.base_lr * getattr(cfg, 'param_lr_factor', 0.1)}]
        opt = torch.optim.Adam(param_groups, eps=cfg.adam_eps)
        sch = _get_scheduler(opt, cfg.lr_strategy, steps_remaining if steps_remaining > 0 else 1)
        last_layer = []
        for net in [model.model_psi, model.model_p, model.model_tau]:
            if hasattr(net, 'fcs') and len(net.fcs) > 0: last_layer.extend([p for p in net.fcs[-1].parameters() if p.requires_grad])
        if not last_layer: last_layer = net_params + phys_params
        return opt, sch, last_layer, net_params + phys_params
    
    if staged_training:
        print(f"\n  [Staged Training] Fase 1: Cinematica e Reologia.")
        set_model_trainable(model, ['psi', 'tau'])
        physics_problem.pde_weights = {'momentum': 0.0, 'constitutive': base_pde_weights.get('constitutive', 1.0)} 
        current_active_bcs = ['u', 'v', 'txx', 'txy', 'tyy']
        set_physics_trainable(physics_problem, []) 
    else:
        set_model_trainable(model, ['psi', 'p', 'tau'])
        current_active_bcs = None
        set_physics_trainable(physics_problem, ['mu_s', 'mu_p', 'lam', 'eps', 'alpha'])

    optimizer, scheduler, _last_layer_trainable, trainable_params = _rebuild_optimizer(warmup_epochs_1 if staged_training else epochs)
    alpha_dynamic = 0.9

    for epoch in pbar:
        if staged_training and epoch == warmup_epochs_1:
            set_physics_trainable(physics_problem, ['mu_p', 'lam', 'eps', 'alpha'])
            optimizer, scheduler, _last_layer_trainable, trainable_params = _rebuild_optimizer(half_epochs - warmup_epochs_1)
        if staged_training and epoch == half_epochs:
            print(f"\n  [Staged Training] Fase 2: Dinamica (psi+p).")
            set_model_trainable(model, ['psi', 'p'])
            physics_problem.pde_weights = {'momentum': base_pde_weights.get('momentum', 10.0), 'constitutive': 0.0}
            current_active_bcs = ['u', 'v', 'p']
            set_physics_trainable(physics_problem, [])
            optimizer, scheduler, _last_layer_trainable, trainable_params = _rebuild_optimizer(warmup_epochs_2 - half_epochs)
        if staged_training and epoch == warmup_epochs_2:
            set_physics_trainable(physics_problem, ['mu_s'])
            optimizer, scheduler, _last_layer_trainable, trainable_params = _rebuild_optimizer(epochs - warmup_epochs_2)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        
        xb, yb = _sample_minibatch(xy_int, obs_int, cfg.minibatch_internal, _device)
        xbc, ybc = _sample_minibatch(xy_bc, bc_targets, cfg.minibatch_boundary, _device)
        xph = xb.clone().requires_grad_(True) if lambda_data > 0 else _sample_minibatch(collocation_points, None, cfg.minibatch_internal, _device)[0].clone().requires_grad_(True)

        loss, loss_dict = compute_pinn_loss(
            model, x_data=xb, y_data=yb, x_bc=xbc, y_bc=ybc, physics_problem=physics_problem, x_physics=xph,
            lambda_data=lambda_data, lambda_bc=lambda_bc, lambda_physics=target_lambda_physics,
            mode=cfg.mode, variance_weights=cfg.variance_weights, active_bcs=current_active_bcs,
            group_weights=cfg.group_weights
        )

        if cfg.dynamic_weighting and (epoch + 1) % cfg.update_weights_every == 0:
            if lambda_bc > 0 and 'bc_loss' in loss_dict and loss_dict['bc_loss'].requires_grad:
                g_bc = torch.autograd.grad(loss_dict['bc_loss'], _last_layer_trainable, retain_graph=True, allow_unused=True)
                norm_bc = max([g.norm(2) for g in g_bc if g is not None]).item() if any(g is not None for g in g_bc) else 0.0
                if target_lambda_physics > 0 and 'pde_loss' in loss_dict and loss_dict['pde_loss'].requires_grad:
                    g_ph = torch.autograd.grad(loss_dict['pde_loss'], _last_layer_trainable, retain_graph=True, allow_unused=True)
                    m_ph = max([g.norm(2) for g in g_ph if g is not None]).item() if any(g is not None for g in g_ph) else 0.0
                    if m_ph > 1e-12: target_lambda_physics = alpha_dynamic * target_lambda_physics + (1-alpha_dynamic) * (norm_bc / m_ph) * lambda_bc
                if lambda_data > 0 and 'data_loss' in loss_dict and loss_dict['data_loss'].requires_grad:
                    g_dt = torch.autograd.grad(loss_dict['data_loss'], _last_layer_trainable, retain_graph=True, allow_unused=True)
                    m_dt = max([g.norm(2) for g in g_dt if g is not None]).item() if any(g is not None for g in g_dt) else 0.0
                    if m_dt > 1e-12: lambda_data = alpha_dynamic * lambda_data + (1-alpha_dynamic) * (norm_bc / m_dt) * lambda_bc

        loss.backward(inputs=trainable_params)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        optimizer.step()
        if cfg.lr_strategy in ['step_decay', 'cosine']: scheduler.step()

        history_entry = {k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in loss_dict.items()}
        history_entry.update({'weight_data': lambda_data, 'weight_bc': lambda_bc, 'weight_phys': target_lambda_physics})
        if getattr(physics_problem, 'inverse_mode', False):
            eff = physics_problem.get_logged_parameters()
            history_entry.update({'param_etas': eff['mu_s'], 'param_etap': eff['mu_p'], 'param_lam': eff['lam'], 'param_epsilon': eff['eps'], 'param_alpha': eff['alpha']})
        loss_history.update(epoch, history_entry, lr=optimizer.param_groups[0]['lr'])

        if (epoch + 1) % 100 == 0:
            pbar.set_postfix({'Loss': f"{loss.item():.2e}"})
            if (epoch + 1) % cfg.plot_every == 0:
                generate_epoch_diagnostic_plot(model, physics_problem, xy_grid, T_exact_grid, triang, epoch, plots_dir, cfg.plot_every, cfg.val_label, plot_files)
    pbar.close()
    return loss_history, plot_files, lambda_data, lambda_bc, target_lambda_physics, base_pde_weights

def _run_lbfgs_phase(model, physics_problem, cfg, data_internal, data_boundary, collocation_points, loss_history, lambda_data, lambda_bc, target_lambda_physics, base_pde_weights):
    _dtype = next(model.parameters()).dtype
    xy_int, obs_int = data_internal
    xy_bc, dir_bc, neu_bc, norm_bc = data_boundary
    bc_targets = (dir_bc, neu_bc, norm_bc)
    
    print("\n  [Staged Training] Fase 3: Raffinamento L-BFGS.")
    set_model_trainable(model, ['psi', 'p', 'tau'])
    physics_problem.pde_weights = base_pde_weights
    
    if cfg.precision_mode == 'staged':
        torch.set_default_dtype(torch.float64)
        model.double(); physics_problem.double()
        xy_int, obs_int, xy_bc = xy_int.double(), obs_int.double(), xy_bc.double()
        bc_targets = tuple(t.double() for t in bc_targets)
        xph_full = xy_int.clone().requires_grad_(True) if lambda_data > 0 else collocation_points.double().requires_grad_(True)
    else:
        xph_full = xy_int.clone().requires_grad_(True) if lambda_data > 0 else collocation_points.clone().requires_grad_(True)

    optimizer_lbfgs = torch.optim.LBFGS(list(model.parameters()) + [p for p in physics_problem.parameters() if p.requires_grad], lr=1.0, max_iter=cfg.max_lbfgs_iters, tolerance_grad=1e-9, tolerance_change=1e-12, history_size=50 if IS_1050TI else 300, line_search_fn="strong_wolfe")

    l_it = [0]
    pbar = tqdm(total=cfg.max_lbfgs_iters, desc="Training VE (L-BFGS)", mininterval=2.0)
    c_size = 500 if IS_1050TI else 2000
    
    def closure():
        optimizer_lbfgs.zero_grad()
        loss, loss_dict = compute_pinn_loss(model, x_data=xy_int, y_data=obs_int, x_bc=xy_bc, y_bc=bc_targets, physics_problem=physics_problem, x_physics=xph_full, lambda_data=lambda_data, lambda_bc=lambda_bc, lambda_physics=target_lambda_physics, mode=cfg.mode, variance_weights=cfg.variance_weights, active_bcs=None, force_data_loss=(l_it[0]%50==0), chunk_size=c_size, group_weights=cfg.group_weights)
        if l_it[0] % 50 == 0:
            history_entry = {k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in loss_dict.items()}
            history_entry.update({'weight_data': lambda_data, 'weight_bc': lambda_bc, 'weight_phys': target_lambda_physics})
            loss_history.update(cfg.epochs + l_it[0], history_entry, lr=1.0)
        l_it[0] += 1; pbar.update(1); pbar.set_postfix({'Loss': f"{loss.item():.2e}"})
        return loss

    optimizer_lbfgs.step(closure)
    pbar.close()
    if cfg.precision_mode == 'staged':
        torch.set_default_dtype(_dtype); model.to(_dtype); physics_problem.to(_dtype)
    return loss_history

def train_ViscoelasticPINN(model, config, data_internal, data_boundary, validation_grid, physics_problem, collocation_points, plots_dir, final_dir, stress_exact_grids=None):
    os.makedirs(plots_dir, exist_ok=True); os.makedirs(final_dir, exist_ok=True)
    loss_history = TrainingHistory()
    loss_history, plot_files, ld, lbc, lp, bpw = _run_adam_phase(model, physics_problem, config, data_internal, data_boundary, validation_grid, collocation_points, loss_history, plots_dir)
    loss_history = _run_lbfgs_phase(model, physics_problem, config, data_internal, data_boundary, collocation_points, loss_history, ld, lbc, lp, bpw)
    
    print("Training completato. Generazione artifacts...")
    model.eval()
    with torch.set_grad_enabled(True):
        xg, Te, triang = validation_grid
        u_f, v_f, p_f, tau_f = physics_problem.get_velocity(model, xg.clone().requires_grad_(True))
        generate_final_training_plots(final_dir, plots_dir, triang, Te.cpu(), u_f.detach().cpu().view(-1), p_f, model(xg), stress_exact_grids, plot_files, config.epochs, config.val_label)
    
    loss_history.plot_losses(adam_epochs=config.epochs, save_path=os.path.join(final_dir, 'VE_loss_history.png'), experiment_name=config.experiment_name, smoothing_alpha=0.95)
    return loss_history

def compute_pinn_loss(model, x_data, y_data, x_bc=None, y_bc=None, x_physics=None, physics_problem=None, lambda_data=1.0, lambda_bc=1.0, lambda_physics=1.0, mode='standard', variance_weights=None, force_data_loss=False, chunk_size=None, group_weights=None, **kwargs):
    if chunk_size is not None:
        loss_accum = {'data_loss': 0.0, 'bc_loss': 0.0, 'pde_loss': 0.0, 'total_loss': 0.0}
        dev, dtype = next(model.parameters()).device, next(model.parameters()).dtype
        
        if x_data is not None and x_data.numel() > 0:
            for i in range(0, x_data.shape[0], chunk_size):
                xc, yc = x_data[i:i+chunk_size], y_data[i:i+chunk_size]
                cl, cd = compute_pinn_loss(model, xc, yc, None, None, None, physics_problem, lambda_data, 0, 0, mode, variance_weights, force_data_loss, None, group_weights, **kwargs)
                loss_accum['data_loss'] += cd.get('data_loss', 0.0) * (xc.shape[0]/x_data.shape[0])
                if cl.requires_grad: (cl * (xc.shape[0]/x_data.shape[0])).backward()
        
        if x_bc is not None and x_bc.numel() > 0:
            cl, cd = compute_pinn_loss(model, None, None, x_bc, y_bc, None, physics_problem, 0, lambda_bc, 0, mode, variance_weights, False, None, group_weights, **kwargs)
            loss_accum['bc_loss'] = cd.get('bc_loss', 0.0)
            for k, v in cd.items(): 
                if k.startswith('loss_bc_'): loss_accum[k] = v
            if cl.requires_grad: cl.backward()

        if x_physics is not None and x_physics.numel() > 0:
            for i in range(0, x_physics.shape[0], chunk_size):
                xc = x_physics[i:i+chunk_size]
                cl, cd = compute_pinn_loss(model, None, None, None, None, xc, physics_problem, 0, 0, lambda_physics, mode, variance_weights, False, None, group_weights, **kwargs)
                loss_accum['pde_loss'] += cd.get('pde_loss', 0.0) * (xc.shape[0]/x_physics.shape[0])
                if cl.requires_grad: (cl * (xc.shape[0]/x_physics.shape[0])).backward()

        loss_accum['total_loss'] = lambda_data*loss_accum['data_loss'] + lambda_bc*loss_accum['bc_loss'] + lambda_physics*loss_accum['pde_loss']
        return torch.tensor(loss_accum['total_loss'], device=dev, dtype=dtype), loss_accum

    loss_dict, total_loss = {}, 0.0
    if x_data is not None and x_data.numel() > 0:
        up, vp, pp, tp = physics_problem.get_velocity(model, x_data)
        l_dt = 0.5 * (nn.MSELoss()(up, y_data[:,0:1])/variance_weights.get('u',1.0) + nn.MSELoss()(vp, y_data[:,1:2])/variance_weights.get('v',1.0))
        loss_dict['data_loss'] = l_dt; total_loss += lambda_data * l_dt
    
    if x_bc is not None and x_bc.numel() > 0:
        lb, per_g = physics_problem.boundary_loss(model, x_bc, y_bc, variance_weights, kwargs.get('active_bcs'), group_weights)
        loss_dict['bc_loss'] = lb; loss_dict.update(per_g); total_loss += lambda_bc * lb

    if x_physics is not None:
        lp = physics_problem.residual(model, x_physics, variance_weights=variance_weights)
        loss_dict['pde_loss'] = lp; total_loss += lambda_physics * lp
        
    loss_dict['total_loss'] = total_loss
    return total_loss, loss_dict

def compute_viscoelastic_metrics(model, physics_problem, xy_grid_flat, fields_exact_flat, Ny_dom=None, Nx_dom=None):
    model.eval(); dtype = next(model.parameters()).dtype
    xi = xy_grid_flat.clone().to(dtype).requires_grad_(True)
    with torch.set_grad_enabled(True):
        up, vp, pp, tp = physics_problem.get_velocity(model, xi)
    preds = {'u': up.detach().cpu().view(-1), 'p': pp.detach().cpu().view(-1), 'tau_xx': tp[:,0].detach().cpu().view(-1), 'tau_xy': tp[:,1].detach().cpu().view(-1), 'tau_yy': tp[:,2].detach().cpu().view(-1)}
    metrics = {}
    for fn, pf in preds.items():
        ex = fields_exact_flat.get(fn)
        if ex is None: metrics[fn] = (0.0, 0.0); continue
        tf = ex.view(-1).cpu().to(pf.dtype)
        l2 = (torch.norm(pf - tf, 2) / torch.norm(tf, 2)).item() if torch.norm(tf,2) > 1e-10 else 0.0
        metrics[fn] = (l2, 0.0) # Max error non calcolato per brevità
    return metrics
