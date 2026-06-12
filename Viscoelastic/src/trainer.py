import torch
import torch.nn as nn
import numpy as np
import os
import sys
import shutil
from tqdm import tqdm

# Ensure func can be imported from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from func.graphic_func import generate_epoch_diagnostic_plot, generate_final_training_plots
from func.history_tracker import TrainingHistory
from func.hardware_utils import IS_1050TI

from .models import FCN, ViscoelasticCombinedModel, get_activation_name, format_layers_name, initialize_last_layer_zero
from .config import TrainingConfig, set_model_trainable, set_physics_trainable, _get_scheduler

def _sample_minibatch(xy, targets, batch_size, device):
    if batch_size is None or batch_size >= xy.shape[0]:
        return xy, targets
    idx = torch.randperm(xy.shape[0], device=device)[:batch_size]
    if targets is None:
        return xy[idx], None
    if isinstance(targets, tuple):
        return xy[idx], tuple(t[idx] for t in targets)
    return xy[idx], targets[idx]

def _sample_boundary_groups(xy_bc, target_bc, boundary_metadata, batch_size, device):
    if batch_size is None or boundary_metadata is None or len(boundary_metadata) == 0:
        return xy_bc, target_bc, boundary_metadata
        
    dir_target, neu_target, normals = target_bc
    sampled_xy = []
    sampled_dir = []
    sampled_neu = []
    sampled_norm = []
    new_metadata = []
    
    start_idx = 0
    total_original = sum(M for _, M in boundary_metadata)
    
    for g_name, M in boundary_metadata:
        end_idx = start_idx + M
        if total_original > 0:
            g_batch_size = max(1, int(round(M * batch_size / total_original)))
            g_batch_size = min(M, g_batch_size)
        else:
            g_batch_size = M
            
        if g_batch_size < M:
            idx = torch.randperm(M, device=device)[:g_batch_size]
            sampled_xy.append(xy_bc[start_idx:end_idx][idx])
            sampled_dir.append(dir_target[start_idx:end_idx][idx])
            sampled_neu.append(neu_target[start_idx:end_idx][idx])
            sampled_norm.append(normals[start_idx:end_idx][idx])
            new_metadata.append((g_name, g_batch_size))
        else:
            sampled_xy.append(xy_bc[start_idx:end_idx])
            sampled_dir.append(dir_target[start_idx:end_idx])
            sampled_neu.append(neu_target[start_idx:end_idx])
            sampled_norm.append(normals[start_idx:end_idx])
            new_metadata.append((g_name, M))
            
        start_idx = end_idx
        
    return (
        torch.cat(sampled_xy, dim=0),
        (torch.cat(sampled_dir, dim=0), torch.cat(sampled_neu, dim=0), torch.cat(sampled_norm, dim=0)),
        new_metadata
    )

def _run_adam_phase(model, physics_problem, cfg, data_internal, data_boundary, validation_grid, collocation_points, loss_history, plots_dir):
    _device = next(model.parameters()).device
    epochs = cfg.epochs
    half_epochs = epochs+1
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
    warmup_epochs_1_start_active = int(epochs * 0.1) if staged_training else 0 #solo eps e alpha
    warmup_epochs_1_all_active = int(epochs * 0.6) if staged_training else 0 #anche etap e lambda
    warmup_epochs_2 = half_epochs + int(epochs * warmup_ratio) if staged_training else 0
    warmup_non_staged_eps_alpha = int(epochs * 0.1) if not staged_training else 0   # warmup minore (es. 800 epoche)
    warmup_non_staged_all = int(epochs * 0.25) if not staged_training else 0        # sblocco totale (es. 2000 epoche)
    
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
        current_active_bcs = ['u', 'v', 'tau_xx', 'tau_xy', 'tau_yy']
        set_physics_trainable(physics_problem, []) 
    else:
        set_model_trainable(model, ['psi', 'p', 'tau'])
        current_active_bcs = None
        # Partiamo con i parametri fisici congelati se siamo in inverse_mode
        set_physics_trainable(physics_problem, [] if getattr(physics_problem, 'inverse_mode', False) else ['mu_s', 'mu_p', 'lam'])

    optimizer, scheduler, _last_layer_trainable, trainable_params = _rebuild_optimizer(warmup_epochs_1_start_active if staged_training else epochs)
    alpha_dynamic = 0.9

    for epoch in pbar:
        if not staged_training and getattr(physics_problem, 'inverse_mode', False):
            if epoch == warmup_non_staged_eps_alpha:
                pass # eps e alpha rimangono fissati a 0
            elif epoch == warmup_non_staged_all:
                print(f"\n  [Warmup Stage 2] Attivazione di viscosità e lambda (eps, alpha fissati).")
                set_physics_trainable(physics_problem, ['mu_s', 'mu_p', 'lam'])
                optimizer, scheduler, _last_layer_trainable, trainable_params = _rebuild_optimizer(epochs - epoch)
            
        if staged_training and epoch == warmup_epochs_1_start_active:
            set_physics_trainable(physics_problem, ['eps', 'alpha'])
            optimizer, scheduler, _last_layer_trainable, trainable_params = _rebuild_optimizer(warmup_epochs_1_all_active - warmup_epochs_1_start_active)
            
        if staged_training and epoch == warmup_epochs_1_all_active:
            set_physics_trainable(physics_problem, ['eps', 'alpha'])
            optimizer, scheduler, _last_layer_trainable, trainable_params = _rebuild_optimizer(half_epochs - warmup_epochs_1_all_active)
            if cfg.dynamic_weighting:
                lambda_data = cfg.loss_weights.get('data', 1.0)
                lambda_bc = cfg.loss_weights.get('bc', 1.0)
                target_lambda_physics = cfg.loss_weights.get('physics', 1.0)
                print(f"  [Dynamic Weights] Reset a fine warmup (epoca {epoch}): data={lambda_data:.2f}, bc={lambda_bc:.2f}, phys={target_lambda_physics:.2f}")
        if staged_training and epoch == half_epochs:
            print(f"\n  [Staged Training] Fase 2: Dinamica (psi+p).")
            set_model_trainable(model, ['psi', 'p'])
            physics_problem.pde_weights = {'momentum': base_pde_weights.get('momentum', 10.0), 'constitutive': 0.0}
            current_active_bcs = ['u', 'v', 'p']
            set_physics_trainable(physics_problem, [])
            if cfg.dynamic_weighting:
                lambda_data = cfg.loss_weights.get('data', 1.0)
                lambda_bc = cfg.loss_weights.get('bc', 1.0)
                target_lambda_physics = cfg.loss_weights.get('physics', 1.0)
                print(f"  [Dynamic Weights] Reset a inizio Fase 2: data={lambda_data:.2f}, bc={lambda_bc:.2f}, phys={target_lambda_physics:.2f}")
            optimizer, scheduler, _last_layer_trainable, trainable_params = _rebuild_optimizer(warmup_epochs_2 - half_epochs)
        if staged_training and epoch == warmup_epochs_2:
            set_physics_trainable(physics_problem, [])
            optimizer, scheduler, _last_layer_trainable, trainable_params = _rebuild_optimizer(epochs - warmup_epochs_2)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        
        xb, yb = _sample_minibatch(xy_int, obs_int, cfg.minibatch_internal, _device)
        
        # Campionamento proporzionale e contiguo per gruppi geometrici del contorno
        xbc, ybc, epoch_metadata = _sample_boundary_groups(
            xy_bc, bc_targets, physics_problem._boundary_metadata, 
            cfg.minibatch_boundary, _device
        )
        
        orig_metadata = physics_problem._boundary_metadata
        physics_problem._boundary_metadata = epoch_metadata
        
        if target_lambda_physics > 0:
            xph = xb.clone().requires_grad_(True) if lambda_data > 0 else _sample_minibatch(collocation_points, None, cfg.minibatch_internal, _device)[0].clone().requires_grad_(True)
        else:
            xph = None

        loss, loss_dict = compute_pinn_loss(
            model, x_data=xb, y_data=yb, x_bc=xbc, y_bc=ybc, physics_problem=physics_problem, x_physics=xph,
            lambda_data=lambda_data, lambda_bc=lambda_bc, lambda_physics=target_lambda_physics,
            mode=cfg.mode, variance_weights=cfg.variance_weights, active_bcs=current_active_bcs,
            group_weights=cfg.group_weights
        )
        
        physics_problem._boundary_metadata = orig_metadata

        if cfg.dynamic_weighting and (epoch + 1) % cfg.update_weights_every == 0:
            if lambda_bc > 0 and 'bc_loss' in loss_dict and loss_dict['bc_loss'].requires_grad:
                g_bc = torch.autograd.grad(loss_dict['bc_loss'], _last_layer_trainable, retain_graph=True, allow_unused=True)
                valid_g_bc = [g.norm(2) for g in g_bc if g is not None]
                norm_bc = (sum(valid_g_bc) / len(valid_g_bc)).item() if valid_g_bc else 0.0
                loss_dict['grad_bc'] = norm_bc
                
                if target_lambda_physics > 0 and 'pde_loss' in loss_dict and loss_dict['pde_loss'].requires_grad:
                    g_ph = torch.autograd.grad(loss_dict['pde_loss'], _last_layer_trainable, retain_graph=True, allow_unused=True)
                    valid_g_ph = [g.norm(2) for g in g_ph if g is not None]
                    m_ph = (sum(valid_g_ph) / len(valid_g_ph)).item() if valid_g_ph else 0.0
                    loss_dict['grad_pde'] = m_ph
                    if m_ph > 1e-12:
                        ratio_ph = min(norm_bc / m_ph, 100.0)
                        target_lambda_physics = alpha_dynamic * target_lambda_physics + (1-alpha_dynamic) * ratio_ph * lambda_bc
                        target_lambda_physics = min(target_lambda_physics, 500.0)
                        
                if lambda_data > 0 and 'data_loss' in loss_dict and loss_dict['data_loss'].requires_grad:
                    g_dt = torch.autograd.grad(loss_dict['data_loss'], _last_layer_trainable, retain_graph=True, allow_unused=True)
                    valid_g_dt = [g.norm(2) for g in g_dt if g is not None]
                    m_dt = (sum(valid_g_dt) / len(valid_g_dt)).item() if valid_g_dt else 0.0
                    loss_dict['grad_data'] = m_dt
                    if m_dt > 1e-12:
                        ratio_dt = min(norm_bc / m_dt, 100.0)
                        lambda_data = alpha_dynamic * lambda_data + (1-alpha_dynamic) * ratio_dt * lambda_bc
                        lambda_data = min(lambda_data, 500.0)

        loss.backward(inputs=trainable_params)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        if getattr(physics_problem, 'inverse_mode', False):
            phys_params_clip = [p for p in physics_problem.parameters() if p.requires_grad]
            if phys_params_clip:
                torch.nn.utils.clip_grad_norm_(phys_params_clip, cfg.param_clip_norm)
        optimizer.step()
        if getattr(physics_problem, 'inverse_mode', False):
            with torch.no_grad():
                physics_problem.eps.clamp_(min=0.0)
                physics_problem.alpha.clamp_(min=0.0)
                physics_problem.mu_s.clamp_(min=1e-6)
                physics_problem.mu_p.clamp_(min=1e-6)
                physics_problem.lam.clamp_(min=1e-6)
        if cfg.lr_strategy in ['step_decay', 'cosine']: scheduler.step()

        history_entry = {k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in loss_dict.items()}
        history_entry.update({'weight_data': lambda_data, 'weight_bc': lambda_bc, 'weight_phys': target_lambda_physics})
        if getattr(physics_problem, 'inverse_mode', False):
            eff = physics_problem.get_logged_parameters()
            history_entry.update({'param_etas': eff['mu_s'], 'param_etap': eff['mu_p'], 'param_lam': eff['lam'], 'param_epsilon': eff['eps'], 'param_alpha': eff['alpha']})
        loss_history.update(epoch, history_entry, lr=optimizer.param_groups[0]['lr'])

        if (epoch + 1) % 100 == 0:
            pbar.set_postfix({
                'Loss': f"{loss.item():.2e}",
                'LR': f"{optimizer.param_groups[0]['lr']:.2e}"
            })
            if (epoch + 1) % 500 == 0:
                mean_txx = loss_dict.get('mean_abs_f_txx', torch.tensor(0.0)).item()
                mean_tyy = loss_dict.get('mean_abs_f_tyy', torch.tensor(0.0)).item()
                mean_txy = loss_dict.get('mean_abs_f_txy', torch.tensor(0.0)).item()
                pbar.write(f"Epoch {epoch+1:05d} | |f_txx|: {mean_txx:.3e} | |f_tyy|: {mean_tyy:.3e} | |f_txy|: {mean_txy:.3e}")
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
    # eps e alpha rimangono fissati
    set_physics_trainable(physics_problem, ['mu_s', 'mu_p', 'lam'])
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
        loss_accum = {'data_loss': 0.0, 'bc_loss': 0.0, 'pde_loss': 0.0, 'total_loss': 0.0}
        
        # 1. Data Loss Chunking
        if xy_int is not None and xy_int.numel() > 0:
            for i in range(0, xy_int.shape[0], c_size):
                xc = xy_int[i : i + c_size]
                yc = obs_int[i : i + c_size]
                cl, cd = compute_pinn_loss(
                    model,
                    x_data=xc,
                    y_data=yc,
                    x_bc=None,
                    y_bc=None,
                    x_physics=None,
                    physics_problem=physics_problem,
                    lambda_data=lambda_data,
                    lambda_bc=0.0,
                    lambda_physics=0.0,
                    mode=cfg.mode,
                    variance_weights=cfg.variance_weights,
                    group_weights=cfg.group_weights,
                    force_data_loss=(l_it[0] % 50 == 0)
                )
                chunk_weight = xc.shape[0] / xy_int.shape[0]
                loss_accum['data_loss'] += cd.get('data_loss', 0.0) * chunk_weight
                loss_scaled = cl * chunk_weight
                if loss_scaled.requires_grad:
                    loss_scaled.backward()
        
        # 2. Boundary Loss (No chunking needed for BC as it is typically small and wasn't chunked in the original code)
        if xy_bc is not None and xy_bc.numel() > 0:
            cl, cd = compute_pinn_loss(
                model,
                x_data=None,
                y_data=None,
                x_bc=xy_bc,
                y_bc=bc_targets,
                x_physics=None,
                physics_problem=physics_problem,
                lambda_data=0.0,
                lambda_bc=lambda_bc,
                lambda_physics=0.0,
                mode=cfg.mode,
                variance_weights=cfg.variance_weights,
                group_weights=cfg.group_weights
            )
            loss_accum['bc_loss'] = cd.get('bc_loss', 0.0)
            for k, v in cd.items():
                if k.startswith('loss_bc_'):
                    loss_accum[k] = v
            if cl.requires_grad:
                cl.backward()
                
        # 3. Physics Loss Chunking (xph_full)
        if target_lambda_physics > 0 and xph_full is not None and xph_full.numel() > 0:
            for i in range(0, xph_full.shape[0], c_size):
                xc = xph_full[i : i + c_size]
                cl, cd = compute_pinn_loss(
                    model,
                    x_data=None,
                    y_data=None,
                    x_bc=None,
                    y_bc=None,
                    x_physics=xc,
                    physics_problem=physics_problem,
                    lambda_data=0.0,
                    lambda_bc=0.0,
                    lambda_physics=target_lambda_physics,
                    mode=cfg.mode,
                    variance_weights=cfg.variance_weights,
                    group_weights=cfg.group_weights
                )
                chunk_weight = xc.shape[0] / xph_full.shape[0]
                loss_accum['pde_loss'] += cd.get('pde_loss', 0.0) * chunk_weight
                loss_scaled = cl * chunk_weight
                if loss_scaled.requires_grad:
                    loss_scaled.backward()
                    
        # Extract scalar values for logging
        data_loss_val = loss_accum['data_loss'].item() if isinstance(loss_accum['data_loss'], torch.Tensor) else loss_accum['data_loss']
        bc_loss_val = loss_accum['bc_loss'].item() if isinstance(loss_accum['bc_loss'], torch.Tensor) else loss_accum['bc_loss']
        pde_loss_val = loss_accum['pde_loss'].item() if isinstance(loss_accum['pde_loss'], torch.Tensor) else loss_accum['pde_loss']
        
        total_loss_val = lambda_data * data_loss_val + lambda_bc * bc_loss_val + target_lambda_physics * pde_loss_val
        loss_accum['total_loss'] = total_loss_val
        
        if l_it[0] % 50 == 0:
            history_entry = {k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in loss_accum.items()}
            history_entry.update({'weight_data': lambda_data, 'weight_bc': lambda_bc, 'weight_phys': target_lambda_physics})
            loss_history.update(cfg.epochs + l_it[0], history_entry, lr=1.0)
            
        l_it[0] += 1
        pbar.update(1)
        pbar.set_postfix({'Loss': f"{total_loss_val:.2e}"})
        
        dev = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        return torch.tensor(total_loss_val, device=dev, dtype=dtype)

    optimizer_lbfgs.step(closure)
    if getattr(physics_problem, 'inverse_mode', False):
        with torch.no_grad():
            physics_problem.eps.clamp_(min=0.0)
            physics_problem.alpha.clamp_(min=0.0)
            physics_problem.mu_s.clamp_(min=1e-6)
            physics_problem.mu_p.clamp_(min=1e-6)
            physics_problem.lam.clamp_(min=1e-6)
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
        generate_final_training_plots(final_dir, plots_dir, triang, Te.cpu(), u_f.detach().cpu().view(-1), p_f, tau_f, stress_exact_grids, plot_files, config.epochs, config.val_label, data_internal[0], data_boundary[0], collocation_points)
    
    loss_history.plot_losses(adam_epochs=config.epochs, save_path=os.path.join(final_dir, 'VE_loss_history.png'), experiment_name=config.experiment_name, smoothing_alpha=0.95)
    loss_history.plot_gradients(save_path=os.path.join(final_dir, 'VE_gradients_history.png'), experiment_name=config.experiment_name)
    return loss_history

def compute_pinn_loss(model, x_data, y_data, x_bc=None, y_bc=None, x_physics=None, physics_problem=None, lambda_data=1.0, lambda_bc=1.0, lambda_physics=1.0, mode='standard', variance_weights=None, force_data_loss=False, group_weights=None, **kwargs):
    loss_dict, total_loss = {}, 0.0
    if x_data is not None and x_data.numel() > 0:
        up, vp, pp, tp = physics_problem.get_velocity(model, x_data)
        
        loss_u = nn.MSELoss()(up, y_data[:,0:1]) / variance_weights.get('u', 1.0)
        loss_v = nn.MSELoss()(vp, y_data[:,1:2]) / variance_weights.get('v', 1.0)
        data_loss = 0.5 * (loss_u + loss_v)
        
        # Se abbiamo tutti i 6 campi (es. Goal 2: SoloData / comsol_full)
        if y_data.shape[1] >= 6:
            loss_p = nn.MSELoss()(pp, y_data[:,2:3]) / variance_weights.get('p', 1.0)
            loss_txx = nn.MSELoss()(tp[:,0:1], y_data[:,3:4]) / variance_weights.get('tau_xx', 1.0)
            loss_txy = nn.MSELoss()(tp[:,1:2], y_data[:,4:5]) / variance_weights.get('tau_xy', 1.0)
            loss_tyy = nn.MSELoss()(tp[:,2:3], y_data[:,5:6]) / variance_weights.get('tau_yy', 1.0)
            data_loss = (loss_u + loss_v + loss_p + loss_txx + loss_txy + loss_tyy) / 6.0
            
        loss_dict['data_loss'] = data_loss
        total_loss += lambda_data * data_loss
    
    if x_bc is not None and x_bc.numel() > 0:
        bc_loss, per_g = physics_problem.boundary_loss(model, x_bc, y_bc, None, kwargs.get('active_bcs'), group_weights)
        loss_dict['bc_loss'] = bc_loss
        loss_dict.update(per_g)
        total_loss += lambda_bc * bc_loss

    if x_physics is not None and lambda_physics > 0:
        f_u, f_v, f_txx, f_tyy, f_txy = physics_problem.compute_residuals(model, x_physics)
        loss_m = (f_u**2 + f_v**2).mean()
        loss_txx = (f_txx**2).mean()
        loss_tyy = (f_tyy**2).mean()
        loss_txy = (f_txy**2).mean()
        loss_c = loss_txx + loss_tyy + loss_txy
        
        w_mom = physics_problem.pde_weights.get('momentum', 10.0)
        w_const = physics_problem.pde_weights.get('constitutive', 1.0)
        pde_loss = w_mom * loss_m + w_const * loss_c
        
        loss_dict['pde_loss'] = pde_loss
        loss_dict['loss_txx'] = loss_txx
        loss_dict['loss_tyy'] = loss_tyy
        loss_dict['loss_txy'] = loss_txy
        loss_dict['loss_momentum'] = loss_m
        loss_dict['loss_constitutive'] = loss_c
        loss_dict['mean_abs_f_u'] = f_u.abs().mean()
        loss_dict['mean_abs_f_v'] = f_v.abs().mean()
        loss_dict['mean_abs_f_txx'] = f_txx.abs().mean()
        loss_dict['mean_abs_f_tyy'] = f_tyy.abs().mean()
        loss_dict['mean_abs_f_txy'] = f_txy.abs().mean()
        
        total_loss += lambda_physics * pde_loss
    else:
        dev = x_data.device if (x_data is not None and x_data.numel() > 0) else next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        loss_dict['pde_loss'] = torch.tensor(0.0, device=dev, dtype=dtype)
        
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
