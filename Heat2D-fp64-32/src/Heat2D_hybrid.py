import torch
import torch.nn as nn
import os
from tqdm import tqdm
from func.history_tracker import TrainingHistory, compute_pinn_loss
from func.graphic_func import plot2D_final_result

def train_hybrid_logic(
    model, data_internal, data_boundary, validation_grid,
    epochs=40000, physics_problem=None, physics_loss_fn=None, plots_dir='plots', final_dir='Results',
    collocation_points=None, lr_strategy='plateau', loss_weights=None, warmup_epochs=0,
    case_name="Run", dynamic_weighting=False, update_weights_every=100,
    use_staged_precision=True # Toggle per precisione progressiva (FP32 -> FP64)
):
    """
    Logica universale per Adam @ FP32 (ottimizzato TF32) -> L-BFGS @ FP64.
    Se use_staged_precision=True, massimizza la velocità iniziale.
    Se use_staged_precision=False, esegue l'intero training in FP64 puro.
    """
    xy_int, T_int = data_internal
    xy_bc, T_bc = data_boundary
    xy_grid, T_exact_grid, X, Y = validation_grid
    Nx_dom, Ny_dom = X.shape

    if plots_dir: os.makedirs(plots_dir, exist_ok=True)
    if final_dir: os.makedirs(final_dir, exist_ok=True)
    
    loss_history = TrainingHistory()
    if loss_weights is None: loss_weights = {'data': 1.0, 'bc': 1.0, 'physics': 1.0}
    
    lambda_data = loss_weights.get('data', 1.0)
    lambda_bc = loss_weights.get('bc', 1.0)
    lambda_phys_target = loss_weights.get('physics', 1.0)
    alpha_annealing = 0.9

    # --- SETUP PRECISIONE FASE 1 ---
    # Se use_staged_precision=True, usiamo FP32 per la Fase 1.
    # Altrimenti usiamo FP64 per tutto il training.
    phase1_precision = torch.float32 if use_staged_precision else torch.float64
    
    prec_name = "FP64" if not use_staged_precision else "FP32"

    # ABILITAZIONE TF32 (Solo se siamo in modalità Staged e su Ampere+)
    old_matmul_precision = torch.get_float32_matmul_precision()
    if use_staged_precision and torch.cuda.is_available():
        # 'high' abilita TF32 su Ampere (RTX 3080) aumentando la velocità di matmul
        torch.set_float32_matmul_precision('high')

    # --- PHASE 1: ADAM ---
    msg_stage = f"{prec_name} + TF32" if use_staged_precision else "Pure FP64"
    print(f"\n>>> [{case_name}] PHASE 1: Adam ({msg_stage}) - {epochs} epochs")
    model.to(phase1_precision)
    
    # Cast dati alla precisione di Fase 1
    xy_int_low, T_int_low = xy_int.to(phase1_precision), T_int.to(phase1_precision)
    xy_bc_low = xy_bc.to(phase1_precision) if xy_bc is not None else None
    T_bc_low = T_bc.to(phase1_precision) if T_bc is not None else None
    xy_phys_low = collocation_points.to(phase1_precision) if collocation_points is not None else None
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = None
    if lr_strategy == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1000, factor=0.5)

    pbar = tqdm(range(epochs), desc=f"{case_name} Adam ({prec_name})")
    for epoch in pbar:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        
        current_lambda_phys = 0.0 if epoch < warmup_epochs else lambda_phys_target
        
        loss, loss_dict = compute_pinn_loss(
            model, xy_int_low, T_int_low, xy_bc_low, T_bc_low,
            physics_problem=physics_problem,
            physics_loss_fn=physics_loss_fn,
            x_physics=xy_phys_low,
            lambda_data=lambda_data, 
            lambda_bc=lambda_bc, 
            lambda_physics=current_lambda_phys
        )
        
        # LOGICA PESI DINAMICI
        if dynamic_weighting and epoch >= warmup_epochs and (epoch + 1) % update_weights_every == 0:
            if xy_bc_low is not None and 'bc_loss' in loss_dict:
                loss_bc_pure = loss_dict['bc_loss']
                grads_bc = torch.autograd.grad(loss_bc_pure, model.parameters(), retain_graph=True, allow_unused=True)
                max_grad_bc = max([g.norm(2) for g in grads_bc if g is not None]).item() if any(g is not None for g in grads_bc) else 0.0
                
                if current_lambda_phys > 0 and 'pde_loss' in loss_dict:
                    loss_ph_pure = loss_dict['pde_loss']
                    grads_ph = torch.autograd.grad(loss_ph_pure, model.parameters(), retain_graph=True, allow_unused=True)
                    max_grad_ph = max([g.norm(2) for g in grads_ph if g is not None]).item() if any(g is not None for g in grads_ph) else 0.0
                    if max_grad_ph > 1e-12:
                        lambda_phys_target = alpha_annealing * lambda_phys_target + (1-alpha_annealing) * (max_grad_bc / max_grad_ph) * lambda_bc

                if lambda_data > 0 and 'data_loss' in loss_dict:
                    loss_dt_pure = loss_dict['data_loss']
                    grads_dt = torch.autograd.grad(loss_dt_pure, model.parameters(), retain_graph=True, allow_unused=True)
                    max_grad_dt = max([g.norm(2) for g in grads_dt if g is not None]).item() if any(g is not None for g in grads_dt) else 0.0
                    if max_grad_dt > 1e-12:
                        lambda_data = alpha_annealing * lambda_data + (1-alpha_annealing) * (max_grad_bc / max_grad_dt) * lambda_bc

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler: scheduler.step(loss.item())

        if (epoch + 1) % 500 == 0:
            loss_history.update(epoch, {k: v.item() for k, v in loss_dict.items()}, lr=optimizer.param_groups[0]['lr'])
            pbar.set_postfix({'Loss': f"{total_loss.item() if 'total_loss' in globals() else loss.item():.2e}"})

    # --- TRANSITION ---
    print(f"\n>>> [{case_name}] TRANSITION: Switching to FP64 for L-BFGS refinement...")
    
    # Ripristino precisione matmul originale (highest/standard)
    torch.set_float32_matmul_precision(old_matmul_precision)
    
    model.to(torch.float64)
    xy_int_64, T_int_64 = xy_int.to(torch.float64), T_int.to(torch.float64)
    xy_bc_64 = xy_bc.to(torch.float64) if xy_bc is not None else None
    T_bc_64 = T_bc.to(torch.float64) if T_bc is not None else None
    xy_phys_64 = collocation_points.to(torch.float64) if collocation_points is not None else None
    
    # --- PHASE 2: L-BFGS @ FP64 ---
    optimizer_lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=5000, line_search_fn="strong_wolfe")
    
    lbfgs_iter = [0]
    def closure():
        optimizer_lbfgs.zero_grad()
        loss, loss_dict = compute_pinn_loss(
            model, xy_int_64, T_int_64, xy_bc_64, T_bc_64,
            physics_problem=physics_problem,
            physics_loss_fn=physics_loss_fn,
            x_physics=xy_phys_64,
            lambda_data=lambda_data, lambda_bc=lambda_bc, lambda_physics=lambda_phys_target
        )
        loss.backward()
        if lbfgs_iter[0] % 10 == 0:
            loss_history.update(epochs + lbfgs_iter[0], {k: v.item() for k, v in loss_dict.items()}, lr=1.0)
        lbfgs_iter[0] += 1
        return loss

    optimizer_lbfgs.step(closure)
    
    model.eval()
    with torch.no_grad():
        T_final = model(xy_grid.to(torch.float64)).reshape(Nx_dom, Ny_dom)
    if final_dir:
        plot2D_final_result(X, Y, T_exact_grid, T_final, epochs + lbfgs_iter[0], 
                           save_path=os.path.join(final_dir, f'{case_name}_final.png'))
    
    return loss_history
