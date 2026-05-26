import torch
import torch.nn as nn

def compute_chunked_gradients(model, physics_problem, xy_int, T_int, xy_bc, T_bc_tuple, xy_physics_full, 
                              mode, variance_weights, lambda_data, lambda_bc, lambda_physics, chunk_size):
    """
    Gestisce l'accumulo dei gradienti a blocchi (chunking) per VRAM limitate.
    Isola la logica matematica dal loop di ottimizzazione principale.
    """
    cur_dtype = next(model.parameters()).dtype
    cur_device = next(model.parameters()).device
    
    total_loss_val = 0.0
    loss_dict = {'data_loss': 0.0, 'bc_loss': 0.0, 'pde_loss': 0.0, 'total_loss': 0.0}
    
    # 1. Data Loss Chunking
    if xy_int is not None and T_int is not None and xy_int.numel() > 0:
        N_data = xy_int.shape[0]
        scale_u = variance_weights.get('u', 1.0) if (mode == 'semi_inverse' and variance_weights is not None) else 1.0
        scale_v = variance_weights.get('v', 1.0) if (mode == 'semi_inverse' and variance_weights is not None) else 1.0
        
        for i in range(0, N_data, chunk_size):
            x_data_c = xy_int[i : i + chunk_size]
            y_data_c = T_int[i : i + chunk_size]
            
            if lambda_data == 0:
                with torch.no_grad():
                    if mode == 'semi_inverse' and physics_problem is not None:
                        u_pred_c, v_pred_c, _, _ = physics_problem.get_velocity(model, x_data_c)
                        u_obs_c = y_data_c[:, 0:1]
                        v_obs_c = y_data_c[:, 1:2]
                        loss_u_c = torch.nn.functional.mse_loss(u_pred_c, u_obs_c, reduction='sum') / (N_data * scale_u)
                        loss_v_c = torch.nn.functional.mse_loss(v_pred_c, v_obs_c, reduction='sum') / (N_data * scale_v)
                        loss_c = 0.5 * (loss_u_c + loss_v_c)
                    else:
                        y_pred_c = model(x_data_c)
                        num_features = T_int.shape[1]
                        loss_c = torch.nn.functional.mse_loss(y_pred_c, y_data_c, reduction='sum') / (N_data * num_features)
            else:
                if mode == 'semi_inverse' and physics_problem is not None:
                    u_pred_c, v_pred_c, _, _ = physics_problem.get_velocity(model, x_data_c)
                    u_obs_c = y_data_c[:, 0:1]
                    v_obs_c = y_data_c[:, 1:2]
                    loss_u_c = torch.nn.functional.mse_loss(u_pred_c, u_obs_c, reduction='sum') / (N_data * scale_u)
                    loss_v_c = torch.nn.functional.mse_loss(v_pred_c, v_obs_c, reduction='sum') / (N_data * scale_v)
                    loss_c = 0.5 * (loss_u_c + loss_v_c)
                else:
                    y_pred_c = model(x_data_c)
                    num_features = T_int.shape[1]
                    loss_c = torch.nn.functional.mse_loss(y_pred_c, y_data_c, reduction='sum') / (N_data * num_features)
                
            loss_dict['data_loss'] += loss_c.item()
            if lambda_data > 0:
                (lambda_data * loss_c).backward()
                
        total_loss_val += lambda_data * loss_dict['data_loss']
        
    # 2. Boundary Loss (Nessun chunking necessario)
    if physics_problem is not None and xy_bc is not None and T_bc_tuple is not None and xy_bc.numel() > 0:
        bc_loss_val = physics_problem.boundary_loss(model, xy_bc, T_bc_tuple, variance_weights=variance_weights, active_bcs=None)
        loss_dict['bc_loss'] = bc_loss_val.item()
        total_loss_val += lambda_bc * bc_loss_val.item()
        if lambda_bc > 0:
            (lambda_bc * bc_loss_val).backward()
            
    # 3. PDE Loss (Physics Residual) Chunking
    if physics_problem is not None and xy_physics_full is not None and xy_physics_full.numel() > 0:
        N_phys = xy_physics_full.shape[0]
        weights = physics_problem.pde_weights
        w_m = weights.get('momentum', 10.0)
        w_c = weights.get('constitutive', 1.0)

        vw = variance_weights if variance_weights is not None else {}
        v_u = max(vw.get('u', 1.0), 1e-8)
        v_v = max(vw.get('v', 1.0), 1e-8)
        v_txx = max(vw.get('txx', 1.0), 1e-8)
        v_tyy = max(vw.get('tyy', 1.0), 1e-8)
        v_txy = max(vw.get('txy', 1.0), 1e-8)
        
        for i in range(0, N_phys, chunk_size):
            x_phys_c = xy_physics_full[i : i + chunk_size]
            if not x_phys_c.requires_grad:
                x_phys_c = x_phys_c.clone().requires_grad_(True)
                
            f_u, f_v, f_txx, f_tyy, f_txy = physics_problem.compute_residuals(model, x_phys_c)
            
            loss_u_c = (f_u ** 2).sum() / (N_phys * v_u)
            loss_v_c = (f_v ** 2).sum() / (N_phys * v_v)
            loss_m_c = loss_u_c + loss_v_c
            
            loss_txx_c = (f_txx ** 2).sum() / (N_phys * v_txx)
            loss_tyy_c = (f_tyy ** 2).sum() / (N_phys * v_tyy)
            loss_txy_c = (f_txy ** 2).sum() / (N_phys * v_txy)
            loss_c_c = loss_txx_c + loss_tyy_c + loss_txy_c
            
            loss_c = w_m * loss_m_c + w_c * loss_c_c
            loss_dict['pde_loss'] += loss_c.item()
            if lambda_physics > 0:
                (lambda_physics * loss_c).backward()
                
        total_loss_val += lambda_physics * loss_dict['pde_loss']
        
    loss_dict['total_loss'] = total_loss_val
    return torch.tensor(total_loss_val, device=cur_device, dtype=cur_dtype), loss_dict
