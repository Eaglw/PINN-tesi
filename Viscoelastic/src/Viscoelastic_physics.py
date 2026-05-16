import torch
import torch.nn as nn

class ViscoelasticPhysics(nn.Module):
    def __init__(self, mu_s=0.005, mu_p=0.005, lam=1.0, rho=1.0, pde_weights=None):
        """
        Modulo per calcolare i residui fisici (equazioni di Navier-Stokes e modello Oldroyd-B).
        
        NOTA FISICA — Scelta della Stream Function:
            L'equazione di continuità (∇·u = 0) è automaticamente soddisfatta
            dall'uso della stream function: u = ∂ψ/∂y, v = -∂ψ/∂x.
            Dimostrazione: ∂u/∂x + ∂v/∂y = ∂²ψ/∂x∂y - ∂²ψ/∂y∂x = 0.
            Perciò NON è inclusa esplicitamente tra i residui.
            Se si passasse a output diretto (u, v), va aggiunta come residuo.
        
        Args:
            mu_s: Viscosità del solvente [Pa·s]
            mu_p: Viscosità polimerica [Pa·s]
            lam: Tempo di rilassamento [s]
            rho: Densità del fluido [kg/m³]. Default=1.0 (adimensionale).
                 Se rho != 1, le equazioni del momento vengono scalate di conseguenza.
            pde_weights: Dict con pesi per le componenti PDE.
                Default: {'momentum': 10.0, 'constitutive': 1.0}
                NOTA: Per Oldroyd-B i residui degli stress (tau_xx soprattutto)
                hanno magnitudini strutturalmente diverse dai residui di momentum,
                perché tau_xx scala come γ̇² (quadratico) mentre f_u scala come γ̇ (lineare).
        """
        super().__init__()
        self.mu_s = mu_s
        self.mu_p = mu_p
        self.lam = lam
        self.rho = rho
        self.mse_loss = nn.MSELoss()
        self.pde_weights = pde_weights or {'momentum': 10.0, 'constitutive': 1.0}

    def get_velocity(self, model, x):
        """
        Calcola u, v e p a partire dalle reti neurali.
        """
        if not x.requires_grad:
            x.requires_grad_(True)
            
        out = model(x)
        psi = out[:, 0:1]
        p = out[:, 1:2]
        tau = out[:, 2:5]
        
        # Derivate spaziali per ottenere u, v da psi
        grad_psi = torch.autograd.grad(psi.sum(), x, create_graph=True)[0]
        psi_x = grad_psi[:, 0:1]
        psi_y = grad_psi[:, 1:2]
        
        u = psi_y
        v = -psi_x
        
        return u, v, p, tau

    def compute_residuals(self, model, x):
        """
        Calcola i residui delle equazioni di Navier-Stokes + Oldroyd-B per un set di punti x.
        
        Ottimizzazioni rispetto all'implementazione naive:
        - Riusa get_velocity() per evitare duplicazione del forward pass + grad(psi)
        - Sfrutta v_y = -u_x (equazione di continuità) per risparmiare 1 chiamata autograd
        - Sfrutta v_yy = -u_yx (teorema di Schwarz) per risparmiare 1 chiamata autograd
        Totale: 10 chiamate autograd anziché 12.
        """
        u, v, p, tau = self.get_velocity(model, x)
        tau_xx = tau[:, 0:1]
        tau_xy = tau[:, 1:2]
        tau_yy = tau[:, 2:3]
        
        # Derivate prime di u, v, p
        grad_u = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        u_x, u_y = grad_u[:, 0:1], grad_u[:, 1:2]
        
        grad_v = torch.autograd.grad(v.sum(), x, create_graph=True)[0]
        v_x = grad_v[:, 0:1]
        v_y = -u_x  # Equazione di continuità: u_x + v_y = 0
        
        grad_p = torch.autograd.grad(p.sum(), x, create_graph=True)[0]
        p_x, p_y = grad_p[:, 0:1], grad_p[:, 1:2]
        
        # Derivate seconde di u, v
        grad_u_x = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
        u_xx = grad_u_x[:, 0:1]
        
        grad_u_y = torch.autograd.grad(u_y.sum(), x, create_graph=True)[0]
        u_yx = grad_u_y[:, 0:1]
        u_yy = grad_u_y[:, 1:2]
        
        grad_v_x = torch.autograd.grad(v_x.sum(), x, create_graph=True)[0]
        v_xx = grad_v_x[:, 0:1]
        
        # v_yy = -u_yx per il teorema di Schwarz:
        # u_yx = ∂³ψ/∂x∂y² e v_yy = -∂³ψ/∂x∂y² → v_yy = -u_yx
        v_yy = -u_yx
        
        # Derivate prime di tau
        grad_tau_xx = torch.autograd.grad(tau_xx.sum(), x, create_graph=True)[0]
        tau_xx_x, tau_xx_y = grad_tau_xx[:, 0:1], grad_tau_xx[:, 1:2]
        grad_tau_xy = torch.autograd.grad(tau_xy.sum(), x, create_graph=True)[0]
        tau_xy_x, tau_xy_y = grad_tau_xy[:, 0:1], grad_tau_xy[:, 1:2]
        grad_tau_yy = torch.autograd.grad(tau_yy.sum(), x, create_graph=True)[0]
        tau_yy_x, tau_yy_y = grad_tau_yy[:, 0:1], grad_tau_yy[:, 1:2]
        
        # Equazioni di Quantità di Moto (Navier-Stokes)
        # ρ(u·∇u) + ∇p - μ_s∇²u - ∇·τ = 0
        f_u = self.rho * (u * u_x + v * u_y) + p_x - self.mu_s * (u_xx + u_yy) - (tau_xx_x + tau_xy_y)
        f_v = self.rho * (u * v_x + v * v_y) + p_y - self.mu_s * (v_xx + v_yy) - (tau_xy_x + tau_yy_y)
        
        # Equazioni Costitutive (Oldroyd-B)
        f_tau_xx = tau_xx + self.lam * (u * tau_xx_x + v * tau_xx_y - 2 * u_x * tau_xx - 2 * u_y * tau_xy) - 2 * self.mu_p * u_x
        f_tau_yy = tau_yy + self.lam * (u * tau_yy_x + v * tau_yy_y - 2 * v_x * tau_xy - 2 * v_y * tau_yy) - 2 * self.mu_p * v_y
        # Upper-Convected Derivative, componente xy:
        # (∇u · τ)_xy = u_x·τ_xy + u_y·τ_yy
        # (τ · ∇u^T)_xy = τ_xx·v_x + τ_xy·v_y
        f_tau_xy = tau_xy + self.lam * (
            u * tau_xy_x + v * tau_xy_y
            - u_x * tau_xy
            - u_y * tau_yy
            - tau_xx * v_x
            - tau_xy * v_y
        ) - self.mu_p * (u_y + v_x)
        
        return f_u, f_v, f_tau_xx, f_tau_yy, f_tau_xy

    def residual(self, model, x, pde_weights=None):
        """
        Ritorna la somma pesata degli MSE dei residui.
        Usa self.pde_weights (configurati nel costruttore) a meno che non venga
        passato un override esplicito.
        """
        weights = pde_weights if pde_weights is not None else self.pde_weights
        w_m = weights.get('momentum', 10.0)
        w_c = weights.get('constitutive', 1.0)
        
        f_u, f_v, f_tau_xx, f_tau_yy, f_tau_xy = self.compute_residuals(model, x)
        zeros = torch.zeros_like(f_u)
        loss_u = self.mse_loss(f_u, zeros)
        loss_v = self.mse_loss(f_v, zeros)
        loss_txx = self.mse_loss(f_tau_xx, zeros)
        loss_tyy = self.mse_loss(f_tau_yy, zeros)
        loss_txy = self.mse_loss(f_tau_xy, zeros)
        return w_m * (loss_u + loss_v) + w_c * (loss_txx + loss_tyy + loss_txy)

    def boundary_loss(self, model, x_bc, target_bc, variance_weights=None, active_bcs=None):
        """
        Calcola la MSE loss sui boundary points, ignorando i valori NaN.
        Supporta condizioni di Dirichlet e Neumann e maschera i campi inattivi.
        """
        if not x_bc.requires_grad:
            x_bc.requires_grad_(True)
            
        u, v, p, tau = self.get_velocity(model, x_bc)
        pred_bc = torch.cat([u, v, p, tau], dim=1)
        
        dir_target, neu_target, normals = target_bc
        keys = ['u', 'v', 'p', 'txx', 'txy', 'tyy']
        
        if active_bcs is not None:
            active_mask = torch.tensor([k in active_bcs for k in keys], device=x_bc.device)
        else:
            active_mask = torch.ones(6, dtype=torch.bool, device=x_bc.device)
            
        total_bc_loss = 0.0
        
        # --- Dirichlet Loss ---
        valid_dir = ~torch.isnan(dir_target) & active_mask
        if valid_dir.any():
            diff_dir = pred_bc - torch.nan_to_num(dir_target, nan=0.0)
            sq_diff_dir = diff_dir ** 2
            
            if variance_weights is not None:
                v_w = [max(variance_weights.get(k, 1.0), 1e-8) for k in keys]
                scales = torch.tensor(v_w, device=x_bc.device)
                sq_diff_dir = sq_diff_dir / scales
                
            total_bc_loss += sq_diff_dir[valid_dir].mean()
            
        # --- Neumann Loss ---
        valid_neu = ~torch.isnan(neu_target) & active_mask
        if valid_neu.any():
            nx = normals[:, 0:1]
            ny = normals[:, 1:2]
            preds = [u, v, p, tau[:, 0:1], tau[:, 1:2], tau[:, 2:3]]
            
            if variance_weights is not None:
                v_w = [max(variance_weights.get(k, 1.0), 1e-8) for k in keys]
                scales = torch.tensor(v_w, device=x_bc.device)
            
            for i, pred in enumerate(preds):
                if valid_neu[:, i].any():
                    grad_pred = torch.autograd.grad(pred.sum(), x_bc, create_graph=True)[0]
                    normal_deriv = grad_pred[:, 0:1] * nx + grad_pred[:, 1:2] * ny
                    
                    target_i = neu_target[:, i:i+1]
                    mask_i = valid_neu[:, i:i+1]
                    
                    diff_neu = normal_deriv - torch.nan_to_num(target_i, nan=0.0)
                    sq_diff_neu = diff_neu ** 2
                    
                    if variance_weights is not None:
                        sq_diff_neu = sq_diff_neu / scales[i]
                        
                    total_bc_loss += sq_diff_neu[mask_i].mean()
                    
        return total_bc_loss

def generate_boundaries(Lx, Ly, u_max, p_exact, stress_exact_dict, Nx, Ny, device):
    """
    Genera le condizioni al contorno per il dominio rettangolare.
    Ritorna 4 tensori: xy_boundary, dirichlet_target, neumann_target, normals
    """
    # 1. Bottom & Top (Wall) -> No-slip (Dirichlet), Neumann per Pressione e Stress
    x_wall = torch.linspace(0, Lx, Nx, device=device).reshape(-1, 1)
    y_wall_bottom = torch.zeros_like(x_wall)
    y_wall_top = torch.full_like(x_wall, Ly)
    
    n_bottom = torch.tensor([[0.0, -1.0]], device=device).expand(Nx, 2)
    n_top    = torch.tensor([[0.0, 1.0]], device=device).expand(Nx, 2)
    
    u_wall = torch.zeros_like(x_wall)
    v_wall = torch.zeros_like(x_wall)
    nan_val  = torch.full_like(u_wall, float('nan'))
    zero_val = torch.zeros_like(u_wall)

    # Dirichlet: u, v = 0. Altri NaN
    wall_dirichlet = torch.cat([u_wall, v_wall, nan_val, nan_val, nan_val, nan_val], dim=1)
    # Neumann: p, tau_xx, tau_xy, tau_yy = 0. u, v = NaN
    wall_neumann = torch.cat([nan_val, nan_val, zero_val, zero_val, zero_val, zero_val], dim=1)
    
    # 2. Inlet/Outlet
    y_inout = torch.linspace(0, Ly, Ny, device=device).reshape(-1, 1)
    u_parabolic = 4 * u_max * (y_inout * (Ly - y_inout)) / (Ly**2)
    v_zero = torch.zeros_like(y_inout)
    
    x_inlet = torch.zeros_like(y_inout)
    x_outlet = torch.full_like(y_inout, Lx)
    
    n_inlet  = torch.tensor([[-1.0, 0.0]], device=device).expand(Ny, 2)
    n_outlet = torch.tensor([[1.0, 0.0]], device=device).expand(Ny, 2)
    
    # Exact values for Dirichlet
    p_outlet = p_exact.reshape(Ny, Nx)[:, -1].reshape(-1, 1).to(device)
    
    txx_exact = stress_exact_dict.get('tau_xx', torch.full((Ny, Nx), float('nan'), device=device)).to(device)
    txy_exact = stress_exact_dict.get('tau_xy', torch.full((Ny, Nx), float('nan'), device=device)).to(device)
    tyy_exact = stress_exact_dict.get('tau_yy', torch.full((Ny, Nx), float('nan'), device=device)).to(device)
    
    txx_inlet  = txx_exact[:, 0].reshape(-1, 1)
    txy_inlet  = txy_exact[:, 0].reshape(-1, 1)
    tyy_inlet  = tyy_exact[:, 0].reshape(-1, 1)
    
    # Inlet Dirichlet: u=parabolico, v=0, tau=exact. p=NaN
    inlet_dirichlet = torch.cat([u_parabolic, v_zero, nan_val[:Ny], txx_inlet, txy_inlet, tyy_inlet], dim=1)
    # Inlet Neumann: p=0. u,v,tau=NaN
    inlet_neumann   = torch.cat([nan_val[:Ny], nan_val[:Ny], zero_val[:Ny], nan_val[:Ny], nan_val[:Ny], nan_val[:Ny]], dim=1)
    
    # Outlet Dirichlet: p=p_exact. u,v,tau=NaN
    outlet_dirichlet = torch.cat([nan_val[:Ny], nan_val[:Ny], p_outlet, nan_val[:Ny], nan_val[:Ny], nan_val[:Ny]], dim=1)
    # Outlet Neumann: tau=0. u,v,p=NaN
    outlet_neumann   = torch.cat([nan_val[:Ny], nan_val[:Ny], nan_val[:Ny], zero_val[:Ny], zero_val[:Ny], zero_val[:Ny]], dim=1)
    
    xy_boundary = torch.cat([
        torch.cat([x_wall, y_wall_bottom], dim=1),
        torch.cat([x_wall, y_wall_top], dim=1),
        torch.cat([x_inlet, y_inout], dim=1), 
        torch.cat([x_outlet, y_inout], dim=1)
    ], dim=0)

    dirichlet_boundary = torch.cat([wall_dirichlet, wall_dirichlet, inlet_dirichlet, outlet_dirichlet], dim=0)
    neumann_boundary   = torch.cat([wall_neumann, wall_neumann, inlet_neumann, outlet_neumann], dim=0)
    normals_boundary   = torch.cat([n_bottom, n_top, n_inlet, n_outlet], dim=0)
    
    return xy_boundary, dirichlet_boundary, neumann_boundary, normals_boundary
