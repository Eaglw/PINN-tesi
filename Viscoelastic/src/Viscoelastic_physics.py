import torch
import torch.nn as nn

class ViscoelasticPhysics(nn.Module):
    def __init__(self, mu_s=0.005, mu_p=0.005, lam=1.0, rho=1.0, pde_weights=None, inverse_mode=False, real_mu_s=None, real_mu_p=None, real_lam=None):
        """
        Modulo per calcolare i residui fisici (equazioni di Navier-Stokes e modello Oldroyd-B).
        
        NOTA FISICA — Scelta della Stream Function:
            L'equazione di continuità (∇·u = 0) è automaticamente soddisfatta
            dall'uso della stream function: u = ∂ψ/∂y, v = -∂ψ/∂x.
            Dimostrazione: ∂u/∂x + ∂v/∂y = ∂²ψ/∂x∂y - ∂²ψ/∂y∂x = 0.
            Perciò NON è inclusa esplicitamente tra i residui.
            Se si passasse a output diretto (u, v), va aggiunta come residuo.
        
        Args:
            mu_s: Viscosità del solvente [Pa·s] (o guess iniziale per inverse problem)
            mu_p: Viscosità polimerica [Pa·s] (o guess iniziale per inverse problem)
            lam: Tempo di rilassamento [s] (o guess iniziale per inverse problem)
            rho: Densità del fluido [kg/m³]. Default=1.0 (adimensionale).
                 Se rho != 1, le equazioni del momento vengono scalate di conseguenza.
            pde_weights: Dict con pesi per le componenti PDE.
                Default: {'momentum': 10.0, 'constitutive': 1.0}
                NOTA: Per Oldroyd-B i residui degli stress (tau_xx soprattutto)
                hanno magnitudini strutturalmente diverse dai residui di momentum,
                perché tau_xx scala come γ̇² (quadratico) mentre f_u scala come γ̇ (lineare).
            inverse_mode: Se True, i parametri mu_s, mu_p, lam diventano addestrabili.
            real_*: Valori reali usati per il plotting e la verifica in modalità inversa.
        """
        super().__init__()
        self.inverse_mode = inverse_mode
        if inverse_mode:
            self.mu_s = nn.Parameter(torch.tensor([mu_s], dtype=torch.float32))
            self.mu_p = nn.Parameter(torch.tensor([mu_p], dtype=torch.float32))
            self.lam = nn.Parameter(torch.tensor([lam], dtype=torch.float32))
            self.real_mu_s = real_mu_s if real_mu_s is not None else mu_s
            self.real_mu_p = real_mu_p if real_mu_p is not None else mu_p
            self.real_lam = real_lam if real_lam is not None else lam
        else:
            self.mu_s = mu_s
            self.mu_p = mu_p
            self.lam = lam
            self.real_mu_s = mu_s
            self.real_mu_p = mu_p
            self.real_lam = lam
            
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
        
        # Valori assoluti per garantire positività fisica (il blocco)
        mu_s_eff = torch.abs(self.mu_s) if isinstance(self.mu_s, torch.Tensor) else abs(self.mu_s)
        mu_p_eff = torch.abs(self.mu_p) if isinstance(self.mu_p, torch.Tensor) else abs(self.mu_p)
        lam_eff  = torch.abs(self.lam) if isinstance(self.lam, torch.Tensor) else abs(self.lam)
        
        # Equazioni di Quantità di Moto (Navier-Stokes)
        # ρ(u·∇u) + ∇p - μ_s∇²u - ∇·τ = 0
        f_u = self.rho * (u * u_x + v * u_y) + p_x - mu_s_eff * (u_xx + u_yy) - (tau_xx_x + tau_xy_y)
        f_v = self.rho * (u * v_x + v * v_y) + p_y - mu_s_eff * (v_xx + v_yy) - (tau_xy_x + tau_yy_y)
        
        # Equazioni Costitutive (Oldroyd-B)
        f_tau_xx = tau_xx + lam_eff * (u * tau_xx_x + v * tau_xx_y - 2 * u_x * tau_xx - 2 * u_y * tau_xy) - 2 * mu_p_eff * u_x
        f_tau_yy = tau_yy + lam_eff * (u * tau_yy_x + v * tau_yy_y - 2 * v_x * tau_xy - 2 * v_y * tau_yy) - 2 * mu_p_eff * v_y
        # Upper-Convected Derivative, componente xy:
        # (∇u · τ)_xy = u_x·τ_xy + u_y·τ_yy
        # (τ · ∇u^T)_xy = τ_xx·v_x + τ_xy·v_y
        f_tau_xy = tau_xy + lam_eff * (
            u * tau_xy_x + v * tau_xy_y
            - u_x * tau_xy
            - u_y * tau_yy
            - tau_xx * v_x
            - tau_xy * v_y
        ) - mu_p_eff * (u_y + v_x)
        
        return f_u, f_v, f_tau_xx, f_tau_yy, f_tau_xy

    def residual(self, model, x, pde_weights=None, variance_weights=None):
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
            active_mask = [k in active_bcs for k in keys]
        else:
            active_mask = [True] * 6
            
        total_bc_loss = 0.0
        
        # --- Dirichlet Loss ---
        valid_dir_float_list = []
        for i, is_active in enumerate(active_mask):
            if is_active:
                valid_dir_float_list.append((~torch.isnan(dir_target[:, i:i+1])).float())
            else:
                valid_dir_float_list.append(torch.zeros_like(dir_target[:, i:i+1]))
        valid_dir_float = torch.cat(valid_dir_float_list, dim=1)
        diff_dir = pred_bc - torch.nan_to_num(dir_target, nan=0.0)
        sq_diff_dir = diff_dir ** 2
        
        if variance_weights is not None:
            sq_diff_dir_list = []
            for i, k in enumerate(keys):
                scale = variance_weights.get(k, 1.0)
                sq_diff_dir_list.append(sq_diff_dir[:, i:i+1] / scale)
            sq_diff_dir = torch.cat(sq_diff_dir_list, dim=1)
            
        sum_dir = (sq_diff_dir * valid_dir_float).sum()
        count_dir = valid_dir_float.sum()
        total_bc_loss += sum_dir / count_dir.clamp_min(1.0)
            
        # --- Neumann Loss ---
        valid_neu_float_list = []
        for i, is_active in enumerate(active_mask):
            if is_active:
                valid_neu_float_list.append((~torch.isnan(neu_target[:, i:i+1])).float())
            else:
                valid_neu_float_list.append(torch.zeros_like(neu_target[:, i:i+1]))
        valid_neu_float = torch.cat(valid_neu_float_list, dim=1)
        nx = normals[:, 0:1]
        ny = normals[:, 1:2]
        preds = [u, v, p, tau[:, 0:1], tau[:, 1:2], tau[:, 2:3]]
        
        for i, pred in enumerate(preds):
            if not active_mask[i]:
                continue
            grad_pred = torch.autograd.grad(pred.sum(), x_bc, create_graph=True)[0]
            normal_deriv = grad_pred[:, 0:1] * nx + grad_pred[:, 1:2] * ny
            
            target_i = neu_target[:, i:i+1]
            mask_i = valid_neu_float[:, i:i+1]
            
            diff_neu = normal_deriv - torch.nan_to_num(target_i, nan=0.0)
            
            if variance_weights is not None:
                scale = variance_weights.get(keys[i], 1.0)
                sq_diff_neu = (diff_neu ** 2) / scale
            else:
                sq_diff_neu = diff_neu ** 2
            
            sum_neu = (sq_diff_neu * mask_i).sum()
            count_neu = mask_i.sum()
            total_bc_loss += sum_neu / count_neu.clamp_min(1.0)
                    
        return total_bc_loss

def generate_boundaries(Lx, Ly, u_max, p_exact, stress_exact_dict, Nx, Ny, device):
    """
    Genera le condizioni al contorno per il dominio rettangolare.
    Ritorna 4 tensori: xy_boundary, dirichlet_target, neumann_target, normals
    
    Slicing Geometrico Rigoroso (senza duplicati):
    - Inlet (x=0): intero lato y [0, Ly].
    - Walls (y=0, Ly): intero lato x (0, Lx] (esclude x=0).
    - Outlet (x=Lx): lato y (0, Ly) (esclude y=0 e y=Ly).
    """
    
    # --- HELPER INTERNI PER L'ASSEMBLAGGIO DELLO STATO ---
    # Variabili fisiche attese: [u, v, p, tau_xx, tau_xy, tau_yy]
    def pack_state(u, v, p, txx, txy, tyy):
        return torch.cat([u, v, p, txx, txy, tyy], dim=1)
        
    def get_nan(ref_tensor):
        return torch.full_like(ref_tensor, float('nan'))
        
    def get_zero(ref_tensor):
        return torch.zeros_like(ref_tensor)

    # ==========================================================
    # 1. INLET (x = 0, y in [0, Ly]) -> Ny punti
    # ==========================================================
    y_inlet = torch.linspace(0, Ly, Ny, device=device).reshape(-1, 1)
    x_inlet = get_zero(y_inlet)
    n_inlet = torch.tensor([[-1.0, 0.0]], device=device).expand(Ny, 2) # Vettore normale
    
    # Dirichlet: Profilo parabolico (u) e velocità trasversale nulla (v)
    u_inlet = 4 * u_max * (y_inlet * (Ly - y_inlet)) / (Ly**2)
    v_inlet = get_zero(y_inlet)
    p_inlet = torch.ones_like(y_inlet)
    
    # Dirichlet: Estrazione sforzi esatti
    txx_exact = stress_exact_dict.get('tau_xx', torch.full((Ny, Nx), float('nan'), device=device)).to(device)
    txy_exact = stress_exact_dict.get('tau_xy', torch.full((Ny, Nx), float('nan'), device=device)).to(device)
    tyy_exact = stress_exact_dict.get('tau_yy', torch.full((Ny, Nx), float('nan'), device=device)).to(device)
    
    txx_inlet = txx_exact[:, 0].reshape(-1, 1)
    txy_inlet = txy_exact[:, 0].reshape(-1, 1)
    tyy_inlet = tyy_exact[:, 0].reshape(-1, 1)
    
    inlet_dirichlet = pack_state(u_inlet, v_inlet, p_inlet, txx_inlet, txy_inlet, tyy_inlet)
    
    # Neumann: Nessuna restrizione all'ingresso
    nan_inlet = get_nan(y_inlet)
    inlet_neumann = pack_state(nan_inlet, nan_inlet, nan_inlet, nan_inlet, nan_inlet, nan_inlet)


    # ==========================================================
    # 2. WALLS (x in (0, Lx], y = 0 e y = Ly) -> Nx - 1 punti cd.
    # ==========================================================
    x_wall_full = torch.linspace(0, Lx, Nx, device=device).reshape(-1, 1)
    x_wall = x_wall_full[1:]  # Escludiamo x=0 (già gestito dall'inlet)
    Nx_wall = Nx - 1
    
    y_bottom = get_zero(x_wall)
    y_top    = torch.full_like(x_wall, Ly) #stessa forma di x_wall ma con Ly come contenuto
    
    n_bottom = torch.tensor([[0.0, -1.0]], device=device).expand(Nx_wall, 2)
    n_top    = torch.tensor([[0.0,  1.0]], device=device).expand(Nx_wall, 2)
    
    # Dirichlet: Condizione di aderenza alla parete (No-slip)
    u_wall   = get_zero(x_wall)
    v_wall   = get_zero(x_wall)
    nan_wall = get_nan(x_wall)
    
    wall_dirichlet = pack_state(u_wall, v_wall, nan_wall, nan_wall, nan_wall, nan_wall)
    
    # Neumann: Gradiente di pressione normale nullo alla parete (dp/dn = 0)
    zero_wall = get_zero(x_wall)
    wall_neumann = pack_state(nan_wall, nan_wall, zero_wall, nan_wall, nan_wall, nan_wall)


    # ==========================================================
    # 3. OUTLET (x = Lx, y in (0, Ly)) -> Ny - 2 punti
    # ==========================================================
    y_outlet_full = torch.linspace(0, Ly, Ny, device=device).reshape(-1, 1)
    y_outlet = y_outlet_full[1:-1]  # Escludiamo gli spigoli governati dalle pareti
    Ny_outlet = Ny - 2
    
    x_outlet = torch.full_like(y_outlet, Lx)
    n_outlet = torch.tensor([[1.0, 0.0]], device=device).expand(Ny_outlet, 2)
    
    # Dirichlet: Scarico a pressione zero
    p_outlet   = get_zero(y_outlet)
    nan_outlet = get_nan(y_outlet)
    
    outlet_dirichlet = pack_state(nan_outlet, nan_outlet, p_outlet, nan_outlet, nan_outlet, nan_outlet)
    
    # Neumann: Flusso in uscita libero
    outlet_neumann = pack_state(nan_outlet, nan_outlet, nan_outlet, nan_outlet, nan_outlet, nan_outlet)


    # ==========================================================
    # 4. CONCATENAZIONE GLOBALE
    # ==========================================================
    xy_boundary = torch.cat([
        torch.cat([x_inlet, y_inlet], dim=1),
        torch.cat([x_wall, y_bottom], dim=1),
        torch.cat([x_wall, y_top], dim=1),
        torch.cat([x_outlet, y_outlet], dim=1)
    ], dim=0)

    dirichlet_boundary = torch.cat([
        inlet_dirichlet, 
        wall_dirichlet, 
        wall_dirichlet,  # Mappato due volte per bottom e top
        outlet_dirichlet
    ], dim=0)
    
    neumann_boundary = torch.cat([
        inlet_neumann, 
        wall_neumann, 
        wall_neumann,    # Mappato due volte per bottom e top
        outlet_neumann
    ], dim=0)
    
    normals_boundary = torch.cat([
        n_inlet, 
        n_bottom, 
        n_top, 
        n_outlet
    ], dim=0)
    
    return xy_boundary, dirichlet_boundary, neumann_boundary, normals_boundary