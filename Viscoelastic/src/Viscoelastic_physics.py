import torch
import torch.nn as nn

class ViscoelasticPhysics(nn.Module):
    def __init__(self, mu_s=0.005, mu_p=0.005, lam=1.0, rho=1.0):
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
        """
        super().__init__()
        self.mu_s = mu_s
        self.mu_p = mu_p
        self.lam = lam
        self.rho = rho
        self.mse_loss = nn.MSELoss()

    def get_velocity(self, model, x):
        """
        Calcola u, v e p a partire dalle reti neurali.
        """
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
        """
        if not x.requires_grad:
            x.requires_grad_(True)
            
        out = model(x)
        psi = out[:, 0:1]
        p = out[:, 1:2]
        tau_xx = out[:, 2:3]
        tau_xy = out[:, 3:4]
        tau_yy = out[:, 4:5]
        
        # u, v da psi
        grad_psi = torch.autograd.grad(psi.sum(), x, create_graph=True)[0]
        psi_x, psi_y = grad_psi[:, 0:1], grad_psi[:, 1:2]
        u = psi_y
        v = -psi_x
        
        # Derivate prime di u, v, p
        grad_u = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        u_x, u_y = grad_u[:, 0:1], grad_u[:, 1:2]
        grad_v = torch.autograd.grad(v.sum(), x, create_graph=True)[0]
        v_x, v_y = grad_v[:, 0:1], grad_v[:, 1:2]
        grad_p = torch.autograd.grad(p.sum(), x, create_graph=True)[0]
        p_x, p_y = grad_p[:, 0:1], grad_p[:, 1:2]
        
        # Derivate seconde di u, v
        grad_u_x = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
        u_xx = grad_u_x[:, 0:1]
        grad_u_y = torch.autograd.grad(u_y.sum(), x, create_graph=True)[0]
        u_yy = grad_u_y[:, 1:2]
        grad_v_x = torch.autograd.grad(v_x.sum(), x, create_graph=True)[0]
        v_xx = grad_v_x[:, 0:1]
        grad_v_y = torch.autograd.grad(v_y.sum(), x, create_graph=True)[0]
        v_yy = grad_v_y[:, 1:2]
        
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
        # (nabla_u · tau)_xy = u_x*tau_xy + u_y*tau_yy
        # (tau · nabla_u^T)_xy = tau_xx*v_x + tau_xy*v_y
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
        
        Args:
            pde_weights: Dict con pesi individuali per le componenti PDE.
                Default: {'momentum': 1.0, 'constitutive': 1.0}
                
                NOTA: Per Oldroyd-B i residui degli stress (tau_xx soprattutto)
                hanno magnitudini strutturalmente diverse dai residui di momentum,
                perché tau_xx scala come γ̇² (quadratico) mentre f_u scala come γ̇ (lineare).
                Se il training non converge sulla parte di momentum, provare
                pde_weights={'momentum': 5.0, 'constitutive': 1.0}.
        """
        if pde_weights is None:
            pde_weights = {'momentum': 1.0, 'constitutive': 1.0}
        w_m = pde_weights.get('momentum', 1.0)
        w_c = pde_weights.get('constitutive', 1.0)
        
        f_u, f_v, f_tau_xx, f_tau_yy, f_tau_xy = self.compute_residuals(model, x)
        zeros = torch.zeros_like(f_u)
        loss_u = self.mse_loss(f_u, zeros)
        loss_v = self.mse_loss(f_v, zeros)
        loss_txx = self.mse_loss(f_tau_xx, zeros)
        loss_tyy = self.mse_loss(f_tau_yy, zeros)
        loss_txy = self.mse_loss(f_tau_xy, zeros)
        return w_m * (loss_u + loss_v) + w_c * (loss_txx + loss_tyy + loss_txy)

    def boundary_loss(self, model, x_bc, target_bc):
        """
        Calcola la MSE loss sui boundary points, ignorando i valori NaN.
        Il target_bc deve contenere: [u, v, p, tau_xx, tau_xy, tau_yy].
        """
        if not x_bc.requires_grad:
            x_bc.requires_grad_(True)
            
        u, v, p, tau = self.get_velocity(model, x_bc)
        # Unifichiamo le predizioni in un unico tensore [u, v, p, txx, txy, tyy]
        pred_bc = torch.cat([u, v, p, tau], dim=1)
        
        mask = ~torch.isnan(target_bc)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=x_bc.device, requires_grad=True)
            
        return self.mse_loss(pred_bc[mask], target_bc[mask])

def generate_boundaries(Lx, Ly, u_max, p_exact, stress_exact_dict, Nx, Ny, device):
    """
    Genera le condizioni al contorno per il dominio rettangolare.
    Include: u, v, p, tau_xx, tau_xy, tau_yy.
    I target non forniti vengono impostati a NaN.
    """
    xy_boundary_list = []
    target_boundary_list = []
    
    # 1. Bottom & Top (Wall) -> No-slip + Stress analitici
    x_wall = torch.linspace(0, Lx, Nx+2)[1:-1].reshape(-1, 1).to(device)
    y_wall_bottom = torch.zeros_like(x_wall).to(device)
    y_wall_top = torch.full_like(x_wall, Ly).to(device)
    
    # Estraggo gli stress analitici alle pareti (se disponibili nel dict)
    txx_exact = stress_exact_dict.get('tau_xx', torch.full((Ny, Nx), float('nan'))).to(device)
    txy_exact = stress_exact_dict.get('tau_xy', torch.full((Ny, Nx), float('nan'))).to(device)
    tyy_exact = stress_exact_dict.get('tau_yy', torch.full((Ny, Nx), float('nan'))).to(device)

    # Nota: y=0 è riga 0, y=Ly è riga Ny-1
    txx_bottom = txx_exact[0, :].reshape(-1, 1)
    txx_top    = txx_exact[-1, :].reshape(-1, 1)
    txy_bottom = txy_exact[0, :].reshape(-1, 1)
    txy_top    = txy_exact[-1, :].reshape(-1, 1)
    tyy_bottom = tyy_exact[0, :].reshape(-1, 1)
    tyy_top    = tyy_exact[-1, :].reshape(-1, 1)

    u_wall = torch.zeros_like(x_wall)
    v_wall = torch.zeros_like(x_wall)
    nan_p  = torch.full_like(u_wall, float('nan'))

    wall_bottom_target = torch.cat([u_wall, v_wall, nan_p, txx_bottom, txy_bottom, tyy_bottom], dim=1)
    wall_top_target    = torch.cat([u_wall, v_wall, nan_p, txx_top, txy_top, tyy_top], dim=1)
    
    # 2. Inlet/Outlet -> Velocità Parabolica + Pressione + Stress
    y_inout = torch.linspace(0, Ly, Ny).reshape(-1, 1).to(device)
    u_parabolic = 4 * u_max * (y_inout * (Ly - y_inout)) / (Ly**2)
    v_zero = torch.zeros_like(y_inout)
    
    x_inlet = torch.zeros_like(y_inout)
    x_outlet = torch.full_like(y_inout, Lx)
    
    p_inlet  = p_exact.reshape(Ny, Nx)[:, 0].reshape(-1, 1).to(device)
    p_outlet = p_exact.reshape(Ny, Nx)[:, -1].reshape(-1, 1).to(device)
    
    # Stress Inlet/Outlet
    txx_inlet  = txx_exact[:, 0].reshape(-1, 1)
    txy_inlet  = txy_exact[:, 0].reshape(-1, 1)
    tyy_inlet  = tyy_exact[:, 0].reshape(-1, 1)
    txx_outlet = txx_exact[:, -1].reshape(-1, 1)
    txy_outlet = txy_exact[:, -1].reshape(-1, 1)
    tyy_outlet = tyy_exact[:, -1].reshape(-1, 1)

    inlet_target = torch.cat([u_parabolic, v_zero, p_inlet, txx_inlet, txy_inlet, tyy_inlet], dim=1)
    outlet_target = torch.cat([
        torch.full_like(v_zero, float('nan')),  # u libera
        torch.full_like(v_zero, float('nan')),  # v libera
        p_outlet,
        txx_outlet, txy_outlet, tyy_outlet
    ], dim=1)
    
    xy_boundary = torch.cat([
        torch.cat([x_wall, y_wall_bottom], dim=1),
        torch.cat([x_wall, y_wall_top], dim=1),
        torch.cat([x_inlet, y_inout], dim=1), 
        torch.cat([x_outlet, y_inout], dim=1)
    ], dim=0)

    target_boundary = torch.cat([
        wall_bottom_target,
        wall_top_target,
        inlet_target,
        outlet_target
    ], dim=0)
    
    return xy_boundary, target_boundary
