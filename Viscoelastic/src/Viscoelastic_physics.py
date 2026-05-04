import torch
import torch.nn as nn

class ViscoelasticPhysics(nn.Module):
    def __init__(self, mu_s=0.005, mu_p=0.005, lam=1.0):
        """
        Modulo per calcolare i residui fisici (equazioni di Navier-Stokes e modello Oldroyd-B).
        """
        super().__init__()
        self.mu_s = mu_s
        self.mu_p = mu_p
        self.lam = lam
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
        psi_x = torch.autograd.grad(psi.sum(), x, create_graph=True)[0][:, 0:1]
        psi_y = torch.autograd.grad(psi.sum(), x, create_graph=True)[0][:, 1:2]
        
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
        psi_x = torch.autograd.grad(psi.sum(), x, create_graph=True)[0][:, 0:1]
        psi_y = torch.autograd.grad(psi.sum(), x, create_graph=True)[0][:, 1:2]
        u = psi_y
        v = -psi_x
        
        # Derivate prime di u, v, p
        u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0][:, 0:1]
        u_y = torch.autograd.grad(u.sum(), x, create_graph=True)[0][:, 1:2]
        v_x = torch.autograd.grad(v.sum(), x, create_graph=True)[0][:, 0:1]
        v_y = torch.autograd.grad(v.sum(), x, create_graph=True)[0][:, 1:2]
        p_x = torch.autograd.grad(p.sum(), x, create_graph=True)[0][:, 0:1]
        p_y = torch.autograd.grad(p.sum(), x, create_graph=True)[0][:, 1:2]
        
        # Derivate seconde di u, v
        u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0][:, 0:1]
        u_yy = torch.autograd.grad(u_y.sum(), x, create_graph=True)[0][:, 1:2]
        v_xx = torch.autograd.grad(v_x.sum(), x, create_graph=True)[0][:, 0:1]
        v_yy = torch.autograd.grad(v_y.sum(), x, create_graph=True)[0][:, 1:2]
        
        # Derivate prime di tau
        tau_xx_x = torch.autograd.grad(tau_xx.sum(), x, create_graph=True)[0][:, 0:1]
        tau_xx_y = torch.autograd.grad(tau_xx.sum(), x, create_graph=True)[0][:, 1:2]
        tau_xy_x = torch.autograd.grad(tau_xy.sum(), x, create_graph=True)[0][:, 0:1]
        tau_xy_y = torch.autograd.grad(tau_xy.sum(), x, create_graph=True)[0][:, 1:2]
        tau_yy_x = torch.autograd.grad(tau_yy.sum(), x, create_graph=True)[0][:, 0:1]
        tau_yy_y = torch.autograd.grad(tau_yy.sum(), x, create_graph=True)[0][:, 1:2]
        
        # Equazioni di Quantità di Moto (Navier-Stokes)
        f_u = (u * u_x + v * u_y) + p_x - self.mu_s * (u_xx + u_yy) - (tau_xx_x + tau_xy_y)
        f_v = (u * v_x + v * v_y) + p_y - self.mu_s * (v_xx + v_yy) - (tau_xy_x + tau_yy_y)
        
        # Equazioni Costitutive (Oldroyd-B)
        f_tau_xx = tau_xx + self.lam * (u * tau_xx_x + v * tau_xx_y - 2 * u_x * tau_xx - 2 * u_y * tau_xy) - 2 * self.mu_p * u_x
        f_tau_yy = tau_yy + self.lam * (u * tau_yy_x + v * tau_yy_y - 2 * v_x * tau_xy - 2 * v_y * tau_yy) - 2 * self.mu_p * v_y
        f_tau_xy = tau_xy + self.lam * (u * tau_xy_x + v * tau_xy_y - v_x * tau_xx - u_y * tau_yy) - self.mu_p * (u_y + v_x)
        
        return f_u, f_v, f_tau_xx, f_tau_yy, f_tau_xy

    def residual(self, model, x):
        """
        Ritorna la somma degli MSE dei residui.
        """
        f_u, f_v, f_tau_xx, f_tau_yy, f_tau_xy = self.compute_residuals(model, x)
        zeros = torch.zeros_like(f_u)
        loss_u = self.mse_loss(f_u, zeros)
        loss_v = self.mse_loss(f_v, zeros)
        loss_txx = self.mse_loss(f_tau_xx, zeros)
        loss_tyy = self.mse_loss(f_tau_yy, zeros)
        loss_txy = self.mse_loss(f_tau_xy, zeros)
        return loss_u + loss_v + loss_txx + loss_tyy + loss_txy

    def boundary_loss(self, model, x_bc, target_bc):
        """
        Calcola la MSE loss sui boundary points, ignorando i valori NaN.
        Il target_bc deve contenere: [u, v, p].
        """
        if not x_bc.requires_grad:
            x_bc.requires_grad_(True)
            
        u, v, p, _ = self.get_velocity(model, x_bc)
        pred_bc = torch.cat([u, v, p], dim=1)
        
        mask = ~torch.isnan(target_bc)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=x_bc.device, requires_grad=True)
            
        return self.mse_loss(pred_bc[mask], target_bc[mask])

def generate_boundaries(Lx, Ly, u_max, p_exact, P_grid, Nx, Ny, device):
    """
    Genera le condizioni al contorno per il dominio rettangolare (u, v, p).
    I target non forniti vengono impostati a NaN.
    """
    xy_boundary_list = []
    uvp_boundary_list = []
    
    # Bottom & Top (Wall) -> No-slip
    x_wall = torch.linspace(0, Lx, Nx).reshape(-1, 1).to(device)
    y_wall_bottom = torch.zeros_like(x_wall).to(device)
    y_wall_top = torch.full_like(x_wall, Ly).to(device)
    
    bottom_wall = torch.cat([x_wall, y_wall_bottom], dim=1)
    top_wall = torch.cat([x_wall, y_wall_top], dim=1)
    
    u_wall = torch.zeros_like(x_wall).to(device)
    v_wall = torch.zeros_like(x_wall).to(device)
    
    # Left & Right (Inlet / Outlet) -> Pressure
    y_inout = torch.linspace(0, Ly, Ny).reshape(-1, 1).to(device)
    x_inlet = torch.zeros_like(y_inout).to(device)
    x_outlet = torch.full_like(y_inout, Lx).to(device)
    
    inlet = torch.cat([x_inlet, y_inout], dim=1)
    outlet = torch.cat([x_outlet, y_inout], dim=1)
    
    p_inlet = p_exact.reshape(Ny, Nx)[:, 0].reshape(-1, 1).to(device)
    p_outlet = p_exact.reshape(Ny, Nx)[:, -1].reshape(-1, 1).to(device)
    
    # Appende tutto (usando NaN dove non c'è BC)
    xy_boundary_list.extend([bottom_wall, top_wall])
    uvp_boundary_list.extend([
        torch.cat([u_wall, v_wall, torch.full_like(u_wall, float('nan'))], dim=1),
        torch.cat([u_wall, v_wall, torch.full_like(u_wall, float('nan'))], dim=1)
    ])
    
    xy_boundary_list.extend([inlet, outlet])
    uvp_boundary_list.extend([
        torch.cat([torch.full_like(p_inlet, float('nan')), torch.full_like(p_inlet, float('nan')), p_inlet], dim=1),
        torch.cat([torch.full_like(p_outlet, float('nan')), torch.full_like(p_outlet, float('nan')), p_outlet], dim=1)
    ])
    
    xy_boundary = torch.cat(xy_boundary_list, dim=0)
    uvp_boundary = torch.cat(uvp_boundary_list, dim=0)
    
    return xy_boundary, uvp_boundary
