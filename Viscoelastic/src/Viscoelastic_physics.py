import torch
import torch.nn as nn
import torch.nn.functional as F

def _softplus_inverse(x):
    """Calcola y tale che softplus(y) = x. Per inizializzare parametri reparametrizzati."""
    if x > 20.0:
        return x  # Per grandi valori, softplus(x) ≈ x
    if x < 1e-8:
        return -20.0  # softplus(-20) ≈ 2e-9 ≈ 0
    return float(torch.log(torch.exp(torch.tensor(x, dtype=torch.float64)) - 1.0).item())


# ==============================================================================
# STRUTTURA E FUNZIONAMENTO DELLE CONDIZIONI AL CONTORNO (BC)
# ==============================================================================
DEFAULT_BC_RULES = {
    'Inlet': {
        'dirichlet': {'u': 'csv', 'v': 'csv', 'p': 'csv', 'tau_xx': 'csv', 'tau_xy': 'csv', 'tau_yy': 'csv'},
        'neumann': {}
    },
    'Walls': {
        'dirichlet': {'u': 0.0, 'v': 0.0},
        'neumann': {'p': 0.0}
    },
    'Outlet': {
        'dirichlet': {'p': 'csv'},
        'neumann': {'tau_xx': 0.0, 'tau_xy': 0.0, 'tau_yy': 0.0}
    }
}


class ViscoelasticPhysics(nn.Module):
    def __init__(self, mu_s=0.005, mu_p=0.005, lam=1.0, eps=0.0, alpha=0.0, rho=1.0,
                 U_ref=1.0, H_ref=1.0,
                 pde_weights=None, inverse_mode=False,
                 real_mu_s=None, real_mu_p=None, real_lam=None, real_eps=None, real_alpha=None,
                 bc_rules=None):
        """
        Modulo per calcolare i residui fisici in forma ADIMENSIONALE con RISCALAMENTO VARIABILI.
        
        LOGICA DI RESCALING:
            Per evitare problemi numerici quando (1-beta) è molto piccolo, la rete neurale
            non predice lo stress fisico tau, ma lo stress riscalato tau_tilde:
                tau = (1 - beta) * tau_tilde
            In questo modo, le equazioni costitutive diventano di ordine O(1) e non richiedono
            moltiplicatori enormi che destabilizzano il gradiente.
        """
        super().__init__()
        self.inverse_mode = inverse_mode
        self.U_ref = U_ref
        self.H_ref = H_ref
        self.bc_rules = bc_rules or DEFAULT_BC_RULES
        self._boundary_metadata = None
        
        if inverse_mode:
            self.mu_s = nn.Parameter(torch.tensor([_softplus_inverse(max(mu_s - 1e-6, 1e-9))], dtype=torch.float32))
            self.mu_p = nn.Parameter(torch.tensor([_softplus_inverse(max(mu_p - 1e-6, 1e-9))], dtype=torch.float32))
            self.lam = nn.Parameter(torch.tensor([_softplus_inverse(max(lam - 1e-6, 1e-9))], dtype=torch.float32))
            self.eps = nn.Parameter(torch.tensor([_softplus_inverse(max(eps - 1e-8, 1e-9))], dtype=torch.float32))
            self.alpha = nn.Parameter(torch.tensor([_softplus_inverse(max(alpha - 1e-8, 1e-9))], dtype=torch.float32))
            self.real_mu_s = real_mu_s if real_mu_s is not None else mu_s
            self.real_mu_p = real_mu_p if real_mu_p is not None else mu_p
            self.real_lam = real_lam if real_lam is not None else lam
            self.real_eps = real_eps if real_eps is not None else eps
            self.real_alpha = real_alpha if real_alpha is not None else alpha
        else:
            self.register_buffer('mu_s', torch.tensor([mu_s], dtype=torch.float32))
            self.register_buffer('mu_p', torch.tensor([mu_p], dtype=torch.float32))
            self.register_buffer('lam', torch.tensor([lam], dtype=torch.float32))
            self.register_buffer('eps', torch.tensor([eps], dtype=torch.float32))
            self.register_buffer('alpha', torch.tensor([alpha], dtype=torch.float32))
            self.real_mu_s = real_mu_s if real_mu_s is not None else mu_s
            self.real_mu_p = real_mu_p if real_mu_p is not None else mu_p
            self.real_lam = real_lam if real_lam is not None else lam
            self.real_eps = real_eps if real_eps is not None else eps
            self.real_alpha = real_alpha if real_alpha is not None else alpha
            
        self.rho = rho
        self.mse_loss = nn.MSELoss()
        self.pde_weights = pde_weights or {'momentum': 10.0, 'constitutive': 1.0}

    @classmethod
    def from_dataset(cls, dataset, device='cpu', **kwargs):
        if isinstance(dataset, dict) and 'params' in dataset:
            params = dataset['params']
        elif hasattr(dataset, 'params'):
            params = dataset.params
        else:
            raise ValueError("Dataset parameters not found.")
            
        mu_s = params.get('mu_s', 0.005)
        mu_p = params.get('mu_p', 0.005)
        lam = params.get('lam', 1.0)
        eps = params.get('eps', 0.0)
        alpha = params.get('alpha', 0.0)
        rho = params.get('rho', 1.0)
        scales = dataset.get('scales', {})
        return cls(mu_s=mu_s, mu_p=mu_p, lam=lam, eps=eps, alpha=alpha, rho=rho,
                   U_ref=scales.get('U_ref', 1.0), H_ref=scales.get('H', 1.0),
                   inverse_mode=False, **kwargs).to(device)

    def _get_effective_params(self):
        if self.inverse_mode:
            return {
                'mu_s': 1e-6 + F.softplus(self.mu_s),
                'mu_p': 1e-6 + F.softplus(self.mu_p),
                'lam': 1e-6 + F.softplus(self.lam),
                'eps': 1e-8 + F.softplus(self.eps),
                'alpha': 1e-8 + F.softplus(self.alpha),
            }
        return {'mu_s': self.mu_s, 'mu_p': self.mu_p, 'lam': self.lam, 'eps': self.eps, 'alpha': self.alpha}

    def get_logged_parameters(self):
        eff = self._get_effective_params()
        return {k: v.item() if hasattr(v, 'item') else float(v) for k, v in eff.items()}

    def _get_nondim_params(self):
        eff = self._get_effective_params()
        mu_tot = eff['mu_s'] + eff['mu_p']
        return {
            'Re': self.rho * self.U_ref * self.H_ref / mu_tot,
            'Wi': eff['lam'] * self.U_ref / self.H_ref,
            'beta': eff['mu_s'] / mu_tot,
            'eps': eff['eps'],
            'alpha': eff['alpha'],
        }

    def get_velocity(self, model, x):
        """Restituisce u, v, p e lo stress FISICO tau (già de-riscalato)."""
        if not x.requires_grad: x = x.clone().requires_grad_(True)
        out = model(x)
        psi, p, tau_tilde = out[:, 0:1], out[:, 1:2], out[:, 2:5]
        grad_psi = torch.autograd.grad(psi.sum(), x, create_graph=True)[0]
        u, v = grad_psi[:, 1:2], -grad_psi[:, 0:1]
        
        # De-riscalamento per ottenere lo stress fisico tau
        beta = self._get_nondim_params()['beta']
        tau = (1.0 - beta) * tau_tilde
        return u, v, p, tau

    def compute_residuals(self, model, x):
        """Calcola i residui PDE operando sulle variabili riscalate (tilde)."""
        # 1. Recupero parametri e output rete (riscalati)
        nd = self._get_nondim_params()
        Re, Wi, beta, eps, alpha = nd['Re'], nd['Wi'], nd['beta'], nd['eps'], nd['alpha']
        one_m_beta = 1.0 - beta
        
        out = model(x)
        psi, p, tau_tilde = out[:, 0:1], out[:, 1:2], out[:, 2:5]
        tt_xx, tt_xy, tt_yy = tau_tilde[:, 0:1], tau_tilde[:, 1:2], tau_tilde[:, 2:3]
        
        # 2. Cinematica da Stream Function
        grad_psi = torch.autograd.grad(psi.sum(), x, create_graph=True)[0]
        u, v = grad_psi[:, 1:2], -grad_psi[:, 0:1]
        
        grad_u = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        u_x, u_y = grad_u[:, 0:1], grad_u[:, 1:2]
        grad_v = torch.autograd.grad(v.sum(), x, create_graph=True)[0]
        v_x, v_y = grad_v[:, 0:1], -u_x
        
        # 3. Derivate Seconde e Pressione
        grad_p = torch.autograd.grad(p.sum(), x, create_graph=True)[0]
        p_x, p_y = grad_p[:, 0:1], grad_p[:, 1:2]
        u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0][:, 0:1]
        u_yx = torch.autograd.grad(u_y.sum(), x, create_graph=True)[0][:, 0:1]
        u_yy = torch.autograd.grad(u_y.sum(), x, create_graph=True)[0][:, 1:2]
        v_xx = torch.autograd.grad(v_x.sum(), x, create_graph=True)[0][:, 0:1]
        v_yy = -u_yx
        
        # 4. Derivate Stress Riscalato (tilde)
        g_txx = torch.autograd.grad(tt_xx.sum(), x, create_graph=True)[0]
        g_txy = torch.autograd.grad(tt_xy.sum(), x, create_graph=True)[0]
        g_tyy = torch.autograd.grad(tt_yy.sum(), x, create_graph=True)[0]
        tt_xx_x, tt_xx_y = g_txx[:, 0:1], g_txx[:, 1:2]
        tt_xy_x, tt_xy_y = g_txy[:, 0:1], g_txy[:, 1:2]
        tt_yy_x, tt_yy_y = g_tyy[:, 0:1], g_tyy[:, 1:2]
        
        # 5. Momentum (Navier-Stokes) - Lo stress entra come (1-beta)*div(tau_tilde)
        f_u = Re * (u * u_x + v * u_y) + p_x - beta * (u_xx + u_yy) - one_m_beta * (tt_xx_x + tt_xy_y)
        f_v = Re * (u * v_x + v * v_y) + p_y - beta * (v_xx + v_yy) - one_m_beta * (tt_xy_x + tt_yy_y)

        # 6. Costitutiva Riscalata (Tutti i termini sono O(1))
        # f = 1 + eps*Wi*tr(tau_tilde)
        f_PTT = 1.0 + eps * Wi * (tt_xx + tt_yy)
        
        upper_xx = (u * tt_xx_x + v * tt_xx_y - 2 * u_x * tt_xx - 2 * u_y * tt_xy)
        upper_yy = (u * tt_yy_x + v * tt_yy_y - 2 * v_x * tt_xy - 2 * v_y * tt_yy)
        upper_xy = (u * tt_xy_x + v * tt_xy_y - u_x * tt_xy - u_y * tt_yy - tt_xx * v_x - tt_xy * v_y)

        f_txx = f_PTT * tt_xx + Wi * upper_xx + alpha * Wi * (tt_xx**2 + tt_xy**2) - 2.0 * u_x
        f_tyy = f_PTT * tt_yy + Wi * upper_yy + alpha * Wi * (tt_xy**2 + tt_yy**2) - 2.0 * v_y
        f_txy = f_PTT * tt_xy + Wi * upper_xy + alpha * Wi * tt_xy * (tt_xx + tt_yy) - (u_y + v_x)
        
        return f_u, f_v, f_txx, f_tyy, f_txy

    def residual(self, model, x, pde_weights=None, variance_weights=None):
        weights = pde_weights or self.pde_weights
        vw = variance_weights or {}
        
        f_u, f_v, f_txx, f_tyy, f_txy = self.compute_residuals(model, x)
                
        loss_m = (f_u**2 / max(vw.get('u', 1.0), 1e-8)).mean() + (f_v**2 / max(vw.get('v', 1.0), 1e-8)).mean()
        loss_c = (f_txx**2 / max(vw.get('txx_nd', 1.0), 1e-8)).mean() + \
                 (f_tyy**2 / max(vw.get('tyy_nd', 1.0), 1e-8)).mean() + \
                 (f_txy**2 / max(vw.get('txy_nd', 1.0), 1e-8)).mean()

        return weights.get('momentum', 10.0) * loss_m + weights.get('constitutive', 1.0) * loss_c

    def boundary_loss(self, model, x_bc, target_bc, variance_weights=None, active_bcs=None, group_weights=None):
        dir_target, neu_target, normals = target_bc
        nx, ny = normals[:, 0:1], normals[:, 1:2]
        
        keys = ['u', 'v', 'p', 'txx', 'txy', 'tyy']
        var_w = torch.ones((1, 6), device=x_bc.device)
        if variance_weights:
            for i, k in enumerate(keys): var_w[0, i] = variance_weights.get(k, 1.0)
                
        total_bc_loss = 0.0
        per_group_losses = {}

        if not hasattr(self, '_boundary_metadata') or not self._boundary_metadata:
            if not x_bc.requires_grad: x_bc = x_bc.clone().requires_grad_(True)
            u, v, p, tau = self.get_velocity(model, x_bc)
            pred_bc = torch.cat([u, v, p, tau], dim=1) 
            return self._compute_raw_bc_loss(pred_bc, x_bc, dir_target, neu_target, nx, ny, var_w, active_bcs, keys), {}

        start_idx = 0
        for g_name, M in self._boundary_metadata:
            end_idx = start_idx + M
            g_weight = group_weights.get(g_name, 1.0) if group_weights else 1.0
            
            # Estraiamo le coordinate locali per questo gruppo e abilitiamo i gradienti
            x_group = x_bc[start_idx:end_idx].clone().requires_grad_(True)
            
            # Forward pass locale specifico per il gruppo
            u_g, v_g, p_g, tau_g = self.get_velocity(model, x_group)
            pred_g = torch.cat([u_g, v_g, p_g, tau_g], dim=1)
            
            g_loss = self._compute_raw_bc_loss(pred_g, x_group, 
                                              dir_target[start_idx:end_idx], neu_target[start_idx:end_idx], 
                                              nx[start_idx:end_idx], ny[start_idx:end_idx], 
                                              var_w, active_bcs, keys)
            
            per_group_losses[f"loss_bc_{g_name}"] = g_loss.item()
            total_bc_loss += g_weight * g_loss
            start_idx = end_idx

        return total_bc_loss, per_group_losses

    def _compute_raw_bc_loss(self, pred, x, dir_t, neu_t, nx, ny, var_w, active_bcs, keys):
        loss = 0.0
        for i in range(6):
            if active_bcs and keys[i] not in active_bcs: continue
            # Dirichlet
            mask_d = (~torch.isnan(dir_t[:, i:i+1])).float()
            if mask_d.sum() > 0:
                diff = pred[:, i:i+1] - torch.nan_to_num(dir_t[:, i:i+1], nan=0.0)
                loss += (mask_d * (diff**2) / var_w[0, i]).sum() / mask_d.sum().clamp_min(1.0)
            # Neumann
            mask_n = (~torch.isnan(neu_t[:, i:i+1])).float()
            if mask_n.sum() > 0:
                p_i = pred[:, i:i+1]
                g_p = torch.autograd.grad(p_i.sum(), x, create_graph=True, retain_graph=True)[0]
                diff_n = (g_p[:, 0:1]*nx + g_p[:, 1:2]*ny) - torch.nan_to_num(neu_t[:, i:i+1], nan=0.0)
                loss += (mask_n * (diff_n**2) / var_w[0, i]).sum() / mask_n.sum().clamp_min(1.0)
        return loss

    def apply_boundary_conditions(self, boundary_groups):
        bc_rules = self.bc_rules
        FIELD_TO_COL = {'u': 0, 'v': 1, 'p': 2, 'tau_xx': 3, 'tau_xy': 4, 'tau_yy': 5}
        target_device, target_dtype = self.mu_s.device, self.mu_s.dtype
        xy_f, dir_f, neu_f, norm_f = [], [], [], []
        self._boundary_metadata = []

        print("\n" + "="*60 + f"\n{'BC AUDIT REPORT':^60}\n" + "-"*60)
        print(f"{'Group':<20} | {'Points':<8} | {'Rules'}")
        
        for g_key, group in boundary_groups.items():
            r_key = next((rk for rk in bc_rules if rk.lower() in g_key.lower() or g_key.lower() in rk.lower()), None)
            if not r_key: continue
            
            M = group['indices'].numel()
            rules = bc_rules[r_key]
            active = [f"{t[0].upper()}:{','.join(rules[t].keys())}" for t in ['dirichlet', 'neumann'] if rules.get(t)]
            print(f"{g_key:<20} | {M:<8} | {' '.join(active)}")

            d_g = torch.full((M, 6), float('nan'), device=target_device, dtype=target_dtype)
            n_g = torch.full((M, 6), float('nan'), device=target_device, dtype=target_dtype)

            for t, target_tensor in [('dirichlet', d_g), ('neumann', n_g)]:
                for field, val in rules.get(t, {}).items():
                    if field in FIELD_TO_COL:
                        target_tensor[:, FIELD_TO_COL[field]] = group['fields'][field].to(target_device, target_dtype).view(-1) if val == 'csv' else float(val)

            xy_f.append(group['xy'].to(target_device, target_dtype))
            norm_f.append(group['norm'].to(target_device, target_dtype))
            dir_f.append(d_g); neu_f.append(n_g)
            self._boundary_metadata.append((g_key, M))

        print("="*60 + "\n")
        return torch.cat(xy_f, 0), torch.cat(dir_f, 0), torch.cat(neu_f, 0), torch.cat(norm_f, 0)
