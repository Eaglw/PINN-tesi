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


class ViscoelasticPhysics(nn.Module):
    def __init__(self, mu_s=0.005, mu_p=0.005, lam=1.0, eps=0.0, alpha=0.0, rho=1.0,
                 U_ref=1.0, H_ref=1.0,
                 pde_weights=None, inverse_mode=False,
                 real_mu_s=None, real_mu_p=None, real_lam=None, real_eps=None, real_alpha=None):
        """
        Modulo per calcolare i residui fisici in forma ADIMENSIONALE.
        
        Le PDE sono scritte in termini di Re, Wi, β calcolati on-the-fly
        dai parametri dimensionali (mu_s, mu_p, lam) e dalle scale di riferimento
        (U_ref, H_ref).
        
        NOTA FISICA — Scelta della Stream Function:
            L'equazione di continuità (∇·u = 0) è automaticamente soddisfatta
            dall'uso della stream function: u = ∂ψ/∂y, v = -∂ψ/∂x.
            Perciò NON è inclusa esplicitamente tra i residui.
        
        Forma adimensionale:
            Momentum: Re(u·∇u) + ∇p - β∇²u - ∇·τ = 0
            Costitutiva: f·τ + Wi·∇̊τ + α·Wi/(1-β)·(τ·τ) - 2(1-β)·D = 0
                dove f = 1 + ε·Wi/(1-β)·tr(τ) (PTT)
        
        Args:
            mu_s: Viscosità del solvente [Pa·s] (o guess iniziale per inverse problem)
            mu_p: Viscosità polimerica [Pa·s] (o guess iniziale per inverse problem)
            lam: Tempo di rilassamento [s] (o guess iniziale per inverse problem)
            eps: Parametro di mobilità PTT (o guess iniziale)
            alpha: Parametro di mobilità Giesekus (o guess iniziale)
            rho: Densità del fluido [kg/m³].
            U_ref: Velocità di riferimento per adimensionalizzazione [m/s].
            H_ref: Lunghezza di riferimento (altezza canale) [m].
            pde_weights: Dict con pesi per le componenti PDE.
                Default: {'momentum': 10.0, 'constitutive': 1.0}
            inverse_mode: Se True, i parametri mu_s, mu_p, lam diventano addestrabili.
            real_*: Valori reali usati per il plotting e la verifica in modalità inversa.
        """
        super().__init__()
        self.inverse_mode = inverse_mode
        self.U_ref = U_ref
        self.H_ref = H_ref
        
        if inverse_mode:
            # Reparametrizzazione: shifted softplus
            # raw = softplus_inverse(guess - offset)
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
        """
        Costruisce il modulo fisico estraendo i parametri esatti dal dataset.
        Supporta sia il formato .pt legacy sia il formato COMSOL con 'scales'.
        Forza inverse_mode=False perché è pensato per il forward problem.
        """
        if isinstance(dataset, dict) and 'params' in dataset:
            params = dataset['params']
        elif hasattr(dataset, 'params'):
            params = dataset.params
        else:
            raise ValueError("Il dataset fornito non contiene i parametri (chiave 'params' o attributo 'params').")
            
        mu_s = params.get('mu_s', 0.005)
        mu_p = params.get('mu_p', 0.005)
        lam = params.get('lam', 1.0)
        eps = params.get('eps', 0.0)
        alpha = params.get('alpha', 0.0)
        rho = params.get('rho', 1.0)
        
        # Estrai scale di riferimento per adimensionalizzazione (COMSOL format)
        scales = dataset.get('scales', {})
        U_ref = scales.get('U_ref', 1.0)
        H_ref = scales.get('H', 1.0)
        
        return cls(
            mu_s=mu_s, mu_p=mu_p, lam=lam, eps=eps, alpha=alpha, rho=rho,
            U_ref=U_ref, H_ref=H_ref,
            inverse_mode=False,
            real_mu_s=mu_s, real_mu_p=mu_p, real_lam=lam, real_eps=eps, real_alpha=alpha,
            **kwargs
        ).to(device)

    def _get_effective_params(self):
        """Restituisce i parametri fisici effettivi usati nelle equazioni.
        In forward mode: valori diretti (buffer).
        In inverse mode: softplus(raw) con clamp di sicurezza post-softplus.
        """
        if self.inverse_mode:
            return {
                'mu_s': 1e-6 + F.softplus(self.mu_s),
                'mu_p': 1e-6 + F.softplus(self.mu_p),
                'lam': 1e-6 + F.softplus(self.lam),
                'eps': 1e-8 + F.softplus(self.eps),
                'alpha': 1e-8 + F.softplus(self.alpha),
            }
        else:
            return {
                'mu_s': self.mu_s,
                'mu_p': self.mu_p,
                'lam': self.lam,
                'eps': self.eps,
                'alpha': self.alpha,
            }

    def _get_nondim_params(self):
        """Calcola i numeri adimensionali (Re, Wi, β, ε, α) dai parametri dimensionali correnti.
        
        Re = ρ·U_ref·H_ref / (μ_s + μ_p)
        Wi = λ·U_ref / H_ref
        β  = μ_s / (μ_s + μ_p)
        ε e α sono già adimensionali.
        """
        eff = self._get_effective_params()
        mu_s_eff = eff['mu_s']
        mu_p_eff = eff['mu_p']
        mu_tot = mu_s_eff + mu_p_eff
        
        Re = self.rho * self.U_ref * self.H_ref / mu_tot
        Wi = eff['lam'] * self.U_ref / self.H_ref
        beta = mu_s_eff / mu_tot
        
        return {
            'Re': Re,
            'Wi': Wi,
            'beta': beta,
            'eps': eff['eps'],
            'alpha': eff['alpha'],
        }

    def get_velocity(self, model, x):
        """
        Calcola u, v e p a partire dalle reti neurali.
        """
        if not x.requires_grad:
            x = x.clone().requires_grad_(True)
            
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
        Calcola i residui delle PDE in FORMA ADIMENSIONALE.
        
        Momentum:  Re(u·∇u) + ∇p - β∇²u - ∇·τ = 0
        Costitutiva (Oldroyd-B + PTT + Giesekus):
            f·τ + Wi·∇̊τ + α·Wi/(1-β)·(τ·τ) - 2(1-β)·D = 0
            dove f = 1 + ε·Wi/(1-β)·tr(τ) (PTT coefficient)
        
        I numeri adimensionali (Re, Wi, β) sono calcolati on-the-fly
        dai parametri dimensionali correnti tramite _get_nondim_params().
        
        Ottimizzazioni autograd invariate:
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
        
        # v_yy = -u_yx per il teorema di Schwarz
        v_yy = -u_yx
        
        # Derivate prime di tau
        grad_tau_xx = torch.autograd.grad(tau_xx.sum(), x, create_graph=True)[0]
        tau_xx_x, tau_xx_y = grad_tau_xx[:, 0:1], grad_tau_xx[:, 1:2]
        grad_tau_xy = torch.autograd.grad(tau_xy.sum(), x, create_graph=True)[0]
        tau_xy_x, tau_xy_y = grad_tau_xy[:, 0:1], grad_tau_xy[:, 1:2]
        grad_tau_yy = torch.autograd.grad(tau_yy.sum(), x, create_graph=True)[0]
        tau_yy_x, tau_yy_y = grad_tau_yy[:, 0:1], grad_tau_yy[:, 1:2]
        
        # Parametri adimensionali calcolati on-the-fly
        nd = self._get_nondim_params()
        Re    = nd['Re']
        Wi    = nd['Wi']
        beta  = nd['beta']
        eps   = nd['eps']
        alpha = nd['alpha']
        one_m_beta = 1.0 - beta  # (1-β) = μ_p/μ_tot
        
        # ═══════════════════════════════════════════════════
        # Momentum (Navier-Stokes adimensionale)
        # Re(u·∇u) + ∇p - β∇²u - ∇·τ = 0
        # ═══════════════════════════════════════════════════
        f_u = Re * (u * u_x + v * u_y) + p_x - beta * (u_xx + u_yy) - (tau_xx_x + tau_xy_y)
        f_v = Re * (u * v_x + v * v_y) + p_y - beta * (v_xx + v_yy) - (tau_xy_x + tau_yy_y)

        # ═══════════════════════════════════════════════════
        # Upper-Convected Derivative (stessa struttura, ora pesata da Wi)
        # ═══════════════════════════════════════════════════
        upper_xx = (u * tau_xx_x + v * tau_xx_y - 2 * u_x * tau_xx - 2 * u_y * tau_xy)
        upper_yy = (u * tau_yy_x + v * tau_yy_y - 2 * v_x * tau_xy - 2 * v_y * tau_yy)
        upper_xy = (u * tau_xy_x + v * tau_xy_y - u_x * tau_xy - u_y * tau_yy - tau_xx * v_x - tau_xy * v_y)

        # PTT coefficient: f = 1 + ε·Wi/(1-β)·tr(τ)
        PTT_coeff = 1.0 + eps * Wi / one_m_beta.clamp(min=1e-8) * (tau_xx + tau_yy)

        # Giesekus coefficient: α·Wi/(1-β)
        G_coeff = alpha * Wi / one_m_beta.clamp(min=1e-8)

        # ═══════════════════════════════════════════════════
        # Equazioni Costitutive adimensionali
        # f·τ + Wi·∇̊τ + G·(τ·τ) - 2(1-β)·D = 0
        # ═══════════════════════════════════════════════════
        f_tau_xx = PTT_coeff * tau_xx + Wi * upper_xx + G_coeff * (tau_xx**2 + tau_xy**2) - 2 * one_m_beta * u_x
        f_tau_yy = PTT_coeff * tau_yy + Wi * upper_yy + G_coeff * (tau_xy**2 + tau_yy**2) - 2 * one_m_beta * v_y
        f_tau_xy = PTT_coeff * tau_xy + Wi * upper_xy + G_coeff * tau_xy * (tau_xx + tau_yy) - one_m_beta * (u_y + v_x)
        
        return f_u, f_v, f_tau_xx, f_tau_yy, f_tau_xy

    def residual(self, model, x, pde_weights=None, variance_weights=None):
        """
        Calcola la somma pesata delle Loss sui residui delle PDE.
        """
        weights = pde_weights if pde_weights is not None else self.pde_weights #Pesa di default se non passati diversamente
        w_m = weights.get('momentum', 10.0)
        w_c = weights.get('constitutive', 1.0)

        vw = variance_weights if variance_weights is not None else {} #Pesi delle singole componenti ad 1 se non passati diversamente
        v_u = vw.get('u', 1.0)
        v_v = vw.get('v', 1.0)
        v_txx = vw.get('txx', 1.0)
        v_tyy = vw.get('tyy', 1.0)
        v_txy = vw.get('txy', 1.0)
        
        f_u, f_v, f_tau_xx, f_tau_yy, f_tau_xy = self.compute_residuals(model, x) #calcolo residui da sopra
                
        # Loss Momentum (Navier-Stokes) possiamo usare mean invece che nn.MSEloss perchè vogliamo la loss=0
        loss_u = (f_u ** 2 / max(v_u, 1e-8)).mean()
        loss_v = (f_v ** 2 / max(v_v, 1e-8)).mean()
        loss_m = loss_u + loss_v
        
        # Loss Costitutiva (Oldroyd-B)
        loss_txx = (f_tau_xx ** 2 / max(v_txx, 1e-8)).mean()
        loss_tyy = (f_tau_yy ** 2 / max(v_tyy, 1e-8)).mean()
        loss_txy = (f_tau_xy ** 2 / max(v_txy, 1e-8)).mean()
        loss_c = loss_txx + loss_tyy + loss_txy

        return w_m * loss_m + w_c * loss_c

    def boundary_loss(self, model, x_bc, target_bc, variance_weights=None, active_bcs=None):
        """
        Calcola la funzione di costo (Loss) basata sull'Errore Quadratico Medio (MSE) 
        sui punti di contorno (boundary points).
        """
        if not x_bc.requires_grad:
            x_bc = x_bc.clone().requires_grad_(True) #check di sicurezza per calcolo gradienti
        
        u, v, p, tau = self.get_velocity(model, x_bc) #previsioni del modello
        
        pred_bc = torch.cat([u, v, p, tau], dim=1) #Tensore Npunti x 6 [u, v, p, tau_xx, tau_xy, tau_yy]
        device = pred_bc.device
        
        dir_target, neu_target, normals = target_bc #split target
        nx, ny = normals[:, 0:1], normals[:, 1:2] #vettori normali
        keys = ['u', 'v', 'p', 'txx', 'txy', 'tyy'] # ordine variabili
        
        var_w = torch.ones((1, 6), device=device)
        active_mask = torch.ones((1, 6), dtype=torch.bool, device=device) #ottimizzazione per broadcasting per solo le bc attive o non nulle
        
        if variance_weights is not None: #normalizzazione dei contributi sulla varianza
            for i, k in enumerate(keys):
                var_w[0, i] = variance_weights.get(k, 1.0) 
                
        if active_bcs is not None: #impostiamo quali bc sono attive
            for i, k in enumerate(keys):
                active_mask[0, i] = (k in active_bcs) 

        total_bc_loss = 0.0

        # --- 3. LOSS DI DIRICHLET ---
        for i in range(6):
            valid_dir_i = (~torch.isnan(dir_target[:, i:i+1])) & active_mask[:, i:i+1]
            mask_i = valid_dir_i.float()
            if mask_i.sum() > 0:
                diff_i = pred_bc[:, i:i+1] - torch.nan_to_num(dir_target[:, i:i+1], nan=0.0)
                sq_diff_i = (diff_i ** 2) / var_w[0, i]
                total_bc_loss += (sq_diff_i * mask_i).sum() / mask_i.sum().clamp_min(1.0)

        # --- 4. LOSS DI NEUMANN ---
        # Cache per sapere quali colonne hanno condizioni al contorno di Neumann (non-NaN)
        # in modo da evitare la sincronizzazione GPU-CPU ripetuta (incompatibile con torch.compile).
        if not hasattr(self, '_neu_active_mask_cache'):
            self._neu_active_mask_cache = {}
        
        neu_cache_key = id(neu_target)
        if neu_cache_key not in self._neu_active_mask_cache:
            self._neu_active_mask_cache[neu_cache_key] = [
                bool((~torch.isnan(neu_target[:, j])).any().item())
                for j in range(6)
            ]
        
        has_neu_data = self._neu_active_mask_cache[neu_cache_key]

        for i in range(6):
            if not has_neu_data[i]:
                continue
            if active_bcs is not None and keys[i] not in active_bcs:
                continue
                
            pred_i = pred_bc[:, i:i+1]
            grad_pred = torch.autograd.grad(pred_i.sum(), x_bc, create_graph=True)[0] #grad i-esima variabile
            
            normal_deriv = grad_pred[:, 0:1] * nx + grad_pred[:, 1:2] * ny # calcolo della derivata
            
            diff_neu = normal_deriv - torch.nan_to_num(neu_target[:, i:i+1], nan=0.0) # differenza + sistituiamo i NaN
            
            sq_diff_neu = (diff_neu ** 2) / var_w[0, i] #quadrato pesato
            
            valid_neu_i = (~torch.isnan(neu_target[:, i:i+1])) & active_mask[:, i:i+1]
            mask_i = valid_neu_i.float()
            total_bc_loss += (sq_diff_neu * mask_i).sum() / mask_i.sum().clamp_min(1.0) #operazioni per sommare solo i valori corretti

        return total_bc_loss