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
        'dirichlet': {'u': 'csv', 'v': 'csv', 'tau_xx': '0', 'tau_xy': '0', 'tau_yy': '0'},
        'neumann': {}
    },
    'Walls': {
        'dirichlet': {'u': 0.0, 'v': 0.0},
        #'neumann': {'p': 0.0}
    },
    'Outlet': {
        'dirichlet': {'u': 0},
        'custom': ['outlet-stress']
        #'neumann': {'tau_xx': 0.0, 'tau_xy': 0.0, 'tau_yy': 0.0}
    },
    'Walls-dritte':{
        'dirichlet': {'u': 0.0, 'v': 0.0},
        #'neumann': {'p': 0.0}
    }
}


class ViscoelasticPhysics(nn.Module):
    def __init__(self, mu_s=0.005, mu_p=0.005, lam=1.0, eps=0.0, alpha=0.0, rho=1.0,
                 U_ref=1.0, H_ref=1.0,
                 pde_weights=None, inverse_mode=False,
                 real_mu_s=None, real_mu_p=None, real_lam=None, real_eps=None, real_alpha=None,
                 bc_rules=None):
        """
        Modulo per calcolare i residui fisici in forma ADIMENSIONALE.
        
        La rete neurale prevede direttamente il tensore degli sforzi adimensionale tau.
        Il parametro (1-beta) = mu_p / mu_tot appare esplicitamente nelle equazioni
        costitutive per bilanciare il contributo polimerico e la viscosità del solvente.
        """
        super().__init__()
        self.inverse_mode = inverse_mode
        self.U_ref = U_ref
        self.H_ref = H_ref
        self.bc_rules = bc_rules or DEFAULT_BC_RULES
        self._boundary_metadata = None
        
        if inverse_mode:
            self.mu_s = nn.Parameter(torch.tensor([mu_s], dtype=torch.float32))
            self.mu_p = nn.Parameter(torch.tensor([mu_p], dtype=torch.float32))
            self.lam = nn.Parameter(torch.tensor([lam], dtype=torch.float32))
            self.eps = nn.Parameter(torch.tensor([eps], dtype=torch.float32))
            self.alpha = nn.Parameter(torch.tensor([alpha], dtype=torch.float32))
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
        return {'mu_s': self.mu_s, 'mu_p': self.mu_p, 'lam': self.lam, 'eps': self.eps, 'alpha': self.alpha}

    def get_logged_parameters(self):
        eff = self._get_effective_params()
        return {k: v.item() if hasattr(v, 'item') else float(v) for k, v in eff.items()}

    def _get_nondim_params(self):
        eff = self._get_effective_params()
        # Per evitare disallineamenti di scala con il dataset (che è adimensionalizzato usando i valori reali),
        # usiamo real_mu_s + real_mu_p come riferimento costante.
        real_mu_tot = self.real_mu_s + self.real_mu_p
        return {
            'Re': self.rho * self.U_ref * self.H_ref / real_mu_tot,
            'Wi': eff['lam'] * self.U_ref / self.H_ref,
            'beta': eff['mu_s'] / real_mu_tot,
            'eps': eff['eps'],
            'alpha': eff['alpha'],
            'beta_poly': eff['mu_p'] / real_mu_tot,
            'etas': eff['mu_s'],
            'etap': eff['mu_p'],
        }

    def get_velocity(self, model, x):
        """Restituisce u, v, p e lo stress adimensionale tau."""
        if not x.requires_grad: x = x.clone().requires_grad_(True)
        out = model(x)
        psi, p, tau = out[:, 0:1], out[:, 1:2], out[:, 2:5]
        grad_psi = torch.autograd.grad(psi.sum(), x, create_graph=True)[0]
        u, v = grad_psi[:, 1:2], -grad_psi[:, 0:1]
        return u, v, p, tau

    def compute_residuals(self, model, x):
        """Calcola i residui PDE adimensionali operando direttamente su tau."""
        # 1. Recupero parametri e output rete
        nd = self._get_nondim_params()
        Re, Wi, beta, eps, alpha = nd['Re'], nd['Wi'], nd['beta'], nd['eps'], nd['alpha']
        beta_poly = nd['beta_poly']
        
        out = model(x)
        psi, p, tau = out[:, 0:1], out[:, 1:2], out[:, 2:5]
        tau_xx, tau_xy, tau_yy = tau[:, 0:1], tau[:, 1:2], tau[:, 2:3]
        
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
        
        # 4. Derivate Stress
        g_txx = torch.autograd.grad(tau_xx.sum(), x, create_graph=True)[0]
        g_txy = torch.autograd.grad(tau_xy.sum(), x, create_graph=True)[0]
        g_tyy = torch.autograd.grad(tau_yy.sum(), x, create_graph=True)[0]
        tau_xx_x, tau_xx_y = g_txx[:, 0:1], g_txx[:, 1:2]
        tau_xy_x, tau_xy_y = g_txy[:, 0:1], g_txy[:, 1:2]
        tau_yy_x, tau_yy_y = g_tyy[:, 0:1], g_tyy[:, 1:2]
        
        # 5. Momentum (Navier-Stokes)
        f_u = Re * (u * u_x + v * u_y) + p_x - beta * (u_xx + u_yy) - (tau_xx_x + tau_xy_y)
        f_v = Re * (u * v_x + v * v_y) + p_y - beta * (v_xx + v_yy) - (tau_xy_x + tau_yy_y)
 
        # 6. Costitutive (Oldroyd-B / PTT / Giesekus)
        # Il parametro beta_poly bilancia lo scaling fisico dello sforzo polimerico
        f_PTT = 1.0 + (eps * Wi / beta_poly) * (tau_xx + tau_yy)
        
        upper_xx = (u * tau_xx_x + v * tau_xx_y - 2 * u_x * tau_xx - 2 * u_y * tau_xy)
        upper_yy = (u * tau_yy_x + v * tau_yy_y - 2 * v_x * tau_xy - 2 * v_y * tau_yy)
        upper_xy = (u * tau_xy_x + v * tau_xy_y - u_x * tau_xy - u_y * tau_yy - tau_xx * v_x - tau_xy * v_y)
 
        f_txx = f_PTT * tau_xx + Wi * upper_xx + (alpha * Wi / beta_poly) * (tau_xx**2 + tau_xy**2) - 2.0 * beta_poly * u_x
        f_tyy = f_PTT * tau_yy + Wi * upper_yy + (alpha * Wi / beta_poly) * (tau_xy**2 + tau_yy**2) - 2.0 * beta_poly * v_y
        f_txy = f_PTT * tau_xy + Wi * upper_xy + (alpha * Wi / beta_poly) * tau_xy * (tau_xx + tau_yy) - beta_poly * (u_y + v_x)
        
        return f_u, f_v, f_txx, f_tyy, f_txy
 
    def residual(self, model, x, pde_weights=None, variance_weights=None):
        weights = pde_weights or self.pde_weights
        vw = variance_weights or {}
        
        f_u, f_v, f_txx, f_tyy, f_txy = self.compute_residuals(model, x)
                
        loss_m = (f_u**2 + f_v**2).mean()
        loss_c = (f_txx**2 / max(vw.get('tau_xx', 1.0), 1e-8)).mean() + \
                 (f_tyy**2 / max(vw.get('tau_yy', 1.0), 1e-8)).mean() + \
                 (f_txy**2 / max(vw.get('tau_xy', 1.0), 1e-8)).mean()
 
        return weights.get('momentum', 10.0) * loss_m + weights.get('constitutive', 1.0) * loss_c
 
    def boundary_loss(self, model, x_bc, target_bc, variance_weights=None, active_bcs=None, group_weights=None):
            """
            Orchestratore della loss al contorno. Affetta i mega-tensori in base
            ai gruppi fisici (es. Inlet, Walls) e delega il calcolo dell'errore.
            """
            # 1. Inizializzazione sicura dei dizionari (evita errori se l'utente passa None)
            variance_weights = variance_weights or {}
            group_weights = group_weights or {}
            
            # Unpacking pulito dei tensori target e delle normali
            dir_target, neu_target, normals = target_bc
            nx = normals[:, 0:1]
            ny = normals[:, 1:2]
            
            # 2. Vettore dei pesi di varianza (bilanciamento tra unità di misura diverse)
            keys = ['u', 'v', 'p', 'tau_xx', 'tau_xy', 'tau_yy']
            var_w = torch.ones((1, 6), device=x_bc.device)
            for i, key in enumerate(keys):
                var_w[0, i] = variance_weights.get(key, 1.0)
                    
            total_bc_loss = torch.tensor(0.0, device=x_bc.device, dtype=x_bc.dtype)
            per_group_losses = {}
 
            # 3. Fallback di emergenza (se manca il metadata, processa tutto in un colpo solo)
            if not hasattr(self, '_boundary_metadata') or not self._boundary_metadata:
                if not x_bc.requires_grad: 
                    x_bc = x_bc.clone().requires_grad_(True)
                
                u, v, p, tau = self.get_velocity(model, x_bc)
                pred_bc = torch.cat([u, v, p, tau], dim=1) 
                raw_loss = self._compute_raw_bc_loss(pred_bc, x_bc, dir_target, neu_target, nx, ny, var_w, active_bcs, keys)
                return raw_loss, {"loss_bc_all": raw_loss.item() if hasattr(raw_loss, 'item') else float(raw_loss)}
 
            # 4. Ciclo di Slicing sui Gruppi Fisici
            start_idx = 0
            for group_name, num_points in self._boundary_metadata:
                end_idx = start_idx + num_points
                
                # Peso di importanza del gruppo (es. forzare l'Inlet pesa 10, il Wall pesa 1)
                g_weight = group_weights.get(group_name, 1.0)
                
                # --- AFFETTAMENTO (SLICING) DEI TENSORI ---
                # Estraiamo SOLO i dati geometrici e target di questo specifico contorno.
                dir_slice = dir_target[start_idx:end_idx]
                neu_slice = neu_target[start_idx:end_idx]
                nx_slice = nx[start_idx:end_idx]
                ny_slice = ny[start_idx:end_idx]
                
                # Creiamo un nuovo nodo nel grafo computazionale isolato per questo gruppo
                x_slice = x_bc[start_idx:end_idx].clone().requires_grad_(True)
                
                # --- FORWARD PASS LOCALE ---
                # Interroghiamo la rete neurale fornendole SOLO le coordinate di questo gruppo
                u_g, v_g, p_g, tau_g = self.get_velocity(model, x_slice)
                pred_slice = torch.cat([u_g, v_g, p_g, tau_g], dim=1)
                
                # --- CALCOLO MATEMATICO DELL'ERRORE ---
                g_loss = self._compute_raw_bc_loss(
                    pred=pred_slice, 
                    x=x_slice, 
                    dir_t=dir_slice, 
                    neu_t=neu_slice, 
                    nx=nx_slice, 
                    ny=ny_slice, 
                    var_w=var_w, 
                    active_bcs=active_bcs, 
                    keys=keys
                )
                
                # --- AGGIUNTA CONDIZIONI PERSONALIZZATE ---
                if group_name in self.bc_rules:
                    rules = self.bc_rules[group_name]
                    if 'custom' in rules:
                        for custom_rule in rules['custom']:
                            if custom_rule == 'normal_stress':
                                nd = self._get_nondim_params()
                                beta = nd['beta']
                                eta_p=nd['etap']
                                grad_u = torch.autograd.grad(u_g.sum(), x_slice, create_graph=True)[0]
                                u_x = grad_u[:, 0:1]
                                res_ns = p_g - 2.0 * beta*eta_p * u_x - tau_g[:, 0:1]
                                g_loss += (res_ns ** 2).mean()
                
                # --- AGGIORNAMENTO TOTALI ---
                per_group_losses[f"loss_bc_{group_name}"] = g_loss.item() if hasattr(g_loss, 'item') else float(g_loss)
                total_bc_loss += g_weight * g_loss
                
                # Avanzamento del puntatore per il prossimo gruppo
                start_idx = end_idx

            return total_bc_loss, per_group_losses

    def _compute_raw_bc_loss(self, pred, x, dir_t, neu_t, nx, ny, var_w, active_bcs, keys):
            """
            Calcola l'errore quadratico medio (MSE) sui bordi.
            
            Args:
                pred: Tensore delle predizioni [u, v, p, tau_xx, tau_xy, tau_yy]
                x: Coordinate spaziali (richiede gradienti per Neumann)
                dir_t, neu_t: Tensori target per Dirichlet e Neumann (con NaN nei punti liberi)
                nx, ny: Componenti del vettore normale uscente al contorno
                var_w: Pesi di varianza per bilanciare le loss delle diverse variabili
                active_bcs: (Opzionale) Lista dei campi da forzare
                keys: Nomi delle variabili corrispondenti agli indici 0-5
            """
            total_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

            for i, var_name in enumerate(keys):
                # 1. Filtro variabili attive
                if active_bcs and var_name not in active_bcs:
                    continue

                var_weight = var_w[0, i]
                pred_val = pred[:, i:i+1]  # Estraiamo la singola variabile (es. solo 'u')

                # ==============================================================================
                # A. CONDIZIONI DI DIRICHLET (Valore Imposto, es. u = 0 al muro)
                # ==============================================================================
                target_d = dir_t[:, i:i+1]
                mask_d = (~torch.isnan(target_d)).float()
                valid_points_d = mask_d.sum()

                if valid_points_d > 0:
                    # Dobbiamo pulire il target sostituendo i NaN con 0 PRIMA di sottrarre.
                    clean_target_d = torch.nan_to_num(target_d, nan=0.0)
                    
                    diff_d = pred_val - clean_target_d
                    squared_error_d = (diff_d ** 2) / var_weight
                    
                    # Moltiplichiamo per la maschera (azzera gli errori fittizi) e facciamo la media
                    loss_d = (mask_d * squared_error_d).sum() / valid_points_d
                    total_loss += loss_d

                # ==============================================================================
                # B. CONDIZIONI DI NEUMANN (Flusso Imposto, es. derivata normale nulla)
                # ==============================================================================
                target_n = neu_t[:, i:i+1]
                mask_n = (~torch.isnan(target_n)).float()
                valid_points_n = mask_n.sum()

                if valid_points_n > 0:
                    # 1. Calcolo del gradiente spaziale: ∇f = (∂f/∂x, ∂f/∂y)
                    # create_graph=True è essenziale perché questo gradiente farà parte della Loss, 
                    # e durante la backpropagation servirà calcolare il gradiente di questo gradiente.
                    grad_pred = torch.autograd.grad(pred_val.sum(), x, create_graph=True, retain_graph=True)[0]
                    
                    grad_x = grad_pred[:, 0:1]
                    grad_y = grad_pred[:, 1:2]
                    
                    # 2. Derivata Direzionale: ∇f · n = (∂f/∂x)*nx + (∂f/∂y)*ny
                    directional_derivative = (grad_x * nx) + (grad_y * ny)
                    
                    # 3. Calcolo dell'errore (stessa logica anti-NaN usata per Dirichlet)
                    clean_target_n = torch.nan_to_num(target_n, nan=0.0)
                    
                    diff_n = directional_derivative - clean_target_n
                    squared_error_n = (diff_n ** 2) / var_weight
                    
                    loss_n = (mask_n * squared_error_n).sum() / valid_points_n
                    total_loss += loss_n

            return total_loss

    def apply_boundary_conditions(self, boundary_groups):
        bc_rules = self.bc_rules
        FIELD_TO_COL = {'u': 0, 'v': 1, 'p': 2, 'tau_xx': 3, 'tau_xy': 4, 'tau_yy': 5}
        
        # Pre-assegnazione di device e tipo per garantire coerenza con il modello
        target_device = self.mu_s.device
        target_dtype = self.mu_s.dtype
        
        # Liste che conterranno i tensori parziali prima dell'unione finale
        all_xy, all_dirichlet, all_neumann, all_normals = [], [], [], []
        self._boundary_metadata = []
        
        # Lista per tracciare i bordi orfani
        unmatched_groups = []

        # Intestazione Report
        print("\n" + "="*70)
        print(f"{'BC AUDIT REPORT':^70}")
        print("-" * 70)
        print(f"{'Group Name':<25} | {'Points':<8} | {'Active Rules'}")
        
        for group_name, group_data in boundary_groups.items():
            # 1. MATCHING RIGOROSO (Strict Match)
            # Il nome esportato dalla mesh deve coincidere perfettamente con la chiave nel dizionario
            if group_name not in bc_rules:
                unmatched_groups.append(group_name)
                continue
                
            # 2. Inizializzazione tensori per il gruppo corrente
            num_points = group_data['indices'].numel()
            rules = bc_rules[group_name]
            
            # Creiamo tensori riempiti di NaN
            dirichlet_tensor = torch.full((num_points, 6), float('nan'), device=target_device, dtype=target_dtype)
            neumann_tensor = torch.full((num_points, 6), float('nan'), device=target_device, dtype=target_dtype)

            # 3. Formattazione log per la console
            active_rules_str = []
            for bc_type in ['dirichlet', 'neumann']:
                if rules.get(bc_type):
                    fields_str = ','.join(rules[bc_type].keys())
                    active_rules_str.append(f"{bc_type[0].upper()}:{fields_str}")
            if rules.get('custom'):
                custom_str = ','.join(rules['custom'])
                active_rules_str.append(f"C:{custom_str}")
            print(f"{group_name:<25} | {num_points:<8} | {' '.join(active_rules_str)}")

            # 4. Popolamento dei tensori con i valori target
            condition_types = [('dirichlet', dirichlet_tensor), ('neumann', neumann_tensor)]
            
            for cond_type, target_tensor in condition_types:
                type_rules = rules.get(cond_type, {})
                
                for field, value in type_rules.items():
                    if field not in FIELD_TO_COL:
                        continue
                        
                    col_idx = FIELD_TO_COL[field]
                    
                    if value == 'csv':
                        # Condizione variabile da dati pre-calcolati (es. profilo Inlet velocità)
                        field_data = group_data['fields'][field].to(target_device, target_dtype).view(-1)
                        target_tensor[:, col_idx] = field_data
                    else:
                        # Condizione costante (es. no-slip ai muri)
                        target_tensor[:, col_idx] = float(value)

            # 5. Salvataggio in lista dei dati elaborati
            all_xy.append(group_data['xy'].to(target_device, target_dtype))
            all_normals.append(group_data['norm'].to(target_device, target_dtype))
            all_dirichlet.append(dirichlet_tensor)
            all_neumann.append(neumann_tensor)
            
            self._boundary_metadata.append((group_name, num_points))

        print("="*70)
        
        # --- SISTEMA DI ALLERTA BORDIN NON MATCHATI ---
        if unmatched_groups:
            print("\n" + "!"*70)
            print(f"{' WARNING: UNMATCHED BOUNDARY GROUPS ':^70}")
            print("!" * 70)
            print("I seguenti gruppi sono presenti nella mesh, ma NON hanno una regola")
            print("corrispondente nel dizionario 'bc_rules'. Verranno ignorati:")
            for ug in unmatched_groups:
                print(f"  -> {ug}")
            print("Se questo e' intenzionale, ignora il messaggio. Altrimenti controlla la nomenclatura")
            print("!"*70 + "\n")

        # 6. Concatenazione finale
        return (
            torch.cat(all_xy, dim=0),
            torch.cat(all_dirichlet, dim=0),
            torch.cat(all_neumann, dim=0),
            torch.cat(all_normals, dim=0)
        )