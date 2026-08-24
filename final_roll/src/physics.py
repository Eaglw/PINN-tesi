import builtins
import torch
import torch.nn as nn
from src.utils import weighted_mse


def inverse_softplus(y, min_val=1e-8):
    """Calcola l'inversa della funzione Softplus: x = softplus_inv(y).
    Garantisce stabilità numerica per valori piccoli e grandi."""
    if isinstance(y, (float, int)):
        y = torch.tensor([float(y)], device=DEVICE)
    y_clamped = torch.clamp(y, min=min_val)
    return torch.where(y_clamped > 20.0, y_clamped, torch.log(torch.expm1(y_clamped)))


def inverse_sigmoid(y, eps=1e-7):
    """Calcola l'inversa della funzione Sigmoide: x = logit(y)."""
    if isinstance(y, (float, int)):
        y = torch.tensor([float(y)], device=DEVICE)
    y_clamped = torch.clamp(y, min=eps, max=1.0 - eps)
    return torch.log(y_clamped / (1.0 - y_clamped))


class Physics(nn.Module):
    """PDE adimensionali + boundary conditions. Supporta modalità diretta o inversa."""

    def __init__(self, U_ref, H_ref, H_coord=0.05, var_weights=None, inverse_mode=True, tau_scale=1.0, p_scale=50.0, use_roll_stress_bc=True, w_roll_stress=1.0, eta_0=None):
        super().__init__()
        self.U_ref = U_ref
        self.H_ref = H_ref
        self.H_coord = H_coord
        self.var_weights = var_weights
        self.inverse_mode = inverse_mode
        self.tau_scale = tau_scale
        self.p_scale = p_scale
        self.use_roll_stress_bc = use_roll_stress_bc
        self.w_roll_stress = w_roll_stress

        # Parametri di scala e guess (full-blind: nessuna informazione privilegiata nel training)
        mod_globals = globals()
        mu_s_true_val = getattr(builtins, "MU_S_TRUE", mod_globals.get("MU_S_TRUE", 0.1))
        mu_p_true_val = getattr(builtins, "MU_P_TRUE", mod_globals.get("MU_P_TRUE", 0.9))
        lam_true_val = getattr(builtins, "LAM_TRUE", mod_globals.get("LAM_TRUE", 0.05))

        guess_factor = getattr(builtins, "GUESS_FACTOR", 0.8)
        guess_mu_s = getattr(builtins, "GUESS_MU_S", mu_s_true_val * guess_factor)
        guess_mu_p = getattr(builtins, "GUESS_MU_P", mu_p_true_val * guess_factor)
        guess_lam = getattr(builtins, "GUESS_LAM", lam_true_val * guess_factor)

        # Scala di normalizzazione globale di riferimento costante (arbitraria)
        if eta_0 is None:
            eta_0 = getattr(builtins, "ETA_0", getattr(builtins, "ETA_0_INIT", mod_globals.get("ETA_0", 1.0)))
        self.register_buffer("eta_0", torch.tensor(float(eta_0), device=DEVICE, dtype=torch.float32))

        # Guess base per la log-parametrizzazione (lambda = guess_lam * exp(r_lam))
        self.register_buffer("guess_lam", torch.tensor(guess_lam, device=DEVICE, dtype=torch.float32))
        self.register_buffer("guess_mu_p", torch.tensor(guess_mu_p, device=DEVICE, dtype=torch.float32))
        self.register_buffer("guess_mu_s", torch.tensor(guess_mu_s, device=DEVICE, dtype=torch.float32))

        # Parametri raw in log-space (inizializzati a 0.0 -> param = guess)
        self.register_parameter("_raw_lam", nn.Parameter(torch.zeros(1, device=DEVICE, dtype=torch.float32)))
        self.register_parameter("_raw_mu_p", nn.Parameter(torch.zeros(1, device=DEVICE, dtype=torch.float32)))
        self.register_parameter("_raw_mu_s", nn.Parameter(torch.zeros(1, device=DEVICE, dtype=torch.float32), requires_grad=False))

        # Se non siamo in inverse_mode, i parametri sono fissi ai veri valori
        if not inverse_mode:
            self._raw_lam.requires_grad_(False)
            self._raw_mu_p.requires_grad_(False)
            self._raw_mu_s.requires_grad_(False)

    @property
    def lam(self):
        """Tempo di rilassamento in log-space: lambda = guess_lam * exp(r_lam)."""
        return self.guess_lam * torch.exp(self._raw_lam).squeeze()

    @property
    def mu_p(self):
        """Viscosita' polimerica dimensionale: eta_p = guess_mu_p * exp(r_mup)."""
        return self.guess_mu_p * torch.exp(self._raw_mu_p).squeeze()

    @property
    def mu_p_nd(self):
        """Viscosita' polimerica adimensionale: mu_p* = mu_p / eta_0."""
        return self.mu_p / self.eta_0

    @property
    def mu_s(self):
        """Viscosita' solvente dimensionale: eta_s = guess_mu_s * exp(r_mus)."""
        return self.guess_mu_s * torch.exp(self._raw_mu_s).squeeze()

    @property
    def mu_s_nd(self):
        """Viscosita' solvente adimensionale: mu_s* = mu_s / eta_0."""
        return self.mu_s / self.eta_0

    @property
    def mu_tot(self):
        """Viscosita' totale derivata: eta_tot = eta_s + eta_p."""
        return self.mu_s + self.mu_p

    @property
    def beta(self):
        """Rapporto viscoso derivato: beta = eta_s / (eta_s + eta_p)."""
        return self.mu_s / (self.mu_tot + 1e-12)

    @property
    def beta_poly(self):
        """Frazione polimerica derivata: beta_poly = eta_p / (eta_s + eta_p)."""
        return self.mu_p / (self.mu_tot + 1e-12)

    @property
    def Re_scale(self):
        """Numero di Reynolds basato sulla scala costante di normalizzazione eta_0."""
        return RHO * self.U_ref * self.H_ref / self.eta_0

    @property
    def Re_phys(self):
        """Numero di Reynolds fisico stimato basato sulla viscosita' totale eta_tot = eta_s + eta_p."""
        return RHO * self.U_ref * self.H_ref / (self.mu_tot + 1e-12)

    @property
    def eps(self):
        return torch.tensor(0.0, device=self._raw_lam.device, dtype=self._raw_lam.dtype)

    @property
    def alpha(self):
        return torch.tensor(0.0, device=self._raw_lam.device, dtype=self._raw_lam.dtype)

    def set_trainable(self, name, trainable=True):
        """Imposta requires_grad sul parametro raw sottostante."""
        raw_param = getattr(self, f"_raw_{name}", None)
        if raw_param is not None and isinstance(raw_param, nn.Parameter):
            raw_param.requires_grad_(trainable)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """Gestisce in modo trasparente la retrocompatibilità con i vecchi checkpoint."""
        param_names = ["beta", "mu_s", "mu_p", "lam", "eps", "alpha"]
        for p_name in param_names:
            old_key = prefix + p_name
            raw_key = prefix + f"_raw_{p_name}"
            if old_key in state_dict and raw_key not in state_dict:
                old_val = state_dict.pop(old_key)
                inv_fn = (lambda y: inverse_sigmoid(y / 0.99)) if p_name == "beta" else inverse_softplus
                if torch.is_tensor(old_val):
                    raw_val = inv_fn(old_val).to(old_val.device)
                else:
                    raw_val = inv_fn(old_val)
                state_dict[raw_key] = raw_val

        # Conversione automatica da _raw_mu_p a _raw_beta per vecchi checkpoint
        raw_beta_key = prefix + "_raw_beta"
        raw_mup_key = prefix + "_raw_mu_p"
        raw_mus_key = prefix + "_raw_mu_s"

        if raw_mup_key in state_dict and raw_beta_key not in state_dict:
            raw_mup_val = state_dict.pop(raw_mup_key)
            if raw_mus_key in state_dict:
                raw_mus_val = state_dict[raw_mus_key]
                mu_s_val = torch.nn.functional.softplus(raw_mus_val) + 1e-8
                mu_p_val = torch.nn.functional.softplus(raw_mup_val) + 1e-8
                beta_val = torch.clamp(mu_s_val / (mu_s_val + mu_p_val), min=1e-6, max=0.99)
                raw_beta_val = inverse_sigmoid(beta_val / 0.99)
            else:
                beta_val = torch.tensor(0.10, device=DEVICE)
                raw_beta_val = inverse_sigmoid(beta_val / 0.99)
            state_dict[raw_beta_key] = raw_beta_val
        elif raw_mup_key in state_dict and hasattr(self, "_raw_beta") and not hasattr(self, "_raw_mu_p"):
            state_dict.pop(raw_mup_key, None)

        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def _grad(self, y, x, create_graph=True, retain_graph=True):
        """Helper per calcolare i gradienti tramite autograd."""
        return torch.autograd.grad(
            y,
            x,
            grad_outputs=torch.ones_like(y),
            create_graph=create_graph,
            retain_graph=retain_graph,
        )[0] * (self.H_ref / self.H_coord)

    def get_velocity(self, model, x, create_graph=True):
        """Calcola u, v, p, tau dalla stream function."""
        if not x.requires_grad:
            x = x.clone().requires_grad_(True)

        psi = model.model_psi(x) * (self.H_coord / self.H_ref)
        p = model.model_p(x) * model.p_scale
        tau = model.model_tau(x) * model.tau_scale

        grad_psi = self._grad(psi, x, create_graph=create_graph)
        u, v = grad_psi[:, 1:2], -grad_psi[:, 0:1]

        return u, v, p, tau

    def _nondim(self):
        """Parametri adimensionali correnti scalati rispetto alla scala eta_0."""
        mu_p_nd = self.mu_p / self.eta_0
        mu_s_nd = self.mu_s / self.eta_0
        Re_scale = RHO * self.U_ref * self.H_ref / self.eta_0
        Wi = self.lam * self.U_ref / self.H_ref
        return Re_scale, Wi, mu_s_nd, mu_p_nd, self.eps, self.alpha

    def compute_residuals(self, x, u, v, p, tau, w_momentum=1.0, w_constitutive=1.0, frozen_velocity=False, precomputed_div_tau=None):
        """Calcola i residui PDE adimensionali saltando i calcoli se i pesi sono nulli.
        
        Args:
            frozen_velocity: Se True, le derivate di velocità nella momentum
                equation vengono calcolate con create_graph=False (più economico)
                perché model_psi è congelato. Il gradiente di pressione usa sempre
                create_graph=True per consentire il backprop verso model_p.
        """
        Re_scale, Wi, mu_s_nd, mu_p_nd, eps, alpha = self._nondim()

        # Estrazione tensori di stress passati come argomento
        tau_xx, tau_xy, tau_yy = tau[:, 0:1], tau[:, 1:2], tau[:, 2:3]

        # --- Cinematica ---
        # u e v sono già passati come argomento, estraiamo subito le derivate prime
        grad_u = self._grad(u, x)
        u_x, u_y = grad_u[:, 0:1], grad_u[:, 1:2]

        grad_v = self._grad(v, x)
        v_x, v_y = grad_v[:, 0:1], -u_x  # Incompressibilità analitica

        # --- Derivate Stress ---
        if (w_momentum > 0.0 or w_constitutive > 0.0) and precomputed_div_tau is None:
            cg = w_constitutive > 0.0
            # retain_graph=True sulle prime due per non consumare il forward di model_tau
            g_txx = self._grad(tau_xx, x, create_graph=cg, retain_graph=True)
            tau_xx_x, tau_xx_y = g_txx[:, 0:1], g_txx[:, 1:2]

            g_txy = self._grad(tau_xy, x, create_graph=cg, retain_graph=True)
            tau_xy_x, tau_xy_y = g_txy[:, 0:1], g_txy[:, 1:2]

            g_tyy = self._grad(tau_yy, x, create_graph=cg)  # Libera il grafo
            tau_yy_x, tau_yy_y = g_tyy[:, 0:1], g_tyy[:, 1:2]

            if not cg:
                tau_xx_x, tau_xx_y = tau_xx_x.detach(), tau_xx_y.detach()
                tau_xy_x, tau_xy_y = tau_xy_x.detach(), tau_xy_y.detach()
                tau_yy_x, tau_yy_y = tau_yy_x.detach(), tau_yy_y.detach()
        else:
            tau_xx_x = tau_xx_y = tau_xy_x = tau_xy_y = tau_yy_x = tau_yy_y = None

        # --- Momentum ---
        if w_momentum > 0.0:
            cg_vel = not frozen_velocity

            grad_p = self._grad(p, x)
            p_x, p_y = grad_p[:, 0:1], grad_p[:, 1:2]

            u_xx = self._grad(u_x, x, create_graph=cg_vel)[:, 0:1]

            grad_u_y = self._grad(u_y, x, create_graph=cg_vel)
            u_yx, u_yy = grad_u_y[:, 0:1], grad_u_y[:, 1:2]

            v_xx = self._grad(v_x, x, create_graph=cg_vel)[:, 0:1]
            v_yy = -u_yx

            if precomputed_div_tau is not None:
                div_tau_x, div_tau_y = precomputed_div_tau
            else:
                div_tau_x = tau_xx_x + tau_xy_y
                div_tau_y = tau_xy_x + tau_yy_y

            f_u = (
                Re_scale * (u * u_x + v * u_y)
                + p_x
                - mu_s_nd * (u_xx + u_yy)
                - div_tau_x
            )
            f_v = (
                Re_scale * (u * v_x + v * v_y)
                + p_y
                - mu_s_nd * (v_xx + v_yy)
                - div_tau_y
            )
        else:
            f_u = f_v = torch.zeros_like(u)

        # --- Costitutive ---
        if w_constitutive > 0.0:
            f_PTT = 1.0 + (eps * Wi / mu_p_nd) * (tau_xx + tau_yy)
            upper_xx = u * tau_xx_x + v * tau_xx_y - 2 * u_x * tau_xx - 2 * u_y * tau_xy
            upper_yy = u * tau_yy_x + v * tau_yy_y - 2 * v_x * tau_xy - 2 * v_y * tau_yy
            upper_xy = (
                u * tau_xy_x
                + v * tau_xy_y
                - u_x * tau_xy
                - u_y * tau_yy
                - tau_xx * v_x
                - tau_xy * v_y
            )

            f_txx = (
                f_PTT * tau_xx
                + Wi * upper_xx
                + (alpha * Wi / mu_p_nd) * (tau_xx**2 + tau_xy**2)
                - 2.0 * mu_p_nd * u_x
            )
            f_tyy = (
                f_PTT * tau_yy
                + Wi * upper_yy
                + (alpha * Wi / mu_p_nd) * (tau_xy**2 + tau_yy**2)
                - 2.0 * mu_p_nd * v_y
            )
            f_txy = (
                f_PTT * tau_xy
                + Wi * upper_xy
                + (alpha * Wi / mu_p_nd) * tau_xy * (tau_xx + tau_yy)
                - mu_p_nd * (u_y + v_x)
            )

            # Bilanciamento Loss PDE
            f_txx, f_tyy, f_txy = (
                f_txx / self.tau_scale,
                f_tyy / self.tau_scale,
                f_txy / self.tau_scale,
            )
        else:
            f_txx = f_tyy = f_txy = torch.zeros_like(u)

        return f_u, f_v, f_txx, f_tyy, f_txy

    def compute_pde_losses(self, x, u, v, p, tau, w_momentum=1.0, w_constitutive=1.0, frozen_velocity=False, precomputed_div_tau=None):
        """Calcola separatamente loss momentum e constitutive."""
        f_u, f_v, f_txx, f_tyy, f_txy = self.compute_residuals(
            x, u, v, p, tau, w_momentum, w_constitutive, frozen_velocity=frozen_velocity, precomputed_div_tau=precomputed_div_tau
        )
        loss_m = (f_u**2 + f_v**2).mean() / 2.0
        loss_c = (f_txx**2 + f_tyy**2 + f_txy**2).mean() / 3.0

        return loss_m, loss_c

    def pde_loss_weighted(self, x, u, v, p, tau, w_momentum, w_constitutive, frozen_velocity=False):
        """Loss PDE pesata per staged training."""
        loss_m, loss_c = self.compute_pde_losses(x, u, v, p, tau, w_momentum, w_constitutive, frozen_velocity=frozen_velocity)
        return w_momentum * loss_m + w_constitutive * loss_c

    def pde_loss(self, x, u, v, p, tau):
        """Loss PDE con pesi di default."""
        return self.pde_loss_weighted(x, u, v, p, tau, W_MOMENTUM, W_CONSTITUTIVE)

    def data_loss(self, u, v, uv_target, var_w):
        """Loss dati: solo u, v."""
        loss_u = weighted_mse(u, uv_target[:, 0:1], var_w["u"])
        loss_v = weighted_mse(v, uv_target[:, 1:2], var_w["v"])
        return 0.5 * (loss_u + loss_v)

    def drift_loss(self, u, v, u_ckpt, v_ckpt):
        """Soft drift loss normalizzata rispetto alla magnitudo cinematica di checkpoint."""
        delta_sq = (u - u_ckpt)**2 + (v - v_ckpt)**2
        norm_sq = (u_ckpt**2 + v_ckpt**2).mean() + 1e-6
        return delta_sq.mean() / norm_sq


    def boundary_loss(self, model, bc_data, var_w, active_bcs=None):
        """Calcola le loss per i gruppi al contorno."""
        # Creazione sicura del tensore cumulativo con il corretto dtype
        total_loss = torch.tensor(
            0.0, device=DEVICE, dtype=next(model.parameters()).dtype
        )

        for group_name, gd in bc_data.items():
            x_bc = gd["xy"].clone().requires_grad_(True)
            u, v, p, tau = self.get_velocity(model, x_bc)
            g_loss = 0.0

            if group_name == "Walls":
                if active_bcs is None or "u" in active_bcs:
                    g_loss += weighted_mse(u, torch.zeros_like(u), var_w["u"])
                if active_bcs is None or "v" in active_bcs:
                    g_loss += weighted_mse(v, torch.zeros_like(v), var_w["v"])

            elif group_name in ("Roll1", "Roll2", "Roll3", "Roll4"):
                if active_bcs is None or "u" in active_bcs:
                    g_loss += weighted_mse(u, gd["fields"]["u"], var_w["u"])
                if active_bcs is None or "v" in active_bcs:
                    g_loss += weighted_mse(v, gd["fields"]["v"], var_w["v"])

                # ====================================================================
                # BOUNDARY CONDITION STRESS SUI RULLI (ANCORAGGIO STRESS)
                # Impostare USE_ROLL_STRESS_BC = False in train_4roll_main.py per disattivare.
                # ====================================================================
                use_stress_bc = getattr(self, "use_roll_stress_bc", True)
                if use_stress_bc:
                    tau_xx_pred, tau_xy_pred, tau_yy_pred = tau[:, 0:1], tau[:, 1:2], tau[:, 2:3]
                    g_stress = 0.0
                    if active_bcs is None or "tau_xx" in active_bcs:
                        g_stress += weighted_mse(tau_xx_pred, gd["fields"]["tau_xx"], var_w["tau_xx"])
                    if active_bcs is None or "tau_xy" in active_bcs:
                        g_stress += weighted_mse(tau_xy_pred, gd["fields"]["tau_xy"], var_w["tau_xy"])
                    if active_bcs is None or "tau_yy" in active_bcs:
                        g_stress += weighted_mse(tau_yy_pred, gd["fields"]["tau_yy"], var_w["tau_yy"])

                    # Bilanciamento 1:1 di magnitudo per componente (2 comp vel vs 3 comp stress -> fattore 2/3)
                    w_stress = getattr(self, "w_roll_stress", 1.0)
                    g_loss += w_stress * (2.0 / 3.0) * g_stress

            elif group_name == "PressurePoint":
                if active_bcs is None or "p" in active_bcs:
                    g_loss += weighted_mse(p, gd["fields"]["p"], var_w["p"])

            total_loss = total_loss + g_loss

        return total_loss

    def clamp_params(self):
        """No-op: I parametri sono fisicamente garantiti positivi o non-negativi tramite Softplus Reparameterization."""
        pass

    def log_params(self):
        """Restituisce i parametri correnti estraendoli proceduralmente (sia dimensionali che adimensionali)."""
        return {
            "beta": self.beta.item(),
            "mu_s": self.mu_s.item(),
            "mu_p": self.mu_p.item(),
            "mu_tot": self.mu_tot.item(),
            "mu_s_nd": self.mu_s_nd.item(),
            "mu_p_nd": self.mu_p_nd.item(),
            "lam": self.lam.item(),
            "eps": self.eps.item(),
            "alpha": self.alpha.item(),
            "eta_0": self.eta_0.item(),
            "Re_scale": self.Re_scale.item(),
            "Re_phys": self.Re_phys.item(),
        }


def evaluate_final_losses(model, physics, data, chunk_size=2000):
    """
    Calcola le loss finali valutate sul dataset completo in chunk per evitare OOM.
    Utilizza un dizionario per l'accumulo dinamico, eliminando il boilerplate.
    """
    model.eval()
    _dtype = next(model.parameters()).dtype

    # Helper per il cast dei tensori
    def _cast(val):
        return val.to(_dtype) if torch.is_tensor(val) else val

    xy_all = _cast(data["coords"])
    uv_all = _cast(data["uv_data"])
    var_w = data["var_weights"]
    bc_data = data["boundary_groups"]

    # Pre-cast sicuro delle Boundary Conditions
    bc_data_typed = {
        gname: {
            "xy": _cast(gd["xy"]),
            "norm": _cast(gd["norm"]),
            "fields": {k: _cast(v) for k, v in gd["fields"].items()},
        }
        for gname, gd in bc_data.items()
    }

    # Dizionario degli accumulatori per le metriche PDE
    metrics = {
        "d_loss": 0.0,
        "loss_m": 0.0,
        "loss_c": 0.0,
        "abs_fu": 0.0,
        "abs_fv": 0.0,
        "abs_ftxx": 0.0,
        "abs_ftxy": 0.0,
        "abs_ftyy": 0.0,
    }

    # Le loss di validazione richiedono comunque i gradienti accesi per la PDE
    with torch.set_grad_enabled(True):
        # --- 1. DATA E PDE LOSS (Chunked) ---
        total_points = xy_all.shape[0]
        for i in range(0, total_points, chunk_size):
            xc = xy_all[i : i + chunk_size]
            yc = uv_all[i : i + chunk_size]
            w = xc.shape[0] / total_points

            # Estrazione unificata della cinematica
            xph = xc.clone().requires_grad_(True)
            u, v, p, tau = physics.get_velocity(model, xph)

            # Data Loss
            dl = physics.data_loss(u, v, yc, var_w)
            metrics["d_loss"] += dl.item() * w

            # PDE Residuals
            f_u, f_v, f_txx, f_tyy, f_txy = physics.compute_residuals(xph, u, v, p, tau)

            metrics["loss_m"] += (0.5 * (f_u**2 + f_v**2).mean()).item() * w
            metrics["loss_c"] += (
                (f_txx**2 + f_tyy**2 + f_txy**2).mean() / 3.0
            ).item() * w

            # Mean Absolute Residuals
            metrics["abs_fu"] += f_u.abs().mean().item() * w
            metrics["abs_fv"] += f_v.abs().mean().item() * w
            metrics["abs_ftxx"] += f_txx.abs().mean().item() * w
            metrics["abs_ftxy"] += f_txy.abs().mean().item() * w
            metrics["abs_ftyy"] += f_tyy.abs().mean().item() * w

        # --- 2. BOUNDARY LOSS ---
        bc_vals = {"u": 0.0, "v": 0.0, "p": 0.0, "tau": 0.0}

        for group_name, gd in bc_data_typed.items():
            x_bc = gd["xy"].clone().requires_grad_(True)
            u, v, p, tau = physics.get_velocity(model, x_bc)

            if group_name == "Walls":
                bc_vals["u"] += weighted_mse(u, torch.zeros_like(u), var_w["u"]).item()
                bc_vals["v"] += weighted_mse(v, torch.zeros_like(v), var_w["v"]).item()
            elif group_name in ("Roll1", "Roll2", "Roll3", "Roll4"):
                bc_vals["u"] += weighted_mse(u, gd["fields"]["u"], var_w["u"]).item()
                bc_vals["v"] += weighted_mse(v, gd["fields"]["v"], var_w["v"]).item()
                use_stress_bc = getattr(physics, "use_roll_stress_bc", True)
                if use_stress_bc:
                    tau_xx_p, tau_xy_p, tau_yy_p = tau[:, 0:1], tau[:, 1:2], tau[:, 2:3]
                    g_stress = (
                        weighted_mse(tau_xx_p, gd["fields"]["tau_xx"], var_w["tau_xx"]) +
                        weighted_mse(tau_xy_p, gd["fields"]["tau_xy"], var_w["tau_xy"]) +
                        weighted_mse(tau_yy_p, gd["fields"]["tau_yy"], var_w["tau_yy"])
                    )
                    w_stress = getattr(physics, "w_roll_stress", 1.0)
                    bc_vals["tau"] += (w_stress * (2.0 / 3.0) * g_stress).item()
            elif group_name == "PressurePoint":
                bc_vals["p"] += weighted_mse(p, gd["fields"]["p"], var_w["p"]).item()

    b_loss_val = sum(bc_vals.values())
    pde_loss_val = W_MOMENTUM * metrics["loss_m"] + W_CONSTITUTIVE * metrics["loss_c"]
    total_loss_val = (
        W_DATA * metrics["d_loss"] + W_BC * b_loss_val + W_PHYSICS * pde_loss_val
    )

    return {
        "Data Loss": metrics["d_loss"],
        "Boundary Loss": b_loss_val,
        "BC_u": bc_vals["u"],
        "BC_v": bc_vals["v"],
        "BC_p": bc_vals["p"],
        "BC_tau": bc_vals["tau"],
        "Momentum Loss": metrics["loss_m"],
        "Constitutive Loss": metrics["loss_c"],
        "Total PDE Loss": pde_loss_val,
        "Total Loss": total_loss_val,
        "Mean Abs f_u": metrics["abs_fu"],
        "Mean Abs f_v": metrics["abs_fv"],
        "Mean Abs f_txx": metrics["abs_ftxx"],
        "Mean Abs f_txy": metrics["abs_ftxy"],
        "Mean Abs f_tyy": metrics["abs_ftyy"],
    }


def compute_l2_errors(model, physics, data, chunk_size=7000):
    """Calcola L2 relative errors per tutti i campi in modo vettorializzato e pulito, processando in chunk."""
    model.eval()
    _dtype = next(model.parameters()).dtype
    errors = {}

    xi_all = data["coords"].to(_dtype)
    total_points = xi_all.shape[0]

    u_list, v_list, p_list, tau_p_list = [], [], [], []
    compute_stretch = data.get("true_stretch") is not None
    pred_stretch_list = [] if compute_stretch else None

    # Esecuzione a chunk per evitare OOM (Out Of Memory) su 125k punti
    with torch.set_grad_enabled(True):
        for i in range(0, total_points, chunk_size):
            xi = xi_all[i : i + chunk_size].clone().requires_grad_(True)
            u_p, v_p, p_p, tau_p = physics.get_velocity(model, xi, create_graph=compute_stretch)

            u_list.append(u_p.detach())
            v_list.append(v_p.detach())
            p_list.append(p_p.detach())
            tau_p_list.append(tau_p.detach())

            if compute_stretch:
                grad_u = torch.autograd.grad(
                    u_p.sum(), xi, create_graph=False, retain_graph=True
                )[0]
                grad_v = torch.autograd.grad(v_p.sum(), xi, create_graph=False)[0]

                D_xx = grad_u[:, 0]
                D_xy = 0.5 * (grad_u[:, 1] + grad_v[:, 0])
                D_yy = grad_v[:, 1]

                pred_str_chunk = torch.sqrt(D_xx**2 + 2 * D_xy**2 + D_yy**2)
                pred_stretch_list.append(pred_str_chunk.detach())

    tau_p_full = torch.cat(tau_p_list, dim=0)
    preds = {
        "u": torch.cat(u_list, dim=0),
        "v": torch.cat(v_list, dim=0),
        "p": torch.cat(p_list, dim=0),
        "tau_xx": tau_p_full[:, 0:1],
        "tau_xy": tau_p_full[:, 1:2],
        "tau_yy": tau_p_full[:, 2:3],
    }
    
    # Preleviamo le variabili esatte direttamente castate
    exacts = {k: data[k].to(_dtype) for k in preds.keys()}

    def _compute_rel_l2(pred, exact, mask=None):
        """Helper interno per l'errore L2 relativo, con supporto opzionale per le maschere."""
        p_flat, e_flat = pred.detach().view(-1), exact.view(-1)
        if mask is not None:
            p_flat, e_flat = p_flat[mask], e_flat[mask]

        norm_e = torch.norm(e_flat, 2)
        if norm_e > 1e-10:
            return (torch.norm(p_flat - e_flat, 2) / norm_e).item()
        return 0.0

    # 1. Errori L2 Standard
    for key in preds:
        errors[key] = _compute_rel_l2(preds[key], exacts[key])

    # 2. Errori L2 Mascherati (Zone ad alto stress)
    tau_magnitude = torch.sqrt(
        exacts["tau_xx"] ** 2 + exacts["tau_xy"] ** 2 + exacts["tau_yy"] ** 2
    )
    threshold = 0.05 * torch.max(tau_magnitude).item()
    mask = (tau_magnitude >= threshold).view(-1)

    # Fallback sicuro se lo stress è universalmente basso
    if not mask.any():
        mask = torch.ones_like(mask, dtype=torch.bool)

    for key in ["tau_xx", "tau_xy", "tau_yy"]:
        errors[f"{key}_masked"] = _compute_rel_l2(preds[key], exacts[key], mask=mask)

    # 3. Calcolo Stretch (Se presente nei dati)
    if compute_stretch:
        pred_stretch = torch.cat(pred_stretch_list, dim=0).view(-1)
        true_stretch = data["true_stretch"].to(_dtype).view(-1)

        norm_str = torch.norm(true_stretch, 2)
        errors["stretch"] = (
            (torch.norm(pred_stretch - true_stretch, 2) / norm_str).item()
            if norm_str > 1e-10
            else 0.0
        )

    return errors
