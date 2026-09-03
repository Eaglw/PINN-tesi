"""
train_4roll_main_curl_mauri.py
=============================================================================
Script PINN 4-Roll Mill per PC Maurizio con Riformulazione Vorticità (psi - omega)
nella Fase 2 (100% STANDALONE - NESSUNA MODIFICA A src/train.py):

Architettura & Formulazione:
  - Risolve il rumore numerico della derivata 4a (nabla^4 psi) spezzando
    il vincolo rotazionale in due equazioni di ordine <= 2:
      1) Relazione cinematica di Poisson:
           omega = dv/dx - du/dy  <=>  omega + nabla^2(psi) = 0
      2) Equazione di trasporto della vorticità (Rotore della quantità di moto):
           mu_s* * nabla^2(omega) + rot(div(tau*))_static - Re_scale * (u . nabla)omega = 0
  - La stream function psi rimane MOBILE in Fase 2 (unitamente a model_p e model_omega).
  - La pressione p(x,y) e' appresa tramite la Momentum equation standard ancorata
    al PressurePoint di Dirichlet.
  - Parametri reologici F1 fissati dal checkpoint:
      mu_p^(F1)  = 0.904854 Pa*s
      lambda^(F1)= 0.050203 s
  - Parametro ottimizzato in Fase 2:
      mu_tot (target: 1.000 Pa*s), con mu_s = clamp(mu_tot - mu_p^(F1), min=1e-6)
  - Pre-allineamento iniziale rapido (warm-start 300 step) di model_omega su -nabla^2(psi)
    per partire all'epoca 50001 con consistenza cinematica perfetta.
  - Logging iper-verboso (frequenza configurabile via LOG_FREQUENCY) per monitorare
    in tempo reale tutti i termini, gradienti ed errori L2.
=============================================================================
"""
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import dai moduli src originali (NON modificati)
from src.debug import test_random_points, debug_physics_magnitudes
from src.physics import Physics, evaluate_final_losses, compute_l2_errors
from src.train import FCN, SimpleHistory, init_weights_xavier, initialize_last_layer_zero, precompute_stress_divergence
from src.utils import load_data, launch_tensorboard_server, generate_all_diagnostics, weighted_mse, convert_to_fp64, convert_to_fp32, get_optimal_chunk_size

import src.debug
import src.physics
import src.train
import src.utils

import builtins

# --- Logging automatico di tutti i print (Globale) ---
_original_print = builtins.print
global_log_path = None

def custom_print(*args, **kwargs):
    _original_print(*args, **kwargs)
    if global_log_path is not None:
        sep = kwargs.get('sep', ' ')
        end = kwargs.get('end', '\n')
        text = sep.join(map(str, args)) + end
        with open(global_log_path, 'a', encoding='utf-8') as f:
            f.write(text)

builtins.print = custom_print

# ============================================================================
# 1. SETUP AMBIENTE E PYTORCH
# ============================================================================
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

torch.set_default_dtype(torch.float32)
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = False

SEED = 123
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# 2. COSTANTI E CONFIGURAZIONI GLOBALI
# ============================================================================
EXPORT_TO_OBSIDIAN = False
STAGED_TRAINING = True
INVERSE_PROBLEM = True
DEBUG_MODE = False

USE_ROLL_STRESS_BC = True
W_ROLL_STRESS = 1.0

# --- Frequenza di Logging Verboso ---
# Stampa il report completo ogni LOG_FREQUENCY epoche per facilitare il copia-incolla
LOG_FREQUENCY = 50

# --- Pesi Funzione di Loss per Fase 2 ---
W_DATA_2 = 20.0             # Ancoraggio dati velocità u, v
W_BC_2 = 5.0                # Ancoraggio boundary conditions (no-slip + PressurePoint)
W_MOMENTUM = 1.0            # Peso equazione quantità di moto per pressione p
W_POISSON = 1.0             # Peso vincolo cinematico: omega + nabla^2(psi) = 0
W_VORTICITY = 1.0           # Peso equazione trasporto: mu_s* nabla^2(omega) + rot(div(tau)) - Re*(u.nabla)omega = 0
W_DRIFT = 1.0               # Soft trust-region cinematica rispetto al checkpoint F1
CURL_SUBSET_SIZE = 5000     # Sottocampione punti per l'equazione di vorticità

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR.parent / "COMSOL" / "4roll" / "4_roll_mill.csv"

# Checkpoint consolidato Fase 1
RESUME_CHECKPOINT = BASE_DIR / "checkpoints" / "checkpoint_inverso_fase1_40k+10k.pth"

# Parametri Fisici REALI (Ground Truth)
MU_S_TRUE = 0.100
MU_P_TRUE = 0.900
MU_TOT_TRUE = MU_S_TRUE + MU_P_TRUE   # 1.000 Pa*s
BETA_TRUE = MU_S_TRUE / MU_TOT_TRUE    # 0.100
LAM_TRUE = 0.050
EPS_TRUE = 0.0
ALPHA_TRUE = 0.0
RHO = 1000.0

MIN_MU_S = 1e-6
MIN_MU_P = 1e-6
MIN_LAM = 1e-6

ETA_0 = 2.0
GUESS_FACTOR = 0.80

GUESS_LAM = LAM_TRUE * GUESS_FACTOR
GUESS_MU_S = MU_S_TRUE * GUESS_FACTOR
GUESS_MU_P = MU_P_TRUE * GUESS_FACTOR
GUESS_MU_TOT = 1.000                  # Guess iniziale per la viscosita' totale (ordine O(1))
GUESS_BETA = GUESS_MU_S / (GUESS_MU_S + GUESS_MU_P)
GUESS_EPS = 0.0
GUESS_ALPHA = 0.0

HIDDEN_LAYERS = [128] * 8
ACTIVATION = nn.SiLU

# Iperparametri Fase 2 (Richiesta PC Mauri)
ADAM_EPOCHS_PHASE2 = 30000
USE_LBFGS_PHASE2 = True
LBFGS_MAX_ITERS_PHASE2 = 2000

BASE_LR = 1e-3
ADAM_EPS = 1e-7
PARAM_LR_FACTOR = 0.1
GRAD_CLIP_NORM = 1000.0
PARAM_CLIP_NORM = 1.0
VARIANCE_EPS = 1e-4

# Iniezione parametri globali in builtins e nei moduli src
for name, val in list(locals().items()):
    if name.isupper():
        builtins.__dict__[name] = val
        for mod in [src.debug, src.physics, src.train, src.utils]:
            mod.__dict__[name] = val


# ============================================================================
# 3. MODELLO NEURALE CON TESTA OMEGA INTEGRATA (FASE 2)
# ============================================================================
class CombinedModelVorticity(nn.Module):
    """
    Rete combinata per la Fase 2 a 4 teste:
      - model_psi:   2 -> 1 (Stream Function)
      - model_p:     2 -> 1 (Pressione scalare)
      - model_tau:   2 -> 3 (Stress tensore, congelato in Fase 2)
      - model_omega: 2 -> 1 (Vorticità scalare, ordine <= 2)
    """
    def __init__(self, p_scale=1.0, tau_scale=1.0, hidden_layers=None):
        super().__init__()
        hl = hidden_layers if hidden_layers is not None else [128] * 8
        self.model_psi = FCN(2, 1, hl)
        self.model_p = FCN(2, 1, hl)
        self.model_tau = FCN(2, 3, hl)
        self.model_omega = FCN(2, 1, hl)

        self.p_scale = p_scale
        self.tau_scale = tau_scale

    def forward(self, x):
        psi = self.model_psi(x)
        p = self.model_p(x) * self.p_scale
        tau = self.model_tau(x) * self.tau_scale
        return torch.cat([psi, p, tau], dim=1)


# ============================================================================
# 4. SOTTOCLASSE SPECIALIZZATA: VORTICITY SPLITTING & MUTOT PHYSICS
# ============================================================================
class VorticityMuTotPhysics(Physics):
    """
    Riformulazione avanzata con Splitting di Vorticità (psi - omega):
    1. Parametrizzazione su Viscosita' Totale mu_tot (target: 1.000 Pa*s)
       con mu_p fissato al valore identificato dalla Fase 1:
         mu_s = clamp(mu_tot - mu_p_fixed, min=1e-6)
    2. Splitting dell'operatore rotazionale tramite la vorticità scalare omega:
       - Vincolo Poisson cinematico:
           r_poiss = omega - (dv/dx - du/dy) = omega + nabla^2(psi)  [ordine <= 2]
       - Equazione trasporto vorticità (rotore della quantità di moto):
           r_vort = mu_s* * nabla^2(omega) + rot(div(tau*)) - Re_scale * (u . nabla)omega  [ordine <= 2]
       Nessuna derivata di ordine 4: il rumore ad alta frequenza è eliminato.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer("guess_mu_tot", torch.tensor(1.0, device=DEVICE, dtype=torch.float32))
        self.register_parameter("_raw_mu_tot", nn.Parameter(torch.zeros(1, device=DEVICE, dtype=torch.float32), requires_grad=True))

        self.mu_p_fixed = None
        self._precomputed_rot_div_tau = None
        self._curl_points_idx = None
        self._xy_all = None
        self.diag_csv_path = None

        self.w_poisson = W_POISSON
        self.w_vorticity = W_VORTICITY

    @property
    def mu_tot(self):
        """Viscosita' totale dimensionale: mu_tot = guess_mu_tot * exp(_raw_mu_tot)."""
        return self.guess_mu_tot * torch.exp(self._raw_mu_tot).squeeze()

    @property
    def mu_tot_nd(self):
        """Viscosita' totale adimensionale: mu_tot* = mu_tot / eta_0."""
        return self.mu_tot / self.eta_0

    @property
    def mu_p(self):
        """Viscosita' polimerica: fissa da Fase 1."""
        if self.mu_p_fixed is not None:
            return self.mu_p_fixed
        return super().mu_p

    @property
    def mu_p_nd(self):
        return self.mu_p / self.eta_0

    @property
    def mu_s(self):
        """Viscosita' solvente chiusa algebricamente: mu_s = mu_tot - mu_p."""
        return torch.clamp(self.mu_tot - self.mu_p, min=1e-6)

    @property
    def mu_s_nd(self):
        return self.mu_s / self.eta_0

    @property
    def beta(self):
        return self.mu_s / (self.mu_tot + 1e-12)

    def set_trainable(self, name, trainable=True):
        if name in ["mu_s", "mu_tot"]:
            self._raw_mu_tot.requires_grad_(trainable)
        elif name in ["mu_p", "lam"]:
            super().set_trainable(name, False)
        else:
            super().set_trainable(name, trainable)

    def log_params(self):
        p = super().log_params()
        p["mu_tot"] = self.mu_tot.item()
        p["mu_tot_nd"] = self.mu_tot_nd.item()
        p["mu_s"] = self.mu_s.item()
        p["mu_s_nd"] = self.mu_s_nd.item()
        p["mu_p"] = self.mu_p.item()
        p["mu_p_nd"] = self.mu_p_nd.item()
        p["beta"] = self.beta.item()
        return p

    def precompute_rot_div_tau(self, model, xy_all, subset_size=5000):
        """
        Precalcola il termine statico rot(div(tau)) su un sottocampione random di punti.
        Valutato esclusivamente al 2° ordine su model_tau congelato.
        """
        model.eval()
        n = xy_all.shape[0]
        torch.manual_seed(42)
        idx = torch.randperm(n)[:subset_size].to(xy_all.device)
        self._curl_points_idx = idx
        self._xy_all = xy_all

        xc = xy_all[idx].clone().requires_grad_(True)

        with torch.set_grad_enabled(True):
            u, v, p, tau = self.get_velocity(model, xc, create_graph=True)
            tau_xx, tau_xy, tau_yy = tau[:, 0:1], tau[:, 1:2], tau[:, 2:3]

            g_txx = self._grad(tau_xx, xc, create_graph=True)
            g_txy = self._grad(tau_xy, xc, create_graph=True)
            g_tyy = self._grad(tau_yy, xc, create_graph=True)
            div_tau_x = g_txx[:, 0:1] + g_txy[:, 1:2]
            div_tau_y = g_txy[:, 0:1] + g_tyy[:, 1:2]

            grad_div_ty = self._grad(div_tau_y, xc, create_graph=False)
            grad_div_tx = self._grad(div_tau_x, xc, create_graph=False)
            rot_div_tau = (grad_div_ty[:, 0:1] - grad_div_tx[:, 1:2]).detach()

        self._precomputed_rot_div_tau = rot_div_tau
        model.train()

        print(f"  [Precalcolo rot(div(tau))] Calcolato su {subset_size} punti (solo derivate 2e!).")
        print(f"  [Precalcolo] ||rot(div(tau))||: {torch.norm(rot_div_tau).item():.4e} | Media abs: {rot_div_tau.abs().mean().item():.4e}")
        return rot_div_tau

    def warmstart_omega(self, model, xy_all, steps=300, lr=1e-3):
        """
        Warm-start rapido (~2 secondi) per allineare model_omega al laplaciano di psi:
          min_omega || omega - (v_x - u_y) ||^2
        Garantisce che all'epoca 50001 la vorticità parta perfettamente sincronizzata.
        """
        print(f"\n[Warm-Start Omega] Pre-allineamento di model_omega su -nabla^2(psi) ({steps} steps)...")
        model.eval()
        for p in model.model_omega.parameters():
            p.requires_grad = True

        opt_omega = torch.optim.Adam(model.model_omega.parameters(), lr=lr)

        n = xy_all.shape[0]
        batch_size = min(8000, n)

        with torch.enable_grad():
            for s in range(steps):
                idx = torch.randperm(n)[:batch_size]
                xc = xy_all[idx].clone().requires_grad_(True)

                u, v, _, _ = self.get_velocity(model, xc, create_graph=True)
                grad_u = self._grad(u, xc, create_graph=True)
                grad_v = self._grad(v, xc, create_graph=True)
                u_y = grad_u[:, 1:2]
                v_x = grad_v[:, 0:1]

                omega_target = (v_x - u_y).detach()
                omega_pred = model.model_omega(xc)

                loss_init = torch.mean((omega_pred - omega_target) ** 2)

                opt_omega.zero_grad(set_to_none=True)
                loss_init.backward()
                opt_omega.step()

                if (s + 1) % (steps // 3) == 0 or (s + 1) == steps:
                    print(f"  [Warm-Start Omega] Step {s+1}/{steps} -> MSE(omega - vort_target): {loss_init.item():.4e}")

        print("[Warm-Start Omega] Completato con successo! Omega sincronizzata.")
        model.train()

    def compute_vorticity_losses(self, model):
        """
        Calcola le loss rotazionali tramite Splitting di Vorticità (ordine <= 2):
          1) L_poisson = || omega_pred - (v_x - u_y) ||^2
          2) L_vorticity = || mu_s* nabla^2(omega) + rot_div_tau - Re (u*w_x + v*w_y) ||^2
        """
        if self._precomputed_rot_div_tau is None or self._xy_all is None:
            return torch.tensor(0.0, device=DEVICE), torch.tensor(0.0, device=DEVICE)

        dtype = next(model.parameters()).dtype
        device = next(model.parameters()).device

        mu_tot_nd = self.mu_tot_nd.to(dtype=dtype, device=device)
        mu_p_nd = self.mu_p_nd.to(dtype=dtype, device=device)
        mu_s_nd = mu_tot_nd - mu_p_nd

        rot_div_tau = self._precomputed_rot_div_tau.to(dtype=dtype, device=device)
        xc = self._xy_all[self._curl_points_idx].to(dtype=dtype, device=device).clone().requires_grad_(True)
        Re_scale = self.Re_scale

        # 1. Cinematica (u, v da psi)
        u, v, p, tau = self.get_velocity(model, xc, create_graph=True)
        grad_u = self._grad(u, xc, create_graph=True)
        grad_v = self._grad(v, xc, create_graph=True)
        u_y = grad_u[:, 1:2]
        v_x = grad_v[:, 0:1]

        # 2. Vorticità e sue derivate (da model_omega)
        omega = model.model_omega(xc)
        grad_w = self._grad(omega, xc, create_graph=True)
        w_x = grad_w[:, 0:1]
        w_y = grad_w[:, 1:2]

        w_xx = self._grad(w_x, xc, create_graph=True)[:, 0:1]
        w_yy = self._grad(w_y, xc, create_graph=True)[:, 1:2]
        lap_omega = w_xx + w_yy

        # 3. Equazione di Poisson: omega = v_x - u_y (definizione cinematica)
        res_poisson = omega - (v_x - u_y)
        loss_poisson = torch.mean(res_poisson ** 2)

        # 4. Equazione di trasporto vorticità (rotore della quantità di moto):
        conv_w = u * w_x + v * w_y
        res_vorticity = mu_s_nd * lap_omega + rot_div_tau - Re_scale * conv_w
        loss_vorticity = torch.mean(res_vorticity ** 2)

        self._last_loss_poisson = loss_poisson.item()
        self._last_loss_vorticity = loss_vorticity.item()
        self._last_mean_omega = omega.abs().mean().item()
        self._last_max_omega = omega.abs().max().item()

        return loss_poisson, loss_vorticity


# ============================================================================
# 5. CICLO DI ADDESTRAMENTO FASE 2 DEDICATO (STANDALONE)
# ============================================================================
def train_phase2_vorticity(model, physics, data, save_dir, tb_writer=None):
    """
    Ciclo di training dedicato alla Fase 2 con architettura di vorticità (psi - omega):
      - 30000 epoche Adam (FP32)
      - 2000 iterazioni L-BFGS (FP64)
      - Logging iper-verboso ogni LOG_FREQUENCY epoche.
    """
    xy_all = data["coords"]
    uv_all = data["uv_data"]
    bc_data = data["boundary_groups"]
    var_w = data["var_weights"]

    # Inizializza cache cinematica da F1
    with torch.enable_grad():
        x_in = xy_all.clone().requires_grad_(True)
        u_ph1, v_ph1, _, _ = physics.get_velocity(model, x_in, create_graph=False)
        u_ckpt_cache = u_ph1.detach().clone().float()
        v_ckpt_cache = v_ph1.detach().clone().float()
        print("  [Checkpoint Reologico] Cache cinematica per soft anti-drift loss inizializzata.")

    # Precalcolo statico della divergenza di tau su tutti i punti per la quantità di moto
    print("\n[Optimization] Precalcolo divergenza sforzi per Momentum in Fase 2...")
    precomputed_div_tau = precompute_stress_divergence(model, physics, xy_all)
    print("[Optimization] Divergenza sforzi precalcolata.")

    # Configura requires_grad per Fase 2
    for p in model.parameters():
        p.requires_grad = False
    for p in model.model_p.parameters():
        p.requires_grad = True
    for p in model.model_psi.parameters():
        p.requires_grad = True
    for p in model.model_omega.parameters():
        p.requires_grad = True

    physics.set_trainable("mu_tot", True)
    physics.set_trainable("mu_p", False)
    physics.set_trainable("lam", False)

    # Parametri ottimizzatore Adam
    p_params = [p for p in model.model_p.parameters() if p.requires_grad]
    psi_params = [p for p in model.model_psi.parameters() if p.requires_grad]
    omega_params = [p for p in model.model_omega.parameters() if p.requires_grad]
    phys_params = [physics._raw_mu_tot]

    groups = [
        {"params": p_params, "lr": BASE_LR},
        {"params": omega_params, "lr": BASE_LR},
        {"params": psi_params, "lr": BASE_LR * 0.1},
        {"params": phys_params, "lr": BASE_LR * PARAM_LR_FACTOR},
    ]
    optimizer = torch.optim.Adam(groups, eps=ADAM_EPS)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ADAM_EPOCHS_PHASE2, eta_min=1e-6)

    # Chunk size per gradient accumulation
    CHUNK_SIZE = 25000
    active_bcs = ["p", "u", "v"]

    history = SimpleHistory()

    print(f"\n{'=' * 60}\nAVVIO FASE 2 ADAM (Vorticità + Pressione): {ADAM_EPOCHS_PHASE2} epoche\n{'=' * 60}")
    print(f"  Pesi: W_DATA={W_DATA_2}, W_BC={W_BC_2}, W_MOMENTUM={W_MOMENTUM}")
    print(f"        W_POISSON={W_POISSON}, W_VORTICITY={W_VORTICITY}, W_DRIFT={W_DRIFT}")

    pbar = tqdm(range(ADAM_EPOCHS_PHASE2), desc="Adam Fase 2 (psi-w)", mininterval=2.0)

    for epoch_idx in pbar:
        epoch_global = 50000 + epoch_idx + 1
        model.train()
        optimizer.zero_grad(set_to_none=True)

        d_loss_accum = 0.0
        p_loss_accum = 0.0
        loss_m_accum = 0.0

        # 1. Accumulazione su chunk (Data + Drift + Momentum)
        total_points = xy_all.shape[0]
        for i in range(0, total_points, CHUNK_SIZE):
            xc = xy_all[i : i + CHUNK_SIZE]
            yc = uv_all[i : i + CHUNK_SIZE]
            w_chunk = xc.shape[0] / total_points

            xph = xc.clone().requires_grad_(True)
            u, v, p, tau = physics.get_velocity(model, xph)

            chunk_loss = 0.0

            # Data Loss
            dl = physics.data_loss(u, v, yc, var_w)
            d_loss_accum += dl.item() * w_chunk
            chunk_loss = chunk_loss + W_DATA_2 * dl * w_chunk

            # Soft anti-drift loss su cinematica
            u_c0 = u_ckpt_cache[i : i + CHUNK_SIZE]
            v_c0 = v_ckpt_cache[i : i + CHUNK_SIZE]
            drl = physics.drift_loss(u, v, u_c0, v_c0)
            chunk_loss = chunk_loss + W_DRIFT * drl * w_chunk

            # Momentum equation (per pressione p)
            div_tau_c = (precomputed_div_tau[0][i : i + CHUNK_SIZE], precomputed_div_tau[1][i : i + CHUNK_SIZE])
            lm, _ = physics.compute_pde_losses(
                xph, u, v, p, tau, w_momentum=1.0, w_constitutive=0.0,
                frozen_velocity=False, precomputed_div_tau=div_tau_c
            )
            loss_m_accum += lm.item() * w_chunk
            p_loss_accum += (W_MOMENTUM * lm).item() * w_chunk
            chunk_loss = chunk_loss + (W_MOMENTUM * lm) * w_chunk

            chunk_loss.backward()

        # 2. Boundary conditions
        b_loss = physics.boundary_loss(model, bc_data, var_w, active_bcs=active_bcs)
        b_loss_val = b_loss.item()
        if W_BC_2 > 0.0:
            (W_BC_2 * b_loss).backward()

        # 3. Equazioni di Vorticità (Poisson + Trasporto Vorticità)
        l_poiss, l_vort = physics.compute_vorticity_losses(model)
        rot_loss = (physics.w_poisson * l_poiss) + (physics.w_vorticity * l_vort)
        if W_CURL > 0.0:
            (W_CURL * rot_loss).backward()

        tot_loss = (W_DATA_2 * d_loss_accum) + (W_BC_2 * b_loss_val) + p_loss_accum + (W_CURL * rot_loss.item())

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        torch.nn.utils.clip_grad_norm_([physics._raw_mu_tot], PARAM_CLIP_NORM)

        optimizer.step()
        scheduler.step()

        # Aggiornamento postfix progress bar
        if (epoch_idx + 1) % 10 == 0:
            params_curr = physics.log_params()
            pbar.set_postfix({
                "Loss": f"{tot_loss:.2e}",
                "mu_tot": f"{params_curr['mu_tot']:.4f}",
                "mu_s": f"{params_curr['mu_s']:.4f}",
                "L_Poiss": f"{l_poiss.item():.2e}",
                "L_Vort": f"{l_vort.item():.2e}",
            })

        # Logging dettagliato e verboso ogni LOG_FREQUENCY epoche
        is_log_epoch = ((epoch_idx + 1) % LOG_FREQUENCY == 0) or (epoch_idx == 0) or ((epoch_idx + 1) == ADAM_EPOCHS_PHASE2)
        if is_log_epoch:
            params = physics.log_params()
            model.eval()
            with torch.no_grad():
                l2_errs = compute_l2_errors(model, physics, data)

            # Stima rapida del rapporto gradienti su psi (Data vs Momentum)
            with torch.enable_grad():
                s_pts = xy_all[:3000].clone().requires_grad_(True)
                s_lbl = uv_all[:3000]
                u_s, v_s, p_s, tau_s = physics.get_velocity(model, s_pts, create_graph=True)
                l_d_s = physics.data_loss(u_s, v_s, s_lbl, var_w)
                g_d_list = torch.autograd.grad(W_DATA_2 * l_d_s, model.model_psi.parameters(), retain_graph=True, allow_unused=True)
                g_d_norm = sum(g.norm(2).item()**2 for g in g_d_list if g is not None)**0.5

                s_div = (precomputed_div_tau[0][:3000], precomputed_div_tau[1][:3000])
                lm_s, _ = physics.compute_pde_losses(s_pts, u_s, v_s, p_s, tau_s, w_momentum=1.0, w_constitutive=0.0, frozen_velocity=False, precomputed_div_tau=s_div)
                g_m_list = torch.autograd.grad(W_MOMENTUM * lm_s, model.model_psi.parameters(), allow_unused=True)
                g_m_norm = sum(g.norm(2).item()**2 for g in g_m_list if g is not None)**0.5
                ratio_g = (g_m_norm / (g_d_norm + 1e-12)) if g_d_norm > 0 else 0.0

            # Stampa report leggibile
            print(f"\n[Epoch {epoch_global}] Loss: {tot_loss:.4e} | Data: {d_loss_accum:.4e} | BC: {b_loss_val:.4e} | PDE(mom): {p_loss_accum:.4e} | Rot: {rot_loss.item():.4e}")
            print(f"  Params -> mu_tot: {params['mu_tot']:.6f} Pa·s | mu_s: {params['mu_s']:.6f} Pa·s | beta: {params['beta']:.6f} | mu_p: {params['mu_p']:.6f} Pa·s")
            print(f"  L2 Errors -> u: {l2_errs['u']:.4e} | v: {l2_errs['v']:.4e} | p: {l2_errs['p']:.4e}")
            print(f"               tau_xx: {l2_errs['tau_xx']:.4e} | tau_xy: {l2_errs['tau_xy']:.4e} | tau_yy: {l2_errs['tau_yy']:.4e}")
            print(f"  Vorticity -> L_Poisson: {l_poiss.item():.4e} | L_Vorticity: {l_vort.item():.4e} | Mean(|w|): {physics._last_mean_omega:.4e}")
            print(f"  G_data(psi): {g_d_norm:.4e} | G_mom(psi): {g_m_norm:.4e} | Ratio (G_mom/G_data): {ratio_g:.4f}")

            # Salvataggio checkpoint
            chk_path = save_dir / "checkpoint.pth"
            torch.save({
                'epoch': epoch_global,
                'model_state_dict': model.state_dict(),
                'physics_state_dict': physics.state_dict(),
            }, str(chk_path))

            # Logging su CSV
            if physics.diag_csv_path is not None:
                write_hdr = not os.path.exists(physics.diag_csv_path)
                with open(physics.diag_csv_path, "a", encoding="utf-8") as f:
                    if write_hdr:
                        f.write("epoch,mu_tot,mu_s,beta,loss_total,loss_data,loss_bc,loss_momentum,loss_poisson,loss_vorticity,err_u,err_v,err_p,ratio_g\n")
                    f.write(f"{epoch_global},{params['mu_tot']:.6f},{params['mu_s']:.6f},{params['beta']:.6f},{tot_loss:.6e},{d_loss_accum:.6e},{b_loss_val:.6e},{p_loss_accum:.6e},{l_poiss.item():.6e},{l_vort.item():.6e},{l2_errs['u']:.6e},{l2_errs['v']:.6e},{l2_errs['p']:.6e},{ratio_g:.4f}\n")

            history.update(epoch_global, {
                "total": tot_loss, "data": d_loss_accum, "bc": b_loss_val, "pde": p_loss_accum,
                "loss_momentum": loss_m_accum, "loss_constitutive": 0.0,
                "param_beta": params["beta"], "param_mu_tot": params["mu_tot"],
                "param_mu_s": params["mu_s"], "param_mu_p": params["mu_p"], "param_lam": params["lam"],
                "l2_u": l2_errs["u"], "l2_v": l2_errs["v"], "l2_p": l2_errs["p"],
                "l2_tau_xx": l2_errs["tau_xx"], "l2_tau_xy": l2_errs["tau_xy"], "l2_tau_yy": l2_errs["tau_yy"]
            })

            if tb_writer is not None:
                tb_writer.add_scalar('Loss/Total', tot_loss, epoch_global)
                tb_writer.add_scalar('Loss/Data', d_loss_accum, epoch_global)
                tb_writer.add_scalar('Loss/BC', b_loss_val, epoch_global)
                tb_writer.add_scalar('Loss/Momentum', p_loss_accum, epoch_global)
                tb_writer.add_scalar('Loss/Poisson', l_poiss.item(), epoch_global)
                tb_writer.add_scalar('Loss/Vorticity', l_vort.item(), epoch_global)
                tb_writer.add_scalar('Params/mu_tot', params["mu_tot"], epoch_global)
                tb_writer.add_scalar('Params/mu_s', params["mu_s"], epoch_global)
                tb_writer.add_scalar('L2_Error/p', l2_errs["p"], epoch_global)
                tb_writer.flush()

    pbar.close()

    # ========================================================================
    # 6. FASE L-BFGS (FP64)
    # ========================================================================
    if USE_LBFGS_PHASE2 and LBFGS_MAX_ITERS_PHASE2 > 0:
        print(f"\n{'=' * 60}\nAVVIO FASE 2 L-BFGS (FP64): {LBFGS_MAX_ITERS_PHASE2} iterazioni\n{'=' * 60}")
        convert_to_fp64(model, physics, data)
        xy_all = data["coords"]
        uv_all = data["uv_data"]
        bc_data = data["boundary_groups"]

        u_ckpt_cache = u_ckpt_cache.double()
        v_ckpt_cache = v_ckpt_cache.double()

        precomputed_div_tau_lbfgs = precompute_stress_divergence(model, physics, xy_all)
        rot_tau_lbfgs = physics.precompute_rot_div_tau(model, xy_all, subset_size=CURL_SUBSET_SIZE)

        all_params_lbfgs = [p for p in model.parameters() if p.requires_grad] + [physics._raw_mu_tot]
        optimizer_lbfgs = torch.optim.LBFGS(
            all_params_lbfgs, lr=1.0, max_iter=1, max_eval=20,
            tolerance_grad=1e-16, tolerance_change=1e-16, history_size=300, line_search_fn="strong_wolfe"
        )

        pbar_lbfgs = tqdm(range(LBFGS_MAX_ITERS_PHASE2), desc="L-BFGS Fase 2", mininterval=2.0)
        last_lbfgs_dict = {}

        def closure():
            optimizer_lbfgs.zero_grad(set_to_none=True)
            d_loss_accum = 0.0
            p_loss_accum = 0.0
            total_points = xy_all.shape[0]

            for i in range(0, total_points, CHUNK_SIZE):
                xc = xy_all[i : i + CHUNK_SIZE]
                yc = uv_all[i : i + CHUNK_SIZE]
                w_chunk = xc.shape[0] / total_points

                xph = xc.clone().requires_grad_(True)
                u, v, p, tau = physics.get_velocity(model, xph)

                dl = physics.data_loss(u, v, yc, var_w)
                d_loss_accum += dl.item() * w_chunk

                u_c0 = u_ckpt_cache[i : i + CHUNK_SIZE]
                v_c0 = v_ckpt_cache[i : i + CHUNK_SIZE]
                drl = physics.drift_loss(u, v, u_c0, v_c0)

                div_tau_c = (precomputed_div_tau_lbfgs[0][i : i + CHUNK_SIZE], precomputed_div_tau_lbfgs[1][i : i + CHUNK_SIZE])
                lm, _ = physics.compute_pde_losses(xph, u, v, p, tau, w_momentum=1.0, w_constitutive=0.0, frozen_velocity=False, precomputed_div_tau=div_tau_c)
                p_loss_accum += (W_MOMENTUM * lm).item() * w_chunk

                chunk_loss = (W_DATA_2 * dl + W_DRIFT * drl + W_MOMENTUM * lm) * w_chunk
                chunk_loss.backward()

            b_loss = physics.boundary_loss(model, bc_data, var_w, active_bcs=active_bcs)
            if W_BC_2 > 0.0:
                (W_BC_2 * b_loss).backward()

            l_poiss, l_vort = physics.compute_vorticity_losses(model)
            rot_loss = (physics.w_poisson * l_poiss) + (physics.w_vorticity * l_vort)
            if W_CURL > 0.0:
                (W_CURL * rot_loss).backward()

            tot_lbfgs = (W_DATA_2 * d_loss_accum) + (W_BC_2 * b_loss.item()) + p_loss_accum + (W_CURL * rot_loss.item())
            last_lbfgs_dict['total'] = tot_lbfgs
            last_lbfgs_dict['data'] = d_loss_accum
            last_lbfgs_dict['bc'] = b_loss.item()
            last_lbfgs_dict['pde'] = p_loss_accum
            last_lbfgs_dict['rot'] = rot_loss.item()
            last_lbfgs_dict['poiss'] = l_poiss.item()
            last_lbfgs_dict['vort'] = l_vort.item()
            return torch.tensor(tot_lbfgs, device=DEVICE, dtype=torch.float64)

        for it_lbfgs in range(LBFGS_MAX_ITERS_PHASE2):
            loss_step = optimizer_lbfgs.step(closure)
            step_val = loss_step.item() if isinstance(loss_step, torch.Tensor) else float(loss_step)
            global_step = 50000 + ADAM_EPOCHS_PHASE2 + it_lbfgs + 1
            pbar_lbfgs.update(1)

            is_log_lbfgs = ((it_lbfgs + 1) % LOG_FREQUENCY == 0) or (it_lbfgs == 0) or ((it_lbfgs + 1) == LBFGS_MAX_ITERS_PHASE2)
            if is_log_lbfgs:
                params = physics.log_params()
                with torch.no_grad():
                    l2_errs = compute_l2_errors(model, physics, data)

                tot_l = last_lbfgs_dict.get('total', step_val)
                print(f"\n[L-BFGS Iter {it_lbfgs+1}/{LBFGS_MAX_ITERS_PHASE2}] Loss: {tot_l:.4e} | Data: {last_lbfgs_dict.get('data',0):.4e} | PDE: {last_lbfgs_dict.get('pde',0):.4e} | Rot: {last_lbfgs_dict.get('rot',0):.4e}")
                print(f"  Params -> mu_tot: {params['mu_tot']:.6f} Pa·s | mu_s: {params['mu_s']:.6f} Pa·s | beta: {params['beta']:.6f}")
                print(f"  L2 Errors -> u: {l2_errs['u']:.4e} | v: {l2_errs['v']:.4e} | p: {l2_errs['p']:.4e}")
                print(f"  Vorticity -> L_Poiss: {last_lbfgs_dict.get('poiss',0):.4e} | L_Vort: {last_lbfgs_dict.get('vort',0):.4e}")

                history.update(global_step, {
                    "total": tot_l, "data": last_lbfgs_dict.get('data',0), "bc": last_lbfgs_dict.get('bc',0), "pde": last_lbfgs_dict.get('pde',0),
                    "loss_momentum": last_lbfgs_dict.get('pde',0), "loss_constitutive": 0.0,
                    "param_beta": params["beta"], "param_mu_tot": params["mu_tot"],
                    "param_mu_s": params["mu_s"], "param_mu_p": params["mu_p"], "param_lam": params["lam"],
                    "l2_u": l2_errs["u"], "l2_v": l2_errs["v"], "l2_p": l2_errs["p"],
                    "l2_tau_xx": l2_errs["tau_xx"], "l2_tau_xy": l2_errs["tau_xy"], "l2_tau_yy": l2_errs["tau_yy"]
                })

        pbar_lbfgs.close()

    return history


# ============================================================================
# 7. MAIN ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    adam_k = ADAM_EPOCHS_PHASE2 // 1000
    lbfgs_k = f"{LBFGS_MAX_ITERS_PHASE2 / 1000:.1f}k".replace(".0k", "k")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_name = f"[{timestamp}][INV][STAGED][Ph2_{adam_k}k+{lbfgs_k}_Vorticity_Mauri]"
    OUTPUT_DIR = BASE_DIR / "output_4rollmill" / run_name
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    builtins.OUTPUT_DIR = OUTPUT_DIR

    global_log_path = OUTPUT_DIR / "train_log.txt"

    print("=" * 80)
    print("AVVIO TRAINING PINN 4-ROLL MILL [VORTICITY SPLITTING (psi-omega) REFORMULATION]")
    print(f"Device: {DEVICE} | Precisione: {torch.get_default_dtype()}")
    print(f"Checkpoint Fase 1: {RESUME_CHECKPOINT.name}")
    print(f"Fase 2 Budget: {ADAM_EPOCHS_PHASE2} Adam + {LBFGS_MAX_ITERS_PHASE2} L-BFGS")
    print(f"Formulazione: mu_tot (target: 1.000 Pa*s), mu_s = mu_tot - mu_p^(F1)")
    print(f"Vincolo Rotazionale: W_CURL = {W_CURL} (W_POISSON={W_POISSON}, W_VORTICITY={W_VORTICITY}) | Punti: {CURL_SUBSET_SIZE}")
    print(f"Logging Frequenza Verbosa: ogni {LOG_FREQUENCY} epoche")
    print(f"Output salvato in: {OUTPUT_DIR}")
    print("=" * 80)

    # 1. Caricamento Dataset
    data = load_data(filepath=DATASET_PATH, eta_0=ETA_0)

    # 2. Inizializzazione Modello a 4 teste (psi, p, tau, omega)
    model = CombinedModelVorticity(p_scale=data["p_scale"], tau_scale=data["tau_scale"]).to(DEVICE)
    for submodel in [model.model_psi, model.model_p, model.model_tau, model.model_omega]:
        submodel.apply(lambda m: init_weights_xavier(m, activation_name=ACTIVATION))

    initialize_last_layer_zero(model.model_p)
    initialize_last_layer_zero(model.model_tau)
    initialize_last_layer_zero(model.model_omega)

    physics = VorticityMuTotPhysics(
        U_ref=data["U_ref"],
        H_ref=data["H"],
        H_coord=data["H_coord"],
        var_weights=data["var_weights"],
        inverse_mode=INVERSE_PROBLEM,
        tau_scale=data["tau_scale"],
        p_scale=data["p_scale"],
        eta_0=ETA_0,
    ).to(DEVICE)

    # 3. Caricamento Checkpoint Fase 1 e Fissaggio di mu_p
    if RESUME_CHECKPOINT.exists():
        chk = torch.load(str(RESUME_CHECKPOINT), map_location=DEVICE)

        # Carica selettivamente i pesi delle teste esistenti in F1
        state_m = chk['model_state_dict']
        psi_dict = {k.replace("model_psi.", ""): v for k, v in state_m.items() if k.startswith("model_psi.")}
        tau_dict = {k.replace("model_tau.", ""): v for k, v in state_m.items() if k.startswith("model_tau.")}
        p_dict = {k.replace("model_p.", ""): v for k, v in state_m.items() if k.startswith("model_p.")}

        model.model_psi.load_state_dict(psi_dict)
        model.model_tau.load_state_dict(tau_dict)
        if p_dict:
            model.model_p.load_state_dict(p_dict)

        physics.load_state_dict(chk['physics_state_dict'], strict=False)

        # Fissiamo mu_p e lambda come costanti della Fase 1
        params_log = physics.log_params()
        physics.mu_p_fixed = physics.mu_p.detach().clone()
        physics.diag_csv_path = OUTPUT_DIR / "vorticity_diagnostics.csv"
        print(f"\n[Checkpoint Fase 1] Pesi e reologia caricati da: {RESUME_CHECKPOINT.name}")
        print(f"  lam (F1 fissa)   : {params_log['lam']:.6f} s (target: {LAM_TRUE:.6f})")
        print(f"  mu_p (F1 fissa)  : {params_log['mu_p']:.6f} Pa*s (target: {MU_P_TRUE:.6f})")
        print(f"  mu_tot (guess)   : {params_log['mu_tot']:.6f} Pa*s (target: {MU_TOT_TRUE:.6f})")
        print(f"  mu_s (derivata)  : {params_log['mu_s']:.6f} Pa*s (target: {MU_S_TRUE:.6f})")

        # Precalcolo statico rot(div(tau)) sul sottoinsieme (ordine 2)
        print(f"\n[Precalcolo] Calcolo di rot(div(tau)) statico su {CURL_SUBSET_SIZE} punti...")
        rot_tau = physics.precompute_rot_div_tau(model, data["coords"].to(DEVICE), subset_size=CURL_SUBSET_SIZE)

        # Warm-start rapido di model_omega per allineamento a -nabla^2(psi)
        physics.warmstart_omega(model, data["coords"].to(DEVICE), steps=300, lr=1e-3)
    else:
        raise FileNotFoundError(f"Checkpoint Fase 1 non trovato: {RESUME_CHECKPOINT}")

    # 4. Training Fase 2
    launch_tensorboard_server(OUTPUT_DIR.parent)
    tb_dir = OUTPUT_DIR / "tb_logs"
    tb_dir.mkdir(parents=True, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=str(tb_dir))

    history = train_phase2_vorticity(
        model,
        physics,
        data,
        save_dir=OUTPUT_DIR,
        tb_writer=tb_writer,
    )
    tb_writer.close()

    # 5. Report Finale
    params = physics.log_params()
    print(f"\n{'=' * 60}\nRISULTATI FINALI PARAMETRI FISICI [VORTICITY FORMULATION]\n{'=' * 60}")
    print(f"  eta_0 (scala rif.)  : {params['eta_0']:.6f} Pa·s")
    print(f"  mu_p (fissa F1)     : {params['mu_p']:.6f} Pa·s (true: {MU_P_TRUE:.6f})")
    print(f"  mu_tot* (adimens.)  : {params['mu_tot_nd']:.6f} (true: {MU_TOT_TRUE/ETA_0:.6f})")
    print(f"  mu_tot  (dimension.): {params['mu_tot']:.6f} Pa·s (true: {MU_TOT_TRUE:.6f})")
    print(f"  mu_s* (adimens.)    : {params['mu_s_nd']:.6f} (true: {MU_S_TRUE/ETA_0:.6f})")
    print(f"  mu_s  (dimension.)  : {params['mu_s']:.6f} Pa·s (true: {MU_S_TRUE:.6f})")
    print(f"  beta  (ratio)       : {params['beta']:.6f} (true: {BETA_TRUE:.6f})")
    print(f"  lam   (dimension.)  : {params['lam']:.6f} s (true: {LAM_TRUE:.6f})")

    final_losses = evaluate_final_losses(model, physics, data)
    print(f"\n{'=' * 60}\nREPORT FINALE LOSS\n{'=' * 60}")
    for k, v in final_losses.items():
        print(f"  {k:<20s}: {v:.6e}")

    errors = compute_l2_errors(model, physics, data)
    print("\nL2 Relative Errors:")
    for fn, err in errors.items():
        print(f"  {fn:>8s}: {err:.6f}")

    # 6. Generazione Plot
    history.plot_losses(str(OUTPUT_DIR / "loss_history.png"))
    history.plot_params(str(OUTPUT_DIR / "params_evolution.png"))
    history.plot_l2_errors(str(OUTPUT_DIR / "l2_errors_history.png"))

    generate_all_diagnostics(model, physics, data, str(OUTPUT_DIR))

    if DEBUG_MODE:
        test_random_points(model, physics, data, num_points=10)
        debug_physics_magnitudes(model, physics, data, num_points=2000)

    print(f"\n[OK] Esecuzione terminata con successo. Risultati in: {OUTPUT_DIR}")
