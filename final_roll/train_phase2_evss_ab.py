"""
scratch/train_phase2_evss_ab.py
=============================================================================
Script Fase 2 EVSS per PC Personale (A/B Testing vs PC Maurizio Standard)
=============================================================================

Caratteristiche e Riforme Numeriche Implementate:
  1. Setup e Budget Identici a PC Maurizio:
     - Dataset: COMSOL/4roll/4_roll_mill.csv (125,456 nodi).
     - Inizializzazione: final_roll/checkpoints/checkpoint_inverso_fase1_40k+10k.pth.
     - Budget standard: 20.000 epoche Adam + 2.000 iterazioni L-BFGS (supporta --smoke-test con 2+2).
  2. Precalcolo della Sorgente Elastica EVSS (Fase 1 Frozen):
     - Sigma_F1 = tau_F1 - mu_p_F1 * grad(u_F1)
     - div(Sigma_F1) = div(tau_F1) - mu_p_F1 * lap(u_F1)
     - Precalcolo statico una tantum su tutti i punti di collocazione.
  3. Formulazione Navier-Stokes EVSS Scalata (scale_mom = 400.0 Pa/m):
     - fu_dim = rho * conv_u + px - mu_tot * lap_u - div_sigma_x  [Pa/m]
     - fv_dim = rho * conv_v + py - mu_tot * lap_v - div_sigma_y  [Pa/m]
     - loss_mom = 0.5 * (((fu / scale_mom)**2 + (fv / scale_mom)**2).mean())
     - Garantisce valori di loss normalizzati di ordine O(10^-2 - 10^0).
  4. Parametrizzazione mu_tot & Softplus mu_s:
     - mu_tot ottimizzato in log-space (guess iniziale = 0.984854 Pa*s).
     - mu_s = softplus(mu_tot - mu_p_fixed, beta=20.0), garantendo mu_s > 0.
  5. Ancoraggio Hard Algebrico della Pressione:
     - p(x) = p_scale * (model_p(x) - model_p(x_anchor)) + p_ref.
     - Vincolo esatto a livello algebrico ad ogni step, senza penalità puntuali soft.
  6. Modello psi Mobile Controllato:
     - Micro-LR (1e-4, ossia 0.1 * BASE_LR).
     - Soft functional proximity penalty (loss_prox) rispetto alla cinematica di F1.
  7. Igiene Numerica & Precisione Differenziata:
     - TF32 disabilitato per standard IEEE FP32.
     - Adam EPS differenziato: 1e-8 per reti neurali, 1e-15 per il parametro fisico mu_tot.
     - Gradient clipping rigido GRAD_CLIP_NORM = 5.0.
     - L-BFGS @ FP64 con convert_to_fp64 e assert_fp64_integrity, history_size = 300, strong_wolfe.
=============================================================================
"""
import argparse
import builtins
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Percorsi di Progetto
SCRATCH_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRATCH_DIR.parent
FINAL_ROLL_DIR = PROJECT_ROOT / "final_roll"
sys.path.insert(0, str(FINAL_ROLL_DIR))

# Configurazione Ambiente PyTorch
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.set_default_dtype(torch.float32)

# [Proposta A] Disabilitazione TF32 per conformità IEEE FP32 standard
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False

SEED = 123
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Costanti Fisiche di Riferimento (Ground Truth COMSOL)
DATASET_PATH = PROJECT_ROOT / "COMSOL" / "4roll" / "4_roll_mill.csv"
RESUME_CHECKPOINT = FINAL_ROLL_DIR / "checkpoints" / "checkpoint_inverso_fase1_40k+10k.pth"

ETA_0 = 2.0
GUESS_FACTOR = 0.80
MU_S_TRUE = 0.100
MU_P_TRUE = 0.900
MU_TOT_TRUE = 1.000
BETA_TRUE = MU_S_TRUE / MU_TOT_TRUE  # 0.100
LAM_TRUE = 0.050
EPS_TRUE = 0.0
ALPHA_TRUE = 0.0
RHO = 1000.0

ACTIVATION = nn.SiLU
HIDDEN_LAYERS = [128] * 8

# Pesi Funzione di Loss per Fase 2
W_DATA_2 = 20.0
W_BC_2 = 5.0
W_MOMENTUM = 1.0
W_PROX = 0.1  # Soft functional proximity penalty weight

# Parametri Iper-ottimizzazione
BASE_LR = 1e-3
ADAM_EPS = 1e-8
ADAM_EPS_PHYS = 1e-15
PARAM_LR_FACTOR = 0.1
GRAD_CLIP_NORM = 5.0
SCALE_MOM = 400.0  # Pa/m (Proposta AA)

# Budget Standard Fase 2 (A/B Test vs Mauri)
ADAM_EPOCHS_PHASE2 = 20000
USE_LBFGS_PHASE2 = True
LBFGS_MAX_ITERS_PHASE2 = 2000
LOG_FREQUENCY = 100
BATCH_CHUNK_SIZE = 8192

# Supporto opzione --smoke-test
if "--smoke-test" in sys.argv:
    print("\n[ATTENZIONE] Modalita' --smoke-test attiva: 2 epoche Adam + 2 iterazioni L-BFGS.")
    ADAM_EPOCHS_PHASE2 = 2
    LBFGS_MAX_ITERS_PHASE2 = 2
    LOG_FREQUENCY = 1

# Iniezione parametri globali nei moduli src prima del loro import
for name, val in list(locals().items()):
    if name.isupper():
        builtins.__dict__[name] = val

# Import Moduli dal Core final_roll/src
import src.debug
import src.physics
import src.train
import src.utils

from src.train import CombinedModel
from src.physics import Physics, compute_l2_errors
from src.utils import load_data, convert_to_fp64, convert_to_fp32, assert_fp64_integrity
from src.debug import diagnose_identifiability


# ============================================================================
# CLASSE FISICA EVSS SPECIALIZZATA FASE 2
# ============================================================================
class EVSSPhase2Physics(Physics):
    """
    Fisica Fase 2 con formulazione EVSS (Elastic-Viscous Split Stress):
      - Parametrizza mu_tot in log-space: mu_tot = guess_mu_tot * exp(raw_mu_tot).
      - Congela mu_p da Fase 1 e protegge la viscosità solvente via softplus:
        mu_s = softplus(mu_tot - mu_p_fixed, beta=20.0) > 0.
      - Calcola il residuo dimensionale del momento in Pa/m e lo scala per scale_mom = 400.0 Pa/m.
    """
    def __init__(self, *args, mu_p_fixed=0.904854, guess_mu_tot=None, scale_mom=400.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("mu_p_fixed", torch.tensor(float(mu_p_fixed), device=DEVICE, dtype=torch.float32))
        
        # Allineamento iniziale con la stima di guess: mu_tot_guess = guess_mu_s (0.080) + mu_p_fixed
        if guess_mu_tot is None:
            guess_mu_s = MU_S_TRUE * GUESS_FACTOR  # 0.0800 Pa*s
            guess_mu_tot = guess_mu_s + float(mu_p_fixed)  # ~0.984854 Pa*s
        self.register_buffer("guess_mu_tot", torch.tensor(float(guess_mu_tot), device=DEVICE, dtype=torch.float32))
        self.register_parameter("_raw_mu_tot", nn.Parameter(torch.zeros(1, device=DEVICE, dtype=torch.float32), requires_grad=True))
        self.register_buffer("scale_mom_buf", torch.tensor(float(scale_mom), device=DEVICE, dtype=torch.float32))

    @property
    def mu_tot(self):
        """Viscosita' totale dimensionale [Pa*s]: eta_tot = guess_mu_tot * exp(r_tot)."""
        return self.guess_mu_tot * torch.exp(self._raw_mu_tot).squeeze()

    @property
    def mu_tot_nd(self):
        """Viscosita' totale adimensionale: mu_tot* = mu_tot / eta_0."""
        return self.mu_tot / self.eta_0

    @property
    def mu_p(self):
        """Viscosita' polimerica dimensionale fissata da Fase 1."""
        return self.mu_p_fixed

    @property
    def mu_p_nd(self):
        """Viscosita' polimerica adimensionale fissata: mu_p* = mu_p / eta_0."""
        return self.mu_p_fixed / self.eta_0

    @property
    def mu_s(self):
        """Viscosita' solvente dimensionale protetta via softplus (Proposta AC): mu_s = softplus(mu_tot - mu_p, beta=20.0)."""
        return torch.nn.functional.softplus(self.mu_tot - self.mu_p_fixed, beta=20.0)

    @property
    def mu_s_nd(self):
        """Viscosita' solvente adimensionale: mu_s* = mu_s / eta_0."""
        return self.mu_s / self.eta_0

    @property
    def beta(self):
        """Rapporto viscoso derivato: beta = mu_s / (mu_tot + 1e-12)."""
        return self.mu_s / (self.mu_tot + 1e-12)

    @property
    def Re_phys(self):
        """Numero di Reynolds fisico basato sulla viscosita' totale: Re_phys = rho * U_ref * H_ref / mu_tot."""
        return RHO * self.U_ref * self.H_ref / (self.mu_tot + 1e-12)

    def compute_momentum_loss(self, x, u, v, p, div_sigma_chunk, scale_mom=None):
        """
        Calcola il residuo di Navier-Stokes in formulazione EVSS scalato con scale_mom:
          fu_dim = rho * conv_u + px - mu_tot * lap_u - div_sigma_x  [Pa/m]
          fv_dim = rho * conv_v + py - mu_tot * lap_v - div_sigma_y  [Pa/m]
          loss_mom = 0.5 * (((fu_dim / scale_mom)**2 + (fv_dim / scale_mom)**2).mean())
        """
        if scale_mom is None:
            scale_mom = self.scale_mom_buf

        gu = self._grad(u, x, create_graph=True)
        ux, uy = gu[:, 0:1], gu[:, 1:2]
        gv = self._grad(v, x, create_graph=True)
        vx, vy = gv[:, 0:1], gv[:, 1:2]

        gp = self._grad(p, x, create_graph=True)
        px, py = gp[:, 0:1], gp[:, 1:2]

        uxx = self._grad(ux, x, create_graph=True)[:, 0:1]
        uyy = self._grad(uy, x, create_graph=True)[:, 1:2]
        lap_u = uxx + uyy

        vxx = self._grad(vx, x, create_graph=True)[:, 0:1]
        lap_v = vxx - uxx  # Equazione di continuita': vyy = -uxx

        div_sigma_x, div_sigma_y = div_sigma_chunk

        conv_x = u * ux + v * uy
        conv_y = u * vx + v * vy

        # Fattore di scala dimensionale per convertire i gradienti adimensionali in Pa/m:
        # scale_phys = (eta_0 * U_ref) / (H_ref ** 2)
        scale_phys = (self.eta_0 * self.U_ref) / (self.H_ref ** 2)

        # Residui dimensionali in Pa/m:
        # Re_scale * conv_x * scale_phys = rho * (u_dim * grad) u_dim
        # px * scale_phys = grad(p_dim)
        # mu_tot_nd * lap_u * scale_phys = mu_tot * lap(u_dim)
        # div_sigma_x * scale_phys = div(Sigma_dim)
        fu_dim = (self.Re_scale * conv_x + px - self.mu_tot_nd * lap_u - div_sigma_x) * scale_phys
        fv_dim = (self.Re_scale * conv_y + py - self.mu_tot_nd * lap_v - div_sigma_y) * scale_phys

        loss_mom = 0.5 * (((fu_dim / scale_mom) ** 2 + (fv_dim / scale_mom) ** 2).mean())
        return loss_mom, fu_dim, fv_dim


# ============================================================================
# PRECALCOLO DELLA SORGENTE ELASTICA EVSS
# ============================================================================
def precompute_evss_elastic_source(model, physics, coords, mu_p_nd, chunk_size=4096):
    """
    Precalcola staticamente div(Sigma_F1) = div(tau_F1) - mu_p_nd * lap(u_F1)
    sui punti di collocazione dal checkpoint consolidato di Fase 1.
    """
    model.eval()
    n_pts = coords.shape[0]
    div_sigma_x_list = []
    div_sigma_y_list = []

    print("[EVSS Precompute] Precalcolo sorgente elastica div(Sigma_F1)...")
    with torch.enable_grad():
        for i in range(0, n_pts, chunk_size):
            xc = coords[i : i + chunk_size].clone().requires_grad_(True)
            u, v, _, tau = physics.get_velocity(model, xc, create_graph=True)
            txx, txy, tyy = tau[:, 0:1], tau[:, 1:2], tau[:, 2:3]

            gu = physics._grad(u, xc, create_graph=True)
            ux, uy = gu[:, 0:1], gu[:, 1:2]
            gv = physics._grad(v, xc, create_graph=True)
            vx = gv[:, 0:1]

            uxx = physics._grad(ux, xc, create_graph=False)[:, 0:1]
            uyy = physics._grad(uy, xc, create_graph=False)[:, 1:2]
            lap_u = (uxx + uyy).detach()

            vxx = physics._grad(vx, xc, create_graph=False)[:, 0:1]
            lap_v = (vxx - uxx).detach()

            gtxx = physics._grad(txx, xc, create_graph=False)
            gtxy = physics._grad(txy, xc, create_graph=False)
            gtyy = physics._grad(tyy, xc, create_graph=False)
            div_tx = (gtxx[:, 0:1] + gtxy[:, 1:2]).detach()
            div_ty = (gtxy[:, 0:1] + gtyy[:, 1:2]).detach()

            div_sig_x = (div_tx - mu_p_nd * lap_u).detach()
            div_sig_y = (div_ty - mu_p_nd * lap_v).detach()

            div_sigma_x_list.append(div_sig_x)
            div_sigma_y_list.append(div_sig_y)

    div_sigma_x = torch.cat(div_sigma_x_list, dim=0)
    div_sigma_y = torch.cat(div_sigma_y_list, dim=0)
    print(f"  -> div(Sigma_F1) completato: x in [{div_sigma_x.min().item():.4e}, {div_sigma_x.max().item():.4e}], y in [{div_sigma_y.min().item():.4e}, {div_sigma_y.max().item():.4e}]")
    return div_sigma_x, div_sigma_y


# ============================================================================
# CICLO PRINCIPALE DI ADDESTRAMENTO FASE 2 EVSS
# ============================================================================
def run_phase2_evss():
    print("\n" + "=" * 90)
    print(f"AVVIO ADDESTRAMENTO FASE 2 EVSS (A/B Test PC Personale)")
    print(f"Budget: {ADAM_EPOCHS_PHASE2} Adam + {LBFGS_MAX_ITERS_PHASE2} L-BFGS | Device: {DEVICE}")
    print("=" * 90)

    # 1. Directory di Output e Logging
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    smoke_tag = "_smoke" if "--smoke-test" in sys.argv else ""
    run_name = f"run_{run_timestamp}_phase2_evss_{ADAM_EPOCHS_PHASE2}adam+{LBFGS_MAX_ITERS_PHASE2}lbfgs{smoke_tag}"
    output_dir = SCRATCH_DIR / "output_phase2_evss" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=str(output_dir / "tb_logs"))

    # 2. Caricamento Dati
    print(f"\n[Data] Caricamento dataset COMSOL: {DATASET_PATH}...")
    data = load_data(DATASET_PATH, use_fp64=False, eta_0=ETA_0)
    xy_all = data["coords"]
    uv_all = data["uv_data"]
    bc_data = data["boundary_groups"]
    var_w = data["var_weights"]
    n_pts = xy_all.shape[0]

    # Verifica Ancoraggio Hard Pressione
    x_anchor = data.get("x_anchor")
    p_ref_anchor = data.get("p_ref")
    assert x_anchor is not None, "Punto di ancoraggio pressione non trovato in data!"
    print(f"  [Ancoraggio Hard AB] x_anchor = {x_anchor.cpu().numpy().tolist()}, p_ref = {p_ref_anchor.item():.4f}")

    # 3. Istanzia Modello con Hard Anchor e Carica Checkpoint Fase 1
    print(f"\n[Model] Istanzia CombinedModel e carica checkpoint: {RESUME_CHECKPOINT}...")
    model = CombinedModel(
        p_scale=data["p_scale"],
        tau_scale=data["tau_scale"],
        x_anchor=x_anchor,
        p_ref=p_ref_anchor,
    ).to(DEVICE)

    if not RESUME_CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint Fase 1 non trovato in: {RESUME_CHECKPOINT}")

    chk = torch.load(str(RESUME_CHECKPOINT), map_location=DEVICE)
    model.load_state_dict(chk["model_state_dict"], strict=False)
    print(f"  -> Checkpoint Fase 1 caricato con successo (Epoca registrata: {chk.get('epoch', 'N/A')}).")

    # Estrai parametri reologici Fase 1
    f1_phys = chk["physics_state_dict"]
    if "_raw_mu_p" in f1_phys and "guess_mu_p" in f1_phys:
        mu_p_f1 = (f1_phys["guess_mu_p"] * torch.exp(f1_phys["_raw_mu_p"])).item()
    elif "_raw_mu_p" in f1_phys:
        mu_p_f1 = (0.72 * torch.exp(f1_phys["_raw_mu_p"])).item()
    else:
        mu_p_f1 = 0.904854
    print(f"  -> Parametro reologico F1 congelato: mu_p = {mu_p_f1:.6f} Pa*s")

    # 4. Istanzia Fisica EVSS Specializzata
    physics = EVSSPhase2Physics(
        U_ref=data["U_ref"],
        H_ref=data["H"],
        H_coord=data["H_coord"],
        var_weights=data["var_weights"],
        inverse_mode=True,
        tau_scale=data["tau_scale"],
        p_scale=data["p_scale"],
        eta_0=ETA_0,
        mu_p_fixed=mu_p_f1,
        scale_mom=SCALE_MOM,
    ).to(DEVICE)

    # Diagnostica Preventiva di Identificabilita' di Leray (Proposta AE)
    try:
        diagnose_identifiability(model, physics, xy_all)
    except Exception as e:
        print(f"  [Diagnostica AE] Avviso calcolo rho_id: {e}")

    # 5. Cache Cinematica di Fase 1 (per Soft Anti-Drift Proximity Penalty)
    print("\n[Kinematics Cache] Precalcolo campi cinematici di riferimento F1...")
    with torch.enable_grad():
        u_ck_list, v_ck_list = [], []
        for i in range(0, n_pts, BATCH_CHUNK_SIZE):
            xc_c = xy_all[i : i + BATCH_CHUNK_SIZE].clone().requires_grad_(True)
            uc, vc, _, _ = physics.get_velocity(model, xc_c, create_graph=False)
            u_ck_list.append(uc.detach())
            v_ck_list.append(vc.detach())
        u_ckpt_cache = torch.cat(u_ck_list, dim=0)
        v_ckpt_cache = torch.cat(v_ck_list, dim=0)
    print("  -> Cache cinematica per soft proximity penalty completata.")

    # 6. Precalcolo Sorgente Elastica EVSS
    div_sigma_x, div_sigma_y = precompute_evss_elastic_source(
        model, physics, xy_all, physics.mu_p_nd, chunk_size=BATCH_CHUNK_SIZE
    )

    # 7. Configurazione Parametri Addestrabili (Fase 2 Decoupled)
    # model_tau: 100% CONGELATO
    for p in model.parameters():
        p.requires_grad = False
    # model_p e model_psi: MOBILI
    for p in model.model_p.parameters():
        p.requires_grad = True
    for p in model.model_psi.parameters():
        p.requires_grad = True

    # Parametro fisico trainable: mu_tot
    param_var = physics._raw_mu_tot
    param_var.requires_grad = True

    # [Proposta B] Differentiated Adam EPS: 1e-8 per la rete neurale, 1e-15 per il parametro fisico
    optimizer = torch.optim.Adam([
        {"params": model.model_p.parameters(), "lr": BASE_LR, "eps": ADAM_EPS},
        {"params": model.model_psi.parameters(), "lr": 1e-4, "eps": ADAM_EPS},  # Micro-LR controllato
        {"params": [param_var], "lr": BASE_LR * PARAM_LR_FACTOR, "eps": ADAM_EPS_PHYS},
    ], eps=ADAM_EPS)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, ADAM_EPOCHS_PHASE2), eta_min=1e-6
    )

    # 8. Valutazione Stato Iniziale
    init_errs = compute_l2_errors(model, physics, data, chunk_size=BATCH_CHUNK_SIZE)
    print(f"\n[Stato Iniziale Fase 2 EVSS]")
    print(f"  mu_tot: {physics.mu_tot.item():.4f} Pa*s (True: {MU_TOT_TRUE:.4f})")
    print(f"  mu_s:   {physics.mu_s.item():.4f} Pa*s (True: {MU_S_TRUE:.4f})")
    print(f"  beta:   {physics.beta.item():.4f} (True: {BETA_TRUE:.4f})")
    print(f"  L2 Errors -> u: {init_errs['u']:.4e} | v: {init_errs['v']:.4e} | p: {init_errs['p']:.4e} | tau_xy: {init_errs['tau_xy']:.4e}")

    history = {
        "epoch": [],
        "loss_tot": [],
        "loss_data": [],
        "loss_bc": [],
        "loss_mom": [],
        "loss_prox": [],
        "mu_tot": [],
        "mu_s": [],
        "beta": [],
        "l2_u": [],
        "l2_v": [],
        "l2_p": [],
    }

    # ========================================================================
    # STAGE 1: OTTIMIZZAZIONE ADAM
    # ========================================================================
    print("\n" + "-" * 115)
    print(f"{'Epoca':<7} | {'Loss Tot':<10} | {'L_Data':<10} | {'L_BC':<10} | {'L_Mom':<10} | {'L_Prox':<10} | {'mu_s':<8} | {'mu_tot':<8} | {'E_u':<9} | {'E_p':<9}")
    print("-" * 115)

    t0_adam = time.time()

    for epoch in range(1, ADAM_EPOCHS_PHASE2 + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        d_loss_accum = 0.0
        m_loss_accum = 0.0
        prox_loss_accum = 0.0

        # Chunking dei punti di collocazione per gradient accumulation sicura
        for i in range(0, n_pts, BATCH_CHUNK_SIZE):
            xc = xy_all[i : i + BATCH_CHUNK_SIZE]
            yc = uv_all[i : i + BATCH_CHUNK_SIZE]
            uck = u_ckpt_cache[i : i + BATCH_CHUNK_SIZE]
            vck = v_ckpt_cache[i : i + BATCH_CHUNK_SIZE]
            sig_x = div_sigma_x[i : i + BATCH_CHUNK_SIZE]
            sig_y = div_sigma_y[i : i + BATCH_CHUNK_SIZE]
            w_chunk = xc.shape[0] / n_pts

            xph = xc.clone().requires_grad_(True)
            u, v, p, _ = physics.get_velocity(model, xph, create_graph=True)

            # 1. Data Loss su velocita' u, v
            dl = physics.data_loss(u, v, yc, var_w)
            d_loss_accum += dl.item() * w_chunk

            # 2. Soft Functional Proximity Penalty rispetto alla cinematica di F1
            pl = physics.drift_loss(u, v, uck, vck)
            prox_loss_accum += pl.item() * w_chunk

            # 3. Residuo Navier-Stokes EVSS Scalato (scale_mom = 400.0 Pa/m)
            ml, _, _ = physics.compute_momentum_loss(xph, u, v, p, (sig_x, sig_y), scale_mom=SCALE_MOM)
            m_loss_accum += ml.item() * w_chunk

            chunk_loss = (W_DATA_2 * dl + W_PROX * pl + W_MOMENTUM * ml) * w_chunk
            chunk_loss.backward()

        # 4. Boundary Loss (Solo u, v poiché la pressione è ancorata algebricamente)
        bl = physics.boundary_loss(model, bc_data, var_w, active_bcs=["u", "v"])
        b_loss_val = bl.item()
        if bl.requires_grad:
            (W_BC_2 * bl).backward()

        # [Proposta H & Run 23] Gradient Clipping Rigido a 5.0
        trainable_params = [p for p in model.parameters() if p.requires_grad] + [param_var]
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=GRAD_CLIP_NORM)

        optimizer.step()
        scheduler.step()

        tot_loss = W_DATA_2 * d_loss_accum + W_BC_2 * b_loss_val + W_MOMENTUM * m_loss_accum + W_PROX * prox_loss_accum
        cur_mu_tot = physics.mu_tot.item()
        cur_mu_s = physics.mu_s.item()
        cur_beta = physics.beta.item()

        # TensorBoard Logging Scalare
        tb_writer.add_scalar("Loss/Total", tot_loss, epoch)
        tb_writer.add_scalar("Loss/Data", d_loss_accum, epoch)
        tb_writer.add_scalar("Loss/BC", b_loss_val, epoch)
        tb_writer.add_scalar("Loss/Momentum", m_loss_accum, epoch)
        tb_writer.add_scalar("Loss/Proximity", prox_loss_accum, epoch)
        tb_writer.add_scalar("Params/mu_tot", cur_mu_tot, epoch)
        tb_writer.add_scalar("Params/mu_s", cur_mu_s, epoch)
        tb_writer.add_scalar("Params/beta", cur_beta, epoch)

        # Logging periodico a console
        if epoch % LOG_FREQUENCY == 0 or epoch == ADAM_EPOCHS_PHASE2 or epoch == 1:
            errs = compute_l2_errors(model, physics, data, chunk_size=BATCH_CHUNK_SIZE)
            history["epoch"].append(epoch)
            history["loss_tot"].append(tot_loss)
            history["loss_data"].append(d_loss_accum)
            history["loss_bc"].append(b_loss_val)
            history["loss_mom"].append(m_loss_accum)
            history["loss_prox"].append(prox_loss_accum)
            history["mu_tot"].append(cur_mu_tot)
            history["mu_s"].append(cur_mu_s)
            history["beta"].append(cur_beta)
            history["l2_u"].append(errs["u"])
            history["l2_v"].append(errs["v"])
            history["l2_p"].append(errs["p"])

            tb_writer.add_scalar("Errors/L2_u", errs["u"], epoch)
            tb_writer.add_scalar("Errors/L2_v", errs["v"], epoch)
            tb_writer.add_scalar("Errors/L2_p", errs["p"], epoch)

            print(f"{epoch:<7} | {tot_loss:<10.4e} | {d_loss_accum:<10.4e} | {b_loss_val:<10.4e} | {m_loss_accum:<10.4e} | {prox_loss_accum:<10.4e} | {cur_mu_s:<8.4f} | {cur_mu_tot:<8.4f} | {errs['u']:<9.4e} | {errs['p']:<9.4e}")

    t_adam_elapsed = time.time() - t0_adam
    print(f"\n[Adam] Completato in {t_adam_elapsed:.2f}s.")

    # ========================================================================
    # STAGE 2: RAFFINAMENTO FISICO L-BFGS (FP64 SCIENTIFIC PRECISION)
    # ========================================================================
    if USE_LBFGS_PHASE2 and LBFGS_MAX_ITERS_PHASE2 > 0:
        print("\n" + "=" * 90)
        print(f"AVVIO RAFFINAMENTO L-BFGS @ FP64 (Iterazioni max: {LBFGS_MAX_ITERS_PHASE2})")
        print("=" * 90)

        # [Proposta M] Conversione ad alta precisione FP64 con asserzione guard
        print("[FP64 Guard] Conversione centralizzata di model, physics e tensori a float64...")
        convert_to_fp64(model, physics, data)
        div_sigma_x = div_sigma_x.double()
        div_sigma_y = div_sigma_y.double()
        u_ckpt_cache = u_ckpt_cache.double()
        v_ckpt_cache = v_ckpt_cache.double()
        assert_fp64_integrity(model, physics, data)
        print("  -> Asserzione FP64 Integrity superata con successo!")

        xy_all = data["coords"]
        uv_all = data["uv_data"]
        bc_data = data["boundary_groups"]
        var_w = data["var_weights"]

        # Trainable parameters FP64
        lbfgs_params = (
            [p for p in model.model_p.parameters() if p.requires_grad]
            + [p for p in model.model_psi.parameters() if p.requires_grad]
            + [physics._raw_mu_tot]
        )

        # [Run 23 Tuning] L-BFGS con history_size = 300 e strong_wolfe line search
        optimizer_lbfgs = torch.optim.LBFGS(
            lbfgs_params,
            lr=1.0,
            max_iter=1,  # 1 iterazione per chiamata closure, controllata esternamente
            max_eval=20,
            tolerance_grad=1e-16,
            tolerance_change=1e-16,
            history_size=300,
            line_search_fn="strong_wolfe",
        )

        pbar_lbfgs = tqdm(total=LBFGS_MAX_ITERS_PHASE2, desc="L-BFGS Phase 2 EVSS", mininterval=1.0)
        lbfgs_iter = 0

        def closure():
            nonlocal lbfgs_iter
            optimizer_lbfgs.zero_grad(set_to_none=True)

            d_accum = 0.0
            m_accum = 0.0
            p_accum = 0.0

            # Chunking FP64
            for i in range(0, n_pts, BATCH_CHUNK_SIZE):
                xc = xy_all[i : i + BATCH_CHUNK_SIZE]
                yc = uv_all[i : i + BATCH_CHUNK_SIZE]
                uck = u_ckpt_cache[i : i + BATCH_CHUNK_SIZE]
                vck = v_ckpt_cache[i : i + BATCH_CHUNK_SIZE]
                sig_x = div_sigma_x[i : i + BATCH_CHUNK_SIZE]
                sig_y = div_sigma_y[i : i + BATCH_CHUNK_SIZE]
                w_chunk = xc.shape[0] / n_pts

                xph = xc.clone().requires_grad_(True)
                u, v, p, _ = physics.get_velocity(model, xph, create_graph=True)

                dl = physics.data_loss(u, v, yc, var_w)
                d_accum += dl.item() * w_chunk

                pl = physics.drift_loss(u, v, uck, vck)
                p_accum += pl.item() * w_chunk

                ml, _, _ = physics.compute_momentum_loss(xph, u, v, p, (sig_x, sig_y), scale_mom=SCALE_MOM)
                m_accum += ml.item() * w_chunk

                chunk_loss = (W_DATA_2 * dl + W_PROX * pl + W_MOMENTUM * ml) * w_chunk
                chunk_loss.backward()

            bl = physics.boundary_loss(model, bc_data, var_w, active_bcs=["u", "v"])
            b_val = bl.item()
            if bl.requires_grad:
                (W_BC_2 * bl).backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(lbfgs_params, max_norm=GRAD_CLIP_NORM)

            tot_loss_val = W_DATA_2 * d_accum + W_BC_2 * b_val + W_MOMENTUM * m_accum + W_PROX * p_accum
            return torch.tensor(tot_loss_val, device=DEVICE, dtype=torch.float64)

        for it in range(1, LBFGS_MAX_ITERS_PHASE2 + 1):
            loss_val = optimizer_lbfgs.step(closure)
            pbar_lbfgs.update(1)
            global_step = ADAM_EPOCHS_PHASE2 + it

            cur_mu_tot = physics.mu_tot.item()
            cur_mu_s = physics.mu_s.item()
            cur_beta = physics.beta.item()

            tb_writer.add_scalar("Loss/Total", loss_val.item(), global_step)
            tb_writer.add_scalar("Params/mu_tot", cur_mu_tot, global_step)
            tb_writer.add_scalar("Params/mu_s", cur_mu_s, global_step)
            tb_writer.add_scalar("Params/beta", cur_beta, global_step)

        pbar_lbfgs.close()
        print("[L-BFGS] Raffinamento completato.")

    # 9. Valutazione Finale
    print("\n" + "=" * 90)
    print("REPORT FINALE FASE 2 EVSS")
    print("=" * 90)

    final_errs = compute_l2_errors(model, physics, data, chunk_size=BATCH_CHUNK_SIZE)
    final_mu_tot = physics.mu_tot.item()
    final_mu_s = physics.mu_s.item()
    final_beta = physics.beta.item()

    err_mu_s = abs(final_mu_s - MU_S_TRUE) / MU_S_TRUE * 100.0
    err_mu_tot = abs(final_mu_tot - MU_TOT_TRUE) / MU_TOT_TRUE * 100.0
    err_beta = abs(final_beta - BETA_TRUE) / BETA_TRUE * 100.0

    print(f"Parametri Fisici Stimati:")
    print(f"  mu_tot: {final_mu_tot:.6f} Pa*s (True: {MU_TOT_TRUE:.4f} | Errore Relativo: {err_mu_tot:.2f}%)")
    print(f"  mu_s:   {final_mu_s:.6f} Pa*s (True: {MU_S_TRUE:.4f} | Errore Relativo: {err_mu_s:.2f}%)")
    print(f"  beta:   {final_beta:.6f} (True: {BETA_TRUE:.4f} | Errore Relativo: {err_beta:.2f}%)")
    print(f"Errori L2 Relativi sui Campi:")
    print(f"  u:      {final_errs['u']:.4e}")
    print(f"  v:      {final_errs['v']:.4e}")
    print(f"  p:      {final_errs['p']:.4e}")
    print(f"  tau_xx: {final_errs['tau_xx']:.4e}")
    print(f"  tau_xy: {final_errs['tau_xy']:.4e}")
    print(f"  tau_yy: {final_errs['tau_yy']:.4e}")

    # Verifica Protezione Softplus
    assert final_mu_s > 0.0, f"VIOLAZIONE: mu_s non e' strettamente positivo! mu_s={final_mu_s}"

    # 10. Salvataggio Checkpoint Finale e Metriche
    final_ckpt_path = output_dir / "checkpoint_evss_phase2_final.pth"
    torch.save({
        "epoch": ADAM_EPOCHS_PHASE2 + (LBFGS_MAX_ITERS_PHASE2 if USE_LBFGS_PHASE2 else 0),
        "model_state_dict": model.state_dict(),
        "physics_state_dict": physics.state_dict(),
        "final_params": {
            "mu_tot": final_mu_tot,
            "mu_s": final_mu_s,
            "beta": final_beta,
        },
        "final_errs": final_errs,
        "history": history,
    }, str(final_ckpt_path))
    print(f"\n[Checkpoint] Salvato con successo in: {final_ckpt_path}")

    # Plot Riassuntivo delle Metriche
    try:
        plot_path = output_dir / "summary_evss_metrics.png"
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        if history["epoch"]:
            axs[0, 0].plot(history["epoch"], history["loss_mom"], label="L_Mom (EVSS)", color="purple")
            axs[0, 0].set_yscale("log")
            axs[0, 0].set_title("Loss Momentum Scalata")
            axs[0, 0].grid(True)
            axs[0, 0].legend()

            axs[0, 1].plot(history["epoch"], history["mu_s"], label="mu_s (learned)", color="blue")
            axs[0, 1].axhline(MU_S_TRUE, color="black", linestyle="--", label="mu_s (true)")
            axs[0, 1].set_title("Evoluzione Viscosita' Solvente mu_s")
            axs[0, 1].grid(True)
            axs[0, 1].legend()

            axs[1, 0].plot(history["epoch"], history["l2_p"], label="L2 Error p", color="green")
            axs[1, 0].set_yscale("log")
            axs[1, 0].set_title("Errore L2 Pressione")
            axs[1, 0].grid(True)
            axs[1, 0].legend()

            axs[1, 1].plot(history["epoch"], history["beta"], label="beta (learned)", color="red")
            axs[1, 1].axhline(BETA_TRUE, color="black", linestyle="--", label="beta (true)")
            axs[1, 1].set_title("Evoluzione Rapporto Viscoso beta")
            axs[1, 1].grid(True)
            axs[1, 1].legend()

            plt.tight_layout()
            plt.savefig(str(plot_path), dpi=150)
            plt.close()
            print(f"[Plot] Grafico riassuntivo salvato in: {plot_path}")
    except Exception as e:
        print(f"[Plot] Avviso generazione plot: {e}")

    tb_writer.close()
    print("\n[Completato] Esecuzione Fase 2 EVSS conclusa con successo!")


if __name__ == "__main__":
    run_phase2_evss()
