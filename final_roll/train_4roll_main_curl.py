"""
train_4roll_main_mutot.py (e train_4roll_main_curl.py)
=====================================================
Script PINN 4-Roll Mill con Riformulazione della Fase 2 su Viscosita' Totale (mu_tot)
e vincolo di irrotazionalita' curl(F) = 0.

Riformulazione:
  - mu_p e lambda sono FISSI dai valori identificati nella Fase 1:
      mu_p^(F1) = 0.904854 Pa*s
      lambda^(F1) = 0.050203 s
  - La Fase 2 ottimizza la viscosita' totale mu_tot (guess: 1.0, target: 1.000 Pa*s)
  - La viscosita' solvente e' ricavata analiticamente:
      mu_s = mu_tot - mu_p^(F1)  (target: 0.100 Pa*s)
  - Il vincolo curl(F) = 0 e' riscritto sulla componente non-Newtoniana:
      curl(F) = mu_tot* * curl(lap(u)) + curl(div(tau - 2*mu_p*D) - Re*(u.nabla)u) = 0

Checkpoint di partenza: checkpoints/checkpoint_inverso_fase1_40k+10k.pth
NON modifica alcun file in src/.
"""
import os
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
from src.train import CombinedModel, train
from src.utils import load_data, launch_tensorboard_server, generate_all_diagnostics

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

# --- Vincolo curl(F) = 0 su mu_tot ---
USE_CURL_CONSTRAINT = True
W_CURL = 1.0             # Peso della loss curl(F) = 0
CURL_SUBSET_SIZE = 5000   # Punti su cui valutare il curl (sottocampione per efficienza VRAM)

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR.parent / "COMSOL" / "4roll" / "4_roll_mill.csv"

# Checkpoint consolidato Fase 1
RESUME_CHECKPOINT = BASE_DIR / "checkpoints" / "checkpoint_inverso_fase1_40k+10k.pth"

# Parametri Fisici REALI
MU_S_TRUE = 0.100
MU_P_TRUE = 0.900
MU_TOT_TRUE = MU_S_TRUE + MU_P_TRUE # 1.000 Pa*s
BETA_TRUE = MU_S_TRUE / MU_TOT_TRUE  # 0.100
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
GUESS_MU_TOT = 1.000  # Guess iniziale per la viscosita' totale (ordine O(1))
GUESS_BETA = GUESS_MU_S / (GUESS_MU_S + GUESS_MU_P)
GUESS_EPS = 0.0
GUESS_ALPHA = 0.0

HIDDEN_LAYERS = [128] * 8
ACTIVATION = nn.SiLU

# Iperparametri Fase 1 (gia' completata nel checkpoint)
ADAM_EPOCHS_PHASE1 = 40000
USE_LBFGS_PHASE1 = True
LBFGS_MAX_ITERS_PHASE1 = 10000

# Iperparametri Fase 2: 30k Adam + 2k L-BFGS
ADAM_EPOCHS_PHASE2 = 30000
USE_LBFGS_PHASE2 = True
LBFGS_MAX_ITERS_PHASE2 = 2000

BASE_LR = 1e-3
ADAM_EPS = 1e-7
PARAM_LR_FACTOR = 0.1
GRAD_CLIP_NORM = 1000.0
PARAM_CLIP_NORM = 1.0

WARMUP_UNLOCK_EPOCH = 0
WARMUP_PHASE2_EPOCHS = 0

# Pesi Funzione di Loss
W_DATA_1 = 1.0
W_BC_1 = 5.0
W_CONSTITUTIVE = 1.0

W_DATA_2 = 20.0
W_BC_2 = 5.0
W_MOMENTUM = 1.0
W_DRIFT = 0.0
VARIANCE_EPS = 1e-4

# Iniezione parametri globali in builtins e nei moduli src
for name, val in list(locals().items()):
    if name.isupper():
        builtins.__dict__[name] = val
        for mod in [src.debug, src.physics, src.train, src.utils]:
            mod.__dict__[name] = val


# ============================================================================
# 3. SOTTOCLASSE SPECIALIZZATA: MUTOT & CURL PHYSICS
# ============================================================================
class MuTotCurlPhysics(Physics):
    """
    Estende Physics introducendo:
    1. Parametrizzazione su Viscosita' Totale mu_tot (target: 1.000 Pa*s)
       con mu_p fissato al valore identificato dalla Fase 1:
         mu_s = clamp(mu_tot - mu_p_fixed, min=1e-6)
    2. Vincolo di irrotazionalita' curl(F) = 0 in Fase 2:
         curl(F) = mu_tot* * curl(lap(u)) + curl(div(tau - 2*mu_p*D) - Re*(u.nabla)u) = 0
       dove il secondo termine b'(x,y) e' precalcolato all'inizio della Fase 2.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Parametro addestrabile per la viscosita' totale: mu_tot = guess_mu_tot * exp(_raw_mu_tot)
        self.register_buffer("guess_mu_tot", torch.tensor(1.0, device=DEVICE, dtype=torch.float32))
        self.register_parameter("_raw_mu_tot", nn.Parameter(torch.zeros(1, device=DEVICE, dtype=torch.float32), requires_grad=False))

        # Valore fisso di mu_p memorizzato dalla Fase 1
        self.mu_p_fixed = None

        # Buffers per il vincolo curl
        self._precomputed_b_prime = None
        self._curl_points_idx = None
        self._xy_all = None

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
        """Viscosita' polimerica: fissa da Fase 1 se impostata."""
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
        """In Fase 2 train.py chiama set_trainable('mu_s', True): noi sblocchiamo _raw_mu_tot!"""
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

    def precompute_curl_b_prime(self, model, xy_all, subset_size=5000):
        """
        Precalcola il termine b'(x,y) = curl( div(tau - 2*mu_p*D) - Re*(u.nabla)u )
        su un sottocampione random di punti.
        """
        model.eval()
        n = xy_all.shape[0]
        torch.manual_seed(42)
        idx = torch.randperm(n)[:subset_size].to(xy_all.device)
        self._curl_points_idx = idx
        self._xy_all = xy_all

        xc = xy_all[idx].clone().requires_grad_(True)
        Re_scale = self.Re_scale
        mu_p_nd_val = self.mu_p_nd.item()

        with torch.set_grad_enabled(True):
            u, v, p, tau = self.get_velocity(model, xc, create_graph=True)
            tau_xx, tau_xy, tau_yy = tau[:, 0:1], tau[:, 1:2], tau[:, 2:3]

            grad_u = self._grad(u, xc, create_graph=True)
            u_x, u_y = grad_u[:, 0:1], grad_u[:, 1:2]
            grad_v = self._grad(v, xc, create_graph=True)
            v_x, v_y = grad_v[:, 0:1], grad_v[:, 1:2]

            # Laplaciano di velocita'
            u_xx = self._grad(u_x, xc, create_graph=True)[:, 0:1]
            u_yy = self._grad(u_y, xc, create_graph=True)[:, 1:2]
            v_xx = self._grad(v_x, xc, create_graph=True)[:, 0:1]
            v_yy = self._grad(v_y, xc, create_graph=True)[:, 1:2]

            lap_u_x = u_xx + u_yy
            lap_u_y = v_xx + v_yy

            # Divergenza dello stress polimerico tau
            g_txx = self._grad(tau_xx, xc, create_graph=True)
            g_txy = self._grad(tau_xy, xc, create_graph=True)
            g_tyy = self._grad(tau_yy, xc, create_graph=True)
            div_tau_x = g_txx[:, 0:1] + g_txy[:, 1:2]
            div_tau_y = g_txy[:, 0:1] + g_tyy[:, 1:2]

            # Divergenza dello stress non-Newtoniano: div(tau_E) = div(tau) - mu_p* * lap(u)
            div_tau_E_x = div_tau_x - mu_p_nd_val * lap_u_x
            div_tau_E_y = div_tau_y - mu_p_nd_val * lap_u_y

            # Termine convettivo
            conv_x = u * u_x + v * u_y
            conv_y = u * v_x + v * v_y

            # B' = div(tau_E) - Re * conv
            B_prime_x = div_tau_E_x - Re_scale * conv_x
            B_prime_y = div_tau_E_y - Re_scale * conv_y

            # curl(B') = dB'_y/dx - dB'_x/dy
            grad_Bpy = self._grad(B_prime_y, xc, create_graph=False)
            grad_Bpx = self._grad(B_prime_x, xc, create_graph=False)
            b_prime = (grad_Bpy[:, 0:1] - grad_Bpx[:, 1:2]).detach()

            # Calcolo diagnostico di a = curl(lap(u))
            grad_lap_y = self._grad(lap_u_y, xc, create_graph=False)
            grad_lap_x = self._grad(lap_u_x, xc, create_graph=False)
            a = (grad_lap_y[:, 0:1] - grad_lap_x[:, 1:2]).detach()

        self._precomputed_b_prime = b_prime
        model.train()

        # Diagnostica analitica della proiezione su mu_tot
        mu_tot_nd_opt = - torch.dot(a.squeeze(), b_prime.squeeze()) / (torch.norm(a)**2 + 1e-16)
        mu_tot_opt = mu_tot_nd_opt.item() * self.eta_0.item()
        mu_s_opt = mu_tot_opt - self.mu_p.item()

        print(f"  [Precalcolo mu_tot] b'(x,y) precalcolato su {subset_size} punti.")
        print(f"  [Precalcolo mu_tot] |a| medio: {a.abs().mean().item():.4e} | |b'| medio: {b_prime.abs().mean().item():.4e}")
        print(f"  [Precalcolo mu_tot] mu_tot* ottimale (proiezione): {mu_tot_opt:.4f} Pa*s (Target: {MU_TOT_TRUE:.4f})")
        print(f"  [Precalcolo mu_tot] mu_s ricavata (mu_tot - mu_p): {mu_s_opt:.4f} Pa*s (Target: {MU_S_TRUE:.4f})")
        return a, b_prime

    def compute_curl_loss(self, model):
        """Calcola la loss curl(F) = 0 su punti precalcolati compatibile sia con FP32 che FP64."""
        if self._precomputed_b_prime is None or self._xy_all is None:
            return torch.tensor(0.0, device=DEVICE)

        dtype = next(model.parameters()).dtype
        device = next(model.parameters()).device

        mu_tot_nd = self.mu_tot_nd.to(dtype=dtype, device=device)

        xc = self._xy_all[self._curl_points_idx].to(dtype=dtype, device=device).clone().requires_grad_(True)
        b_prime = self._precomputed_b_prime.to(dtype=dtype, device=device)

        u, v, p, tau = self.get_velocity(model, xc, create_graph=True)
        grad_u = self._grad(u, xc, create_graph=True)
        u_x, u_y = grad_u[:, 0:1], grad_u[:, 1:2]
        grad_v = self._grad(v, xc, create_graph=True)
        v_x, v_y = grad_v[:, 0:1], grad_v[:, 1:2]

        u_xx = self._grad(u_x, xc, create_graph=True)[:, 0:1]
        u_yy = self._grad(u_y, xc, create_graph=True)[:, 1:2]
        v_xx = self._grad(v_x, xc, create_graph=True)[:, 0:1]
        v_yy = self._grad(v_y, xc, create_graph=True)[:, 1:2]

        Ax = u_xx + u_yy
        Ay = v_xx + v_yy

        grad_Ay = self._grad(Ay, xc, create_graph=True)
        grad_Ax = self._grad(Ax, xc, create_graph=True)
        a = grad_Ay[:, 0:1] - grad_Ax[:, 1:2]

        curl_F = mu_tot_nd * a + b_prime
        return torch.mean(curl_F ** 2)

    def boundary_loss(self, model, bc_data, var_w=None, active_bcs=None, **kwargs):
        """Aggiunge il vincolo curl(F)=0 formulato su mu_tot alla loss al contorno in Fase 2."""
        total_loss = super().boundary_loss(model, bc_data, var_w, active_bcs=active_bcs, **kwargs)
        if active_bcs is not None and "p" in active_bcs:
            use_curl = getattr(builtins, "USE_CURL_CONSTRAINT", True)
            if use_curl and self._precomputed_b_prime is not None:
                w_curl = getattr(builtins, "W_CURL", 1.0)
                l_curl = self.compute_curl_loss(model)
                total_loss = total_loss + w_curl * l_curl
        return total_loss


# ============================================================================
# 4. MAIN ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    layers_str = f"{len(HIDDEN_LAYERS)}x{HIDDEN_LAYERS[0]}"
    mode_tag = "INV" if INVERSE_PROBLEM else "DIR"
    strategy_tag = "STAGED" if STAGED_TRAINING else "MONO"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    adam_k = ADAM_EPOCHS_PHASE2 // 1000
    lbfgs_k = f"{LBFGS_MAX_ITERS_PHASE2 / 1000:.1f}k".replace(".0k", "k")
    run_name = f"[{timestamp}][INV][STAGED][Ph2_{adam_k}k+{lbfgs_k}_MuTot_CurlF]"
    OUTPUT_DIR = BASE_DIR / "output_4rollmill" / run_name
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    builtins.OUTPUT_DIR = OUTPUT_DIR

    global_log_path = OUTPUT_DIR / "train_log.txt"

    print("=" * 75)
    print("AVVIO TRAINING PINN 4-ROLL MILL [MUTOT & CURL(F) = 0 REFORMULATION]")
    print(f"Device: {DEVICE} | Precisione: {torch.get_default_dtype()}")
    print(f"Checkpoint Fase 1: {RESUME_CHECKPOINT.name}")
    print(f"Fase 2 Budget: {ADAM_EPOCHS_PHASE2} Adam + {LBFGS_MAX_ITERS_PHASE2} L-BFGS")
    print(f"Formulazione: mu_tot (target: 1.000 Pa*s), mu_s = mu_tot - mu_p^(F1)")
    print(f"Vincolo Rotazionale: W_CURL = {W_CURL} | Sottocampione = {CURL_SUBSET_SIZE} punti")
    print(f"Output salvato in: {OUTPUT_DIR}")
    print("=" * 75)

    # 1. Caricamento Dataset
    data = load_data(filepath=DATASET_PATH, eta_0=ETA_0)

    # 2. Inizializzazione Modello e Fisica (usando MuTotCurlPhysics)
    model = CombinedModel(p_scale=data["p_scale"], tau_scale=data["tau_scale"]).to(DEVICE)
    physics = MuTotCurlPhysics(
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
        model.load_state_dict(chk['model_state_dict'])
        physics.load_state_dict(chk['physics_state_dict'], strict=False)

        # Fissiamo mu_p e lambda come costanti della Fase 1
        params_log = physics.log_params()
        physics.mu_p_fixed = physics.mu_p.detach().clone()
        print(f"\n[Checkpoint Fase 1] Pesi e reologia caricati da: {RESUME_CHECKPOINT.name}")
        print(f"  lam (F1 fissa)   : {params_log['lam']:.6f} s (target: {LAM_TRUE:.6f})")
        print(f"  mu_p (F1 fissa)  : {params_log['mu_p']:.6f} Pa*s (target: {MU_P_TRUE:.6f})")
        print(f"  mu_tot (guess)   : {params_log['mu_tot']:.6f} Pa*s (target: {MU_TOT_TRUE:.6f})")
        print(f"  mu_s (derivata)  : {params_log['mu_s']:.6f} Pa*s (target: {MU_S_TRUE:.6f})")

        # Precalcolo del termine b' sul sottoinsieme
        print(f"\n[Precalcolo mu_tot] Calcolo del termine non-Newtoniano b'(x,y) su {CURL_SUBSET_SIZE} punti...")
        a_diag, bp_diag = physics.precompute_curl_b_prime(model, data["coords"].to(DEVICE), subset_size=CURL_SUBSET_SIZE)
        print("[Precalcolo mu_tot] Completato con successo.\n")
    else:
        raise FileNotFoundError(f"Checkpoint Fase 1 non trovato: {RESUME_CHECKPOINT}")

    # 4. Training
    launch_tensorboard_server(OUTPUT_DIR.parent)
    tb_dir = OUTPUT_DIR / "tb_logs"
    tb_dir.mkdir(parents=True, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=str(tb_dir))

    history = train(
        model,
        physics,
        data,
        resume_checkpoint=RESUME_CHECKPOINT,
        save_dir=OUTPUT_DIR,
        tb_writer=tb_writer,
    )
    tb_writer.close()

    # 5. Report Finale
    params = physics.log_params()
    print(f"\n{'=' * 60}\nRISULTATI FINALI PARAMETRI FISICI [MUTOT FORMULATION]\n{'=' * 60}")
    print(f"  eta_0 (scala rif.)  : {params['eta_0']:.6f} Pa*s")
    print(f"  mu_p (fissa F1)     : {params['mu_p']:.6f} Pa*s (true: {MU_P_TRUE:.6f})")
    print(f"  mu_tot* (adimens.)  : {params['mu_tot_nd']:.6f} (true: {MU_TOT_TRUE/ETA_0:.6f})")
    print(f"  mu_tot  (dimension.): {params['mu_tot']:.6f} Pa*s (true: {MU_TOT_TRUE:.6f})")
    print(f"  mu_s* (adimens.)    : {params['mu_s_nd']:.6f} (true: {MU_S_TRUE/ETA_0:.6f})")
    print(f"  mu_s  (dimension.)  : {params['mu_s']:.6f} Pa*s (true: {MU_S_TRUE:.6f})")
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
