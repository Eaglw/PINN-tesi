"""
train_4roll_main_mauri.py
=============================================================================
Script PINN 4-Roll Mill per PC Maurizio con Splitting Alternato (Gauss-Seidel)
nella Fase 2 (100% STANDALONE - NESSUNA MODIFICA A src/):

Architettura & Formulazione:
  - Splitting Alternato disaccoppiato (K_A = 5, K_B = 1) ispirato a ViscoelasticNet:
      * Sub-Step A (x5 step) -> Blocco Idrodinamico & Pressione:
          - model_psi: SBLOCCATA / TRAINABLE
          - model_p: SBLOCCATA / TRAINABLE
          - mu_s: SBLOCCATA / TRAINABLE (da epoca 0, NESSUN WARMUP)
          - model_tau: CONGELATO (requires_grad = False)
          - Loss: W_MOM * L_mom + W_DATA * L_data + W_BC * L_bc + W_DRIFT * L_drift
      * Sub-Step B (x1 step) -> Blocco Reologico:
          - model_psi: SBLOCCATA / TRAINABLE (ponte cinematico continuo)
          - model_tau: SBLOCCATO / TRAINABLE
          - model_p: CONGELATO (requires_grad = False)
          - mu_s: CONGELATO (requires_grad = False)
          - mu_p, lam: CONGELATI da Fase 1 (0.904854 Pa*s, 0.050203 s)
          - Loss: W_CONST * L_const + W_DATA * L_data + W_ROLL_STRESS * L_bc_stress + W_DRIFT * L_drift
  - Eliminazione totale di L_div_consist e vincolo di rotore (curl).
  - Nessun warmup: mu_s attivo e stimato dall'epoca 50001 fin da subito.
  - Logging ogni LOG_FREQUENCY = 100 epoche con report dettagliato.
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

# Import dai moduli src originali (NON MODIFICATI)
from src.debug import test_random_points, debug_physics_magnitudes
from src.physics import Physics, evaluate_final_losses, compute_l2_errors
from src.train import (
    CombinedModel,
    SimpleHistory,
    init_weights_xavier,
    initialize_last_layer_zero,
    precompute_stress_divergence,
)
from src.utils import (
    load_data,
    launch_tensorboard_server,
    generate_all_diagnostics,
    convert_to_fp64,
    convert_to_fp32,
    get_optimal_chunk_size,
)

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

# Frequenza di Logging Verboso (richiesta: ogni 100 epoche)
LOG_FREQUENCY = 100

# Rapporto di splitting alternato: K_A step su Idrodinamica per ogni K_B step su Reologia
K_A = 5
K_B = 1

# Pesi Funzione di Loss per Fase 2
W_DATA_2 = 20.0
W_BC_2 = 5.0
W_MOMENTUM = 1.0
W_CONSTITUTIVE = 1.0
W_DRIFT = 2.0

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR.parent / "COMSOL" / "4roll" / "4_roll_mill.csv"

# Checkpoint consolidato Fase 1
RESUME_CHECKPOINT = BASE_DIR / "checkpoints" / "checkpoint_inverso_fase1_40k+10k.pth"

# Parametri Fisici REALI (Ground Truth per valutazione)
MU_S_TRUE = 0.100
MU_P_TRUE = 0.900
MU_TOT_TRUE = 1.000
BETA_TRUE = 0.100
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
GUESS_MU_TOT = GUESS_MU_S + GUESS_MU_P
GUESS_BETA = GUESS_MU_S / GUESS_MU_TOT
GUESS_EPS = 0.0
GUESS_ALPHA = 0.0

HIDDEN_LAYERS = [128] * 8
ACTIVATION = nn.SiLU

# Iperparametri Fase 2 per PC Mauri (Opzione B: ~32.5 ore totali)
ADAM_EPOCHS_PHASE2 = 10000
USE_LBFGS_PHASE2 = True
LBFGS_MAX_ITERS_PHASE2 = 2000

# Supporto opzionale per test rapido locale di collaudo (es. --smoke-test)
if "--smoke-test" in sys.argv:
    print("\n[ATTENZIONE] Modalita' --smoke-test attiva: 2 macro-epoche Adam e 2 iterazioni L-BFGS.")
    ADAM_EPOCHS_PHASE2 = 2
    LOG_FREQUENCY = 1
    LBFGS_MAX_ITERS_PHASE2 = 2

# NESSUN WARMUP (richiesto esplicitamente dall'utente: mu_s attivo da epoca 0)
WARMUP_PHASE2_EPOCHS = 0

BASE_LR = 1e-3
ADAM_EPS = 1e-7
PARAM_LR_FACTOR = 0.1
GRAD_CLIP_NORM = 1000.0
PARAM_CLIP_NORM = 1.0
VARIANCE_EPS = 1e-4

# Iniezione parametri globali nei moduli src
for name, val in list(locals().items()):
    if name.isupper():
        builtins.__dict__[name] = val
        for mod in [src.debug, src.physics, src.train, src.utils]:
            mod.__dict__[name] = val

# Directory di output per PC Mauri
budget_tag = f"Ph2_{ADAM_EPOCHS_PHASE2//1000}k+{LBFGS_MAX_ITERS_PHASE2//1000}k_Alternating_KA{K_A}_KB{K_B}"
run_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
config_name = f"[{run_timestamp}][INV][STAGED][{budget_tag}][mauri]"

OUTPUT_DIR = BASE_DIR / "output_4rollmill" / config_name
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
global_log_path = OUTPUT_DIR / "train_log.txt"


# ============================================================================
# 3. CICLO DI ADDESTRAMENTO ALTERNATO (FASE 2 STANDALONE)
# ============================================================================
def train_phase2_alternating(model, physics, data, save_dir, tb_writer=None):
    """
    Ciclo di training Fase 2 a Splitting Alternato (Gauss-Seidel):
      - Sub-Step A (x K_A): allena psi, p, mu_s su Momentum + Data + BC (tau congelato).
      - Sub-Step B (x K_B): allena tau e psi su Costitutiva + Data + BC stress (p e mu_s congelati).
      - L-BFGS finale (FP64) per la raffinazione di precisione.
    """
    xy_all = data["coords"]
    uv_all = data["uv_data"]
    bc_data = data["boundary_groups"]
    var_w = data["var_weights"]
    total_points = xy_all.shape[0]

    # Inizializza cache cinematica da F1 per la soft anti-drift loss
    with torch.enable_grad():
        x_in = xy_all.clone().requires_grad_(True)
        u_ph1, v_ph1, _, _ = physics.get_velocity(model, x_in, create_graph=False)
        u_ckpt_cache = u_ph1.detach().clone().float()
        v_ckpt_cache = v_ph1.detach().clone().float()
        print("  [Checkpoint Reologico] Cache cinematica per soft anti-drift loss inizializzata.")

    # Congela parametri reologici fissati in Fase 1
    physics.set_trainable("mu_p", False)
    physics.set_trainable("lam", False)
    physics.set_trainable("mu_s", True)  # Attivo da subito (nessun warmup)

    # Parametri ottimizzatore Blocco A: psi, p, mu_s
    params_A = [
        {"params": model.model_p.parameters(), "lr": BASE_LR},
        {"params": model.model_psi.parameters(), "lr": BASE_LR * 0.1},
        {"params": [physics._raw_mu_s], "lr": BASE_LR * PARAM_LR_FACTOR},
    ]
    opt_A = torch.optim.Adam(params_A, eps=ADAM_EPS)
    sched_A = torch.optim.lr_scheduler.CosineAnnealingLR(opt_A, T_max=ADAM_EPOCHS_PHASE2 * K_A, eta_min=1e-6)

    # Parametri ottimizzatore Blocco B: tau e psi (psi sempre sbloccata)
    params_B = [
        {"params": model.model_tau.parameters(), "lr": BASE_LR * 0.1},
        {"params": model.model_psi.parameters(), "lr": BASE_LR * 0.05},
    ]
    opt_B = torch.optim.Adam(params_B, eps=ADAM_EPS)
    sched_B = torch.optim.lr_scheduler.CosineAnnealingLR(opt_B, T_max=ADAM_EPOCHS_PHASE2 * K_B, eta_min=1e-6)

    active_bcs_A = ["p", "u", "v"]
    active_bcs_B = ["u", "v", "tau_xx", "tau_xy", "tau_yy"] if USE_ROLL_STRESS_BC else ["u", "v"]

    # ====================================================================
    # PROBE DINAMICO VRAM PER FASE A E FASE B
    # ====================================================================
    print("\n[Optimization] Precalcolo iniziale divergenza sforzi (FP32)...")
    precomputed_div_tau = precompute_stress_divergence(model, physics, xy_all)
    print("  -> Divergenza sforzi iniziale completata.")

    # 1. Probe dinamico per Sub-Step A (Idrodinamica: psi, p, mu_s)
    def test_closure_A(c):
        xc = xy_all[:c]
        xph = xc.clone().requires_grad_(True)
        u, v, p, tau = physics.get_velocity(model, xph, create_graph=True)
        div_c = (precomputed_div_tau[0][:c], precomputed_div_tau[1][:c])
        lm, _ = physics.compute_pde_losses(
            xph, u, v, p, tau,
            w_momentum=1.0, w_constitutive=0.0,
            frozen_velocity=False,
            precomputed_div_tau=div_c
        )
        dl = physics.data_loss(u, v, uv_all[:c], var_w)
        loss = lm + dl
        loss.backward()

    print("\n[VRAM Optimization] Determinazione dinamica chunk size per Sub-Step A (Idrodinamica)...")
    CHUNK_SIZE_A = get_optimal_chunk_size(
        phase=2, safety_factor=0.60, min_chunk=1000, max_chunk=12000,
        model=model, test_closure=test_closure_A
    )

    # 2. Probe dinamico per Sub-Step B (Reologia: tau, psi)
    def test_closure_B(c):
        xc = xy_all[:c]
        xph = xc.clone().requires_grad_(True)
        u, v, p, tau = physics.get_velocity(model, xph, create_graph=True)
        _, lc = physics.compute_pde_losses(
            xph, u, v, p, tau,
            w_momentum=0.0, w_constitutive=1.0,
            frozen_velocity=False
        )
        dl = physics.data_loss(u, v, uv_all[:c], var_w)
        loss = lc + dl
        loss.backward()

    print("\n[VRAM Optimization] Determinazione dinamica chunk size per Sub-Step B (Reologia)...")
    CHUNK_SIZE_B = get_optimal_chunk_size(
        phase=1, safety_factor=0.60, min_chunk=1000, max_chunk=20000,
        model=model, test_closure=test_closure_B
    )

    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()

    history = SimpleHistory()

    print(f"\n{'=' * 70}")
    print(f"AVVIO FASE 2: SPLITTING ALTERNATO GAUSS-SEIDEL ({ADAM_EPOCHS_PHASE2} Macro-Epoche)")
    print(f"  Rapporto Step: K_A={K_A} (Idrodinamica: psi, p, mu_s) vs K_B={K_B} (Reologia: tau, psi)")
    print(f"  psi SEMPRE SBLOCCATA in entrambi i sub-step (ponte cinematico)")
    print(f"  Warmup: NESSUNO (mu_s libero da epoca 0) | Log ogni {LOG_FREQUENCY} epoche")
    print(f"  Chunk Sicuri: Sub-Step A={CHUNK_SIZE_A} pts | Sub-Step B={CHUNK_SIZE_B} pts")
    print(f"  Pesi: W_DATA={W_DATA_2}, W_BC={W_BC_2}, W_MOM={W_MOMENTUM}, W_CONST={W_CONSTITUTIVE}, W_DRIFT={W_DRIFT}")
    print(f"{'=' * 70}\n")

    pbar = tqdm(range(ADAM_EPOCHS_PHASE2), desc="Adam Alternato (A5-B1)", mininterval=2.0)

    for epoch_idx in pbar:
        epoch_global = 50000 + epoch_idx + 1
        model.train()

        loss_m_accum = 0.0
        loss_c_accum = 0.0
        d_loss_accum = 0.0
        b_loss_accum = 0.0

        # ====================================================================
        # SUB-STEP A (K_A iterazioni): IDRODINAMICA & PRESSIONE (psi, p, mu_s)
        # ====================================================================
        for step_a in range(K_A):
            # Configura gradi di libertà: tau CONGELATO, psi e p SBLOCCATI
            for p in model.model_tau.parameters():
                p.requires_grad = False
            for p in model.model_p.parameters():
                p.requires_grad = True
            for p in model.model_psi.parameters():
                p.requires_grad = True
            physics._raw_mu_s.requires_grad = True

            opt_A.zero_grad(set_to_none=True)
            step_a_loss_m = 0.0

            # Calcolo su chunk dinamico protetto per conservazione VRAM
            for i in range(0, total_points, CHUNK_SIZE_A):
                xc = xy_all[i : i + CHUNK_SIZE_A]
                yc = uv_all[i : i + CHUNK_SIZE_A]
                u_c0 = u_ckpt_cache[i : i + CHUNK_SIZE_A]
                v_c0 = v_ckpt_cache[i : i + CHUNK_SIZE_A]
                w_chunk = xc.shape[0] / total_points

                div_tau_c = (
                    precomputed_div_tau[0][i : i + CHUNK_SIZE_A],
                    precomputed_div_tau[1][i : i + CHUNK_SIZE_A]
                )

                xph = xc.clone().requires_grad_(True)
                u, v, p, tau = physics.get_velocity(model, xph, create_graph=True)

                # 1. Momentum loss (w_momentum=1.0, w_constitutive=0.0)
                lm, _ = physics.compute_pde_losses(
                    xph, u, v, p, tau,
                    w_momentum=1.0, w_constitutive=0.0,
                    frozen_velocity=False,
                    precomputed_div_tau=div_tau_c
                )
                step_a_loss_m += lm.item() * w_chunk

                # 2. Data loss & Soft Anti-drift loss
                dl = physics.data_loss(u, v, yc, var_w)
                drl = physics.drift_loss(u, v, u_c0, v_c0)

                chunk_loss_A = (W_MOMENTUM * lm + W_DATA_2 * dl + W_DRIFT * drl) * w_chunk
                chunk_loss_A.backward()

            # Boundary conditions Blocco A (Dirichlet PressurePoint + No-slip)
            bl_A = physics.boundary_loss(model, bc_data, var_w, active_bcs=active_bcs_A)
            if W_BC_2 > 0.0:
                (W_BC_2 * bl_A).backward()

            torch.nn.utils.clip_grad_norm_(model.model_p.parameters(), GRAD_CLIP_NORM)
            torch.nn.utils.clip_grad_norm_(model.model_psi.parameters(), GRAD_CLIP_NORM)
            torch.nn.utils.clip_grad_norm_([physics._raw_mu_s], PARAM_CLIP_NORM)

            opt_A.step()
            sched_A.step()

            if step_a == K_A - 1:
                loss_m_accum = step_a_loss_m
                b_loss_accum = bl_A.item()

        # ====================================================================
        # SUB-STEP B (K_B iterazioni): REOLOGIA & CINEMATICA (tau, psi)
        # ====================================================================
        for step_b in range(K_B):
            # Configura gradi di libertà: p e mu_s CONGELATI, tau e psi SBLOCCATI
            for p in model.model_p.parameters():
                p.requires_grad = False
            physics._raw_mu_s.requires_grad = False
            for p in model.model_tau.parameters():
                p.requires_grad = True
            for p in model.model_psi.parameters():
                p.requires_grad = True

            opt_B.zero_grad(set_to_none=True)
            step_b_loss_c = 0.0
            step_b_loss_d = 0.0

            for i in range(0, total_points, CHUNK_SIZE_B):
                xc = xy_all[i : i + CHUNK_SIZE_B]
                yc = uv_all[i : i + CHUNK_SIZE_B]
                u_c0 = u_ckpt_cache[i : i + CHUNK_SIZE_B]
                v_c0 = v_ckpt_cache[i : i + CHUNK_SIZE_B]
                w_chunk = xc.shape[0] / total_points

                xph = xc.clone().requires_grad_(True)
                u, v, p, tau = physics.get_velocity(model, xph, create_graph=True)

                # 1. Constitutive loss (w_momentum=0.0, w_constitutive=1.0)
                _, lc = physics.compute_pde_losses(
                    xph, u, v, p, tau,
                    w_momentum=0.0, w_constitutive=1.0,
                    frozen_velocity=False
                )
                step_b_loss_c += lc.item() * w_chunk

                # 2. Data loss & Anti-drift su psi
                dl = physics.data_loss(u, v, yc, var_w)
                drl = physics.drift_loss(u, v, u_c0, v_c0)
                step_b_loss_d += dl.item() * w_chunk

                chunk_loss_B = (W_CONSTITUTIVE * lc + W_DATA_2 * dl + W_DRIFT * drl) * w_chunk
                chunk_loss_B.backward()

            # Boundary conditions Blocco B (No-slip + Roller Stress BC nativo)
            bl_B_total = W_BC_2 * physics.boundary_loss(model, bc_data, var_w, active_bcs=active_bcs_B)
            bl_B_total.backward()

            torch.nn.utils.clip_grad_norm_(model.model_tau.parameters(), GRAD_CLIP_NORM)
            torch.nn.utils.clip_grad_norm_(model.model_psi.parameters(), GRAD_CLIP_NORM)

            opt_B.step()
            sched_B.step()

            if step_b == K_B - 1:
                loss_c_accum = step_b_loss_c
                d_loss_accum = step_b_loss_d

        # Ricalcola la divergenza di tau aggiornata per il prossimo Sub-step A
        precomputed_div_tau = precompute_stress_divergence(model, physics, xy_all)

        # Metriche riassuntive
        tot_loss = (W_MOMENTUM * loss_m_accum) + (W_CONSTITUTIVE * loss_c_accum) + (W_DATA_2 * d_loss_accum) + (W_BC_2 * b_loss_accum)

        # Aggiornamento postfix progress bar
        if (epoch_idx + 1) % 10 == 0:
            params_curr = physics.log_params()
            pbar.set_postfix({
                "Loss": f"{tot_loss:.2e}",
                "mu_s": f"{params_curr['mu_s']:.4f}",
                "L_mom": f"{loss_m_accum:.2e}",
                "L_const": f"{loss_c_accum:.2e}"
            })

        # Logging dettagliato ogni LOG_FREQUENCY epoche (100)
        if (epoch_idx + 1) % LOG_FREQUENCY == 0 or (epoch_idx + 1) == ADAM_EPOCHS_PHASE2:
            model.eval()
            with torch.no_grad():
                l2_errs = compute_l2_errors(model, physics, data)
                params_eval = physics.log_params()

            print(f"\n{'-' * 70}")
            print(f"[REPORT EPOCA {epoch_global}] (Adam Alternato Iter {epoch_idx + 1}/{ADAM_EPOCHS_PHASE2})")
            print(f"{'-' * 70}")
            print(f"  > Loss Totale       : {tot_loss:.4e} | Data: {d_loss_accum:.4e} | BC: {b_loss_accum:.4e}")
            print(f"  > Loss Momentum (A) : {loss_m_accum:.4e} | Loss Costitutiva (B): {loss_c_accum:.4e}")
            print(f"  > PARAMETRI FISICI:")
            print(f"      mu_s   : {params_eval['mu_s']:.6f} Pa*s (Target reale: {MU_S_TRUE:.4f}, Delta: {(params_eval['mu_s'] - MU_S_TRUE)/MU_S_TRUE*100:+.2f}%)")
            print(f"      mu_tot : {params_eval['mu_tot']:.6f} Pa*s (Target reale: {MU_TOT_TRUE:.4f}, Delta: {(params_eval['mu_tot'] - MU_TOT_TRUE)/MU_TOT_TRUE*100:+.2f}%)")
            print(f"      mu_p   : {params_eval['mu_p']:.6f} Pa*s (Fisso da F1)")
            print(f"      lam    : {params_eval['lam']:.6f} s (Fisso da F1)")
            print(f"      beta   : {params_eval['beta']:.6f} (Target reale: {BETA_TRUE:.4f})")
            print(f"  > ERRORI L2 RELATIVI:")
            print(f"      u: {l2_errs['u']:.4e} ({l2_errs['u']*100:.2f}%) | v: {l2_errs['v']:.4e} ({l2_errs['v']*100:.2f}%)")
            print(f"      p: {l2_errs['p']:.4e} ({l2_errs['p']*100:.2f}%)")
            print(f"      tau_xx: {l2_errs['tau_xx']:.4e} | tau_xy: {l2_errs['tau_xy']:.4e} | tau_yy: {l2_errs['tau_yy']:.4e}")
            print(f"{'-' * 70}\n")

            loss_dict = {
                "total": tot_loss,
                "data": d_loss_accum,
                "bc": b_loss_accum,
                "loss_momentum": loss_m_accum,
                "loss_constitutive": loss_c_accum,
                "l2_u": l2_errs["u"],
                "l2_v": l2_errs["v"],
                "l2_p": l2_errs["p"],
                "l2_tau_xx": l2_errs["tau_xx"],
                "l2_tau_xy": l2_errs["tau_xy"],
                "l2_tau_yy": l2_errs["tau_yy"],
                "param_mu_s": params_eval["mu_s"],
                "param_mu_tot": params_eval["mu_tot"]
            }
            history.update(epoch_global, loss_dict)

            if tb_writer is not None:
                tb_writer.add_scalar("Loss/Total", tot_loss, epoch_global)
                tb_writer.add_scalar("Loss/Momentum", loss_m_accum, epoch_global)
                tb_writer.add_scalar("Loss/Constitutive", loss_c_accum, epoch_global)
                tb_writer.add_scalar("Params/mu_s", params_eval["mu_s"], epoch_global)
                tb_writer.add_scalar("Params/mu_tot", params_eval["mu_tot"], epoch_global)
                tb_writer.add_scalar("L2_Error/p", l2_errs["p"], epoch_global)
                tb_writer.add_scalar("L2_Error/u", l2_errs["u"], epoch_global)
                tb_writer.flush()

            # Salvataggio periodico checkpoint
            state_chk = {
                "epoch": epoch_global,
                "model_state_dict": model.state_dict(),
                "physics_state_dict": physics.state_dict(),
                "history_state_dict": history.state_dict(),
            }
            torch.save(state_chk, save_dir / "checkpoint.pth")

    pbar.close()

    # Salvataggio fine Fase 2 Adam
    torch.save({
        "epoch": 50000 + ADAM_EPOCHS_PHASE2,
        "model_state_dict": model.state_dict(),
        "physics_state_dict": physics.state_dict(),
        "history_state_dict": history.state_dict(),
    }, save_dir / "checkpoint_phase2_adam.pth")
    print(f"\n[Checkpoint] Fase 2 Adam salvato in: {save_dir / 'checkpoint_phase2_adam.pth'}")

    # ====================================================================
    # L-BFGS FASE 2 REFINEMENT (FP64)
    # ====================================================================
    if USE_LBFGS_PHASE2 and LBFGS_MAX_ITERS_PHASE2 > 0:
        print(f"\n{'=' * 70}")
        print(f"AVVIO L-BFGS FASE 2: RAFFINAMENTO AD ALTA PRECISIONE ({LBFGS_MAX_ITERS_PHASE2} iters, FP64)")
        print(f"{'=' * 70}")

        convert_to_fp64(model, physics, data)
        torch.cuda.empty_cache()

        xy_all = data["coords"]
        uv_all = data["uv_data"]
        bc_data = data["boundary_groups"]

        # In L-BFGS: tau resta congelato allo stato armonizzato di Adam, psi e p e mu_s rifiniti
        for p in model.parameters():
            p.requires_grad = False
        for p in model.model_p.parameters():
            p.requires_grad = True
        for p in model.model_psi.parameters():
            p.requires_grad = True
        physics._raw_mu_s.requires_grad = True

        # Precalcolo statico della divergenza dello stress tau in FP64
        print("  Precalcolo divergenza di tau per L-BFGS (FP64)...")
        div_tau_lbfgs = precompute_stress_divergence(model, physics, xy_all)
        print("  -> Divergenza per L-BFGS pronta.")

        CHUNK_SIZE_LBFGS = min(CHUNK_SIZE_A, 5000)
        print(f"  -> Chunk Size protetto per L-BFGS (FP64): {CHUNK_SIZE_LBFGS} punti")

        opt_lbfgs = torch.optim.LBFGS(
            [p for p in model.parameters() if p.requires_grad] + [physics._raw_mu_s],
            lr=1.0,
            max_iter=LBFGS_MAX_ITERS_PHASE2,
            history_size=50,
            tolerance_grad=1e-12,
            tolerance_change=1e-14,
            line_search_fn="strong_wolfe"
        )

        lbfgs_iter = 0
        pbar_lbfgs = tqdm(total=LBFGS_MAX_ITERS_PHASE2, desc="L-BFGS Fase 2 (FP64)")

        def closure():
            nonlocal lbfgs_iter
            opt_lbfgs.zero_grad(set_to_none=True)

            tot_loss_val = 0.0
            for i in range(0, total_points, CHUNK_SIZE_LBFGS):
                xc = xy_all[i : i + CHUNK_SIZE_LBFGS]
                yc = uv_all[i : i + CHUNK_SIZE_LBFGS]
                w_chunk = xc.shape[0] / total_points

                div_chunk = (
                    div_tau_lbfgs[0][i : i + CHUNK_SIZE_LBFGS],
                    div_tau_lbfgs[1][i : i + CHUNK_SIZE_LBFGS]
                )

                xph = xc.clone().requires_grad_(True)
                u, v, p, tau = physics.get_velocity(model, xph, create_graph=True)

                lm, _ = physics.compute_pde_losses(
                    xph, u, v, p, tau,
                    w_momentum=1.0, w_constitutive=0.0,
                    frozen_velocity=False,
                    precomputed_div_tau=div_chunk
                )
                dl = physics.data_loss(u, v, yc, var_w)

                chunk_loss = (W_MOMENTUM * lm + W_DATA_2 * dl) * w_chunk
                chunk_loss.backward()
                tot_loss_val += chunk_loss.item()

            bl = physics.boundary_loss(model, bc_data, var_w, active_bcs=active_bcs_A)
            (W_BC_2 * bl).backward()
            tot_loss_val += (W_BC_2 * bl).item()

            lbfgs_iter += 1
            pbar_lbfgs.update(1)

            if lbfgs_iter % 50 == 0:
                with torch.no_grad():
                    l2_e = compute_l2_errors(model, physics, data)
                    p_curr = physics.log_params()
                print(f"\n  [L-BFGS Iter {lbfgs_iter}] Loss: {tot_loss_val:.4e} | mu_s: {p_curr['mu_s']:.6f} | L2(p): {l2_e['p']*100:.2f}% | L2(u): {l2_e['u']*100:.2f}%")

            return torch.tensor(tot_loss_val, device=DEVICE, dtype=torch.float64)

        opt_lbfgs.step(closure)
        pbar_lbfgs.close()

        torch.save({
            "model_state_dict": model.state_dict(),
            "physics_state_dict": physics.state_dict(),
            "history_state_dict": history.state_dict()
        }, save_dir / "checkpoint_lbfgs_phase2.pth")
        print(f"\n[Checkpoint] L-BFGS Fase 2 salvato in: {save_dir / 'checkpoint_lbfgs_phase2.pth'}")

    return history


# ============================================================================
# 4. MAIN ENTRY POINT
# ============================================================================
def main():
    print("=" * 80)
    print("PINN 4-ROLL MILL: ADDESTRAMENTO FASE 2 ALTERNATA (PC MAURIZIO)")
    print(f"Device: {DEVICE} | Checkpoint: {RESUME_CHECKPOINT.name}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print("=" * 80)

    # 1. Carica dataset e modelli
    data = load_data(filepath=DATASET_PATH, eta_0=ETA_0)

    model = CombinedModel(p_scale=data["p_scale"], tau_scale=data["tau_scale"]).to(DEVICE)
    physics = Physics(
        U_ref=data["U_ref"],
        H_ref=data["H"],
        H_coord=data["H_coord"],
        var_weights=data["var_weights"],
        inverse_mode=True,
        tau_scale=data["tau_scale"],
        p_scale=data["p_scale"],
        eta_0=ETA_0,
    ).to(DEVICE)

    # 2. Carica Checkpoint Fase 1
    if not RESUME_CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint non trovato: {RESUME_CHECKPOINT}")

    chk = torch.load(str(RESUME_CHECKPOINT), map_location=DEVICE)
    model.load_state_dict(chk['model_state_dict'])
    physics.load_state_dict(chk['physics_state_dict'], strict=False)

    params_f1 = physics.log_params()
    print(f"\n[Fase 1 Consolidata Caricata con Successo]")
    print(f"  mu_p : {params_f1['mu_p']:.6f} Pa*s (Target: {MU_P_TRUE:.4f})")
    print(f"  lam  : {params_f1['lam']:.6f} s (Target: {LAM_TRUE:.4f})")
    print(f"  mu_s (guess iniziale): {params_f1['mu_s']:.6f} Pa*s (Target: {MU_S_TRUE:.4f})")
    print(f"  eta_0: {ETA_0:.4f} Pa*s")

    # 3. Setup TensorBoard
    launch_tensorboard_server(OUTPUT_DIR.parent)
    tb_dir = OUTPUT_DIR / "tb_logs"
    tb_dir.mkdir(parents=True, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=str(tb_dir))

    # 4. Esecuzione Addestramento
    history = train_phase2_alternating(model, physics, data, save_dir=OUTPUT_DIR, tb_writer=tb_writer)
    tb_writer.close()

    # 5. Report Finale e Generazione Diagnostiche
    print(f"\n{'=' * 70}\nREPORT FINALE PARAMETRI IDENTIFICATI\n{'=' * 70}")
    params_final = physics.log_params()
    for k, v in params_final.items():
        print(f"  {k:<15s}: {v:.6f}")

    errors_final = compute_l2_errors(model, physics, data)
    print(f"\n{'=' * 70}\nERRORI L2 RELATIVI FINALI\n{'=' * 70}")
    for k, v in errors_final.items():
        print(f"  {k:<15s}: {v:.6e} ({v*100:.2f}%)")

    # Plot e grafici
    print(f"\nGenerazione grafici e diagnostiche in corso in: {OUTPUT_DIR} ...")
    history.plot_losses(str(OUTPUT_DIR / "loss_history.png"))
    history.plot_params(str(OUTPUT_DIR / "params_evolution.png"))
    history.plot_l2_errors(str(OUTPUT_DIR / "l2_errors_history.png"))
    generate_all_diagnostics(model, physics, data, str(OUTPUT_DIR))

    print(f"\n{'=' * 70}\nRUN COMPLETATA CON SUCCESSO SUL PC DI MAURIZIO!\n{'=' * 70}")


if __name__ == "__main__":
    main()
