"""
train_4roll_main_mauri.py
=============================================================================
Script PINN 4-Roll Mill per PC Maurizio: Addestramento Fase 2 Standard Disaccoppiata.

Formulazione Standard Disaccoppiata (ViscoelasticNet Framework):
  - Inizializzazione dal checkpoint consolidato di Fase 1 (checkpoint_inverso_fase1_40k+10k.pth, epoca 49999).
  - Tensore degli sforzi tau (model_tau) rigorosamente CONGELATO (requires_grad = False).
  - Stream function psi (model_psi) MOBILE con micro-learning rate controllato (1e-4 = 0.1 * BASE_LR).
  - Pressione p (model_p) SBLOCCATA e addestrabile (lr = BASE_LR = 1e-3).
  - Parametrizzazione viscosità totale mu_tot con calcolo mu_s protetto da softplus (Proposta AC).
  - Ancoraggio algebrico HARD della pressione su model_p (Proposta AB) ed esclusione della penalty soft.
  - Adimensionalizzazione e riscalamento del residuo di Navier-Stokes per scale_mom = (eta_0 * U_ref) / (H_coord ** 2) (Proposta AA).
  - Igiene numerica: TF32 disabilitato, Adam EPS differenziato (1e-8 / 1e-15), Gradient Clipping a 5.0.
  - L-BFGS Fase 2: history_size = 300, line_search_fn = "strong_wolfe" (Run 23 tuning).
  - Diagnostica preventiva del rapporto di identificabilità di Leray rho_id a inizio Fase 2 (Proposta AE).
  - Budget standard: 20.000 epoche Adam + 2.000 iterazioni L-BFGS (2 + 2 con flag --smoke-test).
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

# Import dai moduli src
from src.debug import test_random_points, debug_physics_magnitudes
from src.physics import Physics, evaluate_final_losses, compute_l2_errors
from src.train import CombinedModel, initialize_last_layer_zero, init_weights_xavier, train
from src.utils import (
    load_data,
    plot_fields,
    plot_high_stress_regions,
    launch_tensorboard_server,
    generate_all_diagnostics,
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
# [Proposta A] Disabilita TF32 per garantire la piena precisione FP32 standard IEEE (23-bit mantissa)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
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

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR.parent / "COMSOL" / "4roll" / "4_roll_mill.csv"

# Checkpoint consolidato Fase 1
RESUME_CHECKPOINT = BASE_DIR / "checkpoints" / "checkpoint_inverso_fase1_40k+10k.pth"

# Parametri Fisici REALI (Ground Truth)
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

# Budget Fase 1 (usato da src/train per indicizzare la transizione al checkpoint F1 40k+10k)
ADAM_EPOCHS_PHASE1 = 40000
USE_LBFGS_PHASE1 = True
LBFGS_MAX_ITERS_PHASE1 = 10000

# Budget Fase 2 per PC Maurizio (Standard: 20.000 Adam + 2.000 L-BFGS)
ADAM_EPOCHS_PHASE2 = 20000
USE_LBFGS_PHASE2 = True
LBFGS_MAX_ITERS_PHASE2 = 2000

# Nessun warmup su mu_tot in Fase 2 (attivo e stimato da epoca 0)
WARMUP_PHASE2_EPOCHS = 0
USE_MU_TOT_PARAM = True

# Supporto per esecuzione rapida di collaudo (--smoke-test)
if "--smoke-test" in sys.argv:
    print("\n[ATTENZIONE] Modalita' --smoke-test attiva: 2 epoche Adam e 2 iterazioni L-BFGS.")
    ADAM_EPOCHS_PHASE2 = 2
    LBFGS_MAX_ITERS_PHASE2 = 2

BASE_LR = 1e-3
ADAM_EPS = 1e-8
ADAM_EPS_PHYS = 1e-15
PARAM_LR_FACTOR = 0.1
GRAD_CLIP_NORM = 5.0
PARAM_CLIP_NORM = 1.0

# Pesi di Loss
W_DATA_1 = 1.0
W_BC_1 = 5.0
W_CONSTITUTIVE = 1.0

W_DATA_2 = 20.0
W_BC_2 = 5.0
W_MOMENTUM = 1.0
W_DRIFT = 0.0
VARIANCE_EPS = 1e-4

# ============================================================================
# 3. INIZIALIZZAZIONE OUTPUT
# ============================================================================
layers_str = f"{len(HIDDEN_LAYERS)}x{HIDDEN_LAYERS[0]}"
mode_tag = "INV" if INVERSE_PROBLEM else "DIR"
strategy_tag = "STAGED" if STAGED_TRAINING else "MONO"

def _format_iters(n):
    if n == 0:
        return "0"
    if n % 1000 == 0:
        return f"{n // 1000}k"
    return f"{n / 1000:.1f}k"

budget_tag = f"Ph2_{_format_iters(ADAM_EPOCHS_PHASE2)}+{_format_iters(LBFGS_MAX_ITERS_PHASE2)}_Warmup{_format_iters(WARMUP_PHASE2_EPOCHS)}"
run_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
config_name = f"[{run_timestamp}][{mode_tag}][{strategy_tag}][{budget_tag}][mauri]"

OUTPUT_DIR = BASE_DIR / "output_4rollmill" / config_name
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
global_log_path = OUTPUT_DIR / "train_log.txt"

# Iniezione dinamica dei parametri globali nei moduli src e builtins
for name, val in list(globals().items()):
    if name.isupper():
        builtins.__dict__[name] = val
        for module in [src.debug, src.physics, src.train, src.utils]:
            module.__dict__[name] = val

# ============================================================================
# 4. MAIN ENTRY POINT
# ============================================================================
def main():
    print("=" * 80)
    print("PINN 4-ROLL MILL: ADDESTRAMENTO FASE 2 STANDARD DISACCOPPIATA (PC MAURIZIO)")
    print(f"Device: {DEVICE} | Dtype: {torch.get_default_dtype()}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Checkpoint di partenza: {RESUME_CHECKPOINT.name}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print("=" * 80)

    # 1. Caricamento Dati
    data = load_data(filepath=DATASET_PATH, eta_0=ETA_0)

    # 2. Inizializzazione Modello e Fisica con Ancoraggio Hard e Stress Vettoriale
    model = CombinedModel(
        p_scale=data["p_scale"],
        tau_scale=data["tau_scale_vec"],
        x_anchor=data.get("x_anchor"),
        p_ref=data.get("p_ref", 0.0),
    ).to(DEVICE)
    for submodel in [model.model_psi, model.model_p, model.model_tau]:
        submodel.apply(lambda m: init_weights_xavier(m, activation_name=ACTIVATION))

    initialize_last_layer_zero(model.model_p)
    initialize_last_layer_zero(model.model_tau)

    physics = Physics(
        U_ref=data["U_ref"],
        H_ref=data["H"],
        H_coord=data["H_coord"],
        var_weights=data["var_weights"],
        inverse_mode=INVERSE_PROBLEM,
        tau_scale=data["tau_scale_vec"],
        p_scale=data["p_scale"],
        eta_0=ETA_0,
    ).to(DEVICE)

    # Verifica presenza checkpoint Fase 1
    if not RESUME_CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint Fase 1 non trovato: {RESUME_CHECKPOINT}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModello inizializzato: {total_params:,} parametri totali")
    print(f"Configurazione Fase 2 Standard:")
    print(f"  - model_tau: CONGELATO")
    print(f"  - model_p: ADDESTRABILE (lr={BASE_LR})")
    print(f"  - model_psi: MOBILE (micro-lr={BASE_LR * 0.1})")
    print(f"  - mu_tot: ADDESTRABILE (softplus mu_s protetto da valori negativi)")
    print(f"  - Ancoraggio Pressione: HARD ALGEBRICO (p(x0) = p_ref esatto, penalty soft rimossa)")
    print(f"  - Riscalamento Momento: scale_mom = {physics.scale_mom.item():.4f} Pa/m")
    print(f"  - Budget: {ADAM_EPOCHS_PHASE2} Adam + {LBFGS_MAX_ITERS_PHASE2} L-BFGS (history_size=300, strong_wolfe)")

    # 3. Setup TensorBoard
    launch_tensorboard_server(OUTPUT_DIR.parent)
    tb_dir = OUTPUT_DIR / "tb_logs"
    tb_dir.mkdir(parents=True, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=str(tb_dir))

    # 4. Esecuzione Addestramento Standard Fase 2
    history = train(
        model,
        physics,
        data,
        resume_checkpoint=RESUME_CHECKPOINT,
        save_dir=OUTPUT_DIR,
        tb_writer=tb_writer
    )
    tb_writer.close()

    # 5. Report Risultati Finali
    params = physics.log_params()
    print(f"\n{'=' * 60}\nRISULTATI FINALI PARAMETRI FISICI (Dimensionali e Adimensionali)\n{'=' * 60}")
    print(f"  eta_0 (scala rif.) : {params['eta_0']:.6f} Pa·s")
    print(f"  mu_p* (adimens.)   : {params['mu_p_nd']:.6f}  (true: {MU_P_TRUE/ETA_0:.6f})")
    print(f"  mu_p  (dimension.) : {params['mu_p']:.6f} Pa·s (true: {MU_P_TRUE:.6f})")
    print(f"  mu_s* (adimens.)   : {params['mu_s_nd']:.6f}  (true: {MU_S_TRUE/ETA_0:.6f})")
    print(f"  mu_s  (dimension.) : {params['mu_s']:.6f} Pa·s (true: {MU_S_TRUE:.6f})")
    print(f"  mu_tot (dimension.): {params['mu_tot']:.6f} Pa·s (true: {MU_TOT_TRUE:.6f})")
    print(f"  beta  (ratio)      : {params['beta']:.6f}  (true: {BETA_TRUE:.6f})")
    print(f"  lam   (dimension.) : {params['lam']:.6f} s (true: {LAM_TRUE:.6f})")
    print(f"  eps   (PTT)        : {params['eps']:.6f}  (true: {EPS_TRUE:.6f})")
    print(f"  alpha (Giesekus)   : {params['alpha']:.6f}  (true: {ALPHA_TRUE:.6f})")

    final_losses = evaluate_final_losses(model, physics, data)
    print(f"\n{'=' * 60}\nREPORT FINALE DETTAGLIATO\n{'=' * 60}")
    for k, v in final_losses.items():
        print(f"  {k:<20s}: {v:.6e}")

    errors = compute_l2_errors(model, physics, data)
    print("\nL2 Relative Errors:")
    for fn, err in errors.items():
        print(f"  {fn:>8s}: {err:.6f} ({err*100:.2f}%)")

    # 6. Generazione Plot e Diagnostiche
    print(f"\nGenerazione diagnostiche e plot in: {OUTPUT_DIR} ...")
    history.plot_losses(str(OUTPUT_DIR / "loss_history.png"))
    history.plot_params(str(OUTPUT_DIR / "params_evolution.png"))
    history.plot_l2_errors(str(OUTPUT_DIR / "l2_errors_history.png"))
    generate_all_diagnostics(model, physics, data, str(OUTPUT_DIR))

    print(f"\n[OK] Run Fase 2 Standard completata con successo sul PC di Maurizio! Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
