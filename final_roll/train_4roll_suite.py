import os
import sys
import argparse
import json
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
from src.utils import load_data, plot_fields, launch_tensorboard_server

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
        try:
            with open(global_log_path, 'a', encoding='utf-8') as f:
                f.write(text)
        except Exception:
            pass

builtins.print = custom_print

# ============================================================================
# PARSER ARGOMENTI CLI PER SUITE DI TEST
# ============================================================================
parser = argparse.ArgumentParser(description="PINN 4-Roll Mill - Suite di Test Fase 1 (Invarianza eta_0)")
parser.add_argument("--eta0", type=float, default=2.0, help="Viscosita' di riferimento eta_0 (default: 2.0)")
parser.add_argument("--epochs-ph1", type=int, default=40000, help="Epoche Adam Fase 1 (default: 40000)")
parser.add_argument("--lbfgs-ph1", type=int, default=10000, help="Iterazioni L-BFGS Fase 1 (default: 10000)")
parser.add_argument("--no-lbfgs", action="store_true", help="Disattiva L-BFGS Fase 1")
parser.add_argument("--resume", type=str, default=None, help="Percorso eventuale checkpoint da riprendere (default: None)")
parser.add_argument("--seed", type=int, default=123, help="Seed casuale (default: 123)")
parser.add_argument("--tag", type=str, default="", help="Tag aggiuntivo per il nome della run")
parser.add_argument("--no-tb", action="store_true", help="Disattiva avvio automatico server TensorBoard")
args, unknown = parser.parse_known_args()

# ============================================================================
# 1. SETUP AMBIENTE E PYTORCH
# ============================================================================
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

torch.set_default_dtype(torch.float32)
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = False

SEED = args.seed
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

RESUME_CHECKPOINT = Path(args.resume) if args.resume else None

# Parametri Fisici REALI (Ground Truth)
MU_S_TRUE = 0.1
MU_P_TRUE = 0.9
MU_TOT_TRUE = MU_S_TRUE + MU_P_TRUE
BETA_TRUE = MU_S_TRUE / MU_TOT_TRUE
LAM_TRUE = 0.05
EPS_TRUE = 0.0
ALPHA_TRUE = 0.0
RHO = 1000.0

MIN_MU_S = 1e-6
MIN_MU_P = 1e-6
MIN_LAM = 1e-6

# Scala di normalizzazione globale di riferimento (da CLI)
ETA_0 = float(args.eta0)

# Fattore di perturbazione per i parametri del problema inverso
GUESS_FACTOR = 0.80

GUESS_LAM = LAM_TRUE * GUESS_FACTOR
GUESS_MU_S = MU_S_TRUE * GUESS_FACTOR
GUESS_MU_P = MU_P_TRUE * GUESS_FACTOR
GUESS_MU_TOT = GUESS_MU_S + GUESS_MU_P
GUESS_BETA = GUESS_MU_S / GUESS_MU_TOT
GUESS_EPS = 0.0
GUESS_ALPHA = 0.0

# Architettura Neural Network
HIDDEN_LAYERS = [128] * 8
ACTIVATION = nn.SiLU

# Iperparametri di Training: SOLO FASE 1 (Cinematica & Reologia)
ADAM_EPOCHS_PHASE1 = args.epochs_ph1
USE_LBFGS_PHASE1 = not args.no_lbfgs
LBFGS_MAX_ITERS_PHASE1 = args.lbfgs_ph1 if USE_LBFGS_PHASE1 else 0

# Fase 2 disattivata per il benchmark reologico
ADAM_EPOCHS_PHASE2 = 0
USE_LBFGS_PHASE2 = False
LBFGS_MAX_ITERS_PHASE2 = 0

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

# ============================================================================
# 3. INIZIALIZZAZIONE OUTPUT
# ============================================================================
layers_str = f"{len(HIDDEN_LAYERS)}x{HIDDEN_LAYERS[0]}"

def _format_iters(n):
    if n == 0:
        return "0"
    if n % 1000 == 0:
        return f"{n // 1000}k"
    return f"{n / 1000:.1f}k"

mode_tag = "INV" if INVERSE_PROBLEM else "DIR"
strategy_tag = "STAGED" if STAGED_TRAINING else "MONO"
budget_tag = f"Ph1_{_format_iters(ADAM_EPOCHS_PHASE1)}+{_format_iters(LBFGS_MAX_ITERS_PHASE1)}"
eta_tag = f"eta0_{ETA_0:.2f}".replace('.', '_')
extra_tag = f"_{args.tag}" if args.tag else ""
run_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

config_name = f"[{mode_tag}][{strategy_tag}][{budget_tag}][{eta_tag}{extra_tag}][{run_timestamp}]"

OUTPUT_DIR = BASE_DIR / "output_4rollmill" / "suite_eta0" / config_name
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

global_log_path = OUTPUT_DIR / "train_log.txt"

# Iniezione dinamica dei parametri globali nei moduli di src
for module in [src.debug, src.physics, src.train, src.utils]:
    for name, val in list(globals().items()):
        if name.isupper():
            module.__dict__[name] = val

if __name__ == "__main__":
    print(f"Device: {DEVICE} | Dtype: {torch.get_default_dtype()}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Configurazione Suite: ETA_0={ETA_0:.4f} | Adam Ph1={ADAM_EPOCHS_PHASE1} | L-BFGS Ph1={LBFGS_MAX_ITERS_PHASE1}\n")
    print("=" * 60)

    # 1. Caricamento Dati
    data = load_data(eta_0=ETA_0)

    # 2. Inizializzazione Modello e Fisica
    model = CombinedModel(p_scale=data["p_scale"], tau_scale=data["tau_scale"]).to(DEVICE)
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
        tau_scale=data["tau_scale"],
        p_scale=data["p_scale"],
        eta_0=ETA_0,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModello: {total_params:,} parametri totali")
    print(f"Modalità: SUITE INVERSO FASE 1 (Cinematica & Reologia)")
    print(f"  - eta_0: {physics.eta_0.item():.4f} Pa·s")
    print(f"  - Guess Iniziali: lam={physics.lam.item():.4f} s (true: {LAM_TRUE}), mu_p={physics.mu_p.item():.4f} Pa·s (true: {MU_P_TRUE})")
    print(f"  - Output Scales: p_scale={data['p_scale']:.4f}, tau_scale={data['tau_scale']:.4f}")
    print("=" * 60)

    # 3. TensorBoard
    if not args.no_tb:
        launch_tensorboard_server(OUTPUT_DIR.parent)

    tb_dir = OUTPUT_DIR / "tb_logs"
    tb_dir.mkdir(parents=True, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=str(tb_dir))

    # 4. Training
    history = train(
        model, 
        physics, 
        data, 
        resume_checkpoint=str(RESUME_CHECKPOINT) if RESUME_CHECKPOINT else None,
        save_dir=OUTPUT_DIR,
        tb_writer=tb_writer
    )

    tb_writer.close()

    # 5. Report Risultati Finali
    params = physics.log_params()
    mu_p_final = params['mu_p']
    lam_final = params['lam']
    mu_p_err = abs(mu_p_final - MU_P_TRUE) / MU_P_TRUE
    lam_err = abs(lam_final - LAM_TRUE) / LAM_TRUE

    print(f"\n{'=' * 60}\nRISULTATI FINALI PARAMETRI FISICI (eta_0 = {ETA_0:.4f})\n{'=' * 60}")
    print(f"  eta_0               : {params['eta_0']:.6f} Pa·s")
    print(f"  mu_p* (adimens.)    : {params['mu_p_nd']:.6f}  (true: {MU_P_TRUE/ETA_0:.6f})")
    print(f"  mu_p  (dimension.)  : {mu_p_final:.6f} Pa·s (true: {MU_P_TRUE:.6f}) -> Err Rel: {mu_p_err*100:.2f}%")
    print(f"  lam   (dimension.)  : {lam_final:.6f} s      (true: {LAM_TRUE:.6f}) -> Err Rel: {lam_err*100:.2f}%")

    final_losses = evaluate_final_losses(model, physics, data)
    print(f"\n{'=' * 60}\nREPORT LOSS FINALI\n{'=' * 60}")
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

    # Salva dizionario di riepilogo JSON per l'orchestratore
    summary_data = {
        "eta_0": ETA_0,
        "adam_epochs": ADAM_EPOCHS_PHASE1,
        "lbfgs_iters": LBFGS_MAX_ITERS_PHASE1,
        "mu_p_final": mu_p_final,
        "mu_p_true": MU_P_TRUE,
        "mu_p_rel_err": mu_p_err,
        "lam_final": lam_final,
        "lam_true": LAM_TRUE,
        "lam_rel_err": lam_err,
        "tau_scale": float(data["tau_scale"]),
        "p_scale": float(data["p_scale"]),
        "final_losses": {k: float(v) for k, v in final_losses.items()},
        "l2_errors": {k: float(v) for k, v in errors.items()},
        "epochs_history": history.epochs,
        "loss_history": {k: [float(x) if x is not None else None for x in v] for k, v in history.losses.items()}
    }

    with open(OUTPUT_DIR / "suite_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\n[OK] Run Suite completata con successo. Risultati in: {OUTPUT_DIR}")
