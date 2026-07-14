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

# Setup base directory
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))
if str(BASE_DIR.parent) not in sys.path:
    sys.path.append(str(BASE_DIR.parent))

# Import dai moduli src
from src.debug import test_random_points, debug_physics_magnitudes
from src.physics import Physics, evaluate_final_losses, compute_l2_errors
from src.train import CombinedModel, initialize_last_layer_zero, init_weights_xavier, train
from src.utils import load_data, plot_fields, plot_high_stress_regions

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
torch.set_float32_matmul_precision("high")  # Abilita TF32 per matmul (Ampere+)
torch.backends.cudnn.benchmark = False

# Fissiamo i seed per la riproducibilità
SEED = 123
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# 2. COSTANTI E CONFIGURAZIONI GLOBALI
# ============================================================================

# --- Opzioni di Controllo ---
EXPORT_TO_OBSIDIAN = False  # False per ambiente Kaggle
STAGED_TRAINING = True  # True: staged (Fase 1: psi+tau, Fase 2: psi+p)
INVERSE_PROBLEM = False  # True: semi-inverso, False: diretto
DEBUG_MODE = False

# --- Checkpointing ---
RESUME_CHECKPOINT = None

# --- Auto-detect dataset path (Kaggle / Local) ---
if "KAGGLE_KERNEL_RUN_TYPE" in os.environ or not (BASE_DIR.parent / "COMSOL" / "4roll" / "4_roll_mill.csv").exists():
    import glob
    matches = glob.glob("/kaggle/input/**/4_roll_mill.csv", recursive=True)
    if matches:
        DATASET_PATH = Path(matches[0])
    else:
        # Ricerca ricorsiva locale come fallback
        matches_local = glob.glob("**/4_roll_mill.csv", recursive=True)
        if matches_local:
            DATASET_PATH = Path(matches_local[0]).resolve()
        else:
            DATASET_PATH = BASE_DIR.parent / "COMSOL" / "4roll" / "4_roll_mill.csv"
else:
    DATASET_PATH = BASE_DIR.parent / "COMSOL" / "4roll" / "4_roll_mill.csv"

# --- Parametri Fisici REALI (Ground Truth) ---
MU_S_TRUE = 0.1  # Viscosità solvente [Pa·s]
MU_P_TRUE = 0.9  # Viscosità polimerica [Pa·s]
LAM_TRUE = 0.05  # Tempo di rilassamento [s]
EPS_TRUE = 0.0  # Parametro PTT
ALPHA_TRUE = 0.0  # Parametro Giesekus
RHO = 1000.0  # Densità [kg/m³]

# --- Costanti e Guess Iniziali ---
MIN_MU_S = 1e-6
MIN_MU_P = 1e-6
MIN_LAM = 1e-6

GUESS_MULTIPLIER = 0.8
GUESS_MU_S = MU_S_TRUE * GUESS_MULTIPLIER
GUESS_MU_P = MU_P_TRUE * GUESS_MULTIPLIER
GUESS_LAM = LAM_TRUE * GUESS_MULTIPLIER
GUESS_EPS = 0.0
GUESS_ALPHA = 0.0

# --- Architettura Neural Network ---
HIDDEN_LAYERS = [128] * 8  # 8 hidden layers da 128 neuroni
ACTIVATION = nn.SiLU

# --- Iperparametri di Training ---
ADAM_EPOCHS_PHASE1 = 200000  # Allungato come richiesto
ADAM_EPOCHS_PHASE2 = 80000   # Allungato come richiesto
USE_LBFGS_PHASE1 = True
USE_LBFGS_PHASE2 = True
LBFGS_MAX_ITERS_PHASE1 = 10000
LBFGS_MAX_ITERS_PHASE2 = 10000
BASE_LR = 1e-3
ADAM_EPS = 1e-7
PARAM_LR_FACTOR = 0.1
GRAD_CLIP_NORM = 1000.0
PARAM_CLIP_NORM = 1.0

WARMUP_UNLOCK_EPOCH = int(0.2 * ADAM_EPOCHS_PHASE1)

# --- Pesi Funzione di Loss ---
W_BC = 5.0      # Aumentato da 2.0 a 5.0 per vincolare meglio l'ancoraggio del punto di pressione
W_PHYSICS = 3.0
W_DATA = 1.0    # Attivato (1.0) per permettere a psi di addestrarsi usando anche i dati di velocità di COMSOL
W_MOMENTUM = 1.0
W_CONSTITUTIVE = 1.0
VARIANCE_EPS = 1e-4

# ============================================================================
# 3. INIZIALIZZAZIONE OUTPUT
# ============================================================================
layers_str = f"{len(HIDDEN_LAYERS)}x{HIDDEN_LAYERS[0]}"
config_name = f"{DATASET_PATH.stem}_Kaggle3_L{layers_str}_E{ADAM_EPOCHS_PHASE1+ADAM_EPOCHS_PHASE2}_{ACTIVATION.__name__}_staged{STAGED_TRAINING}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

OUTPUT_DIR = BASE_DIR / "output_4rollmill" / config_name
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

global_log_path = OUTPUT_DIR / "train_log.txt"

# Iniezione dinamica dei parametri globali nei moduli di src per risolvere la mancanza di config
for module in [src.debug, src.physics, src.train, src.utils]:
    for name, val in list(globals().items()):
        if name.isupper():
            module.__dict__[name] = val

if __name__ == "__main__":
    print(f"Device: {DEVICE} | Dtype: {torch.get_default_dtype()}")
    print(f"Dataset: {DATASET_PATH}\n")
    print("=" * 60)

    # 1. Caricamento Dati
    data = load_data()

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
    ).to(DEVICE)

    # Recap Configurazione
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModello: {total_params:,} parametri totali")
    print(f" Valori veri: mu_s={MU_S_TRUE}, mu_p={MU_P_TRUE}, lam={LAM_TRUE}")

    # 3. Training Setup TensorBoard
    tb_dir = OUTPUT_DIR / "tb_logs"
    tb_dir.mkdir(parents=True, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=str(tb_dir))
    
    # Esecuzione del training
    history = train(
        model, 
        physics, 
        data, 
        resume_checkpoint=RESUME_CHECKPOINT,
        save_dir=OUTPUT_DIR,
        tb_writer=tb_writer
    )
    
    tb_writer.close()

    # 4. Report Risultati Finali
    params = physics.log_params()
    print(f"\n{'=' * 60}\nRISULTATI FINALI PARAMETRI FISICI\n{'=' * 60}")
    for p_name, true_val in zip(
        ["mu_s", "mu_p", "lam", "eps", "alpha"],
        [MU_S_TRUE, MU_P_TRUE, LAM_TRUE, EPS_TRUE, ALPHA_TRUE],
    ):
        print(f"  {p_name:<5s}: {params[p_name]:.6f}  (true: {true_val})")

    final_losses = evaluate_final_losses(model, physics, data)
    print(f"\n{'=' * 60}\nREPORT FINALE DETTAGLIATO\n{'=' * 60}")
    for k, v in final_losses.items():
        print(f"  {k:<20s}: {v:.6e}")

    errors = compute_l2_errors(model, physics, data)
    print("\nL2 Relative Errors:")
    for fn, err in errors.items():
        print(f"  {fn:>8s}: {err:.6f}")

    # 5. Salvataggio del checkpoint finale
    final_checkpoint_path = OUTPUT_DIR / "final_model_checkpoint.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'physics_state_dict': physics.state_dict(),
        'history_state_dict': history.state_dict()
    }, final_checkpoint_path)
    print(f"\n[Checkpoint Finale] Salvato con successo in: {final_checkpoint_path}")

    # 6. Generazione Plot
    history.plot_losses(str(OUTPUT_DIR / "loss_history.png"))
    history.plot_params(str(OUTPUT_DIR / "params_evolution.png"))
    history.plot_l2_errors(str(OUTPUT_DIR / "l2_errors_history.png"))
    
    from src.utils import generate_all_diagnostics
    generate_all_diagnostics(model, physics, data, str(OUTPUT_DIR))

    print(f"\n[OK] Esecuzione terminata. Risultati e plot salvati in: {OUTPUT_DIR}")
