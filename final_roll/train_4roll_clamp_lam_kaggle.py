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
import matplotlib.tri as mtri
from tqdm import tqdm

# Import dai moduli src
from src.debug import test_random_points, debug_physics_magnitudes
from src.physics import Physics, evaluate_final_losses, compute_l2_errors, inverse_softplus
from src.train import CombinedModel, initialize_last_layer_zero, init_weights_xavier, train
from src.utils import load_data, plot_fields, plot_high_stress_regions, launch_tensorboard_server
from src.utils import get_optimal_chunk_size

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
# 2. FISICA PERSONALIZZATA CON CLAMP SU LAMBDA (lambda >= 0.035)
# ============================================================================
MIN_LAM_CLAMP = 0.035

class PhysicsClampedLam(Physics):
    """
    Variante della classe Physics in cui il valore del tempo di rilassamento lambda
    è vincolato ad essere >= MIN_LAM_CLAMP (es. 0.035).
    Questo impedisce al modello di scivolare verso il minimo locale newtoniano (lambda -> 0)
    quando non vengono fornite condizioni al contorno sullo stress.
    """
    def __init__(self, *args, min_lam_clamp=MIN_LAM_CLAMP, **kwargs):
        self.min_lam_clamp = min_lam_clamp
        super().__init__(*args, **kwargs)

    @property
    def lam(self):
        softplus_lam = torch.nn.functional.softplus(self._raw_lam) + 1e-8
        return torch.clamp(softplus_lam, min=self.min_lam_clamp)

    def clamp_params(self):
        super().clamp_params()
        min_raw = inverse_softplus(self.min_lam_clamp)
        with torch.no_grad():
            if self._raw_lam < min_raw:
                self._raw_lam.copy_(min_raw)


# ============================================================================
# 3. COSTANTI E CONFIGURAZIONI GLOBALI
# ============================================================================

# --- Opzioni di Controllo ---
EXPORT_TO_OBSIDIAN = True
STAGED_TRAINING = True
INVERSE_PROBLEM = True
DEBUG_MODE = False

# --- Boundary Conditions dello Stress (DISATTIVATE per questo esperimento) ---
USE_ROLL_STRESS_BC = False
W_ROLL_STRESS = 0.0

# --- Percorsi Base ---
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR.parent / "COMSOL" / "4roll" / "4_roll_mill.csv"

# --- Checkpointing ---
RESUME_CHECKPOINT = None

# --- Parametri Fisici REALI (Ground Truth) ---
MU_S_TRUE = 0.1
MU_P_TRUE = 0.9
BETA_TRUE = MU_S_TRUE / (MU_S_TRUE + MU_P_TRUE)  # 0.10
LAM_TRUE = 0.05
EPS_TRUE = 0.0
ALPHA_TRUE = 0.0
RHO = 1000.0

# --- Costanti e Guess Iniziali ---
MIN_MU_S = 1e-6
MIN_MU_P = 1e-6
MIN_LAM = 1e-6

GUESS_MULTIPLIER = 0.8
GUESS_MU_S = MU_S_TRUE * GUESS_MULTIPLIER
GUESS_MU_P = MU_P_TRUE * GUESS_MULTIPLIER
GUESS_BETA = 0.05
GUESS_LAM = LAM_TRUE * GUESS_MULTIPLIER  # 0.04
GUESS_EPS = 0.05
GUESS_ALPHA = 0.05

# --- Architettura Neural Network ---
HIDDEN_LAYERS = [128] * 8
ACTIVATION = nn.SiLU

# --- Iperparametri di Training ---
ADAM_EPOCHS_PHASE1 = 30000
ADAM_EPOCHS_PHASE2 = 0
USE_LBFGS_PHASE1 = True
USE_LBFGS_PHASE2 = False
LBFGS_MAX_ITERS_PHASE1 = 10000
LBFGS_MAX_ITERS_PHASE2 = 0
BASE_LR = 1e-3
ADAM_EPS = 1e-7
PARAM_LR_FACTOR = 0.1
GRAD_CLIP_NORM = 1000.0
PARAM_CLIP_NORM = 1.0

WARMUP_UNLOCK_EPOCH = 0

# --- Pesi Funzione di Loss ---
W_BC = 5.0
W_PHYSICS = 3.0
W_DATA = 1.0
W_MOMENTUM = 1.0
W_CONSTITUTIVE = 1.0
VARIANCE_EPS = 1e-4

# ============================================================================
# 4. INIZIALIZZAZIONE OUTPUT
# ============================================================================
layers_str = f"{len(HIDDEN_LAYERS)}x{HIDDEN_LAYERS[0]}"
config_name = f"clamp_lam_{MIN_LAM_CLAMP}_noStressBC_{DATASET_PATH.stem}_L{layers_str}_E{ADAM_EPOCHS_PHASE1+ADAM_EPOCHS_PHASE2}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

if RESUME_CHECKPOINT is not None:
    OUTPUT_DIR = Path(RESUME_CHECKPOINT).parent
else:
    OUTPUT_DIR = BASE_DIR / "output_4rollmill" / config_name
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

global_log_path = OUTPUT_DIR / "train_log.txt"

# Iniezione dinamica dei parametri globali nei moduli di src
for module in [src.debug, src.physics, src.train, src.utils]:
    for name, val in list(globals().items()):
        if name.isupper():
            module.__dict__[name] = val

if __name__ == "__main__":
    print(f"Device: {DEVICE} | Dtype: {torch.get_default_dtype()}")
    print(f"Dataset: {DATASET_PATH}\n")
    print("=" * 60)
    print(f"ESPERIMENTO: NO STRESS BC + CLAMP LAMBDA >= {MIN_LAM_CLAMP}")
    print(f"  - Stress BC rulli: {USE_ROLL_STRESS_BC}")
    print(f"  - Lambda Clamp min: {MIN_LAM_CLAMP} (valore reale target: {LAM_TRUE})")
    print("=" * 60)

    # 1. Caricamento Dati
    data = load_data()

    # 2. Inizializzazione Modello e Fisica Clamped
    model = CombinedModel(p_scale=data["p_scale"], tau_scale=data["tau_scale"]).to(
        DEVICE
    )
    for submodel in [model.model_psi, model.model_p, model.model_tau]:
        submodel.apply(lambda m: init_weights_xavier(m, activation_name=ACTIVATION))

    initialize_last_layer_zero(model.model_p)
    initialize_last_layer_zero(model.model_tau)

    physics = PhysicsClampedLam(
        U_ref=data["U_ref"],
        H_ref=data["H"],
        H_coord=data["H_coord"],
        var_weights=data["var_weights"],
        inverse_mode=INVERSE_PROBLEM,
        tau_scale=data["tau_scale"],
        p_scale=data["p_scale"],
        use_roll_stress_bc=USE_ROLL_STRESS_BC,
        w_roll_stress=W_ROLL_STRESS,
        min_lam_clamp=MIN_LAM_CLAMP,
    ).to(DEVICE)

    # Recap Configurazione
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModello: {total_params:,} parametri totali")
    print("Modalità: PROBLEMA INVERSO CON CLAMP SU LAMBDA E NO STRESS BC")
    print(f" Guess lam: {GUESS_LAM} (true: {LAM_TRUE}, min clamp: {MIN_LAM_CLAMP})")

    obsidian_dest_dir = None
    obsidian_run_name = None

    if EXPORT_TO_OBSIDIAN and not RESUME_CHECKPOINT:
        from src.utils import init_run_in_obsidian
        config_details = {
            "dataset": DATASET_PATH.name,
            "epochs": ADAM_EPOCHS_PHASE1 + ADAM_EPOCHS_PHASE2,
            "inverse_problem": INVERSE_PROBLEM,
            "staged_training": STAGED_TRAINING,
            "use_roll_stress_bc": USE_ROLL_STRESS_BC,
            "min_lam_clamp": MIN_LAM_CLAMP,
            "activation": ACTIVATION.__name__,
            "network": layers_str,
            "lbfgs_phase1": USE_LBFGS_PHASE1,
            "lbfgs_phase2": USE_LBFGS_PHASE2
        }
        obsidian_dest_dir, obsidian_run_name = init_run_in_obsidian(config_name, config_details)

    # 3. Training
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
        tb_writer=tb_writer
    )

    tb_writer.close()

    # 4. Report Risultati Finali
    params = physics.log_params()
    print(f"\n{'=' * 60}\nRISULTATI FINALI PARAMETRI FISICI\n{'=' * 60}")
    for p_name, true_val in zip(
        ["beta", "mu_s", "mu_p", "lam", "eps", "alpha"],
        [BETA_TRUE, MU_S_TRUE, MU_P_TRUE, LAM_TRUE, EPS_TRUE, ALPHA_TRUE],
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

    # 5. Generazione Plot
    history.plot_losses(str(OUTPUT_DIR / "loss_history.png"))
    history.plot_params(str(OUTPUT_DIR / "params_evolution.png"))
    history.plot_l2_errors(str(OUTPUT_DIR / "l2_errors_history.png"))

    from src.utils import generate_all_diagnostics
    generate_all_diagnostics(model, physics, data, str(OUTPUT_DIR))

    if EXPORT_TO_OBSIDIAN and obsidian_dest_dir:
        from src.utils import finalize_run_in_obsidian

        results_details = {
            "status": "completed",
            "min_lam_clamp": MIN_LAM_CLAMP,
            "use_roll_stress_bc": USE_ROLL_STRESS_BC,
        }
        for p_name in ["beta", "mu_s", "mu_p", "lam", "eps", "alpha"]:
            if p_name in params:
                results_details[f"Param {p_name}"] = f"{params[p_name]:.6f}"

        for k, v in final_losses.items():
            results_details[f"Loss {k}"] = f"{v:.6e}"

        for fn, err in errors.items():
            results_details[f"Error {fn}"] = f"{err:.6f}"

        finalize_run_in_obsidian(
            dest_dir=obsidian_dest_dir,
            source_dir=str(OUTPUT_DIR),
            run_folder_name=obsidian_run_name,
            results_details=results_details
        )

    print(f"\n[OK] Esecuzione esperimento terminata. Plot e risultati salvati in: {OUTPUT_DIR}")
