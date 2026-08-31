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
from src.physics import Physics, evaluate_final_losses, compute_l2_errors
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
#os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

torch.set_default_dtype(torch.float32)
torch.set_float32_matmul_precision("high")  # Abilita TF32 per matmul (Ampere+)
torch.backends.cudnn.benchmark = False  # GPU con input size fissi: benchmark seleziona l'algoritmo più veloce
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
EXPORT_TO_OBSIDIAN = False  # True: esporta i log e i plot nel vault Obsidian a fine run
STAGED_TRAINING = True  # True: staged (Fase 1: psi+tau, Fase 2: psi+p)
INVERSE_PROBLEM = True  # True: semi-inverso, False: diretto
DEBUG_MODE = False  # True: stampa info e test avanzati (es. magnitudo PDE)

# --- Boundary Conditions dello Stress sui Rulli (ANCORAGGIO STRESS) ---
# Impostare a False se si desidera rimuovere la BC dello stress sui 4 rulli.
USE_ROLL_STRESS_BC = True
W_ROLL_STRESS = 1.0  # Peso dello stress BC rispetto al velocity BC sui rulli (pesato 1:1 per componente)

# --- Percorsi Base ---
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR.parent / "COMSOL" / "4roll" / "4_roll_mill.csv"

# --- Checkpointing ---
RESUME_CHECKPOINT = BASE_DIR / "checkpoints" / "checkpoint_inverso_fase1_40k+10k.pth"

# --- Parametri Fisici REALI (Ground Truth) ---
MU_S_TRUE = 0.1  # Viscosità solvente [Pa·s]
MU_P_TRUE = 0.9  # Viscosità polimerica [Pa·s]
MU_TOT_TRUE = MU_S_TRUE + MU_P_TRUE  # Viscosità totale [Pa·s] (1.0)
BETA_TRUE = MU_S_TRUE / MU_TOT_TRUE  # Rapporto di viscosità (0.10)
LAM_TRUE = 0.05  # Tempo di rilassamento [s]
EPS_TRUE = 0.0  # Parametro PTT (bloccato a 0)
ALPHA_TRUE = 0.0  # Parametro Giesekus (bloccato a 0)
RHO = 1000.0  # Densità [kg/m³]

# --- Costanti e Calcolo Dinamico dei Guess Iniziali (Log-Space Parametrization) ---
MIN_MU_S = 1e-6
MIN_MU_P = 1e-6
MIN_LAM = 1e-6

# Scala di normalizzazione globale di riferimento (arbitraria, default 2.0 Pa*s)
ETA_0 = 2.0

# Fattore di perturbazione per i parametri del problema inverso (es. 0.80 = 80% del valore reale)
GUESS_FACTOR = 0.80

GUESS_LAM = LAM_TRUE * GUESS_FACTOR                      # 0.05 * 0.80 = 0.0400 s
GUESS_MU_S = MU_S_TRUE * GUESS_FACTOR                    # 0.10 * 0.80 = 0.0800 Pa·s
GUESS_MU_P = MU_P_TRUE * GUESS_FACTOR                    # 0.90 * 0.80 = 0.7200 Pa·s
GUESS_MU_TOT = GUESS_MU_S + GUESS_MU_P                  # 0.8000 Pa·s
GUESS_BETA = GUESS_MU_S / GUESS_MU_TOT                  # 0.1000
GUESS_EPS = 0.0
GUESS_ALPHA = 0.0

# --- Architettura Neural Network ---
HIDDEN_LAYERS = [128] * 8  # 8 hidden layers da 128 neuroni
ACTIVATION = nn.SiLU

# --- Iperparametri di Training a 2 Fasi Disaccoppiate ---
# Fase 1: Cinematica & Reologia (model_psi, model_tau -> lam, mu_p)
ADAM_EPOCHS_PHASE1 = 40000
USE_LBFGS_PHASE1 = True
LBFGS_MAX_ITERS_PHASE1 = 10000

# Fase 2: Idrodinamica & Pressione (model_p, model_psi con mu_s sbloccato dopo warmup)
ADAM_EPOCHS_PHASE2 = 10000
USE_LBFGS_PHASE2 = True
LBFGS_MAX_ITERS_PHASE2 = 500

BASE_LR = 1e-3
ADAM_EPS = 1e-7
PARAM_LR_FACTOR = 0.1
GRAD_CLIP_NORM = 1000.0
PARAM_CLIP_NORM = 1.0

WARMUP_UNLOCK_EPOCH = 0  # 0: parametri attivi fin da epoca 0 in Fase 1; >0: sblocco senza reset Adam
WARMUP_PHASE2_EPOCHS = 5000  # Epoche iniziali Adam Fase 2 con mu_s frozen per pre-formare il campo di pressione

# --- Pesi Funzione di Loss (Architettura Staged Disaccoppiata) ---
# Fase 1: Cinematica & Reologia (model_psi, model_tau -> lam, mu_p)
W_DATA_1 = 1.0          # Peso dati velocita' (u, v) in Fase 1
W_BC_1 = 5.0            # Peso boundary conditions (no-slip + stress rulli) in Fase 1
W_CONSTITUTIVE = 1.0    # Peso equazione costitutiva reologica (Oldroyd-B / PTT / Giesekus)

# Fase 2: Idrodinamica & Pressione (model_p, model_psi -> mu_s)
W_DATA_2 = 20.0         # Bilanciamento quantitativo gradienti su model_psi in Fase 2
W_BC_2 = 5.0            # Peso ancoraggio punto di pressione e boundary in Fase 2
W_MOMENTUM = 1.0        # Peso equazione di conservazione quantita' di moto (Navier-Stokes)

W_DRIFT = 0.0           # Soft anti-drift penalty ausiliaria
VARIANCE_EPS = 1e-4

# ============================================================================
# 3. INIZIALIZZAZIONE OUTPUT
# ============================================================================
layers_str = f"{len(HIDDEN_LAYERS)}x{HIDDEN_LAYERS[0]}"

# Generazione dinamica dei tag di nomenclatura in base ai parametri effettivi
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

config_name = f"[{run_timestamp}][{mode_tag}][{strategy_tag}][{budget_tag}]"

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
    if DEBUG_MODE:
        print("DEBUG REPORT CONFIGURAZIONE INIZIALE:")
        print("  - Formula Weighted MSE: Mean( ((pred - target) ** 2) / var )")
        print("  - Definizione U_ref:    max(sqrt(u_raw**2 + v_raw**2))")
        print("=" * 60)

    # 1. Caricamento Dati
    data = load_data(eta_0=ETA_0)

    # 2. Inizializzazione Modello e Fisica
    model = CombinedModel(p_scale=data["p_scale"], tau_scale=data["tau_scale"]).to(
        DEVICE
    )
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

    # Recap Configurazione
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModello: {total_params:,} parametri totali")
    if INVERSE_PROBLEM:
        print("Modalità: PROBLEMA INVERSO (FASE 1 ONLY - Estensione Cinematica & Reologia)")
        print(f"  - Obiettivo: Raffinamento intensivo dei campi (psi, tau) e parametri (lam, mu_p)")
        print(f"  - Scala di Riferimento: eta_0={physics.eta_0.item():.2f} Pa·s")
        print(f"  - Valori Attuali Caricati: lam={physics.lam.item():.4f} s (true: {LAM_TRUE}), mu_p={physics.mu_p.item():.4f} Pa·s (true: {MU_P_TRUE})")
        print(f"  - Budget di Training: {ADAM_EPOCHS_PHASE1} Adam (FP32) + {LBFGS_MAX_ITERS_PHASE1} L-BFGS (FP64)")
    else:
        print("Modalità: PROBLEMA DIRETTO")

    obsidian_dest_dir = None
    obsidian_run_name = None
    
    if EXPORT_TO_OBSIDIAN:
        from src.utils import init_run_in_obsidian
        config_details = {
            "dataset": DATASET_PATH.name,
            "eta_0": ETA_0,
            "epochs": ADAM_EPOCHS_PHASE1 + ADAM_EPOCHS_PHASE2,
            "inverse_problem": INVERSE_PROBLEM,
            "staged_training": STAGED_TRAINING,
            "activation": ACTIVATION.__name__,
            "network": layers_str,
            "lbfgs_phase1": USE_LBFGS_PHASE1,
            "lbfgs_phase2": USE_LBFGS_PHASE2
        }
        obsidian_dest_dir, obsidian_run_name = init_run_in_obsidian(config_name, config_details)

    # 3. Training
    # Avvia automaticamente TensorBoard monitorando la directory radice degli output
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
        print(f"  {fn:>8s}: {err:.6f}")

    # 5. Generazione Plot
    history.plot_losses(str(OUTPUT_DIR / "loss_history.png"))
    history.plot_params(str(OUTPUT_DIR / "params_evolution.png"))
    history.plot_l2_errors(str(OUTPUT_DIR / "l2_errors_history.png"))
    
    from src.utils import generate_all_diagnostics
    generate_all_diagnostics(model, physics, data, str(OUTPUT_DIR))

    # 6. Test di Validazione Fisica
    if DEBUG_MODE:
        test_random_points(model, physics, data, num_points=10)
        debug_physics_magnitudes(model, physics, data, num_points=2000)


    if EXPORT_TO_OBSIDIAN and obsidian_dest_dir:
        from src.utils import finalize_run_in_obsidian
        
        results_details = {
            "status": "completed"
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
        
    print(f"\n[OK] Esecuzione terminata. Plot salvati in: {OUTPUT_DIR}")
