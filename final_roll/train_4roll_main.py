import os
import shutil
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
from src.utils import (
    load_data,
    plot_fields,
    plot_high_stress_regions,
    launch_tensorboard_server,
    get_optimal_chunk_size,
    build_dataset_tag,
    resolve_dataset_path,
    parse_dataset_metadata,
    update_inverse_benchmarks_database,
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
#os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

torch.set_default_dtype(torch.float32)
# [Proposta A] Disabilita TF32 per garantire la piena precisione FP32 standard IEEE (23-bit mantissa)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
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

# --- Parametri Fisici REALI (Ground Truth) - Setup Giesekus Mesh Study 12k (da zero) ---
MU_S_TRUE = 0.5  # Viscosità solvente [Pa·s]
MU_P_TRUE = 0.5  # Viscosità polimerica [Pa·s]
MU_TOT_TRUE = MU_S_TRUE + MU_P_TRUE  # Viscosità totale [Pa·s] (1.0)
BETA_TRUE = MU_S_TRUE / MU_TOT_TRUE  # Rapporto di viscosità (0.50)
LAM_TRUE = 0.7  # Tempo di rilassamento [s]
EPS_TRUE = 0.0  # Parametro PTT (bloccato a 0)
ALPHA_TRUE = 0.35  # Parametro Giesekus (bloccato a 0.35)
RHO = 1000.0  # Densità [kg/m³]

# Risoluzione mesh e parametri fisici da CLI (default '12k', sovrascrivibile es. --mesh 5k --alpha 0.1 o --dataset <nome>)
import sys
MESH_TAG = "12k"
WARMUP_UNLOCK_EPOCH = 0
TRANSFER_CKPT_PATH = None
ADAM_EPOCHS_OVERRIDE = None
LBFGS_ITERS_OVERRIDE = None

for i, arg in enumerate(sys.argv):
    if arg == "--mesh" and i + 1 < len(sys.argv):
        MESH_TAG = sys.argv[i + 1]
    elif arg.startswith("--mesh="):
        MESH_TAG = arg.split("=")[1]
    elif arg == "--alpha" and i + 1 < len(sys.argv):
        ALPHA_TRUE = float(sys.argv[i + 1])
    elif arg.startswith("--alpha="):
        ALPHA_TRUE = float(arg.split("=")[1])
    elif arg == "--eps" and i + 1 < len(sys.argv):
        EPS_TRUE = float(sys.argv[i + 1])
    elif arg.startswith("--eps="):
        EPS_TRUE = float(arg.split("=")[1])
    elif arg == "--warmup" and i + 1 < len(sys.argv):
        WARMUP_UNLOCK_EPOCH = int(sys.argv[i + 1])
    elif arg.startswith("--warmup="):
        WARMUP_UNLOCK_EPOCH = int(arg.split("=")[1])
    elif arg == "--transfer-ckpt" and i + 1 < len(sys.argv):
        TRANSFER_CKPT_PATH = sys.argv[i + 1]
    elif arg.startswith("--transfer-ckpt="):
        TRANSFER_CKPT_PATH = arg.split("=")[1]
    elif arg == "--transfer" and i + 1 < len(sys.argv):
        TRANSFER_CKPT_PATH = sys.argv[i + 1]
    elif arg.startswith("--transfer="):
        TRANSFER_CKPT_PATH = arg.split("=")[1]
    elif arg == "--adam1" and i + 1 < len(sys.argv):
        ADAM_EPOCHS_OVERRIDE = int(sys.argv[i + 1])
    elif arg.startswith("--adam1="):
        ADAM_EPOCHS_OVERRIDE = int(arg.split("=")[1])
    elif arg == "--lbfgs1" and i + 1 < len(sys.argv):
        LBFGS_ITERS_OVERRIDE = int(sys.argv[i + 1])
    elif arg.startswith("--lbfgs1="):
        LBFGS_ITERS_OVERRIDE = int(arg.split("=")[1])
    elif arg == "--dataset" and i + 1 < len(sys.argv):
        _meta = parse_dataset_metadata(sys.argv[i + 1])
        LAM_TRUE = _meta["lam_true"]
        MU_P_TRUE = _meta["mu_p_true"]
        MU_S_TRUE = _meta["mu_s_true"]
        ALPHA_TRUE = _meta["alpha_true"]
        EPS_TRUE = _meta["eps_true"]
        MESH_TAG = _meta["mesh_tag"]
    elif arg.startswith("--dataset="):
        _meta = parse_dataset_metadata(arg.split("=")[1])
        LAM_TRUE = _meta["lam_true"]
        MU_P_TRUE = _meta["mu_p_true"]
        MU_S_TRUE = _meta["mu_s_true"]
        ALPHA_TRUE = _meta["alpha_true"]
        EPS_TRUE = _meta["eps_true"]
        MESH_TAG = _meta["mesh_tag"]

# Ricalcolo grandezze derivate dopo eventuale parsing CLI
MU_TOT_TRUE = MU_S_TRUE + MU_P_TRUE
BETA_TRUE = MU_S_TRUE / MU_TOT_TRUE

# Tag identificativo standard della configurazione reologica (L-P-S-A-E_M)
PARAM_TAG = build_dataset_tag(LAM_TRUE, MU_P_TRUE, MU_S_TRUE, ALPHA_TRUE, EPS_TRUE, MESH_TAG)

# --- Percorsi Base & Dataset (Strict: errore bloccante se non trovato) ---
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = resolve_dataset_path(BASE_DIR.parent / "COMSOL" / "4roll" / "Datasets", PARAM_TAG)

# --- Checkpointing (Disattivato per training ex-novo da zero) ---
RESUME_CHECKPOINT = None

# --- Costanti e Calcolo Dinamico dei Guess Iniziali (Log-Space Parametrization) ---
MIN_MU_S = 1e-6
MIN_MU_P = 1e-6
MIN_LAM = 1e-6

# Scala di normalizzazione globale di riferimento (arbitraria, default 2.0 Pa*s)
ETA_0 = 2.0

# Fattore di perturbazione per i parametri del problema inverso (es. 0.80 = 80% del valore reale)
GUESS_FACTOR = 0.80

GUESS_LAM = LAM_TRUE * GUESS_FACTOR                      # 0.10 * 0.80 = 0.0800 s
GUESS_MU_S = MU_S_TRUE * GUESS_FACTOR                    # 0.50 * 0.80 = 0.4000 Pa·s
GUESS_MU_P = MU_P_TRUE * GUESS_FACTOR                    # 0.50 * 0.80 = 0.4000 Pa·s
GUESS_MU_TOT = GUESS_MU_S + GUESS_MU_P                  # 0.8000 Pa·s
GUESS_BETA = GUESS_MU_S / GUESS_MU_TOT                  # 0.5000
GUESS_EPS = 0.25
GUESS_ALPHA = 0.25
TRAIN_ALPHA = True
TRAIN_EPS = True

# --- Architettura Neural Network ---
HIDDEN_LAYERS = [128] * 8  # 8 hidden layers da 128 neuroni
ACTIVATION = nn.SiLU

# --- Iperparametri di Training a 2 Fasi Disaccoppiate ---
# Fase 1: Cinematica & Reologia (model_psi, model_tau -> lam, mu_p, alpha, eps)
ADAM_EPOCHS_PHASE1 = 40000
USE_LBFGS_PHASE1 = True
LBFGS_MAX_ITERS_PHASE1 = 10000

# Se specificato override o transfer learning default (20k + 5k)
if ADAM_EPOCHS_OVERRIDE is not None:
    ADAM_EPOCHS_PHASE1 = ADAM_EPOCHS_OVERRIDE
elif TRANSFER_CKPT_PATH is not None:
    ADAM_EPOCHS_PHASE1 = 20000

if LBFGS_ITERS_OVERRIDE is not None:
    LBFGS_MAX_ITERS_PHASE1 = LBFGS_ITERS_OVERRIDE
elif TRANSFER_CKPT_PATH is not None:
    LBFGS_MAX_ITERS_PHASE1 = 5000

# Fase 2: Disattivata (solo Fase 1 ex-novo)
ADAM_EPOCHS_PHASE2 = 0
USE_LBFGS_PHASE2 = False
LBFGS_MAX_ITERS_PHASE2 = 0

BASE_LR = 2.5e-3
ETA_MIN = 2.5e-6
ADAM_EPS = 1e-8
# [Proposta B] Epsilon differenziato per non soffocare gradienti fisici piccoli
ADAM_EPS_PHYS = 1e-15
PARAM_LR_FACTOR = 1.0    # LR parametri fisici = BASE_LR * 1.0 = 2.5e-3 (stesso range cosine annealing da 2.5e-3 a 2.5e-6)
# [Proposta H & Run 23] Gradient clipping rigido a 5.0 per prevenire salti numerici
GRAD_CLIP_NORM = 5.0
PARAM_CLIP_NORM = 1.0

WARMUP_UNLOCK_EPOCH = globals().get("WARMUP_UNLOCK_EPOCH", 0)  # da CLI (--warmup N) o 0 default (parametri attivi da subito)
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

def _format_iters(n):
    if n == 0:
        return "0"
    if n % 1000 == 0:
        return f"{n // 1000}k"
    return f"{n / 1000:.1f}k"

if ADAM_EPOCHS_PHASE2 > 0 or USE_LBFGS_PHASE2:
    budget_tag = f"Ph2_{_format_iters(ADAM_EPOCHS_PHASE2)}+{_format_iters(LBFGS_MAX_ITERS_PHASE2)}_Warmup{_format_iters(WARMUP_PHASE2_EPOCHS)}"
else:
    tl_prefix = "TL_" if TRANSFER_CKPT_PATH is not None else ""
    budget_tag = f"{tl_prefix}Ph1_{_format_iters(ADAM_EPOCHS_PHASE1)}+{_format_iters(LBFGS_MAX_ITERS_PHASE1)}"
    if WARMUP_UNLOCK_EPOCH > 0:
        budget_tag += f"_Warmup{_format_iters(WARMUP_UNLOCK_EPOCH)}"

if RESUME_CHECKPOINT is not None and RESUME_CHECKPOINT.exists():
    OUTPUT_DIR = RESUME_CHECKPOINT.parent
    config_name = OUTPUT_DIR.name
else:
    run_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    config_name = f"[{run_timestamp}][{mode_tag}][{PARAM_TAG}][{budget_tag}]"
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

    # 2b. Caricamento Pesi da Transfer Learning (se specificato)
    if TRANSFER_CKPT_PATH is not None:
        transfer_p = Path(TRANSFER_CKPT_PATH)
        if not transfer_p.is_absolute():
            transfer_p = BASE_DIR / transfer_p
        if not transfer_p.exists():
            ckpt_matches = list((BASE_DIR / "checkpoints").rglob(transfer_p.name))
            if ckpt_matches:
                transfer_p = ckpt_matches[0]
        if not transfer_p.exists():
            raise FileNotFoundError(f"[Transfer Learning] Checkpoint donatore non trovato: {TRANSFER_CKPT_PATH}")

        print(f"\n[Transfer Learning] Caricamento pesi pre-addestrati da:\n  {transfer_p}")
        source_chk = torch.load(str(transfer_p), map_location=DEVICE)
        model_dict = model.state_dict()
        pretrained_dict = {
            k: v for k, v in source_chk['model_state_dict'].items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        model.load_state_dict(pretrained_dict, strict=False)
        print(f"[Transfer Learning] Caricati con successo {len(pretrained_dict)}/{len(model_dict)} tensori di pesi per la rete.")
        print("[Transfer Learning] Parametri fisici RESETTATI rigorosamente ai valori di Guess target:")
        print(f"  - lambda_guess: {physics.lam.item():.4f} s (Target reale: {LAM_TRUE:.4f} s, Guess: {GUESS_LAM:.4f} s)")
        print(f"  - mu_p_guess:   {physics.mu_p.item():.4f} Pa·s (Target reale: {MU_P_TRUE:.4f} Pa·s, Guess: {GUESS_MU_P:.4f} Pa·s)")
        print(f"  - alpha_guess:  {physics.alpha.item():.4f}   (Target reale: {ALPHA_TRUE:.4f}, Guess: {GUESS_ALPHA:.4f})")
        print(f"  - eps_guess:    {physics.eps.item():.4f}   (Target reale: {EPS_TRUE:.4f}, Guess: {GUESS_EPS:.4f})")
        print("  - L'ottimizzatore Adam partirà da epoca 0 su questi pesi pre-addestrati.")

    # Recap Configurazione
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModello: {total_params:,} parametri totali")
    if INVERSE_PROBLEM:
        mode_desc = "TRANSFER LEARNING FASE 1" if TRANSFER_CKPT_PATH is not None else "FASE 1 ONLY"
        print(f"Modalità: PROBLEMA INVERSO ({mode_desc} - Estensione Cinematica & Reologia)")
        print(f"  - Obiettivo: Raffinamento intensivo dei campi (psi, tau) e parametri (lam, mu_p, alpha, eps)")
        print(f"  - Scala di Riferimento: eta_0={physics.eta_0.item():.2f} Pa·s")
        print(f"  - Valori Attuali Caricati: lam={physics.lam.item():.4f} s (true: {LAM_TRUE}), mu_p={physics.mu_p.item():.4f} Pa·s (true: {MU_P_TRUE}), alpha={physics.alpha.item():.4f} (true: {ALPHA_TRUE}), eps={physics.eps.item():.4f} (true: {EPS_TRUE})")
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

    # 7. Summary Benchmark Tabellare & Salvataggio JSON (Facile consultazione)
    import json
    err_lam_pct = 100.0 * (params['lam'] - LAM_TRUE) / (LAM_TRUE + 1e-12)
    err_mup_pct = 100.0 * (params['mu_p'] - MU_P_TRUE) / (MU_P_TRUE + 1e-12)
    err_alpha_pct = 100.0 * (params['alpha'] - ALPHA_TRUE) / (ALPHA_TRUE + 1e-12) if ALPHA_TRUE > 0.0 else None
    err_eps_pct = 100.0 * (params['eps'] - EPS_TRUE) / (EPS_TRUE + 1e-12) if EPS_TRUE > 0.0 else None
    avg_l2_uv = 0.5 * (errors.get('u', 0.0) + errors.get('v', 0.0))
    avg_l2_diag = 0.5 * (errors.get('tau_xx', 0.0) + errors.get('tau_yy', 0.0))
    
    summary_metrics = {
        "mesh_tag": MESH_TAG,
        "param_tag": PARAM_TAG,
        "n_points": int(len(data.get("coords", data.get("points", [])))),
        "lambda_estimated": float(params['lam']),
        "lambda_true": float(LAM_TRUE),
        "lambda_error_pct": float(err_lam_pct),
        "mu_p_estimated": float(params['mu_p']),
        "mu_p_true": float(MU_P_TRUE),
        "mu_p_error_pct": float(err_mup_pct),
        "alpha_estimated": float(params['alpha']),
        "alpha_true": float(ALPHA_TRUE),
        "alpha_error_pct": float(err_alpha_pct) if err_alpha_pct is not None else None,
        "eps_estimated": float(params['eps']),
        "eps_true": float(EPS_TRUE),
        "eps_error_pct": float(err_eps_pct) if err_eps_pct is not None else None,
        "l2_u": float(errors.get('u', 0.0)),
        "l2_v": float(errors.get('v', 0.0)),
        "l2_uv_mean": float(avg_l2_uv),
        "l2_tau_xx": float(errors.get('tau_xx', 0.0)),
        "l2_tau_xy": float(errors.get('tau_xy', 0.0)),
        "l2_tau_yy": float(errors.get('tau_yy', 0.0)),
        "l2_tau_diag_mean": float(avg_l2_diag),
        "final_loss_total": float(final_losses.get('total_loss', 0.0)),
        "final_loss_data": float(final_losses.get('data_loss', 0.0)),
        "final_loss_bc": float(final_losses.get('bc_loss', 0.0)),
        "final_loss_pde": float(final_losses.get('pde_loss', 0.0))
    }
    
    summary_json_path = OUTPUT_DIR / "metrics_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f_json:
        json.dump(summary_metrics, f_json, indent=2)

    # Aggiorna automaticamente i registri centralizzati inverse_runs.csv e inverse_runs.md
    update_inverse_benchmarks_database()

    print(f"\n{'=' * 75}")
    print(f"  >>> BENCHMARK METRICS SUMMARY [{MESH_TAG}] <<<")
    print(f"{'=' * 75}")
    print(f"  Mesh:                  {MESH_TAG} ({len(data.get('coords', data.get('points', []))):,} nodi)")
    print(f"  Final Loss Totale:     {final_losses.get('total_loss', 0.0):.6e}")
    print(f"  Errore L2 (u, v):      {avg_l2_uv*100:.4f}%  (u: {errors.get('u', 0.0)*100:.4f}%, v: {errors.get('v', 0.0)*100:.4f}%)")
    print(f"  Errore L2 tau_xy:      {errors.get('tau_xy', 0.0)*100:.4f}%")
    print(f"  Errore L2 tau_diag:    {avg_l2_diag*100:.4f}%  (xx: {errors.get('tau_xx', 0.0)*100:.4f}%, yy: {errors.get('tau_yy', 0.0)*100:.4f}%)")
    print(f"  ---------------------------------------------------------------------------")
    print(f"  Parametro lambda:      {params['lam']:.6f} s   [True: {LAM_TRUE:.4f} s | Err: {err_lam_pct:+.2f}%]")
    print(f"  Parametro mu_p:        {params['mu_p']:.6f} Pa·s [True: {MU_P_TRUE:.4f} Pa·s | Err: {err_mup_pct:+.2f}%]")
    if err_alpha_pct is not None:
        print(f"  Parametro alpha:       {params['alpha']:.6f}     [True: {ALPHA_TRUE:.4f} | Err: {err_alpha_pct:+.2f}% | Guess: {GUESS_ALPHA:.2f}]")
    else:
        print(f"  Parametro alpha:       {params['alpha']:.6f}     [True: {ALPHA_TRUE:.4f} | Guess: {GUESS_ALPHA:.2f}]")
    if err_eps_pct is not None:
        print(f"  Parametro eps:         {params['eps']:.6f}     [True: {EPS_TRUE:.4f} | Err: {err_eps_pct:+.2f}% | Guess: {GUESS_EPS:.2f}]")
    else:
        print(f"  Parametro eps:         {params['eps']:.6f}     [True: {EPS_TRUE:.4f} | Guess: {GUESS_EPS:.2f}]")
    print(f"{'=' * 75}")
    print(f"  [JSON Salvato]: {summary_json_path}")
    print(f"{'=' * 75}\n")
    
    # 8. Archiviazione con Quality Gate per Checkpoint Donatori (Fase 1)
    f1_ckpt_in_run = OUTPUT_DIR / "checkpoint_lbfgs_phase1.pth"
    if f1_ckpt_in_run.exists():
        f1_tag = budget_tag
        if ALPHA_TRUE > 0:
            subfolder = "giesekus"
        elif EPS_TRUE > 0:
            subfolder = "ptt"
        else:
            subfolder = "oldroyd"
            
        MAX_ERR_LAM = 10.5      # Max ~10% errore relativo su lambda
        MAX_ERR_MUP = 10.5      # Max ~10% errore relativo su mu_p
        MAX_ERR_TAU_XY = 8.0    # Max 8% errore L2 su tau_xy
        MAX_ERR_UV = 5.0        # Max 5% errore L2 su (u, v) (rilassato)
        
        err_tau_xy_pct = errors.get('tau_xy', 0.0) * 100.0
        err_uv_pct = avg_l2_uv * 100.0
        
        is_quality_ok = (
            abs(err_lam_pct) <= MAX_ERR_LAM and
            abs(err_mup_pct) <= MAX_ERR_MUP and
            err_tau_xy_pct <= MAX_ERR_TAU_XY and
            err_uv_pct <= MAX_ERR_UV
        )
        
        if is_quality_ok:
            dest_dir = BASE_DIR / "checkpoints" / subfolder
            dest_dir.mkdir(parents=True, exist_ok=True)
            f1_dest = dest_dir / f"checkpoint_inverso_fase1_{PARAM_TAG}_{f1_tag}.pth"
            shutil.copy2(f1_ckpt_in_run, f1_dest)
            print(f"[Quality Gate: SUPERATO] Checkpoint donatore archiviato in:\n  -> {f1_dest}")
        else:
            print(f"[Quality Gate: NON SUPERATO] Checkpoint NON archiviato in checkpoints/{subfolder}:")
            print(f"  Soglie: |Err lam| <= {MAX_ERR_LAM}%, |Err mu_p| <= {MAX_ERR_MUP}%, Err tau_xy <= {MAX_ERR_TAU_XY}%, Err uv <= {MAX_ERR_UV}%")
            print(f"  Valori: |Err lam| = {abs(err_lam_pct):.2f}%, |Err mu_p| = {abs(err_mup_pct):.2f}%, Err tau_xy = {err_tau_xy_pct:.2f}%, Err uv = {err_uv_pct:.2f}%")
            print(f"  (Il checkpoint rimane disponibile nella cartella di run: {OUTPUT_DIR})")
        
    print(f"\n[OK] Esecuzione terminata. Plot salvati in: {OUTPUT_DIR}")
