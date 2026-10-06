"""
train_direct_kaggle.py
=============================================================================
Script PINN 4-Roll Mill per il Problema Diretto (CFD Forward Simulation).
Progettato per esecuzione su Kaggle GPU e postazioni locali.

Caratteristiche del Setup Diretto:
  - Zero dati interni di velocità (W_DATA = 0.0): la rete non riceve mai i campi
    u e v interni da COMSOL; i punti della mesh fungono puramente da punti di
    collocazione per i residui delle PDE.
  - Zero dati di stress al contorno (USE_ROLL_STRESS_BC = False): non viene fornito
    alcun dato di sforzo sui rulli o pareti (equivalente al setup FEM di COMSOL).
  - Boundary Conditions pesate (W_BC = 50.0): peso dominante sulle pareti no-slip
    e sulla rotazione imposta dai 4 rulli cilindrici per contrastare il minimo banale.
  - Ancoraggio Hard Algebrico della Pressione: model_p garantisce per costruzione
    algebrica p(x_0) == p_ref senza soft-loss.
  - Simulazione Accoppiata: Navier-Stokes (W_MOMENTUM = 1.0) e Legge Costitutiva
    (W_CONSTITUTIVE = 1.0) entrambe attive contemporaneamente.
  - Parametri Fisici Fissi: lambda, mu_p, mu_s, alpha, eps congelati ai valori reali.
  - Dataset Default: mesh 5K (4_roll_mill_L0.1-P0.5-S0.5-A0-E0-M5k.csv).
  - Pipeline di Ottimizzazione: Adam (FP32) + L-BFGS (FP64).
  - Diagnostica Completa: metriche L2 relative rispetto a COMSOL, contour plots 2D,
    TensorBoard e log standardizzati.
=============================================================================
"""

import os
import sys
import builtins
import shutil
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

# --- Logging automatico di tutti i print (Globale) ---
_original_print = builtins.print
global_log_path = None


def custom_print(*args, **kwargs):
    _original_print(*args, **kwargs)
    if global_log_path is not None:
        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")
        text = sep.join(map(str, args)) + end
        with open(global_log_path, "a", encoding="utf-8") as f:
            f.write(text)


builtins.print = custom_print

# ============================================================================
# 1. SETUP AMBIENTE, PYTORCH E DISPOSITIVO
# ============================================================================
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

torch.set_default_dtype(torch.float32)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False

SEED = 123
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
builtins.DEVICE = DEVICE

# ============================================================================
# 2. CONFIGURAZIONE FISICA, PESI E IPERPARAMETRI
# ============================================================================
# Modalità Diretta
INVERSE_PROBLEM = False
STAGED_TRAINING = False  # Simulazione accoppiata monolitica (Momentum + Costitutiva attive)
DEBUG_MODE = False

# Boundary Conditions
USE_ROLL_STRESS_BC = False  # Nessuna condizione al contorno di sforzo (puro setup CFD)
W_ROLL_STRESS = 0.0

# Pesi Funzione di Loss per il Problema Diretto
W_DATA = 0.0          # ZERO dati interni di velocità
W_DATA_1 = 0.0
W_BC = 50.0           # Peso dominante sulle boundary conditions di velocità
W_BC_1 = 50.0
W_MOMENTUM = 1.0      # Equazione di Navier-Stokes / Stokes
W_CONSTITUTIVE = 1.0  # Equazione costitutiva reologica (Oldroyd-B / Giesekus / PTT)
W_DRIFT = 0.0

# Architettura Neurale
HIDDEN_LAYERS = [128] * 8
ACTIVATION = nn.SiLU

# Iperparametri Ottimizzatore
ADAM_EPOCHS_PHASE1 = 40000
USE_LBFGS_PHASE1 = True
LBFGS_MAX_ITERS_PHASE1 = 10000

ADAM_EPOCHS_PHASE2 = 0
USE_LBFGS_PHASE2 = False
LBFGS_MAX_ITERS_PHASE2 = 0

BASE_LR = 2.5e-3
ETA_MIN = 2.5e-6
ADAM_EPS = 1e-8
ADAM_EPS_PHYS = 1e-15
PARAM_LR_FACTOR = 1.0
GRAD_CLIP_NORM = 5.0
PARAM_CLIP_NORM = 1.0

# Densità e Scala di Riferimento
RHO = 1000.0
ETA_0 = 1.0  # Scala globale eta_0 (1.0 Pa*s di default per normalizzazione non-dimensionale)

# Parametri Fisici Default (Oldroyd-B baseline M5k)
LAM_TRUE = 0.1
MU_P_TRUE = 0.5
MU_S_TRUE = 0.5
ALPHA_TRUE = 0.0
EPS_TRUE = 0.0
MESH_TAG = "5k"

# ============================================================================
# 3. PARSING RIGA DI COMANDO & OPZIONI DI RUN
# ============================================================================
SMOKE_TEST = False
CUSTOM_DATASET_ARG = None

i = 1
while i < len(sys.argv):
    arg = sys.argv[i]
    if arg == "--dataset" and i + 1 < len(sys.argv):
        CUSTOM_DATASET_ARG = sys.argv[i + 1]
        i += 2
    elif arg.startswith("--dataset="):
        CUSTOM_DATASET_ARG = arg.split("=")[1]
        i += 1
    elif arg == "--mesh" and i + 1 < len(sys.argv):
        MESH_TAG = sys.argv[i + 1]
        i += 2
    elif arg.startswith("--mesh="):
        MESH_TAG = arg.split("=")[1]
        i += 1
    elif arg == "--w-bc" and i + 1 < len(sys.argv):
        W_BC = float(sys.argv[i + 1])
        W_BC_1 = W_BC
        i += 2
    elif arg.startswith("--w-bc="):
        W_BC = float(arg.split("=")[1])
        W_BC_1 = W_BC
        i += 1
    elif arg == "--adam" and i + 1 < len(sys.argv):
        ADAM_EPOCHS_PHASE1 = int(sys.argv[i + 1])
        i += 2
    elif arg.startswith("--adam="):
        ADAM_EPOCHS_PHASE1 = int(arg.split("=")[1])
        i += 1
    elif arg == "--lbfgs" and i + 1 < len(sys.argv):
        LBFGS_MAX_ITERS_PHASE1 = int(sys.argv[i + 1])
        i += 2
    elif arg.startswith("--lbfgs="):
        LBFGS_MAX_ITERS_PHASE1 = int(arg.split("=")[1])
        i += 1
    elif arg == "--no-lbfgs":
        USE_LBFGS_PHASE1 = False
        LBFGS_MAX_ITERS_PHASE1 = 0
        i += 1
    elif arg == "--smoke-test":
        SMOKE_TEST = True
        ADAM_EPOCHS_PHASE1 = 5
        LBFGS_MAX_ITERS_PHASE1 = 2
        i += 1
    else:
        i += 1

# ============================================================================
# 4. RISOLUZIONE DEI PERCORSI (KAGGLE & LOCALE)
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent

# Aggiungi final_roll al sys.path se necessario
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(BASE_DIR.parent) not in sys.path:
    sys.path.insert(0, str(BASE_DIR.parent))

from src.debug import test_random_points, debug_physics_magnitudes
from src.physics import Physics, evaluate_final_losses, compute_l2_errors
from src.train import CombinedModel, initialize_last_layer_zero, init_weights_xavier, train
from src.utils import (
    load_data,
    generate_all_diagnostics,
    build_dataset_tag,
    resolve_dataset_path,
    parse_dataset_metadata,
)
import src.debug
import src.physics
import src.train
import src.utils

# Risoluzione percorso dataset
def find_dataset_file(param_tag, custom_arg=None):
    if custom_arg is not None:
        p = Path(custom_arg)
        if p.is_file():
            return p.resolve()
        # Cerca nelle cartelle note
        search_dirs = [
            BASE_DIR.parent / "COMSOL" / "4roll" / "Datasets",
            BASE_DIR.parent / "COMSOL" / "4roll",
            Path("/kaggle/input"),
            Path("/kaggle/working"),
            BASE_DIR,
        ]
        for sdir in search_dirs:
            if sdir.is_dir():
                matches = list(sdir.rglob(f"*{p.name}*"))
                if matches:
                    return matches[0].resolve()
        raise FileNotFoundError(f"Dataset custom non trovato: {custom_arg}")

    # Lookup standard tramite PARAM_TAG
    candidate_dirs = [
        BASE_DIR.parent / "COMSOL" / "4roll" / "Datasets",
        BASE_DIR.parent / "COMSOL" / "4roll",
        Path("/kaggle/input"),
        Path("/kaggle/working"),
    ]
    for cdir in candidate_dirs:
        if cdir.is_dir():
            try:
                found = resolve_dataset_path(cdir, param_tag)
                if found.is_file():
                    return found.resolve()
            except Exception:
                pass
            matches = list(cdir.rglob(f"*{param_tag}*.csv"))
            if matches:
                return matches[0].resolve()

    raise FileNotFoundError(
        f"[STRICT DATASET ERROR] Impossibile trovare il dataset per il tag: {param_tag}\n"
        f"Verifica che il dataset sia presente in COMSOL/4roll/Datasets o specifica --dataset <percorso>."
    )


# Parsing metadati fisici se specificato file custom
if CUSTOM_DATASET_ARG is not None:
    try:
        _meta = parse_dataset_metadata(CUSTOM_DATASET_ARG)
        LAM_TRUE = _meta["lam_true"]
        MU_P_TRUE = _meta["mu_p_true"]
        MU_S_TRUE = _meta["mu_s_true"]
        ALPHA_TRUE = _meta["alpha_true"]
        EPS_TRUE = _meta["eps_true"]
        MESH_TAG = _meta["mesh_tag"]
    except Exception as e:
        print(f"[Avviso] Parsing metadati da nome file: {e}. Uso parametri default.")

MU_TOT_TRUE = MU_S_TRUE + MU_P_TRUE
BETA_TRUE = MU_S_TRUE / MU_TOT_TRUE
PARAM_TAG = build_dataset_tag(LAM_TRUE, MU_P_TRUE, MU_S_TRUE, ALPHA_TRUE, EPS_TRUE, MESH_TAG)
DATASET_PATH = find_dataset_file(PARAM_TAG, CUSTOM_DATASET_ARG)

# ============================================================================
# 5. CONFIGURAZIONE OUTPUT & LOGGING
# ============================================================================
def _format_iters(n):
    if n == 0:
        return "0"
    if n % 1000 == 0:
        return f"{n // 1000}k"
    return f"{n / 1000:.1f}k"


budget_tag = f"Ph1_{_format_iters(ADAM_EPOCHS_PHASE1)}+{_format_iters(LBFGS_MAX_ITERS_PHASE1)}"
run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
config_name = f"[{run_timestamp}][DIR][{PARAM_TAG}][{budget_tag}]"

OUTPUT_DIR = BASE_DIR / "output_4rollmill" / config_name
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
global_log_path = OUTPUT_DIR / "train_log.txt"

# Iniezione globale dei parametri di configurazione
for k, v in list(globals().items()):
    if k.isupper():
        setattr(builtins, k, v)
        for mod in [src.debug, src.physics, src.train, src.utils]:
            setattr(mod, k, v)

# ============================================================================
# 6. ESECUZIONE SIMULAZIONE DIRETTA PINN
# ============================================================================
if __name__ == "__main__":
    print("=" * 75)
    print("  SIMULAZIONE CFD DIRETTA VISCOELASTICA TRAMITE PINN (4-ROLL MILL)")
    print("=" * 75)
    print(f"Device: {DEVICE} | Precisione Default: {torch.get_default_dtype()}")
    print(f"Dataset COMSOL di Riferimento: {DATASET_PATH.resolve()}")
    print(f"Output Directory: {OUTPUT_DIR.resolve()}\n")

    print("CONFIGURAZIONE DEL PROBLEMA DIRETTO:")
    print(f"  - Modalità: PROBLEMA DIRETTO (Parametri fisici bloccati ai valori veri)")
    print(f"  - Parametri Fisici: lambda={LAM_TRUE:.4f} s | mu_p={MU_P_TRUE:.4f} Pa*s | mu_s={MU_S_TRUE:.4f} Pa*s | alpha={ALPHA_TRUE:.4f} | eps={EPS_TRUE:.4f}")
    print(f"  - Supervisione Dati Interni: W_DATA = {W_DATA} (ZERO dati di velocità interni forniti alla rete)")
    print(f"  - Condizioni al Contorno: W_BC = {W_BC} (Parete No-Slip + Rotazione 4 Rulli)")
    print(f"  - Stress sui Rulli: DISATTIVATO (USE_ROLL_STRESS_BC = {USE_ROLL_STRESS_BC})")
    print(f"  - Ancoraggio Pressione: ALGEBRICO HARD (model_p vincola p(x_0) == p_ref per costruzione)")
    print(f"  - Equazioni PDE Attive: Momentum (w={W_MOMENTUM}) + Costitutiva (w={W_CONSTITUTIVE})")
    print(f"  - Budget Ottimizzazione: {ADAM_EPOCHS_PHASE1} epoche Adam (FP32) + {LBFGS_MAX_ITERS_PHASE1} iterazioni L-BFGS (FP64)")
    print("=" * 75 + "\n")

    # 1. Caricamento Dataset COMSOL
    data = load_data(filepath=str(DATASET_PATH), eta_0=ETA_0)

    # 2. Inizializzazione Modello con Hard Pressure Anchor
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

    # 3. Inizializzazione Fisica
    physics = Physics(
        U_ref=data["U_ref"],
        H_ref=data["H"],
        H_coord=data["H_coord"],
        var_weights=data["var_weights"],
        inverse_mode=INVERSE_PROBLEM,
        tau_scale=data["tau_scale_vec"],
        p_scale=data["p_scale"],
        use_roll_stress_bc=USE_ROLL_STRESS_BC,
        w_roll_stress=W_ROLL_STRESS,
        eta_0=ETA_0,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Rete Neurale: CombinedModel ({len(HIDDEN_LAYERS)} layer x {HIDDEN_LAYERS[0]} neuroni, {total_params:,} parametri)")
    print(f"Ancoraggio Hard Pressione Attivo: {getattr(model, 'hard_anchor', False)}")

    # 4. Training (Adam FP32 + L-BFGS FP64)
    tb_dir = OUTPUT_DIR / "tb_logs"
    tb_dir.mkdir(parents=True, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=str(tb_dir))

    history = train(
        model=model,
        physics=physics,
        data=data,
        resume_checkpoint=None,
        save_dir=OUTPUT_DIR,
        tb_writer=tb_writer,
    )
    tb_writer.close()

    # 5. Valutazione e Report Finale Errori Relativi L2 vs COMSOL
    print(f"\n{'=' * 75}\nVALUTAZIONE ACCURATEZZA SIMULAZIONE DIRETTA (Errori Relativi L2 vs COMSOL)\n{'=' * 75}")
    errors = compute_l2_errors(model, physics, data)
    for field_name, err in errors.items():
        print(f"  L2 Relativo {field_name:<15s}: {err:.6e} ({err * 100:.2f}%)")

    final_losses = evaluate_final_losses(model, physics, data)
    print(f"\nResidui Finali PDE e Boundary Conditions:")
    for k, v in final_losses.items():
        print(f"  {k:<20s}: {v:.6e}")

    # 6. Generazione Grafici e Mappe Diagnostiche
    print(f"\nGenerazione figure e mappe diagnostiche 2D in corso...")
    history.plot_losses(str(OUTPUT_DIR / "loss_history.png"))
    history.plot_l2_errors(str(OUTPUT_DIR / "l2_errors_history.png"))
    generate_all_diagnostics(model, physics, data, str(OUTPUT_DIR))

    print(f"\n[Completato con Successo] Tutti i risultati sono stati salvati in:\n  {OUTPUT_DIR.resolve()}")
