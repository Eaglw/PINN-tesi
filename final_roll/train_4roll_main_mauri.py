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
  - Formulazione Navier-Stokes nativamente adimensionale (scale_mom = 1.0, Proposta AA rettificata).
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

# Parametri Fisici REALI (Ground Truth)
MU_S_TRUE = 0.500
MU_P_TRUE = 0.500
MU_TOT_TRUE = 1.000
BETA_TRUE = 0.500
LAM_TRUE = 0.100
EPS_TRUE = 0.0
ALPHA_TRUE = 0.0
MESH_TAG = "125k"
RHO = 1000.0
WARMUP_UNLOCK_EPOCH = 0
TRANSFER_CKPT_PATH = None
ADAM_EPOCHS_OVERRIDE = None
LBFGS_ITERS_OVERRIDE = None

# Risoluzione mesh e parametri fisici da CLI (default '125k', sovrascrivibile es. --mesh 5k --alpha 0.1 o --dataset <nome>)
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

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = resolve_dataset_path(BASE_DIR.parent / "COMSOL" / "4roll" / "Datasets", PARAM_TAG)

# Checkpoint di partenza: disattivato per training ex-novo da zero
RESUME_CHECKPOINT = None

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
GUESS_EPS = 0.25
GUESS_ALPHA = 0.25
TRAIN_ALPHA = True
TRAIN_EPS = True

HIDDEN_LAYERS = [128] * 8
ACTIVATION = nn.SiLU

# Budget Fase 1: Cinematica & Reologia (40k Adam + 10k L-BFGS di raffinamento)
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

# Budget Fase 2: Disattivata (focus su benchmark di convergenza mesh in Fase 1)
ADAM_EPOCHS_PHASE2 = 0
USE_LBFGS_PHASE2 = False
LBFGS_MAX_ITERS_PHASE2 = 0

# Warmup Fase 2 (opzionale, default 0 epoche: mu_tot attivo da subito)
WARMUP_PHASE2_EPOCHS = 0
USE_MU_TOT_PARAM = True

# Supporto per esecuzione rapida di collaudo (--smoke-test)
if "--smoke-test" in sys.argv:
    print("\n[ATTENZIONE] Modalita' --smoke-test attiva: 2 epoche Adam F1, 2 iterazioni L-BFGS F1, 2 epoche Adam F2, 2 iterazioni L-BFGS F2.")
    ADAM_EPOCHS_PHASE1 = 2
    LBFGS_MAX_ITERS_PHASE1 = 2
    ADAM_EPOCHS_PHASE2 = 2
    LBFGS_MAX_ITERS_PHASE2 = 2

BASE_LR = 2.5e-3
ETA_MIN = 2.5e-6
ADAM_EPS = 1e-8
ADAM_EPS_PHYS = 1e-15
PARAM_LR_FACTOR = 1.0
GRAD_CLIP_NORM = 5.0
PARAM_CLIP_NORM = 1.0

# Pesi di Loss
W_DATA_1 = 1.0
W_BC_1 = 5.0
W_CONSTITUTIVE = 1.0

W_DATA_2 = 20.0
W_BC_2 = 5.0
W_MOMENTUM = 1.0
W_DRIFT = 0.1
VARIANCE_EPS = 1e-4

# ============================================================================
# 3. INIZIALIZZAZIONE OUTPUT
# ============================================================================
layers_str = f"{len(HIDDEN_LAYERS)}x{HIDDEN_LAYERS[0]}"
mode_tag = "INV" if INVERSE_PROBLEM else "DIR"

def _format_iters(n):
    if n == 0:
        return "0"
    if n % 1000 == 0:
        return f"{n // 1000}k"
    return f"{n / 1000:.1f}k"

tl_prefix = "TL_" if TRANSFER_CKPT_PATH is not None else ""
budget_tag = f"{tl_prefix}Ph1_{_format_iters(ADAM_EPOCHS_PHASE1)}+{_format_iters(LBFGS_MAX_ITERS_PHASE1)}_Ph2_{_format_iters(ADAM_EPOCHS_PHASE2)}+{_format_iters(LBFGS_MAX_ITERS_PHASE2)}"
if WARMUP_UNLOCK_EPOCH > 0:
    budget_tag += f"_Warmup{_format_iters(WARMUP_UNLOCK_EPOCH)}"
run_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
config_name = f"[{run_timestamp}][{mode_tag}][{PARAM_TAG}][{budget_tag}][mauri]"

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
    print("PINN 4-ROLL MILL: ADDESTRAMENTO COMPLETO END-TO-END FASE 1 + FASE 2 (PC MAURIZIO)")
    print(f"Device: {DEVICE} | Dtype: {torch.get_default_dtype()}")
    print(f"Dataset: {DATASET_PATH}")
    ckpt_msg = RESUME_CHECKPOINT.name if RESUME_CHECKPOINT is not None else "Nessuno (Partenza da zero: Fase 1 + Fase 2)"
    print(f"Checkpoint di partenza: {ckpt_msg}")
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

    # 2b. Caricamento Pesi da Transfer Learning (se specificato)
    if TRANSFER_CKPT_PATH is not None:
        transfer_p = Path(TRANSFER_CKPT_PATH)
        if not transfer_p.is_absolute():
            transfer_p = BASE_DIR / transfer_p
        if not transfer_p.exists():
            alt_p = BASE_DIR / "checkpoints" / transfer_p.name
            if alt_p.exists():
                transfer_p = alt_p
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

    # Verifica eventuale checkpoint di ripresa
    if RESUME_CHECKPOINT is not None and not RESUME_CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint specificato non trovato: {RESUME_CHECKPOINT}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModello inizializzato: {total_params:,} parametri totali")
    print(f"Configurazione Addestramento Completo (ViscoelasticNet Staged):")
    print(f"  - Fase 1: {ADAM_EPOCHS_PHASE1} Adam + {LBFGS_MAX_ITERS_PHASE1} L-BFGS (Stima lambda ed eta_p, p congelata)")
    print(f"  - Fase 2: {ADAM_EPOCHS_PHASE2} Adam + {LBFGS_MAX_ITERS_PHASE2} L-BFGS (tau congelato, psi mobile, stima eta_s)")
    print(f"  - Ancoraggio Pressione: HARD ALGEBRICO (p(x0) = p_ref esatto, penalty soft rimossa)")
    print(f"  - Formulazione Momento: Nativamente Adimensionale (scale_mom = {physics.scale_mom.item():.1f})")

    # 3. Setup TensorBoard
    launch_tensorboard_server(OUTPUT_DIR.parent)
    tb_dir = OUTPUT_DIR / "tb_logs"
    tb_dir.mkdir(parents=True, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=str(tb_dir))

    # 4. Esecuzione Addestramento Completo (Fase 1 -> Fase 2)
    history = train(
        model,
        physics,
        data,
        resume_checkpoint=RESUME_CHECKPOINT,
        save_dir=OUTPUT_DIR,
        tb_writer=tb_writer
    )
    tb_writer.close()

    # 5. Archiviazione Checkpoint Fase 1 per benchmark e test futuri
    f1_ckpt_in_run = OUTPUT_DIR / "checkpoint_lbfgs_phase1.pth"
    if f1_ckpt_in_run.exists():
        f1_tag = f"Ph1_{_format_iters(ADAM_EPOCHS_PHASE1)}+{_format_iters(LBFGS_MAX_ITERS_PHASE1)}"
        f1_dest = BASE_DIR / "checkpoints" / f"checkpoint_inverso_fase1_{PARAM_TAG}_{f1_tag}.pth"
        import shutil
        shutil.copy2(f1_ckpt_in_run, f1_dest)
        print(f"\n[Checkpoint F1] Checkpoint consolidato Fase 1 archiviato in: {f1_dest}")

    # 6. Report Risultati Finali
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

    # 7. Generazione Plot e Diagnostiche
    print(f"\nGenerazione diagnostiche e plot in: {OUTPUT_DIR} ...")
    history.plot_losses(str(OUTPUT_DIR / "loss_history.png"))
    history.plot_params(str(OUTPUT_DIR / "params_evolution.png"))
    # 8. Summary Benchmark Tabellare & Salvataggio JSON (Facile consultazione)
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

    print(f"\n[OK] Run Completa End-to-End conclusa con successo sul PC di Maurizio! Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
