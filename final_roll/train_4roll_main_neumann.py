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

# Import dai moduli src originali (NON modificati)
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

# --- Condizione di Neumann sulla Pressione alle Pareti (Nuova Logica) ---
USE_NEUMANN_BC = True
W_NEUMANN = 50.0

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR.parent / "COMSOL" / "4roll" / "4_roll_mill.csv"

# Checkpoint consolidato Fase 1
RESUME_CHECKPOINT = BASE_DIR / "checkpoints" / "checkpoint_inverso_fase1_40k+10k.pth"

# Parametri Fisici REALI
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

# Iperparametri Fase 1 (gia' completata nel checkpoint)
ADAM_EPOCHS_PHASE1 = 40000
USE_LBFGS_PHASE1 = True
LBFGS_MAX_ITERS_PHASE1 = 10000

# Iperparametri Fase 2 (Richiesta specifica: 30k Adam + 2k L-BFGS)
ADAM_EPOCHS_PHASE2 = 30000
USE_LBFGS_PHASE2 = True
LBFGS_MAX_ITERS_PHASE2 = 2000

BASE_LR = 1e-3
ADAM_EPS = 1e-7
PARAM_LR_FACTOR = 0.1
GRAD_CLIP_NORM = 1000.0
PARAM_CLIP_NORM = 1.0

# Niente warmup asimmetrico: mu_s, p e psi attivi insieme da inizio Fase 2
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
# 3. SOTTOCLASSE SPECIALIZZATA NEUMANN PHYSICS
# ============================================================================
class NeumannPhysics(Physics):
    """
    Estende Physics implementando la condizione di Neumann ESATTA della quantita' di moto:
    dp/dn = n . ( (mu_s / eta_0) * nabla^2(u) + div(tau) )
    sulle 4 pareti esterne (Walls). Nessun dato COMSOL di pressione viene utilizzato.
    """
    def boundary_loss(self, model, bc_data, var_w=None, active_bcs=None, **kwargs):
        # 1. Calcolo normale della loss al contorno standard (u=0 su walls, u e tau su rolls, p=0 su PressurePoint)
        total_loss = super().boundary_loss(model, bc_data, var_w, active_bcs=active_bcs, **kwargs)

        # 2. In Fase 2, la pressione e' attiva ("p" in active_bcs)
        if active_bcs is not None and "p" in active_bcs and "Walls" in bc_data:
            use_neumann = getattr(builtins, "USE_NEUMANN_BC", True)
            if use_neumann:
                walls_xy = bc_data["Walls"]["xy"]
                w_pts = walls_xy.clone().requires_grad_(True)
                u, v, p, tau = self.get_velocity(model, w_pts, create_graph=True)

                # Derivate prime di velocita' per Laplaciano
                grad_u = self._grad(u, w_pts, create_graph=True)
                u_x, u_y = grad_u[:, 0:1], grad_u[:, 1:2]
                grad_v = self._grad(v, w_pts, create_graph=True)
                v_x, v_y = grad_v[:, 0:1], -u_x

                # Derivate seconde di velocita' (Laplaciano)
                u_xx = self._grad(u_x, w_pts, create_graph=True)[:, 0:1]
                grad_u_y = self._grad(u_y, w_pts, create_graph=True)
                u_yx, u_yy = grad_u_y[:, 0:1], grad_u_y[:, 1:2]
                v_xx = self._grad(v_x, w_pts, create_graph=True)[:, 0:1]
                v_yy = -u_yx

                lap_u = u_xx + u_yy
                lap_v = v_xx + v_yy

                # Divergenza dello stress tau alle pareti
                tau_xx, tau_xy, tau_yy = tau[:, 0:1], tau[:, 1:2], tau[:, 2:3]
                g_txx = self._grad(tau_xx, w_pts, create_graph=True)
                g_txy = self._grad(tau_xy, w_pts, create_graph=True)
                g_tyy = self._grad(tau_yy, w_pts, create_graph=True)
                div_tau_x = g_txx[:, 0:1] + g_txy[:, 1:2]
                div_tau_y = g_txy[:, 0:1] + g_tyy[:, 1:2]

                # Target esatto della quantita' di moto: T = (mu_s / eta_0) * lap(u) + div(tau)
                mu_s_nd = self.mu_s / self.eta_0
                Tx = mu_s_nd * lap_u + div_tau_x
                Ty = mu_s_nd * lap_v + div_tau_y

                # Gradiente di pressione predetto
                g_p = self._grad(p, w_pts, create_graph=True)
                px, py = g_p[:, 0:1], g_p[:, 1:2]

                # Maschere geometriche per pareti verticali e orizzontali
                x_w = walls_xy[:, 0:1]
                y_w = walls_xy[:, 1:2]
                eps_tol = 1e-4

                vert_mask = (x_w < eps_tol) | (x_w > 1.0 - eps_tol)
                horiz_mask = (y_w < eps_tol) | (y_w > 1.0 - eps_tol)

                # Residuo normale: (px - Tx)^2 su verticali, (py - Ty)^2 su orizzontali
                res_vert = (px[vert_mask] - Tx[vert_mask]) ** 2
                res_horiz = (py[horiz_mask] - Ty[horiz_mask]) ** 2

                l_neumann_exact = torch.mean(torch.cat([res_vert, res_horiz], dim=0))

                w_n = getattr(builtins, "W_NEUMANN", 50.0)
                total_loss = total_loss + w_n * l_neumann_exact

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
    run_name = f"[{timestamp}][INV][STAGED][Ph2_{adam_k}k+{lbfgs_k}_Neumann]"
    OUTPUT_DIR = BASE_DIR / "output_4rollmill" / run_name
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    builtins.OUTPUT_DIR = OUTPUT_DIR

    global_log_path = OUTPUT_DIR / "train_log.txt"

    print("=" * 75)
    print(f"AVVIO TRAINING PINN 4-ROLL MILL [NEUMANN WALL BC]")
    print(f"Device: {DEVICE} | Precisione: {torch.get_default_dtype()}")
    print(f"Checkpoint Fase 1: {RESUME_CHECKPOINT.name}")
    print(f"Fase 2 Budget: {ADAM_EPOCHS_PHASE2} Adam + {LBFGS_MAX_ITERS_PHASE2} L-BFGS")
    print(f"Neumann Wall BC (dp/dn = 0): Attivo con peso W_NEUMANN = {W_NEUMANN}")
    print(f"Output salvato in: {OUTPUT_DIR}")
    print("=" * 75)

    # 1. Caricamento Dataset
    data = load_data(filepath=DATASET_PATH, eta_0=ETA_0)

    # 2. Inizializzazione Modello e Fisica (usando NeumannPhysics)
    model = CombinedModel(p_scale=data["p_scale"], tau_scale=data["tau_scale"]).to(DEVICE)
    physics = NeumannPhysics(
        U_ref=data["U_ref"],
        H_ref=data["H"],
        H_coord=data["H_coord"],
        var_weights=data["var_weights"],
        inverse_mode=INVERSE_PROBLEM,
        tau_scale=data["tau_scale"],
        p_scale=data["p_scale"],
        eta_0=ETA_0,
    ).to(DEVICE)

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
        tb_writer=tb_writer,
    )
    tb_writer.close()

    # 4. Report Finale
    params = physics.log_params()
    print(f"\n{'=' * 60}\nRISULTATI FINALI PARAMETRI FISICI [NEUMANN RUN]\n{'=' * 60}")
    print(f"  eta_0 (scala rif.) : {params['eta_0']:.6f} Pa·s")
    print(f"  mu_p* (adimens.)   : {params['mu_p_nd']:.6f}  (true: {MU_P_TRUE/ETA_0:.6f})")
    print(f"  mu_p  (dimension.) : {params['mu_p']:.6f} Pa·s (true: {MU_P_TRUE:.6f})")
    print(f"  mu_s* (adimens.)   : {params['mu_s_nd']:.6f}  (true: {MU_S_TRUE/ETA_0:.6f})")
    print(f"  mu_s  (dimension.) : {params['mu_s']:.6f} Pa·s (true: {MU_S_TRUE:.6f})")
    print(f"  mu_tot (dimension.): {params['mu_tot']:.6f} Pa·s (true: {MU_TOT_TRUE:.6f})")
    print(f"  beta  (ratio)      : {params['beta']:.6f}  (true: {BETA_TRUE:.6f})")
    print(f"  lam   (dimension.) : {params['lam']:.6f} s (true: {LAM_TRUE:.6f})")

    final_losses = evaluate_final_losses(model, physics, data)
    print(f"\n{'=' * 60}\nREPORT FINALE LOSS\n{'=' * 60}")
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

    if DEBUG_MODE:
        test_random_points(model, physics, data, num_points=10)
        debug_physics_magnitudes(model, physics, data, num_points=2000)

    print(f"\n[OK] Esecuzione terminata con successo. Risultati in: {OUTPUT_DIR}")
