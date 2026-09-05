import os
import sys
from datetime import datetime
from pathlib import Path

# Assicura che la directory final_roll sia sempre nel PYTHONPATH a runtime
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import cKDTree
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Import dai moduli src
from src.debug import test_random_points, debug_physics_magnitudes
from src.physics import Physics, evaluate_final_losses, compute_l2_errors
from src.train import FCN, initialize_last_layer_zero, init_weights_xavier
from src.utils import (
    load_data,
    weighted_mse,
    convert_to_fp32,
    convert_to_fp64,
    launch_tensorboard_server
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
        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")
        text = sep.join(map(str, args)) + end
        with open(global_log_path, "a", encoding="utf-8") as f:
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
# 2. COSTANTI E PERCORSI STANDARD
# ============================================================================
DATASET_PATH = BASE_DIR.parent / "COMSOL" / "4roll" / "4_roll_mill.csv"
DERIVATIVES_CACHE_PATH = BASE_DIR.parent / "COMSOL" / "4roll" / "comsol_derivatives_mls.pt"

# Parametri Fisici REALI (Ground Truth COMSOL)
MU_S_TRUE = 0.1       # Viscosità solvente [Pa·s]
MU_P_TRUE = 0.9       # Viscosità polimerica [Pa·s]
MU_TOT_TRUE = 1.0     # Viscosità totale [Pa·s]
BETA_TRUE = 0.10      # Rapporto viscosità
LAM_TRUE = 0.05       # Tempo di rilassamento [s]
EPS_TRUE = 0.0        # PTT (0)
ALPHA_TRUE = 0.0      # Giesekus (0)
RHO = 1000.0          # Densità [kg/m³]

# Scala di normalizzazione globale (Pa·s)
ETA_0 = 2.0

# Guess Iniziale Perturbato per il Problema Inverso (80% del valore vero)
GUESS_FACTOR = 0.80
GUESS_MU_S = MU_S_TRUE * GUESS_FACTOR      # 0.0800 Pa·s
GUESS_MU_P = MU_P_TRUE                    # 0.9000 Pa·s (fissato al vero valore COMSOL)
GUESS_LAM = LAM_TRUE                      # 0.0500 s (fissato al vero valore COMSOL)

MIN_MU_S = 1e-6
MIN_MU_P = 1e-6
MIN_LAM = 1e-6

# Architettura Rete Solo Pressione (model_p)
HIDDEN_LAYERS = [128] * 8
ACTIVATION = nn.SiLU
VARIANCE_EPS = 1e-4

# Budget Fase 2: Idrodinamica & Viscosità Solvente (mu_s) calibrato per ~6-7h di training
ADAM_EPOCHS_PHASE2 = 80000
USE_LBFGS_PHASE2 = True
LBFGS_MAX_ITERS_PHASE2 = 6000
WARMUP_PHASE2_EPOCHS = 0      # NESSUN WARMUP: mu_s attivo e addestrabile fin da epoca 0

# Iperparametri Ottimizzatore
BASE_LR = 1e-3
ADAM_EPS = 1e-7
PARAM_LR_FACTOR = 0.1        # LR per mu_s: 1e-4
GRAD_CLIP_NORM = 1000.0
PARAM_CLIP_NORM = 1.0

# Pesi Funzione di Loss: SOLO Momentum e Pressure Point
W_MOMENTUM = 1.0
W_BC_PRES = 10.0             # Ancoraggio forte del punto di Dirichlet p(x0, y0) = p_comsol(x0, y0)

# Chunk Size per gestione VRAM
CHUNK_SIZE_ADAM = 16384
CHUNK_SIZE_LBFGS = 8192

# ============================================================================
# 3. INIEZIONE GLOBALI PER MODULI SRC
# ============================================================================
for module in [src.debug, src.physics, src.train, src.utils]:
    for name, val in list(globals().items()):
        if name.isupper():
            module.__dict__[name] = val
            builtins.__dict__[name] = val

# ============================================================================
# 4. CALCOLO DERIVATE SPAZIALI COMSOL TRAMITE MOVING LEAST SQUARES (MLS)
# ============================================================================
def compute_or_load_comsol_derivatives(data, cache_path, k_neighbors=32):
    """
    Calcola (o carica dalla cache) le derivate spaziali ad alta precisione
    direttamente dai nodi COMSOL tramite Moving Least Squares di 3° grado.
    Restituisce conv_u, conv_v, lap_u, lap_v, div_tau_x, div_tau_y adimensionali.
    """
    if cache_path.exists():
        print(f"\n[Cache] Caricamento derivate COMSOL precalcolate da: {cache_path}")
        cache = torch.load(cache_path, map_location=DEVICE)
        print("  Derivate COMSOL caricate con successo!")
        return cache

    print("\n" + "=" * 70)
    print("CALCOLO DERIVATE SPAZIALI DAI DATI COMSOL (Moving Least Squares 3° Grado)")
    print("=" * 70)

    coords_np = data["coords"].cpu().numpy()
    u_np = data["u"].cpu().numpy()
    v_np = data["v"].cpu().numpy()
    txx_np = data["tau_xx"].cpu().numpy()
    txy_np = data["tau_xy"].cpu().numpy()
    tyy_np = data["tau_yy"].cpu().numpy()

    H_ref = data["H"]
    H_coord = data["H_coord"]
    s = H_ref / H_coord

    print(f"Costruzione albero KDTree su {len(coords_np):,} punti...")
    tree = cKDTree(coords_np)
    dists, nbrs = tree.query(coords_np, k=k_neighbors)

    print("Risoluzione batched GPU del sistema polinomiale locale...")
    coords_t = torch.tensor(coords_np, device=DEVICE, dtype=torch.float64)
    nbrs_t = torch.tensor(nbrs, device=DEVICE, dtype=torch.long)
    dists_t = torch.tensor(dists, device=DEVICE, dtype=torch.float64)

    dx = coords_t[nbrs_t, 0] - coords_t[:, 0:1]
    dy = coords_t[nbrs_t, 1] - coords_t[:, 1:2]
    h = dists_t[:, -1:] / 2.0
    w = torch.exp(- (dx**2 + dy**2) / (2 * h**2 + 1e-16))

    ones = torch.ones_like(dx)
    A = torch.stack([
        ones, dx, dy, 0.5 * dx**2, dx * dy, 0.5 * dy**2,
        (dx**3) / 6.0, (dx**2 * dy) / 2.0, (dx * dy**2) / 2.0, (dy**3) / 6.0
    ], dim=-1) * w.unsqueeze(-1)

    ATA = torch.matmul(A.transpose(1, 2), A) + 1e-12 * torch.eye(10, device=DEVICE, dtype=torch.float64).unsqueeze(0)
    fields = torch.tensor(np.column_stack([u_np, v_np, txx_np, txy_np, tyy_np]), device=DEVICE, dtype=torch.float64)
    fields_nbrs = fields[nbrs_t] * w.unsqueeze(-1)
    ATB = torch.matmul(A.transpose(1, 2), fields_nbrs)

    coeff = torch.linalg.solve(ATA, ATB)

    # Derivate prime con fattore di scala s = H_ref / H_coord
    ux = coeff[:, 1:2, 0] * s
    uy = coeff[:, 2:3, 0] * s
    vx = coeff[:, 1:2, 1] * s
    vy = coeff[:, 2:3, 1] * s

    # Laplaciano velocità (derivate seconde) con s^2
    lap_u = (coeff[:, 3:4, 0] + coeff[:, 5:6, 0]) * (s**2)
    lap_v = (coeff[:, 3:4, 1] + coeff[:, 5:6, 1]) * (s**2)

    # Divergenza dello stress con fattore di scala s
    div_tx = (coeff[:, 1:2, 2] + coeff[:, 2:3, 3]) * s
    div_ty = (coeff[:, 1:2, 3] + coeff[:, 2:3, 4]) * s

    # Convezione: u*ux + v*uy
    u_f = fields[:, 0:1]
    v_f = fields[:, 1:2]
    conv_u = u_f * ux + v_f * uy
    conv_v = u_f * vx + v_f * vy

    # Verifica fisica incompressibilità
    div_u = ux + vy
    u_mag = torch.mean(torch.sqrt(u_f**2 + v_f**2))
    err_incompr = torch.mean(torch.abs(div_u)).item() / (u_mag.item() + 1e-12) * 100.0

    print(f"  Media |div(u)|: {torch.mean(torch.abs(div_u)).item():.4e}")
    print(f"  Errore relativo incompressibilita' COMSOL: {err_incompr:.3f}%")

    cache = {
        "conv_u": conv_u.float(),
        "conv_v": conv_v.float(),
        "lap_u": lap_u.float(),
        "lap_v": lap_v.float(),
        "div_tau_x": div_tx.float(),
        "div_tau_y": div_ty.float(),
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_path)
    print(f"[OK] Derivate salvate in cache: {cache_path}")
    return cache


# ============================================================================
# 5. MOTORE DI TRAINING PER LA FASE 2 (SOLO model_p E mu_s SENZA WARMUP)
# ============================================================================
def train_phase2(model_p, physics, data, derivatives, save_dir, tb_writer=None):
    """
    Fase 2 Pura:
    - NESSUNA rete per psi o tau (i termini cinematici e di sforzo sono numerici esatti da COMSOL).
    - L'unica rete neurale addestrata è model_p.
    - Parametro fisico addestrato: mu_s (attivo fin da epoca 0, nessun warmup).
    - Loss: W_MOMENTUM * Momentum + W_BC_PRES * PressurePoint.
    """
    xy_all = data["coords"]
    p_true = data["p"]
    var_w = data["var_weights"]
    p_scale = data["p_scale"]
    H_ref = data["H"]
    H_coord = data["H_coord"]
    scale_grad = H_ref / H_coord

    p_pt_data = data["boundary_groups"]["PressurePoint"]
    p_pt_xy = p_pt_data["xy"]
    p_pt_true = p_pt_data["fields"]["p"]

    # Termini differenziali COMSOL
    conv_u_all = derivatives["conv_u"].to(DEVICE)
    conv_v_all = derivatives["conv_v"].to(DEVICE)
    lap_u_all = derivatives["lap_u"].to(DEVICE)
    lap_v_all = derivatives["lap_v"].to(DEVICE)
    div_tx_all = derivatives["div_tau_x"].to(DEVICE)
    div_ty_all = derivatives["div_tau_y"].to(DEVICE)

    # History per tracking
    history = {
        "epoch": [],
        "loss_tot": [],
        "loss_mom": [],
        "loss_pres": [],
        "mu_s": [],
        "mu_s_rel_err": [],
        "l2_p": [],
    }

    # Configurazione Fisica: mu_s è immediatamente addestrabile
    physics.inverse_mode = True
    physics.set_trainable("mu_s", True)  # ATTIVO FIN DA SUBITO (WARMUP ZERO)
    physics.set_trainable("mu_p", False)
    physics.set_trainable("lam", False)

    # Inizializza ultimo layer di model_p a zero per partenza simmetrica e neutra
    initialize_last_layer_zero(model_p)

    # Ottimizzatore Adam (Due gruppi: pesi model_p + parametro mu_s attivo)
    p_params = [p for p in model_p.parameters() if p.requires_grad]
    phys_param = [physics._raw_mu_s]

    opt_groups = [
        {"params": p_params, "lr": BASE_LR},
        {"params": phys_param, "lr": BASE_LR * PARAM_LR_FACTOR},  # LR dedicato 1e-4 da epoca 0
    ]
    optimizer_adam = torch.optim.Adam(opt_groups, eps=ADAM_EPS)
    scheduler_adam = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_adam, T_max=ADAM_EPOCHS_PHASE2, eta_min=1e-6
    )

    print("\n" + "=" * 70)
    print(f"AVVIO ADDESTRAMENTO FASE 2 PURA (ADAM: {ADAM_EPOCHS_PHASE2} epoche, NO WARMUP)")
    print(f"  Loss: {W_MOMENTUM} * Momentum + {W_BC_PRES} * PressurePoint")
    print(f"  Rete attiva: model_p ({sum(p.numel() for p in p_params):,} pesi)")
    print(f"  Reti psi e tau: ELIMINATE (sostituite da derivate COMSOL)")
    print(f"  Parametro da identificare: mu_s (guess: {physics.mu_s.item():.4f} Pa·s, target: {MU_S_TRUE:.4f} Pa·s)")
    print("=" * 70)

    # Funzione di step loss con accumulo chunked
    def compute_step_loss(model_p, points, conv_u, conv_v, lap_u, lap_v, div_tx, div_ty,
                          p_pt_xy_in, p_pt_true_in, p_scale, scale_grad, chunk_size):
        loss_mom_accum = 0.0
        n_pts = points.shape[0]
        Re_scale = physics.Re_scale

        for i in range(0, n_pts, chunk_size):
            xc = points[i : i + chunk_size]
            w_chunk = xc.shape[0] / n_pts

            xph = xc.clone().requires_grad_(True)
            p_pred = model_p(xph) * p_scale

            grad_p = torch.autograd.grad(
                p_pred, xph,
                grad_outputs=torch.ones_like(p_pred),
                create_graph=True
            )[0] * scale_grad

            p_x = grad_p[:, 0:1]
            p_y = grad_p[:, 1:2]

            cu = conv_u[i : i + chunk_size]
            cv = conv_v[i : i + chunk_size]
            lu = lap_u[i : i + chunk_size]
            lv = lap_v[i : i + chunk_size]
            dtx = div_tx[i : i + chunk_size]
            dty = div_ty[i : i + chunk_size]

            # Ricalcolo di mu_s_nd ad ogni chunk per rigenerare il grafo computazionale ed evitare errori al backward
            mu_s_nd = physics.mu_s / physics.eta_0

            f_u = Re_scale * cu + p_x - mu_s_nd * lu - dtx
            f_v = Re_scale * cv + p_y - mu_s_nd * lv - dty

            lm = 0.5 * torch.mean(f_u**2 + f_v**2)
            chunk_loss = W_MOMENTUM * lm * w_chunk
            loss_mom_accum += lm.item() * w_chunk

            if isinstance(chunk_loss, torch.Tensor):
                chunk_loss.backward()

        # Vincolo di Dirichlet sul punto di pressione (usa le coordinate fornite per preservare il dtype)
        x_pt = p_pt_xy_in.clone().requires_grad_(True)
        p_pred_pt = model_p(x_pt) * p_scale
        l_pres = weighted_mse(p_pred_pt, p_pt_true_in, var_w["p"])
        loss_pres_val = l_pres.item()

        pres_chunk_loss = W_BC_PRES * l_pres
        if isinstance(pres_chunk_loss, torch.Tensor):
            pres_chunk_loss.backward()

        tot_loss = (W_MOMENTUM * loss_mom_accum) + (W_BC_PRES * loss_pres_val)
        return tot_loss, loss_mom_accum, loss_pres_val

    # Loop Adam (mu_s sempre mobile fin dal primo step)
    pbar = tqdm(range(ADAM_EPOCHS_PHASE2), desc="Adam Phase 2", mininterval=2.0)
    for epoch in pbar:
        model_p.train()
        optimizer_adam.zero_grad(set_to_none=True)

        tot_loss, l_mom, l_pres = compute_step_loss(
            model_p, xy_all, conv_u_all, conv_v_all, lap_u_all, lap_v_all, div_tx_all, div_ty_all,
            p_pt_xy, p_pt_true, p_scale, scale_grad, CHUNK_SIZE_ADAM
        )

        torch.nn.utils.clip_grad_norm_(model_p.parameters(), GRAD_CLIP_NORM)
        if physics.inverse_mode and physics._raw_mu_s.requires_grad:
            torch.nn.utils.clip_grad_norm_([physics._raw_mu_s], PARAM_CLIP_NORM)

        optimizer_adam.step()
        scheduler_adam.step()

        # Postfix progress bar
        if (epoch + 1) % 10 == 0 or epoch == 0 or (epoch + 1) == ADAM_EPOCHS_PHASE2:
            current_mus = physics.mu_s.item()
            err_mus = abs(current_mus - MU_S_TRUE) / MU_S_TRUE * 100.0
            pbar.set_postfix({
                "Loss": f"{tot_loss:.2e}",
                "Mom": f"{l_mom:.2e}",
                "Pres": f"{l_pres:.2e}",
                "mu_s": f"{current_mus:.4f}",
                "Err(%)": f"{err_mus:.1f}%"
            })

        # Report dettagliato periodico con errore L2(p)
        log_full = ((epoch + 1) % max(1, ADAM_EPOCHS_PHASE2 // 30) == 0) or (epoch == 0) or ((epoch + 1) == ADAM_EPOCHS_PHASE2)
        if log_full:
            current_mus = physics.mu_s.item()
            err_mus = abs(current_mus - MU_S_TRUE) / MU_S_TRUE * 100.0

            model_p.eval()
            with torch.no_grad():
                p_eval = model_p(xy_all) * p_scale
                l2_p = (torch.norm(p_eval - p_true) / torch.norm(p_true)).item()

            history["epoch"].append(epoch + 1)
            history["loss_tot"].append(tot_loss)
            history["loss_mom"].append(l_mom)
            history["loss_pres"].append(l_pres)
            history["mu_s"].append(current_mus)
            history["mu_s_rel_err"].append(err_mus)
            history["l2_p"].append(l2_p)

            print(f"\n[Adam Epoca {epoch+1:5d}/{ADAM_EPOCHS_PHASE2}] "
                  f"Loss Tot: {tot_loss:.4e} | Mom: {l_mom:.4e} | Pres BC: {l_pres:.4e}")
            print(f"  -> mu_s: {current_mus:.6f} Pa·s (Target: {MU_S_TRUE:.4f} | Errore: {err_mus:.2f}%)")
            print(f"  -> L2 Errore Pressione: {l2_p:.4e} ({l2_p * 100:.2f}%)")

            if tb_writer is not None:
                tb_writer.add_scalar("Loss/Total", tot_loss, epoch + 1)
                tb_writer.add_scalar("Loss/Momentum", l_mom, epoch + 1)
                tb_writer.add_scalar("Loss/PressurePoint", l_pres, epoch + 1)
                tb_writer.add_scalar("Params/mu_s", current_mus, epoch + 1)
                tb_writer.add_scalar("Params/mu_s_RelErr_pct", err_mus, epoch + 1)
                tb_writer.add_scalar("Errors/L2_p", l2_p, epoch + 1)

    # Salvataggio checkpoint Adam
    chk_adam_path = save_dir / "checkpoint_phase2_adam.pth"
    torch.save({
        "model_p_state_dict": model_p.state_dict(),
        "physics_state_dict": physics.state_dict(),
        "history": history
    }, chk_adam_path)
    print(f"\n[Checkpoint] Adam Fase 2 salvato in: {chk_adam_path}")

    # ==================================================================
    # FASE L-BFGS (FP64)
    # ==================================================================
    if USE_LBFGS_PHASE2 and LBFGS_MAX_ITERS_PHASE2 > 0:
        print("\n" + "=" * 70)
        print(f"FASE L-BFGS 2: {LBFGS_MAX_ITERS_PHASE2} iterazioni (FP64 ad altissima precisione)")
        print("=" * 70)

        # Conversione a FP64
        model_p.double()
        physics.double()
        xy_64 = xy_all.double()
        p_pt_xy_64 = p_pt_xy.double()
        p_pt_true_64 = p_pt_true.double()
        p_true_64 = p_true.double()

        conv_u_64 = conv_u_all.double()
        conv_v_64 = conv_v_all.double()
        lap_u_64 = lap_u_all.double()
        lap_v_64 = lap_v_all.double()
        div_tx_64 = div_tx_all.double()
        div_ty_64 = div_ty_all.double()

        physics.set_trainable("mu_s", True)
        all_trainable_64 = list(model_p.parameters()) + [physics._raw_mu_s]

        optimizer_lbfgs = torch.optim.LBFGS(
            all_trainable_64,
            lr=1.0,
            max_iter=1,
            max_eval=20,
            tolerance_grad=1e-18,
            tolerance_change=1e-18,
            history_size=150,
            line_search_fn="strong_wolfe",
        )

        last_step_vals = {}

        def closure_lbfgs():
            optimizer_lbfgs.zero_grad(set_to_none=True)
            tot_loss, l_mom, l_pres = compute_step_loss(
                model_p, xy_64, conv_u_64, conv_v_64, lap_u_64, lap_v_64, div_tx_64, div_ty_64,
                p_pt_xy_64, p_pt_true_64, p_scale, scale_grad, CHUNK_SIZE_LBFGS
            )
            last_step_vals["tot"] = tot_loss
            last_step_vals["mom"] = l_mom
            last_step_vals["pres"] = l_pres
            return torch.tensor(tot_loss, device=DEVICE, dtype=torch.float64)

        pbar_lbfgs = tqdm(range(LBFGS_MAX_ITERS_PHASE2), desc="L-BFGS Phase 2", mininterval=2.0)
        for it in pbar_lbfgs:
            optimizer_lbfgs.step(closure_lbfgs)

            current_mus = physics.mu_s.item()
            err_mus = abs(current_mus - MU_S_TRUE) / MU_S_TRUE * 100.0
            tot_l = last_step_vals.get("tot", 0.0)
            pbar_lbfgs.set_postfix({"Loss": f"{tot_l:.2e}", "mu_s": f"{current_mus:.4f}", "Err(%)": f"{err_mus:.1f}%"})

            log_lbfgs = ((it + 1) % max(1, LBFGS_MAX_ITERS_PHASE2 // 20) == 0) or (it == 0) or ((it + 1) == LBFGS_MAX_ITERS_PHASE2)
            if log_lbfgs:
                global_it = ADAM_EPOCHS_PHASE2 + it + 1
                with torch.no_grad():
                    p_eval_64 = model_p(xy_64) * p_scale
                    l2_p = (torch.norm(p_eval_64 - p_true_64) / torch.norm(p_true_64)).item()

                history["epoch"].append(global_it)
                history["loss_tot"].append(tot_l)
                history["loss_mom"].append(last_step_vals.get("mom", 0.0))
                history["loss_pres"].append(last_step_vals.get("pres", 0.0))
                history["mu_s"].append(current_mus)
                history["mu_s_rel_err"].append(err_mus)
                history["l2_p"].append(l2_p)

                print(f"\n[L-BFGS Iter {it+1:4d}/{LBFGS_MAX_ITERS_PHASE2}] "
                      f"Loss: {tot_l:.4e} | Mom: {last_step_vals.get('mom', 0.0):.4e}")
                print(f"  -> mu_s: {current_mus:.6f} Pa·s (Target: {MU_S_TRUE:.4f} | Errore: {err_mus:.2f}%)")
                print(f"  -> L2 Errore Pressione: {l2_p:.4e} ({l2_p * 100:.2f}%)")

                if tb_writer is not None:
                    tb_writer.add_scalar("Loss/Total", tot_l, global_it)
                    tb_writer.add_scalar("Params/mu_s", current_mus, global_it)
                    tb_writer.add_scalar("Errors/L2_p", l2_p, global_it)

        # Ripristino a FP32
        model_p.float()
        physics.float()

    # Salvataggio checkpoint finale
    final_chk_path = save_dir / "checkpoint_phase2_final.pth"
    torch.save({
        "model_p_state_dict": model_p.state_dict(),
        "physics_state_dict": physics.state_dict(),
        "history": history
    }, final_chk_path)
    print(f"\n[Checkpoint Finale] Salvato in: {final_chk_path}")

    return history


# ============================================================================
# 6. REPORT E GENERAZIONE PLOT DIAGNOSTICI
# ============================================================================
def generate_phase2_diagnostics(model_p, physics, data, history, output_dir):
    """Genera plot di confronto e report statistico finale per la Fase 2."""
    print("\n" + "=" * 70)
    print("GENERAZIONE GRAFICI DIAGNOSTICI E REPORT FINALE")
    print("=" * 70)

    xy_all = data["coords"]
    p_true = data["p"]
    x_np = xy_all[:, 0].cpu().numpy()
    y_np = xy_all[:, 1].cpu().numpy()
    p_scale = data["p_scale"]

    model_p.eval()
    with torch.no_grad():
        p_pred = (model_p(xy_all) * p_scale).cpu().numpy().flatten()
        p_true_np = p_true.cpu().numpy().flatten()
        err_abs = np.abs(p_pred - p_true_np)
        l2_err_p = np.linalg.norm(p_pred - p_true_np) / np.linalg.norm(p_true_np)

    # 1. Plot Evoluzione mu_s
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(history["epoch"], history["mu_s"], color="blue", lw=2, label=r"Learned $\mu_s$")
    ax1.axhline(MU_S_TRUE, color="black", linestyle="--", lw=2, label=rf"True $\mu_s$ ({MU_S_TRUE} Pa·s)")
    ax1.axhline(GUESS_MU_S, color="gray", linestyle=":", lw=1.5, label=rf"Initial Guess ({GUESS_MU_S} Pa·s)")
    ax1.set_ylabel(r"$\mu_s$ [Pa·s]", fontsize=12)
    ax1.set_title(r"Evoluzione Viscosità Solvente $\mu_s$ (Zero Reti Psi/Tau - Dati COMSOL Pura MLS)", fontsize=14)
    ax1.grid(True, ls="--", alpha=0.6)
    ax1.legend(fontsize=11)

    ax2.plot(history["epoch"], history["mu_s_rel_err"], color="crimson", lw=2, label=r"Errore Relativo $\mu_s$ (%)")
    ax2.set_xlabel("Epoca / Iterazione Globale", fontsize=12)
    ax2.set_ylabel("Errore Relativo (%)", fontsize=12)
    ax2.set_yscale("log")
    ax2.grid(True, ls="--", alpha=0.6)
    ax2.legend(fontsize=11)

    plt.tight_layout()
    plot_mus_path = output_dir / "params_evolution_mus.png"
    plt.savefig(plot_mus_path, dpi=150)
    plt.close()
    print(f"  [Plot] Curva evoluzione mu_s salvata: {plot_mus_path.name}")

    # 2. Plot Loss History
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history["epoch"], history["loss_tot"], color="black", lw=2, label="Loss Totale")
    ax.plot(history["epoch"], history["loss_mom"], color="purple", lw=1.5, label="Momentum Loss")
    ax.plot(history["epoch"], history["loss_pres"], color="green", lw=1.5, label="PressurePoint Loss")
    ax.set_yscale("log")
    ax.set_xlabel("Epoca / Iterazione", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("History Funzione di Loss (Fase 2: Momentum + Dirichlet Point)", fontsize=14)
    ax.grid(True, ls="--", alpha=0.6)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plot_loss_path = output_dir / "loss_history.png"
    plt.savefig(plot_loss_path, dpi=150)
    plt.close()
    print(f"  [Plot] History delle loss salvata: {plot_loss_path.name}")

    # 3. Mappe di Contorno 2D della Pressione
    triang = mtri.Triangulation(x_np, y_np)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    v_min, v_max = min(p_true_np.min(), p_pred.min()), max(p_true_np.max(), p_pred.max())

    c0 = axes[0].tricontourf(triang, p_true_np, levels=60, cmap="viridis", vmin=v_min, vmax=v_max)
    axes[0].set_title("Pressione COMSOL Ground Truth ($p_{true}$)", fontsize=12)
    axes[0].set_aspect("equal")
    plt.colorbar(c0, ax=axes[0])

    c1 = axes[1].tricontourf(triang, p_pred, levels=60, cmap="viridis", vmin=v_min, vmax=v_max)
    axes[1].set_title("Pressione Predetta PINN ($p_{pred}$)", fontsize=12)
    axes[1].set_aspect("equal")
    plt.colorbar(c1, ax=axes[1])

    c2 = axes[2].tricontourf(triang, err_abs, levels=60, cmap="inferno")
    axes[2].set_title(f"Errore Assoluto $|p_{{pred}} - p_{{true}}|$\n(L2 Relativo: {l2_err_p*100:.2f}%)", fontsize=12)
    axes[2].set_aspect("equal")
    plt.colorbar(c2, ax=axes[2])

    for ax in axes:
        ax.set_xlabel("x*")
        ax.set_ylabel("y*")

    plt.tight_layout()
    plot_p_path = output_dir / "pressure_field_comparison.png"
    plt.savefig(plot_p_path, dpi=150)
    plt.close()
    print(f"  [Plot] Confronto campi di pressione salvato: {plot_p_path.name}")

    # Report finale a console
    final_mus = physics.mu_s.item()
    final_err = abs(final_mus - MU_S_TRUE) / MU_S_TRUE * 100.0
    print("\n" + "=" * 70)
    print("RISULTATI FINALI ESPERIMENTO FASE 2 PURA:")
    print("=" * 70)
    print(f"  Viscosità Solvente Reale (Target):  {MU_S_TRUE:.6f} Pa·s")
    print(f"  Guess Iniziale Inverso:             {GUESS_MU_S:.6f} Pa·s")
    print(f"  Viscosità Solvente Identificata:    {final_mus:.6f} Pa·s")
    print(f"  Errore Relativo Finale su mu_s:     {final_err:.4f}%")
    print(f"  Errore L2 Relativo sulla Pressione: {l2_err_p * 100:.4f}%")
    print("=" * 70)


# ============================================================================
# 7. MAIN ENTRYPOINT
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("ESPERIMENTO PINN: FASE 2 PURA CON DERIVATE COMSOL MLS (ZERO RETI PSI/TAU)")
    print("=" * 70)
    print(f"Device: {DEVICE} | Dtype di default: {torch.get_default_dtype()}")
    print(f"Dataset Path: {DATASET_PATH}")
    print(f"Derivatives Cache Path: {DERIVATIVES_CACHE_PATH}")

    # Setup Cartella di Output
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_name = f"[{run_timestamp}][INV][PHASE2_MLS_DIRECT][Ph2_{ADAM_EPOCHS_PHASE2//1000}k+{LBFGS_MAX_ITERS_PHASE2//1000}k]"
    OUTPUT_DIR = BASE_DIR / "output_4rollmill" / run_name
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    global_log_path = OUTPUT_DIR / "train_log.txt"
    print(f"Cartella Risultati: {OUTPUT_DIR}\n")

    # 1. Caricamento Dati COMSOL
    data = load_data(filepath=DATASET_PATH, eta_0=ETA_0)

    # 2. Calcolo o caricamento derivate COMSOL Moving Least Squares
    derivatives = compute_or_load_comsol_derivatives(data, DERIVATIVES_CACHE_PATH)

    # Se invocato con flag --precompute-only, esce qui dopo aver salvato la cache
    if "--precompute-only" in sys.argv:
        print(f"\n[OK] Modalità --precompute-only completata! Cache derivate salvata in: {DERIVATIVES_CACHE_PATH}")
        sys.exit(0)

    # 3. Inizializzazione Rete Solo Pressione e Fisica
    model_p = FCN(n_input=2, n_output=1, hidden_layers=HIDDEN_LAYERS).to(DEVICE)
    model_p.apply(lambda m: init_weights_xavier(m, activation_name=ACTIVATION))

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

    # 4. Avvio TensorBoard (con fallback in caso di ambiente headless o server remoto)
    try:
        launch_tensorboard_server(OUTPUT_DIR.parent)
    except Exception as e:
        print(f"[TensorBoard] Server automatico non avviato ({e}). Procedo con l'addestramento.")
    tb_dir = OUTPUT_DIR / "tb_logs"
    tb_dir.mkdir(parents=True, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=str(tb_dir))

    # 5. Esecuzione Fase 2
    history = train_phase2(
        model_p=model_p,
        physics=physics,
        data=data,
        derivatives=derivatives,
        save_dir=OUTPUT_DIR,
        tb_writer=tb_writer
    )

    tb_writer.close()

    # 6. Report e Diagnostica Finale
    generate_phase2_diagnostics(model_p, physics, data, history, OUTPUT_DIR)

    print(f"\n[FINE ESPERIMENTO] Risultati salvati in: {OUTPUT_DIR}")
