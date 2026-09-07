import os
import sys
from datetime import datetime
from pathlib import Path

# Assicura che la directory final_roll sia sempre nel PYTHONPATH
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

# Import moduli di supporto
from src.physics import Physics
from src.train import FCN, init_weights_xavier
from src.utils import load_data, weighted_mse, launch_tensorboard_server

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
# 2. COSTANTI E PARAMETRI FISICI (PROBLEMA DIRETTO)
# ============================================================================
DATASET_PATH = BASE_DIR.parent / "COMSOL" / "4roll" / "4_roll_mill.csv"
DERIVATIVES_CACHE_PATH = BASE_DIR.parent / "COMSOL" / "4roll" / "comsol_derivatives_mls.pt"

# Parametri Fisici REALI (Ground Truth COMSOL)
MU_S_TRUE = 0.1       # Viscosità solvente FISSA [Pa·s]
MU_P_TRUE = 0.9       # Viscosità polimerica FISSA [Pa·s]
MU_TOT_TRUE = 1.0     # Viscosità totale [Pa·s]
ETA_0 = 2.0           # Scala di riferimento globale [Pa·s]
RHO = 1000.0          # Densità [kg/m³]

# Architettura Rete Solo Pressione (model_p)
HIDDEN_LAYERS = [128] * 8
ACTIVATION = nn.SiLU
VARIANCE_EPS = 1e-4

# Budget Test Diretto (Veloce e Preciso per Problema Diretto Convesso)
ADAM_EPOCHS = 20000
USE_LBFGS = True
LBFGS_MAX_ITERS = 2000

# Iperparametri Ottimizzatore
BASE_LR = 1e-3
ADAM_EPS = 1e-7
GRAD_CLIP_NORM = 1000.0

# Pesi Funzione di Loss: SOLO Momentum e Singolo PressurePoint
W_MOMENTUM = 1.0
W_BC_PRES = 10.0      # Ancoraggio Dirichlet del singolo punto di pressione

# Chunk Size VRAM
CHUNK_SIZE_ADAM = 16384
CHUNK_SIZE_LBFGS = 8192

# Iniezione parametri per i moduli src
for module in [src.debug, src.physics, src.train, src.utils]:
    for name, val in list(globals().items()):
        if name.isupper():
            module.__dict__[name] = val
            builtins.__dict__[name] = val


# ============================================================================
# 3. CALCOLO O CARICAMENTO DERIVATE SPAZIALI COMSOL (MLS)
# ============================================================================
def compute_or_load_comsol_derivatives(data, cache_path, k_neighbors=32):
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

    tree = cKDTree(coords_np)
    dists, nbrs = tree.query(coords_np, k=k_neighbors)

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

    ux = coeff[:, 1:2, 0] * s
    uy = coeff[:, 2:3, 0] * s
    vx = coeff[:, 1:2, 1] * s
    vy = coeff[:, 2:3, 1] * s

    lap_u = (coeff[:, 3:4, 0] + coeff[:, 5:6, 0]) * (s**2)
    lap_v = (coeff[:, 3:4, 1] + coeff[:, 5:6, 1]) * (s**2)

    div_tx = (coeff[:, 1:2, 2] + coeff[:, 2:3, 3]) * s
    div_ty = (coeff[:, 1:2, 3] + coeff[:, 2:3, 4]) * s

    u_f = fields[:, 0:1]
    v_f = fields[:, 1:2]
    conv_u = u_f * ux + v_f * uy
    conv_v = u_f * vx + v_f * vy

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
# 4. TRAINING FASE 2 DIRETTA (mu_s FISSO A 0.10, SOLO model_p)
# ============================================================================
def train_direct_p(model_p, physics, data, derivatives, save_dir, tb_writer=None):
    xy_all = data["coords"]
    p_true = data["p"]
    var_w = data["var_weights"]
    p_scale = data["p_scale"]
    scale_grad = data["H"] / data["H_coord"]

    p_pt_data = data["boundary_groups"]["PressurePoint"]
    p_pt_xy = p_pt_data["xy"]
    p_pt_true = p_pt_data["fields"]["p"]

    conv_u_all = derivatives["conv_u"].to(DEVICE)
    conv_v_all = derivatives["conv_v"].to(DEVICE)
    lap_u_all = derivatives["lap_u"].to(DEVICE)
    lap_v_all = derivatives["lap_v"].to(DEVICE)
    div_tx_all = derivatives["div_tau_x"].to(DEVICE)
    div_ty_all = derivatives["div_tau_y"].to(DEVICE)

    # Costanti Adimensionali FISSE
    Re_scale = physics.Re_scale
    mu_s_nd = physics.mu_s / physics.eta_0  # Valore fisso = 0.10 / 2.0 = 0.05

    history = {
        "epoch": [],
        "loss_tot": [],
        "loss_mom": [],
        "loss_pres": [],
        "l2_p": [],
    }

    optimizer_adam = torch.optim.Adam(model_p.parameters(), lr=BASE_LR, eps=ADAM_EPS)
    scheduler_adam = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_adam, T_max=ADAM_EPOCHS, eta_min=1e-6)

    print("\n" + "=" * 70)
    print(f"AVVIO TEST 1: FASE 2 DIRETTA (mu_s = {physics.mu_s.item():.4f} Pa·s FISSO, {ADAM_EPOCHS} epoche)")
    print(f"  Loss: {W_MOMENTUM} * Momentum + {W_BC_PRES} * Singolo PressurePoint")
    print(f"  Rete addestrata: SOLO model_p ({sum(p.numel() for p in model_p.parameters()):,} pesi)")
    print(f"  mu_s trainable: {physics._raw_mu_s.requires_grad}")
    print("=" * 70)

    def compute_step_loss(model_p, points, conv_u, conv_v, lap_u, lap_v, div_tx, div_ty,
                          p_pt_xy_in, p_pt_true_in, p_scale, scale_grad, chunk_size, mu_s_val_nd):
        loss_mom_accum = 0.0
        n_pts = points.shape[0]

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

            f_u = Re_scale * cu + p_x - mu_s_val_nd * lu - dtx
            f_v = Re_scale * cv + p_y - mu_s_val_nd * lv - dty

            lm = 0.5 * torch.mean(f_u**2 + f_v**2)
            chunk_loss = W_MOMENTUM * lm * w_chunk
            loss_mom_accum += lm.item() * w_chunk

            if isinstance(chunk_loss, torch.Tensor):
                chunk_loss.backward()

        x_pt = p_pt_xy_in.clone().requires_grad_(True)
        p_pred_pt = model_p(x_pt) * p_scale
        l_pres = weighted_mse(p_pred_pt, p_pt_true_in, var_w["p"])
        loss_pres_val = l_pres.item()

        pres_chunk_loss = W_BC_PRES * l_pres
        if isinstance(pres_chunk_loss, torch.Tensor):
            pres_chunk_loss.backward()

        tot_loss = (W_MOMENTUM * loss_mom_accum) + (W_BC_PRES * loss_pres_val)
        return tot_loss, loss_mom_accum, loss_pres_val

    # Loop Adam
    pbar = tqdm(range(ADAM_EPOCHS), desc="Adam Phase 2 Diretto", mininterval=2.0)
    for epoch in pbar:
        model_p.train()
        optimizer_adam.zero_grad(set_to_none=True)

        tot_loss, l_mom, l_pres = compute_step_loss(
            model_p, xy_all, conv_u_all, conv_v_all, lap_u_all, lap_v_all, div_tx_all, div_ty_all,
            p_pt_xy, p_pt_true, p_scale, scale_grad, CHUNK_SIZE_ADAM, mu_s_nd
        )

        torch.nn.utils.clip_grad_norm_(model_p.parameters(), GRAD_CLIP_NORM)
        optimizer_adam.step()
        scheduler_adam.step()

        if (epoch + 1) % 10 == 0 or epoch == 0 or (epoch + 1) == ADAM_EPOCHS:
            pbar.set_postfix({"Loss": f"{tot_loss:.2e}", "Mom": f"{l_mom:.2e}", "Pres": f"{l_pres:.2e}"})

        log_full = ((epoch + 1) % max(1, ADAM_EPOCHS // 25) == 0) or (epoch == 0) or ((epoch + 1) == ADAM_EPOCHS)
        if log_full:
            model_p.eval()
            with torch.no_grad():
                p_eval = model_p(xy_all) * p_scale
                l2_p = (torch.norm(p_eval - p_true) / torch.norm(p_true)).item()

            history["epoch"].append(epoch + 1)
            history["loss_tot"].append(tot_loss)
            history["loss_mom"].append(l_mom)
            history["loss_pres"].append(l_pres)
            history["l2_p"].append(l2_p)

            print(f"\n[Adam Epoca {epoch+1:5d}/{ADAM_EPOCHS}] "
                  f"Loss Tot: {tot_loss:.4e} | Mom: {l_mom:.4e} | Pres BC: {l_pres:.4e}")
            print(f"  -> L2 Errore Pressione: {l2_p:.4e} ({l2_p * 100:.2f}%)")

            if tb_writer is not None:
                tb_writer.add_scalar("Loss/Total", tot_loss, epoch + 1)
                tb_writer.add_scalar("Loss/Momentum", l_mom, epoch + 1)
                tb_writer.add_scalar("Loss/PressurePoint", l_pres, epoch + 1)
                tb_writer.add_scalar("Errors/L2_p", l2_p, epoch + 1)

    # Salvataggio Checkpoint Adam
    torch.save({
        "model_p_state_dict": model_p.state_dict(),
        "history": history
    }, save_dir / "checkpoint_direct_adam.pth")

    # ==================================================================
    # FASE L-BFGS (FP64)
    # ==================================================================
    if USE_LBFGS and LBFGS_MAX_ITERS > 0:
        print("\n" + "=" * 70)
        print(f"FASE L-BFGS 2 (DIRETTO): {LBFGS_MAX_ITERS} iterazioni (FP64 ad altissima precisione)")
        print("=" * 70)

        model_p.double()
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

        mu_s_nd_64 = float(mu_s_nd)

        optimizer_lbfgs = torch.optim.LBFGS(
            model_p.parameters(),
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
                p_pt_xy_64, p_pt_true_64, p_scale, scale_grad, CHUNK_SIZE_LBFGS, mu_s_nd_64
            )
            last_step_vals["tot"] = tot_loss
            last_step_vals["mom"] = l_mom
            last_step_vals["pres"] = l_pres
            return torch.tensor(tot_loss, device=DEVICE, dtype=torch.float64)

        pbar_lbfgs = tqdm(range(LBFGS_MAX_ITERS), desc="L-BFGS Phase 2 Diretto", mininterval=2.0)
        for it in pbar_lbfgs:
            optimizer_lbfgs.step(closure_lbfgs)

            tot_l = last_step_vals.get("tot", 0.0)
            pbar_lbfgs.set_postfix({"Loss": f"{tot_l:.2e}"})

            log_lbfgs = ((it + 1) % max(1, LBFGS_MAX_ITERS // 20) == 0) or (it == 0) or ((it + 1) == LBFGS_MAX_ITERS)
            if log_lbfgs:
                global_it = ADAM_EPOCHS + it + 1
                with torch.no_grad():
                    p_eval_64 = model_p(xy_64) * p_scale
                    l2_p = (torch.norm(p_eval_64 - p_true_64) / torch.norm(p_true_64)).item()

                history["epoch"].append(global_it)
                history["loss_tot"].append(tot_l)
                history["loss_mom"].append(last_step_vals.get("mom", 0.0))
                history["loss_pres"].append(last_step_vals.get("pres", 0.0))
                history["l2_p"].append(l2_p)

                print(f"\n[L-BFGS Iter {it+1:4d}/{LBFGS_MAX_ITERS}] "
                      f"Loss: {tot_l:.4e} | Mom: {last_step_vals.get('mom', 0.0):.4e}")
                print(f"  -> L2 Errore Pressione: {l2_p:.4e} ({l2_p * 100:.2f}%)")

                if tb_writer is not None:
                    tb_writer.add_scalar("Loss/Total", tot_l, global_it)
                    tb_writer.add_scalar("Errors/L2_p", l2_p, global_it)

        model_p.float()

    torch.save({
        "model_p_state_dict": model_p.state_dict(),
        "history": history
    }, save_dir / "checkpoint_direct_final.pth")

    return history


# ============================================================================
# 5. GENERAZIONE DIAGNOSTICHE E REPORT FINALE
# ============================================================================
def generate_direct_diagnostics(model_p, data, history, output_dir):
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

    # 1. Plot Loss History
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history["epoch"], history["loss_tot"], color="black", lw=2, label="Loss Totale")
    ax.plot(history["epoch"], history["loss_mom"], color="purple", lw=1.5, label="Momentum Loss")
    ax.plot(history["epoch"], history["loss_pres"], color="green", lw=1.5, label="PressurePoint Loss")
    ax.set_yscale("log")
    ax.set_xlabel("Epoca / Iterazione")
    ax.set_ylabel("Loss")
    ax.set_title("History Loss (Problema Diretto Pressione: mu_s = 0.10 Fisso, 1 PressurePoint)")
    ax.grid(True, ls="--", alpha=0.6)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_history.png", dpi=150)
    plt.close()

    # 2. Plot L2 Error History
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history["epoch"], [e * 100.0 for e in history["l2_p"]], color="crimson", lw=2, label="Errore L2 Pressione (%)")
    ax.set_yscale("log")
    ax.set_xlabel("Epoca / Iterazione")
    ax.set_ylabel("Errore L2 (%)")
    ax.set_title("Evoluzione Errore L2 Pressione nel Problema Diretto")
    ax.grid(True, ls="--", alpha=0.6)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "l2_error_p_history.png", dpi=150)
    plt.close()

    # 3. Mappe di Contorno 2D
    triang = mtri.Triangulation(x_np, y_np)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    v_min, v_max = min(p_true_np.min(), p_pred.min()), max(p_true_np.max(), p_pred.max())

    c0 = axes[0].tricontourf(triang, p_true_np, levels=60, cmap="viridis", vmin=v_min, vmax=v_max)
    axes[0].set_title("Pressione COMSOL Ground Truth ($p_{true}$)")
    axes[0].set_aspect("equal")
    plt.colorbar(c0, ax=axes[0])

    c1 = axes[1].tricontourf(triang, p_pred, levels=60, cmap="viridis", vmin=v_min, vmax=v_max)
    axes[1].set_title("Pressione Predetta PINN ($p_{pred}$)")
    axes[1].set_aspect("equal")
    plt.colorbar(c1, ax=axes[1])

    c2 = axes[2].tricontourf(triang, err_abs, levels=60, cmap="inferno")
    axes[2].set_title(f"Errore Assoluto $|p_{{pred}} - p_{{true}}|$\n(L2 Relativo: {l2_err_p*100:.2f}%)")
    axes[2].set_aspect("equal")
    plt.colorbar(c2, ax=axes[2])

    for ax in axes:
        ax.set_xlabel("x*")
        ax.set_ylabel("y*")

    plt.tight_layout()
    plt.savefig(output_dir / "pressure_field_comparison.png", dpi=150)
    plt.close()

    print("\n" + "=" * 70)
    print("RISULTATI FINALI TEST 1 (PROBLEMA DIRETTO MLS):")
    print("=" * 70)
    print(f"  Viscosità Solvente Fissa:           {MU_S_TRUE:.6f} Pa·s")
    print(f"  Errore L2 Relativo sulla Pressione: {l2_err_p * 100:.4f}%")
    print("=" * 70)


# ============================================================================
# 6. MAIN ENTRYPOINT
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("TEST 1: FASE 2 DIRETTA CON DERIVATE COMSOL MLS (mu_s = 0.10 FISSO)")
    print("=" * 70)
    print(f"Device: {DEVICE} | Dtype: {torch.get_default_dtype()}")
    print(f"Dataset Path: {DATASET_PATH}")
    print(f"Derivatives Cache: {DERIVATIVES_CACHE_PATH}")

    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_name = f"[{run_timestamp}][DIR][PHASE2_MLS_FIXED_MUS][Ph2_{ADAM_EPOCHS//1000}k+{LBFGS_MAX_ITERS//1000}k]"
    OUTPUT_DIR = BASE_DIR / "output_4rollmill" / run_name
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    global_log_path = OUTPUT_DIR / "train_log.txt"

    data = load_data(filepath=DATASET_PATH, eta_0=ETA_0)
    derivatives = compute_or_load_comsol_derivatives(data, DERIVATIVES_CACHE_PATH)

    model_p = FCN(n_input=2, n_output=1, hidden_layers=HIDDEN_LAYERS).to(DEVICE)
    model_p.apply(lambda m: init_weights_xavier(m, activation_name=ACTIVATION))

    physics = Physics(
        U_ref=data["U_ref"],
        H_ref=data["H"],
        H_coord=data["H_coord"],
        var_weights=data["var_weights"],
        inverse_mode=False,
        tau_scale=data["tau_scale"],
        p_scale=data["p_scale"],
        eta_0=ETA_0,
    ).to(DEVICE)

    # Assicura rigidamente che mu_s sia esattamente 0.10 e non addestrabile
    physics.guess_mu_s.copy_(torch.tensor(MU_S_TRUE, device=DEVICE))
    physics._raw_mu_s.data.zero_()
    physics._raw_mu_s.requires_grad = False

    try:
        launch_tensorboard_server(OUTPUT_DIR.parent)
    except Exception as e:
        print(f"[TensorBoard] Server automatico non avviato ({e}).")

    tb_dir = OUTPUT_DIR / "tb_logs"
    tb_dir.mkdir(parents=True, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=str(tb_dir))

    history = train_direct_p(
        model_p=model_p,
        physics=physics,
        data=data,
        derivatives=derivatives,
        save_dir=OUTPUT_DIR,
        tb_writer=tb_writer
    )
    tb_writer.close()

    generate_direct_diagnostics(model_p, data, history, OUTPUT_DIR)
    print(f"\n[FINE TEST 1] Risultati salvati in: {OUTPUT_DIR}")
