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
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Import moduli ufficiali src
from src.physics import Physics, compute_l2_errors, evaluate_final_losses
from src.train import CombinedModel, precompute_stress_divergence
from src.utils import load_data, weighted_mse, convert_to_fp32, convert_to_fp64, launch_tensorboard_server

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
# 2. COSTANTI E PARAMETRI FISICI (TEST 2: DIRETTO DA CHECKPOINT FASE 1)
# ============================================================================
DATASET_PATH = BASE_DIR.parent / "COMSOL" / "4roll" / "4_roll_mill.csv"
CHECKPOINT_PATH = BASE_DIR / "checkpoints" / "checkpoint_inverso_fase1_40k+10k.pth"

# Parametri Fisici REALI
MU_S_TRUE = 0.1       # Viscosità solvente FISSA al valore reale [Pa·s]
MU_P_TRUE = 0.9       # Viscosità polimerica [Pa·s]
MU_TOT_TRUE = 1.0     # Viscosità totale [Pa·s]
ETA_0 = 2.0           # Scala di normalizzazione globale [Pa·s]
RHO = 1000.0          # Densità [kg/m³]

# Architettura Network
HIDDEN_LAYERS = [128] * 8
ACTIVATION = nn.SiLU
VARIANCE_EPS = 1e-4

# Budget Fase 2 Diretta
ADAM_EPOCHS = 30000
USE_LBFGS = True
LBFGS_MAX_ITERS = 2000

# Iperparametri Ottimizzatore
BASE_LR = 1e-3
ADAM_EPS = 1e-7
GRAD_CLIP_NORM = 1000.0

# Pesi Funzione di Loss (Fase 2 Diretta con Moduli src)
W_DATA = 20.0         # Peso dati velocità (u, v) su model_psi
W_MOMENTUM = 1.0      # Peso equazione di Navier-Stokes
W_BC_PRES = 10.0      # Ancoraggio del SINGOLO PressurePoint

# Chunk Size Gestione VRAM
CHUNK_SIZE_ADAM = 16384
CHUNK_SIZE_LBFGS = 8192

# Iniezione parametri per i moduli src
for module in [src.debug, src.physics, src.train, src.utils]:
    for name, val in list(globals().items()):
        if name.isupper():
            module.__dict__[name] = val
            builtins.__dict__[name] = val


# ============================================================================
# 3. TRAINING ENGINE FASE 2 DIRETTA DA CHECKPOINT
# ============================================================================
def train_direct_from_checkpoint(model, physics, data, save_dir, tb_writer=None):
    xy_all = data["coords"]
    uv_all = data["uv_data"]
    p_true = data["p"]
    var_w = data["var_weights"]
    p_pt_data = data["boundary_groups"]["PressurePoint"]
    p_pt_xy = p_pt_data["xy"]
    p_pt_true = p_pt_data["fields"]["p"]

    total_points = xy_all.shape[0]

    # Precalcolo della divergenza dello stress tau (tau è congelato)
    print("\n[Precomputation] Precalcolo divergenza dello stress tau dai pesi di Fase 1...")
    div_tau_x, div_tau_y = precompute_stress_divergence(model, physics, xy_all, chunk_size=5000)
    print("[Precomputation] Divergenza completata con successo!")

    history = {
        "epoch": [],
        "loss_tot": [],
        "loss_mom": [],
        "loss_data": [],
        "loss_pres": [],
        "l2_p": [],
        "l2_u": [],
        "l2_v": [],
    }

    # Ottimizzatore Adam: model_p a pieno LR, model_psi a LR attenuato
    p_params = list(model.model_p.parameters())
    psi_params = list(model.model_psi.parameters())

    opt_groups = [
        {"params": p_params, "lr": BASE_LR},
        {"params": psi_params, "lr": BASE_LR * 0.1},
    ]
    optimizer_adam = torch.optim.Adam(opt_groups, eps=ADAM_EPS)
    scheduler_adam = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_adam, T_max=ADAM_EPOCHS, eta_min=1e-6)

    print("\n" + "=" * 70)
    print(f"AVVIO TEST 2: FASE 2 DIRETTA DA CHECKPOINT FASE 1 ({ADAM_EPOCHS} epoche)")
    print(f"  mu_s = {physics.mu_s.item():.4f} Pa·s FISSO (trainable: {physics._raw_mu_s.requires_grad})")
    print(f"  Loss: {W_MOMENTUM}*Mom + {W_DATA}*Data_uv + {W_BC_PRES}*PressurePoint")
    print(f"  Reti attive: model_p (train), model_psi (train micro-lr), model_tau (FROZEN)")
    print("=" * 70)

    def compute_step_loss(model, points, labels, p_pt_xy_in, p_pt_true_in, div_tx, div_ty, chunk_size):
        d_loss_accum = 0.0
        p_loss_accum = 0.0
        n_pts = points.shape[0]

        for i in range(0, n_pts, chunk_size):
            xc = points[i : i + chunk_size]
            yc = labels[i : i + chunk_size]
            w_chunk = xc.shape[0] / n_pts

            xph = xc.clone().requires_grad_(True)
            u, v, p, tau = physics.get_velocity(model, xph)

            # Data Loss su velocità (ancora model_psi)
            dl = physics.data_loss(u, v, yc, var_w)
            d_loss_accum += dl.item() * w_chunk

            # Momentum Loss con divergenza tau precalcolata
            chunk_div_tau = (div_tx[i : i + chunk_size], div_ty[i : i + chunk_size])
            lm, _ = physics.compute_pde_losses(
                xph, u, v, p, tau, w_momentum=W_MOMENTUM, w_constitutive=0.0,
                frozen_velocity=False, precomputed_div_tau=chunk_div_tau
            )
            p_loss_accum += lm.item() * w_chunk

            chunk_loss = (W_DATA * dl + W_MOMENTUM * lm) * w_chunk
            if isinstance(chunk_loss, torch.Tensor):
                chunk_loss.backward()

        # Vincolo di Dirichlet sul SINGOLO PressurePoint
        x_pt = p_pt_xy_in.clone().requires_grad_(True)
        _, _, p_pred_pt, _ = physics.get_velocity(model, x_pt)
        l_pres = weighted_mse(p_pred_pt, p_pt_true_in, var_w["p"])
        loss_pres_val = l_pres.item()

        pres_chunk_loss = W_BC_PRES * l_pres
        if isinstance(pres_chunk_loss, torch.Tensor):
            pres_chunk_loss.backward()

        tot_loss = (W_DATA * d_loss_accum) + (W_MOMENTUM * p_loss_accum) + (W_BC_PRES * loss_pres_val)
        return tot_loss, p_loss_accum, d_loss_accum, loss_pres_val

    # Loop Adam
    pbar = tqdm(range(ADAM_EPOCHS), desc="Adam Direct Checkpoint", mininterval=2.0)
    for epoch in pbar:
        model.train()
        optimizer_adam.zero_grad(set_to_none=True)

        tot_loss, l_mom, l_data, l_pres = compute_step_loss(
            model, xy_all, uv_all, p_pt_xy, p_pt_true, div_tau_x, div_tau_y, CHUNK_SIZE_ADAM
        )

        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer_adam.step()
        scheduler_adam.step()

        if (epoch + 1) % 10 == 0 or epoch == 0 or (epoch + 1) == ADAM_EPOCHS:
            pbar.set_postfix({"Loss": f"{tot_loss:.2e}", "Mom": f"{l_mom:.2e}", "Pres": f"{l_pres:.2e}"})

        log_full = ((epoch + 1) % max(1, ADAM_EPOCHS // 25) == 0) or (epoch == 0) or ((epoch + 1) == ADAM_EPOCHS)
        if log_full:
            model.eval()
            with torch.no_grad():
                l2_errs = compute_l2_errors(model, physics, data)

            history["epoch"].append(epoch + 1)
            history["loss_tot"].append(tot_loss)
            history["loss_mom"].append(l_mom)
            history["loss_data"].append(l_data)
            history["loss_pres"].append(l_pres)
            history["l2_p"].append(l2_errs["p"])
            history["l2_u"].append(l2_errs["u"])
            history["l2_v"].append(l2_errs["v"])

            print(f"\n[Adam Epoca {epoch+1:5d}/{ADAM_EPOCHS}] "
                  f"Loss: {tot_loss:.4e} | Mom: {l_mom:.4e} | Data: {l_data:.4e} | Pres BC: {l_pres:.4e}")
            print(f"  -> L2 Errore Pressione: {l2_errs['p']:.4e} ({l2_errs['p'] * 100:.2f}%)")
            print(f"  -> L2 Errore Velocità:  u={l2_errs['u']:.4e}, v={l2_errs['v']:.4e}")

            if tb_writer is not None:
                tb_writer.add_scalar("Loss/Total", tot_loss, epoch + 1)
                tb_writer.add_scalar("Loss/Momentum", l_mom, epoch + 1)
                tb_writer.add_scalar("Errors/L2_p", l2_errs["p"], epoch + 1)
                tb_writer.add_scalar("Errors/L2_u", l2_errs["u"], epoch + 1)

    # Salvataggio Checkpoint Adam
    torch.save({
        "model_state_dict": model.state_dict(),
        "physics_state_dict": physics.state_dict(),
        "history": history
    }, save_dir / "checkpoint_direct_adam.pth")

    # ==================================================================
    # FASE L-BFGS (FP64)
    # ==================================================================
    if USE_LBFGS and LBFGS_MAX_ITERS > 0:
        print("\n" + "=" * 70)
        print(f"FASE L-BFGS (FP64): {LBFGS_MAX_ITERS} iterazioni")
        print("=" * 70)

        convert_to_fp64(model, physics, data)
        xy_64 = data["coords"]
        uv_64 = data["uv_data"]
        p_pt_xy_64 = p_pt_xy.double()
        p_pt_true_64 = p_pt_true.double()
        div_tx_64 = div_tau_x.double()
        div_ty_64 = div_tau_y.double()

        for p in model.parameters():
            p.requires_grad = False
        for p in model.model_p.parameters():
            p.requires_grad = True
        for p in model.model_psi.parameters():
            p.requires_grad = True

        optimizer_lbfgs = torch.optim.LBFGS(
            [p for p in model.parameters() if p.requires_grad],
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
            tot_loss, l_mom, l_data, l_pres = compute_step_loss(
                model, xy_64, uv_64, p_pt_xy_64, p_pt_true_64, div_tx_64, div_ty_64, CHUNK_SIZE_LBFGS
            )
            last_step_vals["tot"] = tot_loss
            last_step_vals["mom"] = l_mom
            last_step_vals["data"] = l_data
            last_step_vals["pres"] = l_pres
            return torch.tensor(tot_loss, device=DEVICE, dtype=torch.float64)

        pbar_lbfgs = tqdm(range(LBFGS_MAX_ITERS), desc="L-BFGS Direct Checkpoint", mininterval=2.0)
        for it in pbar_lbfgs:
            optimizer_lbfgs.step(closure_lbfgs)
            tot_l = last_step_vals.get("tot", 0.0)
            pbar_lbfgs.set_postfix({"Loss": f"{tot_l:.2e}"})

            log_lbfgs = ((it + 1) % max(1, LBFGS_MAX_ITERS // 20) == 0) or (it == 0) or ((it + 1) == LBFGS_MAX_ITERS)
            if log_lbfgs:
                global_it = ADAM_EPOCHS + it + 1
                with torch.no_grad():
                    l2_errs = compute_l2_errors(model, physics, data)

                history["epoch"].append(global_it)
                history["loss_tot"].append(tot_l)
                history["loss_mom"].append(last_step_vals.get("mom", 0.0))
                history["loss_data"].append(last_step_vals.get("data", 0.0))
                history["loss_pres"].append(last_step_vals.get("pres", 0.0))
                history["l2_p"].append(l2_errs["p"])
                history["l2_u"].append(l2_errs["u"])
                history["l2_v"].append(l2_errs["v"])

                print(f"\n[L-BFGS Iter {it+1:4d}/{LBFGS_MAX_ITERS}] "
                      f"Loss: {tot_l:.4e} | Mom: {last_step_vals.get('mom', 0.0):.4e}")
                print(f"  -> L2 Errore Pressione: {l2_errs['p']:.4e} ({l2_errs['p'] * 100:.2f}%)")

                if tb_writer is not None:
                    tb_writer.add_scalar("Loss/Total", tot_l, global_it)
                    tb_writer.add_scalar("Errors/L2_p", l2_errs["p"], global_it)

        convert_to_fp32(model, physics, data)

    torch.save({
        "model_state_dict": model.state_dict(),
        "physics_state_dict": physics.state_dict(),
        "history": history
    }, save_dir / "checkpoint_direct_final.pth")

    return history


# ============================================================================
# 4. REPORT FINALE E PLOT
# ============================================================================
def generate_diagnostics(model, physics, data, history, output_dir):
    xy_all = data["coords"]
    p_true = data["p"]
    x_np = xy_all[:, 0].cpu().numpy()
    y_np = xy_all[:, 1].cpu().numpy()

    model.eval()
    with torch.no_grad():
        _, _, p_pred, _ = physics.get_velocity(model, xy_all)
        p_pred_np = p_pred.cpu().numpy().flatten()
        p_true_np = p_true.cpu().numpy().flatten()
        err_abs = np.abs(p_pred_np - p_true_np)
        l2_err_p = np.linalg.norm(p_pred_np - p_true_np) / np.linalg.norm(p_true_np)

    # 1. Plot Loss History
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history["epoch"], history["loss_tot"], color="black", lw=2, label="Loss Totale")
    ax.plot(history["epoch"], history["loss_mom"], color="purple", lw=1.5, label="Momentum Loss")
    ax.plot(history["epoch"], history["loss_data"], color="blue", lw=1.5, label="Data Loss uv")
    ax.plot(history["epoch"], history["loss_pres"], color="green", lw=1.5, label="PressurePoint Loss")
    ax.set_yscale("log")
    ax.set_xlabel("Epoca / Iterazione")
    ax.set_ylabel("Loss")
    ax.set_title("History Loss (Test 2: Diretto da Checkpoint Fase 1, mu_s=0.10 Fisso)")
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
    ax.set_title("Evoluzione Errore L2 Pressione da Checkpoint Fase 1")
    ax.grid(True, ls="--", alpha=0.6)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "l2_error_p_history.png", dpi=150)
    plt.close()

    # 3. Mappe di Contorno 2D
    triang = mtri.Triangulation(x_np, y_np)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    v_min, v_max = min(p_true_np.min(), p_pred_np.min()), max(p_true_np.max(), p_pred_np.max())

    c0 = axes[0].tricontourf(triang, p_true_np, levels=60, cmap="viridis", vmin=v_min, vmax=v_max)
    axes[0].set_title("Pressione COMSOL Ground Truth ($p_{true}$)")
    axes[0].set_aspect("equal")
    plt.colorbar(c0, ax=axes[0])

    c1 = axes[1].tricontourf(triang, p_pred_np, levels=60, cmap="viridis", vmin=v_min, vmax=v_max)
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
    print("RISULTATI FINALI TEST 2 (PROBLEMA DIRETTO DA CHECKPOINT FASE 1):")
    print("=" * 70)
    print(f"  Viscosità Solvente Fissa:           {MU_S_TRUE:.6f} Pa·s")
    print(f"  Errore L2 Relativo sulla Pressione: {l2_err_p * 100:.4f}%")
    print("=" * 70)


# ============================================================================
# 5. MAIN ENTRYPOINT
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("TEST 2: FASE 2 DIRETTA DA CHECKPOINT FASE 1 (mu_s = 0.10 FISSO)")
    print("=" * 70)
    print(f"Device: {DEVICE} | Dtype: {torch.get_default_dtype()}")
    print(f"Dataset Path:    {DATASET_PATH}")
    print(f"Checkpoint Path: {CHECKPOINT_PATH}")

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint non trovato: {CHECKPOINT_PATH}")

    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_name = f"[{run_timestamp}][DIR][PHASE2_CKPT_FIXED_MUS][Ph2_{ADAM_EPOCHS//1000}k+{LBFGS_MAX_ITERS//1000}k]"
    OUTPUT_DIR = BASE_DIR / "output_4rollmill" / run_name
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    global_log_path = OUTPUT_DIR / "train_log.txt"

    data = load_data(filepath=DATASET_PATH, eta_0=ETA_0)

    model = CombinedModel(p_scale=data["p_scale"], tau_scale=data["tau_scale"]).to(DEVICE)
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

    # 1. Carica pesi dal checkpoint di Fase 1 Inversa
    print(f"\n[Checkpoint] Caricamento pesi da: {CHECKPOINT_PATH}")
    chk = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(chk["model_state_dict"])
    physics.load_state_dict(chk["physics_state_dict"], strict=False)
    print("[Checkpoint] Modello e fisica caricati con successo!")

    # 2. Fissa rigidamente mu_s a 0.10
    physics.guess_mu_s.copy_(torch.tensor(MU_S_TRUE, device=DEVICE))
    physics._raw_mu_s.data.zero_()
    physics._raw_mu_s.requires_grad = False
    physics.set_trainable("mu_s", False)
    physics.set_trainable("mu_p", False)
    physics.set_trainable("lam", False)

    # 3. Congela tau, sblocca p e psi
    for p in model.parameters():
        p.requires_grad = False
    for p in model.model_p.parameters():
        p.requires_grad = True
    for p in model.model_psi.parameters():
        p.requires_grad = True

    try:
        launch_tensorboard_server(OUTPUT_DIR.parent)
    except Exception as e:
        print(f"[TensorBoard] Server automatico non avviato ({e}).")

    tb_dir = OUTPUT_DIR / "tb_logs"
    tb_dir.mkdir(parents=True, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=str(tb_dir))

    history = train_direct_from_checkpoint(
        model=model,
        physics=physics,
        data=data,
        save_dir=OUTPUT_DIR,
        tb_writer=tb_writer
    )
    tb_writer.close()

    generate_diagnostics(model, physics, data, history, OUTPUT_DIR)
    print(f"\n[FINE TEST 2] Risultati salvati in: {OUTPUT_DIR}")
