"""
train_4roll_kaggle2.py — Pressure-Only training con derivate MLS dai campi PINN.

Workflow:
  1. Carica checkpoint psi+tau (100k epoche)
  2. Inferenza sulle reti congelate → valori puntuali u, v, tau
  3. Calcola derivate spaziali via MLS (non Autograd!) su quei valori
  4. Addestra solo PressureModel con momentum MLS + PressurePoint BC
"""
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn as nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# Setup base directory
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

# ============================================================================
# 1. SETUP
# ============================================================================
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.set_default_dtype(torch.float32)
torch.set_float32_matmul_precision("high")

SEED = 123
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# 2. CONFIGURAZIONI
# ============================================================================
DATASET_PATH = BASE_DIR.parent / "COMSOL" / "4roll" / "4_roll_mill.csv"
RESUME_CHECKPOINT = BASE_DIR / "checkpoints" / "checkpoint_psi+tau_100k.pth"

# Parametri fisici
MU_S_TRUE = 0.1
MU_P_TRUE = 0.9
LAM_TRUE = 0.05
EPS_TRUE = 0.0
ALPHA_TRUE = 0.0
RHO = 1000.0

GUESS_MULTIPLIER = 0.8
GUESS_MU_S = MU_S_TRUE * GUESS_MULTIPLIER
GUESS_MU_P = MU_P_TRUE * GUESS_MULTIPLIER
GUESS_LAM = LAM_TRUE * GUESS_MULTIPLIER
GUESS_EPS = 0.0
GUESS_ALPHA = 0.0

# Architettura
HIDDEN_LAYERS = [128] * 8
ACTIVATION = nn.SiLU

# Training
ADAM_EPOCHS = 50000
LBFGS_MAX_ITERS = 10000
BASE_LR = 1e-3
ADAM_EPS = 1e-7
GRAD_CLIP_NORM = 5.0

# Loss weights
W_BC = 2.0
W_PHYSICS = 3.0
W_DATA = 0.0
VARIANCE_EPS = 1e-4

# MLS
MLS_K = 50  # Numero di vicini per MLS

# Iniezione costanti nei moduli src
import src.debug, src.physics, src.train, src.utils

for module in [src.debug, src.physics, src.train, src.utils]:
    for name, val in list(globals().items()):
        if name.isupper():
            module.__dict__[name] = val

from src.utils import load_data, plot_fields, plot_high_stress_regions
from src.physics import Physics
from src.train import CombinedModel

# Importa MLS dal kaggle originale
from train_4roll_kaggle import (
    precompute_comsol_derivatives,
    PressureModel,
    init_weights_xavier,
    compute_pressure_l2_error,
    PressureHistory,
    cast_double,
    cast_float,
)

# Output directory
config_name = f"kaggle2_PINN_MLS_E{ADAM_EPOCHS}_L{LBFGS_MAX_ITERS}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
OUTPUT_DIR = BASE_DIR / "output_4rollmill" / config_name
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

log_file_path = OUTPUT_DIR / "train_log.txt"


def log_print(*args, **kwargs):
    print(*args, **kwargs)
    with open(log_file_path, "a", encoding="utf-8") as f:
        print(*args, file=f, **kwargs)


# ============================================================================
# 3. MAIN
# ============================================================================
if __name__ == "__main__":
    log_print(f"Device: {DEVICE} | Dtype: {torch.get_default_dtype()}")
    log_print(f"Dataset: {DATASET_PATH}")
    log_print(f"Checkpoint: {RESUME_CHECKPOINT}")
    log_print(f"Output: {OUTPUT_DIR}\n")
    log_print("=" * 60)

    # -------------------------------------------
    # STEP 1: Caricamento dati e checkpoint PINN
    # -------------------------------------------
    data = load_data()
    xy_all = data["coords"]
    p_all = data["p"]
    var_w = data["var_weights"]
    bc_data = data["boundary_groups"]
    total_points = xy_all.shape[0]

    # Numeri adimensionali
    mu_tot = MU_S_TRUE + MU_P_TRUE
    Re = RHO * data["U_ref"] * data["H"] / mu_tot
    beta = MU_S_TRUE / mu_tot
    s = data["H"] / data["H_coord"]
    log_print(f"Scaling: Re = {Re:.4f}, beta = {beta:.4f}, s = {s:.6f}")

    # Carica il modello PINN completo (psi + tau + p)
    model_pinn = CombinedModel(p_scale=data["p_scale"], tau_scale=data["tau_scale"]).to(DEVICE)
    chk = torch.load(str(RESUME_CHECKPOINT), map_location=DEVICE)
    model_pinn.load_state_dict(chk["model_state_dict"])
    model_pinn.eval()

    physics = Physics(
        U_ref=data["U_ref"],
        H_ref=data["H"],
        H_coord=data["H_coord"],
        var_weights=data["var_weights"],
        inverse_mode=False,
        tau_scale=data["tau_scale"],
        p_scale=data["p_scale"],
    ).to(DEVICE)
    physics.load_state_dict(chk["physics_state_dict"])

    log_print("\n[Checkpoint] Modello PINN caricato con successo.")

    # -------------------------------------------
    # STEP 2: Inferenza sui campi PINN (no grad)
    # -------------------------------------------
    log_print("\n[Inferenza] Calcolo campi u, v, tau dalle reti PINN...")
    chunk_infer = 10000
    u_pinn_list, v_pinn_list, tau_pinn_list = [], [], []

    with torch.no_grad():
        for i in range(0, total_points, chunk_infer):
            xc = xy_all[i : i + chunk_infer]
            # Per calcolare u, v serve autograd su psi (grad_psi)
            xc_g = xc.clone().requires_grad_(True)
            with torch.enable_grad():
                u, v, p, tau = physics.get_velocity(model_pinn, xc_g, create_graph=False)
            u_pinn_list.append(u.detach())
            v_pinn_list.append(v.detach())
            tau_pinn_list.append(tau.detach())

    u_pinn = torch.cat(u_pinn_list, dim=0)
    v_pinn = torch.cat(v_pinn_list, dim=0)
    tau_pinn = torch.cat(tau_pinn_list, dim=0)
    txx_pinn = tau_pinn[:, 0:1]
    txy_pinn = tau_pinn[:, 1:2]
    tyy_pinn = tau_pinn[:, 2:3]

    # Errori L2 dei campi PINN rispetto a COMSOL
    l2_u = (torch.norm(u_pinn - data["u"]) / torch.norm(data["u"])).item()
    l2_v = (torch.norm(v_pinn - data["v"]) / torch.norm(data["v"])).item()
    l2_txx = (torch.norm(txx_pinn - data["tau_xx"]) / torch.norm(data["tau_xx"])).item()
    l2_txy = (torch.norm(txy_pinn - data["tau_xy"]) / torch.norm(data["tau_xy"])).item()
    l2_tyy = (torch.norm(tyy_pinn - data["tau_yy"]) / torch.norm(data["tau_yy"])).item()
    log_print(f"  L2 PINN vs COMSOL: u={l2_u:.4f}, v={l2_v:.4f}")
    log_print(f"                     txx={l2_txx:.4f}, txy={l2_txy:.4f}, tyy={l2_tyy:.4f}")

    # -------------------------------------------
    # STEP 3: Derivate spaziali MLS sui campi PINN
    # -------------------------------------------
    log_print(f"\n[MLS] Calcolo derivate spaziali MLS (K={MLS_K}) sui campi PINN...")
    derivs = precompute_comsol_derivatives(
        xy_all, u_pinn, v_pinn, txx_pinn, txy_pinn, tyy_pinn, DEVICE, chunk_size=5000, K=MLS_K
    )

    # Libera memoria PINN (non serve più)
    del model_pinn, physics, u_pinn_list, v_pinn_list, tau_pinn_list, tau_pinn
    torch.cuda.empty_cache()

    # -------------------------------------------
    # STEP 4: Training Pressione (Adam, FP32)
    # -------------------------------------------
    log_print(f"\n{'=' * 60}")
    log_print(f"FASE ADAM: Training Pressione con MLS - {ADAM_EPOCHS} epoche")
    log_print(f"{'=' * 60}")

    model = PressureModel(p_scale=data["p_scale"]).to(DEVICE)
    model.apply(lambda m: init_weights_xavier(m, activation_name="silu"))

    history = PressureHistory()
    optimizer = torch.optim.Adam(model.parameters(), lr=BASE_LR, eps=ADAM_EPS)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ADAM_EPOCHS, eta_min=1e-6)

    chunk_size = 5000 if DEVICE.type == "cuda" else total_points

    pbar = tqdm(range(ADAM_EPOCHS), desc="Adam (Kaggle2)", mininterval=2.0)
    for epoch in pbar:
        model.train()
        optimizer.zero_grad(set_to_none=True)

        loss_m_accum = 0.0

        for i in range(0, total_points, chunk_size):
            xc = xy_all[i : i + chunk_size]
            w_chunk = xc.shape[0] / total_points

            xc_ph = xc.clone().requires_grad_(True)
            p_pred = model(xc_ph)

            grad_p = torch.autograd.grad(p_pred.sum(), xc_ph, create_graph=True, retain_graph=True)[0]
            p_x = grad_p[:, 0:1]
            p_y = grad_p[:, 1:2]

            # Derivate MLS precomputate
            ux = derivs["u_x"][i : i + chunk_size]
            uy = derivs["u_y"][i : i + chunk_size]
            uxx = derivs["u_xx"][i : i + chunk_size]
            uyy = derivs["u_yy"][i : i + chunk_size]
            vx = derivs["v_x"][i : i + chunk_size]
            vy = derivs["v_y"][i : i + chunk_size]
            vxx = derivs["v_xx"][i : i + chunk_size]
            vyy = derivs["v_yy"][i : i + chunk_size]
            txx_x_val = derivs["txx_x"][i : i + chunk_size]
            txy_y_val = derivs["txy_y"][i : i + chunk_size]
            txy_x_val = derivs["txy_x"][i : i + chunk_size]
            tyy_y_val = derivs["tyy_y"][i : i + chunk_size]

            u_val = u_pinn[i : i + chunk_size]
            v_val = v_pinn[i : i + chunk_size]

            # Residui momentum (stessa formula di Kaggle)
            f_u = Re * (u_val * (ux * s) + v_val * (uy * s)) + p_x * s - beta * ((uxx + uyy) * s**2) - ((txx_x_val + txy_y_val) * s)
            f_v = Re * (u_val * (vx * s) + v_val * (vy * s)) + p_y * s - beta * ((vxx + vyy) * s**2) - ((txy_x_val + tyy_y_val) * s)

            loss_m = (f_u**2 + f_v**2).mean() / 2.0
            chunk_loss = W_PHYSICS * loss_m * w_chunk
            chunk_loss.backward()
            loss_m_accum += loss_m.item() * w_chunk

        # BC: solo PressurePoint (1 nodo)
        gd_pp = bc_data["PressurePoint"]
        x_bc_pp = gd_pp["xy"].clone().requires_grad_(True)
        p_bc_pp = model(x_bc_pp)
        bc_loss = torch.mean(((p_bc_pp - gd_pp["fields"]["p"]) ** 2) / var_w["p"])
        (W_BC * bc_loss).backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()
        scheduler.step()

        tot_loss = W_PHYSICS * loss_m_accum + W_BC * bc_loss.item()

        log_epoch = ((epoch + 1) % 100 == 0) or (epoch == 0) or ((epoch + 1) == ADAM_EPOCHS)
        if log_epoch:
            l2_p_val = compute_pressure_l2_error(model, data, chunk_size)
            if (epoch + 1) % 1000 == 0 or epoch == 0 or (epoch + 1) == ADAM_EPOCHS:
                log_print(
                    f"Adam Epoch {epoch} | Loss: {tot_loss:.6e} | Mom: {loss_m_accum:.6e} "
                    f"| BC PP: {bc_loss.item():.6e} | L2 P: {l2_p_val:.6e}"
                )
            history.update(epoch, tot_loss, loss_m_accum, 0.0, bc_loss.item(), l2_p_val)

        pbar.set_postfix({"L_tot": f"{tot_loss:.2e}", "Mom": f"{loss_m_accum:.2e}", "L2_p": f"{l2_p_val:.2e}" if log_epoch else "..."})
    pbar.close()

    # -------------------------------------------
    # STEP 5: L-BFGS (FP64)
    # -------------------------------------------
    if LBFGS_MAX_ITERS > 0:
        log_print(f"\n{'=' * 60}")
        log_print(f"FASE L-BFGS: Raffinamento Pressione - {LBFGS_MAX_ITERS} iterazioni (FP64)")
        log_print(f"{'=' * 60}")

        model.double()
        torch.set_default_dtype(torch.float64)
        cast_double(data)
        cast_double(derivs)

        xy_all = data["coords"]
        p_all = data["p"]
        var_w = data["var_weights"]
        bc_data = data["boundary_groups"]
        u_pinn = u_pinn.double()
        v_pinn = v_pinn.double()
        total_points = xy_all.shape[0]

        optimizer_lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=1.0,
            max_iter=LBFGS_MAX_ITERS,
            tolerance_grad=1e-12,
            tolerance_change=1e-16,
            history_size=300,
            line_search_fn="strong_wolfe",
        )

        iter_count = [0]

        def closure():
            optimizer_lbfgs.zero_grad()
            loss_m_accum = 0.0

            for i in range(0, total_points, chunk_size):
                xc = xy_all[i : i + chunk_size]
                w_chunk = xc.shape[0] / total_points

                xc_ph = xc.clone().requires_grad_(True)
                p_pred = model(xc_ph)

                grad_p = torch.autograd.grad(p_pred.sum(), xc_ph, create_graph=True, retain_graph=True)[0]
                p_x = grad_p[:, 0:1]
                p_y = grad_p[:, 1:2]

                ux = derivs["u_x"][i : i + chunk_size]
                uy = derivs["u_y"][i : i + chunk_size]
                uxx = derivs["u_xx"][i : i + chunk_size]
                uyy = derivs["u_yy"][i : i + chunk_size]
                vx = derivs["v_x"][i : i + chunk_size]
                vy = derivs["v_y"][i : i + chunk_size]
                vxx = derivs["v_xx"][i : i + chunk_size]
                vyy = derivs["v_yy"][i : i + chunk_size]
                txx_x_val = derivs["txx_x"][i : i + chunk_size]
                txy_y_val = derivs["txy_y"][i : i + chunk_size]
                txy_x_val = derivs["txy_x"][i : i + chunk_size]
                tyy_y_val = derivs["tyy_y"][i : i + chunk_size]

                u_val = u_pinn[i : i + chunk_size]
                v_val = v_pinn[i : i + chunk_size]

                f_u = Re * (u_val * (ux * s) + v_val * (uy * s)) + p_x * s - beta * ((uxx + uyy) * s**2) - ((txx_x_val + txy_y_val) * s)
                f_v = Re * (u_val * (vx * s) + v_val * (vy * s)) + p_y * s - beta * ((vxx + vyy) * s**2) - ((txy_x_val + tyy_y_val) * s)

                loss_m = (f_u**2 + f_v**2).mean() / 2.0
                chunk_loss = W_PHYSICS * loss_m * w_chunk
                chunk_loss.backward()
                loss_m_accum += loss_m.item() * w_chunk

            # BC: solo PressurePoint
            gd_pp = bc_data["PressurePoint"]
            x_bc_pp = gd_pp["xy"].clone().requires_grad_(True)
            p_bc_pp = model(x_bc_pp)
            bc_loss = torch.mean(((p_bc_pp - gd_pp["fields"]["p"]) ** 2) / var_w["p"])
            (W_BC * bc_loss).backward()

            tot_loss = W_PHYSICS * loss_m_accum + W_BC * bc_loss.item()

            iter_count[0] += 1
            if iter_count[0] % 100 == 0 or iter_count[0] == 1 or iter_count[0] == LBFGS_MAX_ITERS:
                l2_p_val = compute_pressure_l2_error(model, data, chunk_size)
                log_print(
                    f"L-BFGS Iter {iter_count[0]} | Loss: {tot_loss:.6e} | Mom: {loss_m_accum:.6e} "
                    f"| BC PP: {bc_loss.item():.6e} | L2 P: {l2_p_val:.6e}"
                )
                history.update(ADAM_EPOCHS + iter_count[0], tot_loss, loss_m_accum, 0.0, bc_loss.item(), l2_p_val)

            return torch.tensor(tot_loss, device=DEVICE, dtype=torch.float64)

        optimizer_lbfgs.step(closure)

    # -------------------------------------------
    # STEP 6: Report Finale
    # -------------------------------------------
    log_print("\n" + "=" * 60 + "\nREPORT FINALE\n" + "=" * 60)
    final_l2_p = compute_pressure_l2_error(model, data, chunk_size)
    log_print(f"  Errore L2 relativo pressione: {final_l2_p:.6f} ({final_l2_p * 100:.2f}%)")

    torch.save(
        {"model_state_dict": model.state_dict(), "history_losses": history.losses},
        OUTPUT_DIR / "final_pressure_model.pth",
    )

    history.plot(str(OUTPUT_DIR))
    log_print(f"\n[OK] Script terminato. Output: {OUTPUT_DIR}")
