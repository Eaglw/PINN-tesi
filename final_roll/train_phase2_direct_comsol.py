"""
train_phase2_direct_comsol.py
=============================================================================
Addestramento Standalone della Pressione da dati COMSOL (Problema Diretto)
senza addestrare preventivamente reti per cinematica e stress (Zero Fase 1).

Paradigma storico convalidato (train_4roll_kaggle.py - commit b4f5547):
1. MLS di 2° grado con coordinate locali rigorosamente scalate tra [-1, 1].
2. Scaling adimensionale basato su mu_tot = 1.0 (Re = 0.0417, beta = 0.10).
3. Gradient clipping rigido a 5.0 (essenziale contro outlier derivativi).
4. W_DATA = 0.0 (nessuna supervisione sui nodi interni di pressione).
5. Singolo punto Dirichlet di ancoraggio (PressurePoint).
6. Ottimizzazione Adam (FP32) seguita da L-BFGS (FP64, history=300).
=============================================================================
"""

import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import scipy.spatial as spatial

# Assicura che la directory final_roll sia nel sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(BASE_DIR.parent) not in sys.path:
    sys.path.insert(0, str(BASE_DIR.parent))

# ============================================================================
# 1. SETUP AMBIENTE E HARDWARE
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
# 2. CONFIGURAZIONI GLOBALI E PARAMETRI FISICI
# ============================================================================
# Rilevamento automatico percorso dataset COMSOL (Kaggle o Locale)
import glob
matches_kaggle = glob.glob("/kaggle/input/**/4_roll_mill.csv", recursive=True)
if matches_kaggle:
    DATASET_PATH = Path(matches_kaggle[0])
elif (BASE_DIR.parent / "COMSOL" / "4roll" / "4_roll_mill.csv").exists():
    DATASET_PATH = BASE_DIR.parent / "COMSOL" / "4roll" / "4_roll_mill.csv"
else:
    matches_local = glob.glob("**/4_roll_mill.csv", recursive=True)
    DATASET_PATH = Path(matches_local[0]).resolve() if matches_local else (BASE_DIR.parent / "COMSOL" / "4roll" / "4_roll_mill.csv")

# Parametri fisici di riferimento (Ground Truth)
MU_S_TRUE = 0.1       # Viscosità solvente [Pa·s]
MU_P_TRUE = 0.9       # Viscosità polimerica [Pa·s]
LAM_TRUE = 0.05       # Tempo di rilassamento [s]
RHO = 1000.0          # Densità [kg/m³]
MU_TOT = MU_S_TRUE + MU_P_TRUE  # 1.0 Pa·s

# Architettura Network
HIDDEN_LAYERS = [128] * 8
ACTIVATION = nn.SiLU

# Iperparametri Training
ADAM_EPOCHS = 20000
LBFGS_MAX_ITERS = 2000
BASE_LR = 1e-3
ADAM_EPS = 1e-7
GRAD_CLIP_NORM = 5.0  # Rigido per evitare picchi numerici

# Pesi Funzione di Loss
W_PHYSICS = 3.0       # Peso della Momentum PDE
W_BC = 2.0            # Peso ancoraggio singolo PressurePoint
W_DATA = 0.0          # ZERO supervisione sui valori interni di pressione
VARIANCE_EPS = 1e-4

# MLS
MLS_K = 25            # 25 vicini per MLS di 2° grado (robusto)

# Iniezione parametri per i moduli src
import src.utils
import src.physics
import src.train
import src.debug
import builtins

for module in [src.debug, src.physics, src.train, src.utils]:
    for name, val in list(globals().items()):
        if name.isupper():
            module.__dict__[name] = val
            builtins.__dict__[name] = val

from src.utils import load_data, plot_fields, plot_high_stress_regions

# Directory di output
config_name = f"[{datetime.now().strftime('%Y-%m-%d_%H-%M')}][DIR][PHASE2_MLS_SCALED][Ph2_{ADAM_EPOCHS//1000}k+{LBFGS_MAX_ITERS//1000}k]"
OUTPUT_DIR = BASE_DIR / "output_4rollmill" / config_name
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

log_file_path = OUTPUT_DIR / "train_log.txt"
def log_print(*args, **kwargs):
    print(*args, **kwargs)
    with open(log_file_path, "a", encoding="utf-8") as f:
        print(*args, file=f, **kwargs)

# ============================================================================
# 3. MOVING LEAST SQUARES (MLS) CON SCALING LOCALE [-1, 1]
# ============================================================================
def precompute_comsol_derivatives_scaled_mls(coords, u, v, txx, txy, tyy, device, K=25, cache_path=None):
    if cache_path and cache_path.exists():
        log_print(f"\n[Cache] Caricamento derivate MLS 2° grado precalcolate da: {cache_path}")
        cache = torch.load(cache_path, map_location=device)
        log_print("  Derivate MLS caricate con successo!")
        return cache

    log_print(f"\n[MLS] Calcolo derivate spaziali con Moving Least Squares di 2° grado (K={K}, scaling locale [-1, 1])...")
    coords_np = coords.cpu().numpy()
    u_np = u.cpu().numpy()
    v_np = v.cpu().numpy()
    txx_np = txx.cpu().numpy()
    txy_np = txy.cpu().numpy()
    tyy_np = tyy.cpu().numpy()

    N = coords_np.shape[0]
    tree = spatial.cKDTree(coords_np)
    distances, indices = tree.query(coords_np, k=K, workers=-1)

    u_x = np.zeros((N, 1), dtype=np.float32)
    u_y = np.zeros((N, 1), dtype=np.float32)
    u_xx = np.zeros((N, 1), dtype=np.float32)
    u_yy = np.zeros((N, 1), dtype=np.float32)

    v_x = np.zeros((N, 1), dtype=np.float32)
    v_y = np.zeros((N, 1), dtype=np.float32)
    v_xx = np.zeros((N, 1), dtype=np.float32)
    v_yy = np.zeros((N, 1), dtype=np.float32)

    txx_x = np.zeros((N, 1), dtype=np.float32)
    txy_y = np.zeros((N, 1), dtype=np.float32)
    txy_x = np.zeros((N, 1), dtype=np.float32)
    tyy_y = np.zeros((N, 1), dtype=np.float32)

    for i in range(N):
        x0 = coords_np[i]
        idx = indices[i]
        dist = distances[i]
        h = max(dist[-1], 1e-4)

        dxy = coords_np[idx] - x0
        dx_scaled = dxy[:, 0] / h
        dy_scaled = dxy[:, 1] / h

        # Base polinomiale di 2° grado con coordinate adimensionali locali [-1, 1]
        X = np.column_stack([
            np.ones(K),
            dx_scaled,
            dy_scaled,
            0.5 * dx_scaled**2,
            0.5 * dy_scaled**2,
            dx_scaled * dy_scaled
        ])

        w = np.exp(- (dist**2) / (h**2))
        W = np.diag(w)

        XTW = X.T @ W
        XTWX = XTW @ X + np.eye(6) * 1e-12

        try:
            inv_XTWX = np.linalg.inv(XTWX)
            c_u = inv_XTWX @ XTW @ u_np[idx]
            c_v = inv_XTWX @ XTW @ v_np[idx]
            c_txx = inv_XTWX @ XTW @ txx_np[idx]
            c_txy = inv_XTWX @ XTW @ txy_np[idx]
            c_tyy = inv_XTWX @ XTW @ tyy_np[idx]

            u_x[i] = c_u[1] / h
            u_y[i] = c_u[2] / h
            u_xx[i] = c_u[3] / (h**2)
            u_yy[i] = c_u[4] / (h**2)

            v_x[i] = c_v[1] / h
            v_y[i] = c_v[2] / h
            v_xx[i] = c_v[3] / (h**2)
            v_yy[i] = c_v[4] / (h**2)

            txx_x[i] = c_txx[1] / h
            txy_y[i] = c_txy[2] / h
            txy_x[i] = c_txy[1] / h
            tyy_y[i] = c_tyy[2] / h
        except np.linalg.LinAlgError:
            pass

    log_print("[MLS] Calcolo derivate completato con successo!")
    cache = {
        "u_x": torch.tensor(u_x, dtype=torch.float32, device=device),
        "u_y": torch.tensor(u_y, dtype=torch.float32, device=device),
        "u_xx": torch.tensor(u_xx, dtype=torch.float32, device=device),
        "u_yy": torch.tensor(u_yy, dtype=torch.float32, device=device),
        "v_x": torch.tensor(v_x, dtype=torch.float32, device=device),
        "v_y": torch.tensor(v_y, dtype=torch.float32, device=device),
        "v_xx": torch.tensor(v_xx, dtype=torch.float32, device=device),
        "v_yy": torch.tensor(v_yy, dtype=torch.float32, device=device),
        "txx_x": torch.tensor(txx_x, dtype=torch.float32, device=device),
        "txy_y": torch.tensor(txy_y, dtype=torch.float32, device=device),
        "txy_x": torch.tensor(txy_x, dtype=torch.float32, device=device),
        "tyy_y": torch.tensor(tyy_y, dtype=torch.float32, device=device),
    }
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache, cache_path)
    return cache

# ============================================================================
# 4. MODELLO NEURALE PRESSIONE
# ============================================================================
class FCN(nn.Module):
    def __init__(self, n_input, n_output, hidden_layers, activation=nn.SiLU):
        super().__init__()
        layers_sizes = [n_input] + hidden_layers + [n_output]
        layers = []
        for i in range(len(layers_sizes) - 1):
            layers.append(nn.Linear(layers_sizes[i], layers_sizes[i + 1]))
            if i < len(layers_sizes) - 2:
                layers.append(activation())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class PressureModel(nn.Module):
    def __init__(self, p_scale=1.0):
        super().__init__()
        self.model_p = FCN(2, 1, HIDDEN_LAYERS, ACTIVATION)
        self.p_scale = p_scale

    def forward(self, x):
        return self.model_p(x) * self.p_scale

def init_weights_xavier(m, activation_name="silu"):
    if isinstance(m, nn.Linear):
        activation_name = activation_name.lower()
        if activation_name == "silu":
            activation_name = "relu"
        gain = nn.init.calculate_gain(activation_name)
        nn.init.xavier_normal_(m.weight, gain=gain)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def cast_double(d):
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            d[k] = v.double()
        elif isinstance(v, dict):
            cast_double(v)

def cast_float(d):
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            d[k] = v.float()
        elif isinstance(v, dict):
            cast_float(v)

def compute_pressure_l2_error(model, data, chunk_size=5000):
    model.eval()
    _dtype = next(model.parameters()).dtype
    xy_all = data["coords"].to(_dtype)
    p_exact = data["p"].to(_dtype)
    p_pred_list = []
    with torch.no_grad():
        for i in range(0, xy_all.shape[0], chunk_size):
            xc = xy_all[i : i + chunk_size]
            p_pred_list.append(model(xc))
    p_pred = torch.cat(p_pred_list, dim=0)

    p_flat = p_pred.view(-1)
    e_flat = p_exact.view(-1)
    norm_e = torch.norm(e_flat, 2)
    if norm_e > 1e-10:
        return (torch.norm(p_flat - e_flat, 2) / norm_e).item()
    return 0.0

class PressureHistory:
    def __init__(self):
        self.epochs = []
        self.losses = {
            "total": [],
            "momentum": [],
            "bc_p": [],
            "l2_p": []
        }

    def update(self, epoch, total, momentum, bc_p, l2_p):
        self.epochs.append(epoch)
        self.losses["total"].append(total)
        self.losses["momentum"].append(momentum)
        self.losses["bc_p"].append(bc_p)
        self.losses["l2_p"].append(l2_p)

    def plot(self, output_dir):
        plt.figure(figsize=(10, 5))
        plt.plot(self.epochs, self.losses["total"], label="Total Loss", color="black", linewidth=2)
        plt.plot(self.epochs, self.losses["momentum"], label="Momentum Loss", color="red", alpha=0.8)
        plt.plot(self.epochs, self.losses["bc_p"], label="PressurePoint BC Loss", color="green", alpha=0.8)
        plt.yscale("log")
        plt.xlabel("Epoca / Iterazione")
        plt.ylabel("Loss")
        plt.title("Pressure-Only Training Loss")
        plt.legend()
        plt.grid(True, ls="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/loss_history.png", dpi=150)
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(self.epochs, self.losses["l2_p"], label="L2 Relative Error (P)", color="purple")
        plt.yscale("log")
        plt.xlabel("Epoca / Iterazione")
        plt.ylabel("Errore L2 Relativo Pressione")
        plt.title("Pressure L2 Relative Error History")
        plt.legend()
        plt.grid(True, ls="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/l2_errors_history.png", dpi=150)
        plt.close()

# ============================================================================
# 5. MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    log_print(f"Device: {DEVICE} | Dtype: {torch.get_default_dtype()}")
    log_print(f"Dataset: {DATASET_PATH}\n")
    log_print(f"Directory Risultati: {OUTPUT_DIR}\n")
    log_print("=" * 70)

    # 1. Caricamento Dati
    data = load_data()
    xy_all = data["coords"]
    u_all = data["u"]
    v_all = data["v"]
    p_all = data["p"]
    txx_all = data["tau_xx"]
    txy_all = data["tau_xy"]
    tyy_all = data["tau_yy"]
    var_w = data["var_weights"]
    bc_data = data["boundary_groups"]
    total_points = xy_all.shape[0]

    # Riferimenti adimensionali coerenti con setup originale (mu_tot = 1.0)
    Re = RHO * data["U_ref"] * data["H"] / MU_TOT
    beta = MU_S_TRUE / MU_TOT
    s = data["H"] / data["H_coord"]

    log_print(f"Scale adimensionali: Re = {Re:.4f}, beta = {beta:.4f}, s = {s:.4f}")
    log_print(f"Pesi Loss: W_PHYSICS = {W_PHYSICS}, W_BC = {W_BC}, W_DATA = {W_DATA}")
    log_print(f"Vincolo Dirichlet: SOLO singolo nodo in {bc_data['PressurePoint']['xy'][0].cpu().numpy()}")

    # 2. Precalcolo derivate MLS di 2° grado scalate
    cache_mls = DATASET_PATH.parent / "comsol_derivatives_mls_deg2.pt"
    derivs = precompute_comsol_derivatives_scaled_mls(
        xy_all, u_all, v_all, txx_all, txy_all, tyy_all, DEVICE, K=MLS_K, cache_path=cache_mls
    )

    # 3. Inizializzazione Modello Pressione
    model = PressureModel(p_scale=data["p_scale"]).to(DEVICE)
    model.apply(lambda m: init_weights_xavier(m, activation_name="silu"))

    history = PressureHistory()

    # 4. Fase 1: Ottimizzazione Adam (FP32)
    optimizer = torch.optim.Adam(model.parameters(), lr=BASE_LR, eps=ADAM_EPS)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ADAM_EPOCHS, eta_min=1e-6)

    chunk_size = 5000 if DEVICE.type == "cuda" else total_points
    pbar = tqdm(range(ADAM_EPOCHS), desc="Adam (Pressure-Only)", mininterval=2.0)

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

            u_val = u_all[i : i + chunk_size]
            v_val = v_all[i : i + chunk_size]

            # Momentum equations con derivate scalate
            f_u = Re * (u_val * (ux * s) + v_val * (uy * s)) + p_x * s - beta * ((uxx + uyy) * s**2) - ((txx_x_val + txy_y_val) * s)
            f_v = Re * (u_val * (vx * s) + v_val * (vy * s)) + p_y * s - beta * ((vxx + vyy) * s**2) - ((txy_x_val + tyy_y_val) * s)

            loss_m = (f_u**2 + f_v**2).mean() / 2.0

            chunk_loss = (W_PHYSICS * loss_m) * w_chunk
            chunk_loss.backward()

            loss_m_accum += loss_m.item() * w_chunk

        # Condizione al contorno sul SINGOLO PressurePoint
        gd = bc_data["PressurePoint"]
        x_bc = gd["xy"].clone().requires_grad_(True)
        p_bc = model(x_bc)
        bc_loss = torch.mean(((p_bc - gd["fields"]["p"]) ** 2) / var_w["p"])
        (W_BC * bc_loss).backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()
        scheduler.step()

        tot_loss = W_PHYSICS * loss_m_accum + W_BC * bc_loss.item()

        log_epoch = ((epoch + 1) % 100 == 0) or (epoch == 0) or ((epoch + 1) == ADAM_EPOCHS)
        if log_epoch:
            l2_p_val = compute_pressure_l2_error(model, data, chunk_size)
            if (epoch + 1) % 1000 == 0 or epoch == 0 or ((epoch + 1) == ADAM_EPOCHS):
                log_print(f"Adam Epoch {epoch+1:5d}/{ADAM_EPOCHS} | Loss: {tot_loss:.4e} | Mom: {loss_m_accum:.4e} | BC P: {bc_loss.item():.4e} | L2 P: {l2_p_val:.4e} ({l2_p_val*100:.2f}%)")
            history.update(epoch + 1, tot_loss, loss_m_accum, bc_loss.item(), l2_p_val)

        pbar.set_postfix({"L_tot": f"{tot_loss:.2e}", "L2_p": f"{l2_p_val*100:.1f}%"})
    pbar.close()

    # 5. Fase 2: Raffinamento L-BFGS (FP64)
    if LBFGS_MAX_ITERS > 0:
        log_print(f"\n{'=' * 70}\nFASE L-BFGS (FP64): {LBFGS_MAX_ITERS} iterazioni massime (History=300, Strong Wolfe)\n{'=' * 70}")

        model.double()
        torch.set_default_dtype(torch.float64)
        cast_double(data)
        cast_double(derivs)

        xy_all = data["coords"]
        u_all = data["u"]
        v_all = data["v"]
        var_w = data["var_weights"]
        bc_data = data["boundary_groups"]

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

                u_val = u_all[i : i + chunk_size]
                v_val = v_all[i : i + chunk_size]

                f_u = Re * (u_val * (ux * s) + v_val * (uy * s)) + p_x * s - beta * ((uxx + uyy) * s**2) - ((txx_x_val + txy_y_val) * s)
                f_v = Re * (u_val * (vx * s) + v_val * (vy * s)) + p_y * s - beta * ((vxx + vyy) * s**2) - ((txy_x_val + tyy_y_val) * s)

                loss_m = (f_u**2 + f_v**2).mean() / 2.0
                chunk_loss = (W_PHYSICS * loss_m) * w_chunk
                chunk_loss.backward()

                loss_m_accum += loss_m.item() * w_chunk

            gd = bc_data["PressurePoint"]
            x_bc = gd["xy"].clone().requires_grad_(True)
            p_bc = model(x_bc)
            bc_loss = torch.mean(((p_bc - gd["fields"]["p"]) ** 2) / var_w["p"])
            (W_BC * bc_loss).backward()

            tot_loss = W_PHYSICS * loss_m_accum + W_BC * bc_loss.item()

            iter_count[0] += 1
            if iter_count[0] % 100 == 0 or iter_count[0] == 1 or iter_count[0] == LBFGS_MAX_ITERS:
                l2_p_val = compute_pressure_l2_error(model, data, chunk_size)
                log_print(f"L-BFGS Iter {iter_count[0]:4d}/{LBFGS_MAX_ITERS} | Loss: {tot_loss:.4e} | Mom: {loss_m_accum:.4e} | BC P: {bc_loss.item():.4e} | L2 P: {l2_p_val:.4e} ({l2_p_val*100:.2f}%)")
                history.update(ADAM_EPOCHS + iter_count[0], tot_loss, loss_m_accum, bc_loss.item(), l2_p_val)

            return torch.tensor(tot_loss, device=DEVICE, dtype=torch.float64)

        optimizer_lbfgs.step(closure)

    # 6. Report Finale e Salvataggio
    log_print("\n" + "=" * 70 + "\nREPORT PRESTAZIONI FINALE\n" + "=" * 70)
    final_l2_p = compute_pressure_l2_error(model, data, chunk_size)
    log_print(f"  Errore L2 Relativo Finale Pressione: {final_l2_p:.6f} ({final_l2_p*100:.2f}%)")

    torch.save({
        "model_state_dict": model.state_dict(),
        "history_losses": history.losses,
    }, OUTPUT_DIR / "final_pressure_model.pth")

    history.plot(str(OUTPUT_DIR))

    # Generazione mappe dei campi finali
    model.eval()
    _dtype = next(model.parameters()).dtype
    p_pred_list = []
    with torch.no_grad():
        for i in range(0, total_points, chunk_size):
            xc = xy_all[i : i + chunk_size].to(_dtype)
            p_pred_list.append(model(xc))
    p_pred = torch.cat(p_pred_list, dim=0)

    predictions = {
        "u": u_all.to(_dtype),
        "v": v_all.to(_dtype),
        "p": p_pred,
        "tau_xx": txx_all.to(_dtype),
        "tau_xy": txy_all.to(_dtype),
        "tau_yy": tyy_all.to(_dtype)
    }
    cast_float(data)
    cast_float(predictions)

    plot_fields(predictions, data, save_path=f"{OUTPUT_DIR}/global_fields.png")
    plot_high_stress_regions(predictions, data, save_path=f"{OUTPUT_DIR}/high_stress.png")

    log_print(f"\n[OK] Risultati e grafici salvati in: {OUTPUT_DIR}")
