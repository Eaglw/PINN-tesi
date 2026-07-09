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
if str(BASE_DIR.parent) not in sys.path:
    sys.path.append(str(BASE_DIR.parent))

# ============================================================================
# 1. SETUP ENVIRONMENT AND PYTORCH
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
# 2. GLOBAL CONFIGURATIONS & PHYSICAL PARAMETERS
# ============================================================================
# Try to auto-detect dataset path, fallback to environment variable or kaggle directory structure
if "KAGGLE_KERNEL_RUN_TYPE" in os.environ or not (BASE_DIR.parent / "COMSOL" / "4roll" / "4_roll_mill.csv").exists():
    import glob
    matches = glob.glob("/kaggle/input/**/4_roll_mill.csv", recursive=True)
    if matches:
        DATASET_PATH = Path(matches[0])
    else:
        # Fallback to current working directory or relative path search
        matches_local = glob.glob("**/4_roll_mill.csv", recursive=True)
        if matches_local:
            DATASET_PATH = Path(matches_local[0]).resolve()
        else:
            DATASET_PATH = BASE_DIR.parent / "COMSOL" / "4roll" / "4_roll_mill.csv"
else:
    DATASET_PATH = BASE_DIR.parent / "COMSOL" / "4roll" / "4_roll_mill.csv"

# Physical parameters (Ground Truth)
MU_S_TRUE = 0.1    # Solvent viscosity [Pa·s]
MU_P_TRUE = 0.9    # Polymeric viscosity [Pa·s]
LAM_TRUE = 0.05    # Relaxation time [s]
EPS_TRUE = 0.0     # PTT parameter
ALPHA_TRUE = 0.0   # Giesekus parameter
RHO = 1000.0       # Density [kg/m³]

MIN_MU_S = 1e-6
MIN_MU_P = 1e-6
MIN_LAM = 1e-6

GUESS_MULTIPLIER = 0.8
GUESS_MU_S = MU_S_TRUE * GUESS_MULTIPLIER
GUESS_MU_P = MU_P_TRUE * GUESS_MULTIPLIER
GUESS_LAM = LAM_TRUE * GUESS_MULTIPLIER
GUESS_EPS = 0.0
GUESS_ALPHA = 0.0

# Network architecture
HIDDEN_LAYERS = [128] * 8
ACTIVATION = nn.SiLU

# Training hyperparameters
ADAM_EPOCHS = 20000
LBFGS_MAX_ITERS = 10000
BASE_LR = 1e-3
ADAM_EPS = 1e-7
GRAD_CLIP_NORM = 5.0

# Loss weights
W_BC = 2.0
W_PHYSICS = 3.0
W_DATA = 0.0  # Set to 0.0 to remove pressure data supervision
VARIANCE_EPS = 1e-4

# Inject variables into src modules to allow load_data() and other utilities to run correctly
import src.utils
import src.physics
import src.train
import src.debug

for module in [src.debug, src.physics, src.train, src.utils]:
    for name, val in list(globals().items()):
        if name.isupper():
            module.__dict__[name] = val

from src.utils import load_data, plot_fields, plot_high_stress_regions

# Create output folder
layers_str = f"{len(HIDDEN_LAYERS)}x{HIDDEN_LAYERS[0]}"
config_name = f"{DATASET_PATH.stem}_PressureOnly_E{ADAM_EPOCHS}_L{LBFGS_MAX_ITERS}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
OUTPUT_DIR = BASE_DIR / "output_4rollmill" / config_name
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Custom logging
log_file_path = OUTPUT_DIR / "train_log.txt"
def log_print(*args, **kwargs):
    print(*args, **kwargs)
    with open(log_file_path, "a", encoding="utf-8") as f:
        print(*args, file=f, **kwargs)

# ===========================================================================
def precompute_comsol_derivatives(coords, u, v, txx, txy, tyy, device, chunk_size=5000, K=30):
    import scipy.spatial as spatial
    log_print(f"Precalculating spatial derivatives using locally scaled MLS (K={K})...")
    
    coords_np = coords.cpu().numpy()
    u_np = u.cpu().numpy()
    v_np = v.cpu().numpy()
    txx_np = txx.cpu().numpy()
    txy_np = txy.cpu().numpy()
    tyy_np = tyy.cpu().numpy()
    
    N = coords_np.shape[0]
    tree = spatial.cKDTree(coords_np)
    distances, indices = tree.query(coords_np, k=K, workers=-1)
    
    u_x = np.zeros((N, 1))
    u_y = np.zeros((N, 1))
    u_xx = np.zeros((N, 1))
    u_yy = np.zeros((N, 1))
    
    v_x = np.zeros((N, 1))
    v_y = np.zeros((N, 1))
    v_xx = np.zeros((N, 1))
    v_yy = np.zeros((N, 1))
    
    txx_x = np.zeros((N, 1))
    txx_y = np.zeros((N, 1))
    txy_x = np.zeros((N, 1))
    txy_y = np.zeros((N, 1))
    tyy_x = np.zeros((N, 1))
    tyy_y = np.zeros((N, 1))
    
    for i in range(N):
        x0 = coords_np[i]
        idx = indices[i]
        dist = distances[i]
        h = max(dist[-1], 1e-4)
        
        dxy = coords_np[idx] - x0
        dx_scaled = dxy[:, 0] / h
        dy_scaled = dxy[:, 1] / h
        
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
        XTWX = XTW @ X
        XTWX += np.eye(6) * 1e-12
        
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
            txx_y[i] = c_txx[2] / h
            txy_x[i] = c_txy[1] / h
            txy_y[i] = c_txy[2] / h
            tyy_x[i] = c_tyy[1] / h
            tyy_y[i] = c_tyy[2] / h
        except np.linalg.LinAlgError:
            pass
            
    log_print("MLS derivative calculation completed successfully!")
    return {
        "u_x": torch.tensor(u_x, dtype=coords.dtype, device=device),
        "u_y": torch.tensor(u_y, dtype=coords.dtype, device=device),
        "u_xx": torch.tensor(u_xx, dtype=coords.dtype, device=device),
        "u_yy": torch.tensor(u_yy, dtype=coords.dtype, device=device),
        "v_x": torch.tensor(v_x, dtype=coords.dtype, device=device),
        "v_y": torch.tensor(v_y, dtype=coords.dtype, device=device),
        "v_xx": torch.tensor(v_xx, dtype=coords.dtype, device=device),
        "v_yy": torch.tensor(v_yy, dtype=coords.dtype, device=device),
        "txx_x": torch.tensor(txx_x, dtype=coords.dtype, device=device),
        "txx_y": torch.tensor(txx_y, dtype=coords.dtype, device=device),
        "txy_x": torch.tensor(txy_x, dtype=coords.dtype, device=device),
        "txy_y": torch.tensor(txy_y, dtype=coords.dtype, device=device),
        "tyy_x": torch.tensor(tyy_x, dtype=coords.dtype, device=device),
        "tyy_y": torch.tensor(tyy_y, dtype=coords.dtype, device=device)
    }

# ============================================================================
# 4. DEFINE MODEL AND CONFIGS
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
        if activation_name == 'silu':
            activation_name = 'relu'
        gain = nn.init.calculate_gain(activation_name)
        nn.init.xavier_normal_(m.weight, gain=gain)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def initialize_last_layer_zero(model):
    last_linear = None
    for module in model.modules():
        if isinstance(module, nn.Linear):
            last_linear = module
    if last_linear is not None:
        nn.init.zeros_(last_linear.weight)
        if last_linear.bias is not None:
            nn.init.zeros_(last_linear.bias)

# Helpers for casting variables recursively
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

# L2 metric calculation
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

# Tracker History Class
class PressureHistory:
    def __init__(self):
        self.epochs = []
        self.losses = {
            "total": [],
            "momentum": [],
            "data_p": [],
            "bc_p": [],
            "l2_p": []
        }

    def update(self, epoch, total, momentum, data_p, bc_p, l2_p):
        self.epochs.append(epoch)
        self.losses["total"].append(total)
        self.losses["momentum"].append(momentum)
        self.losses["data_p"].append(data_p)
        self.losses["bc_p"].append(bc_p)
        self.losses["l2_p"].append(l2_p)

    def plot(self, output_dir):
        plt.figure(figsize=(10, 5))
        plt.plot(self.epochs, self.losses["total"], label="Total Loss", color="black", linewidth=2)
        plt.plot(self.epochs, self.losses["momentum"], label="Momentum Loss", color="red", alpha=0.8)
        plt.plot(self.epochs, self.losses["data_p"], label="Data Loss (P)", color="blue", alpha=0.8)
        plt.plot(self.epochs, self.losses["bc_p"], label="BC Loss (P)", color="green", alpha=0.8)
        plt.yscale("log")
        plt.xlabel("Epoch / Iteration")
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
        plt.xlabel("Epoch / Iteration")
        plt.ylabel("L2 Relative Error")
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
    log_print(f"Saving output to: {OUTPUT_DIR}\n")
    log_print("=" * 60)

    # 1. Load Data
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

    # Adimensional scaling references
    mu_tot = MU_S_TRUE + MU_P_TRUE
    Re = RHO * data["U_ref"] * data["H"] / mu_tot
    beta = MU_S_TRUE / mu_tot
    s = data["H"] / data["H_coord"]

    log_print(f"Computed scaling values: Re = {Re:.4f}, beta = {beta:.4f}, s = {s:.4f}")

    # 2. Precompute spatial derivatives
    derivs = precompute_comsol_derivatives(
        xy_all, u_all, v_all, txx_all, txy_all, tyy_all, DEVICE, chunk_size=5000, K=50
    )

    # 3. Initialize model
    model = PressureModel(p_scale=data["p_scale"]).to(DEVICE)
    model.apply(lambda m: init_weights_xavier(m, activation_name="silu"))
    # DO NOT call initialize_last_layer_zero to avoid Vanishing Gradient Cascade
    # initialize_last_layer_zero(model.model_p)

    history = PressureHistory()

    # 4. Phase 1: Adam optimization (FP32)
    optimizer = torch.optim.Adam(model.parameters(), lr=BASE_LR, eps=ADAM_EPS)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ADAM_EPOCHS, eta_min=1e-6)

    chunk_size = 5000 if DEVICE.type == "cuda" else total_points
    pbar = tqdm(range(ADAM_EPOCHS), desc="Adam (Pressure-Only)", mininterval=2.0)
    for epoch in pbar:
        model.train()
        optimizer.zero_grad(set_to_none=True)

        loss_m_accum = 0.0
        loss_d_p_accum = 0.0

        for i in range(0, total_points, chunk_size):
            xc = xy_all[i : i + chunk_size]
            p_true = p_all[i : i + chunk_size]
            w_chunk = xc.shape[0] / total_points

            xc_ph = xc.clone().requires_grad_(True)
            p_pred = model(xc_ph)

            # Spatial derivative of predicted pressure
            grad_p = torch.autograd.grad(p_pred.sum(), xc_ph, create_graph=True, retain_graph=True)[0]
            p_x = grad_p[:, 0:1]
            p_y = grad_p[:, 1:2]

            # Fetch precomputed derivatives
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

            # Momentum residuals with proper scaling
            f_u = Re * (u_val * (ux * s) + v_val * (uy * s)) + p_x * s - beta * ((uxx + uyy) * s**2) - ((txx_x_val + txy_y_val) * s)
            f_v = Re * (u_val * (vx * s) + v_val * (vy * s)) + p_y * s - beta * ((vxx + vyy) * s**2) - ((txy_x_val + tyy_y_val) * s)

            loss_m = (f_u**2 + f_v**2).mean() / 2.0
            loss_d_p = torch.mean(((p_pred - p_true) ** 2) / var_w["p"])

            chunk_loss = (W_PHYSICS * loss_m + W_DATA * loss_d_p) * w_chunk
            chunk_loss.backward()

            loss_m_accum += loss_m.item() * w_chunk
            loss_d_p_accum += loss_d_p.item() * w_chunk

        # Boundary condition on Walls
        gd = bc_data["Walls"]
        x_bc = gd["xy"].clone().requires_grad_(True)
        p_bc = model(x_bc)
        bc_loss = torch.mean(((p_bc - gd["fields"]["p"]) ** 2) / var_w["p"])
        (W_BC * bc_loss).backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()
        scheduler.step()

        tot_loss = W_PHYSICS * loss_m_accum + W_DATA * loss_d_p_accum + W_BC * bc_loss.item()

        log_epoch = ((epoch + 1) % 100 == 0) or (epoch == 0) or ((epoch + 1) == ADAM_EPOCHS)
        if log_epoch:
            l2_p_val = compute_pressure_l2_error(model, data, chunk_size)
            if (epoch + 1) % 1000 == 0 or epoch == 0 or ((epoch + 1) == ADAM_EPOCHS):
                log_print(f"Adam Epoch {epoch} | Loss: {tot_loss:.6e} | Mom: {loss_m_accum:.6e} | Data P: {loss_d_p_accum:.6e} | BC P: {bc_loss.item():.6e} | L2 P: {l2_p_val:.6e}")
            history.update(epoch, tot_loss, loss_m_accum, loss_d_p_accum, bc_loss.item(), l2_p_val)
            
        pbar.set_postfix({"L_tot": f"{tot_loss:.2e}"})
    pbar.close()

    # 5. Phase 2: L-BFGS optimization (FP64)
    if LBFGS_MAX_ITERS > 0:
        log_print(f"\n{'=' * 60}\nL-BFGS Phase: Refinement of Pressure Model - {LBFGS_MAX_ITERS} iterations (FP64)\n{'=' * 60}")
        
        # Convert all to FP64
        model.double()
        torch.set_default_dtype(torch.float64)
        cast_double(data)
        cast_double(derivs)
        
        xy_all = data["coords"]
        p_all = data["p"]
        u_all = data["u"]
        v_all = data["v"]
        var_w = data["var_weights"]
        bc_data = data["boundary_groups"]
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
            loss_d_p_accum = 0.0
            
            for i in range(0, total_points, chunk_size):
                xc = xy_all[i : i + chunk_size]
                p_true = p_all[i : i + chunk_size]
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
                
                # Momentum residuals with proper scaling
                f_u = Re * (u_val * (ux * s) + v_val * (uy * s)) + p_x * s - beta * ((uxx + uyy) * s**2) - ((txx_x_val + txy_y_val) * s)
                f_v = Re * (u_val * (vx * s) + v_val * (vy * s)) + p_y * s - beta * ((vxx + vyy) * s**2) - ((txy_x_val + tyy_y_val) * s)
                
                loss_m = (f_u**2 + f_v**2).mean() / 2.0
                loss_d_p = torch.mean(((p_pred - p_true) ** 2) / var_w["p"])
                
                chunk_loss = (W_PHYSICS * loss_m + W_DATA * loss_d_p) * w_chunk
                chunk_loss.backward()
                
                loss_m_accum += loss_m.item() * w_chunk
                loss_d_p_accum += loss_d_p.item() * w_chunk
                
            # Boundary condition on Walls
            gd = bc_data["Walls"]
            x_bc = gd["xy"].clone().requires_grad_(True)
            p_bc = model(x_bc)
            bc_loss = torch.mean(((p_bc - gd["fields"]["p"]) ** 2) / var_w["p"])
            (W_BC * bc_loss).backward()
            
            tot_loss = W_PHYSICS * loss_m_accum + W_DATA * loss_d_p_accum + W_BC * bc_loss.item()
            
            iter_count[0] += 1
            if iter_count[0] % 100 == 0 or iter_count[0] == 1 or iter_count[0] == LBFGS_MAX_ITERS:
                l2_p_val = compute_pressure_l2_error(model, data, chunk_size)
                log_print(f"L-BFGS Iter {iter_count[0]} | Loss: {tot_loss:.6e} | Mom: {loss_m_accum:.6e} | Data P: {loss_d_p_accum:.6e} | BC P: {bc_loss.item():.6e} | L2 P: {l2_p_val:.6e}")
                history.update(ADAM_EPOCHS + iter_count[0], tot_loss, loss_m_accum, loss_d_p_accum, bc_loss.item(), l2_p_val)
                
            return torch.tensor(tot_loss, device=DEVICE, dtype=torch.float64)

        optimizer_lbfgs.step(closure)

    # 6. Report Final Results
    log_print("\n" + "=" * 60 + "\nFINAL PERFORMANCE REPORT\n" + "=" * 60)
    final_l2_p = compute_pressure_l2_error(model, data, chunk_size)
    log_print(f"  Final relative L2 error for Pressure: {final_l2_p:.6f} ({final_l2_p*100:.2f}%)")

    # Save model and plots
    log_print(f"\nSaving final model checkpoint and plots to {OUTPUT_DIR}...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'history_losses': history.losses,
    }, OUTPUT_DIR / "final_pressure_model.pth")

    history.plot(str(OUTPUT_DIR))

    # Evaluate all predictions at coordinates to pass to plot_fields
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

    # Cast back data to CPU or float if plot_fields needs it
    cast_float(data)
    cast_float(predictions)

    plot_fields(predictions, data, save_path=f"{OUTPUT_DIR}/global_fields.png")
    plot_high_stress_regions(predictions, data, save_path=f"{OUTPUT_DIR}/high_stress.png")

    log_print("\n[OK] Script run finished successfully!")
