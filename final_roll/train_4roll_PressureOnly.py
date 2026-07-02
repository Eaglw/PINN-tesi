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
from tqdm import tqdm

# Import dai moduli src
from src.debug import test_random_points, debug_physics_magnitudes
from src.physics import Physics, evaluate_final_losses, compute_l2_errors
from src.train import CombinedModel, initialize_last_layer_zero, init_weights_xavier, SimpleHistory
from src.utils import load_data, plot_fields, plot_high_stress_regions, convert_to_fp64
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
SEED = 123
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# 2. COSTANTI E CONFIGURAZIONI GLOBALI
# ============================================================================
EXPORT_TO_OBSIDIAN = True
STAGED_TRAINING = False
INVERSE_PROBLEM = False
DEBUG_MODE = False

RESUME_CHECKPOINT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "checkpoint_psi+tau_100k.pth")

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR.parent / "COMSOL" / "4roll" / "4_roll_mill.csv"

MU_S_TRUE = 0.1
MU_P_TRUE = 0.9
LAM_TRUE = 0.05
EPS_TRUE = 0.0
ALPHA_TRUE = 0.0
RHO = 1000.0

MIN_MU_S = 1e-6
MIN_MU_P = 1e-6
MIN_LAM = 1e-6

GUESS_MULTIPLIER = 0.8
GUESS_MU_S = MU_S_TRUE * GUESS_MULTIPLIER
GUESS_MU_P = MU_P_TRUE * GUESS_MULTIPLIER
GUESS_LAM = LAM_TRUE * GUESS_MULTIPLIER
GUESS_EPS = 0.0
GUESS_ALPHA = 0.0

HIDDEN_LAYERS = [128] * 8
ACTIVATION = nn.SiLU

# Iperparametri specifici per la run Pressure-Only
ADAM_EPOCHS = 50000
LBFGS_ITERS = 5000
BASE_LR = 1e-3
ADAM_EPS = 1e-7
GRAD_CLIP_NORM = 1000.0

W_BC = 2.0
W_PHYSICS = 3.0
W_DATA = 0.0  # Nessuna loss dati per velocità
W_MOMENTUM = 1.0
W_CONSTITUTIVE = 0.0  # Nessuna loss costitutiva
VARIANCE_EPS = 1e-4

# ============================================================================
# 3. INIZIALIZZAZIONE OUTPUT
# ============================================================================
layers_str = f"{len(HIDDEN_LAYERS)}x{HIDDEN_LAYERS[0]}"
config_name = f"{DATASET_PATH.stem}_L{layers_str}_PressureOnly_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

OUTPUT_DIR = BASE_DIR / "output_4rollmill" / config_name
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

global_log_path = OUTPUT_DIR / "train_log.txt"

# Iniezione dinamica dei parametri globali nei moduli di src per risolvere la mancanza di config
for module in [src.debug, src.physics, src.train, src.utils]:
    for name, val in list(globals().items()):
        if name.isupper():
            module.__dict__[name] = val

def launch_tensorboard_server(log_dir):
    import subprocess
    import webbrowser
    import time
    import socket
    import sys
    from pathlib import Path

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 6006))
        s.close()
        port_free = True
    except socket.error:
        port_free = False

    if port_free:
        print("\n[TensorBoard] Avvio del server TensorBoard in corso...")
        venv_bin = Path(sys.executable).parent
        tb_executable = venv_bin / "tensorboard"
        if not tb_executable.exists():
            tb_executable = venv_bin / "tensorboard.exe"
            
        cmd = [
            str(tb_executable) if tb_executable.exists() else "tensorboard",
            "--logdir", str(log_dir),
            "--port", "6006"
        ]
        
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)
        time.sleep(2)
        print("[TensorBoard] Server avviato sulla porta 6006.")
    else:
        print("\n[TensorBoard] Server già attivo o porta 6006 occupata. Utilizzo istanza esistente.")

    print("[TensorBoard] Apertura del browser su http://localhost:6006 ...")
    webbrowser.open("http://localhost:6006")


def precompute_momentum_constants(model, physics, data, chunk_size=5000):
    """
    Precalcola le parti dell'equazione di momentum che dipendono da u, v, tau (tutte congelate).
    """
    model.eval()
    xy_all = data["coords"]
    
    const_u_list = []
    const_v_list = []
    
    with torch.set_grad_enabled(True):
        for i in range(0, xy_all.shape[0], chunk_size):
            xc = xy_all[i : i + chunk_size].clone().requires_grad_(True)
            
            # 1. get velocity and stress
            u, v, p, tau = physics.get_velocity(model, xc, create_graph=True)
            tau_xx, tau_xy, tau_yy = tau[:, 0:1], tau[:, 1:2], tau[:, 2:3]
            
            # 2. get first derivatives of velocity
            grad_u = physics._grad(u, xc, create_graph=True)
            u_x, u_y = grad_u[:, 0:1], grad_u[:, 1:2]
            
            grad_v = physics._grad(v, xc, create_graph=True)
            v_x, v_y = grad_v[:, 0:1], -u_x
            
            # 3. get first derivatives of stress
            g_txx = physics._grad(tau_xx, xc, create_graph=True)
            tau_xx_x, tau_xx_y = g_txx[:, 0:1], g_txx[:, 1:2]
            
            g_txy = physics._grad(tau_xy, xc, create_graph=True)
            tau_xy_x, tau_xy_y = g_txy[:, 0:1], g_txy[:, 1:2]
            
            g_tyy = physics._grad(tau_yy, xc, create_graph=True)
            tau_yy_x, tau_yy_y = g_tyy[:, 0:1], g_tyy[:, 1:2]
            
            # 4. get second derivatives of velocity
            u_xx = physics._grad(u_x, xc, create_graph=True)[:, 0:1]
            grad_u_y = physics._grad(u_y, xc, create_graph=True)
            u_yx, u_yy = grad_u_y[:, 0:1], grad_u_y[:, 1:2]
            
            v_xx = physics._grad(v_x, xc, create_graph=True)[:, 0:1]
            v_yy = -u_yx
            
            # 5. get nondimensional parameters
            Re, Wi, beta, beta_poly, eps, alpha = physics._nondim()
            
            # 6. compute constant parts of momentum equations
            const_u = Re * (u * u_x + v * u_y) - beta * (u_xx + u_yy) - (tau_xx_x + tau_xy_y)
            const_v = Re * (u * v_x + v * v_y) - beta * (v_xx + v_yy) - (tau_xy_x + tau_yy_y)
            
            const_u_list.append(const_u.detach())
            const_v_list.append(const_v.detach())
            
        const_u_all = torch.cat(const_u_list, dim=0)
        const_v_all = torch.cat(const_v_list, dim=0)
        
    return const_u_all, const_v_all


def train_pressure_only(model, physics, data, save_dir=None, resume_checkpoint=None, tb_writer=None):
    """
    Loop di addestramento personalizzato per ottimizzare SOLO la pressione con pre-computazione delle costanti.
    """
    history = SimpleHistory()
    
    # 1. Caricamento del checkpoint pre-addestrato
    if resume_checkpoint is not None and os.path.exists(resume_checkpoint):
        print(f"\n[Checkpoint] Caricamento da: {resume_checkpoint}")
        chk = torch.load(resume_checkpoint, map_location=DEVICE)
        model.load_state_dict(chk['model_state_dict'])
        physics.load_state_dict(chk['physics_state_dict'])
        print("[Checkpoint] Pesi caricati correttamente.")
    else:
        raise FileNotFoundError(f"Checkpoint richiesto non trovato: {resume_checkpoint}")

    # 2. Congelamento reti psi e tau, sblocco solo pressione p
    for p in model.parameters():
        p.requires_grad = False
    for p in model.model_p.parameters():
        p.requires_grad = True

    if physics.inverse_mode:
        for p in physics.parameters():
            p.requires_grad = False

    # Stampa di verifica dello stato di addestramento dei parametri
    print("\n--- STATO DEI PARAMETRI DEL MODELLO ---")
    for name, param in model.named_parameters():
        print(f"  {name:<35} | Requires Grad: {param.requires_grad}")
    print("----------------------------------------\n")

    xy_all = data["coords"]
    bc_data = data["boundary_groups"]
    var_w = data["var_weights"]

    # ==================================================================
    # FASE 1: ADAM (FP32)
    # ==================================================================
    print("\n[Precomputation] Precalcolo termini costanti della momentum in corso (FP32)...")
    const_u_adam, const_v_adam = precompute_momentum_constants(model, physics, data)
    print(f"[Precomputation] Termini precalcolati. Shape: {const_u_adam.shape}")

    optimizer = torch.optim.Adam(model.model_p.parameters(), lr=BASE_LR, eps=ADAM_EPS)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ADAM_EPOCHS, eta_min=1e-6)

    def step_loss_and_backward(points, bc_data, var_w, const_u, const_v):
        # Elaborazione in unica passata (senza chunking per massimizzare la velocità)
        xph = points.clone().requires_grad_(True)
        p = model.model_p(xph) * model.p_scale
        
        # Gradiente di pressione rispetto a x, y
        grad_p = physics._grad(p, xph, create_graph=True)
        p_x, p_y = grad_p[:, 0:1], grad_p[:, 1:2]
        
        # Momentum equations
        f_u = const_u + p_x
        f_v = const_v + p_y
        
        loss_m = (f_u**2 + f_v**2).mean() / 2.0
        (W_PHYSICS * loss_m).backward()
        
        # Boundary Loss (Solo PressurePoint)
        b_loss = physics.boundary_loss(model, bc_data, var_w, active_bcs=["p"])
        b_loss_val = b_loss.item()
        (W_BC * b_loss).backward()
        
        tot_loss = (W_PHYSICS * loss_m.item()) + (W_BC * b_loss_val)
        return tot_loss, loss_m.item(), b_loss_val

    print(f"\n{'=' * 60}\nFASE 1 ADAM (Ottimizzazione Pressione): {ADAM_EPOCHS} epoche\n{'=' * 60}")
    
    pbar = tqdm(range(ADAM_EPOCHS), desc="Adam", mininterval=2.0)
    for epoch in pbar:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        
        tot_loss, loss_m, b_loss_p = step_loss_and_backward(xy_all, bc_data, var_w, const_u_adam, const_v_adam)
        
        torch.nn.utils.clip_grad_norm_(model.model_p.parameters(), GRAD_CLIP_NORM)
        optimizer.step()
        scheduler.step()

        log_loss = ((epoch + 1) % 50 == 0) or (epoch + 1) == ADAM_EPOCHS or epoch == 0
        log_l2 = ((epoch + 1) % max(1, ADAM_EPOCHS // 40) == 0) or (epoch == 0) or ((epoch + 1) == ADAM_EPOCHS)

        if log_loss:
            loss_dict = {
                "total": tot_loss,
                "data": 0.0,
                "bc": b_loss_p,
                "pde": W_PHYSICS * loss_m,
                "loss_momentum": loss_m,
                "loss_constitutive": 0.0
            }
            
            if log_l2:
                print(f"\n[Epoch {epoch+1}] Loss: {tot_loss:.4e} | PDE Momentum: {loss_m:.4e} | BC PressurePoint: {b_loss_p:.4e}")
                model.eval()
                with torch.no_grad():
                    l2_errs = compute_l2_errors(model, physics, data)
                    print(f"  L2 Errors -> u: {l2_errs['u']:.4e} | v: {l2_errs['v']:.4e} | p: {l2_errs['p']:.4e}")
                    print(f"               tau_xx: {l2_errs['tau_xx']:.4e} | tau_xy: {l2_errs['tau_xy']:.4e} | tau_yy: {l2_errs['tau_yy']:.4e}")
                model.train()
                
                loss_dict.update({
                    "l2_u": l2_errs["u"],
                    "l2_v": l2_errs["v"],
                    "l2_p": l2_errs["p"],
                    "l2_tau_xx": l2_errs["tau_xx"],
                    "l2_tau_xy": l2_errs["tau_xy"],
                    "l2_tau_yy": l2_errs["tau_yy"],
                })

                if save_dir is not None:
                    chk_path = os.path.join(save_dir, "checkpoint.pth")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'physics_state_dict': physics.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'history_state_dict': history.state_dict()
                    }, chk_path)
            
            history.update(epoch, loss_dict)

            if tb_writer is not None:
                tb_writer.add_scalar('Loss/Total', tot_loss, epoch)
                tb_writer.add_scalar('Loss/BC', b_loss_p, epoch)
                tb_writer.add_scalar('Loss/PDE', W_PHYSICS * loss_m, epoch)
                tb_writer.add_scalar('Loss/Momentum', loss_m, epoch)
                if log_l2:
                    for k in ['u', 'v', 'p', 'tau_xx', 'tau_xy', 'tau_yy']:
                        tb_writer.add_scalar(f'L2_Error/{k}', l2_errs[k], epoch)

        pbar.set_postfix({
            "Loss": f"{tot_loss:.2e}",
            "Momentum": f"{loss_m:.2e}",
            "BC_p": f"{b_loss_p:.2e}"
        })
    pbar.close()

    # ==================================================================
    # FASE 2: L-BFGS (FP64)
    # ==================================================================
    print(f"\n{'=' * 60}\nFASE L-BFGS (Fine-Tuning Pressione FP64): {LBFGS_ITERS} iterazioni\n{'=' * 60}")
    
    convert_to_fp64(model, physics, data)
    xy_all = data["coords"]
    bc_data = data["boundary_groups"]
    var_w = data["var_weights"]

    # Ripristina i flag di gradiente dopo la conversione in FP64
    for p in model.parameters():
        p.requires_grad = False
    for p in model.model_p.parameters():
        p.requires_grad = True
    if physics.inverse_mode:
        for p in physics.parameters():
            p.requires_grad = False

    print("\n[Precomputation] Precalcolo termini costanti della momentum in corso (FP64)...")
    const_u_lbfgs, const_v_lbfgs = precompute_momentum_constants(model, physics, data)
    print(f"[Precomputation] Termini precalcolati. Shape: {const_u_lbfgs.shape}")

    optimizer_lbfgs = torch.optim.LBFGS(
        model.model_p.parameters(),
        lr=1.0,
        max_iter=LBFGS_ITERS,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        history_size=300,
        line_search_fn="strong_wolfe",
    )

    l_it = [0]
    pbar_lbfgs = tqdm(total=LBFGS_ITERS, desc="L-BFGS", mininterval=2.0)

    def closure():
        optimizer_lbfgs.zero_grad()
        tot_loss, loss_m, b_loss_p = step_loss_and_backward(
            xy_all, bc_data, var_w, const_u_lbfgs, const_v_lbfgs
        )
        loss_tensor = torch.tensor(tot_loss, device=DEVICE)

        log_lbfgs = (l_it[0] % max(1, LBFGS_ITERS // 100) == 0) or (l_it[0] == LBFGS_ITERS - 1)
        if log_lbfgs:
            with torch.no_grad():
                l2_errs = compute_l2_errors(model, physics, data)
                
            loss_dict = {
                "total": tot_loss,
                "data": 0.0,
                "bc": b_loss_p,
                "pde": W_PHYSICS * loss_m,
                "loss_momentum": loss_m,
                "loss_constitutive": 0.0,
                "l2_u": l2_errs["u"],
                "l2_v": l2_errs["v"],
                "l2_p": l2_errs["p"],
                "l2_tau_xx": l2_errs["tau_xx"],
                "l2_tau_xy": l2_errs["tau_xy"],
                "l2_tau_yy": l2_errs["tau_yy"],
            }
            
            history.update(ADAM_EPOCHS + l_it[0], loss_dict)
            
            if tb_writer is not None:
                tb_writer.add_scalar('Loss/Total', tot_loss, ADAM_EPOCHS + l_it[0])
                tb_writer.add_scalar('Loss/BC', b_loss_p, ADAM_EPOCHS + l_it[0])
                tb_writer.add_scalar('Loss/PDE', W_PHYSICS * loss_m, ADAM_EPOCHS + l_it[0])
                tb_writer.add_scalar('Loss/Momentum', loss_m, ADAM_EPOCHS + l_it[0])
                for k in ['u', 'v', 'p', 'tau_xx', 'tau_xy', 'tau_yy']:
                    tb_writer.add_scalar(f'L2_Error/{k}', l2_errs[k], ADAM_EPOCHS + l_it[0])

        l_it[0] += 1
        pbar_lbfgs.update(1)
        pbar_lbfgs.set_postfix({"Loss": f"{tot_loss:.2e}"})
        return loss_tensor

    optimizer_lbfgs.step(closure)
    pbar_lbfgs.close()

    if save_dir is not None:
        chk_path = os.path.join(save_dir, "checkpoint_lbfgs.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'physics_state_dict': physics.state_dict(),
            'history_state_dict': history.state_dict()
        }, chk_path)
        print(f"  [Checkpoint L-BFGS] Salvato in: {chk_path}")

    return history


if __name__ == "__main__":
    print(f"Device: {DEVICE} | Dtype: {torch.get_default_dtype()}")
    print(f"Dataset: {DATASET_PATH}\n")
    print("=" * 60)

    # 1. Caricamento Dati
    data = load_data()

    # 2. Inizializzazione Modello e Fisica
    model = CombinedModel(p_scale=data["p_scale"], tau_scale=data["tau_scale"]).to(DEVICE)
    
    physics = Physics(
        U_ref=data["U_ref"],
        H_ref=data["H"],
        var_weights=data["var_weights"],
        inverse_mode=INVERSE_PROBLEM,
        tau_scale=data["tau_scale"],
        p_scale=data["p_scale"],
    ).to(DEVICE)

    # Inizializziamo TensorBoard
    launch_tensorboard_server(OUTPUT_DIR.parent)
    
    tb_dir = OUTPUT_DIR / "tb_logs"
    tb_dir.mkdir(parents=True, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=str(tb_dir))

    # 3. Avvio dell'addestramento Pressure-Only
    history = train_pressure_only(
        model,
        physics,
        data,
        save_dir=OUTPUT_DIR,
        resume_checkpoint=RESUME_CHECKPOINT,
        tb_writer=tb_writer
    )
    
    tb_writer.close()

    # 4. Report Risultati Finali
    final_losses = evaluate_final_losses(model, physics, data)
    print(f"\n{'=' * 60}\nREPORT FINALE DETTAGLIATO\n{'=' * 60}")
    for k, v in final_losses.items():
        print(f"  {k:<20s}: {v:.6e}")

    errors = compute_l2_errors(model, physics, data)
    print("\nL2 Relative Errors Finali:")
    for fn, err in errors.items():
        print(f"  {fn:>8s}: {err:.6f}")

    # 5. Generazione dei Plot
    history.plot_losses(str(OUTPUT_DIR / "loss_history.png"))
    history.plot_l2_errors(str(OUTPUT_DIR / "l2_errors_history.png"))
    
    from src.utils import generate_all_diagnostics
    generate_all_diagnostics(model, physics, data, str(OUTPUT_DIR))

    # 6. Esportazione Obsidian se abilitata
    if EXPORT_TO_OBSIDIAN:
        from src.utils import init_run_in_obsidian, finalize_run_in_obsidian
        
        config_details = {
            "dataset": DATASET_PATH.name,
            "epochs": ADAM_EPOCHS,
            "inverse_problem": INVERSE_PROBLEM,
            "staged_training": STAGED_TRAINING,
            "activation": ACTIVATION.__name__,
            "network": layers_str,
            "lbfgs_iters": LBFGS_ITERS,
            "pressure_only": True
        }
        
        obsidian_dest_dir, obsidian_run_name = init_run_in_obsidian(config_name, config_details)
        
        if obsidian_dest_dir:
            results_details = {
                "status": "completed",
                "Pressure-Only Run": "Yes"
            }
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

    print(f"\n[OK] Esecuzione terminata. Risultati salvati in: {OUTPUT_DIR}")
