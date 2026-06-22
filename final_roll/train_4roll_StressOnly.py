import os
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn as nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.debug import test_random_points, debug_physics_magnitudes
from src.physics import Physics, evaluate_final_losses, compute_l2_errors
from src.train import CombinedModel, initialize_last_layer_zero, init_weights_xavier, SimpleHistory
from src.utils import load_data, convert_to_fp64
from src.utils import get_optimal_chunk_size
import src.utils

import src.debug
import src.physics
import src.train

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
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

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
CHUNK_SIZE_ADAM_PHASE1 = get_optimal_chunk_size(phase=1)
CHUNK_SIZE_ADAM_PHASE2 = get_optimal_chunk_size(phase=2)
CHUNK_SIZE_LBFGS_PHASE3 = get_optimal_chunk_size(phase=3)

# --- Opzioni di Controllo ---
EXPORT_TO_OBSIDIAN = True  # True: esporta i log e i plot nel vault Obsidian a fine run
STAGED_TRAINING = False # Not used in StressOnly logic but kept for compatibility
INVERSE_PROBLEM = False
USE_LBFGS = True
RESUME_CHECKPOINT = r"C:\Users\eaglw\Documents\PINN tesi\final_roll\output_4rollmill\4_roll_mill_StressOnly_L8x128_E1=50000_E2=30000_SiLU_invFalse_20260619_130621\checkpoint.pth"

CHUNK_SIZE_ADAM = CHUNK_SIZE_ADAM_PHASE1
CHUNK_SIZE_LBFGS = CHUNK_SIZE_LBFGS_PHASE3

# --- Percorsi Base ---
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR.parent / "COMSOL" / "4roll" / "4_roll_mill.csv"

# --- Parametri Fisici REALI (Ground Truth) ---
MU_S_TRUE = 0.1
MU_P_TRUE = 0.9
LAM_TRUE = 1.0
EPS_TRUE = 0.0
ALPHA_TRUE = 0.0
RHO = 1000.0

# --- Costanti e Guess Iniziali ---
MIN_MU_S = 1e-6
MIN_MU_P = 1e-6
MIN_LAM = 1e-6
GUESS_MULTIPLIER = 0.8
GUESS_MU_S = MU_S_TRUE * GUESS_MULTIPLIER
GUESS_MU_P = MU_P_TRUE * GUESS_MULTIPLIER
GUESS_LAM = LAM_TRUE * GUESS_MULTIPLIER
GUESS_EPS = 0.0
GUESS_ALPHA = 0.0

# --- Architettura Neural Network ---
HIDDEN_LAYERS = [128] * 8
ACTIVATION = nn.SiLU

# --- Iperparametri di Training ---
ADAM_EPOCHS_PHASE1 = 50000
LBFGS_ITERS_PHASE1 = 5000
ADAM_EPOCHS_PHASE2 = 30000
LBFGS_ITERS_PHASE2 = 3000
BASE_LR = 1e-3
ADAM_EPS = 1e-7
PARAM_LR_FACTOR = 0.1
GRAD_CLIP_NORM = 5.0
PARAM_CLIP_NORM = 1.0

# --- Pesi Funzione di Loss ---
W_BC = 2.0
W_PHYSICS = 3.0
W_DATA = 1.0
W_MOMENTUM = 1.0
W_CONSTITUTIVE = 1.0
VARIANCE_EPS = 1e-4

# Iniezione dinamica dei parametri globali nei moduli di src per risolvere la mancanza di config
for module in [src.debug, src.physics, src.train, src.utils]:
    for name, val in list(globals().items()):
        if name.isupper():
            module.__dict__[name] = val


# ============================================================================
# NEW: FUNZIONI CUSTOM PER STRESS ONLY
# ============================================================================
def custom_data_loss(u, v, p, uv_target, p_target, var_w):
    """Calcola la MSE pesata separata per velocità e pressione."""
    loss_u = src.utils.weighted_mse(u, uv_target[:, 0:1], var_w["u"])
    loss_v = src.utils.weighted_mse(v, uv_target[:, 1:2], var_w["v"])
    loss_p = src.utils.weighted_mse(p, p_target, var_w["p"])
    return 0.5 * (loss_u + loss_v), loss_p

def convert_to_fp32(model, physics, data):
    """
    Converte in modo centralizzato modello, fisica e dati a FP32 prima di Adam 2.
    """
    def _cast_dict_to_float(d):
        for key, value in d.items():
            if isinstance(value, torch.Tensor):
                d[key] = value.float()
            elif isinstance(value, dict):
                _cast_dict_to_float(value)

    _cast_dict_to_float(data)
    model.float()
    physics.float()

def train_stress_only(model, physics, data, save_dir=None, resume_checkpoint=None):
    """
    Loop di training custom separato in quattro fasi con logica di resume integrata.
    """
    history = SimpleHistory()
    global CHUNK_SIZE_ADAM_PHASE1, CHUNK_SIZE_ADAM_PHASE2, CHUNK_SIZE_LBFGS
    
    xy_all = data["coords"]
    uv_all = data["uv_data"]
    p_all = data["p"]
    var_w = data["var_weights"]
    
    start_epoch = 0
    loaded_opt_state = None
    loaded_sch_state = None

    if resume_checkpoint is not None and os.path.exists(resume_checkpoint):
        print(f"\n[Checkpoint] Caricamento da: {resume_checkpoint}")
        chk = torch.load(resume_checkpoint, map_location=DEVICE)
        model.load_state_dict(chk['model_state_dict'])
        physics.load_state_dict(chk['physics_state_dict'])
        history.load_state_dict(chk['history_state_dict'])
        loaded_opt_state = chk.get('optimizer_state_dict', None)
        loaded_sch_state = chk.get('scheduler_state_dict', None)
        start_epoch = chk.get('epoch', 0) + 1
        print(f"[Checkpoint] Ripresa dall'epoca/iterazione totale {start_epoch}")

    # Assicuriamoci che i parametri fisici non vengano trainati in questo approccio diretto
    if physics.inverse_mode:
        for pname in ["mu_s", "mu_p", "lam", "eps", "alpha"]:
            getattr(physics, pname).requires_grad_(False)

    # Definisco confini fasi in termini di iterazioni totali globali
    phase1_end = ADAM_EPOCHS_PHASE1
    phase1_5_end = phase1_end + LBFGS_ITERS_PHASE1 if USE_LBFGS else phase1_end
    phase2_end = phase1_5_end + ADAM_EPOCHS_PHASE2
    phase2_5_end = phase2_end + LBFGS_ITERS_PHASE2 if USE_LBFGS else phase2_end

    offset = 0

    # ------------------------------------------------------------------
    # FASE 1: ADAM 1 (Data-only: Velocity & Pressure) - FP32
    # ------------------------------------------------------------------
    if start_epoch < phase1_end:
        print(f"\n{'=' * 60}\nFASE 1: ADAM 1 (Data Only: Velocity & Pressure) - {ADAM_EPOCHS_PHASE1} epoche (FP32)\n{'=' * 60}")
        
        def closure_test_phase1(c):
            xc = xy_all[:c].clone().requires_grad_(True)
            uv_c = uv_all[:c]
            p_c = p_all[:c]
            u, v, p_pred, _ = physics.get_velocity(model, xc, create_graph=True)
            loss_uv, loss_p = custom_data_loss(u, v, p_pred, uv_c, p_c, var_w)
            ((loss_uv + loss_p) * 1.0).backward()

        CHUNK_SIZE_ADAM_PHASE1 = get_optimal_chunk_size(phase=1, model=model, test_closure=closure_test_phase1)

        # Congela tau, attiva psi e p
        for p_param in model.parameters():
            p_param.requires_grad = False
        for p_param in model.model_psi.parameters():
            p_param.requires_grad = True
        for p_param in model.model_p.parameters():
            p_param.requires_grad = True

        opt_phase1 = torch.optim.Adam([
            {"params": model.model_psi.parameters(), "lr": BASE_LR},
            {"params": model.model_p.parameters(), "lr": BASE_LR}
        ], eps=ADAM_EPS)
        sch_phase1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_phase1, T_max=ADAM_EPOCHS_PHASE1, eta_min=1e-6)

        if loaded_opt_state is not None:
            opt_phase1.load_state_dict(loaded_opt_state)
            loaded_opt_state = None
        if loaded_sch_state is not None:
            try:
                sch_phase1.load_state_dict(loaded_sch_state)
            except Exception:
                pass
            loaded_sch_state = None

        local_start = start_epoch
        pbar1 = tqdm(range(local_start, ADAM_EPOCHS_PHASE1), desc="Adam Phase 1 (Data)", mininterval=2.0)
        for epoch in pbar1:
            model.train()
            opt_phase1.zero_grad(set_to_none=True)
            
            d_loss_uv_accum, d_loss_p_accum = 0.0, 0.0
            
            for i in range(0, xy_all.shape[0], CHUNK_SIZE_ADAM_PHASE1):
                xc = xy_all[i : i + CHUNK_SIZE_ADAM_PHASE1]
                uv_c = uv_all[i : i + CHUNK_SIZE_ADAM_PHASE1]
                p_c = p_all[i : i + CHUNK_SIZE_ADAM_PHASE1]
                w_chunk = xc.shape[0] / xy_all.shape[0]
                
                xph = xc.clone().requires_grad_(True)
                u, v, p_pred, _ = physics.get_velocity(model, xph, create_graph=True)
                
                loss_uv, loss_p = custom_data_loss(u, v, p_pred, uv_c, p_c, var_w)
                chunk_total_loss = (loss_uv + loss_p) * w_chunk
                chunk_total_loss.backward()
                d_loss_uv_accum += loss_uv.item() * w_chunk
                d_loss_p_accum += loss_p.item() * w_chunk

            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            opt_phase1.step()
            sch_phase1.step()

            tot_loss = d_loss_uv_accum + d_loss_p_accum

            log_loss = ((epoch + 1) % 10 == 0) or (epoch == 0) or ((epoch + 1) == ADAM_EPOCHS_PHASE1)
            log_l2 = ((epoch + 1) % max(1, ADAM_EPOCHS_PHASE1 // 40) == 0) or (epoch == 0) or ((epoch + 1) == ADAM_EPOCHS_PHASE1)
            
            if log_loss:
                loss_dict = {"total": tot_loss, "data": d_loss_uv_accum + d_loss_p_accum, "data_uv": d_loss_uv_accum, "data_p": d_loss_p_accum}
                if log_l2:
                    print(f"\n[Epoch {epoch}] Phase 1 (Data) | Loss Tot: {tot_loss:.4e} | UV: {d_loss_uv_accum:.4e} | P: {d_loss_p_accum:.4e}")
                    model.eval()
                    with torch.no_grad():
                        l2_errs = compute_l2_errors(model, physics, data)
                        print(f"[Epoch {epoch}] L2 Errors:")
                        for k, v in l2_errs.items():
                            print(f"  {k}: {v:.4e}")
                    model.train()
                    
                    loss_dict.update({
                        "l2_u": l2_errs["u"], "l2_v": l2_errs["v"], "l2_p": l2_errs["p"],
                        "l2_tau_xx": l2_errs["tau_xx"], "l2_tau_xy": l2_errs["tau_xy"], "l2_tau_yy": l2_errs["tau_yy"],
                        "l2_tau_xx_masked": l2_errs["tau_xx_masked"], "l2_tau_xy_masked": l2_errs["tau_xy_masked"], "l2_tau_yy_masked": l2_errs["tau_yy_masked"],
                    })

                    if save_dir is not None:
                        chk_path = os.path.join(save_dir, "checkpoint.pth")
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'physics_state_dict': physics.state_dict(),
                            'optimizer_state_dict': opt_phase1.state_dict(),
                            'scheduler_state_dict': sch_phase1.state_dict(),
                            'history_state_dict': history.state_dict()
                        }, chk_path)
                        print(f"  [Checkpoint] Salvato in: {chk_path}")

                history.update(epoch, loss_dict)
                
            pbar1.set_postfix({"L_uv": f"{d_loss_uv_accum:.2e}", "L_p": f"{d_loss_p_accum:.2e}"})
        pbar1.close()
    else:
        print(f"\n[Skip] Fase 1 (Adam 1) completata precedentemente.")
    
    offset = phase1_end

    # ------------------------------------------------------------------
    # FASE 1.5: L-BFGS 1 (Data-only: Velocity & Pressure) - FP64
    # ------------------------------------------------------------------
    if USE_LBFGS:
        if start_epoch < phase1_5_end:
            print(f"\n{'=' * 60}\nFASE 1.5 L-BFGS: Refinamento psi e p - {int(LBFGS_ITERS_PHASE1)} iterazioni (FP64)\n{'=' * 60}")
            convert_to_fp64(model, physics, data)
            xy_all = data["coords"]
            uv_all = data["uv_data"]
            p_all = data["p"]
            var_w = data["var_weights"]
            
            def closure_test_phase1_5(c):
                xc = xy_all[:c].clone().requires_grad_(True)
                uv_c = uv_all[:c]
                p_c = p_all[:c]
                u, v, p_pred, _ = physics.get_velocity(model, xc, create_graph=True)
                loss_uv, loss_p = custom_data_loss(u, v, p_pred, uv_c, p_c, var_w)
                ((loss_uv + loss_p) * 1.0).backward()

            CHUNK_SIZE_LBFGS = get_optimal_chunk_size(phase=1, model=model, test_closure=closure_test_phase1_5)

            for p_param in model.parameters():
                p_param.requires_grad = False
            for p_param in model.model_psi.parameters():
                p_param.requires_grad = True
            for p_param in model.model_p.parameters():
                p_param.requires_grad = True

            optimizer_lbfgs_1 = torch.optim.LBFGS(
                list(model.model_psi.parameters()) + list(model.model_p.parameters()),
                lr=1.0,
                max_iter=int(LBFGS_ITERS_PHASE1),
                tolerance_grad=1e-9,
                tolerance_change=1e-12,
                history_size=300,
                line_search_fn="strong_wolfe",
            )

            if loaded_opt_state is not None:
                print("LBFGS_1 resume: stato dell'optimizer ignorato per evitare instabilità e crash di PyTorch.")
                loaded_opt_state = None

            local_start = start_epoch - offset if start_epoch > offset else 0
            l_it_1 = [local_start]
            pbar_lbfgs_1 = tqdm(total=int(LBFGS_ITERS_PHASE1), initial=local_start, desc="L-BFGS 1 (Data)", mininterval=2.0)

            def closure_1():
                optimizer_lbfgs_1.zero_grad()
                d_loss_uv_accum, d_loss_p_accum = 0.0, 0.0
                
                for i in range(0, xy_all.shape[0], CHUNK_SIZE_LBFGS):
                    xc = xy_all[i : i + CHUNK_SIZE_LBFGS]
                    uv_c = uv_all[i : i + CHUNK_SIZE_LBFGS]
                    p_c = p_all[i : i + CHUNK_SIZE_LBFGS]
                    w_chunk = xc.shape[0] / xy_all.shape[0]
                    
                    xph = xc.clone().requires_grad_(True)
                    u, v, p_pred, _ = physics.get_velocity(model, xph, create_graph=True)
                    
                    loss_uv, loss_p = custom_data_loss(u, v, p_pred, uv_c, p_c, var_w)
                    chunk_total_loss = (loss_uv + loss_p) * w_chunk
                    chunk_total_loss.backward()
                    
                    d_loss_uv_accum += loss_uv.item() * w_chunk
                    d_loss_p_accum += loss_p.item() * w_chunk

                tot_loss = d_loss_uv_accum + d_loss_p_accum
                loss_tensor = torch.tensor(tot_loss, device=DEVICE)

                log_lbfgs = (l_it_1[0] % max(1, int(LBFGS_ITERS_PHASE1) // 100) == 0) or (l_it_1[0] == int(LBFGS_ITERS_PHASE1) - 1)
                if log_lbfgs:
                    loss_dict = {
                        "total": tot_loss,
                        "data": tot_loss,
                        "data_uv": d_loss_uv_accum,
                        "data_p": d_loss_p_accum
                    }
                    
                    print(f"\n[L-BFGS 1 Iter {l_it_1[0]}] Loss Tot: {tot_loss:.4e} | UV: {d_loss_uv_accum:.4e} | P: {d_loss_p_accum:.4e}")
                    model.eval()
                    with torch.no_grad():
                        l2_errs = compute_l2_errors(model, physics, data)
                        print(f"[L-BFGS 1 Iter {l_it_1[0]}] L2 Errors:")
                        for k, v in l2_errs.items():
                            print(f"  {k}: {v:.4e}")
                    model.train()
                    
                    loss_dict.update({
                        "l2_u": l2_errs["u"], "l2_v": l2_errs["v"], "l2_p": l2_errs["p"],
                        "l2_tau_xx": l2_errs["tau_xx"], "l2_tau_xy": l2_errs["tau_xy"], "l2_tau_yy": l2_errs["tau_yy"],
                        "l2_tau_xx_masked": l2_errs["tau_xx_masked"], "l2_tau_xy_masked": l2_errs["tau_xy_masked"], "l2_tau_yy_masked": l2_errs["tau_yy_masked"],
                    })
                    
                    if save_dir is not None:
                        chk_path = os.path.join(save_dir, "checkpoint.pth")
                        torch.save({
                            'epoch': offset + l_it_1[0],
                            'model_state_dict': model.state_dict(),
                            'physics_state_dict': physics.state_dict(),
                            'optimizer_state_dict': optimizer_lbfgs_1.state_dict(),
                            'history_state_dict': history.state_dict()
                        }, chk_path)
                        print(f"  [Checkpoint] Salvato in: {chk_path}")
                    
                    history.update(offset + l_it_1[0], loss_dict)
                    
                l_it_1[0] += 1
                pbar_lbfgs_1.update(1)
                pbar_lbfgs_1.set_postfix({"L_uv": f"{d_loss_uv_accum:.2e}", "L_p": f"{d_loss_p_accum:.2e}"})
                return loss_tensor

            optimizer_lbfgs_1.step(closure_1)
            pbar_lbfgs_1.close()
        else:
            print(f"\n[Skip] Fase 1.5 (L-BFGS 1) completata precedentemente.")
            
        offset = phase1_5_end

    # ------------------------------------------------------------------
    # FASE 2: ADAM 2 (PDE-only: Stress Tensor) - FP32
    # ------------------------------------------------------------------
    if start_epoch < phase2_end:
        print(f"\n{'=' * 60}\nFASE 2: ADAM 2 (PDE Only: Stress) - {ADAM_EPOCHS_PHASE2} epoche (FP32)\n{'=' * 60}")
        
        convert_to_fp32(model, physics, data)
        xy_all = data["coords"]
        uv_all = data["uv_data"]
        p_all = data["p"]
        var_w = data["var_weights"]

        def closure_test_phase2(c):
            xc = xy_all[:c].clone().requires_grad_(True)
            u, v, p_pred, tau_pred = physics.get_velocity(model, xc, create_graph=True)
            loss_m, loss_c = physics.compute_pde_losses(xc, u, v, p_pred, tau_pred, w_momentum=W_MOMENTUM, w_constitutive=W_CONSTITUTIVE)
            (W_PHYSICS * (loss_m + loss_c) * 1.0).backward()

        CHUNK_SIZE_ADAM_PHASE2 = get_optimal_chunk_size(phase=2, model=model, test_closure=closure_test_phase2)

        for p_param in model.parameters():
            p_param.requires_grad = False
        for p_param in model.model_tau.parameters():
            p_param.requires_grad = True

        opt_phase2 = torch.optim.Adam(model.model_tau.parameters(), lr=BASE_LR, eps=ADAM_EPS)
        sch_phase2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_phase2, T_max=ADAM_EPOCHS_PHASE2, eta_min=1e-6)

        if loaded_opt_state is not None:
            try:
                opt_phase2.load_state_dict(loaded_opt_state)
            except ValueError:
                print("Impossibile caricare state Adam_2, continuo.")
            loaded_opt_state = None
        if loaded_sch_state is not None:
            try:
                sch_phase2.load_state_dict(loaded_sch_state)
            except Exception:
                pass
            loaded_sch_state = None

        local_start = start_epoch - offset if start_epoch > offset else 0
        pbar2 = tqdm(range(local_start, ADAM_EPOCHS_PHASE2), desc="Adam Phase 2 (PDE)", mininterval=2.0)
        for epoch in pbar2:
            model.train()
            opt_phase2.zero_grad(set_to_none=True)
            
            p_loss_m_accum, p_loss_c_accum = 0.0, 0.0
            
            for i in range(0, xy_all.shape[0], CHUNK_SIZE_ADAM_PHASE2):
                xc = xy_all[i : i + CHUNK_SIZE_ADAM_PHASE2]
                w_chunk = xc.shape[0] / xy_all.shape[0]
                
                xph = xc.clone().requires_grad_(True)
                u, v, p_pred, tau_pred = physics.get_velocity(model, xph, create_graph=True)
                
                loss_m, loss_c = physics.compute_pde_losses(xph, u, v, p_pred, tau_pred, w_momentum=W_MOMENTUM, w_constitutive=W_CONSTITUTIVE)
                
                chunk_total_loss = W_PHYSICS * (loss_m + loss_c) * w_chunk
                chunk_total_loss.backward()
                
                p_loss_m_accum += loss_m.item() * w_chunk
                p_loss_c_accum += loss_c.item() * w_chunk

            torch.nn.utils.clip_grad_norm_(model.model_tau.parameters(), GRAD_CLIP_NORM)
            opt_phase2.step()
            sch_phase2.step()

            tot_pde = W_PHYSICS * (p_loss_m_accum + p_loss_c_accum)
            
            log_loss = ((epoch + 1) % 10 == 0) or (epoch == 0) or ((epoch + 1) == ADAM_EPOCHS_PHASE2)
            log_l2 = ((epoch + 1) % max(1, ADAM_EPOCHS_PHASE2 // 40) == 0) or (epoch == 0) or ((epoch + 1) == ADAM_EPOCHS_PHASE2)
            
            if log_loss:
                loss_dict = {"total": tot_pde, "pde": tot_pde, "loss_momentum": p_loss_m_accum, "loss_constitutive": p_loss_c_accum}
                
                if log_l2:
                    print(f"\n[Epoch {offset + epoch}] Phase 2 (PDE) | Loss Tot: {tot_pde:.4e} | Mom: {p_loss_m_accum:.4e} | Con: {p_loss_c_accum:.4e}")
                    model.eval()
                    with torch.no_grad():
                        l2_errs = compute_l2_errors(model, physics, data)
                        print(f"[Epoch {offset + epoch}] L2 Errors:")
                        for k, v in l2_errs.items():
                            print(f"  {k}: {v:.4e}")
                    model.train()
                    
                    loss_dict.update({
                        "l2_u": l2_errs["u"], "l2_v": l2_errs["v"], "l2_p": l2_errs["p"],
                        "l2_tau_xx": l2_errs["tau_xx"], "l2_tau_xy": l2_errs["tau_xy"], "l2_tau_yy": l2_errs["tau_yy"],
                        "l2_tau_xx_masked": l2_errs["tau_xx_masked"], "l2_tau_xy_masked": l2_errs["tau_xy_masked"], "l2_tau_yy_masked": l2_errs["tau_yy_masked"],
                    })

                    if save_dir is not None:
                        chk_path = os.path.join(save_dir, "checkpoint.pth")
                        torch.save({
                            'epoch': offset + epoch,
                            'model_state_dict': model.state_dict(),
                            'physics_state_dict': physics.state_dict(),
                            'optimizer_state_dict': opt_phase2.state_dict(),
                            'scheduler_state_dict': sch_phase2.state_dict(),
                            'history_state_dict': history.state_dict()
                        }, chk_path)
                        print(f"  [Checkpoint] Salvato in: {chk_path}")

                history.update(offset + epoch, loss_dict)
                
            pbar2.set_postfix({"L_mom": f"{p_loss_m_accum:.2e}", "L_con": f"{p_loss_c_accum:.2e}"})
        pbar2.close()
    else:
        print(f"\n[Skip] Fase 2 (Adam 2) completata precedentemente.")
        
    offset = phase2_end

    # ------------------------------------------------------------------
    # FASE 2.5: L-BFGS 2 (PDE-only: Stress Tensor) - FP64
    # ------------------------------------------------------------------
    if USE_LBFGS:
        if start_epoch < phase2_5_end:
            print(f"\n{'=' * 60}\nFASE 2.5 L-BFGS: Refinamento stress - {int(LBFGS_ITERS_PHASE2)} iterazioni (FP64)\n{'=' * 60}")
            convert_to_fp64(model, physics, data)
            xy_all = data["coords"]
            
            def closure_test_phase2_5(c):
                xc = xy_all[:c].clone().requires_grad_(True)
                u, v, p_pred, tau_pred = physics.get_velocity(model, xc, create_graph=True)
                loss_m, loss_c = physics.compute_pde_losses(xc, u, v, p_pred, tau_pred, w_momentum=W_MOMENTUM, w_constitutive=W_CONSTITUTIVE)
                (W_PHYSICS * (loss_m + loss_c) * 1.0).backward()

            CHUNK_SIZE_LBFGS = get_optimal_chunk_size(phase=3, model=model, test_closure=closure_test_phase2_5)

            for p_param in model.parameters():
                p_param.requires_grad = False
            for p_param in model.model_tau.parameters():
                p_param.requires_grad = True

            optimizer_lbfgs = torch.optim.LBFGS(
                model.model_tau.parameters(),
                lr=1.0,
                max_iter=int(LBFGS_ITERS_PHASE2),
                tolerance_grad=1e-9,
                tolerance_change=1e-12,
                history_size=300,
                line_search_fn="strong_wolfe",
            )

            if loaded_opt_state is not None:
                print("LBFGS_2 resume: stato dell'optimizer ignorato per evitare instabilità e crash di PyTorch.")
                loaded_opt_state = None

            local_start = start_epoch - offset if start_epoch > offset else 0
            l_it = [local_start]
            pbar_lbfgs = tqdm(total=int(LBFGS_ITERS_PHASE2), initial=local_start, desc="L-BFGS 2 (Stress PDE)", mininterval=2.0)

            def closure():
                optimizer_lbfgs.zero_grad()
                p_loss_m_accum, p_loss_c_accum = 0.0, 0.0
                
                for i in range(0, xy_all.shape[0], CHUNK_SIZE_LBFGS):
                    xc = xy_all[i : i + CHUNK_SIZE_LBFGS]
                    w_chunk = xc.shape[0] / xy_all.shape[0]
                    
                    xph = xc.clone().requires_grad_(True)
                    u, v, p_pred, tau_pred = physics.get_velocity(model, xph, create_graph=True)
                    loss_m, loss_c = physics.compute_pde_losses(xph, u, v, p_pred, tau_pred, w_momentum=W_MOMENTUM, w_constitutive=W_CONSTITUTIVE)
                    
                    chunk_total_loss = W_PHYSICS * (loss_m + loss_c) * w_chunk
                    chunk_total_loss.backward()
                    
                    p_loss_m_accum += loss_m.item() * w_chunk
                    p_loss_c_accum += loss_c.item() * w_chunk

                tot_pde = W_PHYSICS * (p_loss_m_accum + p_loss_c_accum)
                loss_tensor = torch.tensor(tot_pde, device=DEVICE)

                log_lbfgs = (l_it[0] % max(1, int(LBFGS_ITERS_PHASE2) // 100) == 0) or (l_it[0] == int(LBFGS_ITERS_PHASE2) - 1)
                if log_lbfgs:
                    loss_dict = {
                        "total": tot_pde,
                        "pde": tot_pde,
                        "loss_momentum": p_loss_m_accum,
                        "loss_constitutive": p_loss_c_accum
                    }
                    
                    print(f"\n[L-BFGS 2 Iter {l_it[0]}] Loss Tot: {tot_pde:.4e} | Mom: {p_loss_m_accum:.4e} | Con: {p_loss_c_accum:.4e}")
                    model.eval()
                    with torch.no_grad():
                        l2_errs = compute_l2_errors(model, physics, data)
                        print(f"[L-BFGS 2 Iter {l_it[0]}] L2 Errors:")
                        for k, v in l2_errs.items():
                            print(f"  {k}: {v:.4e}")
                    model.train()
                    
                    loss_dict.update({
                        "l2_u": l2_errs["u"], "l2_v": l2_errs["v"], "l2_p": l2_errs["p"],
                        "l2_tau_xx": l2_errs["tau_xx"], "l2_tau_xy": l2_errs["tau_xy"], "l2_tau_yy": l2_errs["tau_yy"],
                        "l2_tau_xx_masked": l2_errs["tau_xx_masked"], "l2_tau_xy_masked": l2_errs["tau_xy_masked"], "l2_tau_yy_masked": l2_errs["tau_yy_masked"],
                    })
                    
                    if save_dir is not None:
                        chk_path = os.path.join(save_dir, "checkpoint.pth")
                        torch.save({
                            'epoch': offset + l_it[0],
                            'model_state_dict': model.state_dict(),
                            'physics_state_dict': physics.state_dict(),
                            'optimizer_state_dict': optimizer_lbfgs.state_dict(),
                            'history_state_dict': history.state_dict()
                        }, chk_path)
                        print(f"  [Checkpoint] Salvato in: {chk_path}")

                    history.update(offset + l_it[0], loss_dict)
                    
                l_it[0] += 1
                pbar_lbfgs.update(1)
                pbar_lbfgs.set_postfix({"L_mom": f"{p_loss_m_accum:.2e}", "L_con": f"{p_loss_c_accum:.2e}"})
                return loss_tensor

            optimizer_lbfgs.step(closure)
            pbar_lbfgs.close()

    return history


# ============================================================================
# MAIN SCRIPT RUNNER
# ============================================================================
if __name__ == "__main__":
    layers_str = f"{len(HIDDEN_LAYERS)}x{HIDDEN_LAYERS[0]}"
    config_name = f"{DATASET_PATH.stem}_StressOnly_L{layers_str}_E1={ADAM_EPOCHS_PHASE1}_E2={ADAM_EPOCHS_PHASE2}_{ACTIVATION.__name__}_inv{INVERSE_PROBLEM}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Se facciamo il resume, potremmo voler mantenere la stessa cartella per non creare cloni, ma il file usa datetime.
    # Se RESUME_CHECKPOINT e' impostato, prendiamo il parent dir dal checkpoint.
    if RESUME_CHECKPOINT is not None and os.path.exists(RESUME_CHECKPOINT):
        OUTPUT_DIR = Path(RESUME_CHECKPOINT).parent
        print(f"\n[INFO] Ripresa addestramento. Output dir: {OUTPUT_DIR}")
    else:
        OUTPUT_DIR = BASE_DIR / "output_4rollmill" / config_name
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    global_log_path = OUTPUT_DIR / "train_log.txt"

    print(f"Device: {DEVICE} | Dtype: {torch.get_default_dtype()}")
    print(f"Dataset: {DATASET_PATH}\n")
    print("=" * 60)
    print("MODELLO STRESS-ONLY SEPARATO (4 FASI)")
    print("Fase 1: Adam su psi e p (solo dati COMSOL, FP32)")
    print("Fase 1.5: L-BFGS su psi e p (solo dati COMSOL, FP64)")
    print("Fase 2: Adam su tau (solo PDE, psi e p congelati, FP32)")
    print("Fase 2.5: L-BFGS su tau (solo PDE, psi e p congelati, FP64)")
    print("=" * 60)

    # 1. Caricamento Dati
    data = load_data()

    # 2. Inizializzazione Modello e Fisica
    model = CombinedModel(p_scale=data["p_scale"], tau_scale=data["tau_scale"]).to(DEVICE)
    for submodel in [model.model_psi, model.model_p, model.model_tau]:
        submodel.apply(lambda m: init_weights_xavier(m, activation_name=ACTIVATION))

    initialize_last_layer_zero(model.model_p)
    initialize_last_layer_zero(model.model_tau)

    physics = Physics(
        U_ref=data["U_ref"],
        H_ref=data["H"],
        var_weights=data["var_weights"],
        inverse_mode=INVERSE_PROBLEM,
        tau_scale=data["tau_scale"],
        p_scale=data["p_scale"],
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModello: {total_params:,} parametri totali")

    # 3. Training
    history = train_stress_only(model, physics, data, save_dir=OUTPUT_DIR, resume_checkpoint=RESUME_CHECKPOINT)

    # 4. Report Risultati Finali
    params = physics.log_params()
    print(f"\n{'=' * 60}\nRISULTATI FINALI PARAMETRI FISICI\n{'=' * 60}")
    for p_name, true_val in zip(
        ["mu_s", "mu_p", "lam", "eps", "alpha"],
        [MU_S_TRUE, MU_P_TRUE, LAM_TRUE, EPS_TRUE, ALPHA_TRUE],
    ):
        print(f"  {p_name:<5s}: {params[p_name]:.6f}  (true: {true_val})")

    final_losses = evaluate_final_losses(model, physics, data)
    print(f"\n{'=' * 60}\nREPORT FINALE DETTAGLIATO\n{'=' * 60}")
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

    # 6. Test di Validazione Fisica
    test_random_points(model, physics, data, num_points=10)
    debug_physics_magnitudes(model, physics, data, num_points=2000)

    if EXPORT_TO_OBSIDIAN:
        from src.utils import export_run_to_obsidian
        config_details = {
            "dataset": DATASET_PATH.name,
            "epochs_phase1": ADAM_EPOCHS_PHASE1,
            "epochs_phase2": ADAM_EPOCHS_PHASE2,
            "inverse_problem": INVERSE_PROBLEM,
            "staged_training": STAGED_TRAINING,
            "activation": ACTIVATION.__name__,
            "network": layers_str,
            "lbfgs": USE_LBFGS
        }
        
        results_details = {
            "status": "completed"
        }
        for p_name in ["mu_s", "mu_p", "lam", "eps", "alpha"]:
            if p_name in params:
                results_details[f"Param {p_name}"] = f"{params[p_name]:.6f}"
                
        for k, v in final_losses.items():
            results_details[f"Loss {k}"] = f"{v:.6e}"
            
        for fn, err in errors.items():
            results_details[f"Error {fn}"] = f"{err:.6f}"
            
        export_run_to_obsidian(
            source_dir=str(OUTPUT_DIR),
            config_name=config_name,
            config_details=config_details,
            results_details=results_details
        )

    print(f"\n[OK] Esecuzione terminata. Plot salvati in: {OUTPUT_DIR}")
