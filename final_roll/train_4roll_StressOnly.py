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
CHUNK_SIZE_ADAM_PHASE1 = 200000
CHUNK_SIZE_ADAM_PHASE2 = 24000
CHUNK_SIZE_LBFGS_PHASE3 = get_optimal_chunk_size(phase=3)

# --- Opzioni di Controllo ---
STAGED_TRAINING = False # Not used in StressOnly logic but kept for compatibility
INVERSE_PROBLEM = False
USE_LBFGS = True
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
ADAM_EPOCHS = 1000*40
#LBFGS_MAX_ITERS = int(0.1 * ADAM_EPOCHS)
LBFGS_MAX_ITERS=1000
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

def train_stress_only(model, physics, data):
    """
    Loop di training custom separato in due fasi:
    1. Metà delle epoche Adam: allena psi e p solo su Dati Comsol.
    2. Altra metà epoche Adam: congela psi e p, allena tau su Momentum + Constitutive PDEs.
    3. Opzionale: L-BFGS solo su tau.
    """
    history = SimpleHistory()
    
    xy_all = data["coords"]
    uv_all = data["uv_data"]
    p_all = data["p"]
    var_w = data["var_weights"]
    
    half_epochs = int(ADAM_EPOCHS * 0.7)

    # Assicuriamoci che i parametri fisici non vengano trainati in questo approccio diretto
    if physics.inverse_mode:
        for pname in ["mu_s", "mu_p", "lam", "eps", "alpha"]:
            getattr(physics, pname).requires_grad_(False)

    # ------------------------------------------------------------------
    # FASE 1: DATA ONLY (Velocity & Pressure)
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}\nFASE 1: DATA ONLY (Velocity & Pressure) - {half_epochs} epoche\n{'=' * 60}")
    
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
    sch_phase1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_phase1, T_max=half_epochs, eta_min=1e-6)

    pbar1 = tqdm(range(half_epochs), desc="Adam Phase 1 (Data)", mininterval=2.0)
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
            # create_graph=True è OBBLIGATORIO anche qui, perché u e v sono derivate di psi.
            # Per backpropagare l'errore di u e v sui pesi di model_psi serve la derivata seconda!
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

        log_loss = ((epoch + 1) % 10 == 0) or (epoch == 0) or ((epoch + 1) == half_epochs)
        log_l2 = ((epoch + 1) % max(1, ADAM_EPOCHS // 40) == 0) or (epoch == 0) or ((epoch + 1) == half_epochs)
        
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

            history.update(epoch, loss_dict)
            
        pbar1.set_postfix({"L_uv": f"{d_loss_uv_accum:.2e}", "L_p": f"{d_loss_p_accum:.2e}"})
    pbar1.close()

    # ------------------------------------------------------------------
    # FASE 2: PDE ONLY (Stress Tensor)
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}\nFASE 2: PDE ONLY (Stress) - {ADAM_EPOCHS - half_epochs} epoche\n{'=' * 60}")
    
    # Congela psi e p, attiva tau
    for p_param in model.parameters():
        p_param.requires_grad = False
    for p_param in model.model_tau.parameters():
        p_param.requires_grad = True

    opt_phase2 = torch.optim.Adam(model.model_tau.parameters(), lr=BASE_LR, eps=ADAM_EPS)
    sch_phase2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_phase2, T_max=(ADAM_EPOCHS - half_epochs), eta_min=1e-6)

    pbar2 = tqdm(range(ADAM_EPOCHS - half_epochs), desc="Adam Phase 2 (PDE)", mininterval=2.0)
    for epoch in pbar2:
        model.train()
        opt_phase2.zero_grad(set_to_none=True)
        
        p_loss_m_accum, p_loss_c_accum = 0.0, 0.0
        
        for i in range(0, xy_all.shape[0], CHUNK_SIZE_ADAM_PHASE2):
            xc = xy_all[i : i + CHUNK_SIZE_ADAM_PHASE2]
            w_chunk = xc.shape[0] / xy_all.shape[0]
            
            xph = xc.clone().requires_grad_(True)
            # create_graph=True perché le PDE calcolano derivate dei tensori
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
        
        log_loss = ((epoch + 1) % 10 == 0) or (epoch == 0) or ((epoch + 1) == (ADAM_EPOCHS - half_epochs))
        log_l2 = ((epoch + 1) % max(1, ADAM_EPOCHS // 40) == 0) or (epoch == 0) or ((epoch + 1) == (ADAM_EPOCHS - half_epochs))
        
        if log_loss:
            loss_dict = {"total": tot_pde, "pde": tot_pde, "loss_momentum": p_loss_m_accum, "loss_constitutive": p_loss_c_accum}
            
            if log_l2:
                print(f"\n[Epoch {half_epochs + epoch}] Phase 2 (PDE) | Loss Tot: {tot_pde:.4e} | Mom: {p_loss_m_accum:.4e} | Con: {p_loss_c_accum:.4e}")
                model.eval()
                with torch.no_grad():
                    l2_errs = compute_l2_errors(model, physics, data)
                    print(f"[Epoch {half_epochs + epoch}] L2 Errors:")
                    for k, v in l2_errs.items():
                        print(f"  {k}: {v:.4e}")
                model.train()
                
                loss_dict.update({
                    "l2_u": l2_errs["u"], "l2_v": l2_errs["v"], "l2_p": l2_errs["p"],
                    "l2_tau_xx": l2_errs["tau_xx"], "l2_tau_xy": l2_errs["tau_xy"], "l2_tau_yy": l2_errs["tau_yy"],
                    "l2_tau_xx_masked": l2_errs["tau_xx_masked"], "l2_tau_xy_masked": l2_errs["tau_xy_masked"], "l2_tau_yy_masked": l2_errs["tau_yy_masked"],
                })

            history.update(half_epochs + epoch, loss_dict)
            
        pbar2.set_postfix({"L_mom": f"{p_loss_m_accum:.2e}", "L_con": f"{p_loss_c_accum:.2e}"})
    pbar2.close()

    # ------------------------------------------------------------------
    # FASE 3: L-BFGS ONLY (Stress Tensor)
    # ------------------------------------------------------------------
    if USE_LBFGS:
        print(f"\n{'=' * 60}\nFASE L-BFGS: {int(LBFGS_MAX_ITERS)} iterazioni (FP64)\n{'=' * 60}")
        convert_to_fp64(model, physics, data)
        xy_all = data["coords"]
        
        # Ci assicuriamo che solo tau sia attivo durante L-BFGS
        for p_param in model.parameters():
            p_param.requires_grad = False
        for p_param in model.model_tau.parameters():
            p_param.requires_grad = True

        optimizer_lbfgs = torch.optim.LBFGS(
            model.model_tau.parameters(),
            lr=1.0,
            max_iter=int(LBFGS_MAX_ITERS),
            tolerance_grad=1e-9,
            tolerance_change=1e-12,
            history_size=300,
            line_search_fn="strong_wolfe",
        )

        l_it = [0]
        pbar_lbfgs = tqdm(total=int(LBFGS_MAX_ITERS), desc="L-BFGS (Stress PDE)", mininterval=2.0)

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

            log_lbfgs = (l_it[0] % max(1, int(LBFGS_MAX_ITERS) // 100) == 0) or (l_it[0] == int(LBFGS_MAX_ITERS) - 1)
            if log_lbfgs:
                loss_dict = {
                    "total": tot_pde,
                    "pde": tot_pde,
                    "loss_momentum": p_loss_m_accum,
                    "loss_constitutive": p_loss_c_accum
                }
                
                print(f"\n[L-BFGS Iter {l_it[0]}] Loss Tot: {tot_pde:.4e} | Mom: {p_loss_m_accum:.4e} | Con: {p_loss_c_accum:.4e}")
                model.eval()
                with torch.no_grad():
                    l2_errs = compute_l2_errors(model, physics, data)
                    print(f"[L-BFGS Iter {l_it[0]}] L2 Errors:")
                    for k, v in l2_errs.items():
                        print(f"  {k}: {v:.4e}")
                model.train()
                
                loss_dict.update({
                    "l2_u": l2_errs["u"], "l2_v": l2_errs["v"], "l2_p": l2_errs["p"],
                    "l2_tau_xx": l2_errs["tau_xx"], "l2_tau_xy": l2_errs["tau_xy"], "l2_tau_yy": l2_errs["tau_yy"],
                    "l2_tau_xx_masked": l2_errs["tau_xx_masked"], "l2_tau_xy_masked": l2_errs["tau_xy_masked"], "l2_tau_yy_masked": l2_errs["tau_yy_masked"],
                })
                
                history.update(ADAM_EPOCHS + l_it[0], loss_dict)
                
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
    config_name = f"{DATASET_PATH.stem}_StressOnly_L{layers_str}_E{ADAM_EPOCHS}_{ACTIVATION.__name__}_inv{INVERSE_PROBLEM}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    OUTPUT_DIR = BASE_DIR / "output_4rollmill" / config_name
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    global_log_path = OUTPUT_DIR / "train_log.txt"

    print(f"Device: {DEVICE} | Dtype: {torch.get_default_dtype()}")
    print(f"Dataset: {DATASET_PATH}\n")
    print("=" * 60)
    print("MODELLO STRESS-ONLY SEPARATO")
    print("Fase 1: psi e p allenati solo su dati COMSOL")
    print("Fase 2: tau allenato solo su PDE (Momentum + Constitutive)")
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
    history = train_stress_only(model, physics, data)

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

    print(f"\n[OK] Esecuzione terminata. Plot salvati in: {OUTPUT_DIR}")
