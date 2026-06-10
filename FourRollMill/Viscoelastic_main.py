import torch
import torch.nn as nn
import matplotlib.pyplot as plt
plt.switch_backend('Agg')  # Forza backend non interattivo per evitare pause
import gc
import sys
import itertools
from datetime import datetime
from pathlib import Path

# --- 1. HARDWARE & CUDA OPTIMIZATIONS ---
torch.set_float32_matmul_precision('high')  # Abilita TF32 per i matmul (RTX 3080/Ampere+)
torch.backends.cudnn.benchmark = False      # Disabilitato per batch dinamici
torch.backends.cudnn.deterministic = False

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# Risoluzione dei percorsi del progetto
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR.parent))

from func.logging_utils import update_results_csv
from func.graphic_func import plot2D_unified_comparison, plot_loss_comparison, plot2D_viscoelastic_comparison

# Import locali Viscoelastic
from FourRollMill.src.models import FCN, ViscoelasticCombinedModel, get_activation_name, format_layers_name
from FourRollMill.src.config import TrainingConfig
from FourRollMill.src.trainer import train_ViscoelasticPINN, compute_viscoelastic_metrics
from FourRollMill.src.Viscoelastic_physics import ViscoelasticPhysics

# --- 2. GRID SEARCH SPACE ---
LAYERS_OPTIONS = [[2, 128, 128, 128, 128, 128, 128, 128, 128, 1]]  # VENet 8x128
EPOCHS_OPTIONS = [13000]
MAX_LBFGS_ITERS = None #Se None, usa il 10% di epoche Adam.
ACTIVATION_OPTIONS = [nn.SiLU]
LR_STRATEGY_OPTIONS = ['cosine']
WEIGHTING_OPTIONS = ['dynamic']

# 0=PurePhys, 1=Phys+Data, 2=SoloData
GOALS_TO_RUN = [1]

GOAL_CONFIGS = {
    0: {'label': 'PurePhys',  'weights': {'bc': 1.0, 'physics': 1.0, 'data': 0.0}, 'mode': 'standard'},
    1: {'label': 'Phys+Data', 'weights': {'bc': 1.0, 'physics': 1.0, 'data': 1.0}, 'mode': 'semi_inverse'},
    2: {'label': 'SoloData',  'weights': {'bc': 1.0, 'physics': 0.0, 'data': 1.0}, 'mode': 'standard'},
}


# --- 3. FIXED CONFIGURATIONS & HYPERPARAMETERS ---
PRECISION_MODE = 'full_64'           # 'full_32' | 'staged' | 'full_64'
SEED = 123
DATASET_OPTIONS = ['4_roll_mill.csv'] #lambda = 1 anche se non c'è nel nome

COMSOL_PARAMS = {
    'mu_s': 0.1,   # Viscosità solvente [Pa·s]
    'mu_p': 0.9,   # Viscosità polimerica [Pa·s]
    'lam': 1,      # Tempo di rilassamento [s]
    'eps': 0.0,      # Parametro PTT
    'alpha': 0.0,    # Parametro Giesekus
    'rho': 1000,      # Densità [kg/m³]
    'omega': 100,  # Giri al minuto[rpm] #non credo serva che tanto lo prendo dal csv

}

# Impostazione del tipo di dato globale iniziale
initial_dtype = torch.float64 if PRECISION_MODE == 'full_64' else torch.float32
torch.set_default_dtype(initial_dtype)

# --- Hyperparameters ---
BASE_LR = 1e-3
ADAM_EPS = 1e-7
STAGED_TRAINING = False

MINIBATCH_INTERNAL = 2048 if PRECISION_MODE == 'full_64' else 2048*2
MINIBATCH_BOUNDARY = 256 if PRECISION_MODE == 'full_64' else 256*2

STATIC_WEIGHTS = {'bc': 10.0, 'physics': 10.0, 'data': 1.0}
STATIC_WEIGHT_STR = "BC=10-PHYS=10-DATA=1"
DYNAMIC_WEIGHT_STR = "Dynamic-Annealing"

PDE_WEIGHTS = {'momentum': 1.0, 'constitutive': 1.0}

VARIANCE_EPS = 1e-4

# --- Inverse Problem Settings ---
INVERSE_PROBLEM = False
GUESS_MULTIPLIER = 0.8
GUESS_MIN_EPS = 0.0
GUESS_MIN_ALPHA = 0.0

LOG_GRADIENTS_EVERY = 500
PLOT_EVERY = 500

BASE_OUTPUT_DIR = BASE_DIR / 'experiments_weighted'
RESULTS_CSV_PATH = BASE_DIR / 'results.csv'


# --- 4. EXPERIMENT UTILITIES ---
def setup_experiment_folder(parent_dir: Path, goal_folder: str):
    exp_dir = parent_dir / goal_folder
    plots_dir = exp_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir, plots_dir


def get_last(hist, key):
    return hist.losses[key][-1] if (key in hist.losses and hist.losses[key]) else 0


# --- 5. MAIN EXECUTION LOOP ---
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if initial_dtype == torch.float64:
        torch.backends.cuda.matmul.allow_tf32 = False
    else:
        torch.backends.cuda.matmul.allow_tf32 = True

    print(f"Using device: {device} with Precision Mode: {PRECISION_MODE} (Initial dtype: {initial_dtype})")

    for dataset_filename in DATASET_OPTIONS:
        DATASET_PATH = BASE_DIR.parent / 'COMSOL'/ '4roll' / dataset_filename
        if not DATASET_PATH.exists():
            raise FileNotFoundError(f"Dataset non trovato in: {DATASET_PATH.resolve()}")

        dataset_name_prefix = DATASET_PATH.stem
        print(f'\n=======================================================')
        print(f'=== PROCESSING DATASET: {dataset_name_prefix.upper()} ===')
        print(f'=======================================================')
        
        # Import aggiornato da src
        from FourRollMill.src.load_comsol import prepare_training_data
        data_bundle = prepare_training_data(
            str(DATASET_PATH), COMSOL_PARAMS,
            initial_dtype, device, variance_eps=VARIANCE_EPS,
            mask_multiplier=5.0
        )
        
        # Estrazione diretta delle strutture
        dataset = data_bundle['dataset']
        xy_pinn_data = data_bundle['data_subsets']['xy']
        psip_pinn_data = data_bundle['data_subsets']['psip']
        uv_pinn_data = data_bundle['data_subsets']['uv']
        
        validation_grid_u = data_bundle['validation_grid']
        stress_exact_grids = data_bundle['stress_exact_grids']
        VAR_WEIGHTS = data_bundle['var_weights']

        params = dataset['params']
        mu_s, mu_p, lam = params['mu_s'], params['mu_p'], params['lam']
        eps, alpha = params.get('eps', 0.0), params.get('alpha', 0.0)
        
        configs = list(itertools.product(LAYERS_OPTIONS, EPOCHS_OPTIONS, ACTIVATION_OPTIONS, LR_STRATEGY_OPTIONS, WEIGHTING_OPTIONS))
        print(f"Starting Weighted Grid Search over {len(configs)} configurations...")

        for layers_config, epochs, act_fn, lr_strat, weight_mode in configs:
            layers_str = format_layers_name(layers_config)
            act_str = get_activation_name(act_fn)
            config_name = f"{dataset_name_prefix}_L{layers_str}_E{epochs}_{act_str}_{lr_strat}_{weight_mode}"
            config_dir = BASE_OUTPUT_DIR / config_name
            config_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n=== Running Configuration: {config_name} ===")
            
            histories = {}
            final_models = {}
            final_phys_problems = {}
            is_dynamic = (weight_mode == 'dynamic')
            current_weight_str = DYNAMIC_WEIGHT_STR if is_dynamic else STATIC_WEIGHT_STR

            # --- LOOP SUI TRAINING GOALS INLINE ---
            for goal in GOALS_TO_RUN:
                goal_cfg = GOAL_CONFIGS[goal]
                inv_mode = INVERSE_PROBLEM and goal != 2
                label = goal_cfg['label']
                mode_param = 'comsol_full' if goal == 2 else 'semi_inverse'
                current_w = dict(goal_cfg['weights'])
                prefix = f"{goal}_{label}"

                print(f"  > {label} ({config_name})")
                exp_dir, plots_dir = setup_experiment_folder(config_dir, prefix)

                # Ripristino seed locale per la riproducibilità del trial
                torch.manual_seed(SEED)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(SEED)

                # Guess dinamici basati sui valori veri del dataset
                guess_mu_s = mu_s * GUESS_MULTIPLIER if inv_mode else mu_s
                guess_mu_p = mu_p * GUESS_MULTIPLIER if inv_mode else mu_p
                guess_lam = lam * GUESS_MULTIPLIER if inv_mode else lam
                guess_eps = max(eps * GUESS_MULTIPLIER, GUESS_MIN_EPS) if inv_mode else eps
                guess_alpha = max(alpha * GUESS_MULTIPLIER, GUESS_MIN_ALPHA) if inv_mode else alpha

                # Costruzione del Physics Problem
                if goal in [0, 2]:
                    phys_problem = ViscoelasticPhysics.from_dataset(
                        dataset, 
                        device=device, 
                        pde_weights=PDE_WEIGHTS
                    )
                    print(f"  > [Goal {goal}] Forward/Data Solver Parameters Loaded from Dataset:")
                    print(f"    mu_s: {phys_problem.mu_s.item():.4f}, mu_p: {phys_problem.mu_p.item():.4f}, lam: {phys_problem.lam.item():.4f}")
                    print(f"    alpha: {phys_problem.alpha.item():.4f}, eps: {phys_problem.eps.item():.4f}")
                else:
                    phys_problem = ViscoelasticPhysics(
                        mu_s=guess_mu_s, 
                        mu_p=guess_mu_p, 
                        lam=guess_lam, 
                        eps=guess_eps,
                        alpha=guess_alpha,
                        pde_weights=PDE_WEIGHTS,
                        inverse_mode=inv_mode,
                        real_mu_s=mu_s,
                        real_mu_p=mu_p,
                        real_lam=lam,
                        real_eps=eps,
                        real_alpha=alpha
                    ).to(device)

                # Costruzione modelli
                layers_psi = layers_config[:-1] + [1]
                layers_p = layers_config[:-1] + [1]
                layers_tau = layers_config[:-1] + [3]

                model_psi = FCN(layers=layers_psi, activation_fn=act_fn).to(device)
                model_p = FCN(layers=layers_p, activation_fn=act_fn).to(device)
                model_tau = FCN(layers=layers_tau, activation_fn=act_fn).to(device)
                model_combined = ViscoelasticCombinedModel(model_psi, model_p, model_tau)

                # Determinazione pesi effettivi, sovrascrive config
                run_is_dynamic = is_dynamic if goal != 2 else False
                effective_w = dict(current_w)
                if not run_is_dynamic and goal != 2:
                    effective_w['bc'] *= STATIC_WEIGHTS['bc']
                    effective_w['physics'] *= STATIC_WEIGHTS['physics']
                    effective_w['data'] *= STATIC_WEIGHTS['data']

                # Selezione dei dati in base al Goal
                if mode_param == 'comsol_full':
                    pinn_data_internal = (xy_pinn_data, psip_pinn_data)
                    var_weights = VAR_WEIGHTS
                elif goal == 1 or mode_param == 'semi_inverse':
                    pinn_data_internal = (xy_pinn_data, uv_pinn_data)
                    var_weights = VAR_WEIGHTS
                else:
                    pinn_data_internal = (xy_pinn_data, psip_pinn_data)
                    var_weights = VAR_WEIGHTS if goal != 2 else None

                # Generazione dinamica delle condizioni al contorno
                xy_master_boundary, dir_master_boundary, neu_master_boundary, norm_master_boundary = phys_problem.apply_boundary_conditions(
                    data_bundle['boundary_groups']
                )
                pinn_data_boundary = (xy_master_boundary, dir_master_boundary, neu_master_boundary, norm_master_boundary)

                max_lbfgs = MAX_LBFGS_ITERS if MAX_LBFGS_ITERS is not None else int(0.1 * epochs)
                train_config = TrainingConfig(
                    epochs=epochs,
                    base_lr=BASE_LR,
                    adam_eps=ADAM_EPS,
                    lr_strategy=lr_strat,
                    staged_training=(STAGED_TRAINING and goal not in [0, 2]),
                    precision_mode=PRECISION_MODE,
                    max_lbfgs_iters=max_lbfgs,
                    minibatch_internal=MINIBATCH_INTERNAL,
                    minibatch_boundary=MINIBATCH_BOUNDARY,
                    dynamic_weighting=run_is_dynamic,
                    loss_weights=effective_w,
                    mode=mode_param,
                    variance_weights=var_weights,
                    log_gradients_every=LOG_GRADIENTS_EVERY,
                    plot_every=PLOT_EVERY,
                    experiment_name=f"Viscoelastic {label}",
                    val_label="u (Velocity)",
                )

                try:
                    history = train_ViscoelasticPINN(
                        model=model_combined,
                        config=train_config,
                        data_internal=pinn_data_internal,
                        data_boundary=pinn_data_boundary,
                        validation_grid=validation_grid_u,
                        physics_problem=phys_problem,
                        collocation_points=xy_pinn_data,
                        plots_dir=str(plots_dir),
                        final_dir=str(exp_dir),
                        stress_exact_grids=stress_exact_grids,
                    )

                    eff = phys_problem.get_logged_parameters()
                    print(f"  [Parametri Fisici Finali - {label}] mu_s: {eff['mu_s']:.5f}, mu_p: {eff['mu_p']:.5f}, lam: {eff['lam']:.5f}, eps: {eff['eps']:.5f}, alpha: {eff['alpha']:.5f}")

                    # Metriche multi-campo evitando lo spacchettamento manuale delle variabili del dataset
                    fields_keys = ['u', 'p', 'tau_xx', 'tau_xy', 'tau_yy']
                    fields_exact_for_metrics = {k: dataset[k] for k in fields_keys}
                    
                    visco_metrics = compute_viscoelastic_metrics(
                        model_combined, phys_problem, data_bundle['xy_grid_flat'], fields_exact_for_metrics
                    )

                    l2_values = [v[0] for v in visco_metrics.values() if v[0] > 1e-10]
                    max_values = [v[1] for v in visco_metrics.values()]
                    l2_avg = sum(l2_values) / len(l2_values) if l2_values else 0.0
                    max_global = max(max_values) if max_values else 0.0

                    lr_log_str = str(BASE_LR) if lr_strat == 'cosine' else f"[{BASE_LR}]"
                    log_data = {
                        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'Architecture': str(layers_config),
                        'Activation_Func': get_activation_name(act_fn), 'Epochs': epochs, 'Run_Type': label,
                        'Optimizer': 'Adam', 'Learning_Rate': lr_log_str,
                        'Loss_Total': get_last(history, 'total_loss'),
                        'Loss_Physics': get_last(history, 'pde_loss'),
                        'Loss_Boundary': get_last(history, 'bc_loss'),
                        'Loss_Data': get_last(history, 'data_loss'),
                        'L2_Relative_Error': l2_avg, 'Max_Relative_Error_Peak': max_global,
                        'L2_u': visco_metrics['u'][0], 'Max_u': visco_metrics['u'][1],
                        'L2_p': visco_metrics['p'][0], 'Max_p': visco_metrics['p'][1],
                        'L2_tau_xx': visco_metrics['tau_xx'][0], 'Max_tau_xx': visco_metrics['tau_xx'][1],
                        'L2_tau_xy': visco_metrics['tau_xy'][0], 'Max_tau_xy': visco_metrics['tau_xy'][1],
                        'L2_tau_yy': visco_metrics['tau_yy'][0], 'Max_tau_yy': visco_metrics['tau_yy'][1],
                        'Seed': SEED, 'Dataset': dataset_name_prefix,
                        'n_points': xy_pinn_data.shape[0] if goal in [1, 2] else 0,
                        'Loss_Weight': current_weight_str
                    }
                    update_results_csv(str(RESULTS_CSV_PATH), log_data)
                    
                    histories[label] = history
                    final_models[label] = model_combined
                    final_phys_problems[label] = phys_problem

                    if inv_mode:
                        history.plot_physical_parameters(
                            true_etas=mu_s,
                            true_etap=mu_p,
                            true_lam=lam,
                            true_epsilon=eps,
                            true_alpha=alpha,
                            save_path=str(exp_dir / 'VE_parameters_evolution.png'),
                            experiment_name=f"Viscoelastic {label}"
                        )

                except Exception as e:
                    print(f"  [X] Errore nel training {label}: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    # Pulizia in caso di fallimento della specifica run per salvaguardare la VRAM
                    if 'model_combined' in locals(): del model_combined
                    if 'model_psi' in locals(): del model_psi
                    if 'model_p' in locals(): del model_p
                    if 'model_tau' in locals(): del model_tau
                    if 'phys_problem' in locals(): del phys_problem
                    
                finally:
                    # Pulizia aggressiva della VRAM ad ogni iterazione
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            if not final_models:
                continue

            # --- COMPARISON PLOTS ---
            print(f"  > Generating Comparisons for {config_name}...")
            results_dir = config_dir / 'comparisons'
            results_dir.mkdir(parents=True, exist_ok=True)

            # Problema fisico di fallback con parametri nominali/ground truth
            comp_phys_problem = ViscoelasticPhysics(
                mu_s=dataset['params']['mu_s'], 
                mu_p=dataset['params']['mu_p'], 
                lam=dataset['params']['lam'], 
                eps=dataset['params'].get('eps', 0.0), 
                alpha=dataset['params'].get('alpha', 0.0), 
                pde_weights=PDE_WEIGHTS
            ).to(device)

            model_results = []
            model_results_multi = []
            for label, model in final_models.items():
                model.eval()
                with torch.set_grad_enabled(True):
                    x_input = data_bundle['xy_grid_flat'].clone().to(next(model.parameters()).dtype).requires_grad_(True)
                    active_phys_problem = final_phys_problems.get(label, comp_phys_problem)
                    u_p, _, p_p, tau_p = active_phys_problem.get_velocity(model, x_input)
                    pred_u = u_p.detach().cpu().to(torch.float32).view(-1)
                    pred_p = p_p.detach().cpu().to(torch.float32).view(-1)
                    pred_txx = tau_p[:, 0].detach().cpu().to(torch.float32).view(-1)
                    pred_txy = tau_p[:, 1].detach().cpu().to(torch.float32).view(-1)
                    pred_tyy = tau_p[:, 2].detach().cpu().to(torch.float32).view(-1)
                
                model_results.append({'T_pred': pred_u, 'label': label})
                model_results_multi.append({
                    'label': label,
                    'fields': {'u': pred_u, 'p': pred_p, 'tau_xx': pred_txx, 'tau_xy': pred_txy, 'tau_yy': pred_tyy}
                })

            hparams = {
                'arch': layers_str, 
                'epochs': str(epochs), 
                'act': act_str, 
                'lr_strategy': lr_strat, 
                'weight': DYNAMIC_WEIGHT_STR if weight_mode == 'dynamic' else STATIC_WEIGHT_STR
            }
            triang = data_bundle['triang']

            if model_results:
                plot2D_unified_comparison(
                    triang, dataset['u'].cpu().view(-1), model_results, hparams, 
                    save_path=str(results_dir / 'Comparison_Unified_ErrorMaps.png')
                )

            if model_results_multi:
                fields_exact_cpu = {
                    'u': dataset['u'].cpu().view(-1), 
                    'p': dataset['p'].cpu().view(-1),
                    'tau_xx': dataset['tau_xx'].cpu().view(-1), 
                    'tau_xy': dataset['tau_xy'].cpu().view(-1), 
                    'tau_yy': dataset['tau_yy'].cpu().view(-1)
                }
                plot2D_viscoelastic_comparison(
                    triang, fields_exact_cpu, model_results_multi, hparams,
                    save_path=str(results_dir / 'Comparison_Viscoelastic_AllFields.png')
                )

            if len(histories) > 1:
                labels_list = list(histories.keys())
                hist_list = [histories[l] for l in labels_list]
                plot_loss_comparison(hist_list, labels_list, save_path=str(results_dir / 'Comparison_Loss_All_Goals.png'))

            # Pulizia VRAM a fine configurazione
            del histories, final_models, final_phys_problems, model_results, model_results_multi, comp_phys_problem
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print("\nWeighted Grid Search configurations completed.")
