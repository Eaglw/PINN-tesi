import torch
import torch.nn as nn
import matplotlib.pyplot as plt
plt.switch_backend('Agg')  # Forza backend non interattivo per evitare pause
import gc
import os
import sys
import itertools
from datetime import datetime

# Ottimizzazioni per GPU Ampere (es. RTX 3080)
torch.set_float32_matmul_precision('high') # Abilita TF32 per i matmul, enorme boost di velocità (fino a 2-3x)
torch.backends.cudnn.benchmark = True # Ottimizza i kernel CUDNN

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# Import funzioni esterne
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from func.logging_utils import update_results_csv
from func.graphic_func import plot2D_unified_comparison, plot_loss_comparison, plot2D_viscoelastic_comparison

# Import locali Viscoelastic
from Viscoelastic.src.Viscoelastic_PINN import (
    train_ViscoelasticPINN, TrainingConfig,
    FCN, ViscoelasticCombinedModel,
    get_activation_name, format_layers_name,
    compute_viscoelastic_metrics
)
from Viscoelastic.src.Viscoelastic_physics import ViscoelasticPhysics


# ╔══════════════════════════════════════════════════╗
# ║              CONFIGURATION BLOCK                 ║
# ╚══════════════════════════════════════════════════╝

# --- Hardware & Precision ---
PRECISION_MODE = 'staged'           # 'full_32' | 'staged' | 'full_64'
SEED = 123

# --- Training Goals ---
DATASET_OPTIONS = ['Oldroyd.csv']

COMSOL_PARAMS = {
    'mu_s': 0.005,   # Viscosità solvente [Pa·s]
    'mu_p': 0.005,   # Viscosità polimerica [Pa·s]
    'lam': 0.1,      # Tempo di rilassamento [s]
    'eps': 0.0,      # Parametro PTT
    'alpha': 0.0,    # Parametro Giesekus
    'rho': 1.0,      # Densità [kg/m³]
}

# 0=PurePhys, 1=Phys+Data, 2=SoloData
GOALS_TO_RUN = [1]

GOAL_CONFIGS = {
    0: {'label': 'PurePhys',  'weights': {'bc': 1.0, 'physics': 1.0, 'data': 0.0}, 'mode': 'standard'},
    1: {'label': 'Phys+Data', 'weights': {'bc': 1.0, 'physics': 1.0, 'data': 1.0}, 'mode': 'semi_inverse'},
    2: {'label': 'SoloData',  'weights': {'bc': 1.0, 'physics': 0.0, 'data': 1.0}, 'mode': 'standard'},
}

# --- Architecture (Grid Search) ---
LAYERS_OPTIONS = [[2, 128, 128, 128, 128, 128, 128, 128, 128, 1]] #VENet 8x128
EPOCHS_OPTIONS = [10000]
ACTIVATION_OPTIONS = [nn.SiLU]
LR_STRATEGY_OPTIONS = ['cosine']
WEIGHTING_OPTIONS = ['dynamic']

# --- L-BFGS ---
MAX_LBFGS_ITERS = int(0.1*EPOCHS_OPTIONS[0])

# --- Optimizer ---
BASE_LR = 1e-3
ADAM_EPS = 1e-7

# --- Staged Training ---
STAGED_TRAINING = True

# --- Mini-Batching ---
MINIBATCH_INTERNAL = 2048
MINIBATCH_BOUNDARY = 256

# --- Loss Weighting ---
STATIC_WEIGHTS = {'bc': 1.0, 'physics': 10.0, 'data': 100.0}
STATIC_WEIGHT_STR = "BC=1-PHYS=10-DATA=100"
DYNAMIC_WEIGHT_STR = "Dynamic-Annealing"

# --- PDE Weights (Momentum vs Constitutive) ---
PDE_WEIGHTS = {'momentum': 10.0, 'constitutive': 1.0}

# --- Data ---
NUM_DATA_SUBSET = 5000
VARIANCE_EPS = 1e-4  # Epsilon per varianze: 1.0 disabilita lo scaling aggressivo, 1e-8 lo abilita

# --- Inverse Problem ---
INVERSE_PROBLEM = True
GUESS_MULTIPLIER = 0.8 # Moltiplicatore per i guess iniziali (es. 0.8 = 80% del valore vero)
GUESS_MIN_EPS = 0.1   # Guess minimo se il valore vero è 0 (per PTT)
GUESS_MIN_ALPHA = 0.1 # Guess minimo se il valore vero è 0 (per Giesekus)

# --- Logging & Plotting ---
# Riduciamo la frequenza per non intasare l'output se l'utente desidera
LOG_GRADIENTS_EVERY = 500
PLOT_EVERY = 500

# --- Paths ---
BASE_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiments_weighted')
RESULTS_CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.csv')



# ╔══════════════════════════════════════════════════╗
# ║              SETUP & DATA LOADING                ║
# ╚══════════════════════════════════════════════════╝

# --- SETUP DISPOSITIVO ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

initial_dtype = torch.float64 if PRECISION_MODE == 'full_64' else torch.float32
torch.set_default_dtype(initial_dtype)

if initial_dtype == torch.float64:
    torch.backends.cuda.matmul.allow_tf32 = False
else:
    torch.backends.cuda.matmul.allow_tf32 = True

# Per reti FC con mini-batch dinamici, benchmark non offre vantaggi
# e causa overhead per il re-profiling ad ogni cambio di dimensione.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False

print(f"Using device: {device} with Precision Mode: {PRECISION_MODE} (Initial dtype: {initial_dtype})")


for dataset_filename in DATASET_OPTIONS:
    # Cerca il file in diverse posizioni possibili per massima flessibilità
    possible_paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset', dataset_filename),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'COMSOL', dataset_filename),
        dataset_filename
    ]
    DATASET_PATH = None
    for p in possible_paths:
        if os.path.exists(p):
            DATASET_PATH = p
            break
            
    if DATASET_PATH is None:
        print(f"❌ Dataset non trovato nelle posizioni note: {possible_paths}")
        sys.exit(1)

    dataset_name_prefix = os.path.basename(DATASET_PATH).replace('.csv', '')
    print(f'\n=======================================================')
    print(f'=== PROCESSING DATASET: {dataset_name_prefix.upper()} ===')
    print(f'=======================================================')
    
    from Viscoelastic.dataset.load_comsol import prepare_training_data
    data_bundle = prepare_training_data(
        DATASET_PATH, COMSOL_PARAMS, NUM_DATA_SUBSET,
        initial_dtype, device, variance_eps=VARIANCE_EPS
    )
    
    dataset = data_bundle['dataset']
    xy_grid_flat = data_bundle['xy_grid_flat']
    triang = data_bundle['triang']
    validation_grid_u = data_bundle['validation_grid']
    stress_exact_grids = data_bundle['stress_exact_grids']
    VAR_WEIGHTS = data_bundle['var_weights']
    
    u_exact = dataset['u']
    v_exact = dataset['v']
    p_exact = dataset['p']
    tau_xx_exact = dataset['tau_xx']
    tau_xy_exact = dataset['tau_xy']
    tau_yy_exact = dataset['tau_yy']
    
    xy_pinn_data = data_bundle['data_subsets']['xy']
    psip_pinn_data = data_bundle['data_subsets']['psip']
    uv_pinn_data = data_bundle['data_subsets']['uv']
    
    xy_master_boundary = data_bundle['boundaries']['xy']
    dir_master_boundary = data_bundle['boundaries']['dir']
    neu_master_boundary = data_bundle['boundaries']['neu']
    norm_master_boundary = data_bundle['boundaries']['norm']

    params = dataset['params']
    mu_s = params['mu_s']
    mu_p = params['mu_p']
    lam = params['lam']
    eps = params.get('eps', 0.0)
    alpha = params.get('alpha', 0.0)


    # ╔══════════════════════════════════════════════════╗
    # ║              HELPER FUNCTIONS                    ║
    # ╚══════════════════════════════════════════════════╝

    def setup_experiment_folder(parent_dir, goal_folder):
        exp_dir = os.path.join(parent_dir, goal_folder)
        plots_dir = os.path.join(exp_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        return exp_dir, plots_dir

    def get_last(hist, key):
        return hist.losses[key][-1] if (key in hist.losses and hist.losses[key]) else 0


    # ╔══════════════════════════════════════════════════╗
    # ║              GRID SEARCH EXECUTION               ║
    # ╚══════════════════════════════════════════════════╝

    configs = list(itertools.product(LAYERS_OPTIONS, EPOCHS_OPTIONS, ACTIVATION_OPTIONS, LR_STRATEGY_OPTIONS, WEIGHTING_OPTIONS))
    print(f"Starting Weighted Grid Search over {len(configs)} configurations...")

    for layers_config, epochs, act_fn, lr_strat, weight_mode in configs:
        torch.set_default_dtype(initial_dtype)
        layers_str = format_layers_name(layers_config)
        act_str = get_activation_name(act_fn)
        config_name = f"{dataset_name_prefix}_L{layers_str}_E{epochs}_{act_str}_{lr_strat}_{weight_mode}"
    
        config_dir = os.path.join(BASE_OUTPUT_DIR, config_name)
        os.makedirs(config_dir, exist_ok=True)
    
        print(f"\n=== Running Configuration: {config_name} ===")
        histories, final_models, final_phys_problems = {}, {}, {}
        is_dynamic = (weight_mode == 'dynamic')
        current_weight_str = DYNAMIC_WEIGHT_STR if is_dynamic else STATIC_WEIGHT_STR

        for goal in GOALS_TO_RUN:
            goal_cfg = GOAL_CONFIGS[goal]
        
            # Instantiate Physics Problem inside the loop so parameters reset for each goal
            inv_mode = INVERSE_PROBLEM and goal != 2
            
            # Calcolo dei guess dinamici basati sui valori veri del dataset
            guess_mu_s = mu_s * GUESS_MULTIPLIER if inv_mode else mu_s
            guess_mu_p = mu_p * GUESS_MULTIPLIER if inv_mode else mu_p
            guess_lam = lam * GUESS_MULTIPLIER if inv_mode else lam
            guess_eps = max(eps * GUESS_MULTIPLIER, GUESS_MIN_EPS) if inv_mode else eps
            guess_alpha = max(alpha * GUESS_MULTIPLIER, GUESS_MIN_ALPHA) if inv_mode else alpha
            
            if goal in [0, 2]:
                phys_problem = ViscoelasticPhysics.from_dataset(
                    dataset, 
                    device=device, 
                    pde_weights=PDE_WEIGHTS
                ).to(initial_dtype)
                
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
                ).to(device).to(initial_dtype)
            label = goal_cfg['label']
            if goal == 2:
                mode_param = 'comsol_full'
            else:
                mode_param = 'semi_inverse'
            current_w = dict(goal_cfg['weights'])

            prefix = f"{goal}_{label}"
            print(f"  > {label} ({config_name})")
        
            exp_dir, plots_dir = setup_experiment_folder(config_dir, prefix)
        
            torch.manual_seed(SEED)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(SEED)
            
            torch.set_default_dtype(initial_dtype)
        
            # Dynamic weighting disattivato per SoloData
            run_is_dynamic = is_dynamic if goal != 2 else False

            # Pesi effettivi: applica pesi statici se non dynamic
            effective_w = dict(current_w)
            if not run_is_dynamic and goal != 2:
                effective_w['bc'] *= STATIC_WEIGHTS['bc']
                effective_w['physics'] *= STATIC_WEIGHTS['physics']
                effective_w['data'] *= STATIC_WEIGHTS['data']

            # Configurazione dati per Goal
            if mode_param == 'comsol_full':
                pinn_data_internal_fresh = (xy_pinn_data, psip_pinn_data)
                var_weights = VAR_WEIGHTS
            elif goal == 1 or mode_param == 'semi_inverse':
                pinn_data_internal_fresh = (xy_pinn_data, uv_pinn_data)
                var_weights = VAR_WEIGHTS
            else:
                pinn_data_internal_fresh = (xy_pinn_data, psip_pinn_data)
                var_weights = VAR_WEIGHTS if goal != 2 else None

            pinn_data_boundary_fresh = (xy_master_boundary, dir_master_boundary, neu_master_boundary, norm_master_boundary)
        
            # Costruzione modello
            layers_psi = layers_config[:-1] + [1]
            layers_p = layers_config[:-1] + [1]
            layers_tau = layers_config[:-1] + [3]
        
            model_psi = FCN(layers=layers_psi, activation_fn=act_fn).to(device).to(initial_dtype)
            model_p = FCN(layers=layers_p, activation_fn=act_fn).to(device).to(initial_dtype)
            model_tau = FCN(layers=layers_tau, activation_fn=act_fn).to(device).to(initial_dtype)
            model_combined = ViscoelasticCombinedModel(model_psi, model_p, model_tau)

            # Costruzione TrainingConfig
            use_staged = STAGED_TRAINING and goal != 2 and goal != 0
            train_config = TrainingConfig(
                epochs=epochs,
                base_lr=BASE_LR,
                adam_eps=ADAM_EPS,
                lr_strategy=lr_strat,
                staged_training=use_staged,
                precision_mode=PRECISION_MODE,
                max_lbfgs_iters=MAX_LBFGS_ITERS,
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
                    data_internal=pinn_data_internal_fresh,
                    data_boundary=pinn_data_boundary_fresh,
                    validation_grid=validation_grid_u,
                    physics_problem=phys_problem,
                    collocation_points=xy_pinn_data,
                    plots_dir=plots_dir,
                    final_dir=exp_dir,
                    stress_exact_grids=stress_exact_grids,
                )
            
                eff = phys_problem.get_logged_parameters()
                print(f"  [Parametri Fisici Finali - {label}] mu_s: {eff['mu_s']:.5f}, mu_p: {eff['mu_p']:.5f}, lam: {eff['lam']:.5f}, eps: {eff['eps']:.5f}, alpha: {eff['alpha']:.5f}")
            
                # Metriche multi-campo
                fields_exact_for_metrics = {
                    'u': u_exact, 'p': p_exact,
                    'tau_xx': tau_xx_exact, 'tau_xy': tau_xy_exact, 'tau_yy': tau_yy_exact
                }
                visco_metrics = compute_viscoelastic_metrics(
                    model_combined, phys_problem, xy_grid_flat, fields_exact_for_metrics
                )
            
                # Metriche aggregate
                l2_values = [v[0] for v in visco_metrics.values() if v[0] > 1e-10]
                max_values = [v[1] for v in visco_metrics.values()]
                l2_avg = sum(l2_values) / len(l2_values) if l2_values else 0.0
                max_global = max(max_values) if max_values else 0.0
            
                lr_log_str = str(BASE_LR) if lr_strat == 'cosine' else f"[{BASE_LR}]"
            
                log_data = {
                    'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'Architecture': str(layers_config),
                    'Activation_Func': act_str, 'Epochs': epochs, 'Run_Type': label,
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
                    'Seed': SEED,
                'Dataset': dataset_name_prefix,
                    'n_points': xy_pinn_data.shape[0] if goal in [1, 2] else 0,
                    'Loss_Weight': current_weight_str
                }
                update_results_csv(RESULTS_CSV_PATH, log_data)
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
                        save_path=os.path.join(exp_dir, 'VE_parameters_evolution.png'),
                        experiment_name=f"Viscoelastic {label}"
                    )
            except Exception as e:
                print(f"  [X] Errore nel training {label}: {e}")
                import traceback
                traceback.print_exc()
                # Pulizia della memoria GPU in caso di errore per non influenzare i goal successivi
                if 'model_combined' in locals(): del model_combined
                if 'model_psi' in locals(): del model_psi
                if 'model_p' in locals(): del model_p
                if 'model_tau' in locals(): del model_tau
                if 'phys_problem' in locals(): del phys_problem
                if 'optimizer' in locals(): del optimizer
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
 
         # --- COMPARISON PLOTS ---
        print(f"  > Generating Comparisons for {config_name}...")
        results_dir = os.path.join(config_dir, 'comparisons')
        os.makedirs(results_dir, exist_ok=True)
    
        # Fallback clean physics problem for velocity evaluation in comparison plots
        comp_phys_problem = ViscoelasticPhysics(
            mu_s=mu_s, mu_p=mu_p, lam=lam, eps=eps, alpha=alpha, pde_weights=PDE_WEIGHTS
        ).to(device).to(initial_dtype)
    
        model_results = []
        model_results_multi = []
        for label, model in final_models.items():
            model.eval()
            with torch.set_grad_enabled(True):
                x_input = xy_grid_flat.clone().to(next(model.parameters()).dtype).requires_grad_(True)
                active_phys_problem = final_phys_problems.get(label, comp_phys_problem)
                u_p, _, p_p, _ = active_phys_problem.get_velocity(model, x_input)
                out = model(x_input)
                pred_u = u_p.detach().cpu().to(torch.float32).view(-1)
                pred_p = p_p.detach().cpu().to(torch.float32).view(-1)
                pred_txx = out[:, 2].detach().cpu().to(torch.float32).view(-1)
                pred_txy = out[:, 3].detach().cpu().to(torch.float32).view(-1)
                pred_tyy = out[:, 4].detach().cpu().to(torch.float32).view(-1)
            model_results.append({'T_pred': pred_u, 'label': label})
            model_results_multi.append({
                'label': label,
                'fields': {'u': pred_u, 'p': pred_p, 'tau_xx': pred_txx, 'tau_xy': pred_txy, 'tau_yy': pred_tyy}
            })
    
        if model_results:
            hparams = {'arch': layers_str, 'epochs': str(epochs), 'act': act_str, 'lr_strategy': lr_strat, 'weight': current_weight_str}
            plot2D_unified_comparison(triang, u_exact.cpu().view(-1), model_results, hparams, save_path=os.path.join(results_dir, 'Comparison_Unified_ErrorMaps.png'))
    
        if model_results_multi:
            fields_exact_cpu = {
                'u': u_exact.cpu().view(-1), 'p': p_exact.cpu().view(-1),
                'tau_xx': tau_xx_exact.cpu().view(-1), 'tau_xy': tau_xy_exact.cpu().view(-1), 'tau_yy': tau_yy_exact.cpu().view(-1)
            }
            plot2D_viscoelastic_comparison(
                triang, fields_exact_cpu, model_results_multi, hparams,
                save_path=os.path.join(results_dir, 'Comparison_Viscoelastic_AllFields.png')
            )
        if len(histories) > 1:
            labels_list = list(histories.keys())
            hist_list = [histories[l] for l in labels_list]
            plot_loss_comparison(hist_list, labels_list, save_path=os.path.join(results_dir, 'Comparison_Loss_All_Goals.png'))

        # --- VRAM Cleanup: previene OOM nelle grid search lunghe ---
        del histories, final_models, model_results, model_results_multi
        if 'final_phys_problems' in locals(): del final_phys_problems
        if 'comp_phys_problem' in locals(): del comp_phys_problem
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nWeighted Grid Search configurations completed.")
