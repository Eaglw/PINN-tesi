import torch
import torch.nn as nn
import matplotlib.pyplot as plt
plt.switch_backend('Agg')  # Forza backend non interattivo per evitare pause
import gc
import os
import sys
import itertools
from datetime import datetime

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Import funzioni esterne
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from func.logging_utils import compute_viscoelastic_metrics, update_results_csv
from func.graphic_func import plot2D_unified_comparison, plot_loss_comparison, plot2D_viscoelastic_comparison

# Import locali Viscoelastic
from Viscoelastic.src.Viscoelastic_PINN import (
    train_ViscoelasticPINN, TrainingConfig,
    FCN, ViscoelasticCombinedModel, VelocityInferenceWrapper,
    get_activation_name, format_layers_name
)
from Viscoelastic.src.Viscoelastic_physics import ViscoelasticPhysics, generate_boundaries


# ╔══════════════════════════════════════════════════╗
# ║              CONFIGURATION BLOCK                 ║
# ╚══════════════════════════════════════════════════╝

# --- Hardware & Precision ---
PRECISION_MODE = 'staged'           # 'full_32' | 'staged' | 'full_64'
SEED = 123

# --- Training Goals ---
# 0=PurePhys, 1=Phys+Data, 2=SoloData
GOALS_TO_RUN = [0, 1, 2]

GOAL_CONFIGS = {
    0: {'label': 'PurePhys',  'weights': {'bc': 1.0, 'physics': 1.0, 'data': 0.0}, 'mode': 'standard'},
    1: {'label': 'Phys+Data', 'weights': {'bc': 1.0, 'physics': 1.0, 'data': 1.0}, 'mode': 'semi_inverse'},
    2: {'label': 'SoloData',  'weights': {'bc': 1.0, 'physics': 0.0, 'data': 1.0}, 'mode': 'standard'},
}

# --- Architecture (Grid Search) ---
LAYERS_OPTIONS = [[2, 128, 128, 128, 128, 128, 128, 128, 128, 1]] #VENet 8x128
EPOCHS_OPTIONS = [15000]
ACTIVATION_OPTIONS = [nn.SiLU]
LR_STRATEGY_OPTIONS = ['cosine']
WEIGHTING_OPTIONS = ['dynamic']

# --- Optimizer ---
BASE_LR = 1e-3
ADAM_EPS = 1e-7

# --- Staged Training ---
STAGED_TRAINING = True

# --- Mini-Batching ---
MINIBATCH_INTERNAL = 1024
MINIBATCH_BOUNDARY = 256

# --- Loss Weighting ---
STATIC_WEIGHTS = {'bc': 1.0, 'physics': 10.0, 'data': 100.0}
STATIC_WEIGHT_STR = "BC=1-PHYS=10-DATA=100"
DYNAMIC_WEIGHT_STR = "Dynamic-Annealing"

# --- PDE Weights (Momentum vs Constitutive) ---
PDE_WEIGHTS = {'momentum': 10.0, 'constitutive': 1.0}

# --- Data ---
NUM_DATA_SUBSET = 5000
VARIANCE_EPS = 1e-8  # Epsilon per varianze: 1.0 disabilita lo scaling aggressivo, 1e-8 lo abilita

# --- Inverse Problem ---
INVERSE_PROBLEM = True
GUESS_MU_S = 0.004  # True is 0.005
GUESS_MU_P = 0.004  # True is 0.005
GUESS_LAM = 0.8     # True is 1.0

# --- L-BFGS ---
MAX_LBFGS_ITERS = 500

# --- Logging & Plotting ---
LOG_GRADIENTS_EVERY = 500
PLOT_EVERY = 500

# --- Paths ---
BASE_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiments_weighted')
RESULTS_CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.csv')
DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset', 'oldroydb_clean.pt')


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

# --- CARICAMENTO DATASET ---
if not os.path.exists(DATASET_PATH):
    print(f"❌ Dataset non trovato in: {DATASET_PATH}")
    sys.exit(1)

dataset = torch.load(DATASET_PATH, map_location=device, weights_only=False)
for key in ['coords', 'u', 'v', 'p', 'psi', 'tau_xx', 'tau_xy', 'tau_yy', 'u_exact', 'p_exact', 'psi_exact', 'tau_xx_exact', 'tau_xy_exact', 'tau_yy_exact']:
    if key in dataset:
        dataset[key] = dataset[key].to(initial_dtype)
params = dataset['params']

Lx, Ly = params['L'], params['H']
mu_s = params.get('mu_s', 0.005)
mu_p = params.get('mu_p', 0.005)
lam = params.get('lam', 1.0)
u_max = params['u_max']
print(f"Dataset caricato: L={Lx}, H={Ly}, mu_s={mu_s}, mu_p={mu_p}, lam={lam}, u_max={u_max}")

# --- Preparazione Grid e Soluzioni Esatte ---
xy_grid_flat = dataset['coords']
u_exact = dataset['u_exact']
p_exact = dataset['p_exact']
psi_exact = dataset['psi_exact']
v_exact = torch.zeros_like(u_exact)
tau_xx_exact = dataset.get('tau_xx_exact', torch.zeros_like(u_exact))
tau_xy_exact = dataset.get('tau_xy_exact', torch.zeros_like(u_exact))
tau_yy_exact = dataset.get('tau_yy_exact', torch.zeros_like(u_exact))

x_sorted = torch.unique(xy_grid_flat[:, 0], sorted=True)
y_sorted = torch.unique(xy_grid_flat[:, 1], sorted=True)
Nx_dom, Ny_dom = len(x_sorted), len(y_sorted)

X = xy_grid_flat[:, 0].reshape(Ny_dom, Nx_dom)
Y = xy_grid_flat[:, 1].reshape(Ny_dom, Nx_dom)
U_grid = u_exact.reshape(Ny_dom, Nx_dom)
P_grid = p_exact.reshape(Ny_dom, Nx_dom)
TAU_XX_grid = tau_xx_exact.reshape(Ny_dom, Nx_dom)
TAU_XY_grid = tau_xy_exact.reshape(Ny_dom, Nx_dom)
TAU_YY_grid = tau_yy_exact.reshape(Ny_dom, Nx_dom)

validation_grid_u = (xy_grid_flat, U_grid, X, Y)
stress_exact_grids = {'p': P_grid, 'tau_xx': TAU_XX_grid, 'tau_xy': TAU_XY_grid, 'tau_yy': TAU_YY_grid}

# Varianze per normalizzazione (Goal 1 - ViscoelasticNet)
sigma2_u   = max(u_exact.var().item(), VARIANCE_EPS)
sigma2_v   = max(v_exact.var().item(), VARIANCE_EPS)
sigma2_p   = max(p_exact.var().item(), VARIANCE_EPS)
sigma2_txx = max(tau_xx_exact.var().item(), VARIANCE_EPS)
sigma2_txy = max(tau_xy_exact.var().item(), VARIANCE_EPS)
sigma2_tyy = max(tau_yy_exact.var().item(), VARIANCE_EPS)
print(f"Variances for normalization: u={sigma2_u:.2e}, v={sigma2_v:.2e}, p={sigma2_p:.2e}, txx={sigma2_txx:.2e}, txy={sigma2_txy:.2e}, tyy={sigma2_tyy:.2e}")

VAR_WEIGHTS = {'u': sigma2_u, 'v': sigma2_v, 'p': sigma2_p, 'txx': sigma2_txx, 'txy': sigma2_txy, 'tyy': sigma2_tyy}

# --- BOUNDARY CONDITIONS ---
xy_master_boundary, dir_master_boundary, neu_master_boundary, norm_master_boundary = generate_boundaries(Lx, Ly, u_max, p_exact, stress_exact_grids, Nx_dom, Ny_dom, device)

# --- Data Subset ---
torch.manual_seed(42)
idx = torch.randperm(xy_grid_flat.shape[0])[:NUM_DATA_SUBSET]
xy_pinn_data = xy_grid_flat[idx]
psip_pinn_data = torch.cat([psi_exact[idx], p_exact[idx], tau_xx_exact[idx], tau_xy_exact[idx], tau_yy_exact[idx]], dim=1)
uv_pinn_data = torch.cat([u_exact[idx], v_exact[idx]], dim=1)

# GPU Pre-cast al dtype iniziale
xy_pinn_data = xy_pinn_data.to(initial_dtype)
psip_pinn_data = psip_pinn_data.to(initial_dtype)
uv_pinn_data = uv_pinn_data.to(initial_dtype)
xy_master_boundary = xy_master_boundary.to(initial_dtype)
dir_master_boundary = dir_master_boundary.to(initial_dtype)
neu_master_boundary = neu_master_boundary.to(initial_dtype)
norm_master_boundary = norm_master_boundary.to(initial_dtype)


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
    config_name = f"L{layers_str}_E{epochs}_{act_str}_{lr_strat}_{weight_mode}"
    
    config_dir = os.path.join(BASE_OUTPUT_DIR, config_name)
    os.makedirs(config_dir, exist_ok=True)
    
    print(f"\n=== Running Configuration: {config_name} ===")
    
    histories, final_models = {}, {}
    is_dynamic = (weight_mode == 'dynamic')
    current_weight_str = DYNAMIC_WEIGHT_STR if is_dynamic else STATIC_WEIGHT_STR

    for goal in GOALS_TO_RUN:
        goal_cfg = GOAL_CONFIGS[goal]
        
        # Instantiate Physics Problem inside the loop so parameters reset for each goal
        inv_mode = INVERSE_PROBLEM and goal != 2
        phys_problem = ViscoelasticPhysics(
            mu_s=GUESS_MU_S if inv_mode else mu_s, 
            mu_p=GUESS_MU_P if inv_mode else mu_p, 
            lam=GUESS_LAM if inv_mode else lam, 
            pde_weights=PDE_WEIGHTS,
            inverse_mode=inv_mode,
            real_mu_s=mu_s,
            real_mu_p=mu_p,
            real_lam=lam
        ).to(device).to(initial_dtype)
        label = goal_cfg['label']
        mode_param = goal_cfg['mode']
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
        if goal == 1:
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
            
            # Metriche multi-campo
            fields_exact_for_metrics = {
                'u': U_grid, 'p': P_grid,
                'tau_xx': TAU_XX_grid, 'tau_xy': TAU_XY_grid, 'tau_yy': TAU_YY_grid
            }
            visco_metrics = compute_viscoelastic_metrics(
                model_combined, phys_problem, xy_grid_flat, fields_exact_for_metrics, Ny_dom, Nx_dom
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
                'n_points': xy_pinn_data.shape[0] if goal in [1, 2] else 0,
                'Loss_Weight': current_weight_str
            }
            update_results_csv(RESULTS_CSV_PATH, log_data)
            histories[label] = history
            final_models[label] = model_combined
            
            if inv_mode:
                history.plot_physical_parameters(
                    true_etas=mu_s,
                    true_etap=mu_p,
                    true_lam=lam,
                    save_path=os.path.join(exp_dir, 'VE_parameters_evolution.png'),
                    experiment_name=f"Viscoelastic {label}"
                )
        except Exception as e:
            print(f"  [X] Errore nel training {label}: {e}")
            import traceback
            traceback.print_exc()

    # --- COMPARISON PLOTS ---
    print(f"  > Generating Comparisons for {config_name}...")
    results_dir = os.path.join(config_dir, 'comparisons')
    os.makedirs(results_dir, exist_ok=True)
    
    model_results = []
    model_results_multi = []
    for label, model in final_models.items():
        model.eval()
        with torch.set_grad_enabled(True):
            x_input = xy_grid_flat.clone().to(next(model.parameters()).dtype).requires_grad_(True)
            u_p, _, p_p, _ = phys_problem.get_velocity(model, x_input)
            out = model(x_input)
            pred_u = u_p.detach().cpu().to(torch.float32).reshape(Ny_dom, Nx_dom)
            pred_p = p_p.detach().cpu().to(torch.float32).reshape(Ny_dom, Nx_dom)
            pred_txx = out[:, 2].detach().cpu().to(torch.float32).reshape(Ny_dom, Nx_dom)
            pred_txy = out[:, 3].detach().cpu().to(torch.float32).reshape(Ny_dom, Nx_dom)
            pred_tyy = out[:, 4].detach().cpu().to(torch.float32).reshape(Ny_dom, Nx_dom)
        model_results.append({'T_pred': pred_u, 'label': label})
        model_results_multi.append({
            'label': label,
            'fields': {'u': pred_u, 'p': pred_p, 'tau_xx': pred_txx, 'tau_xy': pred_txy, 'tau_yy': pred_tyy}
        })
    
    if model_results:
        hparams = {'arch': layers_str, 'epochs': str(epochs), 'act': act_str, 'lr_strategy': lr_strat, 'weight': current_weight_str}
        plot2D_unified_comparison(X.cpu(), Y.cpu(), U_grid.cpu(), model_results, hparams, save_path=os.path.join(results_dir, 'Comparison_Unified_ErrorMaps.png'))
    
    if model_results_multi:
        fields_exact_cpu = {
            'u': U_grid.cpu(), 'p': P_grid.cpu(),
            'tau_xx': TAU_XX_grid.cpu(), 'tau_xy': TAU_XY_grid.cpu(), 'tau_yy': TAU_YY_grid.cpu()
        }
        plot2D_viscoelastic_comparison(
            X.cpu(), Y.cpu(), fields_exact_cpu, model_results_multi, hparams,
            save_path=os.path.join(results_dir, 'Comparison_Viscoelastic_AllFields.png')
        )
    
    if len(histories) > 1:
        labels_list = list(histories.keys())
        hist_list = [histories[l] for l in labels_list]
        plot_loss_comparison(hist_list, labels_list, save_path=os.path.join(results_dir, 'Comparison_Loss_All_Goals.png'))

    # --- VRAM Cleanup: previene OOM nelle grid search lunghe ---
    del histories, final_models, model_results, model_results_multi
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("\nWeighted Grid Search configurations completed.")
