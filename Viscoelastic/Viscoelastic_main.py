import torch
import torch.nn as nn
import os
import sys
import shutil
import itertools
from datetime import datetime

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Import funzioni esterne
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from func.logging_utils import compute_metrics, compute_viscoelastic_metrics, update_results_csv
from func.sampling_utils import generate_internal_points, generate_grid_points
from func.graphic_func import plot2D_unified_comparison, plot_loss_comparison, plot2D_viscoelastic_comparison

# Import locali Viscoelastic
from Viscoelastic.src.Viscoelastic_PINN import train_ViscoelasticPINN, FCN, ViscoelasticCombinedModel, VelocityInferenceWrapper, get_activation_name, format_layers_name
from Viscoelastic.src.Viscoelastic_physics import ViscoelasticPhysics, generate_boundaries

torch.backends.cuda.matmul.allow_tf32 = True  
torch.backends.cudnn.benchmark = True           
torch.backends.cudnn.deterministic = False      

def setup_experiment_folder(parent_dir, goal_folder, description):
    exp_dir = os.path.join(parent_dir, goal_folder)
    plots_dir = os.path.join(exp_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    return exp_dir, plots_dir

# --- SETUP DISPOSITIVO ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)
print(f"Using device: {device} with default dtype: {torch.get_default_dtype()}")

show_plots_interactively = False 

# Cases to run: 0 (Pure Phys), 1 (Phys+Data), 2 (Solo Data)
goals_to_run = [0, 1, 2]

# --- CONFIGURATION FLAGS ---
STAGED_TRAINING = True 

# --- HYPERPARAMETERS GRID SEARCH SETUP ---
layers_options = [[2, 120, 100, 80, 60, 40, 20, 1]]
epochs_options = [10000]
activation_options = [nn.SiLU]
lr_strategies = ['plateau']
weighting_options = ['dynamic']

STATIC_WEIGHTS = {'bc': 1.0, 'physics': 20.0, 'data': 100.0}
STATIC_WEIGHT_STR = "BC=1-PHYS=20-DATA=100"
DYNAMIC_WEIGHT_STR = "Dynamic-Annealing"

base_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiments_weighted')
results_csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.csv')

# --- CARICAMENTO DATASET ---
dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset', 'oldroydb_clean.pt')

if not os.path.exists(dataset_path):
    print(f"❌ Dataset non trovato in: {dataset_path}")
    sys.exit(1)

dataset = torch.load(dataset_path, map_location=device, weights_only=False)
for key in ['coords', 'u', 'v', 'p', 'psi', 'tau_xx', 'tau_xy', 'tau_yy', 'u_exact', 'p_exact', 'psi_exact', 'tau_xx_exact', 'tau_xy_exact', 'tau_yy_exact']:
    if key in dataset:
        dataset[key] = dataset[key].to(torch.float32)
params = dataset['params']

Lx, Ly, mu_s, mu_p, lam, u_max = params['L'], params['H'], params.get('mu_s', 0.005), params.get('mu_p', 0.005), params.get('lam', 1.0), params['u_max']
print(f"Dataset caricato: L={Lx}, H={Ly}, mu_s={mu_s}, mu_p={mu_p}, lam={lam}, u_max={u_max}")

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

margin=2e-2
Nx_grid_master, Ny_grid_master = 40, 40
xy_master_grid = generate_grid_points(Nx_grid_master, Ny_grid_master, Lx, Ly, margin=margin, device=device)

# Controlla che tutti i punti rispettino il dominio
assert xy_master_grid[:, 0].min() >= 0 and xy_master_grid[:, 0].max() <= Lx
assert xy_master_grid[:, 1].min() >= 0 and xy_master_grid[:, 1].max() <= Ly

# --- BOUNDARY CONDITIONS (u, v, p, tau) ---
xy_master_boundary, uvp_master_boundary = generate_boundaries(Lx, Ly, u_max, p_exact, stress_exact_grids, Nx_dom, Ny_dom, device)

num_subset = 1000
torch.manual_seed(42)
idx = torch.randperm(xy_grid_flat.shape[0])[:num_subset]
xy_pinn_data = xy_grid_flat[idx]
psip_pinn_data = torch.cat([psi_exact[idx], p_exact[idx], tau_xx_exact[idx], tau_xy_exact[idx], tau_yy_exact[idx]], dim=1) 

pinn_data_internal = (xy_pinn_data, psip_pinn_data)
pinn_data_boundary = (xy_master_boundary, uvp_master_boundary)

# --- GRID SEARCH EXECUTION ---
configs = list(itertools.product(layers_options, epochs_options, activation_options, lr_strategies, weighting_options))
print(f"Starting Weighted Grid Search over {len(configs)} configurations...")

def get_last(hist, key): 
    return hist.losses[key][-1] if (key in hist.losses and hist.losses[key]) else 0

for layers_config, epochs, act_fn, lr_strat, weight_mode in configs:
    torch.set_default_dtype(torch.float32)
    layers_str = format_layers_name(layers_config)
    act_str = get_activation_name(act_fn)
    config_name = f"L{layers_str}_E{epochs}_{act_str}_{lr_strat}_{weight_mode}"
    
    config_dir = os.path.join(base_output_dir, config_name)
    os.makedirs(config_dir, exist_ok=True)
    
    print(f"\n=== Running Configuration: {config_name} ===")
    
    histories, final_models = {}, {}
    base_lr = 1e-3
    if lr_strat == 'step_decay':
        lr_log_str = f"[{base_lr} -> {base_lr * (0.5**4)}]"
    elif lr_strat == 'plateau':
        lr_log_str = "[plateau min:1e-6]"
    else:
        lr_log_str = str(base_lr)

    is_dynamic = (weight_mode == 'dynamic')
    current_weight_str = DYNAMIC_WEIGHT_STR if is_dynamic else STATIC_WEIGHT_STR

    phys_problem = ViscoelasticPhysics(mu_s=mu_s, mu_p=mu_p, lam=lam)

    for goal in goals_to_run:
        # Mapping dei Goal: 0=PurePhys, 1=Phys+Data, 2=SoloData
        if goal == 0:
            label = "PurePhys"
            current_w = {'bc': 1.0, 'physics': 1.0, 'data': 0.0}
        elif goal == 1:
            label = "Phys+Data"
            current_w = {'bc': 1.0, 'physics': 1.0, 'data': 1.0}
        else: # goal == 2
            label = "SoloData"
            current_w = {'bc': 0.0, 'physics': 0.0, 'data': 1.0}

        prefix = f"{goal}_{label}"
        print(f"  > {label} ({config_name})")
        
        exp_dir, plots_dir = setup_experiment_folder(config_dir, prefix, f"{label} {weight_mode}")
        
        torch.manual_seed(123)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(123)
            
        torch.set_default_dtype(torch.float32)
        pinn_data_internal_fresh = (xy_pinn_data.float(), psip_pinn_data.float())
        pinn_data_boundary_fresh = (xy_master_boundary.float(), uvp_master_boundary.float())
        
        # Forziamo l'ultimo layer
        layers_psi = layers_config[:-1] + [1]
        layers_p = layers_config[:-1] + [1]
        layers_tau = layers_config[:-1] + [3] # tau ha 3 componenti (xx, xy, yy)
        
        model_psi = FCN(layers=layers_psi, activation_fn=act_fn).to(device).to(torch.float32)
        model_p = FCN(layers=layers_p, activation_fn=act_fn).to(device).to(torch.float32)
        model_tau = FCN(layers=layers_tau, activation_fn=act_fn).to(device).to(torch.float32)
        model_combined = ViscoelasticCombinedModel(model_psi, model_p, model_tau)

        # Passiamo una lista unica di parametri all'ottimizzatore
        optimizer_params = list(model_combined.parameters())
        optimizer = torch.optim.Adam(optimizer_params, lr=base_lr)
        
        # Se siamo nel Goal 2 (SoloData), il dynamic weighting non ha senso (una sola componente)
        # Lo disabilitiamo localmente per questa run
        run_is_dynamic = is_dynamic if goal != 2 else False

        # Se non siamo in modalità dinamica, applichiamo i pesi statici (tranne che per SoloData)
        effective_w = dict(current_w)
        if not run_is_dynamic and goal != 2:
            effective_w['bc'] *= STATIC_WEIGHTS['bc']
            effective_w['physics'] *= STATIC_WEIGHTS['physics']
            effective_w['data'] *= STATIC_WEIGHTS['data']

        warmup = 0 if goal == 2 else epochs // 5

        try:
            use_staged = STAGED_TRAINING and goal != 2
            history = train_ViscoelasticPINN(
                model=model_combined, optimizer=optimizer,
                data_internal=pinn_data_internal_fresh, data_boundary=pinn_data_boundary_fresh,
                validation_grid=validation_grid_u, physics_problem=phys_problem,
                epochs=epochs, plots_dir=plots_dir, final_dir=exp_dir,
                show_plots_interactively=show_plots_interactively,
                log_gradients_every=500, collocation_points=xy_master_grid,
                lr_strategy=lr_strat, loss_weights=effective_w, dynamic_weighting=run_is_dynamic,
                update_weights_every=100, warmup_epochs=warmup,
                experiment_name=f"Viscoelastic {label}", val_label="u (Velocity)",
                stress_exact_grids=stress_exact_grids,
                staged_training=use_staged, base_lr=base_lr
            )
            
            # Metriche multi-campo per il caso viscoelastico
            fields_exact_for_metrics = {
                'u': U_grid, 'p': P_grid,
                'tau_xx': TAU_XX_grid, 'tau_xy': TAU_XY_grid, 'tau_yy': TAU_YY_grid
            }
            visco_metrics = compute_viscoelastic_metrics(
                model_combined, phys_problem, xy_grid_flat, fields_exact_for_metrics, Ny_dom, Nx_dom
            )
            
            # Metriche aggregate: media L2 su campi non-banali, max globale
            # Escludo campi con L2=0 (soluzione esatta nulla, es. tau_yy in Poiseuille)
            l2_values = [v[0] for v in visco_metrics.values() if v[0] > 1e-10]
            max_values = [v[1] for v in visco_metrics.values()]
            l2_avg = sum(l2_values) / len(l2_values) if l2_values else 0.0
            max_global = max(max_values) if max_values else 0.0
            
            log_data = {
                'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Architecture': str(layers_config),
                'Activation_Func': act_str, 'Epochs': epochs, 'Run_Type': label,
                'Optimizer': 'Adam', 'Learning_Rate': lr_log_str, 
                'Loss_Total': get_last(history, 'total_loss'), 'Loss_Physics': get_last(history, 'pde_loss'),
                'Loss_Boundary': get_last(history, 'bc_loss'), 'Loss_Data': get_last(history, 'data_loss'),
                'L2_Relative_Error': l2_avg, 'Max_Relative_Error_Peak': max_global,
                'L2_u': visco_metrics['u'][0], 'Max_u': visco_metrics['u'][1],
                'L2_p': visco_metrics['p'][0], 'Max_p': visco_metrics['p'][1],
                'L2_tau_xx': visco_metrics['tau_xx'][0], 'Max_tau_xx': visco_metrics['tau_xx'][1],
                'L2_tau_xy': visco_metrics['tau_xy'][0], 'Max_tau_xy': visco_metrics['tau_xy'][1],
                'L2_tau_yy': visco_metrics['tau_yy'][0], 'Max_tau_yy': visco_metrics['tau_yy'][1],
                'Seed': 123, 'n_points': xy_pinn_data.shape[0] if goal in [1, 2] else 0,
                'Loss_Weight': current_weight_str
            }
            update_results_csv(results_csv_path, log_data)
            histories[label] = history
            final_models[label] = model_combined
        except Exception as e:
            print(f"  [X] Errore nel training {label}: {e}")
            import traceback
            traceback.print_exc()

    print(f"  > Generating Comparisons for {config_name}...")
    results_dir = os.path.join(config_dir, 'comparisons')
    os.makedirs(results_dir, exist_ok=True)
    
    model_results = []
    model_results_multi = []  # Per comparison multi-campo
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
        # X, Y, U_grid possono essere su CUDA; i plot li richiedono su CPU
        plot2D_unified_comparison(X.cpu(), Y.cpu(), U_grid.cpu(), model_results, hparams, save_path=os.path.join(results_dir, 'Comparison_Unified_ErrorMaps.png'))
    
    # Comparison multi-campo per tutti i campi fisici
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

print("\nWeighted Grid Search configurations completed.")
