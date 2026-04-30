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
from func.logging_utils import compute_metrics, update_results_csv
from func.sampling_utils import generate_internal_points, generate_grid_points
from func.graphic_func import plot2D_unified_comparison, plot_loss_comparison

# Import locali Newtonian
from Newtonian.src.Newtonian_PINN import train_NewtonianPINN, FCN, NewtonianCombinedModel, VelocityInferenceWrapper, get_activation_name, format_layers_name
from Newtonian.src.Newtonian_physics import NewtonianPhysics, generate_boundaries

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

# Cases to run: 2 (PINN Data+Phys), 3 (Pure Phys)
# For this refactored version, we loop over these dynamically
goals_to_run = [2, 3]

# --- HYPERPARAMETERS GRID SEARCH SETUP ---
layers_options = [[2,10,10,1]]#,[2, 120, 100, 80, 60, 40, 20, 2]] OPTIM
epochs_options = [100]#,8000] OPTIM
activation_options = [nn.SiLU]
lr_strategies = ['plateau']
weighting_options = ['dynamic']

STATIC_WEIGHTS = {'bc': 1.0, 'physics': 20.0, 'data': 100.0}
STATIC_WEIGHT_STR = "BC=1-PHYS=20-DATA=100"
DYNAMIC_WEIGHT_STR = "Dynamic-Annealing"

base_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiments_weighted')
results_csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.csv')

# --- CARICAMENTO DATASET ---
dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset', 'poiseuille_clean.pt')

if not os.path.exists(dataset_path):
    print(f"❌ Dataset non trovato in: {dataset_path}")
    sys.exit(1)

dataset = torch.load(dataset_path, map_location=device)
for key in ['coords', 'u', 'v', 'p', 'psi', 'u_exact', 'p_exact', 'psi_exact']:
    if key in dataset:
        dataset[key] = dataset[key].to(torch.float32)
params = dataset['params']

Lx, Ly, mu, u_max = params['L'], params['H'], params['mu'], params['u_max']
print(f"Dataset caricato: L={Lx}, H={Ly}, mu={mu}, u_max={u_max}")

xy_grid_flat = dataset['coords']
u_exact = dataset['u_exact']
p_exact = dataset['p_exact']
psi_exact = dataset['psi_exact']
v_exact = torch.zeros_like(u_exact)

x_sorted = torch.unique(xy_grid_flat[:, 0], sorted=True)
y_sorted = torch.unique(xy_grid_flat[:, 1], sorted=True)
Nx_dom, Ny_dom = len(x_sorted), len(y_sorted)

X = xy_grid_flat[:, 0].reshape(Ny_dom, Nx_dom)
Y = xy_grid_flat[:, 1].reshape(Ny_dom, Nx_dom)
U_grid = u_exact.reshape(Ny_dom, Nx_dom)
P_grid = p_exact.reshape(Ny_dom, Nx_dom)
validation_grid_tuple = (xy_grid_flat, U_grid, X, Y)

margin=2e-2
Nx_grid_master, Ny_grid_master = 40, 40
xy_master_grid = generate_grid_points(Nx_grid_master, Ny_grid_master, Lx, Ly, margin=margin, device=device)

# --- BOUNDARY CONDITIONS (u, v, p) ---
xy_master_boundary, uvp_master_boundary = generate_boundaries(Lx, Ly, u_max, p_exact, P_grid, Nx_dom, Ny_dom, device)

num_subset = 1000
idx = torch.randperm(xy_grid_flat.shape[0])[:num_subset]
xy_pinn_data = xy_grid_flat[idx]
psip_pinn_data = torch.cat([psi_exact[idx], p_exact[idx]], dim=1) 

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

    for goal in goals_to_run:
        label = "PINN Data+Phys" if goal == 2 else "PINN PurePhys"
        prefix = f"{goal}_{label.replace(' ', '')}"
        print(f"  > {label} ({config_name})")
        
        exp_dir, plots_dir = setup_experiment_folder(config_dir, prefix, f"{label} {weight_mode}")
        phys_problem = NewtonianPhysics(mu=mu)
        
        # Forziamo l'ultimo layer a 1 per le reti separate
        layers_psi = layers_config[:-1] + [1]
        layers_p = layers_config[:-1] + [1]
        
        model_psi = FCN(layers=layers_psi, activation_fn=act_fn).to(device).to(torch.float32)
        model_p = FCN(layers=layers_p, activation_fn=act_fn).to(device).to(torch.float32)
        model_combined = NewtonianCombinedModel(model_psi, model_p)

        # Passiamo una lista unica di parametri all'ottimizzatore
        params = list(model_combined.parameters())
        optimizer = torch.optim.Adam(params, lr=base_lr)
        
        data_w = 1.0 if goal == 2 else 0.0
        w = {'bc': 1.0, 'physics': 1.0, 'data': data_w} if is_dynamic else {'bc': STATIC_WEIGHTS['bc'], 'physics': STATIC_WEIGHTS['physics'], 'data': data_w}

        history = train_NewtonianPINN(
            model=model_combined, optimizer=optimizer,
            data_internal=pinn_data_internal, data_boundary=pinn_data_boundary,
            validation_grid=validation_grid_tuple, physics_problem=phys_problem,
            epochs=epochs, plots_dir=plots_dir, final_dir=exp_dir,
            show_plots_interactively=show_plots_interactively,
            log_gradients_every=500, collocation_points=xy_master_grid,
            lr_strategy=lr_strat, loss_weights=w, dynamic_weighting=is_dynamic,
            update_weights_every=100, warmup_epochs=0,
            experiment_name=f"Newtonian {label}", val_label="u (Velocity)"
        )
        
        metrics_wrapper = VelocityInferenceWrapper(model_combined, phys_problem)
        l2_err, max_err = compute_metrics(metrics_wrapper, xy_grid_flat, U_grid)
        
        log_data = {
            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Architecture': str(layers_config),
            'Activation_Func': act_str, 'Epochs': epochs, 'Run_Type': label.replace(' ', '_'),
            'Optimizer': 'Adam + L-BFGS', 'Learning_Rate': lr_log_str, 
            'Loss_Total': get_last(history, 'total_loss'), 'Loss_Physics': get_last(history, 'pde_loss'),
            'Loss_Boundary': get_last(history, 'bc_loss'), 'Loss_Data': get_last(history, 'data_loss'),
            'L2_Relative_Error': l2_err, 'Max_Relative_Error_Peak': max_err,
            'Seed': 123, 'n_points': xy_pinn_data.shape[0] if goal == 2 else 0,
            'Loss_Weight': current_weight_str
        }
        update_results_csv(results_csv_path, log_data)
        histories[label] = history
        final_models[label] = model_combined
        if os.path.exists(plots_dir): shutil.rmtree(plots_dir)

    print(f"  > Generating Comparisons for {config_name}...")
    results_dir = os.path.join(config_dir, 'comparisons')
    os.makedirs(results_dir, exist_ok=True)
    
    model_results = []
    for label in ['PINN Data+Phys', 'PINN PurePhys']:
        if label in final_models:
            model = final_models[label].eval()
            with torch.set_grad_enabled(True):
                x_input = xy_grid_flat.clone().to(next(model.parameters()).dtype).requires_grad_(True)
                u_p, _, _ = phys_problem.get_velocity(model, x_input)
                pred = u_p.detach().cpu().to(torch.float32).reshape(Ny_dom, Nx_dom)
            model_results.append({'T_pred': pred, 'label': label})
    
    if model_results:
        hparams = {'arch': layers_str, 'epochs': str(epochs), 'act': act_str, 'lr_strategy': lr_strat, 'weight': current_weight_str}
        plot2D_unified_comparison(X, Y, U_grid, model_results, hparams, save_path=os.path.join(results_dir, 'Comparison_Unified_ErrorMaps.png'))
    
    if 'PINN Data+Phys' in histories and 'PINN PurePhys' in histories:
        plot_loss_comparison([histories['PINN Data+Phys'], histories['PINN PurePhys']], ['PINN Data+Phys', 'PINN PurePhys'], save_path=os.path.join(results_dir, 'Comparison_Loss_DataPhys_vs_PurePhys.png'))

print("\nWeighted Grid Search configurations completed.")
