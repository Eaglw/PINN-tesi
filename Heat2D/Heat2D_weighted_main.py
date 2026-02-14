import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from tqdm import tqdm
import shutil

# Import function for GIF
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from func.graphic_func import save_gif_PIL, plot2D_comparison
from func.logging_utils import compute_metrics, update_results_csv
from func.sampling_utils import generate_internal_points, generate_grid_points, filter_and_refill, check_overlaps
from datetime import datetime

def setup_experiment_folder(parent_dir, goal_folder, description):
    """
    Creates experiment folder and plots folder.
    """
    exp_dir = os.path.join(parent_dir, goal_folder)
    plots_dir = os.path.join(exp_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    return exp_dir, plots_dir

# --- 1. DEFINIZIONE DEL PROBLEMA E SOLUZIONE ANALITICA ---
def soluzione_analitica(x, y, Lx=1.0, Ly=1.0, Nx=50):
    """
    Calcola la soluzione analitica per l'equazione di Laplace (stato stazionario calore).
    """
    T = torch.zeros_like(x)
    const_pi = torch.tensor(np.pi, device=x.device)
    for n in range(1, Nx + 1, 2):
        lambda_n = n * const_pi / Ly
        An = 4 / (n * const_pi)
        term = An * (torch.sinh(lambda_n * x) / torch.sinh(lambda_n * Lx)) * torch.sin(lambda_n * y)
        T += term
    return T

# --- 2. DEFINIZIONE DELLA RETE NEURALE ---
class FCN(nn.Module):
    """Rete Neurale a Connessioni Complete (Fully Connected Network)"""
    def __init__(self, layers, activation_fn=nn.Tanh):
        super().__init__()
        self.activation = activation_fn()
        self.fcs = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.fcs.append(nn.Linear(layers[i], layers[i+1]))
    def forward(self, x):
        for i, layer in enumerate(self.fcs):
            x = layer(x)
            if i < len(self.fcs) - 1:
                x = self.activation(x)
        return x

def get_activation_name(activation_class):
    return activation_class.__name__

def format_layers_name(layers):
    if len(layers) > 3:
        hidden = layers[1:-1]
        if all(x == hidden[0] for x in hidden):
            return f"{layers[0]}_{hidden[0]}x{len(hidden)}_{layers[-1]}"
    return "_".join(map(str, layers))

# Configurazione dispositivo e precisione
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)
print(f"Using device: {device} with default dtype: {torch.get_default_dtype()}")

# Flag per controllare la visualizzazione interattiva dei plot
show_plots_interactively = False 

# Cases to run: 2 (Data+Phys), 3 (Pure Phys)
goal = [2, 3]

# --- HYPERPARAMETERS GRID SEARCH SETUP ---
layers_options = [
    #[2, 50, 50, 50, 50, 1], # Configurazione Originale
    [2, 80, 80, 80, 80, 80, 80, 1],
    [2, 100, 100, 100, 100, 100, 100, 100, 100, 1]  
]

epochs_options = [
    #20000,
    40000
]

activation_options = [
    #nn.Tanh,
    nn.SiLU,
    nn.GELU
]

lr_strategies = [
    #'fixed',
    #'step_decay',
    'plateau'
]

weighting_options = [
    #'static',
    'dynamic'
    ]

# TARGET WEIGHTS
STATIC_WEIGHTS = {'bc': 1.0, 'physics': 20.0, 'data': 100.0}
STATIC_WEIGHT_STR = "BC=1-PHYS=20-DATA=100"
DYNAMIC_WEIGHT_STR = "Dynamic-Annealing"

base_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiments_weighted')
results_csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.csv')

# Dati fissi del problema
Lx, Ly = 1.0, 1.0
Nx_fourier = 50
Nx_dom, Ny_dom = 50, 50
x_grid = torch.linspace(0, Lx, Nx_dom, device=device)
y_grid = torch.linspace(0, Ly, Ny_dom, device=device)
X, Y = torch.meshgrid(x_grid, y_grid, indexing='xy')
T_grid = soluzione_analitica(X, Y, Lx, Ly, Nx=Nx_fourier)
xy_grid_flat = torch.stack([X.flatten(), Y.flatten()], dim=1)

# Preparazione dati Training
torch.manual_seed(123)

# --- GENERAZIONE GRIGLIE FISICA E DATI (Master Sets) ---
Nx_grid_master, Ny_grid_master = 40, 40
xy_master_grid = generate_grid_points(Nx_grid_master, Ny_grid_master, Lx, Ly, margin=1e-5, device=device)

num_master_random = 1600
xy_master_random = generate_internal_points(num_master_random, Lx, Ly, margin=1e-5, device=device)

num_b_side = 100
# Left (x=0) - Esclude angoli
x_b_l = torch.zeros(num_b_side - 2, 1, device=device)
y_b_l = torch.linspace(0, Ly, num_b_side, device=device)[1:-1].reshape(-1, 1)
# Right (x=Lx) - Esclude angoli
x_b_r = torch.ones(num_b_side - 2, 1, device=device) * Lx
y_b_r = torch.linspace(0, Ly, num_b_side, device=device)[1:-1].reshape(-1, 1)
# Bottom (y=0) - Esclude angoli
x_b_b = torch.linspace(0, Lx, num_b_side, device=device)[1:-1].reshape(-1, 1)
y_b_b = torch.zeros(num_b_side - 2, 1, device=device)
# Top (y=Ly) - Esclude angoli
x_b_t = torch.linspace(0, Lx, num_b_side, device=device)[1:-1].reshape(-1, 1)
y_b_t = torch.ones(num_b_side - 2, 1, device=device) * Ly

xy_master_boundary = torch.cat([
    torch.cat([x_b_l, y_b_l], dim=1),
    torch.cat([x_b_r, y_b_r], dim=1),
    torch.cat([x_b_b, y_b_b], dim=1),
    torch.cat([x_b_t, y_b_t], dim=1)
], dim=0)

# Rimozione duplicati (corner) dai bordi
xy_master_boundary = torch.unique(xy_master_boundary, dim=0)

T_master_random = soluzione_analitica(xy_master_random[:, 0:1], xy_master_random[:, 1:2], Lx, Ly, Nx=Nx_fourier)
T_master_boundary = soluzione_analitica(xy_master_boundary[:, 0:1], xy_master_boundary[:, 1:2], Lx, Ly, Nx=Nx_fourier)

# PINN Data+Phys Setup
num_subset = 1000
generator_fn = lambda n: generate_internal_points(n, Lx, Ly, margin=1e-5, device=device)
xy_pinn_data = filter_and_refill(xy_master_grid, generator_fn, num_subset, d_min=1e-4)
T_pinn_data = soluzione_analitica(xy_pinn_data[:, 0:1], xy_pinn_data[:, 1:2], Lx, Ly, Nx=Nx_fourier)
pinn_data_internal = (xy_pinn_data, T_pinn_data)
pinn_data_boundary = (xy_master_boundary, T_master_boundary)

# --- VERIFICA DISGIUNZIONE E OVERLAP ---
print("\n--- Point Overlap Verification ---")
check_overlaps(xy_master_grid, label="Master Grid")
check_overlaps(xy_pinn_data, label="PINN Data Set")
check_overlaps(xy_master_boundary, label="Master Boundary")
check_overlaps(xy_master_random, label="Master Random")
ok_pinn = check_overlaps(torch.cat([xy_master_grid, xy_pinn_data, xy_master_boundary], dim=0), label="PINN Full Set")

if not ok_pinn:
    print("❌ Critical Overlap detected in PINN Full Set. Terminating.")
    sys.exit(1)
print("----------------------------------\n")

validation_grid_tuple = (xy_grid_flat, T_grid, X, Y)

# --- GRID SEARCH EXECUTION ---
total_configs = len(layers_options) * len(epochs_options) * len(activation_options) * len(lr_strategies) * len(weighting_options)
print(f"Starting Weighted Grid Search over {total_configs} configurations...")

for layers_config in layers_options:
    for epochs in epochs_options:
        for act_fn in activation_options:
            for lr_strat in lr_strategies:
                for weight_mode in weighting_options:
            
                    layers_str = format_layers_name(layers_config)
                    act_str = get_activation_name(act_fn)
                    config_name = f"L{layers_str}_E{epochs}_{act_str}_{lr_strat}_{weight_mode}"
                    
                    config_dir = os.path.join(base_output_dir, config_name)
                    os.makedirs(config_dir, exist_ok=True)
                    
                    print(f"\n=== Running Configuration: {config_name} ===")
                    
                    histories = {}
                    final_models = {}

                    base_lr = 1e-3
                    if lr_strat == 'step_decay':
                        final_lr = base_lr * (0.5**4) 
                        lr_log_str = f"[{base_lr} -> {final_lr}]"
                    elif lr_strat == 'plateau':
                        lr_log_str = f"[plateau min:1e-6]"
                    else:
                        lr_log_str = str(base_lr)

                    is_dynamic = (weight_mode == 'dynamic')
                    current_weight_str = DYNAMIC_WEIGHT_STR if is_dynamic else STATIC_WEIGHT_STR

                    # --- 2. PINN Data+Phys ---
                    if 2 in goal:
                        print(f"  > 2. PINN Data+Phys ({config_name})")
                        exp_dir_2, plots_dir_2 = setup_experiment_folder(config_dir, "2_PINN_DataPhys", f"PINN Data+Phys {weight_mode}")
                        from Heat2D.src.Heat2D_PINN import train_modelPINN
                        from Heat2D.src.physics import HeatEquation2D
                        
                        heat_physics = HeatEquation2D()
                        model_2 = FCN(layers=layers_config, activation_fn=act_fn).to(device)
                        optimizer_2 = torch.optim.Adam(model_2.parameters(), lr=base_lr)
                        
                        # Use unit weights for dynamic mode, STATIC_WEIGHTS for static mode
                        w_2 = {'bc': 1.0, 'physics': 1.0, 'data': 1.0} if is_dynamic else STATIC_WEIGHTS

                        history_2 = train_modelPINN(
                            model=model_2,
                            optimizer=optimizer_2,
                            data_internal=pinn_data_internal,
                            data_boundary=pinn_data_boundary,
                            validation_grid=validation_grid_tuple,
                            physics_problem=heat_physics,
                            epochs=epochs,
                            plots_dir=plots_dir_2,
                            final_dir=exp_dir_2,
                            show_plots_interactively=show_plots_interactively,
                            log_gradients_every=500,
                            collocation_points=xy_master_grid,
                            lr_strategy=lr_strat,
                            loss_weights=w_2,
                            dynamic_weighting=is_dynamic,
                            update_weights_every=500,
                            warmup_epochs=0 
                        )
                        
                        l2_err, max_err = compute_metrics(model_2, xy_grid_flat, T_grid)
                        def get_last(hist, key): return hist.losses[key][-1] if (key in hist.losses and hist.losses[key]) else 0
                        
                        log_data = {
                            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'Architecture': str(layers_config),
                            'Activation_Func': get_activation_name(act_fn),
                            'Epochs': epochs,
                            'Run_Type': 'PINN_DataPhys',
                            'Optimizer': 'Adam + L-BFGS',
                            'Learning_Rate': lr_log_str,
                            'Loss_Total': get_last(history_2, 'total_loss'),
                            'Loss_Physics': get_last(history_2, 'pde_loss'),
                            'Loss_Boundary': get_last(history_2, 'bc_loss'), 
                            'L2_Relative_Error': l2_err,
                            'Max_Relative_Error_Peak': max_err,
                            'Seed': 123,
                            'n_points': xy_pinn_data.shape[0],
                            'Loss_Weight': current_weight_str
                        }
                        update_results_csv(results_csv_path, log_data)
                        histories['PINN Data+Phys'] = history_2
                        final_models['PINN Data+Phys'] = model_2
                        if os.path.exists(plots_dir_2): shutil.rmtree(plots_dir_2)

                    # --- 3. PINN PurePhys ---
                    if 3 in goal:
                        print(f"  > 3. PINN PurePhys ({config_name})")
                        exp_dir_3, plots_dir_3 = setup_experiment_folder(config_dir, "3_PINN_PurePhys", f"PINN PurePhys {weight_mode}")
                        from Heat2D.src.Heat2D_PINN import train_modelPINN
                        from Heat2D.src.physics import HeatEquation2D
                        
                        heat_physics = HeatEquation2D()
                        model_3 = FCN(layers=layers_config, activation_fn=act_fn).to(device)
                        optimizer_3 = torch.optim.Adam(model_3.parameters(), lr=base_lr)
                        
                        # For Pure Physics, data weight is always 0.
                        if is_dynamic:
                            w_3 = {'bc': 1.0, 'physics': 1.0, 'data': 0.0}
                        else:
                            w_3 = {'bc': STATIC_WEIGHTS['bc'], 'physics': STATIC_WEIGHTS['physics'], 'data': 0.0}

                        history_3 = train_modelPINN(
                            model=model_3,
                            optimizer=optimizer_3,
                            data_internal=pinn_data_internal,
                            data_boundary=pinn_data_boundary,
                            validation_grid=validation_grid_tuple,
                            physics_problem=heat_physics,
                            epochs=epochs,
                            plots_dir=plots_dir_3,
                            final_dir=exp_dir_3,
                            show_plots_interactively=show_plots_interactively,
                            log_gradients_every=500,
                            collocation_points=xy_master_grid,
                            lr_strategy=lr_strat,
                            loss_weights=w_3,
                            dynamic_weighting=is_dynamic,
                            update_weights_every=500,
                            warmup_epochs=0
                        )

                        l2_err, max_err = compute_metrics(model_3, xy_grid_flat, T_grid)
                        log_data = {
                            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'Architecture': str(layers_config),
                            'Activation_Func': get_activation_name(act_fn),
                            'Epochs': epochs,
                            'Run_Type': 'PINN_PurePhys',
                            'Optimizer': 'Adam + L-BFGS',
                            'Learning_Rate': lr_log_str, 
                            'Loss_Total': get_last(history_3, 'total_loss'),                    
                            'Loss_Physics': get_last(history_3, 'pde_loss'),
                            'Loss_Boundary': get_last(history_3, 'bc_loss'), 
                            'Loss_Data': get_last(history_3, 'data_loss'),
                            'L2_Relative_Error': l2_err,
                            'Max_Relative_Error_Peak': max_err,
                            'Seed': 123,
                            'n_points': 0,
                            'Loss_Weight': current_weight_str
                        }
                        update_results_csv(results_csv_path, log_data)
                        histories['PINN PurePhys'] = history_3
                        final_models['PINN PurePhys'] = model_3
                        if os.path.exists(plots_dir_3): shutil.rmtree(plots_dir_3)

                    # --- COMPARISON LOGIC ---
                    print(f"  > Generating Comparisons for {config_name}...")
                    results_dir = os.path.join(config_dir, 'comparisons')
                    os.makedirs(results_dir, exist_ok=True)
                    
                    if 'PINN Data+Phys' in final_models and 'PINN PurePhys' in final_models:
                        from func.graphic_func import plot2D_unified_comparison
                        model_results = []
                        for label in ['PINN Data+Phys', 'PINN PurePhys']:
                            model = final_models[label]
                            model.eval()
                            with torch.no_grad():
                                pred = model(xy_grid_flat).reshape(Nx_dom, Ny_dom)
                            model_results.append({'T_pred': pred, 'label': label})
                        
                        hparams = {'arch': layers_str, 'epochs': str(epochs), 'act': act_str, 'lr_strategy': lr_strat, 'weight': current_weight_str}
                        plot2D_unified_comparison(X, Y, T_grid, model_results, hparams, save_path=os.path.join(results_dir, 'Comparison_Unified_ErrorMaps.png'))

                    from func.graphic_func import plot_loss_comparison
                    if 'PINN Data+Phys' in histories and 'PINN PurePhys' in histories:
                        plot_loss_comparison([histories['PINN Data+Phys'], histories['PINN PurePhys']], ['PINN Data+Phys', 'PINN PurePhys'], save_path=os.path.join(results_dir, 'Comparison_Loss_DataPhys_vs_PurePhys.png'))
                        
print("\nWeighted Grid Search configurations completed.")
