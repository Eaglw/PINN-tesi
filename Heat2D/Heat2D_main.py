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
from datetime import datetime

def setup_experiment_folder(parent_dir, goal_folder, description):
    """
    Creates experiment folder and plots folder.
    Arguments:
        parent_dir: The parent directory (e.g., experiments/Config_X_Y_Z)
        goal_folder: The specific experiment folder (e.g., 0_NN_Random)
    """
    exp_dir = os.path.join(parent_dir, goal_folder)
    plots_dir = os.path.join(exp_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    return exp_dir, plots_dir

# --- 1. DEFINIZIONE DEL PROBLEMA E SOLUZIONE ANALITICA ---
def soluzione_analitica(x, y, Lx=1.0, Ly=1.0, Nx=50):
    """
    Calcola la soluzione analitica per l'equazione di Laplace (stato stazionario calore).
    T=0 su y=0, y=Ly, x=0; T=1 su x=Lx.
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
            if i < len(self.fcs) - 1: # Apply activation to all but the last layer
                x = self.activation(x)
        return x
    def loss_fn(self, pred, target):
        return nn.MSELoss()(pred, target)

def get_activation_name(activation_class):
    return activation_class.__name__

def format_layers_name(layers):
    # E.g., [2, 50, 50, 1] -> "2_50x2_1"
    if len(layers) > 3:
        # Check if hidden layers are equal
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

"""
Seleziona quali casi eseguire inserendo nell'array goal il corrispettivo numero:
0. NN con dati random
1. NN con dati su griglia
2. PINN con dati e fisica
3. PINN solo fisica
"""
goal = [0, 1, 2, 3]

# --- HYPERPARAMETERS GRID SEARCH SETUP ---
# Opzioni per la Grid Search
layers_options = [
    [2, 50, 50, 50, 50, 1], # Configurazione Originale
    [2, 80, 80, 80, 80, 80, 80, 1]      
]

epochs_options = [
    20000,
    40000
]

activation_options = [
    nn.Tanh,
    nn.SiLU,
    nn.GELU
]

lr_strategies = [
    'fixed',
    'step_decay'
]

base_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiments')
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

# Preparazione dati Training (generati una volta per consistenza)
torch.manual_seed(123)

# --- GENERAZIONE GRIGLIE FISICA E DATI (Master Sets) ---
# 1. Grid Points (Internal): 40x40 = 1600 points
Nx_grid_master, Ny_grid_master = 40, 40
x_grid_int = torch.linspace(0, Lx, Nx_grid_master + 2, device=device)[1:-1]
y_grid_int = torch.linspace(0, Ly, Ny_grid_master + 2, device=device)[1:-1]
X_grid_int, Y_grid_int = torch.meshgrid(x_grid_int, y_grid_int, indexing='xy')
xy_master_grid = torch.stack([X_grid_int.flatten(), Y_grid_int.flatten()], dim=1)

# 2. Random Points (Internal): 1600 points
num_master_random = 1600
xy_master_random = torch.rand((num_master_random, 2), device=device)
xy_master_random[:, 0] *= Lx
xy_master_random[:, 1] *= Ly

# 3. Boundary Points: 400 points (100 per side) - Equidistant
num_b_side = 100
# Left (x=0)
x_b_l = torch.zeros(num_b_side, 1, device=device)
y_b_l = torch.linspace(0, Ly, num_b_side, device=device).reshape(-1, 1)
# Right (x=Lx)
x_b_r = torch.ones(num_b_side, 1, device=device) * Lx
y_b_r = torch.linspace(0, Ly, num_b_side, device=device).reshape(-1, 1)
# Bottom (y=0)
x_b_b = torch.linspace(0, Lx, num_b_side, device=device).reshape(-1, 1)
y_b_b = torch.zeros(num_b_side, 1, device=device)
# Top (y=Ly)
x_b_t = torch.linspace(0, Lx, num_b_side, device=device).reshape(-1, 1)
y_b_t = torch.ones(num_b_side, 1, device=device) * Ly

xy_master_boundary = torch.cat([
    torch.cat([x_b_l, y_b_l], dim=1),
    torch.cat([x_b_r, y_b_r], dim=1),
    torch.cat([x_b_b, y_b_b], dim=1),
    torch.cat([x_b_t, y_b_t], dim=1)
], dim=0)

# Pre-calcolo Soluzione Analitica per i Master Sets
T_master_grid = soluzione_analitica(xy_master_grid[:, 0:1], xy_master_grid[:, 1:2], Lx, Ly, Nx=Nx_fourier)
T_master_random = soluzione_analitica(xy_master_random[:, 0:1], xy_master_random[:, 1:2], Lx, Ly, Nx=Nx_fourier)
T_master_boundary = soluzione_analitica(xy_master_boundary[:, 0:1], xy_master_boundary[:, 1:2], Lx, Ly, Nx=Nx_fourier)

# --- CONFIGURAZIONE CASI ---
# 0. NN Random: 1600 Random + 400 Boundary
xy_train_nn_random = torch.cat([xy_master_random, xy_master_boundary], dim=0)
T_train_nn_random = torch.cat([T_master_random, T_master_boundary], dim=0)
training_data_0 = (xy_train_nn_random, T_train_nn_random)

# 1. NN Grid: 1600 Grid + 400 Boundary
xy_train_nn_grid = torch.cat([xy_master_grid, xy_master_boundary], dim=0)
T_train_nn_grid = torch.cat([T_master_grid, T_master_boundary], dim=0)
training_data_1 = (xy_train_nn_grid, T_train_nn_grid)

# 2. PINN Data+Phys: Phys=Grid(1600), BC=Boundary(400), Data=RandomSubset(1000)
num_subset = 1000
xy_pinn_data = xy_master_random[:num_subset]
T_pinn_data = T_master_random[:num_subset]
# Per PINN Data+Phys, data_internal riceve i 1000 punti random
pinn_data_internal = (xy_pinn_data, T_pinn_data)
pinn_data_boundary = (xy_master_boundary, T_master_boundary)

# 3. PINN Pure Phys: Phys=Grid(1600), BC=Boundary(400)
# Usa gli stessi pinn_data_boundary, ma data_internal viene ignorato se peso=0

print(f"Punti generati per esperimenti:")
print(f" - NN Random: {xy_train_nn_random.shape[0]} punti totali")
print(f" - NN Grid:   {xy_train_nn_grid.shape[0]} punti totali")
print(f" - PINN Data: {xy_pinn_data.shape[0]} punti supervisionati")
print(f" - PINN Phys: {xy_master_grid.shape[0]} punti collocazione")
print(f" - PINN BC:   {xy_master_boundary.shape[0]} punti boundary")

validation_grid_tuple = (xy_grid_flat, T_grid, X, Y)

# --- GRID SEARCH EXECUTION ---
total_configs = len(layers_options) * len(epochs_options) * len(activation_options) * len(lr_strategies)
print(f"Starting Grid Search over {total_configs} configurations...")

for layers_config in layers_options:
    for epochs in epochs_options:
        for act_fn in activation_options:
            for lr_strat in lr_strategies:
            
                # Setup Config Folder Name
                layers_str = format_layers_name(layers_config)
                act_str = get_activation_name(act_fn)
                config_name = f"L{layers_str}_E{epochs}_{act_str}_{lr_strat}"
                
                config_dir = os.path.join(base_output_dir, config_name)
                os.makedirs(config_dir, exist_ok=True)
                
                print(f"\n=== Running Configuration: {config_name} ===")
                
                # Salva immagine soluzione analitica nella root della config per riferimento
                plt.figure(figsize=(8,6))
                cp = plt.contourf(X.cpu().numpy(), Y.cpu().numpy(), T_grid.cpu().numpy(), 50, cmap='inferno')
                plt.colorbar(cp)
                plt.title(f'Analytic Solution ({config_name})')
                plt.savefig(os.path.join(config_dir, 'analytic_sol.png'))
                plt.close()

                histories = {}
                final_models = {}

                # Determine Loggable LR string
                base_lr = 1e-3
                if lr_strat == 'step_decay':
                    # 4 steps of 0.5 decay: 1 -> 0.5 -> 0.25 -> 0.125 -> 0.0625
                    final_lr = base_lr * (0.5**4) 
                    lr_log_str = f"[{base_lr} -> {final_lr}]"
                else:
                    lr_log_str = str(base_lr)

                # --- 0. NN Random ---
                if 0 in goal:
                    print(f"  > 0. NN Random ({config_name})")
                    exp_dir_0, plots_dir_0 = setup_experiment_folder(
                        config_dir,
                        "0_NN_Random", 
                        f"NN Random. Config: {config_name}"
                    )
                    from Heat2D.src.Heat2D_NN import train_modelNN
                    
                    model_0 = FCN(layers=layers_config, activation_fn=act_fn).to(device)
                    optimizer_0 = torch.optim.Adam(model_0.parameters(), lr=base_lr)
                    
                    history_0 = train_modelNN(
                        model=model_0,
                        optimizer=optimizer_0,
                        training_data=training_data_0,
                        validation_grid=validation_grid_tuple,
                        epochs=epochs,
                        plots_dir=plots_dir_0,
                        final_dir=exp_dir_0,
                        show_plots_interactively=show_plots_interactively,
                        lr_strategy=lr_strat
                    )
                    
                    # --- LOGGING 0_NN_Random ---
                    l2_err, max_err = compute_metrics(model_0, xy_grid_flat, T_grid)
                    log_data = {
                        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'Architecture': str(layers_config),
                        'Activation_Func': get_activation_name(act_fn),
                        'Epochs': epochs,
                        'Run_Type': 'NN_Random',
                        'Optimizer': 'Adam', 
                        'Learning_Rate': lr_log_str, 
                        'Loss_Total': history_0.losses['total_loss'][-1] if history_0.losses['total_loss'] else 0,
                        'Loss_Physics': 0,
                        'Loss_Boundary': 0,
                        'Loss_Data': history_0.losses['total_loss'][-1] if history_0.losses['total_loss'] else 0, 
                        'L2_Relative_Error': l2_err,
                        'Max_Relative_Error_Peak': max_err,
                        'Seed': 123,
                        'Loss_Weight': 'not_weighted'
                    }
                    update_results_csv(results_csv_path, log_data)

                    histories['NN Random'] = history_0
                    final_models['NN Random'] = model_0
                    # Cleanup Plots
                    if os.path.exists(plots_dir_0):
                        shutil.rmtree(plots_dir_0)

                # --- 1. NN Grid ---
                if 1 in goal:
                    print(f"  > 1. NN Grid ({config_name})")
                    exp_dir_1, plots_dir_1 = setup_experiment_folder(
                        config_dir,
                        "1_NN_Grid", 
                        f"NN Grid. Config: {config_name}"
                    )
                    from Heat2D.src.Heat2D_NN_griglia import train_modelNN_griglia
                    
                    model_1 = FCN(layers=layers_config, activation_fn=act_fn).to(device)
                    optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=base_lr)
                    
                    history_1 = train_modelNN_griglia(
                        model=model_1,
                        optimizer=optimizer_1,
                        training_data=training_data_1,
                        validation_grid=validation_grid_tuple,
                        epochs=epochs,
                        plots_dir=plots_dir_1,
                        final_dir=exp_dir_1,
                        show_plots_interactively=show_plots_interactively,
                        lr_strategy=lr_strat
                    )
                    
                    # --- LOGGING 1_NN_Grid ---
                    l2_err, max_err = compute_metrics(model_1, xy_grid_flat, T_grid)
                    log_data = {
                        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'Architecture': str(layers_config),
                        'Activation_Func': get_activation_name(act_fn),
                        'Epochs': epochs,
                        'Run_Type': 'NN_Grid',
                        'Optimizer': 'Adam',
                        'Learning_Rate': lr_log_str,
                        'Loss_Total': history_1.losses['total_loss'][-1] if history_1.losses['total_loss'] else 0,
                        'Loss_Physics': 0,
                        'Loss_Boundary': 0,
                        'Loss_Data': history_1.losses['total_loss'][-1] if history_1.losses['total_loss'] else 0,
                        'L2_Relative_Error': l2_err,
                        'Max_Relative_Error_Peak': max_err,
                        'Seed': 123,
                        'Loss_Weight': 'not_weighted'
                    }
                    update_results_csv(results_csv_path, log_data)
                    
                    histories['NN Grid'] = history_1
                    final_models['NN Grid'] = model_1
                    # Cleanup Plots
                    if os.path.exists(plots_dir_1):
                        shutil.rmtree(plots_dir_1)

                # --- 2. PINN Data+Phys ---
                if 2 in goal:
                    print(f"  > 2. PINN Data+Phys ({config_name})")
                    exp_dir_2, plots_dir_2 = setup_experiment_folder(
                        config_dir,
                        "2_PINN_DataPhys", 
                        f"PINN Data+Phys. Config: {config_name}"
                    )
                    from Heat2D.src.Heat2D_PINN import train_modelPINN
                    from Heat2D.src.physics import HeatEquation2D
                    
                    heat_physics = HeatEquation2D()
                    model_2 = FCN(layers=layers_config, activation_fn=act_fn).to(device)
                    optimizer_2 = torch.optim.Adam(model_2.parameters(), lr=base_lr)
                    
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
                        collocation_points=xy_master_grid,
                        lr_strategy=lr_strat
                    )
                    
                    # --- LOGGING 2_PINN_DataPhys ---
                    l2_err, max_err = compute_metrics(model_2, xy_grid_flat, T_grid)
                    # Helper to safely get last loss
                    def get_last(hist, key): return hist.losses[key][-1] if (key in hist.losses and hist.losses[key]) else 0
                    
                    log_data = {
                        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'Architecture': str(layers_config),
                        'Activation_Func': get_activation_name(act_fn),
                        'Epochs': epochs,
                        'Run_Type': 'PINN_DataPhys',
                        'Optimizer': 'Adam',
                        'Learning_Rate': lr_log_str,
                        'Loss_Total': get_last(history_2, 'total_loss'),
                        'Loss_Physics': get_last(history_2, 'pde_loss'),
                        'Loss_Boundary': get_last(history_2, 'bc_loss'), 
                        'Loss_Data': get_last(history_2, 'data_loss'),
                        'L2_Relative_Error': l2_err,
                        'Max_Relative_Error_Peak': max_err,
                        'Seed': 123,
                        'Loss_Weight': 'not_weighted'
                    }
                    update_results_csv(results_csv_path, log_data)

                    histories['PINN Data+Phys'] = history_2
                    final_models['PINN Data+Phys'] = model_2
                    # Cleanup Plots
                    if os.path.exists(plots_dir_2):
                        shutil.rmtree(plots_dir_2)

                # --- 3. PINN PurePhys ---
                if 3 in goal:
                    print(f"  > 3. PINN PurePhys ({config_name})")
                    exp_dir_3, plots_dir_3 = setup_experiment_folder(
                        config_dir,
                        "3_PINN_PurePhys", 
                        f"PINN PurePhys. Config: {config_name}"
                    )
                    from Heat2D.src.Heat2D_PINN import train_modelPINN
                    from Heat2D.src.physics import HeatEquation2D
                    
                    pp_config = {'loss_weights': {'data': 0.0, 'bc': 1.0, 'physics': 1.0}, 'warmup_epochs': 0}
                    heat_physics = HeatEquation2D()
                    model_3 = FCN(layers=layers_config, activation_fn=act_fn).to(device)
                    optimizer_3 = torch.optim.Adam(model_3.parameters(), lr=base_lr)
                    
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
                        loss_weights=pp_config['loss_weights'],
                        warmup_epochs=pp_config['warmup_epochs'],
                        collocation_points=xy_master_grid,
                        lr_strategy=lr_strat
                    )

                    # --- LOGGING 3_PINN_PurePhys ---
                    l2_err, max_err = compute_metrics(model_3, xy_grid_flat, T_grid)
                    def get_last(hist, key): return hist.losses[key][-1] if (key in hist.losses and hist.losses[key]) else 0
                    
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
                        'Loss_Weight': 'not_weighted'
                    }
                    update_results_csv(results_csv_path, log_data)
                    
                    histories['PINN PurePhys'] = history_3
                    final_models['PINN PurePhys'] = model_3
                    # Cleanup Plots
                    if os.path.exists(plots_dir_3):
                        shutil.rmtree(plots_dir_3)

                # --- COMPARISON LOGIC (Per Config) ---
                print(f"  > Generating Comparisons for {config_name}...")
                results_dir = os.path.join(config_dir, 'comparisons')
                os.makedirs(results_dir, exist_ok=True)
                
                # Unified 2x2 Error Map Comparison
                if all(g in goal for g in [0, 1, 2, 3]):
                    from func.graphic_func import plot2D_unified_comparison
                    from func.logging_utils import extract_hyperparams_from_path
                    
                    model_results = []
                    for label in ['NN Random', 'NN Grid', 'PINN Data+Phys', 'PINN PurePhys']:
                        if label in final_models:
                            model = final_models[label]
                            model.eval()
                            with torch.no_grad():
                                pred = model(xy_grid_flat).reshape(Nx_dom, Ny_dom)
                            model_results.append({'T_pred': pred, 'label': label})
                    
                    # Extract hyperparams from path for the title
                    arch, epochs_str, act = extract_hyperparams_from_path(config_dir)
                    hparams = {'arch': arch, 'epochs': epochs_str, 'act': act, 'lr_strategy': lr_strat}
                    
                    if model_results:
                        plot2D_unified_comparison(
                            X, Y, T_grid, 
                            model_results, 
                            hparams, 
                            save_path=os.path.join(results_dir, 'Comparison_Unified_ErrorMaps.png')
                        )

                # Pairwise Loss Comparisons
                from func.graphic_func import plot_loss_comparison
                if 0 in goal and 1 in goal:
                    plot_loss_comparison(
                        [histories['NN Random'], histories['NN Grid']],
                        ['NN Random', 'NN Grid'],
                        save_path=os.path.join(results_dir, 'Comparison_Loss_Random_vs_Grid.png')
                    )

                if 2 in goal and 3 in goal:
                    plot_loss_comparison(
                        [histories['PINN Data+Phys'], histories['PINN PurePhys']],
                        ['PINN Data+Phys', 'PINN PurePhys'],
                        save_path=os.path.join(results_dir, 'Comparison_Loss_DataPhys_vs_PurePhys.png')
                    )
                
                if 1 in goal and 2 in goal:
                    plot_loss_comparison(
                        [histories['NN Grid'], histories['PINN Data+Phys']],
                        ['NN Grid', 'PINN Data+Phys'],
                        save_path=os.path.join(results_dir, 'Comparison_Loss_Grid_vs_PINN.png')
                    )
                        
print("\nAll Grid Search configurations completed.")