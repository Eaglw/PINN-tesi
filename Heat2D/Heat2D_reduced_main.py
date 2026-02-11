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
from func.sampling_utils import generate_internal_points, generate_grid_points
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
Seleziona quali casi eseguire:
0. NN con dati random
1. NN con dati su griglia
"""
goal = [0, 1]

# --- HYPERPARAMETERS GRID SEARCH SETUP ---
# Opzioni per la Grid Search
layers_options = [
    [2, 50, 50, 50, 50, 1], # Solo architettura 50x4
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

# Output directory specifica per esperimenti a punti ridotti
base_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiments_reduced_points')
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

# --- GENERAZIONE GRIGLIE FISICA E DATI (Reduced Sets) ---
# 1. Grid Points (Internal): ~300 points (approx 17x17 = 289)
Nx_grid_red, Ny_grid_red = 17, 18 # 17*18 = 306 points
xy_red_grid = generate_grid_points(Nx_grid_red, Ny_grid_red, Lx, Ly, margin=1e-5, device=device)

# 2. Random Points (Internal): 300 points
num_red_random = 300
xy_red_random = generate_internal_points(num_red_random, Lx, Ly, margin=1e-5, device=device)

# 3. Boundary Points: 200 points (50 per side) - Equidistant
num_b_side = 50
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

xy_red_boundary = torch.cat([
    torch.cat([x_b_l, y_b_l], dim=1),
    torch.cat([x_b_r, y_b_r], dim=1),
    torch.cat([x_b_b, y_b_b], dim=1),
    torch.cat([x_b_t, y_b_t], dim=1)
], dim=0)

# Rimozione duplicati (corner) dai bordi
xy_red_boundary = torch.unique(xy_red_boundary, dim=0)

# Pre-calcolo Soluzione Analitica per i Reduced Sets
T_red_grid = soluzione_analitica(xy_red_grid[:, 0:1], xy_red_grid[:, 1:2], Lx, Ly, Nx=Nx_fourier)
T_red_random = soluzione_analitica(xy_red_random[:, 0:1], xy_red_random[:, 1:2], Lx, Ly, Nx=Nx_fourier)
T_red_boundary = soluzione_analitica(xy_red_boundary[:, 0:1], xy_red_boundary[:, 1:2], Lx, Ly, Nx=Nx_fourier)

# --- CONFIGURAZIONE CASI ---
# 0. NN Random: 300 Random + 200 Boundary = 500
xy_train_nn_random = torch.cat([xy_red_random, xy_red_boundary], dim=0)
T_train_nn_random = torch.cat([T_red_random, T_red_boundary], dim=0)
training_data_0 = (xy_train_nn_random, T_train_nn_random)

# 1. NN Grid: 306 Grid + 200 Boundary = 506
xy_train_nn_grid = torch.cat([xy_red_grid, xy_red_boundary], dim=0)
T_train_nn_grid = torch.cat([T_red_grid, T_red_boundary], dim=0)
training_data_1 = (xy_train_nn_grid, T_train_nn_grid)

n_points_random = xy_train_nn_random.shape[0]
n_points_grid = xy_train_nn_grid.shape[0]

print(f"Punti generati per esperimenti ridotti:")
print(f" - NN Random: {n_points_random} punti totali")
print(f" - NN Grid:   {n_points_grid} punti totali")

validation_grid_tuple = (xy_grid_flat, T_grid, X, Y)

# --- GRID SEARCH EXECUTION ---
total_configs = len(layers_options) * len(epochs_options) * len(activation_options) * len(lr_strategies)
print(f"Starting Reduced Points Grid Search over {total_configs} configurations...")

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
                
                # Salva immagine soluzione analitica
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
                    final_lr = base_lr * (0.5**4) 
                    lr_log_str = f"[{base_lr} -> {final_lr}]"
                else:
                    lr_log_str = str(base_lr)

                # --- 0. NN Random ---
                if 0 in goal:
                    if os.path.exists(os.path.join(config_dir, "0_NN_Random")):
                        print(f"  > 0. NN Random ({config_name}) - SKIPPING (Already exists)")
                        # Load model for comparison if needed? For now just skip training.
                        # To ensure comparison logic works, we might need to load the model. 
                        # But for now let's just assume if it exists we don't re-run.
                        #Comparison logic requires 'final_models' to be populated.
                        # If we skip, we won't have the model in memory.
                        # We can simply skip comparison generation for skipped runs or try to load.
                        # Given the user just wants the CSV logs and folders, skipping is fine.
                    else:
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
                            'Run_Type': 'NN_Random_Red',
                            'Optimizer': 'Adam', 
                            'Learning_Rate': lr_log_str, 
                            'Loss_Total': history_0.losses['total_loss'][-1] if history_0.losses['total_loss'] else 0,
                            'Loss_Physics': 0,
                            'Loss_Boundary': 0,
                            'Loss_Data': history_0.losses['total_loss'][-1] if history_0.losses['total_loss'] else 0, 
                            'L2_Relative_Error': l2_err,
                            'Max_Relative_Error_Peak': max_err,
                            'Seed': 123,
                            'n_points': n_points_random,
                            'Loss_Weight': 'not_weighted'
                        }
                        update_results_csv(results_csv_path, log_data)

                        histories['NN Random'] = history_0
                        final_models['NN Random'] = model_0
                        if os.path.exists(plots_dir_0):
                            shutil.rmtree(plots_dir_0)

                # --- 1. NN Grid ---
                if 1 in goal:
                    if os.path.exists(os.path.join(config_dir, "1_NN_Grid")):
                        print(f"  > 1. NN Grid ({config_name}) - SKIPPING (Already exists)")
                    else:
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
                            'Run_Type': 'NN_Grid_Red',
                            'Optimizer': 'Adam',
                            'Learning_Rate': lr_log_str,
                            'Loss_Total': history_1.losses['total_loss'][-1] if history_1.losses['total_loss'] else 0,
                            'Loss_Physics': 0,
                            'Loss_Boundary': 0,
                            'Loss_Data': history_1.losses['total_loss'][-1] if history_1.losses['total_loss'] else 0,
                            'L2_Relative_Error': l2_err,
                            'Max_Relative_Error_Peak': max_err,
                            'Seed': 123,
                            'n_points': n_points_grid,
                            'Loss_Weight': 'not_weighted'
                        }
                        update_results_csv(results_csv_path, log_data)
                        
                        histories['NN Grid'] = history_1
                        final_models['NN Grid'] = model_1
                        if os.path.exists(plots_dir_1):
                            shutil.rmtree(plots_dir_1)

                # --- COMPARISON LOGIC (Per Config) ---
                print(f"  > Generating Comparisons for {config_name}...")
                results_dir = os.path.join(config_dir, 'comparisons')
                os.makedirs(results_dir, exist_ok=True)
                
                # Unified Error Map Comparison
                if 0 in goal and 1 in goal:
                    if 'NN Random' in final_models and 'NN Grid' in final_models:
                        from func.graphic_func import plot2D_unified_comparison
                        from func.logging_utils import extract_hyperparams_from_path
                        
                        model_results = []
                        for label in ['NN Random', 'NN Grid']:
                            if label in final_models:
                                model = final_models[label]
                                model.eval()
                                with torch.no_grad():
                                    pred = model(xy_grid_flat).reshape(Nx_dom, Ny_dom)
                                model_results.append({'T_pred': pred, 'label': label})
                        
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
                        plot_loss_comparison(
                            [histories['NN Random'], histories['NN Grid']],
                            ['NN Random', 'NN Grid'],
                            save_path=os.path.join(results_dir, 'Comparison_Loss_Random_vs_Grid.png')
                        )
                    else:
                        print(f"  > Skipping comparisons for {config_name} (Models not loaded)")
                        
print("\nAll Reduced Points Grid Search configurations completed.")
