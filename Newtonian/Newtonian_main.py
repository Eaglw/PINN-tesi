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

# Import locali Newtonian
from Newtonian.src.Newtonian_PINN import train_modelPINN
from Newtonian.src.Newtonian_physics import NewtonianPhysics

torch.backends.cuda.matmul.allow_tf32 = True  # Abilitato per velocizzare FP32 su Ampere+
torch.backends.cudnn.benchmark = True           # auto-tuning kernel per size fissa
torch.backends.cudnn.deterministic = False      # più veloce se non serve riproducibilità

def setup_experiment_folder(parent_dir, goal_folder, description):
    """
    Creates experiment folder and plots folder.
    """
    exp_dir = os.path.join(parent_dir, goal_folder)
    plots_dir = os.path.join(exp_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    return exp_dir, plots_dir
    
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
        for layer in self.fcs[:-1]:   # tutti tranne l'ultimo
            x = self.activation(layer(x))
        return self.fcs[-1](x) 
    def loss_fn(self, pred, target):
        return nn.MSELoss()(pred, target)

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
torch.set_default_dtype(torch.float32)
print(f"Using device: {device} with default dtype: {torch.get_default_dtype()}")

# Wrapper per adattare il modello [psi, p] alle funzioni che si aspettano [u]
class NewtonianModelWrapper(nn.Module):
    def __init__(self, model, phys_problem):
        super().__init__()
        self.model = model
        self.phys_problem = phys_problem
    def forward(self, x):
        # Assicura che i gradienti siano attivi per calcolare u = psi_y
        with torch.set_grad_enabled(True):
            if not x.requires_grad: x.requires_grad_(True)
            u, _, _ = self.phys_problem.get_velocity(self.model, x)
        return u.detach() # Restituiamo solo u
    def eval(self):
        self.model.eval()
        return self

# Flag per controllare la visualizzazione interattiva dei plot
show_plots_interactively = False 

# Cases to run: 0 (NN Random), 1 (NN Grid), 2 (PINN Data+Phys), 3 (Pure Phys)
goal = [3]

# --- HYPERPARAMETERS GRID SEARCH SETUP ---
layers_options = [
    [2, 50, 50, 50, 50, 2], # Input (x,y) -> Output (psi, p)
]
    #[2, 80, 80, 80, 80, 80, 80, 1],
    #[2, 100, 100, 100, 100, 100, 100, 100, 100, 1]

epochs_options = [
    200
]

activation_options = [
    nn.Tanh,
    #nn.SiLU,
    #nn.GELU
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

# --- 4. CARICAMENTO DATASET E PARAMETRI ---
# Cerchiamo il dataset in diverse posizioni possibili
possible_paths = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset', 'poiseuille_clean.pt'),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Newtonian', 'dataset', 'poiseuille_clean.pt')
]
dataset_path = None
for p in possible_paths:
    if os.path.exists(p):
        dataset_path = p
        break

if dataset_path is None:
    print(f"❌ Dataset non trovato. Generalo prima con generate_dataset.py")
    sys.exit(1)

dataset = torch.load(dataset_path, map_location=device)
# Forza FP32 per la fase Adam iniziale
for key in ['coords', 'u', 'v', 'p', 'psi', 'u_exact', 'p_exact', 'psi_exact']:
    if key in dataset:
        dataset[key] = dataset[key].to(torch.float32)
params = dataset['params']

# Parametri fisici e di dominio estratti dal dataset
Lx = params['L']
Ly = params['H'] # Usiamo Ly per l'altezza H del canale
mu = params['mu']
u_max = params['u_max']

print(f"Dataset caricato: L={Lx}, H={Ly}, mu={mu}, u_max={u_max}")

# Griglia di validazione dal dataset
xy_grid_flat = dataset['coords']
u_exact = dataset['u_exact']
p_exact = dataset['p_exact']
psi_exact = dataset['psi_exact']
v_exact = torch.zeros_like(u_exact) # Per Poiseuille 2D stazionario v=0

# Ricostruiamo X, Y (2D) dalla griglia piatta per i plot
x_sorted = torch.unique(xy_grid_flat[:, 0], sorted=True)
y_sorted = torch.unique(xy_grid_flat[:, 1], sorted=True)
Nx_dom, Ny_dom = len(x_sorted), len(y_sorted)

X = xy_grid_flat[:, 0].reshape(Ny_dom, Nx_dom)
Y = xy_grid_flat[:, 1].reshape(Ny_dom, Nx_dom)
U_grid = u_exact.reshape(Ny_dom, Nx_dom)
P_grid = p_exact.reshape(Ny_dom, Nx_dom)
V_grid = v_exact.reshape(Ny_dom, Nx_dom)

# Preparazione dati Training
torch.manual_seed(123)
margin=2e-2
# --- GENERAZIONE GRIGLIE FISICA E DATI (Master Sets) ---
Nx_grid_master, Ny_grid_master = 40, 40
xy_master_grid = generate_grid_points(Nx_grid_master, Ny_grid_master, Lx, Ly, margin=margin, device=device)

num_master_random = 1600
xy_master_random = generate_internal_points(num_master_random, Lx, Ly, margin=margin, device=device)

# --- 5. DEFINIZIONE BOUNDARY CONDITIONS (u, v, p) ---
# Usiamo il numero di punti coerente con la griglia del dataset
num_b_y = Ny_dom 
pts_bc = torch.linspace(0, Ly, num_b_y, device=device).reshape(-1, 1) # Da 0 a H

# Profilo Parabolico: u(y) = 4 * u_max * y * (H - y) / H^2
u_parabolic = 4 * u_max * (pts_bc * (Ly - pts_bc)) / (Ly**2)
v_zero = torch.zeros_like(pts_bc)

# 1. Left (x=0) - Inlet: u=parabolico, v=0, p=p_in
bc_left = torch.cat([torch.zeros(num_b_y, 1, device=device), pts_bc], dim=1)
# Estraiamo p_in dal dataset (primo valore di p_exact all'inlet)
p_in = p_exact.flatten()[0] # Assumiamo p costante all'inlet
bc_left_val = torch.cat([u_parabolic, v_zero, torch.ones_like(pts_bc) * p_in], dim=1)

# 2. Right (x=Lx) - Outlet: u=parabolico, v=0, p=p_out
bc_right = torch.cat([torch.ones(num_b_y, 1, device=device) * Lx, pts_bc], dim=1)
p_out = p_exact.flatten()[-1] # Assumiamo p costante all'outlet
bc_right_val = torch.cat([u_parabolic, v_zero, torch.ones_like(pts_bc) * p_out], dim=1)

# 3. Walls (Top/Bottom) - No-slip: u=0, v=0
num_b_x = Nx_dom
pts_x = torch.linspace(0, Lx, num_b_x, device=device).reshape(-1, 1)
# Bottom (y=0)
bc_bottom = torch.cat([pts_x, torch.zeros(num_b_x, 1, device=device)], dim=1)
# Estraiamo la pressione p dal dataset per il bordo inferiore
p_bottom = P_grid[0, :] 
bc_bottom_val = torch.cat([torch.zeros_like(pts_x), torch.zeros_like(pts_x), p_bottom.reshape(-1, 1)], dim=1)

# Top (y=Ly)
bc_top = torch.cat([pts_x, torch.ones(num_b_x, 1, device=device) * Ly], dim=1)
p_top = P_grid[-1, :]
bc_top_val = torch.cat([torch.zeros_like(pts_x), torch.zeros_like(pts_x), p_top.reshape(-1, 1)], dim=1)

xy_master_boundary = torch.cat([bc_left, bc_right, bc_bottom, bc_top], dim=0)
uvp_master_boundary = torch.cat([bc_left_val, bc_right_val, bc_bottom_val, bc_top_val], dim=0)

# Nota: xy_pinn_data and psip_pinn_data devono essere definiti o caricati.
num_subset = 1000
idx = torch.randperm(xy_grid_flat.shape[0])[:num_subset]
xy_pinn_data = xy_grid_flat[idx]
psip_pinn_data = torch.cat([psi_exact[idx], p_exact[idx]], dim=1) # Target [psi, p]

pinn_data_internal = (xy_pinn_data, psip_pinn_data)
pinn_data_boundary = (xy_master_boundary, uvp_master_boundary)

# --- VERIFICA DISGIUNZIONE E OVERLAP ---
print("\n--- Point Overlap Verification (Disabled for Windows stability) ---")
# check_overlaps(xy_pinn_data, label="PINN Data Set")
# check_overlaps(xy_master_boundary, label="Master Boundary")
# check_overlaps(xy_master_random, label="Master Random")
# ok_pinn = check_overlaps(torch.cat([xy_master_grid, xy_pinn_data, xy_master_boundary], dim=0), label="PINN Full Set")
ok_pinn = True # Bypass check

if not ok_pinn:
    print("❌ Critical Overlap detected in PINN Full Set. Terminating.")
    sys.exit(1)
print("----------------------------------\n")

validation_grid_tuple = (xy_grid_flat, U_grid, X, Y)

# --- GRID SEARCH EXECUTION ---
torch.set_default_dtype(torch.float32) # Ensure we start in FP32
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
                    def get_last(hist, key): return hist.losses[key][-1] if (key in hist.losses and hist.losses[key]) else 0

                    # --- 2. PINN Data+Phys ---
                    if 2 in goal:
                        print(f"  > 2. PINN Data+Phys ({config_name})")
                        exp_dir_2, plots_dir_2 = setup_experiment_folder(config_dir, "2_PINN_DataPhys", f"PINN Data+Phys {weight_mode}")
                        phys_problem = NewtonianPhysics(mu=mu)
                        model_2 = FCN(layers=layers_config, activation_fn=act_fn).to(device).to(torch.float32)
                        #model_2 = torch.compile(model_2)
                        optimizer_2 = torch.optim.Adam(model_2.parameters(), lr=base_lr)
                        
                        # Use unit weights for dynamic mode, STATIC_WEIGHTS for static mode
                        w_2 = {'bc': 1.0, 'physics': 1.0, 'data': 1.0} if is_dynamic else STATIC_WEIGHTS

                        history_2 = train_modelPINN(
                            model=model_2,
                            optimizer=optimizer_2,
                            data_internal=pinn_data_internal,
                            data_boundary=pinn_data_boundary,
                            validation_grid=validation_grid_tuple,
                            physics_problem=phys_problem,
                            epochs=epochs,
                            plots_dir=plots_dir_2,
                            final_dir=exp_dir_2,
                            show_plots_interactively=show_plots_interactively,
                            log_gradients_every=500,
                            collocation_points=xy_master_grid,
                            lr_strategy=lr_strat,
                            loss_weights=w_2,
                            dynamic_weighting=is_dynamic,
                            update_weights_every=100,
                            warmup_epochs=0,
                            experiment_name="Newtonian PINN Data+Phys",
                            val_label="u (Velocity)"
                        )
                        
                        metrics_wrapper = NewtonianModelWrapper(model_2, phys_problem)
                        l2_err, max_err = compute_metrics(metrics_wrapper, xy_grid_flat, U_grid)
                        
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
                            'Loss_Data': get_last(history_2, 'data_loss'),
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
                        phys_problem = NewtonianPhysics(mu=mu)
                        model_3 = FCN(layers=layers_config, activation_fn=act_fn).to(device).to(torch.float32)
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
                            physics_problem=phys_problem,
                            epochs=epochs,
                            plots_dir=plots_dir_3,
                            final_dir=exp_dir_3,
                            show_plots_interactively=show_plots_interactively,
                            log_gradients_every=500,
                            collocation_points=xy_master_grid,
                            lr_strategy=lr_strat,
                            loss_weights=w_3,
                            dynamic_weighting=is_dynamic,
                            update_weights_every=100,
                            warmup_epochs=0,
                            experiment_name="Newtonian PINN PurePhys",
                            val_label="u (Velocity)"
                        )

                        metrics_wrapper = NewtonianModelWrapper(model_3, phys_problem)
                        l2_err, max_err = compute_metrics(metrics_wrapper, xy_grid_flat, U_grid)
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
                    
                    from func.graphic_func import plot2D_unified_comparison
                    model_results = []
                    for label in ['NN Random', 'NN Grid', 'PINN Data+Phys', 'PINN PurePhys']:
                        if label in final_models:
                            model = final_models[label]
                            model.eval()
                            with torch.set_grad_enabled(True):
                                # Ensure input has the same dtype as model weights (could be float64 after L-BFGS)
                                dtype = next(model.parameters()).dtype
                                x_input = xy_grid_flat.to(dtype)
                                if not x_input.requires_grad: x_input.requires_grad_(True)
                                
                                if hasattr(phys_problem, 'get_velocity'):
                                    u_p, _, _ = phys_problem.get_velocity(model, x_input)
                                    pred = u_p.detach().cpu().to(torch.float32).reshape(Ny_dom, Nx_dom)
                                else:
                                    pred = model(x_input)[:, 0].detach().cpu().to(torch.float32).reshape(Ny_dom, Nx_dom)
                            model_results.append({'T_pred': pred, 'label': label})
                    
                    if model_results:
                        hparams = {'arch': layers_str, 'epochs': str(epochs), 'act': act_str, 'lr_strategy': lr_strat, 'weight': current_weight_str}
                        plot2D_unified_comparison(X, Y, U_grid, model_results, hparams, save_path=os.path.join(results_dir, 'Comparison_Unified_ErrorMaps.png'))

                    from func.graphic_func import plot_loss_comparison
                    if 'PINN Data+Phys' in histories and 'PINN PurePhys' in histories:
                        plot_loss_comparison([histories['PINN Data+Phys'], histories['PINN PurePhys']], ['PINN Data+Phys', 'PINN PurePhys'], save_path=os.path.join(results_dir, 'Comparison_Loss_DataPhys_vs_PurePhys.png'))
                    
                    if 'NN Grid' in histories and 'PINN Data+Phys' in histories:
                        plot_loss_comparison([histories['NN Grid'], histories['PINN Data+Phys']], ['NN Grid', 'PINN Data+Phys'], save_path=os.path.join(results_dir, 'Comparison_Loss_Grid_vs_PINN.png'))
                        
print("\nWeighted Grid Search configurations completed.")
