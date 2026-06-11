import torch
import torch.nn as nn
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import gc
import sys
import itertools
from datetime import datetime
from pathlib import Path
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# Risoluzione dei percorsi del progetto
BASE_DIR = Path(r"c:\Users\eaglw\Documents\PINN tesi\FourRollMill")
sys.path.append(str(BASE_DIR.parent))

from func.logging_utils import update_results_csv

# Import locali FourRollMill
from FourRollMill.src.models import FCN, ViscoelasticCombinedModel, ScaledViscoelasticCombinedModel, get_activation_name, format_layers_name, initialize_last_layer_zero

from FourRollMill.src.config import TrainingConfig
from FourRollMill.src.trainer import train_ViscoelasticPINN, compute_viscoelastic_metrics, compute_pinn_loss
from FourRollMill.src.Viscoelastic_physics import ViscoelasticPhysics

LAYERS_OPTIONS = [[2, 128, 128, 128, 128, 128, 128, 128, 128, 1]]  # VENet 8x128
EPOCHS_OPTIONS = [500]
MAX_LBFGS_ITERS = 200
ACTIVATION_OPTIONS = [nn.Tanh]
LR_STRATEGY_OPTIONS = ['cosine']
WEIGHTING_OPTIONS = ['static']


GOALS_TO_RUN = [1]

GOAL_CONFIGS = {
    1: {'label': 'Phys+Data', 'weights': {'bc': 1.0, 'physics': 1.0, 'data': 1.0}, 'mode': 'semi_inverse'},
}

PRECISION_MODE = 'staged'
SEED = 123
DATASET_OPTIONS = ['4_roll_mill.csv']

COMSOL_PARAMS = {
    'mu_s': 0.1,
    'mu_p': 0.9,
    'lam': 1,
    'eps': 0.0,
    'alpha': 0.0,
    'rho': 1000,
}

initial_dtype = torch.float32
torch.set_default_dtype(initial_dtype)

BASE_LR = 1e-3
ADAM_EPS = 1e-7
STAGED_TRAINING = False



MINIBATCH_INTERNAL = 2048
MINIBATCH_BOUNDARY = 256

STATIC_WEIGHTS = {'bc': 10.0, 'physics': 10.0, 'data': 1.0}
STATIC_WEIGHT_STR = "BC=10-PHYS=10-DATA=1"
DYNAMIC_WEIGHT_STR = "Dynamic-Annealing"

GROUP_WEIGHTS = {'Walls': 1.0, 'Roll1': 1.0, 'Roll2': 1.0, 'Roll3': 1.0, 'Roll4': 1.0, 'PressurePoint': 10.0}

# Impostiamo momentum a 10.0 e constitutive a 1.0 come nella main config
PDE_WEIGHTS = {'momentum': 10.0, 'constitutive': 1.0}
VARIANCE_EPS = 1e-4

INVERSE_PROBLEM = False
GUESS_MULTIPLIER = 0.8
GUESS_MIN_EPS = 0.0
GUESS_MIN_ALPHA = 0.0

LOG_GRADIENTS_EVERY = 100000
PLOT_EVERY = 100000  # Disable intermediate plotting

BASE_OUTPUT_DIR = BASE_DIR / 'experiments_weighted_test'
RESULTS_CSV_PATH = BASE_DIR / 'results_test.csv'

def setup_experiment_folder(parent_dir: Path, goal_folder: str):
    exp_dir = parent_dir / goal_folder
    plots_dir = exp_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir, plots_dir

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} with Precision Mode: {PRECISION_MODE}")

    for dataset_filename in DATASET_OPTIONS:
        DATASET_PATH = BASE_DIR.parent / 'COMSOL'/ '4roll' / dataset_filename
        dataset_name_prefix = DATASET_PATH.stem
        
        from FourRollMill.src.load_comsol import prepare_training_data
        data_bundle = prepare_training_data(
            str(DATASET_PATH), COMSOL_PARAMS,
            initial_dtype, device, variance_eps=VARIANCE_EPS,
            mask_multiplier=5.0
        )
        
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

        for layers_config, epochs, act_fn, lr_strat, weight_mode in configs:
            layers_str = format_layers_name(layers_config)
            act_str = get_activation_name(act_fn)
            config_name = f"{dataset_name_prefix}_test_L{layers_str}_E{epochs}_{act_str}_{lr_strat}_{weight_mode}"
            config_dir = BASE_OUTPUT_DIR / config_name
            config_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n=== Running Test Configuration: {config_name} ===")
            
            is_dynamic = (weight_mode == 'dynamic')

            for goal in GOALS_TO_RUN:
                goal_cfg = GOAL_CONFIGS[goal]
                inv_mode = INVERSE_PROBLEM and goal != 2
                label = goal_cfg['label']
                mode_param = 'comsol_full' if goal == 2 else 'semi_inverse'
                current_w = dict(goal_cfg['weights'])
                prefix = f"{goal}_{label}"

                exp_dir, plots_dir = setup_experiment_folder(config_dir, prefix)
                torch.manual_seed(SEED)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(SEED)

                # Guess dinamici basati sui valori veri del dataset
                inv_mode = INVERSE_PROBLEM and goal != 2
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

                # Xavier initialization
                def init_weights_xavier(m):
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_normal_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)

                model_psi.apply(init_weights_xavier)
                model_p.apply(init_weights_xavier)
                model_tau.apply(init_weights_xavier)

                # Zero last layer for stability
                initialize_last_layer_zero(model_p)
                initialize_last_layer_zero(model_tau)

                # Data-driven output scaling
                p_data = dataset['p']
                tau_xx_data, tau_xy_data, tau_yy_data = dataset['tau_xx'], dataset['tau_xy'], dataset['tau_yy']
                p_scale = max(abs(p_data.min().item()), abs(p_data.max().item()), 1.0)
                tau_scale = max(
                    abs(tau_xx_data.min().item()), abs(tau_xx_data.max().item()),
                    abs(tau_xy_data.min().item()), abs(tau_xy_data.max().item()),
                    abs(tau_yy_data.min().item()), abs(tau_yy_data.max().item()),
                    1.0
                )
                print(f"  [Output Scaling] p_scale={p_scale:.2f}, tau_scale={tau_scale:.2f}")
                model_combined = ScaledViscoelasticCombinedModel(model_psi, model_p, model_tau, p_scale=p_scale, tau_scale=tau_scale)

                run_is_dynamic = is_dynamic if goal != 2 else False
                effective_w = dict(current_w)
                if not run_is_dynamic and goal != 2:
                    effective_w['bc'] *= STATIC_WEIGHTS['bc']
                    effective_w['physics'] *= STATIC_WEIGHTS['physics']
                    effective_w['data'] *= STATIC_WEIGHTS['data']

                pinn_data_internal = (xy_pinn_data, uv_pinn_data)

                var_weights = VAR_WEIGHTS

                # apply_boundary_conditions crea automaticamente PressurePoint se presente in bc_rules
                xy_master_boundary, dir_master_boundary, neu_master_boundary, norm_master_boundary = phys_problem.apply_boundary_conditions(data_bundle['boundary_groups'])
                pinn_data_boundary = (xy_master_boundary, dir_master_boundary, neu_master_boundary, norm_master_boundary)

                train_config = TrainingConfig(
                    epochs=epochs,
                    base_lr=BASE_LR,
                    adam_eps=ADAM_EPS,
                    lr_strategy=lr_strat,
                    staged_training=(STAGED_TRAINING and goal not in [0, 2]),
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
                    experiment_name=f"Viscoelastic Test {label}",
                    val_label="u (Velocity)",
                    physics_warmup_epochs=100,
                    group_weights=GROUP_WEIGHTS,
                )


                # Pass stress_exact_grids=None to train_ViscoelasticPINN to skip viscoelastic plot generation
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
                    stress_exact_grids=None, # Skip high-dimensional plots to avoid crash
                )

                # 1. Calcolo metriche standard e shiftate
                model_combined.eval()
                with torch.set_grad_enabled(True):
                    xi = data_bundle['xy_grid_flat'].clone().to(device).requires_grad_(True)
                    up, vp, pp, tp = phys_problem.get_velocity(model_combined, xi)
                    
                    up = up.detach().cpu().view(-1)
                    vp = vp.detach().cpu().view(-1)
                    pp = pp.detach().cpu().view(-1)
                    txx = tp[:, 0].detach().cpu().view(-1)
                    txy = tp[:, 1].detach().cpu().view(-1)
                    tyy = tp[:, 2].detach().cpu().view(-1)

                fields_exact = {
                    'u': dataset['u'].cpu().view(-1),
                    'v': dataset['v'].cpu().view(-1),
                    'p': dataset['p'].cpu().view(-1),
                    'tau_xx': dataset['tau_xx'].cpu().view(-1),
                    'tau_xy': dataset['tau_xy'].cpu().view(-1),
                    'tau_yy': dataset['tau_yy'].cpu().view(-1),
                }

                print("\n=============================================")
                print("FINAL EVALUATION METRICS (without pressure BC):")
                print("=============================================")
                
                # Standard L2 relative errors
                for k, pred_field in [('u', up), ('v', vp), ('p', pp), ('tau_xx', txx), ('tau_xy', txy), ('tau_yy', tyy)]:
                    exact_field = fields_exact[k]
                    l2 = (torch.norm(pred_field - exact_field, 2) / torch.norm(exact_field, 2)).item()
                    print(f"  Standard L2 Relative Error for {k:6s}: {l2:.6f}")

                # Shifted L2 relative error for pressure (p_pred - mean(p_pred) vs p_exact - mean(p_exact))
                p_pred_shifted = pp - pp.mean()
                p_exact_shifted = fields_exact['p'] - fields_exact['p'].mean()
                l2_p_shifted = (torch.norm(p_pred_shifted - p_exact_shifted, 2) / torch.norm(p_exact_shifted, 2)).item()
                print(f"  Shifted L2 Relative Error for pressure: {l2_p_shifted:.6f}")

                # 2. Calcolo residui PDE di dettaglio in chunks
                print("\nDetailed PDE residuals (computed in chunks):")
                c_size = 2000
                res_momentum_list = []
                res_const_xx_list = []
                res_const_xy_list = []
                res_const_yy_list = []
                
                with torch.set_grad_enabled(True):
                    for i in range(0, xi.shape[0], c_size):
                        x_chunk = xi[i:i+c_size].clone().requires_grad_(True)
                        f_u, f_v, f_txx, f_tyy, f_txy = phys_problem.compute_residuals(model_combined, x_chunk)
                        
                        chunk_weight = x_chunk.shape[0] / xi.shape[0]
                        res_momentum_list.append((f_u**2 + f_v**2).mean().item() * chunk_weight)
                        res_const_xx_list.append((f_txx**2).mean().item() * chunk_weight)
                        res_const_xy_list.append((f_txy**2).mean().item() * chunk_weight)
                        res_const_yy_list.append((f_tyy**2).mean().item() * chunk_weight)
                        
                res_momentum = sum(res_momentum_list)
                res_const_xx = sum(res_const_xx_list)
                res_const_xy = sum(res_const_xy_list)
                res_const_yy = sum(res_const_yy_list)
                
                print(f"  Momentum Residual:       {res_momentum:.6e}")
                print(f"  Constitutive xx Residual: {res_const_xx:.6e}")
                print(f"  Constitutive xy Residual: {res_const_xy:.6e}")
                print(f"  Constitutive yy Residual: {res_const_yy:.6e}")
                print("=============================================\n")

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
