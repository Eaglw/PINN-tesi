import torch
import sys
import os

# Set paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Viscoelastic.src.load_comsol import prepare_training_data

device = 'cpu'
comsol_params = {
    'mu_s': 0.005,
    'mu_p': 0.005,
    'lam': 0.1,
    'eps': 0.0,
    'alpha': 0.0,
    'rho': 1.0,
}

data_bundle = prepare_training_data(
    'COMSOL/Oldroyd.csv',
    comsol_params,
    num_data_subset=5000,
    initial_dtype=torch.float32,
    device=device
)

var_weights = data_bundle['var_weights']
print("\n--- VARIANCE WEIGHTS ---")
for k, v in var_weights.items():
    print(f"{k}: {v:.6f} (1/var: {1.0/v:.6f})")
