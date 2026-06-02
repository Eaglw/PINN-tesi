import sys
import os
from pathlib import Path

# Aggiunge la directory radice al path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from Viscoelastic.src.load_comsol import prepare_training_data
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_path = str(BASE_DIR / 'COMSOL' / 'Oldroyd_res.csv')

COMSOL_PARAMS = {
    'mu_s': 0.005,
    'mu_p': 0.005,
    'lam': 0.1,
    'eps': 0.0,
    'alpha': 0.0,
    'rho': 1.0,
}

print(f"Tentativo di caricamento da: {dataset_path}")
try:
    data_bundle = prepare_training_data(
        dataset_path, 
        COMSOL_PARAMS, 
        num_data_subset=100, 
        initial_dtype=torch.float32, 
        device=device
    )
    print("\n[SUCCESSO] Il dataset è stato caricato correttamente!")
    
    # Audit delle normali nei gruppi di contorno
    bg = data_bundle['boundary_groups']
    print("\n--- Audit delle normali locali dei gruppi ---")
    for name, group in bg.items():
        coords = group['xy']
        norms = group['norm']
        print(f"Gruppo '{name}': {coords.shape[0]} punti")
        
        # Mostra le coordinate e le normali di alcuni punti significativi (es. i primi 3)
        n_show = min(3, coords.shape[0])
        for i in range(n_show):
            pt = coords[i].cpu().tolist()
            norm = norms[i].cpu().tolist()
            print(f"  Punto {i}: xy = [{pt[0]:.4f}, {pt[1]:.4f}] -> normal = [{norm[0]:.4f}, {norm[1]:.4f}]")
            
except Exception as e:
    print(f"\n[ERRORE] Si è verificato un errore durante il caricamento: {e}")
    import traceback
    traceback.print_exc()
