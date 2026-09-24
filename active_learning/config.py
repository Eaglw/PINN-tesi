"""
Configurazione dei parametri fisici, range di esplorazione,
soglie di convergenza e percorsi per il modulo di Active Learning.
"""

from pathlib import Path
import numpy as np

# Percorsi base
ROOT_DIR = Path(__file__).resolve().parent.parent
INVERSE_RUNS_CSV = ROOT_DIR / "final_roll" / "output_4rollmill" / "inverse_runs.csv"
PLOTS_DIR = Path(__file__).resolve().parent / "plots"

# Risoluzioni di mesh standard del progetto e rispettivo numero indicativo di punti
MESH_NODES = {
    "5k": 5086,
    "12k": 12760,
    "29k": 29401,
    "52k": 52648,
    "88k": 88000,
    "125k": 125000
}

# Range fisici di esplorazione sensati per il problema 4-roll mill
PARAM_BOUNDS = {
    "lambda": (0.005, 2.00),      # Tempo di rilassamento / Weissenberg proxy
    "eta_p": (0.02, 0.98),        # Viscosità polimerica (con eta_s = 1.0 - eta_p)
    "alpha": (0.00, 0.50),        # Parametro mobilità di Giesekus (0 per Oldroyd-B/PTT)
    "eps": (0.00, 0.50),          # Parametro reticolare PTT (0 per Oldroyd-B/Giesekus)
    "log10_n_points": (np.log10(5000), np.log10(130000))  # Scala logaritmica dei punti
}

# Nomi delle feature in input al Gaussian Process
FEATURE_NAMES = ["lambda", "eta_p", "eta_s", "alpha", "eps", "log10_n_points"]

# Metrica di convergenza: errore percentuale relativo massimo dei parametri fisici
# Se l'errore max è sotto il 10%, la PINN è considerata convergente per identificazione inversa.
CONVERGENCE_THRESHOLD_PCT = 10.0
CONVERGENCE_THRESHOLD_LOG = float(np.log10(CONVERGENCE_THRESHOLD_PCT))  # log10(10) = 1.0

# Parametro di esplorazione per Straddle acquisition function: a(x) = beta * sigma(x) - |mu(x) - gamma|
# Valore 1.96 corrisponde all'intervallo di confidenza al 95%.
BETA_EXPLORATION = 1.96

# Modelli reologici ammessi per generazione candidati
SUPPORTED_MODELS = ["Oldroyd-B", "Giesekus", "PTT"]
