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

# Nomi delle feature in input al Gaussian Process (rimossa eta_s collinearmente ridondante)
FEATURE_NAMES = ["lambda", "eta_p", "alpha", "eps", "is_giesekus", "is_ptt", "log10_n_points"]

# Costi computazionali relativi indicativi per risoluzione di mesh (5k baseline = 1.0)
MESH_COST = {
    "5k": 1.0,
    "12k": 2.4,
    "29k": 5.6,
    "52k": 9.8,
    "88k": 16.2,
    "125k": 22.8
}

# Range fisici di esplorazione sensati per il problema 4-roll mill (unica sorgente di verità)
PARAM_BOUNDS = {
    "lambda": (0.005, 2.00),      # Tempo di rilassamento / Weissenberg proxy
    "eta_p": (0.02, 0.98),        # Viscosità polimerica (con eta_s = 1.0 - eta_p)
    "alpha": (0.00, 0.50),        # Parametro mobilità di Giesekus (0 per Oldroyd-B/PTT)
    "eps": (0.00, 0.50),          # Parametro reticolare PTT (0 per Oldroyd-B/Giesekus)
    "is_giesekus": (0.0, 1.0),    # Indicatore binario modello Giesekus
    "is_ptt": (0.0, 1.0),         # Indicatore binario modello PTT
    "log10_n_points": (np.log10(5000), np.log10(130000))  # Scala logaritmica dei punti
}

# Metrica di convergenza: errore percentuale relativo massimo dei parametri fisici
# Se l'errore max è sotto il 10%, la PINN è considerata convergente per identificazione inversa.
CONVERGENCE_THRESHOLD_PCT = 10.0
CONVERGENCE_THRESHOLD_LOG = float(np.log10(CONVERGENCE_THRESHOLD_PCT))  # log10(10) = 1.0

# Parametro di esplorazione per Straddle acquisition function: a(x) = beta * sigma(x) - |mu(x) - gamma|
# Valore 1.96 corrisponde all'intervallo di confidenza al 95%.
BETA_EXPLORATION = 1.96

# Modelli reologici ammessi per generazione candidati
SUPPORTED_MODELS = ["Oldroyd-B", "Giesekus", "PTT"]
