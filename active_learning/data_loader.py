"""
Caricamento e sanitizzazione delle run inverse storiche da inverse_runs.csv
per l'addestramento del Gaussian Process.
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict

from .config import INVERSE_RUNS_CSV, FEATURE_NAMES, MESH_NODES


def parse_eta_s_from_run_id(run_id: str, default_mu_p: float = 0.5) -> float:
    """Estrae il valore di S{eta_s} dal tag della run (es. -S0.5- o -S0.1-)."""
    match = re.search(r"-S([0-9\.]+)-", str(run_id))
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    return float(np.round(1.0 - default_mu_p, 4))


def load_inverse_runs(csv_path: Path = INVERSE_RUNS_CSV) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Legge inverse_runs.csv, calcola l'errore parametrico massimo E_param
    e restituisce la matrice delle feature X (N x 6) e il target logaritmico y (N).
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"File dati non trovato: {csv_path}")

    df = pd.read_csv(csv_path)

    # Rimuove eventuali righe vuote
    df = df.dropna(subset=["lambda_true", "mu_p_true"]).copy()

    records = []
    features_list = []
    targets_log = []

    for _, row in df.iterrows():
        lam = float(row["lambda_true"])
        mu_p = float(row["mu_p_true"])
        
        # eta_s
        eta_s = parse_eta_s_from_run_id(row.get("run_id", ""), default_mu_p=mu_p)

        # alpha e eps
        alpha = float(row["alpha_true"]) if pd.notna(row.get("alpha_true")) else 0.0
        eps = float(row["eps_true"]) if pd.notna(row.get("eps_true")) else 0.0

        # Indicatori binari del modello fluido (Oldroyd-B baseline = [0, 0])
        model_str = str(row.get("fluid_model", "")).lower()
        is_giesekus = 1.0 if "giesekus" in model_str else 0.0
        is_ptt = 1.0 if "ptt" in model_str else 0.0

        # mesh e n_points
        n_pts = row.get("n_points")
        if pd.isna(n_pts) or float(n_pts) <= 0:
            mesh_str = str(row.get("mesh", "5k"))
            n_pts = MESH_NODES.get(mesh_str, 5086)
        n_pts = float(n_pts)
        log10_n = float(np.log10(n_pts))

        # Calcolo errore massimo dei parametri fisici stimati
        errs = []
        if pd.notna(row.get("lambda_err_pct")):
            errs.append(abs(float(row["lambda_err_pct"])))
        if pd.notna(row.get("mu_p_err_pct")):
            errs.append(abs(float(row["mu_p_err_pct"])))
        
        # alpha_err_pct conta solo se alpha_true > 0 o modello Giesekus
        if alpha > 0 and pd.notna(row.get("alpha_err_pct")):
            errs.append(abs(float(row["alpha_err_pct"])))
            
        # eps_err_pct conta solo se eps_true > 0 o modello PTT
        if eps > 0 and pd.notna(row.get("eps_err_pct")):
            errs.append(abs(float(row["eps_err_pct"])))

        max_err = max(errs) if len(errs) > 0 else 5.0
        # Soglia minima di stabilità numerica a 0.05%
        max_err_safe = max(max_err, 0.05)
        log_err = float(np.log10(max_err_safe))

        # Vettore feature pulito (7 dimensioni): eta_s esclusa per evitare collinearità esatta
        features = [lam, mu_p, alpha, eps, is_giesekus, is_ptt, log10_n]
        features_list.append(features)
        targets_log.append(log_err)

        records.append({
            "run_id": row.get("run_id"),
            "fluid_model": row.get("fluid_model", "Unknown"),
            "lambda": lam,
            "eta_p": mu_p,
            "eta_s": eta_s,
            "alpha": alpha,
            "eps": eps,
            "is_giesekus": is_giesekus,
            "is_ptt": is_ptt,
            "n_points": int(n_pts),
            "max_param_err_pct": max_err,
            "log10_max_err": log_err,
            "converged": max_err < 10.0
        })

    X = np.array(features_list, dtype=np.float64)
    y = np.array(targets_log, dtype=np.float64)
    processed_df = pd.DataFrame(records)

    return X, y, processed_df
