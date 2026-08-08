"""
===============================================================================
SCRIPT DI POST-PROCESSING E VALUTAZIONE RUN PINN (4-Roll Mill)
===============================================================================

Descrizione:
  Questo script gestisce la valutazione post-run, la generazione delle metriche
  e la creazione dei grafici diagnostici a partire da un checkpoint (.pth) salvato 
  durante o al termine dell'addestramento.

Funzionalità Principali:
  1. Auto-detection / Selezione Checkpoint:
     - Se avviato senza argomenti, trova automaticamente l'ultima run presente 
       nella cartella 'output_4rollmill/' e seleziona il checkpoint migliore disponibile 
       (dando priorità a checkpoint_lbfgs_phase2.pth, checkpoint.pth, ecc.).
     - Se avviato fornendo un percorso come argomento CLI (es. un file .pth o 
       una cartella di run), carica il checkpoint specificato.

  2. Auto-detection / Selezione Dataset:
     - Se il dataset non viene specificato da CLI, lo script deduce automaticamente 
       il file .csv corretto analizzando il nome della cartella della run e cercando 
       la corrispondenza nella cartella 'COMSOL/'.
     - Stampa a video in modo chiaro il checkpoint ed il dataset utilizzati.

  3. Ripristino dello Stato:
     - Ricostruisce l'architettura 'CombinedModel', la classe di fisica 'Physics' 
       e lo storico 'SimpleHistory'.
     - Ricarica pesi del modello, parametri identificati e cronologia delle loss.

  4. Calcolo Metriche & Report:
     - Stampa i parametri fisici identificati vs Ground Truth (mu_s, mu_p, lam, eps, alpha).
     - Calcola le loss finali sui vincoli PDE, BC e dati.
     - Calcola gli errori relativi L2 per velocità (u, v), pressione (p) e stress (tau_xx, tau_xy, tau_yy).

  5. Output Grafici:
     - Crea/aggiorna i grafici dello storico: loss_history.png, params_evolution.png, l2_errors_history.png.
     - Genera tutte le mappe 2D dei campi e i profili di errore tramite 'generate_all_diagnostics'.
     - Salva tutti i grafici prodotti sia nella cartella principale di run che nella 
       sottocartella dedicata 'postprocess_plots/'.

Uso:
  python postprocess_run.py                                             # Auto-detect ultima run e relativo dataset
  python postprocess_run.py output_4rollmill/NOME_RUN                   # Specificando la cartella di run
  python postprocess_run.py output_4rollmill/NOME_RUN dataset.csv       # Cartella run + dataset specifico
  python postprocess_run.py path/to/checkpoint.pth path/to/dataset.csv # Checkpoint + dataset specifico
===============================================================================
"""

import os
import sys
import argparse
import tempfile
import logging
from pathlib import Path

# Configurazione cartella di cache scrivibile per Matplotlib prima di importarlo
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib_cache")

import matplotlib
matplotlib.use("Agg")
logging.getLogger("matplotlib").setLevel(logging.ERROR)
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import builtins

# Percorsi di base
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

# Import dai moduli di progetto
import src.debug
import src.physics
import src.train
import src.utils

from src.train import CombinedModel, SimpleHistory
from src.physics import Physics, evaluate_final_losses, compute_l2_errors
from src.utils import load_data, generate_all_diagnostics

# Setup PyTorch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)

# Ground truth predefiniti per i plot e le metriche
MU_S_TRUE = 0.1
MU_P_TRUE = 0.9
LAM_TRUE = 1.0
EPS_TRUE = 0.0
ALPHA_TRUE = 0.0
BETA_TRUE = MU_S_TRUE / (MU_S_TRUE + MU_P_TRUE)
ACTIVATION = nn.SiLU


def resolve_checkpoint_and_dir(target_arg=None):
    """Individua il file di checkpoint e la cartella della run da processare."""
    output_base = BASE_DIR / "output_4rollmill"

    if target_arg:
        target_path = Path(target_arg).resolve()
        if target_path.is_file() and target_path.suffix == ".pth":
            return target_path, target_path.parent
        elif target_path.is_dir():
            run_dir = target_path
        else:
            raise FileNotFoundError(f"Percorso specificato non valido: {target_arg}")
    else:
        # Cerca la cartella di output più recente
        if not output_base.exists():
            raise FileNotFoundError(f"La cartella di output {output_base} non esiste.")
        subdirs = [d for d in output_base.iterdir() if d.is_dir()]
        if not subdirs:
            raise FileNotFoundError(f"Nessuna cartella trovata in {output_base}")
        run_dir = max(subdirs, key=lambda d: d.stat().st_mtime)
        print(f"[Auto-detect Run] Selezione dell'ultima run registrata: {run_dir.name}")

    # Priorità di ricerca del checkpoint
    checkpoint_candidates = [
        "checkpoint_lbfgs_phase2.pth",
        "checkpoint.pth",
        "checkpoint_lbfgs_phase1.pth",
        "checkpoint_phase2_adam.pth",
        "checkpoint_phase1_adam.pth",
    ]

    for cand in checkpoint_candidates:
        chk_file = run_dir / cand
        if chk_file.exists():
            return chk_file, run_dir

    # Se non c'è una corrispondenza esatta, cerca qualsiasi file .pth
    pth_files = sorted(run_dir.glob("*.pth"), key=lambda f: f.stat().st_mtime, reverse=True)
    if pth_files:
        return pth_files[0], run_dir

    raise FileNotFoundError(f"Nessun file di checkpoint (.pth) trovato in {run_dir}")


def find_all_available_datasets(comsol_dir):
    """Restituisce un dizionario con i dataset .csv disponibili in COMSOL."""
    datasets = {}
    if not comsol_dir.exists():
        return datasets
    for p in comsol_dir.rglob("*.csv"):
        datasets[p.stem] = p
        datasets[p.name] = p
    return datasets


def resolve_dataset(checkpoint_path, run_dir, explicit_dataset=None):
    """Individua in modo autonomo o esplicito il file di dataset COMSOL (.csv)."""
    comsol_dir = BASE_DIR.parent / "COMSOL"
    available_datasets = find_all_available_datasets(comsol_dir)

    # 1. Se fornito esplicitamente da CLI
    if explicit_dataset:
        exp_path = Path(explicit_dataset).resolve()
        if exp_path.is_file():
            return exp_path
        if explicit_dataset in available_datasets:
            return available_datasets[explicit_dataset]
        if Path(explicit_dataset).name in available_datasets:
            return available_datasets[Path(explicit_dataset).name]
        raise FileNotFoundError(f"Dataset specificato non trovato: {explicit_dataset}")

    # 2. Auto-deduzione dal nome della cartella di run e del file di checkpoint
    search_target = f"{run_dir.name}_{checkpoint_path.stem}".replace("=", "").lower()
    matched_ds = None
    best_match_score = 0

    for ds_key, ds_path in available_datasets.items():
        if ds_key.endswith(".csv"):
            continue
        key_norm = ds_key.replace("=", "").lower()

        # Match diretto di sottostringa
        if key_norm in search_target:
            score = len(key_norm) * 10
        else:
            # Match su token rilevanti (es. 'lambda1' o 'lambda=1')
            tokens = [t for t in key_norm.split("_") if len(t) > 2]
            matched_tokens = [t for t in tokens if t in search_target]
            score = len(matched_tokens) * 2 if matched_tokens else 0

        if score > best_match_score:
            matched_ds = ds_path
            best_match_score = score

    if matched_ds:
        return matched_ds

    # 3. Fallback se c'è un solo file .csv nella cartella COMSOL
    unique_csvs = list(set(available_datasets.values()))
    if len(unique_csvs) == 1:
        return unique_csvs[0]

    # Se non determinato, solleva errore con l'elenco dei dataset trovati
    avail_str = "\n".join([f"  - {p.relative_to(BASE_DIR.parent)}" for p in unique_csvs])
    target_descr = f"{run_dir.name}/{checkpoint_path.name}"
    raise ValueError(
        f"Impossibile auto-dedurre il dataset per '{target_descr}'.\n"
        f"Dataset trovati in COMSOL/:\n{avail_str}\n\n"
        f"Per favore specifica il dataset da riga di comando:\n"
        f"  python postprocess_run.py <run_dir_o_checkpoint> <percorso_dataset.csv>"
    )


def inject_global_constants(dataset_path):
    """Inietta le costanti globali in builtins e nei moduli src."""
    constants = {
        "MU_S_TRUE": MU_S_TRUE,
        "MU_P_TRUE": MU_P_TRUE,
        "LAM_TRUE": LAM_TRUE,
        "EPS_TRUE": EPS_TRUE,
        "ALPHA_TRUE": ALPHA_TRUE,
        "BETA_TRUE": BETA_TRUE,
        "GUESS_MULTIPLIER": 0.8,
        "GUESS_MU_S": MU_S_TRUE * 0.8,
        "GUESS_MU_P": MU_P_TRUE * 0.8,
        "GUESS_LAM": LAM_TRUE * 0.8,
        "GUESS_EPS": 0.05,
        "GUESS_ALPHA": 0.05,
        "HIDDEN_LAYERS": [128] * 8,
        "RHO": 1000.0,
        "VARIANCE_EPS": 1e-4,
        "W_BC": 5.0,
        "W_PHYSICS": 3.0,
        "W_DATA": 1.0,
        "W_MOMENTUM": 1.0,
        "W_CONSTITUTIVE": 1.0,
        "USE_ROLL_STRESS_BC": True,
        "W_ROLL_STRESS": 1.0,
        "ACTIVATION": ACTIVATION,
        "DEVICE": DEVICE,
        "DATASET_PATH": dataset_path,
    }
    for k, v in constants.items():
        setattr(builtins, k, v)
        for mod in [src.debug, src.physics, src.train, src.utils]:
            setattr(mod, k, v)


def main():
    parser = argparse.ArgumentParser(description="Script di Post-Processing PINN (4-Roll Mill)")
    parser.add_argument("target", nargs="?", default=None, help="Percorso della cartella di run o del file checkpoint (.pth)")
    parser.add_argument("dataset", nargs="?", default=None, help="Percorso o nome del file dataset COMSOL (.csv)")
    parser.add_argument("--checkpoint", type=str, default=None, help="File di checkpoint (.pth)")
    parser.add_argument("--dataset-path", type=str, default=None, help="File di dataset COMSOL (.csv)")

    args = parser.parse_args()

    target_input = args.checkpoint or args.target
    dataset_input = args.dataset_path or args.dataset

    # Resolving Checkpoint and Dataset
    checkpoint_path, run_dir = resolve_checkpoint_and_dir(target_input)
    dataset_path = resolve_dataset(checkpoint_path, run_dir, dataset_input)

    # Iniezione del contesto globale
    inject_global_constants(dataset_path)

    # Stampa in evidenza di Checkpoint e Dataset utilizzati
    print(f"\n{'='*70}")
    print(f" POST-PROCESSING E VALUTAZIONE RUN")
    print(f"{'='*70}")
    print(f" Cartella Run : {run_dir}")
    print(f" CHECKPOINT   : {checkpoint_path}")
    print(f" DATASET      : {dataset_path}")
    print(f"{'='*70}\n")

    # Sottocartella dei plot per il postprocessing
    postprocess_dir = run_dir / "postprocess_plots"
    postprocess_dir.mkdir(parents=True, exist_ok=True)

    # 1. Caricamento dati
    print("[1/4] Caricamento dataset...")
    data = load_data(filepath=dataset_path)

    # 2. Inizializzazione e caricamento checkpoint
    print("\n[2/4] Caricamento stato modello e fisica...")
    model = CombinedModel(p_scale=data["p_scale"], tau_scale=data["tau_scale"]).to(DEVICE)
    physics = Physics(
        U_ref=data["U_ref"],
        H_ref=data["H"],
        H_coord=data["H_coord"],
        var_weights=data["var_weights"],
        inverse_mode=True,
        tau_scale=data["tau_scale"],
        p_scale=data["p_scale"],
    ).to(DEVICE)

    chk = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(chk["model_state_dict"])
    physics.load_state_dict(chk["physics_state_dict"])

    history = SimpleHistory()
    if "history_state_dict" in chk:
        history.load_state_dict(chk["history_state_dict"])

    # 3. Valutazione Metriche
    print("\n[3/4] Calcolo metriche ed errori fisici...")
    params = physics.log_params()
    print(f"\n{'='*60}\nPARAMETRI FISICI IDENTIFICATI\n{'='*60}")
    for p_name, true_val in zip(
        ["mu_s", "mu_p", "lam", "eps", "alpha"],
        [MU_S_TRUE, MU_P_TRUE, LAM_TRUE, EPS_TRUE, ALPHA_TRUE],
    ):
        if p_name in params:
            print(f"  {p_name:<5s}: {params[p_name]:.6f}  (true: {true_val})")

    final_losses = evaluate_final_losses(model, physics, data)
    print(f"\n{'='*60}\nLOSS FINALI DETTAGLIATE\n{'='*60}")
    for k, v in final_losses.items():
        print(f"  {k:<20s}: {v:.6e}")

    errors = compute_l2_errors(model, physics, data)
    print(f"\n{'='*60}\nERRORI L2 RELATIVI\n{'='*60}")
    for fn, err in errors.items():
        print(f"  {fn:>8s}: {err:.6f}")

    # 4. Generazione Grafici e Diagnostic
    print("\n[4/4] Generazione grafici e diagnostica dei campi...")
    
    # Genera i grafici principali sia nella sottocartella che nella cartella principale di run
    for target_folder in [postprocess_dir, run_dir]:
        if history.epochs:
            history.plot_losses(str(target_folder / "loss_history.png"))
            history.plot_params(str(target_folder / "params_evolution.png"))
            history.plot_l2_errors(str(target_folder / "l2_errors_history.png"))
    
    # Mappe dei campi di velocità, pressione e stress
    generate_all_diagnostics(model, physics, data, str(postprocess_dir))

    print(f"\n[OK] Post-processing completato con successo!")
    print(f"    Checkpoint utilizzato : {checkpoint_path}")
    print(f"    Dataset utilizzato    : {dataset_path}")
    print(f"    Grafici salvati in    : {postprocess_dir}")


if __name__ == "__main__":
    main()
