"""
Script CLI principale per l'esecuzione del modulo di Active Learning & DoE.
Addestra il Gaussian Process sui dati storici di inverse_runs.csv
e genera il batch dei prossimi 3 esperimenti ottimali per esplorare
i limiti di convergenza della PINN viscoelastica.
"""

import sys
import argparse
from pathlib import Path
import numpy as np

# Aggiunge la root del workspace al sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from active_learning.config import (
    INVERSE_RUNS_CSV,
    CONVERGENCE_THRESHOLD_PCT,
    BETA_EXPLORATION,
    PLOTS_DIR
)
from active_learning.data_loader import load_inverse_runs
from active_learning.gp_boundary import BoundaryGaussianProcess
from active_learning.visualizer import plot_active_learning_doe


def main():
    parser = argparse.ArgumentParser(
        description="Active Learning DoE: suggerisce il prossimo batch di esperimenti per esplorare i limiti della PINN."
    )
    parser.add_argument("--batch-size", type=int, default=3, help="Numero di esperimenti nel batch suggerito (default: 3)")
    parser.add_argument("--beta", type=float, default=BETA_EXPLORATION, help="Peso dell'esplorazione (default: 1.96)")
    parser.add_argument("--threshold", type=float, default=CONVERGENCE_THRESHOLD_PCT, help="Soglia errore di convergenza in %% (default: 10.0)")
    parser.add_argument("--model", type=str, default=None, choices=["Oldroyd-B", "Giesekus", "PTT"], help="Filtra per specifico modello fluido (default: tutti)")
    parser.add_argument("--diverse-models", action="store_true", help="Forza la selezione di modelli costitutivi diversi nel batch")
    parser.add_argument("--composition", type=str, default=None, help="Composizione vincolata del batch separata da virgole (es. 'Giesekus,Giesekus,Oldroyd-B')")
    args = parser.parse_args()

    model_sequence = [m.strip() for m in args.composition.split(",")] if args.composition else None

    print("=" * 80)
    print(" ACTIVE LEARNING & BAYESIAN DoE - FRONTIERA DI CONVERGENZA PINN")
    print("=" * 80)

    # 1. Caricamento dati storici
    print(f"\n[1/4] Caricamento storico run da:\n      {INVERSE_RUNS_CSV}")
    X, y, df = load_inverse_runs(INVERSE_RUNS_CSV)
    n_runs = len(df)
    n_conv = int((df["converged"] == True).sum())
    n_fail = n_runs - n_conv
    print(f"      -> {n_runs} run storiche caricate con successo:")
    print(f"         - Convergenti (errore < {args.threshold}%): {n_conv}")
    print(f"         - Oltre soglia o alto errore: {n_fail}")
    print(f"         - Range lambda storico: [{df['lambda'].min():.2f}, {df['lambda'].max():.2f}]")
    print(f"         - Range eta_p storico:  [{df['eta_p'].min():.2f}, {df['eta_p'].max():.2f}]")

    # 2. Addestramento Gaussian Process
    print(f"\n[2/4] Fitting Gaussian Process Regressor...")
    print(f"      - Kernel: Constant * Matern5/2(ARD) + WhiteKernel")
    print(f"      - Soglia convergenza target: {args.threshold}% (log10 = {args.threshold / 10.0})")
    print(f"      - Parametro di confidenza esplorazione beta: {args.beta}")
    
    gp_model = BoundaryGaussianProcess(
        threshold_log=float(np.log10(args.threshold)),
        beta=args.beta,
        random_state=42
    )
    gp_model.fit(X, y)
    print("      -> GP addestrato con successo.")

    # 3. Selezione Batch tramite Kriging Believer
    filter_info = f" (Filtro modello: {args.model})" if args.model else ""
    diverse_info = " [Modalità Diverse Models Attiva]" if args.diverse_models else ""
    comp_info = f" [Composizione Sequenza: {model_sequence}]" if model_sequence else ""
    effective_b_size = len(model_sequence) if model_sequence else args.batch_size
    print(f"\n[3/4] Ricerca e selezione del Batch di {effective_b_size} NUOVI esperimenti (Kriging Believer){filter_info}{diverse_info}{comp_info}...")
    batch = gp_model.suggest_batch(
        batch_size=args.batch_size,
        fluid_model_filter=args.model,
        diverse_models=args.diverse_models,
        model_sequence=model_sequence
    )
    print("      -> Ottimizzazione completata.\n")

    # 4. Presentazione Tabellare dei Risultati
    print("=" * 80)
    print(f" BATCH SUGGERITO: I PROSSIMI {len(batch)} ESPERIMENTI DA SIMULARE SU COMSOL")
    print("=" * 80)

    # Assicura encoding UTF-8 per console Windows
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    for item in batch:
        print(f"\n[{item['batch_rank']}] NOME FILE COMSOL / DATASET:")
        print(f"    -> {item['filename']}")
        print(f"    Modello Fluido:  {item['fluid_model']}")
        print(f"    Parametri:       lambda = {item['lambda']:.2f} | eta_p = {item['eta_p']:.2f} | eta_s = {item['eta_s']:.2f} | alpha = {item['alpha']:.2f} | eps = {item['eps']:.2f}")
        print(f"    Mesh consigliata: {item['mesh']} ({item['n_points']:,} nodi)")
        print(f"    Stima GP Errore: ~{item['pred_err_pct']:.1f}% (Incertezza epistemica σ: {item['pred_sigma_log']:.2f})")
        print(f"    Acquisition:     {item['acquisition_score']:.3f}")
        print(f"    Razionale:       {item['rationale']}")

    # 5. Generazione Grafico Diagnostico
    print(f"\n[4/4] Generazione mappa della frontiera di convergenza...")
    plot_path = plot_active_learning_doe(gp_model, df, batch)
    print(f"      -> Grafico salvato in: {plot_path}")

    print("\n" + "=" * 80)
    print(" ISTRUZIONI OPERATIVE:")
    print(" 1. Configura ed esporta in COMSOL il dataset per uno o più dei punti sopra indicati.")
    print(" 2. Salva il file CSV in 'COMSOL/4roll/Datasets/<nome_file.csv>'.")
    print(" 3. Esegui il training con 'train_4roll_main.py'.")
    print(" 4. Sincronizza 'inverse_runs.csv' con 'python final_roll/sync_inverse_runs.py'.")
    print(" 5. Rilancia questo script per aggiornare la mappa della frontiera di convergenza!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
