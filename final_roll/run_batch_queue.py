"""
Script automatizzato per eseguire in sequenza la coda dei 3 esperimenti PINN:
1. Giesekus L0.7-A0.35 M12k (Mesh study da zero: 40k Adam + 10k L-BFGS)
2. Giesekus L0.3-A0.50 M5k  (Transfer Learning: 20k Adam + 5k L-BFGS)
3. Oldroyd-B L1.2-S0.02 M5k (Transfer Learning: 20k Adam + 5k L-BFGS)

Al termine della coda esegue automaticamente la sincronizzazione
di inverse_runs.csv e inverse_runs.md tramite sync_inverse_runs.py.
"""

import sys
import subprocess
import time
from datetime import datetime
from pathlib import Path

# Percorsi base
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

# Rilevamento interprete virtual environment
VENV_PYTHON = PROJECT_ROOT / "venv" / "Scripts" / "python.exe"
PYTHON_EXEC = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable

# Definizione ordinata della coda di addestramento
QUEUE = [
    {
        "id": 1,
        "name": "Giesekus L=0.7 A=0.35 M12k (Mesh Study da zero)",
        "dataset": "4_roll_mill_L0.7-P0.5-S0.5-A0.35-E0-M12k.csv",
        "transfer_ckpt": None,  # Cold start da zero
        "adam1": 40000,
        "lbfgs1": 10000,
    },
    {
        "id": 2,
        "name": "Giesekus L=0.3 A=0.50 M5k (Alta Mobilità / Solvente - TL)",
        "dataset": "4_roll_mill_L0.3-P0.05-S0.95-A0.5-E0-M5k.csv",
        "transfer_ckpt": str(
            BASE_DIR / "checkpoints" / "giesekus" / "checkpoint_inverso_fase1_L0.1-P0.5-S0.5-A0.35-E0_M5k_Ph1_40k+10k.pth"
        ),
        "adam1": 20000,
        "lbfgs1": 5000,
    },
    {
        "id": 3,
        "name": "Oldroyd-B L=1.2 S=0.02 M5k (Frontiera HWNP / Quasi-Assenza Solvente - TL)",
        "dataset": "4_roll_mill_L1.2-P0.98-S0.02-A0-E0-M5k.csv",
        "transfer_ckpt": str(
            BASE_DIR / "checkpoints" / "oldroyd" / "checkpoint_inverso_fase1_L0.7-P0.5-S0.5-A0-E0_M5k_TL_Ph1_20k+5k.pth"
        ),
        "adam1": 20000,
        "lbfgs1": 5000,
    },
]


def main():
    print("=" * 80)
    print(" AVVIO CODA AUTOMATIZZATA DI TRAINING PINN (3 RUN)")
    print(f" Data/Ora Inizio:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Interprete Python: {PYTHON_EXEC}")
    print(f" Totale Run in coda: {len(QUEUE)}")
    print("=" * 80)

    # Verifica preliminare esistenza file
    print("\nVerifica preliminare dei dataset e checkpoint richiesti:")
    for task in QUEUE:
        ds_path = PROJECT_ROOT / "COMSOL" / "4roll" / "Datasets" / task["dataset"]
        if not ds_path.exists():
            print(f"[ERRORE BLOCCANTE] Dataset non trovato: {ds_path}")
            sys.exit(1)
        if task["transfer_ckpt"]:
            ckpt_path = Path(task["transfer_ckpt"])
            if not ckpt_path.exists():
                print(f"[ERRORE BLOCCANTE] Checkpoint donor non trovato: {ckpt_path}")
                sys.exit(1)
        status_mode = f"Transfer Learning da {Path(task['transfer_ckpt']).name}" if task["transfer_ckpt"] else "Cold Start da zero"
        print(f"  [OK] Run {task['id']}: {task['dataset']} | {status_mode}")

    print("\nTutti i prerequisiti sono verificati. Inizio esecuzione sequenziale...\n")

    total_start_time = time.time()
    results = []

    for task in QUEUE:
        task_id = task["id"]
        task_name = task["name"]
        dataset = task["dataset"]
        transfer_ckpt = task["transfer_ckpt"]
        adam1 = task["adam1"]
        lbfgs1 = task["lbfgs1"]

        print(f"\n{'#' * 80}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] >>> AVVIO RUN {task_id}/{len(QUEUE)}: {task_name} <<<")
        print(f"  Dataset:     {dataset}")
        print(f"  Modalità:    {'Transfer Learning' if transfer_ckpt else 'Da Zero (Cold Start)'}")
        if transfer_ckpt:
            print(f"  Donor Ckpt:  {Path(transfer_ckpt).name}")
        print(f"  Budget:      {adam1} Adam + {lbfgs1} L-BFGS")
        print(f"{'#' * 80}\n")

        cmd = [
            PYTHON_EXEC,
            str(BASE_DIR / "train_4roll_main.py"),
            f"--dataset={dataset}",
            f"--adam1={adam1}",
            f"--lbfgs1={lbfgs1}",
        ]
        if transfer_ckpt:
            cmd.append(f"--transfer-ckpt={transfer_ckpt}")

        t0 = time.time()
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        elapsed = time.time() - t0

        status = "COMPLETATA" if proc.returncode == 0 else f"FALLITA (code {proc.returncode})"
        results.append({
            "id": task_id,
            "name": task_name,
            "status": status,
            "elapsed_min": elapsed / 60.0
        })

        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Run {task_id} ({task_name}) terminata con stato: {status} in {elapsed/60.0:.1f} minuti.")
        if proc.returncode != 0:
            print(f"[ATTENZIONE] La run {task_id} si è interrotta con codice d'errore {proc.returncode}. Proseguo comunque con la coda.")

    # Sincronizzazione automatica del database delle run
    print("\n" + "=" * 80)
    print("SINCRONIZZAZIONE FINALE DATABASE RUNS (sync_inverse_runs.py)...")
    print("=" * 80)
    try:
        sync_proc = subprocess.run(
            [PYTHON_EXEC, str(BASE_DIR / "sync_inverse_runs.py")],
            cwd=str(PROJECT_ROOT)
        )
        if sync_proc.returncode == 0:
            print("  -> Database inverse_runs.csv e inverse_runs.md aggiornati con successo.")
        else:
            print(f"  [ATTENZIONE] sync_inverse_runs.py terminato con codice {sync_proc.returncode}.")
    except Exception as e:
        print(f"  [ERRORE SINCRONIZZAZIONE] Impossibile eseguire sync_inverse_runs.py: {e}")

    total_elapsed = time.time() - total_start_time
    print("\n" + "=" * 80)
    print("RIASSUNTO FINALE CODA DI TRAINING")
    print(f"Tempo totale trascorso: {total_elapsed/3600.0:.2f} ore ({total_elapsed/60.0:.1f} minuti)")
    print("=" * 80)
    for r in results:
        print(f"  Run {r['id']} [{r['name']}]: {r['status']} ({r['elapsed_min']:.1f} min)")
    print("=" * 80)


if __name__ == "__main__":
    main()
