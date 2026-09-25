import os
import sys
import subprocess
import time
from datetime import datetime
from pathlib import Path

# Percorsi base
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
PYTHON_EXEC = sys.executable

# Definizione della coda di training per i tre nuovi dataset (20k Adam + 5k L-BFGS ciascuno)
QUEUE = [
    {
        "id": 1,
        "name": "Caso 1: Oldroyd-B L=0.7",
        "dataset": "4_roll_mill_L0.7-P0.5-S0.5-A0-E0-M5k.csv",
        "transfer_ckpt": str(BASE_DIR / "checkpoints" / "checkpoint_inverso_fase1_L0.2-P0.5-S0.5_transfer.pth"),
        "adam1": 20000,
        "lbfgs1": 5000,
    },
    {
        "id": 2,
        "name": "Caso 2: Giesekus L=1.0 A=0.5",
        "dataset": "4_roll_mill_L1-P0.5-S0.5-A0.5-E0-M5k.csv",
        "transfer_ckpt": str(BASE_DIR / "output_4rollmill" / "[2026-09-24_17-13][INV][L1-P0.5-S0.5-A0.35-E0_M5k][Ph1_40k+10k]" / "checkpoint_lbfgs_phase1.pth"),
        "adam1": 20000,
        "lbfgs1": 5000,
    },
    {
        "id": 3,
        "name": "Caso 3: Giesekus L=0.7 A=0.35",
        "dataset": "4_roll_mill_L0.7-P0.5-S0.5-A0.35-E0-M5k.csv",
        "transfer_ckpt": str(BASE_DIR / "output_4rollmill" / "[2026-09-24_17-13][INV][L1-P0.5-S0.5-A0.35-E0_M5k][Ph1_40k+10k]" / "checkpoint_lbfgs_phase1.pth"),
        "adam1": 20000,
        "lbfgs1": 5000,
    },
]

def main():
    print("=" * 80)
    print("AVVIO CODA AUTOMATIZZATA TRANSFER LEARNING (3 RUN x 20k Adam + 5k L-BFGS)")
    print(f"Data/Ora Inizio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Interprete Python: {PYTHON_EXEC}")
    print(f"Totale Run in coda: {len(QUEUE)}")
    print("=" * 80)

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
        print(f"  Dataset:       {dataset}")
        print(f"  Donor Ckpt:    {transfer_ckpt}")
        print(f"  Budget:        {adam1} Adam + {lbfgs1} L-BFGS")
        print(f"{'#' * 80}\n")

        cmd = [
            PYTHON_EXEC,
            str(BASE_DIR / "train_4roll_main.py"),
            f"--dataset={dataset}",
            f"--transfer-ckpt={transfer_ckpt}",
            f"--adam1={adam1}",
            f"--lbfgs1={lbfgs1}",
        ]

        t0 = time.time()
        # Esecuzione del processo di training
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
            print(f"[ATTENZIONE] La run {task_id} si e' interrotta con codice d'errore {proc.returncode}.")

    total_elapsed = time.time() - total_start_time
    print("\n" + "=" * 80)
    print("RIASSUNTO FINALE CODA TRANSFER LEARNING")
    print(f"Tempo totale trascorso: {total_elapsed/3600.0:.2f} ore ({total_elapsed/60.0:.1f} minuti)")
    print("=" * 80)
    for r in results:
        print(f"  Run {r['id']} [{r['name']}]: {r['status']} ({r['elapsed_min']:.1f} min)")
    print("=" * 80)

if __name__ == "__main__":
    main()
