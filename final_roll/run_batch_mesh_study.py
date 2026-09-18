#!/usr/bin/env python3
"""
Batch Runner - Studio di Convergenza Mesh PINN (4-Roll Mill)
Esegue in sequenza le run sulle mesh specificate (default: 12k, 29k, 52k)
garantendo l'isolamento della VRAM tra processi e compilando la tabella finale.
"""

import os
import sys
import time
import json
import subprocess
from datetime import datetime
from pathlib import Path

# Directory di lavoro
SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON_EXE = sys.executable

# Meshes da eseguire in sequenza (modificabile da CLI es. python run_batch_mesh_study.py 12k 29k 52k)
DEFAULT_MESHES = ["12k", "29k", "52k"]
MESHES_TO_RUN = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_MESHES

print("=" * 80)
print("  AVVIO BATCH RUNNER - STUDIO DI CONVERGENZA MESH")
print(f"  Data e Ora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Python Interpreter: {PYTHON_EXE}")
print(f"  Sequenza Meshes da Eseguire: {MESHES_TO_RUN}")
print("=" * 80)

run_summaries = []

for idx, mesh in enumerate(MESHES_TO_RUN, 1):
    print(f"\n[{idx}/{len(MESHES_TO_RUN)}] >>> INIZIO TRAINING MESH: {mesh} <<<")
    start_t = time.time()
    
    cmd = [PYTHON_EXE, str(SCRIPT_DIR / "train_4roll_main.py"), "--mesh", mesh]
    
    # Esecuzione in sottoprocesso isolato per rilascio totale VRAM
    proc = subprocess.run(cmd, cwd=str(SCRIPT_DIR))
    
    elapsed_sec = time.time() - start_t
    elapsed_min = elapsed_sec / 60.0
    
    if proc.returncode != 0:
        print(f"\n[ERRORE] La run per mesh {mesh} e' terminata con codice di errore {proc.returncode}!")
        run_summaries.append({
            "mesh": mesh,
            "status": "FAILED",
            "elapsed_min": elapsed_min
        })
    else:
        print(f"\n[OK] Run per mesh {mesh} completata con successo in {elapsed_min:.1f} minuti.")
        # Cerca l'ultimo metrics_summary.json generato per questa mesh
        out_dir = SCRIPT_DIR / "output_4rollmill"
        matching_dirs = sorted(
            [d for d in out_dir.glob(f"*_M{mesh}*") if d.is_dir()],
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        summary_data = None
        if matching_dirs:
            json_file = matching_dirs[0] / "metrics_summary.json"
            if json_file.exists():
                with open(json_file, "r", encoding="utf-8") as f:
                    summary_data = json.load(f)
                    summary_data["folder"] = matching_dirs[0].name
                    summary_data["elapsed_min"] = elapsed_min
                    summary_data["status"] = "SUCCESS"
        
        if summary_data:
            run_summaries.append(summary_data)
        else:
            run_summaries.append({
                "mesh": mesh,
                "status": "SUCCESS (No JSON)",
                "elapsed_min": elapsed_min
            })

# ============================================================================
# COMPILAZIONE TABELLA FINALE
# ============================================================================
print("\n" + "=" * 95)
print("                    RIASSUNTO FINALE BENCHMARK CONVERGENZA MESH")
print("=" * 95)

table_header = (
    f"{'Mesh':<8s} | {'N. Punti':<10s} | {'Tempo (min)':<12s} | "
    f"{'Err L2 (u,v)':<14s} | {'Err L2 tau_xy':<14s} | "
    f"{'lambda [s] (Err%)':<18s} | {'mu_p [Pa*s] (Err%)':<18s}"
)
print(table_header)
print("-" * 95)

for s in run_summaries:
    if s.get("status") == "SUCCESS":
        mesh_str = s.get("mesh_tag", "?")
        n_pts = f"{s.get('n_points', 0):,}"
        t_str = f"{s.get('elapsed_min', 0.0):.1f} min"
        uv_str = f"{s.get('l2_uv_mean', 0.0)*100:.3f}%"
        txy_str = f"{s.get('l2_tau_xy', 0.0)*100:.3f}%"
        lam_val = s.get("lambda_estimated", 0.0)
        lam_err = s.get("lambda_error_pct", 0.0)
        lam_str = f"{lam_val:.5f} ({lam_err:+.1f}%)"
        mup_val = s.get("mu_p_estimated", 0.0)
        mup_err = s.get("mu_p_error_pct", 0.0)
        mup_str = f"{mup_val:.5f} ({mup_err:+.1f}%)"
        print(f"{mesh_str:<8s} | {n_pts:<10s} | {t_str:<12s} | {uv_str:<14s} | {txy_str:<14s} | {lam_str:<18s} | {mup_str:<18s}")
    else:
        m = s.get("mesh", "?")
        st = s.get("status", "UNKNOWN")
        print(f"{m:<8s} | {'N/A':<10s} | {s.get('elapsed_min', 0.0):.1f} min | {st}")

print("=" * 95)
print("\n[OK] Batch completato!")
