#!/usr/bin/env python3
"""
Batch Runner Kaggle: Studio Comparativo delle 3 Ipotesi su Fluido LPTT
Dataset di Riferimento: 4_roll_mill_L1-P0.5-S0.5-A0-E0.3-M5k.csv
Baseline di Riferimento (Locale): [INV][L1-P0.5-S0.5-A0-E0.3_M5k][Ph1_40k+10k] (No Warmup, Softplus)

Le 3 Run eseguite in sequenza entro il limite delle 9 ore di Kaggle (~2.5h ciascuna):
- Run 1 (Fattore Isolato: Warmup):              --warmup 8000 --eps-param softplus
- Run 2 (Fattore Isolato: Parametrizzazione Exp): --warmup 0    --eps-param exp
- Run 3 (Sinergia Combinata: Warmup + Exp):       --warmup 8000 --eps-param exp

Isolamento totale della memoria VRAM garantito dall'esecuzione in sottoprocessi dedicati.
"""

import os
import sys
import time
import json
import subprocess
from datetime import datetime
from pathlib import Path

# Setup percorsi con auto-rilevamento robusto
SCRIPT_DIR = Path(__file__).resolve().parent
if (SCRIPT_DIR / "train_4roll_main.py").exists():
    FINAL_ROLL_DIR = SCRIPT_DIR
elif (SCRIPT_DIR.parent / "final_roll" / "train_4roll_main.py").exists():
    FINAL_ROLL_DIR = SCRIPT_DIR.parent / "final_roll"
else:
    FINAL_ROLL_DIR = SCRIPT_DIR

PYTHON_EXE = sys.executable

# Dataset comune a tutte le run (M5k, L1, P0.5, S0.5, A0, E0.3)
DATASET_NAME = "4_roll_mill_L1-P0.5-S0.5-A0-E0.3-M5k.csv"

# Definizione dei 3 esperimenti
EXPERIMENTS = [
    {
        "id": 1,
        "name": "Kaggle_Run1_WarmupOnly",
        "description": "Effetto Isolato: Warmup (8000 epoche) con Softplus",
        "args": [
            "--dataset", DATASET_NAME,
            "--warmup", "8000",
            "--eps-param", "softplus"
        ]
    },
    {
        "id": 2,
        "name": "Kaggle_Run2_ExpOnly",
        "description": "Effetto Isolato: Parametrizzazione Esponenziale (No Warmup)",
        "args": [
            "--dataset", DATASET_NAME,
            "--warmup", "0",
            "--eps-param", "exp"
        ]
    },
    {
        "id": 3,
        "name": "Kaggle_Run3_WarmupPlusExp",
        "description": "Sinergia Combinata: Warmup (8000 epoche) + Parametrizzazione Esponenziale",
        "args": [
            "--dataset", DATASET_NAME,
            "--warmup", "8000",
            "--eps-param", "exp"
        ]
    }
]

# Baseline di riferimento locale (estratta dall'analisi preliminare)
BASELINE_METRICS = {
    "name": "Baseline (Locale PC)",
    "warmup": "0 epoche",
    "param_eps": "softplus",
    "lambda_est": 0.4211,
    "lambda_err": -57.89,
    "mu_p_est": 0.2128,
    "mu_p_err": -57.44,
    "G_est": 0.5054,
    "G_err": +1.07,
    "eps_est": 0.00017,
    "eps_err": -99.94,
    "status": "COMPLETATA (Trappola Canyon/Softplus)"
}

def print_banner(text):
    print("\n" + "=" * 85)
    print(f"  {text}")
    print("=" * 85)

def find_latest_run_dir(start_time_stamp):
    """Trova la cartella creata più di recente in output_4rollmill."""
    out_base = FINAL_ROLL_DIR / "output_4rollmill"
    candidates = [d for d in out_base.glob("*L1-P0.5-S0.5-A0-E0.3*") if d.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]

def main():
    print_banner("AVVIO BATCH RUNNER KAGGLE: VERIFICA DELLE 3 IPOTESI (LPTT E=0.3)")
    print(f"  Inizio sessione: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Interprete Python: {PYTHON_EXE}")
    print(f"  Cartella di lavoro: {FINAL_ROLL_DIR}")
    print(f"  Dataset di riferimento: {DATASET_NAME}")
    print(f"  Totale esperimenti programmati: {len(EXPERIMENTS)}")

    results = []
    total_start = time.time()

    for exp in EXPERIMENTS:
        exp_id = exp["id"]
        exp_name = exp["name"]
        exp_desc = exp["description"]
        cmd_args = exp["args"]

        print_banner(f"ESPERIMENTO [{exp_id}/{len(EXPERIMENTS)}]: {exp_name}\n  Descrizione: {exp_desc}")
        print(f"  Comando: {PYTHON_EXE} train_4roll_main.py {' '.join(cmd_args)}")

        run_start = time.time()
        start_ts = datetime.now()

        cmd = [PYTHON_EXE, str(FINAL_ROLL_DIR / "train_4roll_main.py")] + cmd_args

        # Esecuzione in sottoprocesso isolato (rilascio totale VRAM e cache CUDA)
        proc = subprocess.run(cmd, cwd=str(FINAL_ROLL_DIR))

        elapsed_sec = time.time() - run_start
        elapsed_min = elapsed_sec / 60.0
        success = (proc.returncode == 0)

        print(f"\n  [Terminato] Esito: {'SUCCESSO' if success else 'ERRORE'} | Durata: {elapsed_min:.1f} minuti")

        # Recupera la cartella di output e le metriche
        run_dir = find_latest_run_dir(start_ts)
        metrics = {
            "id": exp_id,
            "name": exp_name,
            "desc": exp_desc,
            "duration_min": round(elapsed_min, 1),
            "success": success,
            "run_dir": str(run_dir.name) if run_dir else "N/D"
        }

        if run_dir and (run_dir / "metrics_summary.json").exists():
            try:
                with open(run_dir / "metrics_summary.json", "r") as f:
                    ms = json.load(f)
                metrics["lambda_est"] = round(ms.get("lambda_estimated", 0.0), 4)
                metrics["lambda_err"] = round(ms.get("lambda_error_pct", 0.0), 2)
                metrics["mu_p_est"] = round(ms.get("mu_p_estimated", 0.0), 4)
                metrics["mu_p_err"] = round(ms.get("mu_p_error_pct", 0.0), 2)
                l_est = ms.get("lambda_estimated", 1e-12)
                m_est = ms.get("mu_p_estimated", 0.0)
                g_est = m_est / (l_est if abs(l_est) > 1e-12 else 1e-12)
                metrics["G_est"] = round(g_est, 4)
                metrics["G_err"] = round(((g_est - 0.5) / 0.5) * 100, 2)
                metrics["eps_est"] = round(ms.get("eps_estimated", 0.0), 6)
                metrics["eps_err"] = round(ms.get("eps_error_pct", 0.0), 2)
                metrics["l2_uv"] = round(ms.get("l2_uv_mean", 0.0), 5)
                metrics["l2_tau"] = round(ms.get("l2_tau_diag_mean", 0.0), 5)
            except Exception as e:
                print(f"  [Avviso] Errore lettura metrics_summary.json: {e}")
        else:
            # Fallback: lettura diretta da checkpoint.pth se metrics_summary non presente
            if run_dir and (run_dir / "checkpoint.pth").exists():
                try:
                    import torch
                    chk = torch.load(run_dir / "checkpoint.pth", map_location="cpu")
                    psd = chk.get("physics_state_dict", {})
                    glam = psd.get("guess_lam", torch.tensor(0.8)).item()
                    gmup = psd.get("guess_mu_p", torch.tensor(0.4)).item()
                    rlam = psd.get("_raw_lam", torch.tensor(0.0)).item()
                    rmup = psd.get("_raw_mu_p", torch.tensor(0.0)).item()
                    reps = psd.get("_raw_eps", torch.tensor(0.0)).item()

                    l_val = glam * float(torch.exp(torch.tensor(rlam)).item())
                    m_val = gmup * float(torch.exp(torch.tensor(rmup)).item())
                    g_val = m_val / (l_val + 1e-12)

                    # Calcolo eps in base al tipo
                    if "--eps-param" in cmd_args and cmd_args[cmd_args.index("--eps-param") + 1] == "exp":
                        e_val = 0.25 * float(torch.exp(torch.tensor(reps)).item())
                    else:
                        e_val = float(torch.nn.functional.softplus(torch.tensor(reps)).item())

                    metrics["lambda_est"] = round(l_val, 4)
                    metrics["lambda_err"] = round(((l_val - 1.0) / 1.0) * 100, 2)
                    metrics["mu_p_est"] = round(m_val, 4)
                    metrics["mu_p_err"] = round(((m_val - 0.5) / 0.5) * 100, 2)
                    metrics["G_est"] = round(g_val, 4)
                    metrics["G_err"] = round(((g_val - 0.5) / 0.5) * 100, 2)
                    metrics["eps_est"] = round(e_val, 6)
                    metrics["eps_err"] = round(((e_val - 0.3) / 0.3) * 100, 2)
                except Exception as e:
                    print(f"  [Avviso] Errore lettura checkpoint: {e}")

        results.append(metrics)

    total_time_hours = (time.time() - total_start) / 3600.0

    # Stampa e salvataggio Tabella Comparativa Finale
    print_banner(f"SESSIONE KAGGLE COMPLETATA IN {total_time_hours:.2f} ORE")

    report_lines = []
    report_lines.append("# Tabella Comparativa Sperimentale: Verifica delle 3 Ipotesi (LPTT $\\varepsilon=0.3$)")
    report_lines.append(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Durata Totale: {total_time_hours:.2f} h\n")
    report_lines.append("| Esperimento | Modifica Isolata | $\\lambda$ [s] (Vero: 1.0) | $\\mu_p$ [Pa·s] (Vero: 0.5) | $G = \\mu_p/\\lambda$ [Pa] (Vero: 0.5) | $\\varepsilon$ (Vero: 0.3) | Esito vs Baseline |")
    report_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

    # Baseline riga
    report_lines.append(
        f"| **Baseline Locale** | Nessuna (No Warmup, Softplus) | "
        f"{BASELINE_METRICS['lambda_est']:.4f} ({BASELINE_METRICS['lambda_err']:+.1f}%) | "
        f"{BASELINE_METRICS['mu_p_est']:.4f} ({BASELINE_METRICS['mu_p_err']:+.1f}%) | "
        f"{BASELINE_METRICS['G_est']:.4f} ({BASELINE_METRICS['G_err']:+.1f}%) | "
        f"{BASELINE_METRICS['eps_est']:.5f} ({BASELINE_METRICS['eps_err']:+.1f}%) | "
        f"Canyon trap |"
    )

    for r in results:
        l_str = f"{r.get('lambda_est', 'N/D')} ({r.get('lambda_err', 0):+.1f}%)" if 'lambda_est' in r else "N/D"
        m_str = f"{r.get('mu_p_est', 'N/D')} ({r.get('mu_p_err', 0):+.1f}%)" if 'mu_p_est' in r else "N/D"
        g_str = f"{r.get('G_est', 'N/D')} ({r.get('G_err', 0):+.1f}%)" if 'G_est' in r else "N/D"
        e_str = f"{r.get('eps_est', 'N/D')} ({r.get('eps_err', 0):+.1f}%)" if 'eps_est' in r else "N/D"

        # Giudizio sintetico
        if 'lambda_err' in r and abs(r['lambda_err']) < 10.0 and abs(r.get('eps_err', 100)) < 15.0:
            verdict = "**RISOLTO (Identificato)**"
        elif 'lambda_err' in r and abs(r['lambda_err']) < abs(BASELINE_METRICS['lambda_err']):
            verdict = "Migliorato"
        else:
            verdict = "Non risolto"

        report_lines.append(f"| **{r['name']}** | {r['desc']} | {l_str} | {m_str} | {g_str} | {e_str} | {verdict} |")

    report_content = "\n".join(report_lines)
    print("\n" + report_content + "\n")

    summary_file = SCRIPT_DIR / "kaggle_lptt_hypotheses_summary.md"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"Report riassuntivo salvato in: {summary_file}")

if __name__ == "__main__":
    main()
