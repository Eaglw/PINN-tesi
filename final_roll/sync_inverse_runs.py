"""
Script di utilità per aggregare, sincronizzare ed esportare lo storico
completo delle run inverse di PINN in un database CSV centralizzato
(output_4rollmill/inverse_runs.csv) e in tabella Markdown (inverse_runs.md).
"""

import os
import re
import json
from pathlib import Path
import pandas as pd


# Cartella principale dei risultati
OUTPUT_BASE = Path(__file__).resolve().parent / "output_4rollmill"
CSV_OUT = OUTPUT_BASE / "inverse_runs.csv"
MD_OUT = OUTPUT_BASE / "inverse_runs.md"

# Colonne standard del CSV
CSV_COLUMNS = [
    "run_id",
    "timestamp",
    "fluid_model",
    "mesh",
    "n_points",
    "lambda_true",
    "lambda_est",
    "lambda_err_pct",
    "mu_p_true",
    "mu_p_est",
    "mu_p_err_pct",
    "alpha_true",
    "alpha_est",
    "alpha_err_pct",
    "eps_true",
    "eps_est",
    "eps_err_pct",
    "err_L2_u_pct",
    "err_L2_v_pct",
    "err_L2_uv_pct",
    "err_L2_tau_xy_pct",
    "err_L2_tau_xx_pct",
    "err_L2_tau_yy_pct",
    "err_L2_tau_diag_pct",
    "final_loss_total",
    "final_loss_pde",
    "final_loss_bc",
    "final_loss_data",
    "notes"
]

# Run storiche documentate in SUMMARY_RUNS.md antecedenti all'introduzione di metrics_summary.json
HISTORICAL_RUNS = [
    {
        "run_id": "[2026-08-26_16-28][INV][L0.05-P0.9-S0.1-A0-E0_M125k][Ph1_40k+10k]",
        "timestamp": "2026-08-26 16:28",
        "fluid_model": "Oldroyd-B",
        "mesh": "125k",
        "n_points": 125000,
        "lambda_true": 0.05,
        "lambda_est": 0.05020,
        "lambda_err_pct": 0.41,
        "mu_p_true": 0.90,
        "mu_p_est": 0.9049,
        "mu_p_err_pct": 0.54,
        "alpha_true": 0.0,
        "alpha_est": 0.0,
        "alpha_err_pct": None,
        "eps_true": 0.0,
        "eps_est": 0.0,
        "eps_err_pct": None,
        "err_L2_u_pct": 0.04,
        "err_L2_v_pct": 0.04,
        "err_L2_uv_pct": 0.04,
        "err_L2_tau_xy_pct": 0.65,
        "err_L2_tau_xx_pct": 0.77,
        "err_L2_tau_yy_pct": 0.77,
        "err_L2_tau_diag_pct": 0.77,
        "final_loss_total": 0.0,
        "final_loss_pde": 0.0,
        "final_loss_bc": 0.0,
        "final_loss_data": 0.0,
        "notes": "Record Assoluto Storico Fase 1"
    },
    {
        "run_id": "[2026-09-11_21-31][INV][L0.05-P0.5-S0.5-A0-E0_M125k][Ph1_40k+10k][mauri]",
        "timestamp": "2026-09-11 21:31",
        "fluid_model": "Oldroyd-B",
        "mesh": "125k",
        "n_points": 125000,
        "lambda_true": 0.05,
        "lambda_est": 0.05136,
        "lambda_err_pct": 2.72,
        "mu_p_true": 0.50,
        "mu_p_est": 0.5140,
        "mu_p_err_pct": 2.80,
        "alpha_true": 0.0,
        "alpha_est": 0.0,
        "alpha_err_pct": None,
        "eps_true": 0.0,
        "eps_est": 0.0,
        "eps_err_pct": None,
        "err_L2_u_pct": 0.13,
        "err_L2_v_pct": 0.13,
        "err_L2_uv_pct": 0.13,
        "err_L2_tau_xy_pct": 1.80,
        "err_L2_tau_xx_pct": 2.80,
        "err_L2_tau_yy_pct": 2.80,
        "err_L2_tau_diag_pct": 2.80,
        "final_loss_total": 0.0,
        "final_loss_pde": 0.0,
        "final_loss_bc": 0.0,
        "final_loss_data": 0.0,
        "notes": "Fase 1 Mauri (G=10.007 Pa)"
    },
    {
        "run_id": "[2026-09-13_17-13][INV][TRANSFER_LEARNING][L0.1-P0.5-S0.5-A0-E0_M125k][TL_Ph1_20k+5k]",
        "timestamp": "2026-09-13 17:13",
        "fluid_model": "Oldroyd-B",
        "mesh": "125k",
        "n_points": 125000,
        "lambda_true": 0.10,
        "lambda_est": 0.1037,
        "lambda_err_pct": 3.70,
        "mu_p_true": 0.50,
        "mu_p_est": 0.5191,
        "mu_p_err_pct": 3.82,
        "alpha_true": 0.0,
        "alpha_est": 0.0,
        "alpha_err_pct": None,
        "eps_true": 0.0,
        "eps_est": 0.0,
        "eps_err_pct": None,
        "err_L2_u_pct": 0.28,
        "err_L2_v_pct": 0.28,
        "err_L2_uv_pct": 0.28,
        "err_L2_tau_xy_pct": 2.40,
        "err_L2_tau_xx_pct": 3.70,
        "err_L2_tau_yy_pct": 3.70,
        "err_L2_tau_diag_pct": 3.70,
        "final_loss_total": 0.0,
        "final_loss_pde": 0.0,
        "final_loss_bc": 0.0,
        "final_loss_data": 0.0,
        "notes": "Transfer Learning (lam 0.05 -> 0.1)"
    },
    {
        "run_id": "[2026-09-14_06-46][INV][TRANSFER_LEARNING][L0.2-P0.5-S0.5-A0-E0_M125k][TL_Ph1_20k+5k]",
        "timestamp": "2026-09-14 06:46",
        "fluid_model": "Oldroyd-B",
        "mesh": "125k",
        "n_points": 125000,
        "lambda_true": 0.20,
        "lambda_est": 0.2081,
        "lambda_err_pct": 4.05,
        "mu_p_true": 0.50,
        "mu_p_est": 0.5214,
        "mu_p_err_pct": 4.28,
        "alpha_true": 0.0,
        "alpha_est": 0.0,
        "alpha_err_pct": None,
        "eps_true": 0.0,
        "eps_est": 0.0,
        "eps_err_pct": None,
        "err_L2_u_pct": 0.31,
        "err_L2_v_pct": 0.31,
        "err_L2_uv_pct": 0.31,
        "err_L2_tau_xy_pct": 2.80,
        "err_L2_tau_xx_pct": 4.10,
        "err_L2_tau_yy_pct": 4.10,
        "err_L2_tau_diag_pct": 4.10,
        "final_loss_total": 0.0,
        "final_loss_pde": 0.0,
        "final_loss_bc": 0.0,
        "final_loss_data": 0.0,
        "notes": "Transfer Learning (lam 0.1 -> 0.2)"
    }
]


def parse_timestamp_from_name(folder_name: str) -> str:
    m = re.search(r"\[(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2})\]", folder_name)
    if m:
        return f"{m.group(1)} {m.group(2).replace('-', ':')}"
    return ""


def parse_fluid_model(alpha_true: float, eps_true: float) -> str:
    if alpha_true > 0.0:
        return "Giesekus"
    elif eps_true > 0.0:
        return "PTT"
    else:
        return "Oldroyd-B"


def collect_all_runs():
    runs = []
    seen_run_ids = set()

    # 1. Scansione cartelle con metrics_summary.json
    for folder in sorted(OUTPUT_BASE.iterdir()):
        if not folder.is_dir():
            continue
        if "[INV]" not in folder.name or "[STOPPED]" in folder.name:
            continue

        json_file = folder / "metrics_summary.json"
        if not json_file.exists():
            continue

        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Errore lettura {json_file}: {e}")
            continue

        run_id = folder.name
        ts = parse_timestamp_from_name(run_id)
        alpha_t = data.get("alpha_true", 0.0) or 0.0
        eps_t = data.get("eps_true", 0.0) or 0.0
        model = parse_fluid_model(alpha_t, eps_t)
        mesh = data.get("mesh_tag", "")
        if not mesh:
            m_match = re.search(r"_M(\w+)", run_id)
            mesh = m_match.group(1) if m_match else "unknown"

        note = ""
        if "Retry" in run_id or "20-24" in run_id:
            note = "Retry (Guess cieco alpha=0.25)"
        elif "13-54" in run_id:
            note = "1st attempt (Guess alpha=0.08)"
        elif "17-07" in run_id:
            note = "Validazione Giesekus A0.35 (Guess cieco 0.25)"
        elif mesh == "5k" and model == "Oldroyd-B":
            note = "Record Accuratezza Mesh Ultraleggera"

        row = {
            "run_id": run_id,
            "timestamp": ts,
            "fluid_model": model,
            "mesh": mesh,
            "n_points": data.get("n_points", 0),
            "lambda_true": data.get("lambda_true"),
            "lambda_est": data.get("lambda_estimated"),
            "lambda_err_pct": data.get("lambda_error_pct"),
            "mu_p_true": data.get("mu_p_true"),
            "mu_p_est": data.get("mu_p_estimated"),
            "mu_p_err_pct": data.get("mu_p_error_pct"),
            "alpha_true": data.get("alpha_true"),
            "alpha_est": data.get("alpha_estimated"),
            "alpha_err_pct": data.get("alpha_error_pct"),
            "eps_true": data.get("eps_true"),
            "eps_est": data.get("eps_estimated"),
            "eps_err_pct": data.get("eps_error_pct"),
            "err_L2_u_pct": data.get("l2_u", 0.0) * 100 if data.get("l2_u") else None,
            "err_L2_v_pct": data.get("l2_v", 0.0) * 100 if data.get("l2_v") else None,
            "err_L2_uv_pct": data.get("l2_uv_mean", 0.0) * 100 if data.get("l2_uv_mean") else None,
            "err_L2_tau_xy_pct": data.get("l2_tau_xy", 0.0) * 100 if data.get("l2_tau_xy") else None,
            "err_L2_tau_xx_pct": data.get("l2_tau_xx", 0.0) * 100 if data.get("l2_tau_xx") else None,
            "err_L2_tau_yy_pct": data.get("l2_tau_yy", 0.0) * 100 if data.get("l2_tau_yy") else None,
            "err_L2_tau_diag_pct": data.get("l2_tau_diag_mean", 0.0) * 100 if data.get("l2_tau_diag_mean") else None,
            "final_loss_total": data.get("final_loss_total"),
            "final_loss_pde": data.get("final_loss_pde"),
            "final_loss_bc": data.get("final_loss_bc"),
            "final_loss_data": data.get("final_loss_data"),
            "notes": note
        }
        runs.append(row)
        seen_run_ids.add(run_id)

    # 2. Integrazione run storiche se non già presenti
    for hist in HISTORICAL_RUNS:
        if hist["run_id"] not in seen_run_ids:
            runs.append(hist)
            seen_run_ids.add(hist["run_id"])

    # Ordina cronologicamente (più recenti prima o dopo)
    runs.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    return runs


def export_database():
    runs = collect_all_runs()
    df = pd.DataFrame(runs, columns=CSV_COLUMNS)

    # Salva CSV
    df.to_csv(CSV_OUT, index=False)
    print(f"[OK] Database CSV esportato con successo in: {CSV_OUT} ({len(df)} run)")

    # Salva Markdown
    with open(MD_OUT, "w", encoding="utf-8") as f_md:
        f_md.write("# Registro Storico Run Inverse (PINN Viscoelastic Fluid)\n\n")
        f_md.write(f"Database consolidato generato automaticamente da `sync_inverse_runs.py`.\n")
        f_md.write(f"Totale esperimenti catalogati: **{len(df)}**.\n\n")
        
        f_md.write("| Data / Run | Fluido | Mesh | $\\lambda$ True | $\\lambda$ Est (Err%) | $\\mu_p$ True | $\\mu_p$ Est (Err%) | $\\alpha$ True | $\\alpha$ Est (Err%) | $\\varepsilon$ Est | Err $L_2(u,v)$ | Err $L_2(\\tau_{xy})$ | Note |\n")
        f_md.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        
        for r in runs:
            ts = r["timestamp"]
            fl = r["fluid_model"]
            m = r["mesh"]
            lt = f"{r['lambda_true']:.3f} s" if r['lambda_true'] is not None else "-"
            le = f"{r['lambda_est']:.4f} s" if r['lambda_est'] is not None else "-"
            le_err = f"(**{r['lambda_err_pct']:+.1f}%**)" if r['lambda_err_pct'] is not None else ""
            
            mt = f"{r['mu_p_true']:.3f} Pa·s" if r['mu_p_true'] is not None else "-"
            me = f"{r['mu_p_est']:.4f} Pa·s" if r['mu_p_est'] is not None else "-"
            me_err = f"(**{r['mu_p_err_pct']:+.1f}%**)" if r['mu_p_err_pct'] is not None else ""
            
            at = f"{r['alpha_true']:.3f}" if r['alpha_true'] is not None else "-"
            ae = f"{r['alpha_est']:.4f}" if r['alpha_est'] is not None else "-"
            ae_err = f"({r['alpha_err_pct']:+.1f}%)" if r['alpha_err_pct'] is not None else ""
            
            ee = f"{r['eps_est']:.2e}" if r['eps_est'] is not None and r['eps_est'] > 0 else "0.0"
            
            uv = f"**{r['err_L2_uv_pct']:.2f}%**" if r['err_L2_uv_pct'] is not None else "-"
            txy = f"{r['err_L2_tau_xy_pct']:.2f}%" if r['err_L2_tau_xy_pct'] is not None else "-"
            note = r.get("notes", "")
            
            f_md.write(f"| `{ts}` | **{fl}** | **{m}** | {lt} | {le} {le_err} | {mt} | {me} {me_err} | {at} | {ae} {ae_err} | {ee} | {uv} | {txy} | {note} |\n")

    print(f"[OK] Riepilogo Markdown esportato con successo in: {MD_OUT}")


if __name__ == "__main__":
    export_database()
