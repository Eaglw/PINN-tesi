import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
import numpy as np

# Aggiungi cartella final_roll al path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

def run_zero_step_check(eta_values=[0.5, 1.0, 2.0, 5.0, 10.0]):
    """
    Esegue una verifica analitica istantanea a Epoca 0 per verificare che le loss e i gradienti
    rispetto ai parametri fisici dimensionali siano rigorosamente identici (invarianti) per qualsiasi eta_0.
    """
    import torch
    import torch.nn as nn
    from src.utils import load_data
    from src.physics import Physics
    from src.train import CombinedModel, initialize_last_layer_zero, init_weights_xavier
    import src.debug
    import src.physics
    import src.train
    import src.utils

    print("\n" + "=" * 80)
    print("VERIFICA ANALITICA ISTANTANEA A STEP 0 (Invarianza di Scala rispetto a eta_0)")
    print("=" * 80)
    print(f"Valori di eta_0 in test: {eta_values}\n")

    results = []

    for eta_0 in eta_values:
        # Inietta variabili globali necessarie per src
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        globals_dict = {
            "DEVICE": device,
            "DATASET_PATH": BASE_DIR.parent / "COMSOL" / "4roll" / "4_roll_mill.csv",
            "ETA_0": float(eta_0),
            "MU_S_TRUE": 0.1,
            "MU_P_TRUE": 0.9,
            "MU_TOT_TRUE": 1.0,
            "LAM_TRUE": 0.05,
            "EPS_TRUE": 0.0,
            "ALPHA_TRUE": 0.0,
            "RHO": 1000.0,
            "GUESS_FACTOR": 0.8,
            "GUESS_LAM": 0.04,
            "GUESS_MU_S": 0.08,
            "GUESS_MU_P": 0.72,
            "HIDDEN_LAYERS": [128] * 8,
            "ACTIVATION": nn.SiLU,
            "W_DATA_1": 1.0,
            "W_BC_1": 5.0,
            "W_CONSTITUTIVE": 1.0,
            "W_MOMENTUM": 0.0,
            "VARIANCE_EPS": 1e-4,
            "USE_ROLL_STRESS_BC": True,
            "W_ROLL_STRESS": 1.0,
        }
        for mod in [src.debug, src.physics, src.train, src.utils]:
            for k, v in globals_dict.items():
                mod.__dict__[k] = v

        # Fissa lo stesso identico seed
        torch.manual_seed(123)
        np.random.seed(123)

        data = load_data(filepath=globals_dict["DATASET_PATH"], eta_0=eta_0)
        model = CombinedModel(p_scale=data["p_scale"], tau_scale=data["tau_scale"]).to(device)
        for sm in [model.model_psi, model.model_p, model.model_tau]:
            sm.apply(lambda m: init_weights_xavier(m, activation_name=nn.SiLU))
        initialize_last_layer_zero(model.model_p)
        initialize_last_layer_zero(model.model_tau)

        physics = Physics(
            U_ref=data["U_ref"],
            H_ref=data["H"],
            H_coord=data["H_coord"],
            var_weights=data["var_weights"],
            inverse_mode=True,
            tau_scale=data["tau_scale"],
            p_scale=data["p_scale"],
            eta_0=eta_0,
        ).to(device)

        # Calcola loss a step 0
        points = data["coords"][:2000]
        labels = data["uv_data"][:2000]
        xph = points.clone().requires_grad_(True)
        u, v, p, tau = physics.get_velocity(model, xph)

        loss_data = physics.data_loss(u, v, labels, data["var_weights"])
        loss_bc = physics.boundary_loss(model, data["boundary_groups"], data["var_weights"], active_bcs=["u", "v", "tau_xx", "tau_xy", "tau_yy"])
        lm, lc = physics.compute_pde_losses(xph, u, v, p, tau, w_momentum=0.0, w_constitutive=1.0)
        tot_loss = 1.0 * loss_data + 5.0 * loss_bc + 1.0 * lc

        # Backward per estrarre i gradienti
        model.zero_grad()
        physics.zero_grad()
        tot_loss.backward()

        grad_mup = physics._raw_mu_p.grad.item() if physics._raw_mu_p.grad is not None else 0.0
        grad_lam = physics._raw_lam.grad.item() if physics._raw_lam.grad is not None else 0.0

        results.append({
            "eta_0": eta_0,
            "tau_scale": float(data["tau_scale"]),
            "data_loss": loss_data.item(),
            "bc_loss": loss_bc.item(),
            "loss_const": lc.item(),
            "tot_loss": tot_loss.item(),
            "grad_raw_mup": grad_mup,
            "grad_raw_lam": grad_lam,
        })

    # Stampa tabella
    headers = ["eta_0", "tau_scale", "Data Loss", "BC Loss", "Const Loss", "Tot Loss", "dLoss/d(raw_mup)", "dLoss/d(raw_lam)"]
    print(f"{headers[0]:<7} | {headers[1]:<10} | {headers[2]:<11} | {headers[3]:<11} | {headers[4]:<11} | {headers[5]:<11} | {headers[6]:<18} | {headers[7]:<18}")
    print("-" * 115)
    base = results[0]
    for r in results:
        print(f"{r['eta_0']:<7.2f} | {r['tau_scale']:<10.4f} | {r['data_loss']:<11.4e} | {r['bc_loss']:<11.4e} | {r['loss_const']:<11.4e} | {r['tot_loss']:<11.4e} | {r['grad_raw_mup']:<18.6e} | {r['grad_raw_lam']:<18.6e}")

    # Calcola massima deviazione relativa rispetto a eta_0 = 2.0 (o primo elemento)
    idx_ref = [i for i, r in enumerate(results) if abs(r["eta_0"] - 2.0) < 1e-4]
    ref = results[idx_ref[0]] if idx_ref else results[0]
    print("-" * 115)
    max_dev_loss = max(abs(r["tot_loss"] - ref["tot_loss"]) / ref["tot_loss"] for r in results)
    max_dev_mup = max(abs(r["grad_raw_mup"] - ref["grad_raw_mup"]) / (abs(ref["grad_raw_mup"]) + 1e-12) for r in results)
    max_dev_lam = max(abs(r["grad_raw_lam"] - ref["grad_raw_lam"]) / (abs(ref["grad_raw_lam"]) + 1e-12) for r in results)

    print(f"Deviazione Relativa Massima vs Riferimento (eta_0={ref['eta_0']}):")
    print(f"  - Total Loss  : {max_dev_loss * 100:.4f}%")
    print(f"  - Grad raw_mup: {max_dev_mup * 100:.4f}%")
    print(f"  - Grad raw_lam: {max_dev_lam * 100:.4f}%")
    if max_dev_loss < 0.05 and max_dev_mup < 0.05:
        print("\n>>> ESITO CHECK: PERFETTA INVARIANZA DI SCALA DIMOSTRATA NUMERICAMENTE! <<<")
    else:
        print("\n>>> ATTENZIONE: Si riscontra discrepanza nei gradienti. <<<")
    print("=" * 80 + "\n")
    return results

def run_suite_experiments(eta_values, epochs_ph1=10000, lbfgs_ph1=0, no_lbfgs=True, seed=123, tag=""):
    """
    Esegue in sequenza l'addestramento di Fase 1 tramite train_4roll_suite.py per ogni valore di eta_0.
    """
    python_exe = sys.executable
    script_suite = BASE_DIR / "train_4roll_suite.py"

    run_dirs = []

    print("\n" + "=" * 80)
    print(f"AVVIO SUITE DI TRAINING FASE 1: {len(eta_values)} RUNS")
    print(f"  - eta_0 values : {eta_values}")
    print(f"  - Epoche Adam  : {epochs_ph1}")
    print(f"  - L-BFGS iters : {0 if no_lbfgs else lbfgs_ph1}")
    print(f"  - Seed         : {seed}")
    print("=" * 80)

    for i, eta_0 in enumerate(eta_values, 1):
        print(f"\n[{i}/{len(eta_values)}] LANCIO RUN CON eta_0 = {eta_0} ...")
        cmd = [
            python_exe,
            str(script_suite),
            f"--eta0={eta_0}",
            f"--epochs-ph1={epochs_ph1}",
            f"--seed={seed}",
            "--no-tb"
        ]
        if no_lbfgs or lbfgs_ph1 == 0:
            cmd.append("--no-lbfgs")
        else:
            cmd.append(f"--lbfgs-ph1={lbfgs_ph1}")
        if tag:
            cmd.append(f"--tag={tag}")

        ret = subprocess.run(cmd, cwd=str(BASE_DIR))
        if ret.returncode != 0:
            print(f"[ERRORE] Il run per eta_0={eta_0} è terminato con codice di errore {ret.returncode}!")
        else:
            print(f"[OK] Run per eta_0={eta_0} terminato con successo.")

def generate_comparison_plots(suite_dir=None, output_plot_path=None):
    """
    Legge tutti i file suite_summary.json generati nella suite e produce grafici comparativi sovrapposti.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if suite_dir is None:
        suite_dir = BASE_DIR / "output_4rollmill" / "suite_eta0"

    summary_files = list(Path(suite_dir).rglob("suite_summary.json"))
    if not summary_files:
        print(f"[AVVISO] Nessun file suite_summary.json trovato in {suite_dir}")
        return

    # Ordina i summary per valore di eta_0
    summaries = []
    for sf in summary_files:
        try:
            with open(sf, "r", encoding="utf-8") as f:
                data = json.load(f)
                summaries.append((data["eta_0"], data, sf.parent))
        except Exception as e:
            print(f"Errore lettura {sf}: {e}")

    summaries.sort(key=lambda x: x[0])

    print(f"\nCaricati {len(summaries)} run per la comparazione:")
    for eta, s, p in summaries:
        print(f"  - eta_0={eta:5.2f} | mu_p={s['mu_p_final']:.4f} Pa·s (err: {s['mu_p_rel_err']*100:.2f}%) | lam={s['lam_final']:.4f} s (err: {s['lam_rel_err']*100:.2f}%)")

    # Creazione Plot Multi-Pannello
    fig, axs = plt.subplots(2, 2, figsize=(15, 11))
    
    # Palette colori
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(summaries))]

    # 1. Parametro mu_p
    ax = axs[0, 0]
    mu_p_true = summaries[0][1].get("mu_p_true", 0.90)
    for i, (eta, s, _) in enumerate(summaries):
        epochs = s.get("epochs_history", [])
        mup_hist = s.get("loss_history", {}).get("param_mu_p", [])
        valid = [(e, v) for e, v in zip(epochs, mup_hist) if v is not None]
        if valid:
            ep, vv = zip(*valid)
            ax.plot(ep, vv, label=rf"$\eta_0 = {eta:.2f}$ (final: {vv[-1]:.4f})", color=colors[i], linewidth=2.0, alpha=0.85)
    ax.axhline(mu_p_true, color="black", linestyle="--", linewidth=2.0, label=f"True ({mu_p_true:.2f} Pa·s)")
    ax.set_title(r"Evoluzione Viscosità Polimerica $\mu_p$ [Pa·s]", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoca / Iter")
    ax.set_ylabel(r"$\mu_p$ [Pa·s]")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=9)

    # 2. Parametro lambda
    ax = axs[0, 1]
    lam_true = summaries[0][1].get("lam_true", 0.05)
    for i, (eta, s, _) in enumerate(summaries):
        epochs = s.get("epochs_history", [])
        lam_hist = s.get("loss_history", {}).get("param_lam", [])
        valid = [(e, v) for e, v in zip(epochs, lam_hist) if v is not None]
        if valid:
            ep, vv = zip(*valid)
            ax.plot(ep, vv, label=rf"$\eta_0 = {eta:.2f}$ (final: {vv[-1]:.4f})", color=colors[i], linewidth=2.0, alpha=0.85)
    ax.axhline(lam_true, color="black", linestyle="--", linewidth=2.0, label=f"True ({lam_true:.3f} s)")
    ax.set_title(r"Evoluzione Tempo di Rilassamento $\lambda$ [s]", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoca / Iter")
    ax.set_ylabel(r"$\lambda$ [s]")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=9)

    # 3. Loss Costitutiva
    ax = axs[1, 0]
    for i, (eta, s, _) in enumerate(summaries):
        epochs = s.get("epochs_history", [])
        c_hist = s.get("loss_history", {}).get("loss_constitutive", [])
        valid = [(e, v) for e, v in zip(epochs, c_hist) if v is not None and v > 0]
        if valid:
            ep, vv = zip(*valid)
            ax.plot(ep, vv, label=rf"$\eta_0 = {eta:.2f}$", color=colors[i], linewidth=1.5, alpha=0.85)
    ax.set_yscale("log")
    ax.set_title("Loss Costitutiva PDE (Reologia)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoca / Iter")
    ax.set_ylabel("Loss Costitutiva")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=9)

    # 4. Total Loss
    ax = axs[1, 1]
    for i, (eta, s, _) in enumerate(summaries):
        epochs = s.get("epochs_history", [])
        tot_hist = s.get("loss_history", {}).get("total", [])
        valid = [(e, v) for e, v in zip(epochs, tot_hist) if v is not None and v > 0]
        if valid:
            ep, vv = zip(*valid)
            ax.plot(ep, vv, label=rf"$\eta_0 = {eta:.2f}$", color=colors[i], linewidth=1.5, alpha=0.85)
    ax.set_yscale("log")
    ax.set_title("Loss Totale Fase 1", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoca / Iter")
    ax.set_ylabel("Loss Totale")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=9)

    plt.suptitle(r"Suite Invarianza di Scala: Confronto Convergenza Fase 1 vs $\eta_0$", fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if output_plot_path is None:
        output_plot_path = suite_dir / "comparison_eta0_convergence.png"
    plt.savefig(output_plot_path, dpi=180)
    plt.close()
    print(f"\n[OK] Grafico comparativo salvato in: {output_plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orchestratore Suite di Invarianza eta_0")
    parser.add_argument("--check-only", action="store_true", help="Esegue solo il check analitico istantaneo a Step 0")
    parser.add_argument("--plot-only", action="store_true", help="Genera solo i grafici comparativi leggendo le run esistenti")
    parser.add_argument("--eta-list", type=float, nargs="+", default=[0.5, 1.0, 2.0, 5.0], help="Lista di valori eta_0 da testare")
    parser.add_argument("--epochs", type=int, default=10000, help="Numero di epoche Adam per ciascun run")
    parser.add_argument("--lbfgs-ph1", type=int, default=0, help="Iterazioni L-BFGS Fase 1 (0 = disattivato)")
    parser.add_argument("--tag", type=str, default="suite", help="Tag di identificazione")
    args = parser.parse_args()

    if args.check_only:
        run_zero_step_check(eta_values=args.eta_list)
    elif args.plot_only:
        generate_comparison_plots()
    else:
        # 1. Esegui prima il check step 0 per verifica immediata
        run_zero_step_check(eta_values=args.eta_list)

        # 2. Esegui la suite di training
        no_lbfgs_flag = (args.lbfgs_ph1 == 0)
        run_suite_experiments(
            eta_values=args.eta_list,
            epochs_ph1=args.epochs,
            lbfgs_ph1=args.lbfgs_ph1,
            no_lbfgs=no_lbfgs_flag,
            tag=args.tag
        )

        # 3. Genera il plot finale di confronto
        generate_comparison_plots()
