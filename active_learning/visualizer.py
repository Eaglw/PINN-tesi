"""
Modulo di visualizzazione per la frontiera di convergenza e i punti del batch.
"""

from pathlib import Path
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from .config import PLOTS_DIR, CONVERGENCE_THRESHOLD_PCT, CONVERGENCE_THRESHOLD_LOG
from .gp_boundary import BoundaryGaussianProcess


def plot_active_learning_doe(
    gp_model: BoundaryGaussianProcess,
    historical_df: pd.DataFrame,
    selected_batch: List[Dict],
    save_path: Path = PLOTS_DIR / "convergence_boundary_doe.png"
):
    """
    Genera un diagramma a 2 pannelli:
    1. Piano (lambda vs eta_p) per Oldroyd-B / modelli lineari
    2. Piano (lambda vs alpha) per Giesekus
    Mostrando isolivelli di errore, la frontiera critica al 10% e i punti del batch suggerito.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

    # -------------------------------------------------------------
    # Pannello 1: lambda vs eta_p (Fissando alpha=0, eps=0, mesh=5k)
    # -------------------------------------------------------------
    ax1 = axes[0]
    lam_grid = np.linspace(0.05, 1.20, 80)
    etap_grid = np.linspace(0.10, 0.95, 80)
    L_mesh, P_mesh = np.meshgrid(lam_grid, etap_grid)

    pts1 = []
    for l_val, p_val in zip(L_mesh.ravel(), P_mesh.ravel()):
        s_val = float(np.round(1.0 - p_val, 4))
        pts1.append([l_val, p_val, s_val, 0.0, 0.0, np.log10(5086)])
    pts1 = np.array(pts1)

    mu1, sigma1 = gp_model.predict(pts1)
    err1_pct = 10.0 ** mu1
    Z_err1 = err1_pct.reshape(L_mesh.shape)

    # Contorno a colori di errore atteso
    cp1 = ax1.contourf(L_mesh, P_mesh, Z_err1, levels=np.linspace(0, 35, 36), cmap="Spectral_r", extend="both", alpha=0.85)
    cbar1 = fig.colorbar(cp1, ax=ax1)
    cbar1.set_label("Errore Parametrico Atteso (%)", fontsize=10)

    # Linea di isolivello al 10% (Frontiera critica)
    cs1 = ax1.contour(L_mesh, P_mesh, Z_err1, levels=[10.0], colors="black", linewidths=2.5, linestyles="--")
    if len(cs1.levels) > 0:
        ax1.clabel(cs1, fmt={10.0: "Soglia 10%"}, inline=True, fontsize=9)

    # Punti storici
    hist_conv = historical_df[historical_df["converged"] == True]
    hist_fail = historical_df[historical_df["converged"] == False]
    ax1.scatter(hist_conv["lambda"], hist_conv["eta_p"], c="lime", edgecolors="black", s=60, label="Storico: Conv (<10%)", zorder=4)
    ax1.scatter(hist_fail["lambda"], hist_fail["eta_p"], c="red", edgecolors="black", s=60, marker="s", label="Storico: Alto Errore", zorder=4)

    # Punti del batch
    for item in selected_batch:
        ax1.scatter(
            item["lambda"], item["eta_p"],
            c="gold", edgecolors="black", s=180, marker="*",
            zorder=5, label=f"Batch #{item['batch_rank']}" if item['batch_rank'] == 1 else None
        )
        ax1.annotate(
            f"#{item['batch_rank']}",
            (item["lambda"], item["eta_p"]),
            textcoords="offset points", xytext=(6, 6),
            fontweight="bold", fontsize=11, color="navy"
        )

    ax1.set_title("Frontiera di Convergenza PINN: $\\lambda$ vs $\\eta_p$", fontsize=12, fontweight="bold")
    ax1.set_xlabel("$\\lambda$ (Relaxation time)", fontsize=11)
    ax1.set_ylabel("$\\eta_p$ (Polymeric viscosity)", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left", fontsize=8)

    # -------------------------------------------------------------
    # Pannello 2: lambda vs alpha (Giesekus con eta_p=0.5, eta_s=0.5, mesh=5k)
    # -------------------------------------------------------------
    ax2 = axes[1]
    alpha_grid = np.linspace(0.00, 0.50, 80)
    L_mesh2, A_mesh2 = np.meshgrid(lam_grid, alpha_grid)

    pts2 = []
    for l_val, a_val in zip(L_mesh2.ravel(), A_mesh2.ravel()):
        pts2.append([l_val, 0.5, 0.5, a_val, 0.0, np.log10(5086)])
    pts2 = np.array(pts2)

    mu2, sigma2 = gp_model.predict(pts2)
    err2_pct = 10.0 ** mu2
    Z_err2 = err2_pct.reshape(L_mesh2.shape)

    cp2 = ax2.contourf(L_mesh2, A_mesh2, Z_err2, levels=np.linspace(0, 35, 36), cmap="Spectral_r", extend="both", alpha=0.85)
    cbar2 = fig.colorbar(cp2, ax=ax2)
    cbar2.set_label("Errore Parametrico Atteso (%)", fontsize=10)

    cs2 = ax2.contour(L_mesh2, A_mesh2, Z_err2, levels=[10.0], colors="black", linewidths=2.5, linestyles="--")
    if len(cs2.levels) > 0:
        ax2.clabel(cs2, fmt={10.0: "Soglia 10%"}, inline=True, fontsize=9)

    ax2.scatter(hist_conv["lambda"], hist_conv["alpha"], c="lime", edgecolors="black", s=60, zorder=4)
    ax2.scatter(hist_fail["lambda"], hist_fail["alpha"], c="red", edgecolors="black", s=60, marker="s", zorder=4)

    for item in selected_batch:
        ax2.scatter(
            item["lambda"], item["alpha"],
            c="gold", edgecolors="black", s=180, marker="*",
            zorder=5
        )
        ax2.annotate(
            f"#{item['batch_rank']}",
            (item["lambda"], item["alpha"]),
            textcoords="offset points", xytext=(6, 6),
            fontweight="bold", fontsize=11, color="navy"
        )

    ax2.set_title("Effetto Non-lineare Giesekus: $\\lambda$ vs $\\alpha$", fontsize=12, fontweight="bold")
    ax2.set_xlabel("$\\lambda$ (Relaxation time)", fontsize=11)
    ax2.set_ylabel("$\\alpha$ (Giesekus mobility)", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return save_path
