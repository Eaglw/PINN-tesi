"""
Modulo di visualizzazione per la frontiera di convergenza e i punti del batch.
Include diagrammi a 3 pannelli:
1. lambda vs eta_p (Oldroyd-B / bilancio solvente)
2. lambda vs alpha (Giesekus mobility)
3. lambda vs eps (PTT extensibility / distruzione reticolare)
"""

from pathlib import Path
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from .config import PLOTS_DIR, CONVERGENCE_THRESHOLD_PCT, CONVERGENCE_THRESHOLD_LOG, PARAM_BOUNDS
from .gp_boundary import BoundaryGaussianProcess


def plot_active_learning_doe(
    gp_model: BoundaryGaussianProcess,
    historical_df: pd.DataFrame,
    selected_batch: List[Dict],
    save_path: Path = PLOTS_DIR / "convergence_boundary_doe.png"
):
    """
    Genera un diagramma a 3 pannelli:
    1. Piano (lambda vs eta_p) per Oldroyd-B / modelli lineari (is_giesekus=0, is_ptt=0)
    2. Piano (lambda vs alpha) per Giesekus (is_giesekus=1, is_ptt=0)
    3. Piano (lambda vs eps) per PTT (is_giesekus=0, is_ptt=1)
    Mostrando isolivelli di errore atteso, frontiera critica al 10% e punti del batch.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(22, 6.0), dpi=150)

    # Griglie derivate univocamente da PARAM_BOUNDS (unica fonte di verità)
    lam_min, lam_max = PARAM_BOUNDS["lambda"]
    etap_min, etap_max = PARAM_BOUNDS["eta_p"]
    alpha_min, alpha_max = PARAM_BOUNDS["alpha"]
    eps_min, eps_max = PARAM_BOUNDS["eps"]

    lam_grid = np.linspace(lam_min, lam_max, 140)
    etap_grid = np.linspace(etap_min, etap_max, 90)
    alpha_grid = np.linspace(alpha_min, alpha_max, 90)
    eps_grid = np.linspace(eps_min, eps_max, 90)

    # Filtraggio rigoroso dei punti storici per modello fluido
    is_old = historical_df["fluid_model"].str.lower().str.contains("oldroyd") | ((historical_df["alpha"] == 0) & (historical_df["eps"] == 0))
    is_gie = historical_df["fluid_model"].str.lower().str.contains("giesekus") | (historical_df["alpha"] > 0)
    is_ptt = historical_df["fluid_model"].str.lower().str.contains("ptt") | (historical_df["eps"] > 0)

    old_df = historical_df[is_old]
    gie_df = historical_df[is_gie]
    ptt_df = historical_df[is_ptt]

    levels_err = np.linspace(0, 40, 41)

    # -------------------------------------------------------------
    # Pannello 1: lambda vs eta_p (Oldroyd-B puro: alpha=0, eps=0, is_gie=0, is_ptt=0, mesh=5k)
    # -------------------------------------------------------------
    ax1 = axes[0]
    L_mesh1, P_mesh1 = np.meshgrid(lam_grid, etap_grid)

    pts1 = []
    for l_val, p_val in zip(L_mesh1.ravel(), P_mesh1.ravel()):
        # Feature vector a 7D: [lam, eta_p, alpha, eps, is_giesekus, is_ptt, log10_n]
        pts1.append([l_val, p_val, 0.0, 0.0, 0.0, 0.0, np.log10(5086)])
    pts1 = np.array(pts1)

    mu1, sigma1 = gp_model.predict(pts1)
    err1_pct = 10.0 ** mu1
    Z_err1 = err1_pct.reshape(L_mesh1.shape)

    cp1 = ax1.contourf(L_mesh1, P_mesh1, Z_err1, levels=levels_err, cmap="Spectral_r", extend="max", alpha=0.88)
    cbar1 = fig.colorbar(cp1, ax=ax1)
    cbar1.set_label("Errore Parametrico Atteso (%)", fontsize=10)

    cs1 = ax1.contour(L_mesh1, P_mesh1, Z_err1, levels=[10.0], colors="black", linewidths=2.5, linestyles="--")
    if len(cs1.levels) > 0:
        ax1.clabel(cs1, fmt={10.0: "Soglia 10%"}, inline=True, fontsize=9)

    old_conv = old_df[old_df["converged"] == True]
    old_fail = old_df[old_df["converged"] == False]
    ax1.scatter(old_conv["lambda"], old_conv["eta_p"], c="lime", edgecolors="black", s=65, label="Oldroyd-B: Conv (<10%)", zorder=4)
    ax1.scatter(old_fail["lambda"], old_fail["eta_p"], c="red", edgecolors="black", s=65, marker="s", label="Oldroyd-B: Alto Err (≥10%)", zorder=4)

    # Batch points pertinenti per Oldroyd-B
    for item in selected_batch:
        if "oldroyd" in item.get("fluid_model", "").lower() or (item.get("alpha", 0) == 0 and item.get("eps", 0) == 0):
            ax1.scatter(
                item["lambda"], item["eta_p"],
                c="gold", edgecolors="black", s=200, marker="*",
                zorder=6, label=f"Batch #{item['batch_rank']}"
            )
            ax1.annotate(
                f"#{item['batch_rank']} ({item['mesh']})",
                (item["lambda"], item["eta_p"]),
                textcoords="offset points", xytext=(6, 6),
                fontweight="bold", fontsize=10, color="navy"
            )

    ax1.set_title("Oldroyd-B: $\\lambda$ vs $\\eta_p$ ($\\alpha=0, \\varepsilon=0$)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("$\\lambda$ (Relaxation time)", fontsize=11)
    ax1.set_ylabel("$\\eta_p$ (Polymeric viscosity)", fontsize=11)
    ax1.set_xlim(0.0, lam_max * 1.02)
    ax1.set_ylim(0.0, 1.0)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=8)

    # -------------------------------------------------------------
    # Pannello 2: lambda vs alpha (Giesekus: eta_p=0.5, eps=0, is_gie=1, is_ptt=0, mesh=5k)
    # -------------------------------------------------------------
    ax2 = axes[1]
    L_mesh2, A_mesh2 = np.meshgrid(lam_grid, alpha_grid)

    pts2 = []
    for l_val, a_val in zip(L_mesh2.ravel(), A_mesh2.ravel()):
        # Feature vector a 7D: [lam, eta_p, alpha, eps, is_giesekus, is_ptt, log10_n]
        pts2.append([l_val, 0.5, a_val, 0.0, 1.0, 0.0, np.log10(5086)])
    pts2 = np.array(pts2)

    mu2, sigma2 = gp_model.predict(pts2)
    err2_pct = 10.0 ** mu2
    Z_err2 = err2_pct.reshape(L_mesh2.shape)

    cp2 = ax2.contourf(L_mesh2, A_mesh2, Z_err2, levels=levels_err, cmap="Spectral_r", extend="max", alpha=0.88)
    cbar2 = fig.colorbar(cp2, ax=ax2)
    cbar2.set_label("Errore Parametrico Atteso (%)", fontsize=10)

    cs2 = ax2.contour(L_mesh2, A_mesh2, Z_err2, levels=[10.0], colors="black", linewidths=2.5, linestyles="--")
    if len(cs2.levels) > 0:
        ax2.clabel(cs2, fmt={10.0: "Soglia 10%"}, inline=True, fontsize=9)

    gie_conv = gie_df[gie_df["converged"] == True]
    gie_fail = gie_df[gie_df["converged"] == False]
    ax2.scatter(gie_conv["lambda"], gie_conv["alpha"], c="lime", edgecolors="black", s=65, label="Giesekus: Conv (<10%)", zorder=4)
    ax2.scatter(gie_fail["lambda"], gie_fail["alpha"], c="red", edgecolors="black", s=65, marker="s", label="Giesekus: Alto Err (≥10%)", zorder=4)

    # Batch points pertinenti per Giesekus
    for item in selected_batch:
        if "giesekus" in item.get("fluid_model", "").lower() or item.get("alpha", 0) > 0:
            ax2.scatter(
                item["lambda"], item["alpha"],
                c="gold", edgecolors="black", s=200, marker="*",
                zorder=6, label=f"Batch #{item['batch_rank']}"
            )
            ax2.annotate(
                f"#{item['batch_rank']} ({item['mesh']})",
                (item["lambda"], item["alpha"]),
                textcoords="offset points", xytext=(6, 6),
                fontweight="bold", fontsize=10, color="navy"
            )

    ax2.set_title("Giesekus: $\\lambda$ vs $\\alpha$ ($\\eta_p=0.5, \\varepsilon=0$)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("$\\lambda$ (Relaxation time)", fontsize=11)
    ax2.set_ylabel("$\\alpha$ (Giesekus mobility)", fontsize=11)
    ax2.set_xlim(0.0, lam_max * 1.02)
    ax2.set_ylim(-0.02, alpha_max * 1.04)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=8)

    # -------------------------------------------------------------
    # Pannello 3: lambda vs eps (PTT: eta_p=0.5, alpha=0, is_gie=0, is_ptt=1, mesh=5k)
    # -------------------------------------------------------------
    ax3 = axes[2]
    L_mesh3, E_mesh3 = np.meshgrid(lam_grid, eps_grid)

    pts3 = []
    for l_val, e_val in zip(L_mesh3.ravel(), E_mesh3.ravel()):
        # Feature vector a 7D: [lam, eta_p, alpha, eps, is_giesekus, is_ptt, log10_n]
        pts3.append([l_val, 0.5, 0.0, e_val, 0.0, 1.0, np.log10(5086)])
    pts3 = np.array(pts3)

    mu3, sigma3 = gp_model.predict(pts3)
    err3_pct = 10.0 ** mu3
    Z_err3 = err3_pct.reshape(L_mesh3.shape)

    cp3 = ax3.contourf(L_mesh3, E_mesh3, Z_err3, levels=levels_err, cmap="Spectral_r", extend="max", alpha=0.88)
    cbar3 = fig.colorbar(cp3, ax=ax3)
    cbar3.set_label("Errore Parametrico Atteso (%)", fontsize=10)

    cs3 = ax3.contour(L_mesh3, E_mesh3, Z_err3, levels=[10.0], colors="black", linewidths=2.5, linestyles="--")
    if len(cs3.levels) > 0:
        ax3.clabel(cs3, fmt={10.0: "Soglia 10%"}, inline=True, fontsize=9)

    ptt_conv = ptt_df[ptt_df["converged"] == True]
    ptt_fail = ptt_df[ptt_df["converged"] == False]
    ax3.scatter(ptt_conv["lambda"], ptt_conv["eps"], c="lime", edgecolors="black", s=65, label="PTT: Conv (<10%)", zorder=4)
    ax3.scatter(ptt_fail["lambda"], ptt_fail["eps"], c="red", edgecolors="black", s=65, marker="s", label="PTT: Alto Err (≥10%)", zorder=4)

    # Batch points pertinenti per PTT
    for item in selected_batch:
        if "ptt" in item.get("fluid_model", "").lower() or item.get("eps", 0) > 0:
            ax3.scatter(
                item["lambda"], item["eps"],
                c="gold", edgecolors="black", s=200, marker="*",
                zorder=6, label=f"Batch #{item['batch_rank']}"
            )
            ax3.annotate(
                f"#{item['batch_rank']} ({item['mesh']})",
                (item["lambda"], item["eps"]),
                textcoords="offset points", xytext=(6, 6),
                fontweight="bold", fontsize=10, color="navy"
            )

    ax3.set_title("PTT: $\\lambda$ vs $\\varepsilon$ ($\\eta_p=0.5, \\alpha=0$)", fontsize=12, fontweight="bold")
    ax3.set_xlabel("$\\lambda$ (Relaxation time)", fontsize=11)
    ax3.set_ylabel("$\\varepsilon$ (PTT parameter)", fontsize=11)
    ax3.set_xlim(0.0, lam_max * 1.02)
    ax3.set_ylim(-0.02, eps_max * 1.04)
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return save_path
