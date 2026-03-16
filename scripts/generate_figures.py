"""
Generate all figures for the manuscript.

Figures:
    Figure 1 : Issuer P&L distribution — COVID-19 scenario
    Figure 2 : Static vs. dynamic barrier knockout paths (sample)
    Figure 3 : Triage zone decision surface (MSI x USI)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from src.models.csu_pricing import CSUParams, price_csu, compute_issuer_pnl_delta_hedge
from src.simulation.scenarios import SCENARIOS, SCENARIO_LABELS
from src.triage.triage import triage_allocation

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


def figure1_pnl_distribution(n_paths: int = 10_000, seed: int = 42) -> None:
    """Figure 1: Issuer P&L distribution under COVID-19 scenario."""
    K = 100.0
    svj = SCENARIOS["covid_19"]
    csu = CSUParams(K=K, L=1.0, T=30/252, static_buffer=0.10)

    result = price_csu(svj, csu, n_paths, seed=seed)
    hedge_pnl = compute_issuer_pnl_delta_hedge(
        result["S_paths"], result["v_paths"], csu, svj, result["P0_final"]
    )
    pnl_pct = hedge_pnl * 100

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(pnl_pct, bins=80, color="#3A7DCC", edgecolor="none", alpha=0.75, density=True)

    # Annotations
    ax.axvline(np.mean(pnl_pct), color="#E04B3A", lw=1.5, label=f"Mean = {np.mean(pnl_pct):+.2f}%")
    var99 = np.percentile(pnl_pct, 1)
    ax.axvline(var99, color="#F5A623", lw=1.5, ls="--", label=f"VaR$_{{99\%}}$ = {var99:+.2f}%")

    ax.set_xlabel("Issuer P&L (% of Notional)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Issuer P&L Distribution — COVID-19 Scenario\n"
                 f"N = {n_paths:,} paths, 30-day CSU, dynamic barrier",
                 fontsize=12)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig("figures/fig1_pnl_covid.pdf", bbox_inches="tight")
    plt.savefig("figures/fig1_pnl_covid.png", bbox_inches="tight", dpi=150)
    print("Saved figures/fig1_pnl_covid.{pdf,png}")
    plt.close()


def figure2_sample_paths(n_display: int = 20, seed: int = 42) -> None:
    """Figure 2: Sample CSU paths with static vs. dynamic barrier."""
    from src.models.svj import simulate_svj_paths
    from src.triage.triage import compute_dynamic_buffer

    K = 100.0
    svj = SCENARIOS["normal"]
    T = 30 / 252
    n_steps = 30
    n_paths = 200

    S_paths, v_paths = simulate_svj_paths(svj, K, T, n_steps, n_paths, antithetic=False, seed=seed)
    t_grid = np.linspace(0, T * 252, n_steps + 1)

    static_B = K * 1.10

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    for ax, use_dynamic, title in zip(axes,
                                       [False, True],
                                       ["Static Barrier (δ=10%)", "Dynamic Barrier"]):
        for i in range(min(n_display, n_paths)):
            knocked = False
            S = S_paths[i]
            for j in range(1, n_steps + 1):
                t = j / 252
                if use_dynamic:
                    v_hat = max(v_paths[i, j], 0)
                    buf = compute_dynamic_buffer(v_hat, svj.lam, usi=0.6)
                    B = K * (1 + buf)
                else:
                    B = static_B

                if S[j] >= B:
                    knocked = True
                    break

            color = "#E04B3A" if knocked else "#3A7DCC"
            alpha = 0.4
            ax.plot(t_grid[:j+1], S[:j+1], color=color, lw=0.6, alpha=alpha)

        if not use_dynamic:
            ax.axhline(static_B, color="black", lw=1.5, ls="--", label="Barrier B = 110")
        else:
            # Illustrative dynamic barrier band
            v_traj = np.mean(v_paths[:n_display], axis=0)
            B_dyn = [K * (1 + compute_dynamic_buffer(max(v, 0), svj.lam, usi=0.6))
                     for v in v_traj]
            ax.plot(t_grid, B_dyn, color="black", lw=1.5, ls="--", label="Dynamic barrier")

        ax.set_xlabel("Trading days")
        ax.set_ylabel("Asset price" if ax == axes[0] else "")
        ax.set_title(title)
        ax.legend(frameon=False)

        ko_patch = mpatches.Patch(color="#E04B3A", alpha=0.6, label="Knocked out")
        surv_patch = mpatches.Patch(color="#3A7DCC", alpha=0.6, label="Survived")
        ax.legend(handles=[ko_patch, surv_patch], frameon=False)

    plt.suptitle("CSU Path Sample: Static vs. Dynamic Barrier (Normal Market)", y=1.02)
    plt.tight_layout()
    plt.savefig("figures/fig2_paths.pdf", bbox_inches="tight")
    plt.savefig("figures/fig2_paths.png", bbox_inches="tight", dpi=150)
    print("Saved figures/fig2_paths.{pdf,png}")
    plt.close()


def figure3_triage_surface() -> None:
    """Figure 3: Triage decision surface over (MSI, USI) space."""
    msi_grid = np.linspace(0, 2.0, 200)
    usi_grid = np.linspace(0, 1.0, 200)
    MSI, USI = np.meshgrid(msi_grid, usi_grid)

    Z = np.vectorize(triage_allocation)(MSI, USI)

    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = plt.cm.RdYlGn
    c = ax.contourf(MSI, USI, Z, levels=[0, 0.001, 0.26, 0.51, 1.01],
                    colors=["#E04B3A", "#F5A623", "#F8D000", "#4CAF50"], alpha=0.7)

    # Zone labels
    ax.text(1.7, 0.1, "Red\n(α=0)", ha="center", va="center", fontsize=10, color="white", fontweight="bold")
    ax.text(1.2, 0.3, "Orange\n(α=0.25)", ha="center", va="center", fontsize=9, color="white")
    ax.text(0.7, 0.55, "Yellow\n(α=0.50)", ha="center", va="center", fontsize=9)
    ax.text(0.25, 0.85, "Green\n(α=1.0)", ha="center", va="center", fontsize=10, fontweight="bold")

    ax.set_xlabel("Market Severity Index (MSI)", fontsize=12)
    ax.set_ylabel("User Stability Index (USI)", fontsize=12)
    ax.set_title("CSU Risk Triage Decision Surface", fontsize=13)
    plt.tight_layout()
    plt.savefig("figures/fig3_triage.pdf", bbox_inches="tight")
    plt.savefig("figures/fig3_triage.png", bbox_inches="tight", dpi=150)
    print("Saved figures/fig3_triage.{pdf,png}")
    plt.close()


if __name__ == "__main__":
    import os
    os.makedirs("figures", exist_ok=True)
    print("Generating manuscript figures...")
    figure1_pnl_distribution()
    figure2_sample_paths()
    figure3_triage_surface()
    print("All figures saved to figures/")
