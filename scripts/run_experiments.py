"""
Main Monte Carlo experiment reproducing all tables in the paper.

Tables reproduced:
    Table 1 : Crisis scenario parameters (printed, not computed)
    Table 2 : CSU prices under alternative models
    Table 3 : Issuer P&L statistics
    Table 4 : Static vs. Dynamic barrier comparison
    Table 5 : Federated learning performance (printed, not computed here)

Usage:
    python scripts/run_experiments.py

Runtime: ~5 minutes on a modern laptop (10k paths x 4 scenarios).
"""

import numpy as np
import time
from src.models.svj import SVJParams
from src.models.csu_pricing import CSUParams, price_csu, compute_issuer_pnl_delta_hedge
from src.simulation.scenarios import SCENARIOS, SCENARIO_LABELS
from src.triage.triage import compute_dynamic_buffer


def run_table2(seed: int = 42) -> None:
    """Table 2: CSU prices under alternative models."""
    print("\n" + "="*65)
    print("TABLE 2: CSU Prices Under Alternative Models")
    print("L=1, T=30 days, K=100, r=3%")
    print("="*65)

    normal = SCENARIOS["normal"]
    K = 100.0
    buffers = [0.05, 0.10, 0.15, 0.20]
    n_paths = 5_000
    n_steps = 30  # daily

    header = f"{'Buffer':>8} | {'BS':>6} | {'Heston':>8} | {'SVJ':>6} | {'SVJ+Dyn':>8}"
    print(header)
    print("-" * 55)

    for delta in buffers:
        csu = CSUParams(K=K, L=1.0, T=30/252, static_buffer=delta)

        # BS price (analytical for vanilla put, approximate for barrier)
        sigma_bs = np.sqrt(normal.theta)
        from scipy.stats import norm as scipy_norm
        T = 30/252
        d1 = (np.log(K / K) + (normal.r + 0.5 * sigma_bs**2) * T) / (sigma_bs * np.sqrt(T))
        d2 = d1 - sigma_bs * np.sqrt(T)
        bs_put = K * np.exp(-normal.r * T) * scipy_norm.cdf(-d2) - K * scipy_norm.cdf(-d1)
        # Approximate barrier adjustment
        bs_price = bs_put * (1 - delta * 0.5)

        # Heston (SVJ with lam=0)
        heston_params = SVJParams(
            kappa=normal.kappa, theta=normal.theta, sigma=normal.sigma,
            rho=normal.rho, v0=normal.v0, lam=0.0, mu_j=0.0, sigma_j=0.0, r=normal.r
        )
        heston_csu = CSUParams(K=K, L=1.0, T=T, static_buffer=delta)
        h_result = price_csu(heston_params, heston_csu, n_paths, seed=seed)

        # SVJ static
        svj_result = price_csu(normal, csu, n_paths, seed=seed)

        # SVJ dynamic
        def dyn_barrier(step, t):
            v_hat = normal.v0 + (normal.theta - normal.v0) * (1 - np.exp(-normal.kappa * t))
            buf = compute_dynamic_buffer(v_hat, normal.lam, usi=0.6)
            return K * (1 + buf)

        csu_dyn = CSUParams(K=K, L=1.0, T=T, static_buffer=delta)
        dyn_result = price_csu(normal, csu_dyn, n_paths, seed=seed, barrier_fn=dyn_barrier)

        print(f"{delta*100:6.0f}%  | {bs_price:6.2f} | {h_result['P0_raw']:8.2f} | "
              f"{svj_result['P0_raw']:6.2f} | {dyn_result['P0_raw']:8.2f}")


def run_table3(n_paths: int = 10_000, seed: int = 42) -> None:
    """Table 3: Issuer P&L statistics across scenarios."""
    print("\n" + "="*65)
    print("TABLE 3: Issuer P&L Statistics (% of Notional)")
    print(f"N = {n_paths:,} paths per scenario")
    print("="*65)

    K = 100.0
    csu = CSUParams(K=K, L=1.0, T=30/252, static_buffer=0.10)

    header = f"{'Scenario':<18} | {'Mean':>7} | {'Std':>7} | {'VaR99%':>8} | {'MaxLoss':>8}"
    print(header)
    print("-" * 65)

    results = {}
    for key, label in SCENARIO_LABELS.items():
        svj = SCENARIOS[key]
        res = price_csu(svj, csu, n_paths, seed=seed)

        # Delta-hedge P&L
        hedge_pnl = compute_issuer_pnl_delta_hedge(
            res["S_paths"], res["v_paths"], csu, svj, res["P0_final"]
        )
        pnl_pct = hedge_pnl * 100  # as % of K

        mean_pnl = np.mean(pnl_pct)
        std_pnl = np.std(pnl_pct)
        var99 = np.percentile(pnl_pct, 1)
        max_loss = np.min(pnl_pct)

        results[key] = {"mean": mean_pnl, "std": std_pnl, "var99": var99, "max_loss": max_loss}
        print(f"{label:<18} | {mean_pnl:+7.2f}% | {std_pnl:7.2f}% | "
              f"{var99:+8.2f}% | {max_loss:+8.2f}%")

    return results


def run_table4(n_paths: int = 10_000, seed: int = 42) -> None:
    """Table 4: Static vs. Dynamic barrier comparison."""
    print("\n" + "="*65)
    print("TABLE 4: Static vs. Dynamic Barrier Comparison (Normal Market)")
    print("="*65)

    K = 100.0
    svj = SCENARIOS["normal"]

    # Static
    csu_static = CSUParams(K=K, L=1.0, T=30/252, static_buffer=0.10)
    r_static = price_csu(svj, csu_static, n_paths, seed=seed)

    # Dynamic
    def dynamic_barrier(step, t):
        v_hat = svj.v0 + (svj.theta - svj.v0) * (1 - np.exp(-svj.kappa * t))
        buf = compute_dynamic_buffer(v_hat, svj.lam, usi=0.6)
        return K * (1 + buf)

    csu_dyn = CSUParams(K=K, L=1.0, T=30/252, static_buffer=0.10)
    r_dyn = price_csu(svj, csu_dyn, n_paths, seed=seed, barrier_fn=dynamic_barrier)

    def ko_stats(result):
        ko_rate = result["knockout_rate"]
        payoffs = result["payoffs"]
        # Time to knockout approximation: not tracked in current implementation
        # Buyer win rate = fraction with positive payoff (and survived)
        win_rate = np.mean(payoffs > 0)
        return ko_rate, win_rate, result["issuer_pnl"].std() * 100

    ko_s, win_s, std_s = ko_stats(r_static)
    ko_d, win_d, std_d = ko_stats(r_dyn)

    rows = [
        ("Knockout probability", f"{ko_s:.1%}", f"{ko_d:.1%}",
         f"{(ko_d-ko_s)/ko_s:+.1%}"),
        ("Buyer win rate", f"{win_s:.1%}", f"{win_d:.1%}",
         f"{(win_d-win_s)/win_s:+.1%}"),
        ("Issuer P&L std dev", f"{std_s:.2f}%", f"{std_d:.2f}%",
         f"{(std_d-std_s)/std_s:+.1%}"),
    ]

    print(f"{'Metric':<30} | {'Static':>10} | {'Dynamic':>10} | {'Change':>8}")
    print("-" * 65)
    for metric, s, d, chg in rows:
        print(f"{metric:<30} | {s:>10} | {d:>10} | {chg:>8}")


if __name__ == "__main__":
    print("ShortSafe — CSU Monte Carlo Experiments")
    print("Reproducing Tables 2, 3, 4 from manuscript")
    t0 = time.time()

    run_table2(seed=42)
    run_table3(n_paths=10_000, seed=42)
    run_table4(n_paths=10_000, seed=42)

    print(f"\nTotal runtime: {time.time() - t0:.1f}s")
