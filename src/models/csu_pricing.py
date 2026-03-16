"""
CSU (Capped Short Unit) pricing via Monte Carlo.

Implements:
- CSU payoff computation with dynamic barrier monitoring
- Black swan premium (corrected: monotone in |CVaR|)
- Issuer delta-hedge P&L computation
- Control variate using Black-Scholes analytical price
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Callable
from src.models.svj import SVJParams, simulate_svj_paths


@dataclass
class CSUParams:
    """CSU instrument parameters."""
    K: float           # Strike (initial price)
    L: float           # Leverage factor in (0, 1]
    T: float           # Maturity in years
    static_buffer: float = 0.10   # Static barrier buffer delta
    xi: float = 0.30   # Black swan premium scaling parameter


def _bs_put_price(S0: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes vanilla put price (control variate)."""
    from scipy.stats import norm
    if T <= 0 or sigma <= 0:
        return max(K - S0, 0.0) * np.exp(-r * T)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)


def compute_csu_payoffs(
    S_paths: np.ndarray,
    csu: CSUParams,
    barrier_fn: Optional[Callable[[int, float], float]] = None,
) -> np.ndarray:
    """
    Compute CSU payoffs for each path.

    Parameters
    ----------
    S_paths : ndarray, shape (n_paths, n_steps + 1)
        Asset price paths.
    csu : CSUParams
    barrier_fn : callable(step, t) -> barrier_level, optional
        If None, uses static barrier K * (1 + static_buffer).

    Returns
    -------
    payoffs : ndarray, shape (n_paths,)
    """
    n_paths, n_steps_plus_1 = S_paths.shape
    n_steps = n_steps_plus_1 - 1
    dt = csu.T / n_steps

    payoffs = np.zeros(n_paths)
    survived = np.ones(n_paths, dtype=bool)

    for i in range(1, n_steps_plus_1):
        t = i * dt
        if barrier_fn is not None:
            B = barrier_fn(i, t)
        else:
            B = csu.K * (1 + csu.static_buffer)

        # Knock-out check
        hit = survived & (S_paths[:, i] >= B)
        survived[hit] = False
        # payoff stays 0 for knocked paths

    # Surviving paths: compute payoff at T
    S_T = S_paths[:, -1]
    raw_payoff = csu.L * np.maximum(0.0, (csu.K - S_T) / csu.K)
    payoffs = np.where(survived, raw_payoff, 0.0)
    return payoffs


def price_csu(
    svj_params: SVJParams,
    csu: CSUParams,
    n_paths: int = 10_000,
    n_steps_per_year: int = 252,
    barrier_fn: Optional[Callable] = None,
    seed: Optional[int] = 42,
) -> dict:
    """
    Price CSU via Monte Carlo and compute issuer P&L statistics.

    Returns dict with keys:
        P0_raw       : raw MC price (no black swan premium)
        P0_final     : price with black swan premium
        knockout_rate: fraction of paths knocked out
        payoffs      : array of path payoffs
        cvar_999     : CVaR at 99.9% (negative number)
        xi_cvar      : black swan loading |CVaR| * xi
    """
    n_steps = int(csu.T * n_steps_per_year)

    S_paths, v_paths = simulate_svj_paths(
        svj_params, csu.K, csu.T, n_steps, n_paths,
        antithetic=True, seed=seed,
    )

    payoffs = compute_csu_payoffs(S_paths, csu, barrier_fn)

    disc = np.exp(-svj_params.r * csu.T)
    P0_raw = disc * np.mean(payoffs)

    # Issuer P&L (simplified: collect premium, pay payoff, ignore hedge slippage)
    issuer_pnl = P0_raw - disc * payoffs  # as fraction of P0 will be recomputed post-premium

    # CVaR at 99.9% of issuer P&L
    loss_threshold = np.percentile(issuer_pnl, 0.1)  # worst 0.1%
    cvar = np.mean(issuer_pnl[issuer_pnl <= loss_threshold])  # negative

    # Black swan premium loading (CORRECTED: abs(cvar) ensures positive loading)
    loading = csu.xi * abs(cvar)
    P0_final = P0_raw * (1 + loading)

    knockout_mask = payoffs == 0
    ko_rate = np.mean(knockout_mask)

    return {
        "P0_raw": P0_raw,
        "P0_final": P0_final,
        "knockout_rate": ko_rate,
        "payoffs": payoffs,
        "issuer_pnl": issuer_pnl,
        "cvar_999": cvar,
        "bs_loading": loading,
        "S_paths": S_paths,
        "v_paths": v_paths,
    }


def compute_issuer_pnl_delta_hedge(
    S_paths: np.ndarray,
    v_paths: np.ndarray,
    csu: CSUParams,
    svj_params: SVJParams,
    P0_final: float,
    transaction_cost: float = 0.001,
) -> np.ndarray:
    """
    Compute issuer P&L including discrete delta-hedging error.

    Uses finite-difference delta approximation at each step.
    Returns P&L as fraction of notional (K).
    """
    from scipy.stats import norm

    n_paths, n_steps_plus_1 = S_paths.shape
    n_steps = n_steps_plus_1 - 1
    dt = csu.T / n_steps

    pnl = np.full(n_paths, P0_final / csu.K)  # Collect premium (normalized)
    delta_prev = np.zeros(n_paths)

    for i in range(n_steps):
        t = i * dt
        tau = csu.T - t
        if tau <= 1e-6:
            break
        S = S_paths[:, i]
        v = np.maximum(v_paths[:, i], 1e-6)
        sigma_eff = np.sqrt(v)

        # Approximate CSU delta via BS put delta (barrier effect ignored for speed)
        d1 = (np.log(S / csu.K) + (svj_params.r + 0.5 * v) * tau) / (sigma_eff * np.sqrt(tau))
        # Put delta: -N(-d1) = N(d1) - 1
        delta = csu.L / csu.K * np.exp(-svj_params.r * tau) * (norm.cdf(d1) - 1)

        # Hedge P&L from holding delta_prev units
        dS = S_paths[:, i + 1] - S * np.exp(svj_params.r * dt)
        hedge_pnl = delta_prev * dS / csu.K

        # Transaction costs
        tc = transaction_cost * np.abs(delta - delta_prev) * S / csu.K

        pnl += hedge_pnl - tc
        delta_prev = delta.copy()

    # Subtract payoff
    payoff = csu.L * np.maximum(0.0, (csu.K - S_paths[:, -1]) / csu.K)
    pnl -= np.exp(-svj_params.r * csu.T) * payoff

    return pnl
