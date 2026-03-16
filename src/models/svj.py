"""
SVJ (Stochastic Volatility with Jumps) model simulation.

Implements the Heston + Merton-jump model under the risk-neutral measure Q.
Uses the Full Truncation scheme (Lord et al. 2010) to prevent negative variance.

Reference:
    Lord, R., Koekkoek, R., & Van Dijk, D. (2010). A comparison of biased simulation
    schemes for stochastic volatility models. Quantitative Finance, 10(2), 177-194.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class SVJParams:
    """Parameters for the SVJ model."""
    kappa: float       # Mean reversion speed
    theta: float       # Long-run variance
    sigma: float       # Vol-of-vol
    rho: float         # Spot-vol correlation
    v0: float          # Initial variance
    lam: float         # Jump intensity (Poisson rate)
    mu_j: float        # Mean log-jump size
    sigma_j: float     # Std log-jump size
    r: float = 0.03    # Risk-free rate

    @property
    def mu_bar(self) -> float:
        """Jump compensator: E[e^J] - 1."""
        return np.exp(self.mu_j + 0.5 * self.sigma_j ** 2) - 1

    def feller_satisfied(self) -> bool:
        return 2 * self.kappa * self.theta >= self.sigma ** 2


def simulate_svj_paths(
    params: SVJParams,
    S0: float,
    T: float,
    n_steps: int,
    n_paths: int,
    antithetic: bool = True,
    seed: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate SVJ paths using Full Truncation Euler scheme.

    Parameters
    ----------
    params : SVJParams
    S0 : float
        Initial asset price.
    T : float
        Time horizon in years.
    n_steps : int
        Number of time steps.
    n_paths : int
        Number of Monte Carlo paths. If antithetic=True, actual paths = 2 * n_paths.
    antithetic : bool
        Use antithetic variates for variance reduction.
    seed : int, optional

    Returns
    -------
    S : ndarray, shape (n_paths_total, n_steps + 1)
        Asset price paths.
    v : ndarray, shape (n_paths_total, n_steps + 1)
        Variance paths.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)

    # Cholesky decomposition for correlated Brownians
    # [dW_S, dW_v] ~ N(0, [[1, rho], [rho, 1]])
    rho = params.rho
    rho_perp = np.sqrt(1 - rho ** 2)

    base_paths = n_paths
    total_paths = 2 * n_paths if antithetic else n_paths

    S = np.zeros((total_paths, n_steps + 1))
    v = np.zeros((total_paths, n_steps + 1))
    S[:, 0] = S0
    v[:, 0] = params.v0

    for i in range(n_steps):
        # Standard normals
        z1 = rng.standard_normal(base_paths)
        z2 = rng.standard_normal(base_paths)

        # Correlated Brownians
        eps_S = z1
        eps_v = rho * z1 + rho_perp * z2

        if antithetic:
            eps_S = np.concatenate([eps_S, -eps_S])
            eps_v = np.concatenate([eps_v, -eps_v])

        v_pos = np.maximum(v[:, i], 0.0)  # Full truncation

        # Variance update (CIR step)
        v_next = (
            v[:, i]
            + params.kappa * (params.theta - v_pos) * dt
            + params.sigma * np.sqrt(v_pos * dt) * eps_v
        )
        v[:, i + 1] = v_next  # Store raw (may be negative, truncated at use)

        # Jumps
        n_jumps = rng.poisson(params.lam * dt, total_paths)
        jump_sizes = np.where(
            n_jumps > 0,
            rng.normal(params.mu_j, params.sigma_j, total_paths),
            0.0,
        )

        # Log-Euler for price
        log_S = (
            np.log(S[:, i])
            + (params.r - params.lam * params.mu_bar - 0.5 * v_pos) * dt
            + np.sqrt(v_pos * dt) * eps_S
            + jump_sizes
        )
        S[:, i + 1] = np.exp(log_S)

    return S, v


def simulate_svj_terminal(
    params: SVJParams,
    S0: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Full path simulation returning (S_paths, v_paths) for barrier monitoring.
    Wrapper around simulate_svj_paths.
    """
    return simulate_svj_paths(params, S0, T, n_steps, n_paths, antithetic=True, seed=seed)
