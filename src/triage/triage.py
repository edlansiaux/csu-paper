"""
AI Risk Triage System for CSU position sizing.

Inspired by clinical triage methodology: trades are assessed on Market Severity Index (MSI)
and User Stability Index (USI) before allocation.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class MarketFeatures:
    """Input features for MSI computation."""
    pe_zscore: float          # P/E valuation z-score vs 10yr history
    momentum_zscore: float    # RSI/MACD momentum z-score
    vol_ratio: float          # Implied/realized vol ratio
    sentiment_score: float    # NLP sentiment z-score
    weights: tuple = (0.25, 0.20, 0.35, 0.20)


@dataclass
class InvestorProfile:
    """Investor parameters for USI computation."""
    portfolio_return: float       # Recent portfolio mean return (annualized)
    portfolio_vol: float          # Portfolio volatility
    var_95: float                 # 95% VaR (negative number)
    behavior_score: float = 1.0   # 0=worst, 1=neutral (loss chasing penalty)


def compute_msi(features: MarketFeatures) -> float:
    """
    Market Severity Index in [0, inf).
    MSI = w1*Z_val + w2*Z_mom + w3*Z_vol + w4*Z_sent
    Higher MSI = higher market risk.
    """
    w = features.weights
    components = [
        features.pe_zscore,
        features.momentum_zscore,
        features.vol_ratio - 1.0,   # Positive when IV > RV
        features.sentiment_score,
    ]
    return sum(w[i] * max(c, 0) for i, c in enumerate(components))


def compute_usi(profile: InvestorProfile) -> float:
    """
    User Stability Index in [0, 1].
    Measures investor capacity to bear position loss.
    """
    if profile.portfolio_vol <= 0:
        return 0.0
    score = (
        (profile.portfolio_return - profile.var_95)
        / profile.portfolio_vol
        * profile.behavior_score
    )
    return 1.0 / (1.0 + np.exp(-score))  # Logistic squash


def triage_allocation(msi: float, usi: float) -> float:
    """
    Map (MSI, USI) to allocation factor alpha in {0, 0.25, 0.5, 1.0}.

    Triage zones (extended from paper with Orange zone):
    Green  : MSI < 0.5  and USI > 0.7   -> 1.00
    Yellow : MSI < 1.0  and USI >= 0.4  -> 0.50
    Orange : MSI < 1.5  and USI >= 0.2  -> 0.25
    Red    : otherwise                   -> 0.00
    """
    if msi < 0.5 and usi > 0.7:
        return 1.00
    elif msi < 1.0 and usi >= 0.4:
        return 0.50
    elif msi < 1.5 and usi >= 0.2:
        return 0.25
    else:
        return 0.00


def compute_dynamic_buffer(
    v_forecast: float,
    lambda_estimate: float,
    usi: float,
    buffer_min: float = 0.05,
    buffer_max: float = 0.30,
) -> float:
    """
    Compute dynamic barrier buffer delta_t = F(v_hat, lambda_hat, USI).

    This is the public interface used by the manuscript's Monte Carlo (Eq. 13).
    It delegates to the barrier calibration module:
    - In this distribution: reference formula from the manuscript (open-source).
    - In production (ShortSafe): proprietary multi-broker calibrated mapping
      (see src/triage/_barrier_calibration.py).

    The two implementations share this interface exactly. All manuscript
    results are reproducible with the reference implementation.

    Parameters
    ----------
    v_forecast : float
        Forecasted variance (output of EGARCH-LSTM).
    lambda_estimate : float
        Estimated Poisson jump intensity.
    usi : float
        User Stability Index in [0, 1].
    buffer_min, buffer_max : float
        Clipping bounds (paper: [0.05, 0.30]).

    Returns
    -------
    float
        Barrier buffer delta in [buffer_min, buffer_max].
    """
    from src.triage._barrier_calibration import compute_dynamic_buffer_production
    return compute_dynamic_buffer_production(v_forecast, lambda_estimate, usi,
                                             buffer_min, buffer_max)


def max_position_size(alpha: float, capital_max: float, P0_final: float) -> int:
    """
    Maximum number of CSU units.
    N = floor(alpha * C_max / P0_final)
    """
    if P0_final <= 0:
        return 0
    return int(np.floor(alpha * capital_max / P0_final))
