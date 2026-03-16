"""
ShortSafe — EGARCH-LSTM Trained Weights Module

PROPRIETARY — ALL RIGHTS RESERVED
Copyright (c) 2026 Édouard Lansiaux / ShortSafe

This module is a DOCUMENTED STUB. The production weights are maintained
in a private model registry and are NOT distributed.

What the production weights contain
-------------------------------------
- EGARCH(1,1) parameters fitted on 15+ European equity underlyings
  (daily returns 2010–2025, updated monthly via online EM)
- LSTM weights (hidden_size=128, 2 layers, 4-head attention) trained on
  a multi-broker dataset aggregated via the ZKFL-PQ federated protocol
  (arXiv:2603.03398). Training data is never centralized.
- Calibrated blend coefficient alpha (LSTM vs EGARCH mixture weight)
  per volatility regime (low/medium/high/crisis)

Why not distributed
--------------------
The trained weights encode statistical patterns derived from proprietary
broker execution data shared under the ZKFL-PQ privacy guarantee. 
Distributing them would (i) violate the data-sharing agreements with broker
partners, and (ii) constitute the core commercial IP of the ShortSafe
risk triage product.

Reproducibility
---------------
The manuscript's Monte Carlo results do NOT depend on the trained weights.
The paper uses the reference EGARCH stub (see egarch_lstm.py), which is
fully open-source. All tables and figures reproduce correctly without
these weights.

For commercial licensing: edouard.lansiaux@univ-lille.fr
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class _ProprietaryModelWeights:
    """
    Placeholder for production EGARCH-LSTM weights.

    In production, fields are populated from a secure model registry
    (encrypted at rest, access-controlled per environment).
    """

    # EGARCH parameters (per underlying — production has 15+ assets)
    egarch_omega: float = -0.10     # Reference value from manuscript
    egarch_beta: float = 0.90
    egarch_gamma: float = 0.10
    egarch_alpha: float = -0.05
    egarch_initial_sigma2: float = 0.04

    # LSTM architecture (production: hidden=128, layers=2, heads=4)
    # Weights not distributed — only architecture documented
    lstm_hidden_size: int = 128
    lstm_n_layers: int = 2
    lstm_n_heads: int = 4

    # Blend weights per regime
    # Production: alpha_low=0.15, alpha_med=0.30, alpha_high=0.50, alpha_crisis=0.60
    # Reference stub: fixed blend
    blend_reference: float = 0.30

    _production_loaded: bool = False


def load_production_weights(asset_ticker: str = "generic") -> _ProprietaryModelWeights:
    """
    Load production EGARCH-LSTM weights for a given asset.

    In production, retrieves from encrypted model registry.
    This stub returns reference parameters sufficient for manuscript reproducibility.

    Parameters
    ----------
    asset_ticker : str
        Asset identifier. Production supports 15+ European equities.
        Stub ignores this and returns reference parameters.

    Returns
    -------
    _ProprietaryModelWeights
    """
    # Production path (not distributed):
    # return _registry.get(asset_ticker)
    return _ProprietaryModelWeights()


def is_production_available() -> bool:
    """Returns True if production weights are loaded (False in this stub)."""
    return False
