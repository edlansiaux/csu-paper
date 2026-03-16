# Dynamic Capped Short Units — Companion Repository

[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b)](https://arxiv.org/abs/XXXX.XXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

Companion code for:

> **Dynamic Capped Short Units: A Stochastic Volatility Framework for Retail Short Exposure with Bounded Risk**  
> Édouard Lansiaux (STaR-AI Research Group, CHU de Lille / École Centrale Lille)

---

## Overview

This repository reproduces all tables and figures from the manuscript. It implements:

- **SVJ simulation** — Heston stochastic variance + Merton jumps with Full Truncation scheme
- **CSU pricing** — Multi-window barrier Monte Carlo with antithetic variates + control variates
- **Dynamic barrier** — Triage-calibrated knockout threshold
- **Black swan premium** — Corrected monotone formulation (|CVaR|)
- **Issuer P&L** — Discrete delta-hedge computation including transaction costs
- **AI Triage** — MSI + USI + triage decision surface
- **EGARCH-LSTM** — Hybrid volatility forecasting interface

---

## Repository Structure

```
csu-paper/
├── src/
│   ├── models/
│   │   ├── svj.py              # SVJ process simulation (Full Truncation)
│   │   └── csu_pricing.py      # CSU Monte Carlo pricing + issuer P&L
│   ├── simulation/
│   │   └── scenarios.py        # Calibrated crisis scenario parameters
│   ├── triage/
│   │   ├── triage.py           # MSI, USI, triage allocation, dynamic buffer
│   │   └── egarch_lstm.py      # Hybrid EGARCH-LSTM volatility forecaster
│   └── utils/                  # (utilities, to be extended)
├── scripts/
│   ├── run_experiments.py      # Reproduce Tables 2, 3, 4
│   └── generate_figures.py     # Reproduce Figures 1, 2, 3
├── notebooks/
│   └── 01_quickstart.ipynb     # Interactive walkthrough
├── figures/                    # Output directory for manuscript figures
├── requirements.txt
└── README.md
```

---

## Quickstart

### Installation

```bash
git clone https://github.com/edouard-lansiaux/csu-paper
cd csu-paper
pip install -r requirements.txt
```

### Reproduce paper results

```bash
# Reproduce Tables 2, 3, 4 (runtime ~5 min)
python scripts/run_experiments.py

# Generate manuscript figures
python scripts/generate_figures.py
```

### Basic usage

```python
from src.models.svj import SVJParams
from src.models.csu_pricing import CSUParams, price_csu
from src.simulation.scenarios import get_scenario

# Load COVID-19 scenario
svj = get_scenario("covid_19")

# Define CSU instrument
csu = CSUParams(K=100.0, L=1.0, T=30/252, static_buffer=0.10)

# Price via Monte Carlo (10k paths)
result = price_csu(svj, csu, n_paths=10_000, seed=42)
print(f"CSU price:       €{result['P0_raw']:.4f}")
print(f"Price w/ BS prem: €{result['P0_final']:.4f}")
print(f"Knockout rate:   {result['knockout_rate']:.1%}")
print(f"CVaR 99.9%:      {result['cvar_999']:+.4f}")
```

### Dynamic barrier example

```python
from src.triage.triage import compute_dynamic_buffer

def my_dynamic_barrier(step: int, t: float) -> float:
    v_hat = 0.05  # from EGARCH-LSTM forecast
    buf = compute_dynamic_buffer(v_hat, lambda_est=1.0, usi=0.7)
    return 100.0 * (1 + buf)

result_dyn = price_csu(svj, csu, n_paths=10_000, barrier_fn=my_dynamic_barrier)
```

### Triage system

```python
from src.triage.triage import MarketFeatures, InvestorProfile, compute_msi, compute_usi, triage_allocation

market = MarketFeatures(pe_zscore=1.2, momentum_zscore=0.8, vol_ratio=1.5, sentiment_score=0.5)
investor = InvestorProfile(portfolio_return=0.08, portfolio_vol=0.15, var_95=-0.12, behavior_score=0.9)

msi = compute_msi(market)
usi = compute_usi(investor)
alpha = triage_allocation(msi, usi)
print(f"MSI={msi:.2f}, USI={usi:.2f}, Allocation={alpha:.0%}")
```

---

## Key Results

| Scenario | Mean P&L | Std Dev | VaR 99% | Max Loss |
|---|---|---|---|---|
| 2008 Crisis | -0.04% | 0.47% | -1.28% | -2.67% |
| Flash Crash | +0.01% | 0.38% | -0.98% | -1.89% |
| COVID-19 | -0.06% | 0.53% | -1.51% | -2.87% |
| Normal | +0.02% | 0.29% | -0.71% | -1.43% |

---

## Dependencies

See `requirements.txt`. Core:

```
numpy>=1.24
scipy>=1.11
matplotlib>=3.7
pandas>=2.0
```

Optional (for production EGARCH-LSTM):
```
arch>=6.0        # Production EGARCH estimation
torch>=2.0       # PyTorch LSTM training
```

---

## Citation

```bibtex
@misc{lansiaux2025csu,
  title  = {Dynamic Capped Short Units: A Stochastic Volatility Framework
            for Retail Short Exposure with Bounded Risk},
  author = {Lansiaux, \'{E}douard},
  year   = {2025},
  eprint = {XXXX.XXXXX},
  archivePrefix = {arXiv},
}
```

---

## License & Commercial Notice

This repository has a **two-layer license structure**:

| Layer | Files | License |
|---|---|---|
| Academic (reproducibility) | `src/models/`, `src/simulation/`, `src/triage/triage.py`, `src/triage/egarch_lstm.py`, `scripts/` | **MIT** |
| Proprietary stubs | `src/triage/_barrier_calibration.py`, `src/triage/_egarch_weights.py` | **All rights reserved** |

The stubs implement the **reference formulas from the manuscript** and reproduce all
numerical results exactly. The production implementations (trained weights, multi-broker
calibration) are maintained in a private repository.

See [COMMERCIAL_NOTICE.md](COMMERCIAL_NOTICE.md) and [LICENSE](LICENSE) for details.

For commercial licensing: edouard.lansiaux@orange.fr

## Related Work

This work connects to the ZKFL-PQ federated learning architecture:  
> Lansiaux, É. (2025). ZKFL-PQ: Post-quantum zero-knowledge federated learning. [arXiv:2603.03398](https://arxiv.org/abs/2603.03398)
