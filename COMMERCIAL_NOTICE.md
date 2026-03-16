# Commercial Notice

## Academic vs. Commercial Components

This repository is the companion code for:

> Lansiaux, É. (2026). *Dynamic Capped Short Units: A Stochastic Volatility Framework
> for Retail Short Exposure with Bounded Risk.* arXiv:XXXX.XXXXX

It is structured in two layers with different licensing:

---

### Layer 1 — Academic (MIT License)

All mathematical frameworks published in the manuscript are fully open:

| Module | Content | Manuscript ref. |
|---|---|---|
| `src/models/svj.py` | SVJ simulation, Full Truncation scheme | Eq. 1–3, Alg. 1 |
| `src/models/csu_pricing.py` | CSU payoff, black swan premium | Eq. 5–7 |
| `src/simulation/scenarios.py` | Calibrated crisis parameters | Table 1 |
| `src/triage/triage.py` | MSI, USI, triage zones | Eq. 8–9, Table 2 |
| `src/triage/egarch_lstm.py` | EGARCH-LSTM interface + reference stub | Eq. 10–12 |
| `scripts/` | Reproduce Tables 2–4, Figures 1–3 | — |

These files reproduce all numerical results in the manuscript exactly.

---

### Layer 2 — Proprietary (All Rights Reserved)

The following production components are **not distributed** in this repository.
Stubs with the correct interface are provided so that the reproducibility
layer compiles and runs without modification.

| Module | What it contains | Why restricted |
|---|---|---|
| `_barrier_calibration.py` | Production calibration of F(v̂, λ̂, P) | Core IP of ShortSafe CSU product |
| `_egarch_weights.py` | Trained EGARCH-LSTM weights (multi-broker FL) | Trained on proprietary trading data via ZKFL-PQ |
| ShortSafe backend `scoring/engine.py` | PESU PE overvaluation engine | Core IP of ShortSafe PESU product |

The stubs implement the **reference formulas from the paper**, which are sufficient
to reproduce all manuscript results. They do not implement production-grade
calibration.

---

### Licensing Inquiries

For commercial licensing of the production components, contact:

**Dr. Édouard Lansiaux**
ShortSafe / STaR-AI Research Group, CHU de Lille
edouard.lansiaux@univ-lille.fr

---

### What This Means for Reproducibility

The separation is designed so that:

1. **Every table and figure in the manuscript** can be reproduced using the
   MIT-licensed layer alone. Run `python scripts/run_experiments.py`.

2. **Production performance** (live barrier calibration, multi-broker FL
   model accuracy) exceeds the reference stub but is not required for
   academic reproducibility.

3. The **interface contracts** (function signatures, return types) of the
   proprietary stubs are frozen and documented. Third parties can implement
   their own calibration by satisfying these contracts.
