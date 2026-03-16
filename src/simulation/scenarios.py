"""
Empirically calibrated SVJ parameters for historical crisis scenarios.

Calibration anchors:
- 2008 Financial Crisis: VIX peaked ~80% (Oct 2008); sustained 50-70% for weeks
- 2010 Flash Crash: VIX spike from ~17% to ~40% over hours, normalized within days
- COVID-19: VIX peaked ~85% (Mar 2020); sustained 60-80% for 2-3 weeks
- Normal: VIX ~15-20%, consistent with 2014-2019 period
"""

from src.models.svj import SVJParams

SCENARIOS: dict[str, SVJParams] = {
    "2008_crisis": SVJParams(
        kappa=3.0,
        theta=0.35,
        sigma=0.85,
        rho=-0.75,
        v0=0.45,      # ~67% annualized vol (VIX sustained ~65-80%)
        lam=2.5,
        mu_j=-0.06,
        sigma_j=0.12,
        r=0.03,
    ),
    "flash_crash": SVJParams(
        kappa=4.5,
        theta=0.18,
        sigma=0.60,
        rho=-0.55,
        v0=0.15,      # ~39% annualized vol
        lam=2.0,
        mu_j=-0.03,
        sigma_j=0.06,
        r=0.03,
    ),
    "covid_19": SVJParams(
        kappa=2.8,
        theta=0.40,
        sigma=1.20,   # High vol-of-vol: Feller violated, full truncation critical
        rho=-0.85,
        v0=0.50,      # ~71% annualized vol
        lam=4.0,
        mu_j=-0.09,
        sigma_j=0.18,
        r=0.01,       # Near-zero rates in 2020
    ),
    "normal": SVJParams(
        kappa=3.0,
        theta=0.04,
        sigma=0.40,
        rho=-0.65,
        v0=0.04,      # ~20% annualized vol
        lam=0.5,
        mu_j=-0.01,
        sigma_j=0.03,
        r=0.03,
    ),
}

SCENARIO_LABELS = {
    "2008_crisis": "2008 Crisis",
    "flash_crash": "2010 Flash Crash",
    "covid_19": "COVID-19",
    "normal": "Normal Market",
}


def get_scenario(name: str) -> SVJParams:
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario '{name}'. Available: {list(SCENARIOS.keys())}")
    return SCENARIOS[name]
