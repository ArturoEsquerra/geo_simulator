"""Diagnostics: do the simulated regions share a common structure?

Synthetic-control-type estimators need the treated unit to be cointegrated
with (a combination of) the controls (Masini & Medeiros, 2021). These helpers
check that on the pre-period of any panel — simulated or real.
"""

from __future__ import annotations

import warnings
from typing import Optional, Sequence

import numpy as np
import pandas as pd


def _wide(panel: pd.DataFrame, unit: str, value: str, date: str = "date") -> pd.DataFrame:
    return panel.pivot_table(index=date, columns=unit, values=value, aggfunc="sum").sort_index()


def cointegration_report(
    panel: pd.DataFrame,
    unit: str = "state",
    value: str = "conversions",
    period: Optional[str] = "pre",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Engle–Granger test of each unit against the sum of all other units.

    For every unit ``g`` it tests whether ``log(y_g)`` and
    ``log(sum_{h != g} y_h)`` are cointegrated, and runs an ADF test on
    ``log(y_g)`` itself (to show whether the series is non-stationary, i.e.
    whether cointegration is even the relevant question).

    Args:
        panel: Long data with ``date``, ``unit`` and ``value`` columns (e.g.
            ``result.aggregate("state")``).
        unit: Unit column.
        value: Outcome column.
        period: If the panel has a ``period`` column, restrict to this value
            (``"pre"`` by default; ``None`` for all rows).
        alpha: Significance level for the ``cointegrated`` flag.

    Returns:
        One row per unit with the ADF p-value of the level, the Engle–Granger
        p-value and a ``cointegrated`` flag.
    """
    from statsmodels.tsa.stattools import adfuller, coint

    if period is not None and "period" in panel:
        panel = panel[panel["period"] == period]
    wide = _wide(panel, unit, value)
    logs = np.log(wide.clip(lower=0) + 1.0)
    total = wide.sum(axis=1)
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for g in wide.columns:
            y = logs[g].to_numpy()
            others = np.log((total - wide[g]).clip(lower=0) + 1.0).to_numpy()
            if np.std(y) == 0:
                rows.append({unit: g, "adf_p_level": np.nan, "eg_p": np.nan})
                continue
            rows.append({
                unit: g,
                "adf_p_level": adfuller(y, autolag="AIC")[1],
                "eg_p": coint(y, others, trend="c", autolag="AIC")[1],
            })
    report = pd.DataFrame(rows)
    report["cointegrated"] = report["eg_p"] < alpha
    return report


def pre_period_fit(
    panel: pd.DataFrame,
    treatment: Sequence[str],
    unit: str = "state",
    value: str = "conversions",
) -> dict:
    """TBR-style fit of the treated aggregate on the control aggregate.

    Fits ``treated_t = a + b * control_t`` by OLS on the pre-period and
    reports R², MAPE and the residual ADF p-value (a stationary residual is
    what makes the counterfactual extrapolation trustworthy).
    """
    from statsmodels.tsa.stattools import adfuller

    if "period" in panel:
        panel = panel[panel["period"] == "pre"]
    wide = _wide(panel, unit, value)
    treatment = [t for t in treatment if t in wide.columns]
    if not treatment:
        raise ValueError("None of the treatment units are in the panel.")
    y = wide[treatment].sum(axis=1).to_numpy(float)
    x = wide.drop(columns=treatment).sum(axis=1).to_numpy(float)
    design = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adf_p = adfuller(resid, autolag="AIC")[1]
    return {
        "r2": float(1 - resid.var() / y.var()),
        "mape": float(np.mean(np.abs(resid) / np.maximum(y, 1e-9))),
        "intercept": float(coef[0]),
        "slope": float(coef[1]),
        "residual_adf_p": float(adf_p),
    }


def sparsity_report(result, by: Optional[str] = None) -> pd.DataFrame:
    """Share of zero-conversion rows per geo — a warning sign for zip-level analysis."""
    data = result.data
    key = by or result.geo_column
    report = data.groupby(key, observed=True)["conversions"].agg(
        mean="mean", zero_share=lambda s: float((s == 0).mean()))
    return report.sort_values("mean")
