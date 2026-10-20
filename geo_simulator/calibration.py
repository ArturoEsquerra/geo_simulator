"""Estimate the baseline DGP from a historical panel.

Given real (date, geo, conversions) data, ``calibrate`` estimates the
parameters of the common-factor model in ``BaselineConfig`` and the count
noise in ``NoiseConfig``, so that simulated data looks like *your* data
instead of hand-picked defaults.

Estimation, on ``log(y + 0.5)`` of units with enough volume:

1. Remove unit means; estimate the day-of-week pattern (daily data).
2. Market path = cross-sectional mean. Regress on a linear trend (+ a yearly
   harmonic if the history covers a year) -> ``annual_growth``,
   ``yearly_amplitude``. The random-walk innovation std of what remains is
   estimated with a variance ratio,
   ``(Var(x[t+k] - x[t]) - Var(x[t+1] - x[t])) / (k - 1)``, which removes any
   stationary noise riding on the random walk -> ``market_volatility``.
3. Residual panel (unit minus market): its pooled variogram
   ``D(h) = 2 sigma^2 (1 - rho^h) + 2 omega^2 + C h`` separates a stationary
   AR(1) (``idio_sd``, ``idio_ar``), white noise ``omega^2`` and the linear
   divergence ``C`` caused by common random-walk factors with unequal
   loadings (``factor_volatility``, as one equivalent factor).
4. Each unit's white variance is regressed on ``1 / mean``: in logs,
   ``Var = (1 + overdispersion) / mean + 1 / r`` -> ``overdispersion``
   (grain-consistent) and ``nb_concentration`` (size-independent noise).

Parameters that the data cannot identify keep their configured value and are
flagged in the report (e.g. yearly seasonality with < 1 year of history).
"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pandas as pd

from geo_simulator.config import BaselineConfig, CalibrationConfig, NoiseConfig


@dataclasses.dataclass
class CalibrationResult:
    """Calibrated configs plus a report of what was estimated.

    Attributes:
        baseline: ``BaselineConfig`` with the estimated parameters.
        noise: ``NoiseConfig`` with the estimated concentration, expressed at
            the history's granularity (the simulator converts it).
        granularity: ``"day"`` or ``"week"``, inferred from the history.
        report: One row per parameter: ``value``, ``source``
            (``estimated`` / ``configured`` / ``assumed``) and ``note``.
        diagnostics: Extra statistics (units used, trend dispersion, ...).
    """

    baseline: BaselineConfig
    noise: NoiseConfig
    granularity: str
    report: pd.DataFrame
    diagnostics: dict


def _variance_ratio_rw(x: np.ndarray, k: int) -> float:
    """Per-step random-walk innovation variance of ``x`` (>= 0)."""
    x = np.asarray(x, float)
    if len(x) <= k + 2:
        return 0.0
    v1 = np.var(x[1:] - x[:-1])
    vk = np.var(x[k:] - x[:-k])
    return max((vk - v1) / (k - 1), 0.0)


def _mean_ar1_variance_factor(rho: float, m: int = 7) -> float:
    """Var(mean of m consecutive AR(1) values) / Var(one value)."""
    lags = np.abs(np.subtract.outer(np.arange(m), np.arange(m)))
    return float((rho ** lags).sum() / m**2)


def _autocov(e: np.ndarray, lag: int) -> np.ndarray:
    e = e - e.mean(axis=1, keepdims=True)
    if lag == 0:
        return (e * e).mean(axis=1)
    return (e[:, lag:] * e[:, :-lag]).mean(axis=1)


def _fit_variogram(lags: np.ndarray, variogram: np.ndarray) -> dict:
    """Fit ``D(h) = 2 s2 (1 - rho^h) + 2 w + C h`` by bounded least squares."""
    from scipy.optimize import least_squares

    def model(p, h):
        s2, rho, w, c = p
        return 2 * s2 * (1 - rho**h) + 2 * w + c * h

    v1, vmax = float(variogram[0]), float(variogram.max())
    slope0 = max((variogram[-1] - variogram[len(variogram) // 2]) / max(lags[-1] - lags[len(lags) // 2], 1), 0.0)
    x0 = [max(vmax / 4, 1e-8), 0.5, max(v1 / 4, 1e-8), slope0]
    sol = least_squares(lambda p: (model(p, lags) - variogram) / variogram, x0,
                        bounds=([0, 0, 0, 0], [np.inf, 0.98, np.inf, np.inf]))
    s2, rho, w, c = sol.x
    return {"sigma2": float(s2), "rho": float(rho), "white": float(w), "C": float(c), "fitted": model(sol.x, lags)}


def calibrate(config: CalibrationConfig, base: BaselineConfig = None, noise: NoiseConfig = None) -> CalibrationResult:
    """Estimate ``BaselineConfig`` / ``NoiseConfig`` from a historical panel.

    Args:
        config: The history and its column names.
        base: Values for parameters the data cannot identify (defaults to
            ``BaselineConfig()``).
        noise: Values for the noise config (defaults to ``NoiseConfig()``).
    """
    base = base or BaselineConfig()
    noise = noise or NoiseConfig()
    data = config.data[[config.date_col, config.geo_col, config.value_col]].copy()
    data.columns = ["date", "geo", "y"]
    data["date"] = pd.to_datetime(data["date"])
    wide = data.pivot_table(index="date", columns="geo", values="y", aggfunc="sum").sort_index()

    step = pd.Series(wide.index).diff().dt.days.dropna()
    if step.empty:
        raise ValueError("The history needs at least two dates.")
    period_days = int(step.median())
    if period_days not in (1, 7):
        raise ValueError(f"Could not infer daily or weekly data (median step {period_days} days).")
    granularity = "day" if period_days == 1 else "week"
    full_index = pd.date_range(wide.index[0], wide.index[-1], freq=f"{period_days}D")
    wide = wide.reindex(full_index)
    wide = wide.loc[:, wide.isna().mean() <= 0.1].interpolate(limit_direction="both")

    means = wide.mean()
    used = wide.loc[:, means >= config.min_mean]
    n_t, n_g = used.shape
    min_t = 56 if granularity == "day" else 16
    if n_g < 3 or n_t < min_t:
        raise ValueError(f"Calibration needs >= 3 units averaging >= {config.min_mean} per period and "
                         f">= {min_t} periods; got {n_g} units x {n_t} periods.")

    rows, diag = [], {"units_total": wide.shape[1], "units_used": n_g, "periods": n_t, "granularity": granularity}

    def put(name, value, source, note=""):
        rows.append({"parameter": name, "value": value, "source": source, "note": note})

    logs = np.log(used.to_numpy().T + 0.5)               # units x time
    logs -= logs.mean(axis=1, keepdims=True)
    dates = used.index
    days = np.asarray((dates - dates[0]).days, float)

    # 1. Day of week ----------------------------------------------------------
    weekly_pattern = base.weekly_pattern
    if granularity == "day":
        dow = dates.dayofweek.to_numpy()
        effect = np.array([logs[:, dow == d].mean() for d in range(7)])
        effect -= effect.mean()
        logs = logs - effect[dow][None, :]
        weekly_pattern = tuple(np.round(np.exp(effect), 4))
        put("weekly_pattern", weekly_pattern, "estimated")
    else:
        put("weekly_pattern", weekly_pattern, "configured", "weekly history has no day-of-week information")

    # 2. Market path ----------------------------------------------------------
    market = logs.mean(axis=0)
    years = days / 365.25
    has_year = days[-1] >= 364
    design = [np.ones_like(years), years]
    if has_year:
        doy = 2 * np.pi * dates.dayofyear.to_numpy() / 365.25
        design += [np.sin(doy), np.cos(doy)]
    design = np.column_stack(design)
    coef, *_ = np.linalg.lstsq(design, market, rcond=None)
    annual_growth = float(np.expm1(coef[1]))
    span_years = max(years[-1], 1e-9)
    growth_se = float(np.sqrt(_variance_ratio_rw(market - design @ coef, 2) * n_t) / span_years)
    put("annual_growth", round(annual_growth, 4), "estimated",
        f"weakly identified: +/- {growth_se:.3f} (random-walk drift over {span_years:.1f} years)")
    if has_year:
        yearly_amplitude = float(np.hypot(coef[2], coef[3]))
        put("yearly_amplitude", round(yearly_amplitude, 4), "estimated")
    else:
        yearly_amplitude = base.yearly_amplitude
        put("yearly_amplitude", yearly_amplitude, "configured", "needs >= 1 year of history")
    market_resid = market - design @ coef

    k = min(28 // period_days if granularity == "day" else 8, n_t // 4)
    k = max(k, 2)
    per_day = 1.0 / period_days
    market_volatility = float(np.sqrt(_variance_ratio_rw(market_resid, k) * per_day))
    put("market_volatility", round(market_volatility, 5), "estimated", f"variance ratio, k={k}")

    # 3-4. Unit divergence: a pooled variogram of the residual panel ------
    # After removing the market path, each unit's residual is
    #   (common random-walk factors x loadings) + AR(1) + white noise,
    # whose variogram D(h) = mean (r[t+h] - r[t])^2 is
    #   2 sigma^2 (1 - rho^h) + 2 omega^2 + C h.
    # C (the divergence rate) is what makes units drift apart; it is
    # identified even when the number of factors is not.
    resid = logs - market[None, :]
    resid -= resid.mean(axis=1, keepdims=True)
    horizon = max(4, min(n_t // 3, 60 if granularity == "day" else 26))
    lags = np.arange(1, horizon + 1)
    variogram = np.array([np.mean((resid[:, h:] - resid[:, :-h]) ** 2) for h in lags])
    fit = _fit_variogram(lags, variogram)
    sigma2, rho, white, divergence = fit["sigma2"], fit["rho"], fit["white"], fit["C"]
    diag["variogram"] = pd.DataFrame({"lag": lags, "observed": variogram, "fitted": fit["fitted"]})

    divergence_per_day = divergence * per_day
    loading_sd = base.loading_sd
    if divergence_per_day > 1e-9:
        n_factors, factor_volatility = 1, float(np.sqrt(divergence_per_day) / loading_sd)
        put("n_factors", 1, "estimated", "one equivalent factor; only the total divergence rate is identified")
        put("factor_volatility", round(factor_volatility, 5), "estimated",
            f"divergence {divergence_per_day:.2e}/day, loadings scaled to loading_sd={loading_sd}")
    else:
        n_factors, factor_volatility = 0, base.factor_volatility
        put("n_factors", 0, "estimated", "no divergence between units detected")
        put("factor_volatility", factor_volatility, "configured", "unused with n_factors=0")

    # Count noise: per-unit white variance against 1 / mean. In logs,
    # Var = (1 + overdispersion) / mu + 1 / r, so a non-negative regression
    # of each unit's white variance on [1/mu, 1] gives both terms.
    from scipy.optimize import nnls

    mu = used.mean().to_numpy()
    unit_d1 = np.mean((resid[:, 1:] - resid[:, :-1]) ** 2, axis=1)
    unit_white = unit_d1 / 2 - sigma2 * (1 - rho) - divergence / 2
    (slope, intercept), _ = nnls(np.column_stack([1 / mu, np.ones_like(mu)]), unit_white)
    overdispersion = max(slope - 1.0, 0.0)
    nb_concentration = float(1 / intercept) if intercept > 1e-6 else None
    diag["count_noise_fit"] = {"slope_on_1_over_mu": float(slope), "intercept": float(intercept)}

    if granularity == "week":
        # Convert weekly AR(1) to the daily process whose weekly means match it.
        rho_day = rho ** (1 / 7) if rho > 0 else 0.0
        sigma2 = sigma2 / max(_mean_ar1_variance_factor(rho_day), 1e-9)
        rho = rho_day
    idio_sd, idio_ar = float(np.sqrt(sigma2)), float(rho)
    weak = fit["rho"] >= 0.95 or fit["rho"] <= 0.02
    weak_note = "; WEAKLY IDENTIFIED (AR coefficient at its bound: persistence and divergence not separable)" if weak else ""
    put("idio_sd", round(idio_sd, 4), "estimated", "includes state-level shocks (state not identified in history)" + weak_note)
    put("idio_ar", round(idio_ar, 4), "estimated", "daily AR(1) coefficient" + weak_note)
    diag["weakly_identified"] = weak
    put("state_shock_sd", 0.0, "assumed", "absorbed into idio_sd")

    nb = dataclasses.replace(noise, distribution="negative_binomial", overdispersion=float(overdispersion),
                             nb_concentration=nb_concentration)
    put("overdispersion", round(overdispersion, 3), "estimated", "variance multiple of Poisson; grain-consistent")
    put("nb_concentration", round(nb_concentration, 2) if nb_concentration else None, "estimated",
        f"per {granularity}; size-independent noise" if nb_concentration else "no size-independent noise detected")

    # Diagnostics that are reported but deliberately not applied.
    slopes = np.polyfit(years, resid.T, 1)[0]
    diag["unit_trend_dispersion_per_year"] = float(np.std(slopes))
    put("geo_trend_sd", 0.0, "assumed",
        f"kept at 0 to preserve cointegration; observed residual trend dispersion {np.std(slopes):.3f}/year "
        "(includes random-walk divergence)")

    baseline = dataclasses.replace(
        base, weekly_pattern=weekly_pattern, annual_growth=annual_growth, yearly_amplitude=yearly_amplitude,
        market_volatility=market_volatility, n_factors=n_factors, factor_volatility=factor_volatility,
        idio_sd=idio_sd, idio_ar=min(idio_ar, 0.98), state_shock_sd=0.0, geo_trend_sd=0.0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = pd.DataFrame(rows).set_index("parameter")
    return CalibrationResult(baseline=baseline, noise=nb, granularity=granularity, report=report, diagnostics=diag)


def noise_for_granularity(noise: NoiseConfig, history_granularity: str, output_granularity: str) -> NoiseConfig:
    """Convert estimated count noise between days and weeks.

    ``overdispersion`` is unchanged by aggregation. Size-independent noise
    with relative variance ``v = 1 / r`` per day has about ``v / 7`` per week,
    so ``nb_concentration`` scales by 7.
    """
    if not noise.nb_concentration or history_granularity == output_granularity:
        return noise
    factor = 7.0 if output_granularity == "week" else 1 / 7.0
    return dataclasses.replace(noise, nb_concentration=noise.nb_concentration * factor)
