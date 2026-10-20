"""Counterfactual latent intensity: a common-factor model in log space.

See ``BaselineConfig`` for the equation. Everything here is at the level of
the simulated unit (zip, cluster, state or city) and at *daily* resolution;
the simulator aggregates to weeks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from geo_simulator.config import BaselineConfig


def ar1(rng: np.random.Generator, n_series: int, n_steps: int, sd: float, rho: float) -> np.ndarray:
    """Stationary AR(1) paths, shape ``(n_series, n_steps)``, marginal std ``sd``."""
    out = np.empty((n_series, n_steps))
    if n_steps == 0:
        return out
    innovation_sd = sd * np.sqrt(1.0 - rho**2)
    out[:, 0] = rng.normal(0.0, sd, n_series)
    shocks = rng.normal(0.0, innovation_sd, (n_series, n_steps))
    for t in range(1, n_steps):
        out[:, t] = rho * out[:, t - 1] + shocks[:, t]
    return out


def random_walk(rng: np.random.Generator, n_steps: int, sd: float) -> np.ndarray:
    """Driftless Gaussian random walk starting at 0 (an I(1) process)."""
    steps = rng.normal(0.0, sd, n_steps)
    steps[0] = 0.0
    return np.cumsum(steps)


def simulate_latent(
    rng: np.random.Generator,
    shares: np.ndarray,
    state_idx: np.ndarray,
    dates: pd.DatetimeIndex,
    config: BaselineConfig,
) -> tuple[np.ndarray, dict]:
    """Daily latent log-intensity per unit, up to a common scale.

    Args:
        rng: Dedicated generator for the baseline.
        shares: Population share of each unit (sums to 1).
        state_idx: State index of each unit (for the shared state shock).
        dates: Daily dates.
        config: Baseline configuration.

    Returns:
        ``(log_mu, components)`` where ``log_mu`` has shape
        ``(n_units, n_days)`` and ``components`` holds the shared and
        unit-specific pieces for inspection.
    """
    n_regions, n_days = len(shares), len(dates)
    years = np.arange(n_days) / 365.25

    # Region-level, time-invariant draws.
    propensity = rng.normal(0.0, config.propensity_sd, n_regions) - config.propensity_sd**2 / 2
    loadings = rng.normal(0.0, config.loading_sd, (n_regions, config.n_factors))
    seasonal_loading = rng.normal(1.0, config.seasonality_loading_sd, n_regions)
    geo_trend = rng.normal(0.0, config.geo_trend_sd, n_regions)
    phase = rng.uniform(0.0, 2 * np.pi)

    # Shared components: I(1) market factor (+ growth), extra I(1) factors,
    # yearly seasonality, day-of-week, calendar shocks.
    market = np.log1p(config.annual_growth) * years + random_walk(rng, n_days, config.market_volatility)
    factors = np.stack([random_walk(rng, n_days, config.factor_volatility)
                        for _ in range(config.n_factors)]) if config.n_factors else np.zeros((0, n_days))
    doy = 2 * np.pi * dates.dayofyear.to_numpy() / 365.25
    yearly = config.yearly_amplitude * (np.sin(doy + phase) + 0.3 * np.sin(2 * doy + 2 * phase))
    weekly = np.log(np.asarray(config.weekly_pattern, float) / np.mean(config.weekly_pattern))
    day_of_week = weekly[dates.dayofweek.to_numpy()]
    shocks = np.zeros(n_days)
    for when, multiplier in config.shocks:
        hit = dates == pd.Timestamp(when)
        shocks[hit] += np.log(multiplier)

    # State and unit stationary noise (what makes units differ but stay
    # cointegrated). Mean-corrected so E[exp(v + u)] = 1.
    n_states = int(state_idx.max()) + 1
    state_shock = ar1(rng, n_states, n_days, config.state_shock_sd, config.state_shock_ar)
    idio = ar1(rng, n_regions, n_days, config.idio_sd, config.idio_ar)
    stationary = state_shock[state_idx] + idio - (config.state_shock_sd**2 + config.idio_sd**2) / 2

    common = market + day_of_week + shocks
    log_mu = (
        np.log(shares)[:, None] + propensity[:, None] + common[None, :]
        + loadings @ factors
        + seasonal_loading[:, None] * yearly[None, :]
        + geo_trend[:, None] * years[None, :]
        + stationary
    )
    components = {
        "market": market, "factors": factors, "yearly": yearly, "day_of_week": day_of_week,
        "shocks": shocks, "loadings": loadings, "seasonal_loading": seasonal_loading,
        "geo_trend": geo_trend, "propensity": propensity, "state_shock": state_shock,
        "idiosyncratic": idio,
    }
    return log_mu, components
