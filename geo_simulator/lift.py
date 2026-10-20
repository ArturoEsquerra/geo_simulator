"""Lift profiles and coupled injection of the treatment effect.

The counterfactual and treated outcomes share the same random draws, so the
incremental conversions of every row are known exactly:

* Count noise is gamma-Poisson: ``lam ~ Gamma`` (see ``NoiseConfig``) then
  ``Y0 ~ Poisson(lam)``.
* Positive lift ``l``: ``Y1 = Y0 + Poisson(l * lam)``  ->  ``Y1 ~ Poisson((1 + l) lam)``.
* Negative lift ``l``: ``Y1 = Binomial(Y0, 1 + l)``     ->  ``Y1 ~ Poisson((1 + l) lam)``.

In both cases ``E[Y1] = (1 + l) E[Y0]``, with the same latent path.
"""

from __future__ import annotations

import math

import numpy as np

from geo_simulator.config import LiftConfig, NoiseConfig


def ramp_profile(n_test: int, config: LiftConfig) -> np.ndarray:
    """Relative shape of the effect over the test period (plateau = 1)."""
    if config.shape == "constant":
        return np.ones(n_test)
    ramp = config.ramp_periods or max(1, math.ceil(n_test / 3))
    ramp = min(ramp, n_test)
    steps = np.arange(1, n_test + 1)
    return np.minimum(1.0, steps / ramp)


def lift_matrix(
    rng: np.random.Generator,
    expected: np.ndarray,
    treated_unit: np.ndarray,
    n_pre: int,
    n_test: int,
    n_cooldown: int,
    config: LiftConfig,
) -> np.ndarray:
    """Relative lift per (geo, period); zero outside treated rows / periods.

    The lift is scaled so that, over the test period, the treated group's
    expected incremental conversions equal ``config.lift`` times its expected
    counterfactual conversions — for both shapes and any heterogeneity.

    Args:
        rng: Dedicated generator (used only for geo heterogeneity).
        expected: Expected counterfactual conversions, ``(n_geos, n_periods)``.
        treated_unit: Integer id of the treatment unit for each geo (``-1`` for
            control geos). Geos sharing a unit (e.g. zips of a treated state)
            share the same heterogeneity multiplier.
        n_pre, n_test, n_cooldown: Period counts.
        config: Lift configuration.
    """
    lift = np.zeros_like(expected, dtype=float)
    treated = treated_unit >= 0
    if config.lift == 0 or not treated.any():
        return lift

    test = slice(n_pre, n_pre + n_test)
    profile = ramp_profile(n_test, config)
    n_units = treated_unit.max() + 1
    unit_mult = np.exp(rng.normal(0.0, config.geo_heterogeneity, n_units) - config.geo_heterogeneity**2 / 2)
    geo_mult = unit_mult[treated_unit[treated]]

    base = expected[treated, test]
    shape = geo_mult[:, None] * profile[None, :]
    scale = config.lift * base.sum() / (shape * base).sum()
    lift[treated, test] = scale * shape

    if n_cooldown and config.cooldown == "decay":
        decay = config.cooldown_decay ** np.arange(1, n_cooldown + 1)
        last = lift[treated, n_pre + n_test - 1]
        lift[treated, n_pre + n_test:] = last[:, None] * decay[None, :]

    if (lift <= -1).any():
        raise ValueError(
            f"lift={config.lift} with shape={config.shape!r} and geo_heterogeneity="
            f"{config.geo_heterogeneity} needs a per-period lift <= -100% somewhere. "
            "Use a smaller negative lift, a longer ramp, or less heterogeneity."
        )
    return lift


def sample_counts(
    rng: np.random.Generator,
    expected: np.ndarray,
    lift: np.ndarray,
    noise: NoiseConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return coupled ``(counterfactual, observed)`` outcomes.

    ``lift`` is each cell's relative change (positive, zero or negative). The
    gamma intensity has mean ``expected`` and variance
    ``overdispersion * mu + mu**2 / nb_concentration`` (see ``NoiseConfig``).
    """
    if noise.distribution == "none":
        return expected.copy(), expected * (1.0 + lift)

    intensity = expected
    if noise.distribution == "negative_binomial":
        extra = noise.overdispersion + (expected / noise.nb_concentration if noise.nb_concentration else 0.0)
        extra = np.broadcast_to(extra, expected.shape)
        noisy = (extra > 0) & (expected > 0)
        intensity = expected.copy()
        shape = expected[noisy] / extra[noisy]
        intensity[noisy] = rng.gamma(shape, expected[noisy] / shape)
    counterfactual = rng.poisson(intensity)

    observed = counterfactual.copy()
    up = lift > 0
    down = lift < 0
    if up.any():
        observed[up] += rng.poisson(lift[up] * intensity[up])
    if down.any():
        observed[down] = rng.binomial(counterfactual[down], 1.0 + lift[down])
    return counterfactual, observed
