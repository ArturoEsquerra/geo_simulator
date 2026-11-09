"""Configuration dataclasses for the geo-experiment simulator.

Everything that governs a simulation is declared here. The defaults produce a
"well-behaved" panel in which every region shares the same stochastic trends
and seasonality, so the regions are cointegrated and synthetic-control-type
estimators (GeoLift, CausalImpact, TBR/GeoX) are expected to work. The
``BaselineConfig`` stress-test knobs (``geo_trend_sd``, ``loading_sd``,
``propensity_sd``) let you break those assumptions on purpose.
"""

from __future__ import annotations

import dataclasses
from typing import Optional, Sequence

import pandas as pd


GEO_GRAINS = ("zip", "cluster", "state", "city")
GRANULARITIES = ("day", "week")
LIFT_SHAPES = ("constant", "increasing")
COOLDOWN_MODES = ("none", "decay")
DISTRIBUTIONS = ("negative_binomial", "poisson", "none")

# Day-of-week multipliers, Monday .. Sunday (normalised to mean 1 internally).
DEFAULT_WEEKLY_PATTERN = (1.00, 1.03, 1.04, 1.03, 1.06, 0.95, 0.89)


@dataclasses.dataclass
class GeographyConfig:
    """Where the geography (states, zip codes, cities, population) comes from.

    By default the data is downloaded once from GeoNames
    (https://download.geonames.org/export/, CC-BY 4.0) and cached in
    ``cache_dir``.

    Attributes:
        cache_dir: Directory for cached GeoNames files. Defaults to
            ``~/.cache/geo_simulator``.
        custom_geography: Bypass GeoNames entirely. A DataFrame with one row
            per simulated unit and columns ``geo``, ``state``, ``population``
            and, for contamination, ``latitude`` / ``longitude``. Use this for
            offline work, for countries GeoNames does not cover, or to pin an
            exact geography snapshot (``result.geography``) for reproducibility.
        custom_clusters: For ``geo_grain="cluster"``: your own zip -> unit
            mapping as a DataFrame with columns ``zip`` and ``geo`` (and
            optionally ``country``). Replaces the built-in k-means clusters.
            For Google GMA files (``PostCode, CountryCode, Gma, GeoName``)
            use ``load_gma_mapping(path)``; Nielsen DMAs work the same way.
            Leading zeros lost in spreadsheets are restored, and populated
            zips missing from the mapping join the unit of the nearest
            mapped zip (``result.membership["source"]`` says which).
        cluster_seed: Seed of the k-means initialisation for the built-in
            clusters. Independent of the simulation seed, so clusters stay
            fixed across simulations.
        state_population: Optional ``{state: population}`` override. Zip
            weights inside each state are rescaled so the state totals match
            (e.g. official census figures instead of the GeoNames proxy).
        postal_file: GeoNames postal file stem. Defaults to the country code
            (e.g. ``"GB"``); use ``"GB_full"`` / ``"NL_full"`` / ``"CA_full"``
            for full-length postcodes where GeoNames offers them.
        city_min_distance_km: City models only. A city closer than this to a
            larger, already selected city is skipped, so each metro area
            appears once (e.g. Brooklyn is not a separate geo from New York
            City, and its population is not double counted). ``0`` disables.
        min_state_share: Drop states holding less than this share of the
            population (e.g. the US postal file's Marshall Islands, ~0.0003%).
            Near-empty states produce all-zero series that break most
            estimators. ``0`` keeps everything.
    """

    cache_dir: Optional[str] = None
    custom_geography: Optional[pd.DataFrame] = None
    state_population: Optional[dict] = None
    custom_clusters: Optional[pd.DataFrame] = None
    cluster_seed: int = 0
    postal_file: Optional[str] = None
    min_state_share: float = 1e-4
    city_min_distance_km: float = 25.0


@dataclasses.dataclass
class BaselineConfig:
    """Counterfactual (no-intervention) data-generating process.

    The model runs at the chosen geo grain. The latent log-intensity of unit
    ``g`` (a zip, cluster, state or city) in state ``s(g)`` on day ``t`` is::

        log mu[g,t] = a[g] + M[t] + sum_k lam[g,k] F[k,t]
                      + gam[g] * S_year[t] + log W[dow(t)] + d[g] * t
                      + v[s(g),t] + u[g,t]

    * ``M`` — market factor: random walk with drift, loading 1 for everyone.
    * ``F`` — ``n_factors`` extra random-walk factors with loadings
      ``lam ~ N(0, loading_sd)``.
    * ``S_year`` — yearly seasonality, loadings ``gam ~ N(1, seasonality_loading_sd)``.
    * ``W`` — day-of-week pattern shared by all regions.
    * ``v`` — stationary AR(1) shock shared by the units of a state (local
      news, weather, state-level events), so units of the same state are more
      alike than units of different states.
    * ``u`` — stationary AR(1) idiosyncratic noise of the unit.
    * ``d`` — region-specific deterministic trend (stress test, default 0).

    Because every non-stationary component is shared (``M``, ``F``) and the
    state and unit parts (``v``, ``u``) are stationary, log-intensities are
    cointegrated across units. ``d != 0`` breaks this on purpose.

    Pass ``SimulationConfig(calibration=CalibrationConfig(history))`` to
    estimate these parameters from a real panel instead of using defaults.

    Attributes:
        annual_growth: Deterministic national growth per year (0.05 = +5%).
        market_volatility: Daily std of the market random-walk innovations
            (log scale).
        n_factors: Number of additional common stochastic factors.
        factor_volatility: Daily std of each extra factor's innovations.
        loading_sd: Std of region loadings on the extra factors. Larger
            values make regions follow more different paths (but they stay
            cointegrated as a system).
        weekly_pattern: Seven Monday-first day-of-week multipliers.
        yearly_amplitude: Amplitude of the yearly seasonal wave (log scale).
        seasonality_loading_sd: Region heterogeneity in seasonal amplitude.
        idio_sd: Marginal std of the stationary idiosyncratic noise.
        idio_ar: AR(1) coefficient of the idiosyncratic noise (|.| < 1).
        geo_trend_sd: Std of region-specific annual trends. Default 0.
            Non-zero values BREAK cointegration (stress test).
        propensity_sd: Log-normal dispersion of conversion propensity per
            region. 0 keeps conversions strictly proportional to population.
        shocks: Shared shocks as ``[(date, multiplier), ...]``, e.g. a
            national holiday ``("2024-11-29", 1.6)``. Applied to every region.
        state_shock_sd: Marginal std of the state-level shock ``v``.
        state_shock_ar: AR(1) coefficient of the state-level shock.
    """

    annual_growth: float = 0.03
    market_volatility: float = 0.004
    n_factors: int = 2
    factor_volatility: float = 0.003
    loading_sd: float = 0.5
    weekly_pattern: Sequence[float] = DEFAULT_WEEKLY_PATTERN
    yearly_amplitude: float = 0.08
    seasonality_loading_sd: float = 0.1
    idio_sd: float = 0.03
    idio_ar: float = 0.5
    geo_trend_sd: float = 0.0
    propensity_sd: float = 0.0
    shocks: Sequence[tuple] = ()
    state_shock_sd: float = 0.02
    state_shock_ar: float = 0.7


@dataclasses.dataclass
class NoiseConfig:
    """Count noise on top of the latent intensity.

    Counts are gamma-Poisson with mean ``mu`` and variance::

        mu * (1 + overdispersion) + mu**2 / nb_concentration

    * ``overdispersion`` (quasi-Poisson) is *grain-consistent*: zips summed
      to a state have the same overdispersion as a state simulated directly,
      so results are comparable across ``geo_grain`` choices.
    * ``nb_concentration`` adds noise whose relative size does not shrink
      with volume (a classic negative binomial). It is *not* grain-consistent:
      the same value is relatively much noisier for large units. ``None``
      (default) leaves it out.

    Attributes:
        distribution: ``"negative_binomial"`` (gamma-Poisson, default),
            ``"poisson"`` (ignores both parameters) or ``"none"`` (expected
            values, no count noise).
        overdispersion: Extra variance as a multiple of the Poisson variance.
        nb_concentration: Optional size-independent concentration ``r``.
    """

    distribution: str = "negative_binomial"
    overdispersion: float = 1.0
    nb_concentration: Optional[float] = None


@dataclasses.dataclass
class LiftConfig:
    """The true incremental effect of the intervention.

    Attributes:
        lift: Relative lift over the counterfactual during the test period,
            e.g. ``0.05`` for +5%, ``0`` for no effect, ``-0.03`` for -3%.
            Must be greater than -1.
        shape: ``"constant"`` — every time stamp has exactly ``lift``.
            ``"increasing"`` — the effect ramps up during the first
            ``ramp_periods`` time stamps and then plateaus; the plateau is
            solved so that the cumulative lift over the whole test period
            (total incremental / total counterfactual of the treated group)
            equals ``lift``. The plateau is therefore above ``lift``.
        ramp_periods: Length of the ramp for ``"increasing"``. Defaults to a
            third of the test period (at least 1).
        cooldown: ``"none"`` — the effect stops when the test ends.
            ``"decay"`` — the final test-period lift decays geometrically
            through the cooldown (carry-over).
        cooldown_decay: Per-period retention for ``cooldown="decay"``.
        geo_heterogeneity: Log-normal dispersion of the lift across treated
            geos. The geo lifts are rescaled so the treated group's aggregate
            lift still equals ``lift``. 0 = identical lift everywhere.
    """

    lift: float = 0.0
    shape: str = "constant"
    ramp_periods: Optional[int] = None
    cooldown: str = "none"
    cooldown_decay: float = 0.5
    geo_heterogeneity: float = 0.0


@dataclasses.dataclass
class TreatmentConfig:
    """Which geos receive the intervention.

    Attributes:
        treatment_geos: Explicit list of state names
            (``treatment_level="state"``) or unit ids at the chosen grain —
            zip codes, cluster names, city names (``treatment_level="geo"``).
        n_treatment_geos: Number of states / units to draw at random (with the
            simulation seed) when ``treatment_geos`` is not given.
        treatment_level: ``"state"`` (treat every unit of the chosen states),
            ``"geo"`` (treat individual units) or ``None`` (default):
            ``"state"`` for the zip grain, ``"geo"`` otherwise.
        exclude_extremes: When drawing at random, skip the largest and
            smallest 10% of geos (they are hard to match with a convex
            combination of controls).
    """

    treatment_geos: Optional[Sequence[str]] = None
    n_treatment_geos: int = 3
    treatment_level: Optional[str] = None
    exclude_extremes: bool = True


@dataclasses.dataclass
class ContaminationConfig:
    """Spillover of the treatment effect into neighbouring units.

    Each period, a share ``share`` of every treated unit's incremental
    conversions lands in its ``n_neighbors`` nearest units (by centroid
    distance) instead of in the unit itself, e.g. people who live across the
    border, see the campaign and buy there. The *total* effect of the
    campaign is unchanged; it is redistributed. Neighbour weights decay as
    ``exp(-distance / bandwidth_km)`` and are normalised to sum to 1.

    Neighbours may themselves be treated (the effect then stays in the
    treatment group); only spillover into control units contaminates the
    counterfactual. The same rule applies in the cooldown when the effect
    decays into it.

    Attributes:
        share: Share of each treated unit's incremental that bleeds to its
            neighbours, in [0, 1]. 0 (default) = no contamination.
        n_neighbors: Number of nearest units receiving the bleed.
        bandwidth_km: Distance-decay scale. Large values spread the bleed
            evenly over the ``n_neighbors``.
        max_distance_km: Neighbours farther than this receive nothing; if a
            treated unit has no neighbour within range, its effect stays home
            (e.g. Honolulu does not bleed into the mainland). ``None`` removes
            the cap.
    """

    share: float = 0.0
    n_neighbors: int = 5
    bandwidth_km: float = 50.0
    max_distance_km: Optional[float] = 500.0


@dataclasses.dataclass
class CalibrationConfig:
    """Estimate the baseline and noise parameters from a historical panel.

    Attributes:
        data: Long DataFrame with one row per (date, geo).
        date_col, geo_col, value_col: Column names in ``data``.
        min_mean: Units averaging fewer conversions per period are ignored
            for the log-scale estimates (their logs are dominated by count
            noise).
    """

    data: pd.DataFrame
    date_col: str = "date"
    geo_col: str = "geo"
    value_col: str = "conversions"
    min_mean: float = 20.0


@dataclasses.dataclass
class SimulationConfig:
    """Top-level configuration.

    Attributes:
        country: ISO-3166 alpha-2 code (``"US"``, ``"MX"``) or English name.
        geo_grain: The simulated (and output) unit; the factor model runs at
            this grain.
            ``"zip"`` — zip codes with population data;
            ``"cluster"`` — ``n_clusters`` DMA-like groups of zips (or your
            own mapping, ``GeographyConfig.custom_clusters``);
            ``"state"`` — first-level divisions;
            ``"city"`` — the ``n_cities`` largest metro areas.
        n_cities: Number of cities for ``geo_grain="city"``.
        n_clusters: Number of clusters for ``geo_grain="cluster"``.
        granularity: ``"day"`` or ``"week"`` (weeks start on Monday).
        start_date: First pre-period date. For weekly data it is moved back
            to the Monday of its week.
        n_pre_periods: Pre-intervention (historical) periods.
        n_test_periods: Intervention periods.
        n_cooldown_periods: Post-intervention cooldown periods (default 0).
        daily_conversions: Expected national (or all-cities) conversions per
            day, averaged over the pre-period.
        seed: Seed for every random draw. Same config + same geography +
            same seed gives identical output.
        contamination: Spillover of the effect into neighbouring units.
        calibration: Historical panel to estimate ``baseline`` and ``noise``
            from. ``None`` (default) uses the configured values.
    """

    country: str = "US"
    geo_grain: str = "zip"
    n_cities: int = 50
    n_clusters: int = 200
    granularity: str = "day"
    start_date: str = "2024-01-01"
    n_pre_periods: int = 90
    n_test_periods: int = 30
    n_cooldown_periods: int = 0
    daily_conversions: float = 10_000.0
    seed: int = 0
    lift: LiftConfig = dataclasses.field(default_factory=LiftConfig)
    treatment: TreatmentConfig = dataclasses.field(default_factory=TreatmentConfig)
    baseline: BaselineConfig = dataclasses.field(default_factory=BaselineConfig)
    noise: NoiseConfig = dataclasses.field(default_factory=NoiseConfig)
    geography: GeographyConfig = dataclasses.field(default_factory=GeographyConfig)
    contamination: ContaminationConfig = dataclasses.field(default_factory=ContaminationConfig)
    calibration: Optional[CalibrationConfig] = None

    @property
    def treatment_level(self) -> str:
        if self.treatment.treatment_level is not None:
            return self.treatment.treatment_level
        return "state" if self.geo_grain == "zip" else "geo"

    @property
    def n_periods(self) -> int:
        return self.n_pre_periods + self.n_test_periods + self.n_cooldown_periods

    def validate(self) -> None:
        """Raise ``ValueError`` on an inconsistent configuration."""
        _choice("geo_grain", self.geo_grain, GEO_GRAINS)
        _choice("granularity", self.granularity, GRANULARITIES)
        _choice("lift.shape", self.lift.shape, LIFT_SHAPES)
        _choice("lift.cooldown", self.lift.cooldown, COOLDOWN_MODES)
        _choice("noise.distribution", self.noise.distribution, DISTRIBUTIONS)
        _choice("treatment.treatment_level", self.treatment_level, ("state", "geo"))
        if self.n_pre_periods < 1 or self.n_test_periods < 1:
            raise ValueError("n_pre_periods and n_test_periods must be >= 1.")
        if self.n_cooldown_periods < 0:
            raise ValueError("n_cooldown_periods must be >= 0.")
        if self.daily_conversions <= 0:
            raise ValueError("daily_conversions must be > 0.")
        if self.lift.lift <= -1:
            raise ValueError("lift must be > -1 (a -100% lift removes every conversion).")
        if self.geo_grain == "city" and self.n_cities < 2:
            raise ValueError("n_cities must be >= 2.")
        if self.geo_grain == "cluster" and self.n_clusters < 2:
            raise ValueError("n_clusters must be >= 2.")
        c = self.contamination
        if not 0 <= c.share <= 1:
            raise ValueError("contamination.share must be in [0, 1].")
        if c.share > 0 and (c.n_neighbors < 1 or c.bandwidth_km <= 0):
            raise ValueError("contamination needs n_neighbors >= 1 and bandwidth_km > 0.")
        if not 0 <= self.lift.cooldown_decay < 1:
            raise ValueError("cooldown_decay must be in [0, 1).")
        if len(self.baseline.weekly_pattern) != 7:
            raise ValueError("weekly_pattern needs 7 values (Monday..Sunday).")
        for name in ("idio_ar", "state_shock_ar"):
            if not -1 < getattr(self.baseline, name) < 1:
                raise ValueError(f"baseline.{name} must be in (-1, 1) to stay stationary.")
        if self.noise.overdispersion < 0:
            raise ValueError("noise.overdispersion must be >= 0.")
        if self.noise.nb_concentration is not None and self.noise.nb_concentration <= 0:
            raise ValueError("noise.nb_concentration must be > 0 (or None).")


def _choice(name, value, allowed):
    if value not in allowed:
        raise ValueError(f"{name}={value!r}; expected one of {allowed}.")
