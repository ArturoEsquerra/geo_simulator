"""GeoSimulator: build a geo-experiment panel with a known ground truth."""

from __future__ import annotations

import dataclasses
import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from geo_simulator import geography as geography_lib
from geo_simulator.baseline import simulate_latent
from geo_simulator.calibration import CalibrationResult, calibrate, noise_for_granularity
from geo_simulator.config import SimulationConfig
from geo_simulator.contamination import neighbor_matrix, redistribute
from geo_simulator.lift import lift_matrix, sample_counts


logger = logging.getLogger(__name__)

# Independent random streams. Each component gets its own child seed, so
# changing the treatment, the lift or the contamination never changes the
# baseline draws.
_STREAMS = ("baseline", "unused", "treatment", "lift", "counts")

_GEO_COLUMN = {"zip": "zip", "cluster": "cluster", "state": "state", "city": "city"}


class GeoSimulator:
    """Simulate conversions for a geo experiment.

    Example::

        from geo_simulator import GeoSimulator, SimulationConfig, LiftConfig
        cfg = SimulationConfig(country="US", geo_grain="cluster", n_pre_periods=90,
                               n_test_periods=28, daily_conversions=20_000,
                               lift=LiftConfig(lift=0.05), seed=1)
        result = GeoSimulator(cfg).run()
        result.to_csv("geo_data.csv")

    Args:
        config: Simulation configuration.
        geography: Optional pre-built unit table (``geo, state, population``
            and optionally ``latitude, longitude``). Reused by
            ``SimulationResult.with_treatment`` to avoid downloading again and
            to guarantee the same units.
        membership: Optional zip -> unit mapping that goes with ``geography``.
    """

    def __init__(self, config: SimulationConfig, geography: Optional[pd.DataFrame] = None,
                 membership: Optional[pd.DataFrame] = None):
        config.validate()
        self.config = config
        self._geography = geography
        self._membership = membership

    def run(self) -> "SimulationResult":
        cfg = self.config
        streams = dict(zip(_STREAMS, (np.random.default_rng(s) for s in np.random.SeedSequence(cfg.seed).spawn(len(_STREAMS)))))

        # 0. Calibration: estimate the baseline from history if given --------
        calibration = None
        if cfg.calibration is not None:
            calibration = calibrate(cfg.calibration, cfg.baseline, cfg.noise)
            cfg = dataclasses.replace(
                cfg, calibration=None, baseline=calibration.baseline,
                noise=noise_for_granularity(calibration.noise, calibration.granularity, cfg.granularity))
            cfg.validate()

        # 1. Geography --------------------------------------------------------
        if self._geography is not None:
            country_code, membership = cfg.country, self._membership
            geo = geography_lib.finalize_geography(
                self._geography, dataclasses.replace(cfg.geography, state_population=None, min_state_share=0.0))
        else:
            country_code, geo, membership = geography_lib.build_geography(
                cfg.country, cfg.geo_grain, cfg.geography, n_cities=cfg.n_cities, n_clusters=cfg.n_clusters)
        if len(geo) < 2:
            raise ValueError("Need at least two geos.")
        shares = geo["population"].to_numpy() / geo["population"].sum()
        _, state_idx = np.unique(geo["state"].to_numpy(), return_inverse=True)

        # 2. Calendar ---------------------------------------------------------
        days_per_period = 7 if cfg.granularity == "week" else 1
        start = pd.Timestamp(cfg.start_date)
        if cfg.granularity == "week" and start.dayofweek != 0:
            monday = start - pd.Timedelta(days=start.dayofweek)
            logger.warning("start_date %s is not a Monday; weekly data starts on %s.", start.date(), monday.date())
            start = monday
        days = pd.date_range(start, periods=cfg.n_periods * days_per_period, freq="D")
        periods = days[::days_per_period]

        # 3. Latent counterfactual intensity (units x periods) ----------------
        log_mu, components = simulate_latent(streams["baseline"], shares, state_idx, days, cfg.baseline)
        daily = np.exp(log_mu)
        pre_days = cfg.n_pre_periods * days_per_period
        daily *= cfg.daily_conversions / daily[:, :pre_days].sum(axis=0).mean()
        expected = daily.reshape(len(geo), cfg.n_periods, days_per_period).sum(axis=2)
        del daily, log_mu

        # 4. Treatment, effect and contamination ------------------------------
        treatment_geos, treated_unit = self._assign_treatment(cfg, geo, streams["treatment"])
        lift = lift_matrix(streams["lift"], expected, treated_unit, cfg.n_pre_periods,
                           cfg.n_test_periods, cfg.n_cooldown_periods, cfg.lift)
        effect = lift * expected                      # incremental created by the campaign
        neighbors = None
        if cfg.contamination.share > 0:
            if not {"latitude", "longitude"} <= set(geo.columns):
                raise ValueError("Contamination needs unit coordinates: add latitude/longitude to custom_geography.")
            neighbors = neighbor_matrix(geo["latitude"].to_numpy(), geo["longitude"].to_numpy(), cfg.contamination)
        direct, spill_in = redistribute(effect, neighbors, cfg.contamination.share) if neighbors is not None \
            else (effect, np.zeros_like(effect))
        change = np.divide(direct + spill_in, expected, out=np.zeros_like(expected), where=expected > 0)
        if (change <= -1).any():
            raise ValueError("The (contaminated) effect removes more than 100% of some unit's conversions. "
                             "Use a smaller negative lift or less contamination.")

        # 5. Counts -----------------------------------------------------------
        counterfactual, observed = sample_counts(streams["counts"], expected, change, cfg.noise)

        return SimulationResult(
            config=cfg, country_code=country_code, geography=geo, membership=membership, periods=periods,
            expected=expected, effect=effect, direct=direct, spill_in=spill_in, change=change,
            counterfactual=counterfactual, observed=observed, treated_unit=treated_unit,
            treatment_geos=treatment_geos, components=components, neighbors=neighbors, calibration=calibration,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _assign_treatment(cfg: SimulationConfig, geo: pd.DataFrame, rng: np.random.Generator) -> tuple[list, np.ndarray]:
        treatment = cfg.treatment
        unit_col = "state" if cfg.treatment_level == "state" else "geo"
        units = geo.groupby(unit_col)["population"].sum().sort_values()

        if treatment.treatment_geos is not None:
            chosen = [str(g) for g in treatment.treatment_geos]
            unknown = sorted(set(chosen) - set(units.index))
            if unknown:
                raise ValueError(f"Unknown treatment geos ({cfg.treatment_level}): {unknown[:10]}")
        else:
            pool = units.index.to_numpy()
            if treatment.exclude_extremes and len(pool) >= 10:
                cut = len(pool) // 10
                pool = pool[cut:len(pool) - cut]
            n = min(treatment.n_treatment_geos, len(units) - 1)
            if n > len(pool):
                raise ValueError(f"Cannot draw {n} treatment geos from {len(pool)} candidates.")
            chosen = sorted(rng.choice(pool, size=n, replace=False).tolist()) if n > 0 else []

        if len(chosen) >= len(units):
            raise ValueError("Every geo is treated; leave at least one control geo.")
        unit_id = {u: i for i, u in enumerate(chosen)}
        treated_unit = geo[unit_col].map(unit_id).fillna(-1).astype(int).to_numpy()
        return chosen, treated_unit


@dataclasses.dataclass
class SimulationResult:
    """Output of ``GeoSimulator.run``.

    Arrays have shape ``(n_units, n_periods)`` with units in ``geography``
    order:

    * ``expected`` — expected counterfactual conversions.
    * ``effect`` — expected incremental conversions *created* by each treated
      unit's campaign (before contamination).
    * ``direct`` / ``spill_in`` — where that effect lands: the part that
      stays in the unit, and the spillover the unit receives from treated
      neighbours. ``direct + spill_in`` sums to ``effect`` overall.
    * ``change`` — expected relative change of each unit, ``(direct +
      spill_in) / expected``.
    * ``counterfactual`` / ``observed`` — coupled realised outcomes.

    Use ``data``, ``to_csv``, ``aggregate`` and ``ground_truth`` rather than
    the raw arrays.
    """

    config: SimulationConfig
    country_code: str
    geography: pd.DataFrame
    membership: Optional[pd.DataFrame]
    periods: pd.DatetimeIndex
    expected: np.ndarray
    effect: np.ndarray
    direct: np.ndarray
    spill_in: np.ndarray
    change: np.ndarray
    counterfactual: np.ndarray
    observed: np.ndarray
    treated_unit: np.ndarray
    treatment_geos: list
    components: dict
    neighbors: object = None
    calibration: Optional[CalibrationResult] = None

    # -- calendar ------------------------------------------------------
    @property
    def geo_column(self) -> str:
        return _GEO_COLUMN[self.config.geo_grain]

    @property
    def period_labels(self) -> np.ndarray:
        cfg = self.config
        return np.repeat(np.array(["pre", "test", "cooldown"]),
                         [cfg.n_pre_periods, cfg.n_test_periods, cfg.n_cooldown_periods])

    @property
    def test_start(self) -> pd.Timestamp:
        return self.periods[self.config.n_pre_periods]

    @property
    def test_end(self) -> pd.Timestamp:
        """Last date covered by the test (the Sunday of the last week for weekly data)."""
        last = self.periods[self.config.n_pre_periods + self.config.n_test_periods - 1]
        return last + pd.Timedelta(days=6) if self.config.granularity == "week" else last

    @property
    def treated_geos(self) -> list:
        """Unit ids (zips, clusters, states or cities) that receive the treatment."""
        return self.geography.loc[self.treated_unit >= 0, "geo"].tolist()

    # -- data ------------------------------------------------------------
    @property
    def data(self) -> pd.DataFrame:
        """Long panel, one row per (date, unit).

        Columns: ``date``, the unit column (``zip`` / ``cluster`` / ``state`` /
        ``city``), ``state``, ``conversions`` and the ground-truth columns
        ``counterfactual_conversions``, ``incremental_conversions`` (observed
        - counterfactual, including spillover received), ``expected_change``
        (expected relative change), ``treated`` and ``period``.
        """
        n_units, n_periods = self.observed.shape
        geo = self.geography
        counts = np.int32 if self.config.noise.distribution != "none" else np.float64
        # Categoricals and 32-bit numbers keep large daily panels within
        # notebook memory.
        columns = {"date": np.repeat(self.periods.to_numpy(), n_units)}
        geo_codes = pd.Categorical(geo["geo"])
        columns[self.geo_column] = pd.Categorical.from_codes(np.tile(geo_codes.codes, n_periods), geo_codes.categories)
        if self.geo_column != "state":
            state_codes = pd.Categorical(geo["state"])
            columns["state"] = pd.Categorical.from_codes(np.tile(state_codes.codes, n_periods), state_codes.categories)
        columns.update({
            "conversions": self.observed.T.ravel().astype(counts),
            "counterfactual_conversions": self.counterfactual.T.ravel().astype(counts),
            "incremental_conversions": (self.observed - self.counterfactual).T.ravel().astype(counts),
            "expected_change": self.change.T.ravel().astype(np.float32),
            "treated": np.tile(self.treated_unit >= 0, n_periods),
            "period": pd.Categorical(np.repeat(self.period_labels, n_units), categories=["pre", "test", "cooldown"]),
        })
        # Geography is sorted by (state, geo): rows are in (date, state, unit) order.
        return pd.DataFrame(columns)

    def to_csv(self, path: str, minimal: bool = False, **kwargs) -> None:
        """Write the panel to CSV.

        ``minimal=True`` keeps only ``date``, the unit column, ``state`` and
        ``conversions`` (the data an analyst would see). Zip codes are written
        as text; read them back with ``dtype={"zip": str}`` to keep leading
        zeros.
        """
        frame = self.data
        if minimal:
            frame = frame[list(dict.fromkeys(["date", self.geo_column, "state", "conversions"]))]
        frame.to_csv(path, index=False, date_format="%Y-%m-%d", **kwargs)

    def aggregate(self, by: str = "state", include_truth: bool = True) -> pd.DataFrame:
        """Sum the panel to ``by`` x date (e.g. state level for analysis).

        ``treated_share`` is the share of the unit's counterfactual conversions
        in treated units (1.0 for a fully treated state).
        """
        frame = self.data
        sums = ["conversions"] + (["counterfactual_conversions", "incremental_conversions"] if include_truth else [])
        frame["_treated_cf"] = frame["counterfactual_conversions"] * frame["treated"]
        grouped = frame.groupby(["date", by], sort=True, observed=True)
        out = grouped.agg(**{c: (c, "sum") for c in sums + ["_treated_cf"]}, period=("period", "first")).reset_index()
        denom = grouped["counterfactual_conversions"].sum().to_numpy()
        out["treated_share"] = np.divide(out.pop("_treated_cf").to_numpy(), denom,
                                         out=np.zeros(len(out)), where=denom > 0)
        # Plain strings: analysis tools (e.g. GeoX's schema) reject categoricals.
        out[by] = out[by].astype(str)
        out["period"] = out["period"].astype(str)
        if not include_truth:
            out = out.drop(columns=["treated_share"])
        return out

    # -- ground truth ----------------------------------------------------
    def ground_truth(self) -> dict:
        """What a correct analysis should recover — and what contamination hides.

        For the test period (and ``cooldown_*`` for the cooldown):

        * ``configured_lift`` — the lift you asked for.
        * ``total_lift`` — expected incremental created by the campaign /
          expected counterfactual of the treated units. Equals
          ``configured_lift``; contamination does not change it.
        * ``expected_treated_lift`` — the part of the effect that lands in
          treated units / their expected counterfactual. Lower than
          ``total_lift`` when contamination leaks effect to controls.
        * ``realized_lift`` — observed / counterfactual - 1 of the treated
          units on the actual draws: the treated units' realised ATT, what an
          estimator with a *perfect* counterfactual would report.
        * ``control_contamination`` — observed / counterfactual - 1 of the
          control units (0 without contamination).
        * ``identifiable_lift`` — ``(1 + realized_lift) / (1 + control_contamination) - 1``:
          roughly what a control-based estimator (TBR, synthetic control)
          converges to when the controls are contaminated.
        * ``incremental_conversions`` (treated), ``spillover_to_controls``,
          ``total_incremental_conversions`` — realised counts.
        """
        cfg = self.config
        treated = self.treated_unit >= 0
        test = slice(cfg.n_pre_periods, cfg.n_pre_periods + cfg.n_test_periods)
        cool = slice(cfg.n_pre_periods + cfg.n_test_periods, cfg.n_periods)

        def block(sl):
            mu_t = self.expected[treated, sl].sum()
            y1_t, y0_t = self.observed[treated, sl].sum(), self.counterfactual[treated, sl].sum()
            y1_c, y0_c = self.observed[~treated, sl].sum(), self.counterfactual[~treated, sl].sum()
            landed = self.direct[:, sl] + self.spill_in[:, sl]
            realized = y1_t / y0_t - 1 if y0_t else float("nan")
            contamination = y1_c / y0_c - 1 if y0_c else 0.0
            return {
                "total_lift": float(self.effect[treated, sl].sum() / mu_t) if mu_t else 0.0,
                "expected_treated_lift": float(landed[treated].sum() / mu_t) if mu_t else 0.0,
                "realized_lift": float(realized),
                "control_contamination": float(contamination),
                "identifiable_lift": float((1 + realized) / (1 + contamination) - 1),
                "incremental_conversions": float(y1_t - y0_t),
                "spillover_to_controls": float(y1_c - y0_c),
                "total_incremental_conversions": float(y1_t - y0_t + y1_c - y0_c),
                "expected_incremental_conversions": float(self.effect[:, sl].sum()),
                "observed_conversions": float(y1_t),
                "counterfactual_conversions": float(y0_t),
            }

        truth = {"configured_lift": cfg.lift.lift, "shape": cfg.lift.shape,
                 "contamination_share": cfg.contamination.share,
                 "treatment_geos": list(self.treatment_geos), "n_treated_units": int(treated.sum()),
                 "test_start": self.test_start, "test_end": self.test_end, **block(test)}
        if cfg.n_cooldown_periods:
            truth.update({f"cooldown_{k}": v for k, v in block(cool).items()})
        pre = slice(0, cfg.n_pre_periods)
        truth["treated_share_of_conversions"] = float(self.counterfactual[treated, pre].sum() / self.counterfactual[:, pre].sum())
        truth["lift_profile"] = self.lift_profile()
        return truth

    def lift_profile(self) -> pd.DataFrame:
        """Per period: total and treated lift (expected / realised) and control contamination."""
        treated = self.treated_unit >= 0
        mu = self.expected[treated].sum(0)
        landed = self.direct + self.spill_in
        profile = pd.DataFrame({
            "date": self.periods,
            "period": self.period_labels,
            "expected_lift": self.effect[treated].sum(0) / mu,
            "expected_treated_lift": landed[treated].sum(0) / mu,
            "observed": self.observed[treated].sum(0),
            "counterfactual": self.counterfactual[treated].sum(0),
        })
        profile["realized_lift"] = profile["observed"] / profile["counterfactual"] - 1
        controls_cf = self.counterfactual[~treated].sum(0)
        profile["control_contamination"] = np.divide(
            self.observed[~treated].sum(0) - controls_cf, controls_cf,
            out=np.zeros(len(controls_cf)), where=controls_cf > 0)
        return profile

    def summary(self) -> None:
        cfg = self.config
        truth = self.ground_truth()
        print(f"Country {self.country_code} | grain={cfg.geo_grain} | granularity={cfg.granularity} | seed={cfg.seed}"
              + (" | baseline calibrated from history" if self.calibration is not None else ""))
        print(f"Units: {len(self.geography):,} ({self.geo_column}) in {self.geography['state'].nunique()} states | "
              f"periods: {cfg.n_pre_periods} pre + {cfg.n_test_periods} test + {cfg.n_cooldown_periods} cooldown")
        print(f"Dates: {self.periods[0].date()} -> {self.periods[-1].date()} | test {self.test_start.date()} -> {self.test_end.date()}")
        print(f"Treatment ({cfg.treatment_level}): {truth['treatment_geos'][:8]}{' ...' if len(truth['treatment_geos']) > 8 else ''}"
              f" = {truth['n_treated_units']:,} units, {truth['treated_share_of_conversions']:.1%} of pre-period conversions")
        print(f"Lift: configured {truth['configured_lift']:+.2%} ({cfg.lift.shape}) | realized in treated {truth['realized_lift']:+.2%}"
              f" | incremental {truth['incremental_conversions']:+,.0f}")
        if cfg.contamination.share > 0:
            print(f"Contamination: share {cfg.contamination.share:.0%} -> {truth['spillover_to_controls']:+,.0f} conversions in controls"
                  f" ({truth['control_contamination']:+.2%}) | identifiable lift ~ {truth['identifiable_lift']:+.2%}")

    # -- re-use --------------------------------------------------------
    def with_treatment(self, treatment_geos: Optional[Sequence[str]] = None, lift=None, contamination=None,
                       **treatment_kwargs) -> "SimulationResult":
        """Re-run with a different treatment, lift or contamination; same baseline.

        The counterfactual panel is identical (same seed streams, same
        units, same calibrated baseline). Typical use: design the test on the
        pre-period of a ``lift=0`` simulation, then inject the lift into the
        chosen units.

        Args:
            treatment_geos: New treatment states or units.
            lift: A ``LiftConfig`` or a float (replaces ``lift.lift`` only).
            contamination: A ``ContaminationConfig`` or a float (replaces
                ``contamination.share`` only).
            **treatment_kwargs: Other ``TreatmentConfig`` fields.
        """
        cfg = self.config
        treatment = dataclasses.replace(cfg.treatment, **treatment_kwargs)
        if treatment_geos is not None:
            treatment = dataclasses.replace(treatment, treatment_geos=list(treatment_geos))
        lift_cfg = dataclasses.replace(cfg.lift, lift=float(lift)) if isinstance(lift, (int, float)) else (lift or cfg.lift)
        cont = cfg.contamination
        if isinstance(contamination, (int, float)):
            cont = dataclasses.replace(cont, share=float(contamination))
        elif contamination is not None:
            cont = contamination
        new_cfg = dataclasses.replace(cfg, treatment=treatment, lift=lift_cfg, contamination=cont, country=self.country_code)
        result = GeoSimulator(new_cfg, geography=self.geography, membership=self.membership).run()
        result.calibration = self.calibration
        return result

    # -- plotting ----------------------------------------------------------
    def plot(self, ax=None):
        """Treated vs. control conversions, indexed to the pre-period mean."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(12, 4.5))
        treated = self.treated_unit >= 0
        pre = self.config.n_pre_periods
        for mask, label, color in [(~treated, "Control units", "#8a8984"), (treated, "Treated units", "#2a78d6")]:
            series = self.observed[mask].sum(0)
            ax.plot(self.periods, series / series[:pre].mean(), label=label, color=color, linewidth=1.8)
        cf = self.counterfactual[treated].sum(0)
        ax.plot(self.periods, cf / self.observed[treated].sum(0)[:pre].mean(), label="Treated counterfactual",
                color="#2a78d6", linestyle="--", linewidth=1.2)
        ax.axvspan(self.test_start, self.test_end, color="#2a78d6", alpha=0.08, label="Test")
        if self.config.n_cooldown_periods:
            ax.axvspan(self.test_end, self.periods[-1], color="#eb6834", alpha=0.08, label="Cooldown")
        ax.set(title="Conversions indexed to pre-period mean", ylabel="Index (pre-period mean = 1)")
        ax.legend(loc="upper left", ncol=5, frameon=False)
        return ax
