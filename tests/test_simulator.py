"""Tests for GeoSimulator.

Run with:  pytest tests/
Network tests (GeoNames download) run only with:  GEO_SIM_NETWORK=1 pytest tests/
"""

import dataclasses
import os

import numpy as np
import pandas as pd
import pytest

from geo_simulator import (
    BaselineConfig,
    CalibrationConfig,
    ContaminationConfig,
    GeographyConfig,
    GeoSimulator,
    LiftConfig,
    NoiseConfig,
    SimulationConfig,
    TreatmentConfig,
    calibrate,
)
from geo_simulator.calibration import noise_for_granularity
from geo_simulator.diagnostics import cointegration_report, pre_period_fit
from geo_simulator.geography import aggregate_geography, cluster_membership
from geo_simulator.lift import ramp_profile


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def toy_geography(n_states=12, zips_per_state=5, seed=0):
    """States on a 4-wide grid ~2 degrees apart; zips scattered around them."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_states):
        lat0, lon0 = 35 + 2 * (s // 4), -100 + 2 * (s % 4)
        for z in range(zips_per_state):
            rows.append({"geo": f"{s:02d}{z:03d}", "state": f"State {s:02d}",
                         "population": float(rng.lognormal(10, 1)),
                         "latitude": lat0 + rng.uniform(-0.3, 0.3), "longitude": lon0 + rng.uniform(-0.3, 0.3),
                         "place_name": f"Town {s}-{z}"})
    return pd.DataFrame(rows)


def toy_states(n_states=12):
    """One row per state (for geo_grain='state' style tests)."""
    zips = toy_geography(n_states)
    return aggregate_geography(zips, pd.DataFrame({"zip": zips["geo"], "geo": zips["state"]}))


def make_config(**overrides):
    base = dict(
        country="XX",
        n_pre_periods=60,
        n_test_periods=20,
        daily_conversions=50_000,
        seed=7,
        geography=GeographyConfig(custom_geography=toy_geography()),
    )
    base.update(overrides)
    return SimulationConfig(**base)


@pytest.fixture(scope="module")
def result():
    cfg = make_config(lift=LiftConfig(lift=0.08), n_cooldown_periods=5)
    return GeoSimulator(cfg).run()


# ---------------------------------------------------------------------------
# Output shape and schema
# ---------------------------------------------------------------------------


def test_output_schema(result):
    data = result.data
    assert list(data.columns[:4]) == ["date", "zip", "state", "conversions"]
    assert len(data) == 12 * 5 * (60 + 20 + 5)
    assert pd.api.types.is_string_dtype(data["zip"]) and data["zip"].str.len().eq(5).all()
    assert set(data["period"]) == {"pre", "test", "cooldown"}
    assert (data["conversions"] >= 0).all()


def test_daily_volume_matches_request(result):
    pre = result.data.query("period == 'pre'")
    daily = pre.groupby("date")["counterfactual_conversions"].sum().mean()
    assert daily == pytest.approx(50_000, rel=0.02)


def test_csv_round_trip_keeps_leading_zeros(result, tmp_path):
    path = tmp_path / "geo.csv"
    result.to_csv(path, minimal=True)
    back = pd.read_csv(path, dtype={"zip": str})
    assert list(back.columns) == ["date", "zip", "state", "conversions"]
    assert back["zip"].str.startswith("00").any()


# ---------------------------------------------------------------------------
# Reproducibility and coupling
# ---------------------------------------------------------------------------


def test_same_seed_same_data(result):
    again = GeoSimulator(result.config).run()
    np.testing.assert_array_equal(result.observed, again.observed)


def test_different_seed_different_data(result):
    other = GeoSimulator(dataclasses.replace(result.config, seed=8)).run()
    assert not np.array_equal(result.observed, other.observed)


def test_counterfactual_independent_of_treatment_and_lift(result):
    other = result.with_treatment(["State 03", "State 05"], lift=-0.2)
    np.testing.assert_array_equal(result.counterfactual, other.counterfactual)
    np.testing.assert_allclose(result.expected, other.expected)


def test_lift_only_on_treated_test_rows(result):
    data = result.data
    untouched = data[(~data["treated"]) | (data["period"] == "pre")]
    assert (untouched["incremental_conversions"] == 0).all()
    assert (untouched["expected_change"] == 0).all()
    cooldown = data[data["period"] == "cooldown"]
    assert (cooldown["incremental_conversions"] == 0).all()  # cooldown="none"


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lift", [0.1, 0.0, -0.1])
@pytest.mark.parametrize("shape", ["constant", "increasing"])
def test_expected_lift_matches_config(lift, shape):
    cfg = make_config(lift=LiftConfig(lift=lift, shape=shape))
    truth = GeoSimulator(cfg).run().ground_truth()
    assert truth["total_lift"] == pytest.approx(lift, abs=1e-12)
    assert truth["expected_treated_lift"] == pytest.approx(lift, abs=1e-12)  # no contamination


def test_realized_lift_close_to_expected_without_noise():
    cfg = make_config(lift=LiftConfig(lift=0.05), noise=NoiseConfig(distribution="none"))
    truth = GeoSimulator(cfg).run().ground_truth()
    assert truth["realized_lift"] == pytest.approx(0.05, abs=1e-12)


def test_negative_lift_counts_never_exceed_counterfactual():
    res = GeoSimulator(make_config(lift=LiftConfig(lift=-0.3))).run()
    assert (res.observed <= res.counterfactual).all()
    assert res.ground_truth()["realized_lift"] == pytest.approx(-0.3, abs=0.02)


def test_increasing_shape_ramps_then_plateaus():
    res = GeoSimulator(make_config(lift=LiftConfig(lift=0.1, shape="increasing", ramp_periods=5))).run()
    profile = res.lift_profile().query("period == 'test'")["expected_lift"].to_numpy()
    assert np.all(np.diff(profile[:5]) > 0)
    assert profile[5:].std() / profile[5:].mean() < 1e-9
    assert profile[-1] > 0.1  # plateau above the average
    assert profile[0] < 0.1


def test_ramp_profile_shapes():
    assert np.all(ramp_profile(10, LiftConfig(shape="constant")) == 1)
    ramp = ramp_profile(9, LiftConfig(shape="increasing"))
    np.testing.assert_allclose(ramp, [1 / 3, 2 / 3, 1, 1, 1, 1, 1, 1, 1])


def test_cooldown_decay():
    cfg = make_config(lift=LiftConfig(lift=0.1, cooldown="decay", cooldown_decay=0.5), n_cooldown_periods=4,
                      noise=NoiseConfig(distribution="none"))
    profile = GeoSimulator(cfg).run().lift_profile()
    last_test = profile.query("period == 'test'")["expected_lift"].iloc[-1]
    cool = profile.query("period == 'cooldown'")["expected_lift"].to_numpy()
    np.testing.assert_allclose(cool, last_test * 0.5 ** np.arange(1, 5))


def test_geo_heterogeneity_preserves_aggregate():
    cfg = make_config(lift=LiftConfig(lift=0.06, geo_heterogeneity=0.8))
    res = GeoSimulator(cfg).run()
    assert res.ground_truth()["total_lift"] == pytest.approx(0.06, abs=1e-12)
    per_state = res.data.query("period == 'test' and treated").groupby("state", observed=True)["expected_change"].mean()
    assert per_state.nunique() == len(per_state) > 1


def test_too_negative_ramp_raises():
    with pytest.raises(ValueError, match="-100%"):
        GeoSimulator(make_config(lift=LiftConfig(lift=-0.95, shape="increasing"))).run()


# ---------------------------------------------------------------------------
# Treatment assignment
# ---------------------------------------------------------------------------


def test_explicit_state_treatment(result):
    res = result.with_treatment(["State 01"])
    assert set(res.data.query("treated")["state"]) == {"State 01"}


def test_zip_level_treatment():
    cfg = make_config(lift=LiftConfig(lift=0.1),
                      treatment=TreatmentConfig(treatment_geos=["00001", "01002"], treatment_level="geo"))
    res = GeoSimulator(cfg).run()
    assert sorted(res.treated_geos) == ["00001", "01002"]


def test_random_treatment_excludes_extremes():
    cfg = make_config(treatment=TreatmentConfig(n_treatment_geos=3))
    res = GeoSimulator(cfg).run()
    sizes = res.geography.groupby("state")["population"].sum().sort_values()
    assert sizes.index[-1] not in res.treatment_geos and sizes.index[0] not in res.treatment_geos


def test_unknown_treatment_geo_raises():
    with pytest.raises(ValueError, match="Unknown treatment"):
        GeoSimulator(make_config(treatment=TreatmentConfig(treatment_geos=["Atlantis"]))).run()


# ---------------------------------------------------------------------------
# Granularity and levels
# ---------------------------------------------------------------------------


def test_weekly_starts_monday_and_scales():
    cfg = make_config(granularity="week", start_date="2024-01-03", n_pre_periods=20, n_test_periods=6)
    res = GeoSimulator(cfg).run()
    assert (res.periods.dayofweek == 0).all()
    weekly = res.data.query("period == 'pre'").groupby("date")["counterfactual_conversions"].sum().mean()
    assert weekly == pytest.approx(7 * 50_000, rel=0.02)
    assert res.test_end == res.periods[-1] + pd.Timedelta(days=6)


def test_city_level():
    geo = pd.DataFrame({"geo": [f"City {i}" for i in range(15)], "state": [f"S{i % 4}" for i in range(15)],
                        "population": np.linspace(1e5, 2e6, 15)})
    cfg = make_config(geo_grain="city", geography=GeographyConfig(custom_geography=geo), lift=LiftConfig(lift=0.1))
    res = GeoSimulator(cfg).run()
    assert "city" in res.data.columns and res.data["city"].nunique() == 15


def test_state_population_override():
    geo_cfg = GeographyConfig(custom_geography=toy_geography(), state_population={"State 00": 1e9})
    res = GeoSimulator(make_config(geography=geo_cfg)).run()
    pops = res.geography.groupby("state")["population"].sum()
    assert pops["State 00"] == pytest.approx(1e9)
    assert pops.idxmax() == "State 00"


def test_aggregate_to_state(result):
    agg = result.aggregate("state")
    assert len(agg) == 12 * result.config.n_periods
    assert agg["conversions"].sum() == result.observed.sum()
    treated_rows = agg[agg["state"].isin(result.treatment_geos)]
    assert (treated_rows["treated_share"] == 1).all()


# ---------------------------------------------------------------------------
# Common structure (requirement: regions must be cointegrated)
# ---------------------------------------------------------------------------


def test_regions_are_cointegrated_by_default():
    cfg = make_config(n_pre_periods=200, daily_conversions=500_000)
    res = GeoSimulator(cfg).run()
    report = cointegration_report(res.aggregate("state"))
    assert report["cointegrated"].mean() >= 0.8


def test_geo_trends_break_common_structure():
    good = make_config(n_pre_periods=200, daily_conversions=500_000)
    bad = dataclasses.replace(good, baseline=BaselineConfig(geo_trend_sd=0.6, idio_ar=0.95))
    fit_good = pre_period_fit(GeoSimulator(good).run().aggregate("state"), ["State 03"])
    fit_bad = pre_period_fit(GeoSimulator(bad).run().aggregate("state"), ["State 03"])
    assert fit_good["r2"] > fit_bad["r2"]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [
    dict(granularity="month"), dict(geo_grain="county"), dict(contamination=ContaminationConfig(share=1.5)), dict(n_test_periods=0),
    dict(daily_conversions=0), dict(lift=LiftConfig(lift=-1.0)), dict(lift=LiftConfig(shape="linear")),
])
def test_invalid_config_raises(bad):
    with pytest.raises(ValueError):
        GeoSimulator(make_config(**bad))


# ---------------------------------------------------------------------------
# Grains: aggregation, clusters
# ---------------------------------------------------------------------------


def test_state_grain_from_custom_units():
    cfg = make_config(geo_grain="state", geography=GeographyConfig(custom_geography=toy_states()),
                      lift=LiftConfig(lift=0.05))
    res = GeoSimulator(cfg).run()
    assert list(res.data.columns[:3]) == ["date", "state", "conversions"]
    assert res.data["state"].nunique() == 12


def test_cluster_membership_aggregates_without_losing_population():
    zips = toy_geography()
    membership = cluster_membership(zips, n_clusters=6, seed=0)
    assert set(membership["zip"]) == set(zips["geo"]) and membership["geo"].nunique() == 6
    units = aggregate_geography(zips, membership)
    assert units["population"].sum() == pytest.approx(zips["population"].sum())
    assert {"latitude", "longitude"} <= set(units.columns)
    # Fixed seed -> identical clusters.
    pd.testing.assert_frame_equal(membership, cluster_membership(zips, n_clusters=6, seed=0))


# ---------------------------------------------------------------------------
# Contamination
# ---------------------------------------------------------------------------


def contaminated(share, **kw):
    cfg = make_config(geo_grain="state", geography=GeographyConfig(custom_geography=toy_states()),
                      lift=LiftConfig(lift=0.10), noise=NoiseConfig(distribution="none"),
                      treatment=TreatmentConfig(treatment_geos=["State 05"]),
                      contamination=ContaminationConfig(share=share, **kw))
    return GeoSimulator(cfg).run()


def test_contamination_conserves_total_effect():
    res = contaminated(0.3)
    np.testing.assert_allclose((res.direct + res.spill_in).sum(), res.effect.sum())
    truth = res.ground_truth()
    assert truth["total_lift"] == pytest.approx(0.10)
    assert truth["total_incremental_conversions"] == pytest.approx(truth["expected_incremental_conversions"])


def test_contamination_moves_effect_to_control_neighbours():
    truth = contaminated(0.3).ground_truth()
    # One treated state without treated neighbours: exactly 30% of the effect leaves.
    assert truth["expected_treated_lift"] == pytest.approx(0.07)
    assert truth["realized_lift"] == pytest.approx(0.07)
    assert truth["spillover_to_controls"] > 0 and truth["control_contamination"] > 0
    assert truth["identifiable_lift"] < truth["realized_lift"]


def test_contamination_only_reaches_nearest_units():
    res = contaminated(0.3, n_neighbors=2)
    received = res.spill_in.sum(axis=1) > 0
    assert received.sum() == 2 and not received[res.treated_unit >= 0].any()


def test_contamination_respects_max_distance():
    truth = contaminated(0.3, max_distance_km=1.0).ground_truth()
    assert truth["spillover_to_controls"] == 0
    assert truth["expected_treated_lift"] == pytest.approx(0.10)


def test_contamination_keeps_counterfactual(result):
    other = result.with_treatment(contamination=0.5)
    np.testing.assert_array_equal(result.counterfactual, other.counterfactual)


def test_contamination_needs_coordinates():
    geo = toy_geography().drop(columns=["latitude", "longitude"])
    cfg = make_config(geography=GeographyConfig(custom_geography=geo), lift=LiftConfig(lift=0.1),
                      contamination=ContaminationConfig(share=0.2))
    with pytest.raises(ValueError, match="coordinates"):
        GeoSimulator(cfg).run()


# ---------------------------------------------------------------------------
# Calibration from history
# ---------------------------------------------------------------------------


TRUE_BASELINE = BaselineConfig(market_volatility=0.006, n_factors=1, factor_volatility=0.004,
                               idio_sd=0.05, idio_ar=0.6, state_shock_sd=0.0,
                               weekly_pattern=(1.1, 1.05, 1.0, 1.0, 1.1, 0.9, 0.85))


@pytest.fixture(scope="module")
def history():
    geo = toy_geography(n_states=40, zips_per_state=1, seed=3)
    cfg = make_config(geo_grain="city", geography=GeographyConfig(custom_geography=geo), n_pre_periods=400,
                      n_test_periods=1, daily_conversions=100_000, seed=11, baseline=TRUE_BASELINE,
                      noise=NoiseConfig(overdispersion=1.0, nb_concentration=40))
    res = GeoSimulator(cfg).run()
    return res.aggregate("city", include_truth=False).rename(columns={"city": "geo"})


def test_calibration_recovers_parameters(history):
    cal = calibrate(CalibrationConfig(history))
    b = cal.baseline
    assert cal.granularity == "day"
    assert b.idio_sd == pytest.approx(0.05, rel=0.25)
    assert b.idio_ar == pytest.approx(0.6, abs=0.15)
    assert b.market_volatility == pytest.approx(0.006, rel=0.3)
    assert b.factor_volatility * b.loading_sd == pytest.approx(0.004 * 0.5, rel=0.5)
    assert cal.noise.overdispersion == pytest.approx(1.0, abs=0.5)
    assert cal.noise.nb_concentration == pytest.approx(40, rel=0.4)
    expected_dow = np.array(TRUE_BASELINE.weekly_pattern) / np.mean(TRUE_BASELINE.weekly_pattern)
    np.testing.assert_allclose(b.weekly_pattern, expected_dow, atol=0.02)
    assert cal.report.loc["yearly_amplitude", "source"] == "estimated"


def test_simulation_uses_calibrated_baseline(history):
    cfg = make_config(calibration=CalibrationConfig(history), lift=LiftConfig(lift=0.05))
    res = GeoSimulator(cfg).run()
    assert res.calibration is not None
    assert res.config.baseline.idio_sd == pytest.approx(res.calibration.baseline.idio_sd)
    again = res.with_treatment(lift=0.0)
    np.testing.assert_array_equal(res.counterfactual, again.counterfactual)


def test_calibration_rejects_short_history(history):
    with pytest.raises(ValueError, match="Calibration needs"):
        calibrate(CalibrationConfig(history[history["date"] < history["date"].min() + pd.Timedelta(days=20)]))


def test_count_noise_is_grain_consistent():
    """Zips summed to states vs. states simulated directly: similar noise."""
    zips_cfg = make_config(n_pre_periods=120, baseline=BaselineConfig(n_factors=0, idio_sd=0.0, state_shock_sd=0.0,
                                                                     market_volatility=0.0, yearly_amplitude=0.0))
    states_cfg = dataclasses.replace(zips_cfg, geo_grain="state", geography=GeographyConfig(custom_geography=toy_states()))

    def cv(res):
        w = res.aggregate("state").pivot(index="date", columns="state", values="conversions")
        return float((w.std() / w.mean()).mean())

    assert cv(GeoSimulator(zips_cfg).run()) == pytest.approx(cv(GeoSimulator(states_cfg).run()), rel=0.25)


def test_noise_granularity_conversion():
    nb = NoiseConfig(nb_concentration=10)
    assert noise_for_granularity(nb, "day", "week").nb_concentration == pytest.approx(70)
    assert noise_for_granularity(nb, "week", "day").nb_concentration == pytest.approx(10 / 7)
    assert noise_for_granularity(nb, "day", "day").nb_concentration == 10


# ---------------------------------------------------------------------------
# GeoNames (network)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not os.environ.get("GEO_SIM_NETWORK"), reason="set GEO_SIM_NETWORK=1 to download GeoNames")
def test_geonames_us_states_and_clusters():
    from geo_simulator.geography import build_geography, zip_geography

    zips = zip_geography("US", GeographyConfig())
    states = zips.groupby("state")["population"].sum()
    assert states["California"] == pytest.approx(39.4e6, rel=0.05)
    assert states["New York"] == pytest.approx(19.9e6, rel=0.05)
    assert zips["geo"].str.len().eq(5).all()
    _, clusters, membership = build_geography("US", "cluster", GeographyConfig(), n_clusters=100)
    assert len(clusters) == 100 and set(membership["zip"]) == set(zips["geo"])
    assert clusters["population"].sum() == pytest.approx(zips["population"].sum())


@pytest.mark.skipif(not os.environ.get("GEO_SIM_NETWORK"), reason="set GEO_SIM_NETWORK=1 to download GeoNames")
@pytest.mark.parametrize("country, divisions, low, high", [
    ("AR", 24, 40e6, 50e6),   # provinces sharing postal codes must not merge; CABA has no postal codes
    ("CL", 16, 16e6, 19e6),   # Ñuble (created 2018) has no postal codes
    ("MX", 32, 120e6, 130e6),
    ("CA", 13, 35e6, 40e6),
])
def test_geonames_regional_coverage(country, divisions, low, high):
    from geo_simulator.geography import zip_geography

    zips = zip_geography(country, GeographyConfig())
    assert zips["state"].nunique() == divisions
    assert low < zips["population"].sum() < high
