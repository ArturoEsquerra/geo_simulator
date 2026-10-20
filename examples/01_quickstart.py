"""Example 1 — Quickstart: simulate a US geo test and inspect the ground truth.

Run:  python examples/01_quickstart.py
"""

import numpy as np

from geo_simulator import GeoSimulator, LiftConfig, SimulationConfig
from geo_simulator.diagnostics import cointegration_report, pre_period_fit

cfg = SimulationConfig(
    country="US",
    geo_grain="cluster",       # ~200 DMA-like zip clusters ("zip", "state", "city" also work)
    granularity="day",
    n_pre_periods=90,
    n_test_periods=28,
    n_cooldown_periods=7,
    daily_conversions=20_000,
    lift=LiftConfig(lift=0.05, shape="increasing"),
    seed=42,
)

result = GeoSimulator(cfg).run()
result.summary()

print("\nUnit-level output:")
print(result.data.head())

units = result.aggregate("cluster")   # result.aggregate("state") rolls the clusters up to states
print("\nCointegrated clusters:", f"{cointegration_report(units, unit='cluster')['cointegrated'].mean():.0%}")
print("Pre-period fit of the treatment group:", pre_period_fit(units, result.treatment_geos, unit="cluster"))

truth = result.ground_truth()
print(f"\nTruth: realized lift {truth['realized_lift']:+.2%}, incremental {truth['incremental_conversions']:+,.0f}")

result.to_csv("geo_quickstart.csv", minimal=True)
print("Wrote geo_quickstart.csv (date, cluster, state, conversions)")

# Reproducibility: same config and geography -> identical data.
again = GeoSimulator(cfg, geography=result.geography).run()
assert np.array_equal(result.observed, again.observed)
print("Reproducibility verified.")
