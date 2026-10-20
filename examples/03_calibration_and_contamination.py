"""Example 3 — Calibrate the baseline from history, then add contamination.

Run:  python examples/03_calibration_and_contamination.py path/to/history.csv
      (columns: date, geo, conversions). Without a path, a simulated history
      stands in for your data.
"""

import sys

import pandas as pd

from geo_simulator import (CalibrationConfig, ContaminationConfig, GeoSimulator, LiftConfig,
                           SimulationConfig, calibrate)

# 1. History: your own panel, or a simulated stand-in.
if len(sys.argv) > 1:
    history = pd.read_csv(sys.argv[1])
else:
    stand_in = GeoSimulator(SimulationConfig(country="US", geo_grain="city", n_cities=40,
                                             n_pre_periods=180, n_test_periods=1, seed=99)).run()
    history = stand_in.aggregate("city", include_truth=False).rename(columns={"city": "geo"})

calibration = CalibrationConfig(history, date_col="date", geo_col="geo", value_col="conversions")
print(calibrate(calibration).report[["value", "source", "note"]], "\n")

# 2. Simulate DMA-like clusters with the calibrated baseline.
cfg = SimulationConfig(
    country="US", geo_grain="cluster", n_clusters=120,
    n_pre_periods=90, n_test_periods=28, daily_conversions=20_000, seed=5,
    lift=LiftConfig(lift=0.06), calibration=calibration,
)
clean = GeoSimulator(cfg).run()
clean.summary()

# 3. Same world and treatment, with 30% of the effect bleeding into the 5 nearest clusters.
leaky = clean.with_treatment(contamination=ContaminationConfig(share=0.3, n_neighbors=5, bandwidth_km=100))
print()
leaky.summary()
truth = leaky.ground_truth()
print(f"\nTotal effect {truth['total_lift']:+.2%} | lands in treated {truth['realized_lift']:+.2%} | "
      f"controls {truth['control_contamination']:+.3%} | identifiable ~ {truth['identifiable_lift']:+.2%}")
