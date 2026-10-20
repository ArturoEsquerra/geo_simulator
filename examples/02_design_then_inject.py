"""Example 2 — Weekly city-level data: design on the pre-period, then inject the lift.

Run:  python examples/02_design_then_inject.py
"""

from geo_simulator import GeoSimulator, LiftConfig, SimulationConfig
from geo_simulator.diagnostics import pre_period_fit

cfg = SimulationConfig(
    country="Mexico",
    geo_grain="city",
    n_cities=30,
    granularity="week",
    start_date="2024-01-01",   # a Monday
    n_pre_periods=52,
    n_test_periods=8,
    n_cooldown_periods=4,
    daily_conversions=5_000,
    lift=LiftConfig(lift=0.0),  # no effect yet: this is the "historical" data
    seed=7,
)
baseline = GeoSimulator(cfg).run()
cities = baseline.aggregate("city")

# Toy design: among pairs of mid-sized cities, pick the pair with the best
# pre-period fit to the remaining cities (real designs would use GeoX or
# GeoLift market selection here).
by_size = baseline.geography.sort_values("population", ascending=False)["geo"].tolist()
mid = by_size[5:15]
candidates = [[a, b] for i, a in enumerate(mid) for b in mid[i + 1:]]
best = max(candidates, key=lambda c: pre_period_fit(cities, c, unit="city")["r2"])
print("Chosen treatment:", best)

# Same counterfactual, now with a -3% effect (e.g. a go-dark test) that decays in the cooldown.
result = baseline.with_treatment(best, lift=LiftConfig(lift=-0.03, cooldown="decay", cooldown_decay=0.5))
result.summary()
print(result.lift_profile().tail(6))
