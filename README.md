# geo_simulator

Simulate **geo-experiment conversion data with a known lift**: historical (pre-period), test and optional cooldown periods, for a whole country at the **geo grain you choose** — zip codes, DMA-like clusters of zips, states or the largest cities — ready to export to CSV and to analyse with GeoLift, CausalImpact, Meridian GeoX or any other geo-measurement tool. The effect can **bleed into neighbouring units** (contamination), and the baseline can be **estimated from your own historical data**.

The companion to [`meridian_simulator`](../meridian_simulator): that package simulates MMM data with a known ROI; this one simulates geo tests with a known incremental effect.

```python
from geo_simulator import ContaminationConfig, GeoSimulator, LiftConfig, SimulationConfig

cfg = SimulationConfig(
    country="US",                 # ISO code or name
    geo_grain="cluster",          # "zip" | "cluster" (DMA-like) | "state" | "city"
    granularity="day",            # or "week" (weeks start on Monday)
    n_pre_periods=90, n_test_periods=28, n_cooldown_periods=7,
    daily_conversions=20_000,
    lift=LiftConfig(lift=0.05, shape="increasing"),        # +5% over the test, ramping up
    contamination=ContaminationConfig(share=0.2),          # 20% of the effect bleeds to neighbours
    # calibration=CalibrationConfig(history_df),           # estimate the baseline from real data
    seed=42,
)
result = GeoSimulator(cfg).run()
result.summary()
result.to_csv("geo_data.csv", minimal=True)   # date, cluster, state, conversions
truth = result.ground_truth()                 # what a correct analysis recovers
```

```text
Country US | grain=cluster | granularity=day | seed=42
Units: 200 (cluster) in 48 states | periods: 90 pre + 28 test + 7 cooldown
Dates: 2024-01-01 -> 2024-05-04 | test 2024-03-31 -> 2024-04-27
Treatment (geo): ['Baton Rouge, Louisiana', 'Dayton, Ohio', 'Hattiesburg, Mississippi'] = 3 units, 1.0% of pre-period conversions
Lift: configured +5.00% (increasing) | realized in treated +3.90% | incremental +227
Contamination: share 20% -> +68 conversions in controls (+0.01%) | identifiable lift ~ +3.89%
```

## Install

```bash
pip install -e .            # from this folder
pip install -e ".[dev]"     # + pytest
```

The first run for a country downloads GeoNames files (postal codes, `cities1000`, the country dump for state populations; 1–70 MB) into `~/.cache/geo_simulator`. Pass `GeographyConfig(custom_geography=df)` to work offline.

## Features

| Requirement | How |
|---|---|
| Country, pre / test / cooldown periods | `SimulationConfig(country, n_pre_periods, n_test_periods, n_cooldown_periods=0)` |
| Daily conversions | `daily_conversions`: the expected national total per day, averaged over the pre-period |
| Geo grain (the unit that is simulated, analysed and treated) | `geo_grain="zip"` (zips with population data), `"cluster"` (`n_clusters` DMA-like groups, or your own zip→geo mapping), `"state"`, `"city"` (`n_cities`, one per metro) |
| Days or weeks (weeks start Monday) | `granularity="day" \| "week"`; a non-Monday `start_date` is moved back to its Monday |
| States and zip codes per country, proportional to population | GeoNames postal codes + census-based state populations (see [Geography](#geography)) |
| DataFrame → CSV | `result.data`, `result.to_csv(path, minimal=True)`; `result.membership` maps zips to units |
| Seed | `seed`; independent seed streams per component |
| Output: date, unit, state | Plus truth columns: `counterfactual_conversions`, `incremental_conversions`, `expected_change`, `treated`, `period` |
| Constant or increasing lift; positive, zero or negative | `LiftConfig(lift, shape="constant" \| "increasing", ramp_periods)` |
| Common, cointegrated structure across units | Common-factor DGP at the chosen grain (see [Model](#model)); verified with `diagnostics.cointegration_report` |
| Contamination | `ContaminationConfig(share, n_neighbors, bandwidth_km, max_distance_km)` (see [Contamination](#contamination)) |
| Baseline from history, else generated | `CalibrationConfig(history_df)` (see [Calibration](#calibration-from-history)) |

Extras:

* **Design first, then inject.** `result.with_treatment(geos, lift=..., contamination=...)` re-runs with a new treatment group, lift or contamination on the *identical* counterfactual. Design the test on the pre-period of a `lift=0` simulation (e.g. with GeoX or GeoLift market selection), then inject the lift into the chosen units.
* **Treatment** of whole states (`treatment_level="state"`, the default for zips) or individual units (`"geo"`, the default otherwise); explicit (`treatment_geos=[...]`) or random (`n_treatment_geos`, skipping the largest and smallest 10%).
* **Cooldown** with no effect (default) or geometric carry-over (`cooldown="decay"`).
* **Heterogeneous lift** across treated units (`geo_heterogeneity`), rescaled so the aggregate stays exact.
* **Stress-test knobs** that break the synthetic-control assumptions on purpose: `geo_trend_sd`, `loading_sd`, `propensity_sd`, `idio_ar`.
* **Diagnostics:** `cointegration_report`, `pre_period_fit` (TBR-style R², MAPE, residual ADF), `sparsity_report`.

## Model

### Counterfactual (no intervention)

The model runs **at the chosen grain**. For unit *g* in state *s(g)* on day *t*:

```
log μ[g,t] = log share[g] + M[t] + Σ_k λ[g,k] F[k,t] + γ[g] S[t] + log W[dow(t)] + v[s(g),t] + u[g,t]  (+ c)
```

| Term | Meaning | Default |
|---|---|---|
| `share[g]` | Population share of the unit | GeoNames |
| `M[t]` | Market factor: 3%/year growth + Gaussian random walk (I(1)), loading 1 for everyone | σ = 0.004/day |
| `F[k,t]`, `λ[g,k]` | 2 more random-walk factors, unit loadings ~ N(0, 0.5) | σ = 0.003/day |
| `S[t]`, `γ[g]` | Yearly seasonality (two harmonics), loadings ~ N(1, 0.1) | amplitude 0.08 |
| `W` | Day-of-week pattern, shared | Mon…Sun |
| `v[s,t]` | **Stationary** AR(1) shock shared by the units of a state | sd 0.02, ρ 0.7 |
| `u[g,t]` | **Stationary** AR(1) unit noise | sd 0.03, ρ 0.5 |
| `c` | Scale so the pre-period national mean equals `daily_conversions` | |

Every non-stationary component (`M`, `F`) is common, and the state and unit terms (`v`, `u`) are stationary. So the units' log-intensities are **cointegrated**, which is the condition under which synthetic control / TBR / BSTS counterfactuals are consistent ([Masini & Medeiros 2021](https://anson.ucdavis.edu/~rmasini/files/papers/MM-2021-JASA.pdf)). This is the interactive-fixed-effects ("factor") model that underpins the synthetic control literature ([Abadie, Diamond & Hainmueller 2010](https://www.law.upenn.edu/live/files/8950-abadie2010pdf); [Xu 2017](https://github.com/xuyiqing/gsynth)). The state shock makes units of the same state more alike than units of different states — so neighbouring zips are good but not perfect controls.

Weekly data sums the daily intensities over Monday–Sunday.

**Count noise** (`NoiseConfig`) is gamma-Poisson with variance `μ (1 + overdispersion) + μ² / nb_concentration`. The `overdispersion` term (default 1, i.e. twice the Poisson variance) is **grain-consistent**: zips summed to a state have the same noise as a state simulated directly. The optional `nb_concentration` term adds size-independent noise (a classic negative binomial) and is not grain-consistent.

### Treatment effect

For treated unit *i* and test period *t*, the campaign creates `ℓ[i,t] · μ[i,t]` expected incremental conversions:

* **constant:** `ℓ[i,t] = lift`.
* **increasing:** `ℓ[i,t] = p · min(1, (t+1)/ramp_periods)`. The plateau `p` is solved so that Σ ℓμ / Σ μ over the test equals `lift` exactly, so the plateau is above `lift`.

Observed and counterfactual counts are **coupled**: with gamma intensity λ, `Y0 ~ Poisson(λ)`; then `Y1 = Y0 + Poisson(cλ)` for a positive relative change `c` and `Y1 = Binomial(Y0, 1+c)` for a negative one. Both give `Y1 ~ Poisson((1+c)λ)`, so every row's incremental conversions are known exactly.

### Contamination

`ContaminationConfig(share=ρ)` moves a share ρ of each treated unit's daily expected incremental to its `n_neighbors` nearest units (centroid distance, weights `exp(−d / bandwidth_km)` normalised to 1, nothing beyond `max_distance_km` — Honolulu doesn't bleed into the mainland). This is "option A": the **total** effect is conserved; part of it lands outside the treated units (people across the border who see the campaign and buy there). Spillover into other treated units stays in the treatment group; spillover into controls contaminates the counterfactual. At zip grain with state-level treatment, most of the bleed stays inside the treated states and only border zips leak — as in reality.

### Ground truth

`ground_truth()` returns, for the test (and `cooldown_*` for the cooldown):

| Key | Meaning |
|---|---|
| `total_lift` | Total effect created / treated counterfactual. Equals the configured lift, whatever the contamination. |
| `expected_treated_lift` | Part of the effect that lands in treated units / their counterfactual. |
| `realized_lift` | Observed / counterfactual − 1 of the treated units on the actual draws — what an estimator with a perfect counterfactual reports. |
| `control_contamination` | Observed / counterfactual − 1 of the control units. |
| `identifiable_lift` | ≈ (1 + realized) / (1 + contamination) − 1: roughly what a method weighting all controls equally can recover. Methods that weight nearby controls more can do worse. |
| `incremental_conversions`, `spillover_to_controls`, `total_incremental_conversions` | Realised counts. |

## Calibration from history

Pass a historical panel (`date`, `geo`, `conversions`) and the baseline is estimated from it instead of using the defaults; without one, the defaults are used.

```python
from geo_simulator import CalibrationConfig, calibrate
cal = calibrate(CalibrationConfig(history, geo_col="location", value_col="Y"))
cal.report        # value, source (estimated / configured / assumed), note
cfg = SimulationConfig(..., calibration=CalibrationConfig(history, geo_col="location", value_col="Y"))
```

| Parameter | How it is estimated |
|---|---|
| `weekly_pattern` | Day-of-week means of the demeaned log panel (daily history) |
| `annual_growth`, `yearly_amplitude` | Regression of the cross-sectional mean on a trend (+ a yearly harmonic if ≥ 1 year of history). Growth is flagged as weakly identified over short histories. |
| `market_volatility` | Variance ratio of the detrended market path, which removes stationary noise riding on the random walk |
| `factor_volatility` (`n_factors`) | Pooled variogram of unit residuals `D(h) = 2σ²(1−ρʰ) + 2ω² + C·h`: `C` is the rate at which units drift apart (as one equivalent factor) |
| `idio_sd`, `idio_ar` | Same variogram fit (σ², ρ); converted from weekly to daily if needed. Flagged when ρ hits its bound (persistence and divergence not separable). |
| `overdispersion`, `nb_concentration` | Each unit's white variance regressed on 1/mean: `(1 + overdispersion)/μ + 1/r` |
| `state_shock_sd`, `geo_trend_sd` | Not identifiable from a geo-only panel / not applied (would break cointegration); reported |

Recovery on simulated data with known parameters (daily, 200 clusters, 500 days; estimated vs. true): day-of-week within 0.005, market volatility 0.0059 vs 0.006, divergence 0.0047 vs 0.004, unit noise 0.051 / 0.585 vs 0.05 / 0.6, overdispersion 1.11 vs 1.0, `nb_concentration` 50.3 vs 50, yearly amplitude 0.106 vs 0.12. Annual growth is not recovered from 1.4 years (−2.3% vs +5%) because the random walk dominates the trend — the report flags it. Weekly histories recover the count noise but not the split between persistence and divergence. On GeoLift's sample panel (40 cities, 90 days) it finds a strong weekend pattern (Sunday +47%), unit noise ≈ 0.15 (5× the default) and no measurable divergence — real data is noisier than the defaults.

## Geography

GeoNames ([CC-BY 4.0](https://www.geonames.org/)) has postal codes for ~100 countries but **no population per postal code**. The simulator builds it as follows:

1. **Postal codes → state** from `export/zip/{CC}.zip`. Duplicate codes (one row per locality, e.g. Mexico) are collapsed. State labels written under two naming schemes (e.g. Germany's "Bayern"/BY vs. "Bavaria"/02) are merged. Military codes without a state (US APO/FPO) are dropped.
2. **Population per zip:** each place with ≥1,000 inhabitants (`cities1000`, city districts excluded) is assigned to its nearest postal code **within its own state**. **Only zips that receive a place are kept** — the others have no population evidence (US: ~14k of ~41k zips).
3. **State totals** are rescaled to GeoNames' census-based ADM1 populations (`export/dump/{CC}.zip`), matched by name, alternate names or code. This removes double counting (New York City *and* its boroughs) and folds rural population into the nearest town's zip. If the ADM1 figure is missing or implausible (< 30% of the city-based sum; e.g. Hidalgo, MX is listed at 0.3M), the city-based sum is kept. `geography["population_source"]` records which source was used.
4. States with < 0.01% of the population are dropped (`min_state_share`).
5. **Coarser grains aggregate zips — they never truncate them.** `state`: sum of the state's zips. `cluster`: population-weighted k-means of zip centroids (labelled by their largest city, e.g. "New York City, New York"; fixed across simulation seeds via `cluster_seed`), or your own zip→geo mapping (`GeographyConfig(custom_clusters=df)`, e.g. Nielsen DMAs). A unit's `state` is where most of its population lives; its centroid is population-weighted.

Checked across US, MX, DE, ES, FR, GB, BR, IN and CO: full first-level coverage and no double counting (CA 39.4M, NY 19.9M, CDMX 9.2M, 16 Länder / 83M). GeoNames' state totals have **mixed vintages** — Texas (22.9M vs. 29.1M in the 2020 census) and Florida (17.7M vs. 21.5M) are a decade old — so pass official figures with `GeographyConfig(state_population={...})` when exact state sizes matter.

### Regional coverage: Latin America and Canada

| Country | Units with population | Divisions | Population vs. census | Notes |
|---|---:|---:|---|---|
| Mexico | 7,044 | 32 / 32 | 124.8M vs 126.0M (2020) | Hidalgo uses its city-based total (GeoNames figure is wrong) |
| Argentina | 836 | 24 / 24 | 45.5M vs 46.0M (2022) | shared 4-digit codes; CABA has no codes → place-based unit |
| Colombia | 988 | 33 / 33 | 49.3M vs 48.3M (2018) | Bogotá D.C. has only 2 postal codes → use cluster/department grain |
| Peru | 945 | 25 / 26 | 28.6M vs 31.2M (2017) | older GeoNames totals; Lima Province folded into Lima |
| Chile | 224 | 16 / 16 | 17.1M vs 17.6M (2017) | Ñuble (2018) has no codes → 16 place-based units |
| Canada | 838 | 13 / 13 | 37.1M vs 37.0M (2021) | forward sortation areas; `postal_file="CA_full"` for full codes |

Divisions with populated places but no postal codes become place-based units (`{CC}-{admin1}-{rank}`), and postal codes shared across provinces never merge them. The whitepaper (Section 10) and the deck report maps and a common-design study for each market: with ~15% of conversions treated for four weeks, the approximate minimum detectable lift is 3.1 pp in Mexico, 4.6 in Argentina, 4.9 in Colombia, 5.8 in Canada, 6.1 in Peru and 8.3 in Chile.

City models use the `n_cities` most populated places in `cities1000`, picked greedily from the largest and skipping any place within `city_min_distance_km` (25 km) of a city already picked. Each metro area therefore appears once (Brooklyn is not a separate geo from New York City). The metro is labelled and sized by its largest municipality (e.g. Zapopan for Guadalajara), not by its full agglomeration.

## Literature and existing tools

| Source | What it gives us |
|---|---|
| Abadie, Diamond & Hainmueller (2010), *Synthetic Control Methods for Comparative Case Studies*, JASA | The linear factor model that justifies synthetic control. The bias bound shrinks with pre-period length relative to noise, and the treated unit must lie near the convex hull of the controls. |
| Xu (2017), *Generalized Synthetic Control*, Political Analysis ([gsynth](https://github.com/xuyiqing/gsynth)) | Interactive fixed effects; GeoLift's GSC option. Simulation designs draw unit intercepts and factors from distributions, as we do. |
| Ben-Michael, Feller & Rothstein (2021), *The Augmented Synthetic Control Method*, JASA ([augsynth](https://github.com/ebenmichael/augsynth)) | Ridge bias-correction when the pre-period fit is imperfect. This is GeoLift's engine. |
| Chernozhukov, Wüthrich & Zhu (2021), *An Exact and Robust Conformal Inference Method for Counterfactual and Synthetic Controls*, JASA | GeoLift's conformal intervals. |
| Masini & Medeiros (2021, 2022), JASA / JBES | Counterfactual estimators need a **cointegrating** relationship between treated and donors; without it they diverge whatever the true effect. This is why requirement 9 matters. |
| Brodersen et al. (2015), *Inferring causal impact using Bayesian structural time-series*, AoAS ([tfp-causalimpact](https://github.com/google/tfp-causalimpact)) | CausalImpact (BSTS with spike-and-slab regression on control series). |
| Kerman, Wang & Vaver (2017), *Estimating ad effectiveness using geo experiments in a time-based regression framework* ([matched_markets](https://github.com/google/matched_markets)) | TBR, the methodology behind [Meridian GeoX](https://github.com/google/meridian-geox). |
| Chen & Au (2021), *Trimmed Match Design for Randomized Paired Geo Experiments* ([trimmed_match](https://github.com/google/trimmed_match)) | Simulates 100 geos × 6 weeks × 5,000 replicates; emphasises geo heterogeneity and the drift of treated/control comparability during the test. |
| Abadie & Zhao (2021), *Synthetic Controls for Experimental Design* | Choosing treated units *with* synthetic controls, the design-then-inject workflow. |
| [Recast (2025), *Open-Source Geo-Experiment Tools — A Head-to-Head Simulation Study*](https://research.getrecast.com/geolift-sim-study) ([code](https://github.com/getrecast/geolift-simulation-study)) | The closest prior work: 32,000 simulated panels, multiplicative model with shared trend and day-of-week, AR(1) noise. GeoLift: ~93–95% coverage but low power. CausalImpact: most power but ~70% coverage and 28–30% false positives. Their stated limitations (one DGP, no geo-specific trend violations, a single effect size, no zip geography) are what this package adds knobs for. |
| GeoLift's `GeoLift_Test` data (40 US cities × 105 days) | The de-facto benchmark dataset. Real-looking, but the true lift is unknown. We use its pre-period to demonstrate calibration. |
| Arkhangelsky et al. (2021), *Synthetic Difference-in-Differences*, AER | Placebo simulations whose DGP is *estimated from real panels* — the idea behind `CalibrationConfig`. |

**What already exists and what didn't:** GeoLift, GeoX, matched_markets and trimmed_match have power/MDE *simulations* that inject synthetic effects into **your own historical data**. Recast's study generates panels from scratch, but for abstract geos. There was no package that generates a full country's zip / DMA-like / state geography with population-proportional volume, a configurable effect shape, a cooldown, spillover into neighbouring units, a baseline estimated from your own history, and an exact per-row ground truth.

## Limitations

See also Part 9 of the notebook.

1. **The default DGP favours synthetic controls.** Cointegration holds by construction, so with defaults simulated accuracy is an *upper bound*. Calibrate from your own history and use the stress-test knobs.
2. **Calibration is grain-specific and partial.** Unit noise is estimated at the history's grain and carries over imperfectly to other grains; short histories don't identify the annual trend or yearly seasonality; weekly histories can't separate persistence from divergence (flagged). State shocks and unit trends are not estimated/applied.
3. **Contamination is distance-based**: same-day spillover to the nearest units at a fixed share. Real spillover follows commuting and media markets, may be lagged, and differs by channel.
4. **The lift is a multiplier, not a media model.** No spend, saturation or adstock; iROAS needs an assumed spend.
5. **No calendar effects** unless you add shared `shocks`. No local events.
6. **Zip populations are estimated** (nearest-zip assignment of `cities1000`, rescaled to ADM1); rural population sits in the nearest town's zip.
7. **Clusters are DMA-like, not DMAs** (k-means). Pass real DMAs via `custom_clusters`.
8. **Postal coverage varies** (GB/CA/NL use outward codes by default; Brazil is sparse; some countries are missing). For production geography, pass official data with `custom_geography` / `state_population`.
9. **Strict population proportionality** is the default. Real propensity varies (`propensity_sd`).
10. **Zip grain is sparse.** At 20k conversions/day most zip-days are zero; analyse at cluster/state level, or use `granularity="week"`.
11. **Reproducibility needs the same geography snapshot.** GeoNames changes daily. Persist `result.geography` (+ `result.membership`) and pass it back via `custom_geography`.
12. **"Increasing" is a linear ramp** to a plateau above the target. The target is the cumulative, baseline-weighted lift.

## Notebook

[`notebooks/Geo_Simulator_GeoLift_CausalImpact_GeoX.ipynb`](notebooks/Geo_Simulator_GeoLift_CausalImpact_GeoX.ipynb) (Colab-ready) calibrates the baseline from GeoLift's sample panel, simulates US DMA-like clusters, checks cointegration, designs the test with Meridian GeoX, injects the lift, measures it with GeoX, CausalImpact and GeoLift, runs a Monte Carlo (bias, coverage, false positives, power), sweeps contamination, and runs a stress test. Regenerate it with `python notebooks/build_notebook.py`; `..._with_outputs.ipynb` is an executed copy.

## Documentation

| File | What |
|---|---|
| [`presentation/Geo_Simulator_Overview.pptx`](presentation/Geo_Simulator_Overview.pptx) | 37-slide deck in four modules (business need, features incl. Latin America & Canada, using the code, summary) with speaker notes. Rebuild with `python presentation/build_deck.py`. |
| [`whitepaper/geo_simulator_whitepaper.pdf`](whitepaper/geo_simulator_whitepaper.pdf) | Whitepaper: the DGP's mathematics, propositions and proofs (cointegration, aggregation, exact lift, coupling, conservation), calibration estimators, validation. |
| [`whitepaper/geo_simulator_whitepaper.Rmd`](whitepaper/geo_simulator_whitepaper.Rmd) | Knit-ready source (`rmarkdown::render(...)`; needs `rmarkdown`, `knitr`, `stevetemplates` and a LaTeX install). Figures are pre-rendered in `whitepaper/figs`. |
| [`whitepaper/make_figures.py`](whitepaper/make_figures.py) | Regenerates every figure and number in the whitepaper (`results.json`). |

## Tests

```bash
pytest tests/                          # offline
GEO_SIM_NETWORK=1 pytest tests/        # + GeoNames download check
```
