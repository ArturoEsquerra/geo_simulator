"""Geography: states, zip codes, cities and population weights.

Source: GeoNames (https://www.geonames.org, CC-BY 4.0).

* ``export/zip/{CC}.zip``         postal code -> state (admin1), lat/lon.
* ``export/dump/cities1000.zip``  populated places with >= 1,000 inhabitants.
* ``export/dump/admin1CodesASCII.txt``, ``countryInfo.txt``  names.

GeoNames has no population per postal code. Zip weights are built by
assigning every populated place to its nearest postal code of the same state
(great-circle nearest neighbour). Only postal codes that receive at least one
place are kept: the rest have no population evidence. State totals are then
rescaled to GeoNames' first-level (ADM1) population from
``export/dump/{CC}.zip`` when it is available and plausible, which removes
double counting of cities and their districts and folds rural population into
the nearest town's postal code. ``population_source`` says which source set
each state's total.

Coarser grains are built by *aggregating* zips (never by truncating them):

* ``state``   — sum of the state's zips.
* ``cluster`` — population-weighted k-means of zip centroids (a DMA-like
  media-market stand-in), or a user-supplied zip -> geo mapping.
* ``city``    — the largest places of ``cities1000``, one per metro area.

Every unit carries a population-weighted centroid (``latitude``,
``longitude``) used for the contamination neighbourhood matrix.
"""

from __future__ import annotations

import dataclasses
import io
import logging
import os
import pathlib
import unicodedata
import urllib.request
import zipfile
from typing import Optional

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from geo_simulator.config import GeographyConfig


logger = logging.getLogger(__name__)

GEONAMES_URL = "https://download.geonames.org/export"

# Populated-place feature codes kept. Dropped: PPLX (section of a city — would
# double count), PPLH (historical), PPLQ (abandoned), PPLW (destroyed),
# PPLCH (historical capital).
KEEP_FEATURE_CODES = {
    "PPL", "PPLA", "PPLA2", "PPLA3", "PPLA4", "PPLA5", "PPLC", "PPLG",
    "PPLF", "PPLL", "PPLR", "PPLS",
}

_POSTAL_COLUMNS = [
    "country", "postal_code", "place_name", "admin_name1", "admin_code1",
    "admin_name2", "admin_code2", "admin_name3", "admin_code3",
    "latitude", "longitude", "accuracy",
]
_CITY_COLUMNS = [
    "geonameid", "name", "asciiname", "alternatenames", "latitude", "longitude",
    "feature_class", "feature_code", "country_code", "cc2", "admin1_code",
    "admin2_code", "admin3_code", "admin4_code", "population", "elevation",
    "dem", "timezone", "modification_date",
]


# ---------------------------------------------------------------------------
# Downloads and caching
# ---------------------------------------------------------------------------


def _cache_dir(config: GeographyConfig) -> pathlib.Path:
    path = pathlib.Path(config.cache_dir or os.path.join("~", ".cache", "geo_simulator")).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _fetch(relative_url: str, config: GeographyConfig) -> pathlib.Path:
    """Download ``GEONAMES_URL/relative_url`` once and return the local path."""
    target = _cache_dir(config) / relative_url.replace("/", "__")
    if not target.exists():
        url = f"{GEONAMES_URL}/{relative_url}"
        logger.info("Downloading %s", url)
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                payload = response.read()
        except Exception as err:  # noqa: BLE001 — re-raised with context
            raise RuntimeError(
                f"Could not download {url} ({err}). Check your connection, or pass "
                "GeographyConfig(custom_geography=...) to work offline."
            ) from err
        tmp = target.with_suffix(target.suffix + ".part")
        tmp.write_bytes(payload)
        tmp.replace(target)
    return target


def _read_zipped_tsv(path: pathlib.Path, member: str, columns) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        with archive.open(member) as handle:
            return pd.read_csv(
                io.TextIOWrapper(handle, encoding="utf-8"), sep="\t", header=None,
                names=columns, dtype=str, keep_default_na=False, quoting=3,
            )


def resolve_country(country: str, config: GeographyConfig) -> str:
    """Return the ISO alpha-2 code for a code or English country name."""
    code = country.strip()
    if len(code) == 2 and code.isalpha():
        return code.upper()
    # Rows have a variable number of fields, so parse line by line.
    target = _normalize_name(code)
    with open(_fetch("dump/countryInfo.txt", config), encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if not line.startswith("#") and len(fields) > 4 and target in (
                    _normalize_name(fields[4]), _normalize_name(fields[1])):
                return fields[0]
    raise ValueError(f"Unknown country {country!r}. Use an ISO alpha-2 code, e.g. 'US'.")


def load_postal_codes(country_code: str, config: GeographyConfig) -> pd.DataFrame:
    """One row per postal code: ``postal_code, state, latitude, longitude, place_name, aliases``."""
    stem = config.postal_file or country_code
    try:
        path = _fetch(f"zip/{stem}.zip", config)
    except RuntimeError as err:
        raise RuntimeError(
            f"GeoNames has no postal-code file for {country_code!r} ({stem}.zip). "
            "Use level='city' or GeographyConfig(custom_geography=...)."
        ) from err
    member = f"{stem}.txt" if not stem.endswith("_full") else f"{stem}.csv"
    raw = _read_zipped_tsv(path, member, _POSTAL_COLUMNS)
    # Rows without any admin1 (e.g. US military APO/FPO codes) cannot be
    # placed in a state and are dropped.
    raw = raw[(raw["admin_name1"] != "") | (raw["admin_code1"] != "")].copy()
    raw["admin_name1"] = raw["admin_name1"].str.strip()
    raw["state"] = raw["admin_name1"].where(raw["admin_name1"] != "", raw["admin_code1"])
    raw["latitude"] = pd.to_numeric(raw["latitude"], errors="coerce")
    raw["longitude"] = pd.to_numeric(raw["longitude"], errors="coerce")
    # Every name and admin1 code a label is written with; labels that are the
    # same division under two schemes are merged later, through the ADM1 they
    # resolve to (see zip_geography).
    codes = raw.groupby("state")["admin_code1"].agg(lambda c: sorted(set(c) - {""}))
    aliases = {label: sorted({label, *codes.get(label, [])}) for label in raw["state"].unique()}

    # A postal code can appear several times (one row per locality, e.g. MX).
    # Keep one row: the most common state, mean coordinates.
    state = raw.groupby("postal_code")["state"].agg(lambda s: s.value_counts().index[0])
    coords = raw.groupby("postal_code")[["latitude", "longitude"]].mean()
    place = raw.groupby("postal_code")["place_name"].first()
    postal = pd.concat([state, coords, place], axis=1).reset_index()
    postal["aliases"] = postal["state"].map(aliases)
    return postal.sort_values("postal_code").reset_index(drop=True)


def load_admin1(country_code: str, config: GeographyConfig) -> pd.DataFrame:
    """First-level divisions (states) from the GeoNames country dump.

    Columns: ``admin1_code`` (GeoNames scheme, as used by ``cities1000``),
    ``name``, ``alternatenames`` (comma separated, incl. native names and ISO
    codes) and ``population`` (census-based where GeoNames has it).

    Streams ``dump/{CC}.zip`` (tens of MB for large countries) once and caches
    the ADM1 rows as a small CSV.
    """
    cached = _cache_dir(config) / f"adm1__{country_code}.csv"
    if cached.exists():
        return pd.read_csv(cached, dtype=str, keep_default_na=False).astype({"population": float})
    path = _fetch(f"dump/{country_code}.zip", config)
    rows = []
    with zipfile.ZipFile(path) as archive:
        with archive.open(f"{country_code}.txt") as handle:
            for line in io.TextIOWrapper(handle, encoding="utf-8"):
                fields = line.split("\t")
                if len(fields) > 14 and fields[7] == "ADM1":
                    rows.append((fields[10], fields[1], fields[3], float(fields[14] or 0)))
    adm1 = pd.DataFrame(rows, columns=["admin1_code", "name", "alternatenames", "population"])
    adm1 = adm1.sort_values("population", ascending=False).drop_duplicates("admin1_code")
    adm1.to_csv(cached, index=False)
    return adm1


_PREFIXES = ("estado de ", "state of ", "provincia de ", "province of ", "departamento de ", "departamento del ",
             "region de ", "region del ", "land ")
_SUFFIXES = (" province", " provincia", " region", " department", " departamento", " state", " estado")


def _normalize_name(name: str, strip: bool = True) -> str:
    """ASCII, lower case, alphanumerics only; optionally drop administrative
    prefixes/suffixes ("Estado de", "Province", ...)."""
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower().strip()
    if strip:
        for prefix in _PREFIXES:
            if text.startswith(prefix):
                text = text[len(prefix):]
        for suffix in _SUFFIXES:
            if text.endswith(suffix):
                text = text[: -len(suffix)]
    return "".join(ch for ch in text if ch.isalnum())


def _resolve_admin1(postal: pd.DataFrame, adm1: pd.DataFrame) -> dict:
    """Map each postal state label to a GeoNames ADM1 code (or nothing).

    For each label's names, tries in order (keeping only keys that identify a
    single ADM1):

    1. the ADM1's exact name ("Buenos Aires" is the province, not the capital),
    2. its name without administrative prefixes/suffixes ("Salta Province"),
    3. its alternate names (native names, ISO codes such as "BY"),

    then the label's admin1 codes against ADM1 codes.
    """
    def unique_index(pairs):
        index: dict = {}
        for key, code in pairs:
            if key:
                index.setdefault(key, set()).add(code)
        return {k: next(iter(v)) for k, v in index.items() if len(v) == 1}

    rows = list(adm1[["admin1_code", "name", "alternatenames"]].itertuples(index=False))
    exact = unique_index((_normalize_name(n, strip=False), c) for c, n, _ in rows)
    stripped = unique_index((_normalize_name(n), c) for c, n, _ in rows)
    alternates = unique_index((_normalize_name(a), c) for c, n, alts in rows for a in {n, *alts.split(",")} - {""})
    codes = set(adm1["admin1_code"])
    out = {}
    for state, aliases in postal[["state", "aliases"]].drop_duplicates("state").itertuples(index=False):
        names = [a for a in aliases if not (len(a) <= 3 and a.isupper()) and not a.isdigit()]
        short = [a for a in aliases if a not in names]
        hit = None
        for index, strip in ((exact, False), (stripped, True), (alternates, True)):
            hit = next((index[_normalize_name(a, strip)] for a in names if _normalize_name(a, strip) in index), None)
            if hit is not None:
                break
        if hit is None:
            hit = next((a for a in short if a in codes), None)
        if hit is None:
            hit = next((alternates[_normalize_name(a)] for a in short if _normalize_name(a) in alternates), None)
        if hit is not None:
            out[state] = hit
    return out


def load_cities(country_code: str, config: GeographyConfig) -> pd.DataFrame:
    """Populated places of the country: ``name, state, population, lat, lon``."""
    cities = _read_zipped_tsv(_fetch("dump/cities1000.zip", config), "cities1000.txt", _CITY_COLUMNS)
    cities = cities[(cities["country_code"] == country_code)
                    & cities["feature_code"].isin(KEEP_FEATURE_CODES)].copy()
    cities["population"] = pd.to_numeric(cities["population"], errors="coerce").fillna(0)
    cities["latitude"] = pd.to_numeric(cities["latitude"])
    cities["longitude"] = pd.to_numeric(cities["longitude"])

    admin1 = pd.read_csv(
        _fetch("dump/admin1CodesASCII.txt", config), sep="\t", header=None,
        names=["key", "name", "ascii_name", "geonameid"], dtype=str, keep_default_na=False,
    ).set_index("key")["name"]
    keys = country_code + "." + cities["admin1_code"]
    cities["state"] = keys.map(admin1).fillna(cities["admin1_code"]).replace("", "Unknown")
    cities = cities[cities["population"] > 0]
    return cities[["name", "state", "admin1_code", "population", "latitude", "longitude"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Geography tables used by the simulator
# ---------------------------------------------------------------------------


def _unit_vectors(lat, lon) -> np.ndarray:
    lat, lon = np.radians(lat), np.radians(lon)
    return np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def _assign_cities_to_zips(postal, located, city_state, cities, population) -> np.ndarray:
    """Add each city's population to its nearest postal code, in place.

    Returns the name of the largest city assigned to each postal code (empty
    when none), used to label clusters.

    A city goes to the nearest postal code *of its own state* when its state
    is known (``city_state``), otherwise to the nearest postal code overall.
    Restricting to the state stops large metro areas leaking across state
    borders (e.g. Estado de México suburbs into Mexico City postal codes).
    """
    xyz = _unit_vectors(postal["latitude"].fillna(0), postal["longitude"].fillna(0))
    city_xyz = _unit_vectors(cities["latitude"], cities["longitude"])
    city_pop = cities["population"].to_numpy()
    unmatched = city_state.isna().to_numpy().copy()
    target = np.full(len(cities), -1)
    for state, zips in postal.groupby("state").groups.items():
        rows = np.asarray([i for i in zips if located[i]])
        in_state = (city_state == state).to_numpy()
        if not len(rows) or not in_state.any():
            unmatched |= in_state
            continue
        target[in_state] = rows[cKDTree(xyz[rows]).query(city_xyz[in_state])[1]]
    if unmatched.any():
        rows = np.flatnonzero(located)
        target[unmatched] = rows[cKDTree(xyz[rows]).query(city_xyz[unmatched])[1]]
    np.add.at(population, target, city_pop)

    largest = np.full(len(postal), "", dtype=object)
    order = np.argsort(city_pop)  # ascending: the largest city written last wins
    largest[target[order]] = cities["name"].to_numpy()[order]
    return largest


def zip_geography(country_code: str, config: GeographyConfig) -> pd.DataFrame:
    """Zip-level table: ``geo`` (zip), ``state``, ``population``, centroid, ``place_name``.

    Only zips with population evidence (at least one ``cities1000`` place) are
    kept. Divisions that have populated places but no postal codes are
    represented by place-based units with ids ``"{CC}-{admin1}-{rank}"``.
    """
    postal = load_postal_codes(country_code, config)
    cities = load_cities(country_code, config)
    try:
        adm1 = load_admin1(country_code, config)
    except RuntimeError as err:
        logger.warning("ADM1 data unavailable (%s); using city-based state totals.", err)
        adm1 = pd.DataFrame(columns=["admin1_code", "name", "alternatenames", "population"])

    # Labels that resolve to the same ADM1 are one state (e.g. "Bayern" and
    # "Bavaria" in Germany's postal file): keep the label with most zips.
    state_code = _resolve_admin1(postal, adm1)
    zips_per_state = postal["state"].value_counts()
    head = {}
    for state, code in sorted(state_code.items(), key=lambda kv: -zips_per_state[kv[0]]):
        head.setdefault(code, state)
    postal["state"] = postal["state"].map(lambda s: head.get(state_code.get(s), s))
    code_to_state = {code: state for code, state in head.items()}

    # Cities -> postal state through the ADM1 code (same GeoNames scheme);
    # fall back to the city's admin1 name.
    by_name = {_normalize_name(a): head.get(state_code.get(s), s)
               for s, aliases in postal[["state", "aliases"]].itertuples(index=False) for a in aliases}
    city_state = pd.Series([code_to_state.get(code) or by_name.get(_normalize_name(name))
                            for code, name in zip(cities["admin1_code"], cities["state"])],
                           index=cities.index, dtype=object)

    # First-level divisions with populated places but no postal codes at all
    # (Argentina's capital city, Misiones and San Luis; Chile's Ñuble, created
    # after the postal file): their places become units of their own, so their
    # population is kept instead of leaking into a neighbouring state.
    orphan = city_state.isna() & cities["admin1_code"].isin(set(adm1["admin1_code"]) - set(code_to_state))
    extra = cities[orphan].sort_values("population", ascending=False)
    for code, part in extra.groupby("admin1_code"):
        code_to_state[code] = part["state"].iloc[0]
    cities = cities[~orphan]
    city_state = city_state[~orphan]

    population = np.zeros(len(postal))
    located = postal[["latitude", "longitude"]].notna().all(axis=1).to_numpy()
    if not (located.any() and len(cities)):
        raise RuntimeError(f"GeoNames has no coordinates or populated places for {country_code}; "
                           "use GeographyConfig(custom_geography=...).")
    largest_city = _assign_cities_to_zips(postal, located, city_state, cities, population)
    place = np.where(largest_city != "", largest_city, postal["place_name"].to_numpy())
    geo = pd.DataFrame({"geo": postal["postal_code"], "state": postal["state"], "population": population,
                        "latitude": postal["latitude"], "longitude": postal["longitude"],
                        "place_name": place})
    if len(extra):
        rank = extra.groupby("admin1_code").cumcount() + 1
        geo = pd.concat([geo, pd.DataFrame({
            "geo": country_code + "-" + extra["admin1_code"] + "-" + rank.map("{:03d}".format),
            "state": extra["state"], "population": extra["population"], "latitude": extra["latitude"],
            "longitude": extra["longitude"], "place_name": extra["name"]})], ignore_index=True)
        logger.info("%s: %d divisions without postal codes represented by %d place-based units.",
                    country_code, extra["admin1_code"].nunique(), len(extra))
    geo = geo[geo["population"] > 0]

    # State totals: prefer GeoNames' ADM1 population (census-based), which
    # avoids double counting cities and their districts (e.g. New York City
    # and its boroughs, which can roughly double a city-based sum). Fall back
    # to the city-based sum when the ADM1 figure is missing or implausible
    # (< 30% of the city-based sum — a data error, e.g. a missing digit).
    city_based = geo.groupby("state")["population"].sum()
    adm1_pop = adm1.set_index("admin1_code")["population"]
    official = pd.Series({state: adm1_pop[code] for code, state in code_to_state.items()}, dtype=float)
    usable = official[(official > 0) & (official >= 0.3 * city_based.reindex(official.index))]
    geo["population"] *= geo["state"].map(usable / city_based).fillna(1.0).to_numpy()
    geo["population_source"] = np.where(geo["state"].isin(usable.index), "geonames_adm1", "cities1000")
    return finalize_geography(geo, config)


def _one_per_metro(cities: pd.DataFrame, n_cities: int, min_distance_km: float) -> list:
    """Greedy pick, largest first, skipping places near an already picked city."""
    if min_distance_km <= 0:
        return list(cities.index[:n_cities])
    xyz = _unit_vectors(cities["latitude"], cities["longitude"])
    chord = 2 * np.sin(min_distance_km / 6371.0 / 2)  # km -> unit-sphere chord
    picked: list = []
    for i in cities.index:
        if picked and cKDTree(xyz[picked]).query(xyz[i])[0] < chord:
            continue
        picked.append(i)
        if len(picked) == n_cities:
            break
    return picked


def city_geography(country_code: str, n_cities: int, config: GeographyConfig) -> pd.DataFrame:
    """The ``n_cities`` most populated places: ``geo`` (city), ``state``, ``population``."""
    cities = load_cities(country_code, config).sort_values("population", ascending=False).reset_index(drop=True)
    # Disambiguate homonyms ("Springfield") with the state name.
    dup = cities.duplicated("name", keep=False)
    cities.loc[dup, "name"] = cities.loc[dup, "name"] + ", " + cities.loc[dup, "state"]
    cities = cities.drop_duplicates("name").reset_index(drop=True)
    cities = cities.loc[_one_per_metro(cities, n_cities, config.city_min_distance_km)]
    if len(cities) < n_cities:
        logger.warning("Only %d cities available for %s.", len(cities), country_code)
    geo = cities.rename(columns={"name": "geo"})[["geo", "state", "population", "latitude", "longitude"]]
    return finalize_geography(geo, config)


OPTIONAL_COLUMNS = ("latitude", "longitude", "place_name", "population_source")


def finalize_geography(geo: pd.DataFrame, config: GeographyConfig) -> pd.DataFrame:
    """Validate, apply ``state_population`` and ``min_state_share``, sort."""
    missing = {"geo", "state", "population"} - set(geo.columns)
    if missing:
        raise ValueError(f"Geography table is missing columns: {sorted(missing)}")
    keep = ["geo", "state", "population"] + [c for c in OPTIONAL_COLUMNS if c in geo]
    geo = geo[keep].copy()
    geo["geo"] = geo["geo"].astype(str)
    geo["state"] = geo["state"].astype(str)
    geo["population"] = pd.to_numeric(geo["population"]).astype(float)
    if geo["geo"].duplicated().any():
        raise ValueError("Geography table has duplicated 'geo' values.")
    if (geo["population"] < 0).any() or geo["population"].sum() <= 0:
        raise ValueError("Populations must be non-negative with a positive total.")

    if config.state_population:
        unknown = set(config.state_population) - set(geo["state"])
        if unknown:
            raise ValueError(f"state_population has unknown states: {sorted(unknown)}")
        current = geo.groupby("state")["population"].transform("sum")
        target = geo["state"].map(config.state_population)
        rescale = target.notna() & (current > 0)
        geo.loc[rescale, "population"] *= target[rescale] / current[rescale]

    if config.min_state_share > 0:
        share = geo.groupby("state")["population"].transform("sum") / geo["population"].sum()
        dropped = sorted(geo.loc[share < config.min_state_share, "state"].unique())
        if dropped:
            logger.info("Dropping states below min_state_share=%g: %s", config.min_state_share, dropped)
            geo = geo[share >= config.min_state_share]
    geo = geo[geo["population"] > 0]
    return geo.sort_values(["state", "geo"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Aggregation to coarser grains
# ---------------------------------------------------------------------------


def _weighted_centroid(frame: pd.DataFrame) -> tuple[float, float]:
    """Population-weighted centroid on the sphere (lat, lon in degrees)."""
    w = frame["population"].to_numpy()
    xyz = (_unit_vectors(frame["latitude"], frame["longitude"]) * w[:, None]).sum(0)
    lat = np.degrees(np.arctan2(xyz[2], np.hypot(xyz[0], xyz[1])))
    lon = np.degrees(np.arctan2(xyz[1], xyz[0]))
    return float(lat), float(lon)


def aggregate_geography(zips: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a zip table to units given ``membership`` (columns ``zip``, ``geo``).

    A unit's ``state`` is the state holding most of its population (units such
    as DMAs can cross state lines); its centroid is population-weighted.
    """
    merged = zips.merge(membership.rename(columns={"zip": "geo", "geo": "unit"}), on="geo", how="inner")
    if merged.empty:
        raise ValueError("The zip -> geo mapping matches none of the zip codes.")
    rows = []
    has_coords = {"latitude", "longitude"} <= set(merged.columns)
    for unit, part in merged.groupby("unit", sort=True):
        by_state = part.groupby("state")["population"].sum()
        row = {"geo": unit, "state": by_state.idxmax(), "population": part["population"].sum()}
        if has_coords:
            row["latitude"], row["longitude"] = _weighted_centroid(part)
        rows.append(row)
    return pd.DataFrame(rows)


def state_membership(zips: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"zip": zips["geo"], "geo": zips["state"]})


def cluster_membership(zips: pd.DataFrame, n_clusters: int, seed: int = 0, n_iter: int = 50) -> pd.DataFrame:
    """Population-weighted k-means of zip centroids -> DMA-like clusters.

    Clusters are labelled ``"<largest place>, <state>"``. ``seed`` only drives
    the k-means++ initialisation and is independent of the simulation seed,
    so the clusters stay fixed across simulation seeds.
    """
    n_clusters = min(n_clusters, len(zips))
    xyz = _unit_vectors(zips["latitude"], zips["longitude"])
    w = zips["population"].to_numpy()
    rng = np.random.default_rng(seed)

    # Weighted k-means++ initialisation.
    centers = [xyz[rng.choice(len(xyz), p=w / w.sum())]]
    d2 = ((xyz - centers[0]) ** 2).sum(1)
    for _ in range(1, n_clusters):
        p = w * d2
        centers.append(xyz[rng.choice(len(xyz), p=p / p.sum())])
        d2 = np.minimum(d2, ((xyz - centers[-1]) ** 2).sum(1))
    centers = np.array(centers)

    labels = np.zeros(len(xyz), dtype=int)
    for _ in range(n_iter):
        _, new = cKDTree(centers).query(xyz)
        sums = np.zeros_like(centers)
        np.add.at(sums, new, xyz * w[:, None])
        mass = np.bincount(new, weights=w, minlength=n_clusters)
        empty = mass == 0
        centers[~empty] = sums[~empty] / mass[~empty, None]
        centers /= np.linalg.norm(centers, axis=1, keepdims=True)
        if empty.any():  # re-seed empty clusters at the worst-served heavy zips
            far = np.argsort(-w * ((xyz - centers[new]) ** 2).sum(1))[: empty.sum()]
            centers[empty] = xyz[far]
        if np.array_equal(new, labels) and not empty.any():
            break
        labels = new

    frame = zips.assign(cluster=labels)
    names = {}
    for c, part in frame.groupby("cluster"):
        top = part.loc[part["population"].idxmax()]
        names[c] = f"{top.get('place_name', top['geo'])}, {top['state']}"
    label = pd.Series(names)
    dup = label.duplicated(keep=False)
    label[dup] = label[dup] + " #" + label[dup].groupby(label[dup]).cumcount().add(1).astype(str)
    return pd.DataFrame({"zip": frame["geo"], "geo": frame["cluster"].map(label)})


def load_gma_mapping(source, country: Optional[str] = None) -> pd.DataFrame:
    """Read a Google GMA file into a zip -> unit mapping for ``custom_clusters``.

    GMAs are Google's geo units for experiments (comparable to Nielsen DMAs,
    built to limit contamination between units). The file has four columns:
    ``PostCode``, ``CountryCode`` (e.g. ``US``), ``Gma`` (the unit id) and
    ``GeoName`` (the unit's name).

    Args:
        source: Path to the CSV, or a DataFrame with those columns (any case).
        country: Optional ISO alpha-2 code to keep (the file can hold several
            countries). The simulator also filters by country on its own.

    Returns:
        DataFrame with ``zip``, ``geo`` (the GeoName; the Gma id is appended
        when a name is shared by several GMAs or missing), ``gma`` and
        ``country``. Pass it as ``GeographyConfig(custom_clusters=...)`` with
        ``geo_grain="cluster"``.
    """
    raw = pd.read_csv(source, dtype=str, keep_default_na=False) if not isinstance(source, pd.DataFrame) else source.copy()
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    missing = {"postcode", "countrycode", "gma", "geoname"} - set(raw.columns)
    if missing:
        raise ValueError(f"GMA file is missing columns {sorted(missing)}; expected PostCode, CountryCode, Gma, GeoName.")
    out = pd.DataFrame({
        "zip": raw["postcode"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True),
        "country": raw["countrycode"].astype(str).str.strip().str.upper(),
        "gma": raw["gma"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True),
        "name": raw["geoname"].astype(str).str.strip(),
    })
    out = out[(out["zip"] != "") & (out["gma"] != "")]
    if country is not None:
        out = out[out["country"] == country.strip().upper()]
    if out.empty:
        raise ValueError(f"No GMA rows left{'' if country is None else ' for country ' + country!r}.")
    # One label per GMA: its name, made unique with the id when needed.
    per_gma = out.groupby(["country", "gma"])["name"].agg(lambda n: n[n != ""].mode().iloc[0] if (n != "").any() else "")
    labels = per_gma.reset_index()
    shared = labels.duplicated(["country", "name"], keep=False) | (labels["name"] == "")
    labels["geo"] = np.where(shared, (labels["name"] + " (" + labels["gma"] + ")").str.strip(), labels["name"])
    out = out.merge(labels[["country", "gma", "geo"]], on=["country", "gma"])
    return out.drop_duplicates(["country", "zip"])[["zip", "geo", "gma", "country"]].reset_index(drop=True)


def align_custom_clusters(zips: pd.DataFrame, mapping: pd.DataFrame, country_code: Optional[str] = None) -> pd.DataFrame:
    """Match a user zip -> unit mapping to the simulator's zip table.

    * keeps the rows of ``country_code`` when the mapping has a ``country`` column;
    * restores leading zeros lost by spreadsheets (``2134`` -> ``02134``) when
      the country's codes are fixed-width digits;
    * assigns populated zips that the mapping does not list to the unit of
      the nearest listed zip, so no population is dropped.

    Returns ``zip``, ``geo`` and ``source`` (``"mapping"`` or ``"nearest"``)
    for every zip of ``zips``.
    """
    mapping = mapping.rename(columns=lambda c: str(c).strip().lower())
    if not {"zip", "geo"} <= set(mapping.columns):
        raise ValueError("custom_clusters needs columns 'zip' and 'geo' (see load_gma_mapping for GMA files).")
    if country_code and "country" in mapping.columns:
        mapping = mapping[mapping["country"].astype(str).str.upper() == country_code.upper()]
    mapping = mapping[["zip", "geo"]].astype(str)
    mapping["zip"] = mapping["zip"].str.strip()

    ours = zips["geo"]
    widths = ours.str.len().unique()
    if len(widths) == 1 and ours.str.isdigit().all():
        short = mapping["zip"].str.isdigit() & (mapping["zip"].str.len() < widths[0])
        mapping.loc[short, "zip"] = mapping.loc[short, "zip"].str.zfill(int(widths[0]))
    else:  # alphanumeric codes: compare without spaces, upper case
        key = lambda s: s.str.replace(" ", "", regex=False).str.upper()
        lookup = dict(zip(key(ours), ours))
        mapping["zip"] = key(mapping["zip"]).map(lookup).fillna(mapping["zip"])
    mapping = mapping.drop_duplicates("zip")

    out = zips[["geo", "latitude", "longitude"]].rename(columns={"geo": "zip"}).merge(mapping, on="zip", how="left")
    matched = out["geo"].notna().to_numpy()
    if not matched.any():
        raise ValueError("The zip -> geo mapping matches none of the simulator's zip codes "
                         f"(e.g. ours: {ours.head(3).tolist()}, yours: {mapping['zip'].head(3).tolist()}).")
    out["source"] = np.where(matched, "mapping", "nearest")
    if (~matched).any():
        xyz = _unit_vectors(out["latitude"], out["longitude"])
        _, nearest = cKDTree(xyz[matched]).query(xyz[~matched])
        out.loc[~matched, "geo"] = out.loc[matched, "geo"].to_numpy()[nearest]
        logger.info("custom_clusters: %d of %d populated zips were not in the mapping and joined their nearest "
                    "mapped zip's unit.", int((~matched).sum()), len(out))
    return out[["zip", "geo", "source"]]


def build_geography(country: str, grain: str, config: GeographyConfig, n_cities: int = 50,
                    n_clusters: int = 200) -> tuple[str, pd.DataFrame, Optional[pd.DataFrame]]:
    """Return ``(country_code, units, membership)``.

    ``units`` has one row per simulated unit (``geo, state, population`` and,
    when known, ``latitude, longitude``). ``membership`` maps zip codes to
    units for the ``state`` and ``cluster`` grains (``None`` otherwise).
    """
    if config.custom_geography is not None:
        return country, finalize_geography(config.custom_geography, config), None
    code = resolve_country(country, config)
    if grain == "city":
        return code, city_geography(code, n_cities, config), None
    zips = zip_geography(code, config)
    if grain == "zip":
        return code, zips, None
    if grain == "state":
        membership = state_membership(zips)
    elif config.custom_clusters is not None:
        membership = align_custom_clusters(zips, config.custom_clusters, code)
    else:
        membership = cluster_membership(zips, n_clusters, seed=config.cluster_seed)
    units = finalize_geography(aggregate_geography(zips, membership), dataclasses.replace(
        config, state_population=None, min_state_share=0.0))
    membership = membership[membership["zip"].isin(zips["geo"])].reset_index(drop=True)
    return code, units, membership
