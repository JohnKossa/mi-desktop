"""Tests for field introspection and config validation.

    python tests/test_fields.py
    pytest tests/test_fields.py

Runs against the real parcel files in ./data when they are present, and against
synthetic parquet otherwise, so it works on a clean checkout.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fields  # noqa: E402
from config import DEFAULT_WEIGHTS, RunConfig  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data"


def synthetic(path: Path, extra_cols=(), rows: int = 200) -> Path:
    rng = np.random.default_rng(0)
    data = {
        "model_group": rng.choice(["single_family", "condos"], rows),
        "land_class": rng.choice(["SINGLE FAMILY RESIDENTIAL", "RIGHT-OF-WAY"], rows),
        "bldg_area_finished_sqft": rng.uniform(600, 4000, rows),
        "land_area_sqft": rng.uniform(2000, 20000, rows),
        "bldg_age_years": rng.uniform(0, 90, rows),
        "assr_market_value": rng.uniform(50_000, 900_000, rows),
        "latitude": rng.uniform(26, 27, rows),
        "longitude": rng.uniform(-82, -81, rows),
    }
    for c in extra_cols:
        data[c] = rng.uniform(0, 1, rows)
    gdf = gpd.GeoDataFrame(
        data, geometry=[box(i, 0, i + 1, 1) for i in range(rows)], crs=2237
    )
    gdf.to_parquet(path)
    return path


# ==========================================================================
# Name mapping
# ==========================================================================


def test_binned_name_round_trip_is_idempotent():
    assert fields.scored_name("bldg_age_years") == "bldg_age_years_binned"
    assert fields.scored_name("bldg_age_years_binned") == "bldg_age_years_binned"
    assert fields.source_name("bldg_age_years_binned") == "bldg_age_years"
    assert fields.source_name("bldg_age_years") == "bldg_age_years"


def test_default_weight_lookup_bridges_the_suffix():
    """The UI shows source names; DEFAULT_WEIGHTS is keyed on binned ones."""
    got = fields.default_weight_for("bldg_area_finished_sqft", DEFAULT_WEIGHTS)
    assert got == DEFAULT_WEIGHTS["bldg_area_finished_sqft_binned"]
    assert fields.default_weight_for("nonesuch", DEFAULT_WEIGHTS) == 0.0


# ==========================================================================
# Introspection
# ==========================================================================


def test_universe_reports_derived_and_excludes_geometry():
    tmp = Path(tempfile.mkdtemp())
    try:
        uni = fields.inspect_parcel_file(synthetic(tmp / "p.parquet"))
        assert uni.ok
        assert "geometry" not in uni.columns, "geometry must never be offered"
        # assr_*_ppsf are computed later from assr_market_value / areas
        assert set(uni.derived) == {"assr_impr_ppsf", "assr_land_ppsf"}
        assert "bldg_age_years" in uni.numeric
        assert "model_group" in uni.text
        # derived ratios are legitimate scoring/seeding candidates
        assert "assr_land_ppsf" in uni.binnable
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_derived_are_not_offered_when_their_inputs_are_absent():
    tmp = Path(tempfile.mkdtemp())
    try:
        p = tmp / "p.parquet"
        synthetic(p)
        frame = gpd.read_parquet(p).drop(columns=["assr_market_value"])
        frame.to_parquet(p)
        uni = fields.inspect_parcel_file(p)
        assert uni.derived == [], uni.derived
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_missing_file_is_reported_not_raised():
    uni = fields.inspect_parcel_file("/nope/absent.parquet")
    assert not uni.ok and uni.error == "file not found"
    problems = fields.validate(RunConfig(), uni)
    assert len(problems) == 1 and problems[0].setting == "parcel_path"


def test_distinct_values_are_ordered_by_frequency():
    tmp = Path(tempfile.mkdtemp())
    try:
        p = synthetic(tmp / "p.parquet", rows=300)
        vals = fields.distinct_values(p, "model_group")
        assert set(vals) == {"single_family", "condos"}
        assert fields.distinct_values(p, "no_such_column") == []
        assert fields.distinct_values(p, "") == []
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ==========================================================================
# Validation
# ==========================================================================


def test_typo_in_filter_column_is_caught():
    tmp = Path(tempfile.mkdtemp())
    try:
        uni = fields.inspect_parcel_file(synthetic(tmp / "p.parquet"))
        cfg = RunConfig()
        cfg.parcel_filter_column = "modle_group"
        problems = fields.validate(cfg, uni)
        assert any(p.setting == "parcel_filter_column" and p.fatal
                   for p in problems), fields.describe(problems)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_filter_value_that_never_occurs_is_caught():
    tmp = Path(tempfile.mkdtemp())
    try:
        p = synthetic(tmp / "p.parquet")
        uni = fields.inspect_parcel_file(p)
        vals = fields.distinct_values(p, "model_group")
        cfg = RunConfig()
        cfg.parcel_filter_value = "duplexes"
        problems = fields.validate(cfg, uni, filter_values=vals)
        assert any(p.setting == "parcel_filter_value" for p in problems)
        # the correct value must not be flagged
        cfg.parcel_filter_value = "single_family"
        assert not any(p.setting == "parcel_filter_value"
                       for p in fields.validate(cfg, uni, filter_values=vals))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_weight_on_an_unbinned_column_is_caught():
    """The trap free-text entry made easy: the field exists, but is never binned.

    `foo` is a real numeric column, so nothing looks wrong -- but unless `foo` is
    in continuous_variables, `foo_binned` is never created and scoring fails.
    """
    tmp = Path(tempfile.mkdtemp())
    try:
        uni = fields.inspect_parcel_file(synthetic(tmp / "p.parquet"))
        cfg = RunConfig()
        cfg.continuous_variables = ["bldg_age_years"]
        cfg.weights = {
            "bldg_age_years_binned": 1.0,        # fine
            "land_area_sqft_binned": 1.0,        # exists, but not binned
        }
        problems = fields.validate(cfg, uni)
        offenders = [p for p in problems if p.setting == "weights"]
        assert len(offenders) == 1, fields.describe(problems)
        assert "not in the binning list" in offenders[0].detail
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_prebinned_column_can_be_scored_without_being_binned_again():
    tmp = Path(tempfile.mkdtemp())
    try:
        p = synthetic(tmp / "p.parquet", extra_cols=("custom_score_binned",))
        uni = fields.inspect_parcel_file(p)
        assert "custom_score_binned" in uni.prebinned
        cfg = RunConfig()
        cfg.continuous_variables = []
        cfg.weights = {"custom_score_binned": 1.0}
        assert not [p_ for p_ in fields.validate(cfg, uni)
                    if p_.setting == "weights"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_empty_weights_is_fatal():
    tmp = Path(tempfile.mkdtemp())
    try:
        uni = fields.inspect_parcel_file(synthetic(tmp / "p.parquet"))
        cfg = RunConfig()
        cfg.weights = {}
        assert any(p.setting == "weights" and p.fatal
                   for p in fields.validate(cfg, uni))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_missing_land_class_is_advisory_only_and_mode_dependent():
    tmp = Path(tempfile.mkdtemp())
    try:
        p = tmp / "p.parquet"
        synthetic(p)
        gpd.read_parquet(p).drop(columns=["land_class"]).to_parquet(p)
        uni = fields.inspect_parcel_file(p)

        cfg = RunConfig()          # tile adjacency: land class is irrelevant
        assert not [x for x in fields.validate(cfg, uni)
                    if x.setting == "land_class_column"]

        cfg.adjacency_mode = "parcel"
        cfg.obstacle_mode = "all_except"
        flagged = [x for x in fields.validate(cfg, uni)
                   if x.setting == "land_class_column"]
        assert len(flagged) == 1 and not flagged[0].fatal, "should not block a run"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ==========================================================================
# Against the real files, when present
# ==========================================================================


def test_real_files_validate_as_expected():
    lee = DATA / "universe_lee_county_fl.parquet"
    beckham = DATA / "universe_beckham_county_ok.parquet"
    if not lee.exists() or not beckham.exists():
        print("      (skipped: ./data files not present)")
        return

    cfg = RunConfig()
    lee_uni = fields.inspect_parcel_file(lee)
    lee_problems = fields.validate(
        cfg, lee_uni, fields.distinct_values(lee, cfg.parcel_filter_column)
    )
    assert not fields.fatal_problems(lee_problems), fields.describe(lee_problems)

    # Beckham is landlocked, so dist_to_open_water does not exist there -- the
    # stock config must be rejected rather than failing after the shatter.
    bk_uni = fields.inspect_parcel_file(beckham)
    bk_problems = fields.validate(
        cfg, bk_uni, fields.distinct_values(beckham, cfg.parcel_filter_column)
    )
    fatal = fields.fatal_problems(bk_problems)
    assert fatal, "stock config should not validate against Beckham"
    assert all("dist_to_open_water" in p.value for p in fatal), [
        str(p) for p in fatal
    ]

    # ...and after dropping the absent fields it should pass.
    cfg.continuous_variables = [
        c for c in cfg.continuous_variables if c in set(bk_uni.binnable)
    ]
    cfg.seed_fields = [c for c in cfg.seed_fields if c in set(bk_uni.binnable)]
    cfg.weights = {
        fields.scored_name(c): w
        for c in cfg.continuous_variables
        if (w := fields.default_weight_for(c, DEFAULT_WEIGHTS))
    }
    assert not fields.fatal_problems(fields.validate(cfg, bk_uni))


def test_the_two_real_files_are_different_jurisdictions():
    """Guard against a mis-copied data file silently making an A/B test vacuous."""
    lee = DATA / "universe_lee_county_fl.parquet"
    beckham = DATA / "universe_beckham_county_ok.parquet"
    if not lee.exists() or not beckham.exists():
        print("      (skipped: ./data files not present)")
        return
    a = pd.read_parquet(lee, columns=["latitude", "longitude"])
    b = pd.read_parquet(beckham, columns=["latitude", "longitude"])
    assert abs(a.latitude.mean() - b.latitude.mean()) > 1.0, (
        "the two parcel files cover the same latitudes -- is one a copy?"
    )
    assert len(a) != len(b)


# ==========================================================================


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for fn in tests:
        try:
            fn()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {fn.__name__}\n      {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"ERROR {fn.__name__}\n      {type(exc).__name__}: {exc}")
        else:
            print(f"ok    {fn.__name__}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
