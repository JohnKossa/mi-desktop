"""Offline tests for jurisdiction lookup and the bulk/tiled data sources.

    python tests/test_sources.py
    pytest tests/test_sources.py

No network: HTTP seams are stubbed, and the TIGER/Line reader is exercised
against a synthetic zipped shapefile built on the fly.
"""

from __future__ import annotations

import datetime as _dt
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, box

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pipeline  # noqa: E402
import sources  # noqa: E402
import tiger  # noqa: E402
from config import RunConfig  # noqa: E402
from sources import Jurisdiction  # noqa: E402


# ==========================================================================
# Query parsing / LSAD handling
# ==========================================================================


def test_parse_query_state_forms():
    cases = {
        "Fort Myers, FL": ("Fort Myers", "12"),
        "Lee County, FL": ("Lee County", "12"),
        "Lee County Florida": ("Lee County", "12"),
        "Chicago, IL": ("Chicago", "17"),
        "Chicago Illinois": ("Chicago", "17"),
        "Cook County, Illinois": ("Cook County", "17"),
        "Bonita Springs FL": ("Bonita Springs", "12"),
        # "West Virginia" must beat the "Virginia" suffix
        "Charleston West Virginia": ("Charleston", "54"),
        # no state at all
        "Cape Coral": ("Cape Coral", None),
    }
    for query, expected in cases.items():
        assert sources._parse_query(query) == expected, query


def test_strip_lsad():
    cases = {
        "Lee County": "Lee",
        "Fort Myers city": "Fort Myers",
        "East Baton Rouge Parish": "East Baton Rouge",
        "Matanuska-Susitna Borough": "Matanuska-Susitna",
        "Valdez-Cordova Census Area": "Valdez-Cordova",
        "Iosco charter township": "Iosco",
        "Lehigh Acres CDP": "Lehigh Acres",
        # nothing to strip
        "Chicago": "Chicago",
        # must never strip down to nothing
        "County": "County",
    }
    for name, expected in cases.items():
        assert sources._strip_lsad(name) == expected, name


def test_name_variants_cover_both_census_forms():
    """TIGER stores BASENAME "Lee" and NAME "Lee County"; we must match either."""
    assert sources._name_variants("Lee County") == ["Lee County", "Lee"]
    assert sources._name_variants("Chicago") == ["Chicago"]


def test_search_builds_where_clause_matching_basename_and_name():
    """The old clause only tested BASENAME, so "Lee County" matched nothing."""
    captured = {}

    def fake_query_layer(layer_id, where, out_fields="*", **kw):
        captured.setdefault("wheres", []).append(where)
        return {"features": []}

    original = sources._query_layer
    sources._query_layer = fake_query_layer
    try:
        sources.search_jurisdictions("Lee County, FL")
    finally:
        sources._query_layer = original

    assert captured["wheres"], "no layers were queried"
    where = captured["wheres"][0]
    assert "BASENAME" in where and "NAME" in where, where
    assert "'Lee County%'" in where, where
    assert "'Lee%'" in where, "LSAD-stripped variant missing"
    assert "STATE='12'" in where, where


def test_search_ranks_the_named_county_above_similar_places():
    """"Lee County, FL" must return Lee County, not Leesburg or a Lee CDP."""
    def feat(name, basename, geoid, state, size):
        return {
            "properties": {"NAME": name, "BASENAME": basename,
                           "GEOID": geoid, "STATE": state},
            "geometry": box(0, 0, size, size).__geo_interface__,
        }

    responses = {
        sources.LAYER_INCORPORATED_PLACES: [
            feat("Leesburg city", "Leesburg", "1239850", "12", 2),
        ],
        sources.LAYER_CDP: [
            feat("Lee CDP", "Lee", "1239851", "12", 1),
        ],
        sources.LAYER_COUNTY_SUBDIVISIONS: [],
        sources.LAYER_COUNTIES: [
            feat("Lee County", "Lee", "12071", "12", 30),
        ],
    }

    original = sources._query_layer
    sources._query_layer = lambda lid, where, out_fields="*", **kw: {
        "features": responses.get(lid, [])
    }
    try:
        matches = sources.search_jurisdictions("Lee County, FL")
    finally:
        sources._query_layer = original

    assert matches, "no matches"
    assert matches[0].name == "Lee County", [m.name for m in matches]
    assert matches[0].geoid == "12071"
    # Leesburg is only a prefix match, so it must rank below the exact hits.
    assert "Leesburg" not in matches[0].name


def test_counties_layer_is_always_searched():
    """Places used to be able to fill `limit` and short-circuit the county layer."""
    seen = []

    original = sources._query_layer
    sources._query_layer = lambda lid, where, out_fields="*", **kw: (
        seen.append(lid) or {"features": []}
    )
    try:
        sources.search_jurisdictions("Lee County, FL", limit=1)
    finally:
        sources._query_layer = original

    assert sources.LAYER_COUNTIES in seen, seen


# ==========================================================================
# Jurisdiction round-trip
# ==========================================================================


def test_jurisdiction_gdf_round_trip_preserves_state():
    """state_fips must survive to disk -- the bulk fetcher needs it."""
    jur = Jurisdiction(
        name="Lee County", layer_id=82, layer_name="County", geoid="12071",
        state_fips="12", geometry=box(-82, 26.4, -81.5, 26.8), basename="Lee",
    )
    gdf = jur.to_gdf()
    back = pipeline._jurisdiction_from_gdf(gdf, RunConfig())
    assert back.name == "Lee County"
    assert back.state_fips == "12"
    assert back.geoid == "12071"
    assert back.layer_id == 82
    assert back.basename == "Lee"


def test_legacy_jurisdiction_parquet_recovers_state_from_geoid():
    """Run directories written before state_fips existed must still work."""
    legacy = gpd.GeoDataFrame(
        {"name": ["Lee County"], "geoid": ["12071"]},
        geometry=[box(-82, 26.4, -81.5, 26.8)], crs="EPSG:4326",
    )
    back = pipeline._jurisdiction_from_gdf(legacy, RunConfig())
    assert back.state_fips == "12", "should fall back to the GEOID prefix"


# ==========================================================================
# TIGER/Line bulk source
# ==========================================================================


def test_blocks_url_and_year_candidates():
    assert tiger.blocks_url(2025, "12") == (
        "https://www2.census.gov/geo/tiger/TIGER2025/TABBLOCK20/"
        "tl_2025_12_tabblock20.zip"
    )
    # single-digit state codes must be zero-padded
    assert "tl_2025_06_tabblock20.zip" in tiger.blocks_url(2025, "6")

    years = tiger.candidate_years(_dt.date(2026, 7, 28))
    assert years[0] == 2027, "should probe one vintage ahead"
    assert years == sorted(years, reverse=True), "must be newest-first"
    assert min(years) == 2020, "TABBLOCK20 starts at 2020"


def test_resolve_year_picks_newest_available():
    probed = []

    def fake_exists(session, url, timeout=30):
        probed.append(url)
        return "TIGER2024" in url  # pretend 2025+ aren't published yet

    original = tiger._url_exists
    tiger._url_exists = fake_exists
    try:
        year = tiger.resolve_year(
            "12", years=[2027, 2026, 2025, 2024, 2023], use_cache=False
        )
    finally:
        tiger._url_exists = original

    assert year == 2024, year
    # must have stopped at the first hit rather than probing every year
    assert not any("TIGER2023" in u for u in probed), probed


def test_resolve_year_raises_with_actionable_message():
    original = tiger._url_exists
    tiger._url_exists = lambda session, url, timeout=30: False
    try:
        try:
            tiger.resolve_year("12", years=[2025], use_cache=False)
        except RuntimeError as exc:
            assert "TABBLOCK20" in str(exc)
            assert "census blocks off" in str(exc), "should suggest a way forward"
        else:
            raise AssertionError("expected RuntimeError")
    finally:
        tiger._url_exists = original


def _synthetic_block_zip(directory: Path, n_side: int = 20) -> Path:
    """A zipped shapefile shaped like tl_<year>_<ss>_tabblock20.zip."""
    stem = "tl_2025_99_tabblock20"
    cells = [
        box(-82.0 + i * 0.01, 26.0 + j * 0.01,
            -82.0 + (i + 1) * 0.01, 26.0 + (j + 1) * 0.01)
        for i in range(n_side) for j in range(n_side)
    ]
    gdf = gpd.GeoDataFrame(
        {"GEOID20": [f"{k:015d}" for k in range(len(cells))]},
        geometry=cells, crs=4269,  # TIGER ships NAD83 geographic
    )
    shp = directory / f"{stem}.shp"
    gdf.to_file(shp)
    zip_path = directory / f"{stem}.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        for f in os.listdir(directory):
            if f.startswith(stem) and not f.endswith(".zip"):
                z.write(directory / f, f)
    return zip_path


def test_read_blocks_bbox_filters_and_reprojects():
    with tempfile.TemporaryDirectory() as d:
        zip_path = _synthetic_block_zip(Path(d))

        full = tiger.read_blocks(zip_path)
        assert len(full) == 400
        assert full.crs.to_epsg() == 4326, "must come back as 4326"

        # a 3x3-cell window, in lon/lat
        sub = tiger.read_blocks(zip_path, bbox=(-81.98, 26.02, -81.95, 26.05))
        assert 0 < len(sub) < len(full), len(sub)
        assert len(sub) <= 25, "bbox prefilter is not filtering"
        assert sub.geometry.notna().all() and not sub.geometry.is_empty.any()


def test_fetch_blocks_uses_jurisdiction_state_without_network():
    with tempfile.TemporaryDirectory() as d:
        zip_path = _synthetic_block_zip(Path(d))
        calls = []

        original = tiger.download_state_blocks
        tiger.download_state_blocks = lambda ss, progress=None, **kw: (
            calls.append(ss) or zip_path
        )
        try:
            jur = Jurisdiction(
                name="Test", layer_id=82, layer_name="County", geoid="99001",
                state_fips="99", geometry=box(-82.0, 26.0, -81.9, 26.1),
            )
            blocks = tiger.fetch_blocks(jur)
        finally:
            tiger.download_state_blocks = original

        assert calls == ["99"], calls
        assert len(blocks) > 0
        assert blocks.crs.to_epsg() == 4326


def test_pipeline_falls_back_to_rest_when_bulk_fails():
    marker = gpd.GeoDataFrame({"geometry": [box(0, 0, 1, 1)]}, crs="EPSG:4326")
    logged = []

    orig_tiger, orig_rest = tiger.fetch_blocks, sources.fetch_census_blocks
    pipeline.tiger.fetch_blocks = lambda j, progress=None: (_ for _ in ()).throw(
        RuntimeError("download exploded")
    )
    pipeline.sources.fetch_census_blocks = lambda j, progress=None: marker
    try:
        jur = Jurisdiction(
            name="Test", layer_id=82, layer_name="County", geoid="12071",
            state_fips="12", geometry=box(-82, 26.4, -81.5, 26.8),
        )
        out = pipeline.fetch_blocks(jur, RunConfig(), progress=logged.append)
    finally:
        pipeline.tiger.fetch_blocks = orig_tiger
        pipeline.sources.fetch_census_blocks = orig_rest

    assert out is marker, "should have fallen back to the REST fetcher"
    assert any("truncates" in m for m in logged), logged


def test_pipeline_honours_rest_opt_out():
    marker = gpd.GeoDataFrame({"geometry": [box(0, 0, 1, 1)]}, crs="EPSG:4326")
    called = {"bulk": False}

    orig_tiger, orig_rest = tiger.fetch_blocks, sources.fetch_census_blocks
    pipeline.tiger.fetch_blocks = lambda j, progress=None: called.__setitem__("bulk", True)
    pipeline.sources.fetch_census_blocks = lambda j, progress=None: marker
    try:
        jur = Jurisdiction(
            name="Test", layer_id=82, layer_name="County", geoid="12071",
            state_fips="12", geometry=box(-82, 26.4, -81.5, 26.8),
        )
        cfg = RunConfig()
        cfg.census_source = "rest"
        out = pipeline.fetch_blocks(jur, cfg)
    finally:
        pipeline.tiger.fetch_blocks = orig_tiger
        pipeline.sources.fetch_census_blocks = orig_rest

    assert out is marker
    assert not called["bulk"], "census_source='rest' must not hit the bulk path"


# ==========================================================================
# Adaptive Overpass splitting
# ==========================================================================


def test_quadsplit_partitions_exactly():
    bounds = (-82.0, 26.5, -81.7, 26.75)
    cells = sources.quadsplit(bounds)
    assert len(cells) == 4
    total = sum((c[2] - c[0]) * (c[3] - c[1]) for c in cells)
    whole = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
    assert abs(total - whole) < 1e-12, "cells must tile the parent exactly"
    for c in cells:
        assert c[0] >= bounds[0] and c[2] <= bounds[2]
        assert c[1] >= bounds[1] and c[3] <= bounds[3]


def test_overpass_remark_is_treated_as_failure():
    """A truncated response arrives as HTTP 200 with a 'remark' field."""
    for remark in ["runtime error: Query timed out in 'recurse' at line 3",
                   "runtime error: Query run out of memory"]:
        try:
            sources._check_remark({"elements": [], "remark": remark})
        except sources.OverpassTooBig:
            pass
        else:
            raise AssertionError(f"should have rejected: {remark}")
    # a benign remark must pass through
    sources._check_remark({"elements": [], "remark": "some informational note"})


def test_tiled_fetch_subdivides_until_queries_fit():
    """Dense areas must split; the depth needed is discovered, not configured."""
    # Depth 1 cells are ~0.04 sq deg and must fail; depth 2 are ~0.01 and must
    # pass. The threshold sits between them rather than exactly on 0.01, because
    # halving float bounds makes the areas land a hair either side of it.
    area_limit = 0.015
    attempts = []

    def fake_overpass(query, progress=None, timeout=300):
        # recover the bbox from the query text: "(S,W,N,E)"
        inner = query[query.index("(") + 1: query.index(")")]
        s, w, n, e = (float(v) for v in inner.split(","))
        area = (e - w) * (n - s)
        attempts.append(area)
        if area > area_limit:
            raise sources.OverpassTooBig("runtime error: Query timed out")
        return {"elements": [{
            "type": "way",
            "geometry": [{"lon": w, "lat": s}, {"lon": e, "lat": n}],
        }]}

    original_overpass = sources._overpass
    original_pause = sources.OVERPASS_PAUSE_S
    sources._overpass = fake_overpass
    sources.OVERPASS_PAUSE_S = 0.0
    try:
        geoms = sources._fetch_osm_tiled(
            "roads", "way[highway]({bbox});", (-82.0, 26.0, -81.6, 26.4),
            sources._elements_to_lines, use_cache=False,
        )
    finally:
        sources._overpass = original_overpass
        sources.OVERPASS_PAUSE_S = original_pause

    # 0.16 sq deg parent; halving each side per level -> depth 2 gives 0.01
    assert len(geoms) == 16, f"expected 16 leaf cells, got {len(geoms)}"
    assert all(isinstance(g, LineString) for g in geoms)
    assert max(attempts) > area_limit, "the parent query should have been tried first"
    assert sum(1 for a in attempts if a <= area_limit) == 16


def test_tiled_fetch_gives_up_with_a_useful_message():
    original_overpass = sources._overpass
    original_pause = sources.OVERPASS_PAUSE_S
    sources._overpass = lambda q, progress=None, timeout=300: (
        (_ for _ in ()).throw(sources.OverpassTooBig("always too big"))
    )
    sources.OVERPASS_PAUSE_S = 0.0
    try:
        try:
            sources._fetch_osm_tiled(
                "roads", "way[highway]({bbox});", (-82.0, 26.0, -81.0, 27.0),
                sources._elements_to_lines, use_cache=False,
            )
        except RuntimeError as exc:
            msg = str(exc)
            assert "depth" in msg
            assert "turn OSM roads off" in msg, "should offer a way forward"
        else:
            raise AssertionError("expected RuntimeError")
    finally:
        sources._overpass = original_overpass
        sources.OVERPASS_PAUSE_S = original_pause


def test_dedupe_drops_cross_cell_repeats():
    a = LineString([(0, 0), (1, 1)])
    b = LineString([(0, 0), (1, 1)])  # identical geometry, separate object
    c = LineString([(0, 0), (2, 2)])
    out = sources._dedupe([a, b, c, None])
    assert len(out) == 2, [g.wkt for g in out]


# ==========================================================================
# Toggles must actually prevent network calls
# ==========================================================================


def test_every_osm_layer_has_a_toggle_that_prepare_honours():
    """Regression: waterway lines fetched unconditionally.

    `use_osm_roads` and `clip_water` existed, but the waterway-centreline fetch
    had no toggle at all -- so switching every OSM option off in the UI still
    fired an Overpass query, and still failed on a jurisdiction Overpass refused.
    """
    import ast
    import inspect

    import pipeline
    from config import RunConfig

    tree = ast.parse(inspect.getsource(pipeline.prepare))
    guarded = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.IfExp):
            continue
        called = [
            n.func.attr for n in ast.walk(node.body)
            if isinstance(n, ast.Call) and hasattr(n.func, "attr")
        ]
        flags = [n.attr for n in ast.walk(node.test) if isinstance(n, ast.Attribute)]
        for name in called:
            if name.startswith("fetch_osm") or name == "fetch_blocks":
                guarded[name] = flags

    for fetcher in ("fetch_osm_roads", "fetch_osm_waterway_lines",
                    "fetch_osm_water_areas"):
        assert fetcher in guarded, (
            f"{fetcher} is called unconditionally in prepare(); it needs a toggle"
        )
        assert guarded[fetcher], f"{fetcher} has no config flag guarding it"
        for flag in guarded[fetcher]:
            assert hasattr(RunConfig(), flag), f"{flag} is not a RunConfig field"


def test_rate_limiting_is_not_treated_as_too_big():
    """A 429 must back off, not subdivide.

    OverpassTooBig makes the caller split the area into four and issue four more
    queries -- exactly the wrong reply to a server asking us to slow down. Only
    evidence of size (a truncation remark, or every endpoint timing out) should
    trigger a split.
    """
    calls = []

    class FakeResponse:
        def __init__(self, code):
            self.status_code = code

        def json(self):
            return {"elements": []}

        def raise_for_status(self):
            pass

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, **kw):
            calls.append(url)
            return FakeResponse(429)

    orig_session, orig_backoff = sources._session, sources.RATE_LIMIT_BACKOFF_S
    sources._session = FakeSession
    sources.RATE_LIMIT_BACKOFF_S = 0.0
    try:
        try:
            sources._overpass("[out:json];out;")
        except sources.OverpassTooBig as exc:
            raise AssertionError(f"429 was reported as too-big: {exc}")
        except RuntimeError:
            pass  # correct: a plain failure after trying every endpoint
        assert len(calls) == len(sources.OVERPASS_ENDPOINTS), (
            f"should have tried every endpoint, tried {len(calls)}"
        )
    finally:
        sources._session = orig_session
        sources.RATE_LIMIT_BACKOFF_S = orig_backoff


def test_gateway_timeout_on_every_endpoint_does_signal_too_big():
    class FakeResponse:
        status_code = 504

        def json(self):
            return {}

        def raise_for_status(self):
            pass

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, **kw):
            return FakeResponse()

    orig = sources._session
    sources._session = FakeSession
    try:
        try:
            sources._overpass("[out:json];out;")
        except sources.OverpassTooBig:
            pass  # correct: consistent timeouts do point at extent
        else:
            raise AssertionError("expected OverpassTooBig once all endpoints 504")
    finally:
        sources._session = orig


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
