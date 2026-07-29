"""Remote data sources: Census TIGERweb and OpenStreetMap (Overpass).

Everything fetched here is cached to ``.mi_cache/`` as parquet keyed by a hash
of the request, so re-running a jurisdiction is instant and offline.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import LineString, MultiPolygon, Polygon, shape
from shapely.ops import linemerge, polygonize, unary_union

from config import cache_dir

USER_AGENT = "mi-neighborhoods-desktop/0.1 (parcel neighborhood optimizer)"

TIGERWEB = (
    "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/"
    "tigerWMS_Current/MapServer"
)

# TIGERweb layer ids (verified against the tigerWMS_Current service definition)
LAYER_BLOCKS = 12
LAYER_BLOCK_GROUPS = 10
LAYER_COUNTY_SUBDIVISIONS = 22
LAYER_INCORPORATED_PLACES = 28
LAYER_CDP = 30
LAYER_COUNTIES = 82
LAYER_STATES = 80

JURISDICTION_LAYERS: List[Tuple[int, str]] = [
    (LAYER_INCORPORATED_PLACES, "Incorporated Place"),
    (LAYER_CDP, "Census Designated Place"),
    (LAYER_COUNTY_SUBDIVISIONS, "County Subdivision"),
    (LAYER_COUNTIES, "County"),
]

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.osm.ch/api/interpreter",
]

STATE_FIPS: Dict[str, str] = {
    "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06", "CO": "08",
    "CT": "09", "DE": "10", "DC": "11", "FL": "12", "GA": "13", "HI": "15",
    "ID": "16", "IL": "17", "IN": "18", "IA": "19", "KS": "20", "KY": "21",
    "LA": "22", "ME": "23", "MD": "24", "MA": "25", "MI": "26", "MN": "27",
    "MS": "28", "MO": "29", "MT": "30", "NE": "31", "NV": "32", "NH": "33",
    "NJ": "34", "NM": "35", "NY": "36", "NC": "37", "ND": "38", "OH": "39",
    "OK": "40", "OR": "41", "PA": "42", "RI": "44", "SC": "45", "SD": "46",
    "TN": "47", "TX": "48", "UT": "49", "VT": "50", "VA": "51", "WA": "53",
    "WV": "54", "WI": "55", "WY": "56", "AS": "60", "GU": "66", "MP": "69",
    "PR": "72", "VI": "78",
}

STATE_NAMES: Dict[str, str] = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT",
    "delaware": "DE", "district of columbia": "DC", "florida": "FL",
    "georgia": "GA", "hawaii": "HI", "idaho": "ID", "illinois": "IL",
    "indiana": "IN", "iowa": "IA", "kansas": "KS", "kentucky": "KY",
    "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN",
    "mississippi": "MS", "missouri": "MO", "montana": "MT",
    "nebraska": "NE", "nevada": "NV", "new hampshire": "NH",
    "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
    "north carolina": "NC", "north dakota": "ND", "ohio": "OH",
    "oklahoma": "OK", "oregon": "OR", "pennsylvania": "PA",
    "rhode island": "RI", "south carolina": "SC", "south dakota": "SD",
    "tennessee": "TN", "texas": "TX", "utah": "UT", "vermont": "VT",
    "virginia": "VA", "washington": "WA", "west virginia": "WV",
    "wisconsin": "WI", "wyoming": "WY", "puerto rico": "PR",
}

Progress = Callable[[str], None]


def _noop(_: str) -> None:
    pass


# ==========================================================================
# HTTP plumbing
# ==========================================================================


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def _post_json(url: str, data: dict, timeout: int = 180, retries: int = 3) -> dict:
    last: Optional[Exception] = None
    with _session() as s:
        for attempt in range(retries):
            try:
                r = s.post(url, data=data, timeout=timeout)
                r.raise_for_status()
                return r.json()
            except Exception as exc:  # noqa: BLE001 - surfaced after retries
                last = exc
                time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Request to {url} failed after {retries} tries: {last}")


def _cache_path(kind: str, key: str) -> Path:
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return cache_dir() / f"{kind}_{h}.parquet"


def _read_cache(path: Path) -> Optional[gpd.GeoDataFrame]:
    if path.exists():
        try:
            return gpd.read_parquet(path)
        except Exception:
            path.unlink(missing_ok=True)
    return None


def _write_cache(path: Path, gdf: gpd.GeoDataFrame) -> None:
    try:
        gdf.to_parquet(path)
    except Exception:
        pass


# ==========================================================================
# Jurisdiction lookup
# ==========================================================================


@dataclass
class Jurisdiction:
    name: str
    layer_id: int
    layer_name: str
    geoid: str
    state_fips: str
    geometry: object  # shapely geometry in EPSG:4326
    basename: str = ""

    @property
    def label(self) -> str:
        st = next((k for k, v in STATE_FIPS.items() if v == self.state_fips), "")
        return f"{self.name}, {st} ({self.layer_name})" if st else \
            f"{self.name} ({self.layer_name})"

    def bounds(self) -> Tuple[float, float, float, float]:
        return tuple(self.geometry.bounds)  # type: ignore[return-value]

    def to_gdf(self) -> gpd.GeoDataFrame:
        # Every field is persisted, not just the geometry: reopening a run has
        # to be able to rebuild the Jurisdiction, and the bulk block fetcher
        # needs state_fips to know which state file to pull.
        return gpd.GeoDataFrame(
            {
                "name": [self.name],
                "geoid": [self.geoid],
                "basename": [self.basename],
                "state_fips": [self.state_fips],
                "layer_id": [self.layer_id],
                "layer_name": [self.layer_name],
            },
            geometry=[self.geometry],
            crs="EPSG:4326",
        )


def _parse_query(query: str) -> Tuple[str, Optional[str]]:
    """Split "Fort Myers, FL" into ("Fort Myers", "12").

    Also handles the comma-less form ("Lee County Florida") by testing whether
    the query ends with a state name or abbreviation.
    """
    parts = [p.strip() for p in query.split(",") if p.strip()]
    if len(parts) >= 2:
        tail = parts[-1]
        fips = STATE_FIPS.get(tail.upper()) or STATE_FIPS.get(
            STATE_NAMES.get(tail.lower(), ""), ""
        )
        if fips:
            return ", ".join(parts[:-1]), fips

    # No comma: try to peel a trailing state off the end. Longest first, so
    # "West Virginia" wins over "Virginia".
    text = query.strip()
    lowered = text.lower()
    for state_name in sorted(STATE_NAMES, key=len, reverse=True):
        if lowered.endswith(" " + state_name):
            return text[: -(len(state_name) + 1)].strip(), STATE_FIPS[
                STATE_NAMES[state_name]
            ]
    tokens = text.split()
    if len(tokens) >= 2 and tokens[-1].upper() in STATE_FIPS:
        return " ".join(tokens[:-1]), STATE_FIPS[tokens[-1].upper()]

    return text, None


# Legal/statistical area descriptions the Census appends to NAME but leaves out
# of BASENAME: NAME "Lee County" has BASENAME "Lee", NAME "Fort Myers city" has
# BASENAME "Fort Myers". Users type either form, so we search both.
_LSAD_SUFFIXES = {
    "county", "parish", "borough", "census area", "municipality", "municipio",
    "city and borough", "city", "town", "township", "village",
    "cdp", "charter township", "urban county", "consolidated government",
    "metropolitan government", "unified government", "plantation", "gore",
    "district", "precinct", "reservation",
}

# Longest first, so "charter township" is stripped whole rather than leaving
# "Iosco charter" behind, and "city and borough" beats "borough".
LSAD_SUFFIXES = tuple(sorted(_LSAD_SUFFIXES, key=len, reverse=True))


def _strip_lsad(name: str) -> str:
    """"Lee County" -> "Lee";  "Fort Myers city" -> "Fort Myers"."""
    out = name.strip()
    changed = True
    while changed:
        changed = False
        lowered = out.lower()
        for suffix in LSAD_SUFFIXES:
            if lowered.endswith(" " + suffix):
                out = out[: -(len(suffix) + 1)].rstrip()
                changed = True
                break
    return out or name.strip()


def _name_variants(name: str) -> List[str]:
    """The forms worth matching against BASENAME/NAME, longest first."""
    variants = [name.strip()]
    stripped = _strip_lsad(name)
    if stripped and stripped.lower() != name.strip().lower():
        variants.append(stripped)
    return variants


def _esri_escape(value: str) -> str:
    return value.replace("'", "''")


def _query_layer(
    layer_id: int,
    where: str,
    out_fields: str = "*",
    geometry: Optional[dict] = None,
    geometry_type: str = "esriGeometryEnvelope",
    object_ids: Optional[Sequence[int]] = None,
    return_ids_only: bool = False,
) -> dict:
    payload = {
        "f": "geojson" if not return_ids_only else "json",
        "where": where,
        "outFields": out_fields,
        "returnGeometry": "false" if return_ids_only else "true",
        "outSR": "4326",
        "returnIdsOnly": "true" if return_ids_only else "false",
    }
    if geometry is not None:
        payload["geometry"] = json.dumps(geometry)
        payload["geometryType"] = geometry_type
        payload["inSR"] = "4326"
        payload["spatialRel"] = "esriSpatialRelIntersects"
    if object_ids:
        payload["objectIds"] = ",".join(str(i) for i in object_ids)
        payload["where"] = "1=1"
    return _post_json(f"{TIGERWEB}/{layer_id}/query", payload)


def search_jurisdictions(
    query: str, progress: Progress = _noop, limit: int = 25
) -> List[Jurisdiction]:
    """Find candidate study areas matching free text like ``"Fort Myers, FL"``."""
    name, state_fips = _parse_query(query)
    if not name:
        return []

    variants = _name_variants(name)
    results: List[Jurisdiction] = []
    seen: set = set()

    for layer_id, layer_name in JURISDICTION_LAYERS:
        # BASENAME is the bare name ("Lee"), NAME carries the LSAD ("Lee
        # County"). Match either, against both the query as typed and the
        # query with its LSAD suffix removed, so "Lee County, FL", "Lee, FL"
        # and "Chicago, IL" all resolve.
        ors = []
        for variant in variants:
            like = _esri_escape(variant)
            ors.append(f"UPPER(BASENAME) LIKE UPPER('{like}%')")
            ors.append(f"UPPER(NAME) LIKE UPPER('{like}%')")
        clauses = ["(" + " OR ".join(ors) + ")"]
        if state_fips:
            clauses.append(f"STATE='{state_fips}'")
        where = " AND ".join(clauses)
        progress(f"Searching TIGERweb ({layer_name}) for '{name}'...")
        try:
            fc = _query_layer(layer_id, where, out_fields="GEOID,BASENAME,NAME,STATE")
        except Exception as exc:  # noqa: BLE001
            progress(f"  {layer_name} lookup failed: {exc}")
            continue

        for feat in fc.get("features", []) or []:
            props = feat.get("properties") or {}
            geom_json = feat.get("geometry")
            if not geom_json:
                continue
            geoid = str(props.get("GEOID", ""))
            if (layer_id, geoid) in seen:
                continue
            seen.add((layer_id, geoid))
            results.append(
                Jurisdiction(
                    name=str(props.get("NAME") or props.get("BASENAME") or name),
                    layer_id=layer_id,
                    layer_name=layer_name,
                    geoid=geoid,
                    state_fips=str(props.get("STATE", "")),
                    geometry=shape(geom_json),
                    basename=str(props.get("BASENAME", "")),
                )
            )
        # No early break: counties are the last layer searched, so bailing out
        # once `limit` is reached could drop the very match the user typed.

    wanted = {v.lower() for v in variants}

    def rank(j: Jurisdiction) -> tuple:
        forms = {j.name.lower(), j.basename.lower(), _strip_lsad(j.name).lower()}
        if forms & wanted:
            tier = 0                      # exact hit on either form
        elif any(f.startswith(tuple(wanted)) for f in forms if f):
            tier = 1                      # prefix hit
        else:
            tier = 2
        # Within a tier, bigger wins: typing a bare city name usually means the
        # city, and typing a county name means the county, not a like-named CDP.
        return (tier, -float(j.geometry.area))

    results.sort(key=rank)
    return results[:limit]


def get_jurisdiction(query: str, progress: Progress = _noop) -> Jurisdiction:
    """Return the single best match, or raise."""
    matches = search_jurisdictions(query, progress=progress)
    if not matches:
        raise LookupError(
            f"No Census place, CDP, county subdivision or county matched "
            f"'{query}'. Try 'City, ST' or 'Something County, ST'."
        )
    return matches[0]


# ==========================================================================
# Census blocks
# ==========================================================================


def _envelope(bounds: Tuple[float, float, float, float]) -> dict:
    minx, miny, maxx, maxy = bounds
    return {
        "xmin": minx, "ymin": miny, "xmax": maxx, "ymax": maxy,
        "spatialReference": {"wkid": 4326},
    }


def fetch_census_blocks(
    jur: Jurisdiction,
    progress: Progress = _noop,
    layer_id: int = LAYER_BLOCKS,
    chunk: int = 1000,  # TIGERweb caps maxSelectionCount at 2000 per request
    use_cache: bool = True,
) -> gpd.GeoDataFrame:
    """Download 2020 Census blocks intersecting the jurisdiction envelope."""
    key = f"blocks|{layer_id}|{jur.layer_id}|{jur.geoid}"
    path = _cache_path("blocks", key)
    if use_cache:
        cached = _read_cache(path)
        if cached is not None:
            progress(f"Census blocks: {len(cached):,} (cached)")
            return cached

    env = _envelope(jur.bounds())
    progress("Census blocks: requesting object ids...")
    ids_resp = _query_layer(
        layer_id, "1=1", geometry=env, return_ids_only=True
    )
    oids = ids_resp.get("objectIds") or []
    if not oids:
        progress("Census blocks: none returned")
        return gpd.GeoDataFrame({"geometry": []}, crs="EPSG:4326")
    if ids_resp.get("exceededTransferLimit"):
        progress(
            f"Census blocks: TIGERweb truncated the id list at {len(oids):,}. "
            "The study area is very large; blocks near its edge may be missing."
        )

    progress(f"Census blocks: downloading {len(oids):,} features...")
    frames: List[gpd.GeoDataFrame] = []
    for i in range(0, len(oids), chunk):
        batch = oids[i : i + chunk]
        fc = _query_layer(layer_id, "1=1", out_fields="GEOID", object_ids=batch)
        feats = fc.get("features") or []
        if feats:
            frames.append(gpd.GeoDataFrame.from_features(feats, crs="EPSG:4326"))
        progress(
            f"Census blocks: {min(i + chunk, len(oids)):,}/{len(oids):,}"
        )

    if not frames:
        return gpd.GeoDataFrame({"geometry": []}, crs="EPSG:4326")

    blocks = gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True), crs="EPSG:4326"
    )
    blocks = blocks[~blocks.geometry.is_empty & blocks.geometry.notna()]
    blocks = blocks.reset_index(drop=True)
    if use_cache:
        _write_cache(path, blocks)
    return blocks


# ==========================================================================
# OpenStreetMap / Overpass
# ==========================================================================

ROAD_FILTER = (
    '["highway"]["highway"!~"^(footway|path|steps|cycleway|bridleway|'
    'pedestrian|corridor|elevator|proposed|construction|raceway)$"]'
)
WATERWAY_LINE_FILTER = '["waterway"~"^(river|stream|canal|ditch|drain)$"]'
WATER_AREA_FILTERS = [
    '["natural"="water"]',
    '["waterway"="riverbank"]',
    '["landuse"="reservoir"]',
]

AREA_TAG_HINTS = {
    "natural", "landuse", "leisure", "building", "amenity", "water",
    "waterway", "place", "area",
}


class OverpassTooBig(RuntimeError):
    """The query exceeded Overpass's time or memory budget for this area."""


# Overpass answers an over-budget query with HTTP 200, a partial element list,
# and an explanation tucked into "remark" -- so a successful-looking response
# has to be inspected or the study area silently loses roads.
_TRUNCATION_HINTS = ("timed out", "out of memory", "too many", "exceeded")


def _check_remark(data: dict) -> None:
    remark = str(data.get("remark") or "")
    if remark and any(h in remark.lower() for h in _TRUNCATION_HINTS):
        raise OverpassTooBig(remark.strip())


#: How long to wait when an endpoint says it is rate limited.
RATE_LIMIT_BACKOFF_S = 30.0


def _overpass(query: str, progress: Progress = _noop, timeout: int = 300) -> dict:
    """Run one Overpass query, distinguishing "too big" from "busy".

    The distinction matters because the caller's response to ``OverpassTooBig`` is
    to split the area into four and issue four more queries. That is right when
    the query genuinely exceeds a budget, and actively harmful when the server is
    merely overloaded or rate limiting us -- quadrupling the request count is the
    opposite of what a 429 is asking for.

    So: a ``remark`` in a 200 response is definitive evidence of truncation and
    splits immediately. A 429 backs off and moves on. A 504 might be either, so
    every endpoint is tried before concluding the area is too big.
    """
    last: Optional[Exception] = None
    gateway_timeouts = 0
    for url in OVERPASS_ENDPOINTS:
        host = url.split("/")[2]
        try:
            progress(f"Overpass: querying {host}...")
            with _session() as s:
                r = s.post(url, data={"data": query}, timeout=timeout)

                if r.status_code in (429, 503):
                    # Rate limited / no slot free. Splitting would make it worse.
                    progress(
                        f"Overpass: {host} is rate limiting (HTTP "
                        f"{r.status_code}); waiting {RATE_LIMIT_BACKOFF_S:.0f}s "
                        "and trying the next endpoint"
                    )
                    last = RuntimeError(f"HTTP {r.status_code} from {host}")
                    time.sleep(RATE_LIMIT_BACKOFF_S)
                    continue

                if r.status_code == 504:
                    # Could be size, could be an overloaded gateway. Exhaust the
                    # other endpoints before deciding to subdivide.
                    gateway_timeouts += 1
                    progress(f"Overpass: {host} timed out (HTTP 504)")
                    last = RuntimeError(f"HTTP 504 from {host}")
                    continue

                if r.status_code == 400:
                    # Our queries are generated, so a bad request here means the
                    # server rejected the extent rather than the syntax.
                    raise OverpassTooBig(f"HTTP 400 from {host}")

                r.raise_for_status()
                data = r.json()
            _check_remark(data)
            return data
        except OverpassTooBig:
            raise  # splitting the area is the fix, not another endpoint
        except Exception as exc:  # noqa: BLE001
            last = exc
            progress(f"Overpass: {host} failed ({exc}); trying next")
            time.sleep(2)

    if gateway_timeouts == len(OVERPASS_ENDPOINTS):
        # Every endpoint timed out on the same extent, which does point at size.
        raise OverpassTooBig(
            f"all {gateway_timeouts} endpoints returned HTTP 504"
        )
    raise RuntimeError(f"All Overpass endpoints failed: {last}")


def _bbox_str(bounds: Tuple[float, float, float, float]) -> str:
    minx, miny, maxx, maxy = bounds
    return f"{miny},{minx},{maxy},{maxx}"  # Overpass wants S,W,N,E


def quadsplit(
    bounds: Tuple[float, float, float, float]
) -> List[Tuple[float, float, float, float]]:
    """Four equal sub-boxes. Cells overlap by nothing; ways spanning a cut are
    returned by both cells (Overpass clips to the bbox but keeps whole ways), so
    duplicates are dropped downstream."""
    minx, miny, maxx, maxy = bounds
    midx, midy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
    return [
        (minx, miny, midx, midy),
        (midx, miny, maxx, midy),
        (minx, midy, midx, maxy),
        (midx, midy, maxx, maxy),
    ]


#: Deepest the quadtree will go: 4 levels = up to 256 cells, enough to bring a
#: dense metropolitan county under Overpass's per-query budget.
MAX_OVERPASS_DEPTH = 4

#: Courtesy pause between Overpass calls. The endpoints publish fair-use limits
#: and will start refusing service if hammered, so these stay sequential.
OVERPASS_PAUSE_S = 1.0


def _fetch_osm_tiled(
    kind: str,
    query_body: str,
    bounds: Tuple[float, float, float, float],
    parse: Callable[[List[dict]], List],
    progress: Progress = _noop,
    use_cache: bool = True,
    depth: int = 0,
    _counter: Optional[dict] = None,
) -> List:
    """Fetch one Overpass query over ``bounds``, subdividing when it's too big.

    A whole-county road query blows Overpass's memory/time budget, and query
    cost tracks feature density rather than area -- so rather than guess a cell
    size up front, we try the area whole and split into quadrants only where the
    server pushes back. Rural counties resolve in one request; Cook County
    subdivides until each piece fits.

    Each leaf is cached separately, so a download interrupted three quarters of
    the way through resumes instead of restarting.
    """
    _counter = _counter if _counter is not None else {"n": 0}

    key = f"{kind}|{':'.join(f'{v:.5f}' for v in bounds)}"
    path = _cache_path(f"osm_cell_{kind}", key)
    if use_cache:
        cached = _read_cache(path)
        if cached is not None:
            return list(cached.geometry.values)

    bbox = _bbox_str(bounds)
    query = f"[out:json][timeout:300];{query_body.format(bbox=bbox)}out geom;"

    try:
        if _counter["n"]:
            time.sleep(OVERPASS_PAUSE_S)
        _counter["n"] += 1
        data = _overpass(query, progress=progress)
    except OverpassTooBig as exc:
        if depth >= MAX_OVERPASS_DEPTH:
            raise RuntimeError(
                f"Overpass still refuses this area at depth {depth} ({exc}). "
                "Try a smaller jurisdiction, or turn OSM roads off in the "
                "tiling options and rely on census blocks plus the grid."
            ) from exc
        progress(
            f"Overpass: {kind} area too large at depth {depth} ({exc}); "
            "splitting into 4"
        )
        out: List = []
        for cell in quadsplit(bounds):
            out.extend(
                _fetch_osm_tiled(
                    kind, query_body, cell, parse, progress=progress,
                    use_cache=use_cache, depth=depth + 1, _counter=_counter,
                )
            )
        return out

    geoms = parse(data.get("elements") or [])
    progress(f"OSM {kind}: {len(geoms):,} features (depth {depth})")
    if use_cache:
        _write_cache(path, gpd.GeoDataFrame({"geometry": geoms}, crs="EPSG:4326"))
    return geoms


def _way_coords(el: dict) -> List[Tuple[float, float]]:
    return [(p["lon"], p["lat"]) for p in el.get("geometry") or []]


def _is_closed(coords: Sequence[Tuple[float, float]]) -> bool:
    return len(coords) >= 4 and coords[0] == coords[-1]


def _elements_to_lines(elements: Iterable[dict]) -> List[LineString]:
    out: List[LineString] = []
    for el in elements:
        if el.get("type") == "way":
            coords = _way_coords(el)
            if len(coords) >= 2:
                out.append(LineString(coords))
        elif el.get("type") == "relation":
            for member in el.get("members") or []:
                coords = [(p["lon"], p["lat"]) for p in member.get("geometry") or []]
                if len(coords) >= 2:
                    out.append(LineString(coords))
    return out


def _elements_to_polygons(elements: Iterable[dict]) -> List[Polygon]:
    polys: List[Polygon] = []
    for el in elements:
        etype = el.get("type")
        if etype == "way":
            coords = _way_coords(el)
            if _is_closed(coords):
                try:
                    p = Polygon(coords)
                    if not p.is_valid:
                        p = p.buffer(0)
                    if not p.is_empty:
                        polys.append(p)
                except Exception:
                    continue
        elif etype == "relation":
            outer, inner = [], []
            for member in el.get("members") or []:
                coords = [(p["lon"], p["lat"]) for p in member.get("geometry") or []]
                if len(coords) < 2:
                    continue
                (inner if member.get("role") == "inner" else outer).append(
                    LineString(coords)
                )
            built = _rings_to_polygons(outer, inner)
            polys.extend(built)
    return polys


def _rings_to_polygons(
    outer: List[LineString], inner: List[LineString]
) -> List[Polygon]:
    def close(lines: List[LineString]) -> List[Polygon]:
        if not lines:
            return []
        merged = linemerge(lines)
        parts = list(merged.geoms) if merged.geom_type == "MultiLineString" else [merged]
        made = [p for p in polygonize(parts)]
        return made

    outers = close(outer)
    inners = close(inner)
    if not outers:
        return []
    if not inners:
        return outers
    holes = unary_union(inners)
    result = []
    for o in outers:
        try:
            diff = o.difference(holes)
        except Exception:
            diff = o
        if diff.is_empty:
            continue
        if isinstance(diff, MultiPolygon):
            result.extend(list(diff.geoms))
        elif isinstance(diff, Polygon):
            result.append(diff)
    return result


def _dedupe(geoms: Sequence) -> List:
    """Drop geometries repeated across quadtree cells (ways straddling a cut)."""
    seen: set = set()
    out: List = []
    for g in geoms:
        if g is None or g.is_empty:
            continue
        h = g.wkb
        if h in seen:
            continue
        seen.add(h)
        out.append(g)
    return out


def _fetch_osm(
    kind: str,
    query_body: str,
    parse: Callable[[List[dict]], List],
    jur: Jurisdiction,
    label: str,
    progress: Progress = _noop,
    use_cache: bool = True,
) -> gpd.GeoDataFrame:
    key = f"{kind}|{jur.layer_id}|{jur.geoid}"
    path = _cache_path(f"osm_{kind}", key)
    if use_cache:
        cached = _read_cache(path)
        if cached is not None:
            progress(f"OSM {label}: {len(cached):,} features (cached)")
            return cached

    geoms = _fetch_osm_tiled(
        kind, query_body, jur.bounds(), parse,
        progress=progress, use_cache=use_cache,
    )
    before = len(geoms)
    geoms = _dedupe(geoms)
    if before != len(geoms):
        progress(f"OSM {label}: dropped {before - len(geoms):,} cross-cell duplicates")
    progress(f"OSM {label}: {len(geoms):,} features total")

    gdf = gpd.GeoDataFrame({"geometry": geoms}, crs="EPSG:4326")
    if use_cache:
        _write_cache(path, gdf)
    return gdf


def fetch_osm_roads(
    jur: Jurisdiction, progress: Progress = _noop, use_cache: bool = True
) -> gpd.GeoDataFrame:
    """Road centrelines as LineStrings (EPSG:4326)."""
    return _fetch_osm(
        "roads", f"way{ROAD_FILTER}({{bbox}});", _elements_to_lines,
        jur, "roads", progress=progress, use_cache=use_cache,
    )


def fetch_osm_waterway_lines(
    jur: Jurisdiction, progress: Progress = _noop, use_cache: bool = True
) -> gpd.GeoDataFrame:
    """Rivers/streams/canals as LineStrings (EPSG:4326)."""
    return _fetch_osm(
        "waterlines", f"way{WATERWAY_LINE_FILTER}({{bbox}});", _elements_to_lines,
        jur, "waterways", progress=progress, use_cache=use_cache,
    )


def fetch_osm_water_areas(
    jur: Jurisdiction, progress: Progress = _noop, use_cache: bool = True
) -> gpd.GeoDataFrame:
    """Lakes/reservoirs/riverbanks as Polygons (EPSG:4326), for clipping."""
    body = "(" + "".join(
        f"way{f}({{bbox}});relation{f}({{bbox}});" for f in WATER_AREA_FILTERS
    ) + ");"
    return _fetch_osm(
        "waterareas", body, _elements_to_polygons,
        jur, "water areas", progress=progress, use_cache=use_cache,
    )
