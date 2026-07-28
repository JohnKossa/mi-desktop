"""Bulk census geography from TIGER/Line shapefiles.

The TIGERweb REST service is fine for looking up one boundary, but it caps a
single ``returnIdsOnly`` response at 100,000 features, and fetching blocks 1,000
at a time means dozens of round trips for a county. Lee County (~30k blocks)
squeaks under the cap; Cook County (Chicago) does not.

So for blocks we go to the source: Census publishes one zipped shapefile of
2020 tabulation blocks per state at

    https://www2.census.gov/geo/tiger/TIGER<year>/TABBLOCK20/tl_<year>_<ss>_tabblock20.zip

One download (tens of MB), cached on disk indefinitely, then read through GDAL's
``/vsizip/`` with a bounding-box prefilter so only the study area's blocks are
materialised. No caps, no pagination, and the second run of any jurisdiction in
that state is fully offline.

The vintage year isn't hardcoded: available years are probed newest-first at
runtime and the answer is cached, so a new Census release doesn't need a code
change.
"""

from __future__ import annotations

import datetime as _dt
import json
import shutil
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import geopandas as gpd
import pandas as pd
import requests

from config import cache_dir

Progress = Callable[[str], None]


def _noop(_: str) -> None:
    pass


TIGER_BASE = "https://www2.census.gov/geo/tiger"
USER_AGENT = "mi-neighborhoods-desktop/0.1 (parcel neighborhood optimizer)"

#: TIGER/Line vintages to probe, newest first. 2020 is the floor because
#: TABBLOCK20 (2020 census blocks) first appears then.
def candidate_years(today: Optional[_dt.date] = None) -> List[int]:
    year = (today or _dt.date.today()).year
    # Census publishes the next vintage late in the calendar year, so start one
    # ahead and let the probe reject it.
    return list(range(year + 1, 2019, -1))


def blocks_url(year: int, state_fips: str) -> str:
    ss = str(state_fips).zfill(2)
    return f"{TIGER_BASE}/TIGER{year}/TABBLOCK20/tl_{year}_{ss}_tabblock20.zip"


# ==========================================================================
# HTTP
# ==========================================================================


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def _url_exists(session: requests.Session, url: str, timeout: int = 30) -> bool:
    """HEAD the URL, falling back to a one-byte ranged GET.

    Some CDN configurations in front of www2.census.gov answer HEAD with 403
    even when the object is served fine, so a HEAD failure alone isn't proof.
    """
    try:
        r = session.head(url, timeout=timeout, allow_redirects=True)
        if r.status_code == 200:
            return True
        if r.status_code not in (403, 405, 501):
            return False
    except requests.RequestException:
        pass
    try:
        r = session.get(
            url, timeout=timeout, allow_redirects=True,
            headers={"Range": "bytes=0-0"}, stream=True,
        )
        r.close()
        return r.status_code in (200, 206)
    except requests.RequestException:
        return False


# ==========================================================================
# Vintage resolution
# ==========================================================================


def _year_cache_path() -> Path:
    return cache_dir() / "tiger_vintage.json"


def _load_year_cache() -> dict:
    p = _year_cache_path()
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_year_cache(data: dict) -> None:
    try:
        _year_cache_path().write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass


def resolve_year(
    state_fips: str,
    progress: Progress = _noop,
    years: Optional[Sequence[int]] = None,
    use_cache: bool = True,
) -> int:
    """Newest TIGER vintage that actually has this state's block file."""
    ss = str(state_fips).zfill(2)
    cache = _load_year_cache() if use_cache else {}
    if ss in cache:
        return int(cache[ss])

    with _session() as s:
        for year in years or candidate_years():
            url = blocks_url(year, ss)
            progress(f"Probing TIGER{year} for state {ss}...")
            if _url_exists(s, url):
                progress(f"Using TIGER{year} block geography")
                cache[ss] = year
                if use_cache:
                    _save_year_cache(cache)
                return year

    raise RuntimeError(
        f"No TIGER/Line block file found for state {ss}. Checked "
        f"{TIGER_BASE}/TIGER<year>/TABBLOCK20/. Check the network connection, "
        "or turn census blocks off in the tiling options."
    )


# ==========================================================================
# Download
# ==========================================================================


def _tiger_dir() -> Path:
    d = cache_dir() / "tiger"
    d.mkdir(parents=True, exist_ok=True)
    return d


def download_state_blocks(
    state_fips: str,
    progress: Progress = _noop,
    year: Optional[int] = None,
    use_cache: bool = True,
) -> Path:
    """Fetch (or reuse) one state's zipped block shapefile. Returns its path."""
    ss = str(state_fips).zfill(2)
    year = year or resolve_year(ss, progress=progress)
    dest = _tiger_dir() / f"tl_{year}_{ss}_tabblock20.zip"

    if use_cache and dest.exists() and dest.stat().st_size > 0:
        progress(f"Census blocks: reusing {dest.name} "
                 f"({dest.stat().st_size / 1e6:.0f} MB, cached)")
        return dest

    url = blocks_url(year, ss)
    tmp = dest.with_suffix(".part")
    progress(f"Census blocks: downloading {dest.name}...")

    with _session() as s:
        with s.get(url, stream=True, timeout=(30, 300)) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length") or 0)
            done = 0
            step = 8 << 20  # log every 8 MB
            next_log = step
            with open(tmp, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if not chunk:
                        continue
                    fh.write(chunk)
                    done += len(chunk)
                    if done >= next_log:
                        next_log += step
                        if total:
                            progress(
                                f"Census blocks: {done / 1e6:.0f}/{total / 1e6:.0f} MB"
                            )
                        else:
                            progress(f"Census blocks: {done / 1e6:.0f} MB")

    # Rename only once complete, so an interrupted download is never mistaken
    # for a usable cache entry.
    shutil.move(str(tmp), str(dest))
    progress(f"Census blocks: saved {dest.name} ({dest.stat().st_size / 1e6:.0f} MB)")
    return dest


# ==========================================================================
# Reading
# ==========================================================================


def read_blocks(
    zip_path: Path,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    progress: Progress = _noop,
) -> gpd.GeoDataFrame:
    """Read blocks from a TIGER zip, optionally prefiltered to ``bbox``.

    ``bbox`` is lon/lat. TIGER ships EPSG:4269 (NAD83 geographic), which differs
    from EPSG:4326 by well under a metre -- irrelevant for a bounding-box
    prefilter, and the exact trim happens later against the real boundary.
    """
    import pyogrio

    vsi = f"/vsizip/{Path(zip_path).as_posix()}"
    progress(f"Census blocks: reading {Path(zip_path).name}...")
    # columns=[] keeps geometry only; block attributes are unused here because
    # blocks serve purely as cut lines for the shatter.
    gdf = pyogrio.read_dataframe(vsi, bbox=bbox, columns=[])
    progress(f"Census blocks: {len(gdf):,} in the study bbox")

    if gdf.crs is None:
        gdf = gdf.set_crs(4269)
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()]
    return gdf.reset_index(drop=True)


# ==========================================================================
# Top level
# ==========================================================================


def states_for_bounds(
    bounds: Tuple[float, float, float, float], progress: Progress = _noop
) -> List[str]:
    """State FIPS codes intersecting a lon/lat bbox (fallback path only).

    A named place, county subdivision or county always lies within one state, so
    this is only needed when the jurisdiction's own state code is unknown.
    """
    import sources  # imported lazily to avoid a circular import

    minx, miny, maxx, maxy = bounds
    try:
        fc = sources._query_layer(
            sources.LAYER_STATES,
            "1=1",
            out_fields="STATE",
            geometry={
                "xmin": minx, "ymin": miny, "xmax": maxx, "ymax": maxy,
                "spatialReference": {"wkid": 4326},
            },
        )
    except Exception as exc:  # noqa: BLE001
        progress(f"Could not determine states from bounds: {exc}")
        return []
    out = []
    for feat in fc.get("features") or []:
        code = str((feat.get("properties") or {}).get("STATE", "")).zfill(2)
        if code and code not in out:
            out.append(code)
    return out


def fetch_blocks(
    jurisdiction,
    progress: Progress = _noop,
    use_cache: bool = True,
) -> gpd.GeoDataFrame:
    """Blocks covering a jurisdiction, from bulk TIGER/Line shapefiles."""
    bounds = jurisdiction.bounds()

    states = [str(jurisdiction.state_fips).zfill(2)] if jurisdiction.state_fips else []
    if not states:
        progress("Jurisdiction has no state code; resolving from bounds...")
        states = states_for_bounds(bounds, progress=progress)
    if not states:
        raise RuntimeError("Could not determine which state to fetch blocks for.")

    frames: List[gpd.GeoDataFrame] = []
    for ss in states:
        zip_path = download_state_blocks(ss, progress=progress, use_cache=use_cache)
        frames.append(read_blocks(zip_path, bbox=bounds, progress=progress))

    if len(frames) == 1:
        return frames[0]
    return gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True), crs=frames[0].crs
    ).reset_index(drop=True)
