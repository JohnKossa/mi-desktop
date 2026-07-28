"""CRS helpers.

The optimizer's adjacency threshold is expressed in feet, so every geometry has
to live in a projected CRS whose linear unit is feet. The notebook hardcoded
EPSG:2237 (Florida West) / EPSG:2263 (NY Long Island). For an arbitrary
jurisdiction we pick one automatically:

1. Ask the PROJ database for projected CRSs covering the study area, and take
   the best-fitting one whose unit is a (US survey) foot -- i.e. a State Plane
   zone in ftUS.
2. If nothing matches (rare, but true for a few territories), fall back to a
   locally-centred transverse Mercator defined in US survey feet. That always
   works and is accurate to well under a foot across a county-sized area.
"""

from __future__ import annotations

from typing import Optional, Tuple

from pyproj import CRS
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_crs_info

FOOT_UNITS = {"us survey foot", "foot", "foot_us", "ft", "us_survey_foot"}


def _is_feet(crs: CRS) -> bool:
    try:
        for axis in crs.axis_info:
            name = (axis.unit_name or "").lower().replace("-", " ")
            if "foot" in name or "feet" in name:
                return True
    except Exception:
        pass
    return False


def crs_is_feet(crs) -> bool:
    """Public predicate: does this CRS measure distance in feet?"""
    if crs is None:
        return False
    return _is_feet(CRS.from_user_input(crs))


def local_feet_crs(lon: float, lat: float) -> CRS:
    """A transverse Mercator centred on the study area, in US survey feet."""
    proj = (
        f"+proj=tmerc +lat_0={lat:.6f} +lon_0={lon:.6f} +k=0.9999 "
        f"+x_0=0 +y_0=0 +datum=NAD83 +units=us-ft +no_defs"
    )
    return CRS.from_proj4(proj)


def pick_feet_crs(bounds_lonlat: Tuple[float, float, float, float]) -> CRS:
    """Choose a feet-based projected CRS covering ``bounds_lonlat``.

    ``bounds_lonlat`` is (minx, miny, maxx, maxy) in EPSG:4326.
    """
    minx, miny, maxx, maxy = bounds_lonlat
    cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0

    aoi = AreaOfInterest(
        west_lon_degree=minx,
        south_lat_degree=miny,
        east_lon_degree=maxx,
        north_lat_degree=maxy,
    )

    try:
        candidates = query_crs_info(
            auth_name="EPSG",
            pj_types=["PROJECTED_CRS"],
            area_of_interest=aoi,
            contains=True,
        )
    except Exception:
        candidates = []

    best: Optional[CRS] = None
    best_area = float("inf")
    for info in candidates:
        try:
            crs = CRS.from_authority(info.auth_name, info.code)
        except Exception:
            continue
        if not _is_feet(crs):
            continue
        name = (info.name or "").lower()
        # Skip anything that isn't a plain planar zone.
        if "deprecated" in name:
            continue
        aoi_box = info.area_of_use
        if aoi_box is None:
            continue
        area = (aoi_box.east - aoi_box.west) * (aoi_box.north - aoi_box.south)
        # Prefer NAD83-based State Plane over older NAD27 definitions.
        if "nad27" in name:
            area *= 10
        if area < best_area:
            best_area, best = area, crs

    if best is not None:
        return best
    return local_feet_crs(cx, cy)


def describe_crs(crs) -> str:
    c = CRS.from_user_input(crs)
    epsg = c.to_epsg()
    unit = c.axis_info[0].unit_name if c.axis_info else "?"
    label = c.name or "custom"
    return f"{label} (EPSG:{epsg}, {unit})" if epsg else f"{label} ({unit})"
