"""Tile construction: 500 ft grid + census blocks + roads + waterways, shattered.

This is the desktop equivalent of ``scratchpad.ipynb``'s ``shatter_into`` /
``clip_using`` pair, generalised to build the whole tileset from remote sources
and ported to the geopandas 1.x / shapely 2.x API (``union_all`` rather than the
deprecated ``unary_union`` accessor).
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import LineString, MultiLineString, box
from shapely.ops import polygonize, unary_union

Progress = Callable[[str], None]


def _noop(_: str) -> None:
    pass


# ==========================================================================
# Grid
# ==========================================================================


def grid_lines(
    bounds: Tuple[float, float, float, float], size_ft: float
) -> List[LineString]:
    """Horizontal + vertical lines on a ``size_ft`` lattice covering ``bounds``.

    Emitting the grid as lines rather than boxes keeps the linework union an
    order of magnitude cheaper than unioning ~200k square polygons.
    """
    minx, miny, maxx, maxy = bounds
    # Snap the origin to the lattice so tiles are stable across runs/extents.
    x0 = np.floor(minx / size_ft) * size_ft
    y0 = np.floor(miny / size_ft) * size_ft
    xs = np.arange(x0, maxx + size_ft, size_ft)
    ys = np.arange(y0, maxy + size_ft, size_ft)

    lines: List[LineString] = []
    for x in xs:
        lines.append(LineString([(x, y0), (x, ys[-1])]))
    for y in ys:
        lines.append(LineString([(x0, y), (xs[-1], y)]))
    return lines


def grid_cells(
    bounds: Tuple[float, float, float, float], size_ft: float, crs
) -> gpd.GeoDataFrame:
    """The same lattice as square polygons (useful for inspection/debug)."""
    minx, miny, maxx, maxy = bounds
    x0 = np.floor(minx / size_ft) * size_ft
    y0 = np.floor(miny / size_ft) * size_ft
    xs = np.arange(x0, maxx + size_ft, size_ft)
    ys = np.arange(y0, maxy + size_ft, size_ft)
    cells = [
        box(xs[i], ys[j], xs[i + 1], ys[j + 1])
        for i in range(len(xs) - 1)
        for j in range(len(ys) - 1)
    ]
    return gpd.GeoDataFrame({"geometry": cells}, crs=crs)


# ==========================================================================
# Shatter / clip (ported from scratchpad.ipynb)
# ==========================================================================


def _explode_lines(geoms: Sequence) -> List:
    out = []
    for g in geoms:
        if g is None or g.is_empty:
            continue
        if isinstance(g, MultiLineString):
            out.extend(list(g.geoms))
        else:
            out.append(g)
    return out


def shatter(
    line_sources: Sequence[gpd.GeoSeries],
    crs,
    progress: Progress = _noop,
    min_area_sqft: float = 100.0,
) -> gpd.GeoDataFrame:
    """Node all linework together and polygonize it into tiles."""
    all_lines: List = []
    for series in line_sources:
        if series is None or len(series) == 0:
            continue
        s = series
        # Polygons contribute their boundaries; lines contribute themselves.
        geom_types = set(s.geom_type.dropna().unique())
        if geom_types & {"Polygon", "MultiPolygon"}:
            s = s.boundary
        all_lines.extend(_explode_lines(list(s.values)))

    if not all_lines:
        raise ValueError("No linework supplied to shatter().")

    progress(f"Shatter: noding {len(all_lines):,} line segments...")
    noded = unary_union(all_lines)

    progress("Shatter: polygonizing...")
    polys = list(polygonize(noded))
    progress(f"Shatter: {len(polys):,} raw faces")

    tiles = gpd.GeoDataFrame({"geometry": polys}, crs=crs)
    if min_area_sqft > 0:
        before = len(tiles)
        tiles = tiles[tiles.geometry.area >= min_area_sqft]
        if before != len(tiles):
            progress(f"Shatter: dropped {before - len(tiles):,} slivers")
    return tiles.reset_index(drop=True)


def clip_out(
    base: gpd.GeoDataFrame,
    mask: gpd.GeoDataFrame,
    progress: Progress = _noop,
    min_area_sqft: float = 100.0,
) -> gpd.GeoDataFrame:
    """Subtract ``mask`` (e.g. water) from an existing tileset.

    ``build_tileset`` no longer needs this: it removes water from the study-area
    polygon *before* shattering, which is one difference instead of one per tile.
    Kept for clipping a tileset that has already been built.

    Deliberately *not* ``base.geometry.difference(mask.union_all())``. That is the
    obvious one-liner and it is a trap: the difference is elementwise against a
    single MultiPolygon, so every tile pays the cost of the entire mask. With
    380k tiles and a county's worth of OSM water (Cape Coral's canal network
    alone runs to thousands of polygons) that measured at ~11 minutes and looks
    exactly like a hang.

    Instead, index the mask and only touch tiles that actually meet it -- usually
    a small minority -- differencing each against just the mask parts that
    overlap it. Same result, because parts that don't intersect a tile cannot
    change it.
    """
    if mask is None or len(mask) == 0:
        return base
    if base.crs != mask.crs:
        mask = mask.to_crs(base.crs)

    base_geoms = base.geometry.to_numpy()
    mask_geoms = mask.geometry.to_numpy()

    progress(f"Clip: indexing {len(mask):,} mask polygons...")
    pairs = mask.sindex.query(base_geoms, predicate="intersects")
    tile_idx, water_idx = pairs[0], pairs[1]
    if len(tile_idx) == 0:
        progress("Clip: no tile meets the mask; nothing to do")
        return base

    order = np.argsort(tile_idx, kind="stable")
    tile_idx, water_idx = tile_idx[order], water_idx[order]
    affected, starts, counts = np.unique(
        tile_idx, return_index=True, return_counts=True
    )
    progress(
        f"Clip: {len(affected):,} of {len(base):,} tiles meet the mask "
        f"({len(tile_idx):,} overlaps)"
    )

    result = base_geoms.copy()
    step = max(1, len(affected) // 10)
    for k, (pos, start, count) in enumerate(zip(affected, starts, counts)):
        cut = mask_geoms[water_idx[start]] if count == 1 else shapely.union_all(
            mask_geoms[water_idx[start : start + count]]
        )
        result[pos] = shapely.difference(result[pos], cut)
        if k and k % step == 0:
            progress(f"Clip: {k:,}/{len(affected):,} tiles clipped")

    out = gpd.GeoDataFrame({"geometry": result}, crs=base.crs)
    out = out[~out.geometry.is_empty & out.geometry.notna()]
    out = out.explode(ignore_index=True, index_parts=False)
    if min_area_sqft > 0:
        out = out[out.geometry.area >= min_area_sqft]
    progress(f"Clip: {len(out):,} tiles remain")
    return out.reset_index(drop=True)


# ==========================================================================
# Full tileset build
# ==========================================================================


def build_tileset(
    study_area: gpd.GeoDataFrame,
    blocks: Optional[gpd.GeoDataFrame],
    roads: Optional[gpd.GeoDataFrame],
    waterway_lines: Optional[gpd.GeoDataFrame],
    water_areas: Optional[gpd.GeoDataFrame],
    working_crs,
    grid_size_ft: float = 500.0,
    clip_water: bool = True,
    progress: Progress = _noop,
) -> gpd.GeoDataFrame:
    """Assemble the tileset for a jurisdiction.

    All inputs may be in any CRS; everything is reprojected to ``working_crs``
    (which must be feet-based) before the shatter.
    """

    def prep(gdf: Optional[gpd.GeoDataFrame]) -> Optional[gpd.GeoDataFrame]:
        if gdf is None or len(gdf) == 0:
            return None
        g = gdf.to_crs(working_crs)
        g = g[~g.geometry.is_empty & g.geometry.notna()]
        return g if len(g) else None

    area = study_area.to_crs(working_crs)
    area_union = area.geometry.union_all()
    # Grid origin comes from the *un-clipped* bounds so the lattice is identical
    # whether or not water is removed.
    bounds = area_union.bounds

    blocks = prep(blocks)
    roads = prep(roads)
    waterway_lines = prep(waterway_lines)
    water_areas = prep(water_areas)

    if clip_water and water_areas is not None:
        # Cut water out of the study area *before* shattering rather than out of
        # the finished tiles afterwards: one difference on one polygon instead of
        # one per tile. Post-hoc clipping of 380k tiles measured at ~11 minutes.
        # Two things then fall out for free -- the water outlines join the
        # linework via the area boundary, so tiles stop at the shoreline, and the
        # jurisdiction trim below drops water faces without a separate pass.
        progress(f"Tileset: removing {len(water_areas):,} water polygons from the "
                 "study area...")
        t0 = time.perf_counter()
        water_union = water_areas.geometry.union_all()
        remaining = area_union.difference(water_union)
        if remaining.is_empty:
            progress(
                "Tileset: WARNING water covers the entire study area; ignoring the "
                "water clip so there is something left to tile."
            )
        else:
            area_union = remaining
            progress(f"Tileset: water removed in {time.perf_counter() - t0:.1f}s; "
                     "shorelines are now cut lines")

    sources: List[gpd.GeoSeries] = [gpd.GeoSeries([area_union], crs=working_crs)]

    if blocks is not None:
        progress(f"Tileset: {len(blocks):,} census blocks as cut lines")
        sources.append(blocks.geometry)
    if roads is not None:
        progress(f"Tileset: {len(roads):,} road ways as cut lines")
        sources.append(roads.geometry)
    if waterway_lines is not None:
        progress(f"Tileset: {len(waterway_lines):,} waterway ways as cut lines")
        sources.append(waterway_lines.geometry)

    progress(f"Tileset: building {grid_size_ft:.0f} ft grid...")
    glines = grid_lines(bounds, grid_size_ft)
    progress(f"Tileset: {len(glines):,} grid lines")
    sources.append(gpd.GeoSeries(glines, crs=working_crs))

    tiles = shatter(sources, working_crs, progress=progress)

    # Everything outside the study area gets dropped -- including water, now that
    # it has been subtracted from it. The area boundary is in the linework, so
    # every face is cleanly inside or outside.
    progress("Tileset: trimming to the study area...")
    t0 = time.perf_counter()
    points = tiles.geometry.representative_point().to_numpy()
    # prepare() builds an internal index on the polygon. Without it, each of the
    # ~380k point-in-polygon tests walks every ring of what is now a
    # water-perforated multipolygon -- the same quadratic trap the clip had.
    shapely.prepare(area_union)
    # contains(area, point) is exactly the old point.within(area).
    tiles = tiles[shapely.contains(area_union, points)]
    progress(
        f"Tileset: {len(tiles):,} tiles inside the study area "
        f"({time.perf_counter() - t0:.1f}s)"
    )

    tiles = tiles.reset_index(drop=True)
    tiles.index.name = "tile_id"
    progress(f"Tileset: {len(tiles):,} final tiles")
    return tiles


# ==========================================================================
# Parcel <-> tile mapping and tile adjacency
# ==========================================================================


def assign_parcels_to_tiles(
    parcels: gpd.GeoDataFrame,
    tiles: gpd.GeoDataFrame,
    progress: Progress = _noop,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, Dict[int, np.ndarray]]:
    """Join parcels to tiles; orphans become their own virtual tiles.

    Returns ``(parcels_with_tile_id, all_tiles, tile_to_parcels)`` where
    ``tile_to_parcels`` maps tile id -> positional indices into the returned
    parcel frame (matching the optimizer's expectations).
    """
    if tiles.crs != parcels.crs:
        tiles = tiles.to_crs(parcels.crs)

    tiles = tiles.sort_index()
    progress(f"Joining {len(parcels):,} parcels to {len(tiles):,} tiles...")

    # Join on representative point so each parcel lands in exactly one tile
    # (parcels straddling a cut line would otherwise match several).
    # .to_numpy(), not the GeoSeries: passing a Series alongside an explicit
    # index makes geopandas *reindex* it, which would silently produce an
    # all-None geometry column if the two indices ever diverged.
    points = gpd.GeoDataFrame(
        geometry=parcels.geometry.representative_point().to_numpy(),
        index=parcels.index,
        crs=parcels.crs,
    )
    # sjoin names the right-hand index column after the index (here "tile_id"),
    # or "index_right" when it is unnamed -- carry it as an explicit column so
    # neither case needs special handling.
    right = gpd.GeoDataFrame(
        {"_tid": np.asarray(tiles.index, dtype=np.int64)},
        geometry=tiles.geometry.values,
        crs=tiles.crs,
    )
    joined = gpd.sjoin(points, right, how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]

    out = parcels.copy()
    out["tile_id"] = joined["_tid"].reindex(out.index).values

    orphan_mask = out["tile_id"].isna().to_numpy()
    n_orphans = int(orphan_mask.sum())
    max_tile = int(tiles.index.max()) if len(tiles) else -1

    virtual: Optional[gpd.GeoDataFrame] = None
    if n_orphans:
        # A parcel that lands in no tile (outside the shatter, or in a hole
        # punched by the water clip) becomes a tile of its own, per SPEC_TILED.
        progress(f"{n_orphans:,} orphan parcels -> virtual tiles")
        virtual_ids = np.arange(max_tile + 1, max_tile + 1 + n_orphans, dtype=np.int64)
        out.loc[orphan_mask, "tile_id"] = virtual_ids
        virtual = gpd.GeoDataFrame(
            geometry=out.geometry.to_numpy()[orphan_mask],
            index=pd.Index(virtual_ids, name="tile_id"),
            crs=parcels.crs,
        )

    out["tile_id"] = out["tile_id"].astype("int64")
    out = out.reset_index(drop=True)

    base = gpd.GeoDataFrame(
        geometry=tiles.geometry.to_numpy(),
        index=pd.Index(np.asarray(tiles.index, dtype=np.int64), name="tile_id"),
        crs=parcels.crs,
    )
    all_tiles = base if virtual is None else gpd.GeoDataFrame(
        pd.concat([base, virtual]), crs=parcels.crs
    )
    all_tiles = all_tiles.sort_index()

    tile_to_parcels = {
        int(t): np.asarray(idx, dtype=np.int64)
        for t, idx in out.groupby("tile_id").indices.items()
    }
    progress(f"{len(tile_to_parcels):,} tiles contain at least one parcel")
    return out, all_tiles, tile_to_parcels


def calculate_adjacency(
    gdf: gpd.GeoDataFrame,
    threshold_ft: float = 100.0,
    progress: Progress = _noop,
) -> Dict[int, set]:
    """Neighbours within ``threshold_ft``, as an id -> set-of-ids dict.

    Same semantics as the notebook's ``calculate_adjacency`` but vectorised:
    a single bulk spatial-index query instead of one query per geometry.
    """
    if gdf.crs is None:
        raise ValueError("DataFrame has no CRS; project to a feet-based CRS first.")
    units = [
        (a.unit_name or "").lower() for a in gdf.crs.axis_info if hasattr(a, "unit_name")
    ]
    if not any("foot" in u or "feet" in u for u in units):
        raise ValueError(
            f"CRS units are not feet (detected {units}); reproject before adjacency."
        )

    ids = np.asarray(gdf.index)
    geoms = gdf.geometry.values
    sindex = gdf.sindex

    progress(f"Adjacency: querying {len(gdf):,} geometries at {threshold_ft:.0f} ft...")
    try:
        pairs = sindex.query(geoms, predicate="dwithin", distance=threshold_ft)
    except (TypeError, ValueError):
        # Older GEOS without the dwithin predicate: fall back to buffering.
        progress("Adjacency: dwithin unavailable, buffering instead")
        buffered = gdf.geometry.buffer(threshold_ft).values
        pairs = sindex.query(buffered, predicate="intersects")

    left, right = pairs[0], pairs[1]
    keep = left != right
    left, right = left[keep], right[keep]

    adj: Dict[int, set] = {int(i): set() for i in ids}
    for li, ri in zip(ids[left], ids[right]):
        adj[int(li)].add(int(ri))

    edges = sum(len(v) for v in adj.values()) // 2
    progress(
        f"Adjacency: {len(gdf):,} tiles, {edges:,} edges, "
        f"{edges / max(len(gdf), 1):.2f} avg"
    )
    return adj
