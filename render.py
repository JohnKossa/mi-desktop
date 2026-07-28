"""Pure rendering helpers, importable without Qt (used by tests and exports)."""

from __future__ import annotations

from typing import List

import matplotlib
import numpy as np
from shapely.geometry import MultiPolygon, Polygon

# 20 well-separated hues, cycled. Neighborhood ids are arbitrary labels, so a
# qualitative map is the right choice; ids are hashed into the cycle so
# numerically adjacent ids don't land on near-identical colours.
_BASE = np.asarray(matplotlib.colormaps["tab20"].colors, dtype=np.float64)


def neighborhood_colors(n_ids: np.ndarray, seed: int = 12345) -> np.ndarray:
    idx = ((np.asarray(n_ids, dtype=np.int64) * 2654435761 + seed) >> 3) % len(_BASE)
    return _BASE[idx]


def polygon_parts(geom) -> List[np.ndarray]:
    """Exterior rings of a (Multi)Polygon as coordinate arrays."""
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [np.asarray(geom.exterior.coords)]
    if isinstance(geom, MultiPolygon):
        return [np.asarray(p.exterior.coords) for p in geom.geoms if not p.is_empty]
    return []


def build_polygons(geoms, simplify_tolerance: float = 10.0):
    """Flatten tile geometries into (vertex arrays, part -> tile index)."""
    if simplify_tolerance and simplify_tolerance > 0:
        geoms = geoms.simplify(simplify_tolerance, preserve_topology=False)
    verts: List[np.ndarray] = []
    owner: List[int] = []
    for pos, geom in enumerate(geoms.values):
        for part in polygon_parts(geom):
            if len(part) >= 3:
                verts.append(part)
                owner.append(pos)
    return verts, np.asarray(owner, dtype=np.int64)
