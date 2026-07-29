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


#: Tile fills for the diagnostics views. Muted, so the edge overlay reads on top.
IDLE_FILL = (0.88, 0.88, 0.88)
OK_FILL = (0.80, 0.86, 0.80)
DEFECT_FILL = (0.84, 0.15, 0.16)


def fragment_colors(labels: np.ndarray) -> np.ndarray:
    """Colour tiles by connected-component id.

    Uses the same hashed cycle as ``neighborhood_colors``, so a neighborhood
    that is genuinely one piece looks flat here and a shattered one turns to
    confetti -- which is the whole point of the view.
    """
    return neighborhood_colors(labels)


def defect_colors(weak: np.ndarray) -> np.ndarray:
    """Red where a tile has no shared border with its own neighborhood."""
    weak = np.asarray(weak, dtype=bool)
    out = np.tile(np.asarray(OK_FILL, dtype=np.float64), (len(weak), 1))
    out[weak] = DEFECT_FILL
    return out


def flat_fill(n: int, color=IDLE_FILL) -> np.ndarray:
    return np.tile(np.asarray(color, dtype=np.float64), (max(n, 0), 1))


def dimmed(colors: np.ndarray, amount: float = 0.72) -> np.ndarray:
    """Wash colours out toward white, so a highlight on top reads clearly."""
    colors = np.asarray(colors, dtype=np.float64)
    return colors + (1.0 - colors) * float(amount)


def edge_rgba(
    edge_class: np.ndarray,
    class_colors,
    visible: np.ndarray,
    alpha: float = 0.9,
) -> np.ndarray:
    """(E, 4) colours for the adjacency overlay; invisible edges get alpha 0."""
    edge_class = np.asarray(edge_class, dtype=np.int64)
    visible = np.asarray(visible, dtype=bool)
    palette = np.asarray(class_colors, dtype=np.float64)
    out = np.zeros((len(edge_class), 4), dtype=np.float64)
    if not len(edge_class):
        return out
    out[:, :3] = palette[edge_class]
    out[:, 3] = np.where(visible, alpha, 0.0)
    return out


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
