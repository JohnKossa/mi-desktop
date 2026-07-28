"""Live matplotlib map of the tileset, coloured by neighborhood.

Redrawing a county's worth of tiles with ``GeoDataFrame.plot`` every thousand
swaps is far too slow -- it rebuilds every path each time. Instead the tile
geometry is converted to a ``PolyCollection`` exactly once, and each refresh
only pushes a new face-colour array, which is a single NumPy gather plus a
canvas repaint.

This module requires Qt. The colour/geometry helpers it uses live in
``render.py`` so they stay importable in headless contexts.
"""

from __future__ import annotations

from typing import Optional

import matplotlib

matplotlib.use("QtAgg")

import numpy as np  # noqa: E402
from matplotlib.backends.backend_qtagg import (  # noqa: E402
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from render import build_polygons, neighborhood_colors  # noqa: E402

__all__ = ["MapCanvas", "make_toolbar", "neighborhood_colors"]


class MapCanvas(FigureCanvasQTAgg):
    """A pannable/zoomable canvas holding one PolyCollection of tiles."""

    def __init__(self, parent=None) -> None:
        self.fig = Figure(figsize=(8, 8), layout="tight")
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111)
        self._style_axes()

        self.collection: Optional[PolyCollection] = None
        self.part_to_tile: Optional[np.ndarray] = None
        self._title = ""

        self.ax.text(
            0.5, 0.5, "No tileset loaded",
            ha="center", va="center", transform=self.ax.transAxes,
            color="#888", fontsize=12,
        )

    def _style_axes(self) -> None:
        self.ax.set_aspect("equal")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for spine in self.ax.spines.values():
            spine.set_visible(False)

    # ------------------------------------------------------------------
    def set_tiles(
        self, geometries, simplify_tolerance: float = 10.0, boundary=None
    ) -> None:
        """Build the PolyCollection. Call once per tileset."""
        self.ax.clear()
        self._style_axes()
        self._title = ""  # ax.clear() wiped the drawn title; forget the cache

        verts, owner = build_polygons(geometries, simplify_tolerance)
        self.part_to_tile = owner
        self.collection = PolyCollection(
            verts, closed=True, linewidths=0.0, antialiaseds=False
        )
        self.collection.set_facecolors(
            np.tile([0.85, 0.85, 0.85], (max(len(verts), 1), 1))
        )
        self.ax.add_collection(self.collection)

        if boundary is not None and len(boundary):
            try:
                boundary.boundary.plot(
                    ax=self.ax, color="#333333", linewidth=0.8, zorder=5
                )
            except Exception:
                pass

        self.ax.autoscale_view()
        try:
            minx, miny, maxx, maxy = geometries.total_bounds
            pad_x, pad_y = (maxx - minx) * 0.02, (maxy - miny) * 0.02
            self.ax.set_xlim(minx - pad_x, maxx + pad_x)
            self.ax.set_ylim(miny - pad_y, maxy + pad_y)
        except Exception:
            pass
        self.draw_idle()

    # ------------------------------------------------------------------
    def update_colors(self, tile_n_ids: np.ndarray, title: str = "") -> None:
        if self.collection is None or self.part_to_tile is None:
            return
        colors = neighborhood_colors(np.asarray(tile_n_ids))
        self.collection.set_facecolors(colors[self.part_to_tile])
        if title and title != self._title:
            self._title = title
            self.ax.set_title(title, fontsize=10, color="#444")
        self.draw_idle()

    # ------------------------------------------------------------------
    def save_png(self, path: str, dpi: int = 200) -> None:
        self.fig.savefig(path, dpi=dpi, bbox_inches="tight")


def make_toolbar(canvas: MapCanvas, parent=None) -> NavigationToolbar2QT:
    return NavigationToolbar2QT(canvas, parent)
