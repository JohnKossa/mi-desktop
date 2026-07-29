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

from typing import Callable, Optional

import matplotlib

matplotlib.use("QtAgg")

import numpy as np  # noqa: E402
from PySide6 import QtCore  # noqa: E402
from matplotlib.backends.backend_qtagg import (  # noqa: E402
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from render import build_polygons, neighborhood_colors  # noqa: E402

__all__ = ["MapCanvas", "DiagnosticCanvas", "make_toolbar", "neighborhood_colors"]


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
        self.edges: Optional[LineCollection] = None
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
        self.edges = None  # ...and took any edge overlay with it

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
        self.set_face_colors(colors, title)

    def set_face_colors(self, colors: np.ndarray, title: str = "") -> None:
        """Paint tiles from an explicit (n_tiles, 3 or 4) colour array.

        ``update_colors`` is the neighborhood-id shortcut onto this; the
        diagnostics views need to colour by things that are not neighborhood
        ids (fragment index, defect flags) and come in here directly.
        """
        if self.collection is None or self.part_to_tile is None:
            return
        colors = np.asarray(colors, dtype=np.float64)
        self.collection.set_facecolors(colors[self.part_to_tile])
        self.set_title(title)
        self.draw_idle()

    def set_title(self, title: str) -> None:
        if title and title != self._title:
            self._title = title
            self.ax.set_title(title, fontsize=10, color="#444")

    # ------------------------------------------------------------------
    # Adjacency overlay
    # ------------------------------------------------------------------

    def set_edges(self, segments: np.ndarray, linewidth: float = 0.7) -> None:
        """Install the adjacency overlay: one line segment per graph edge.

        Built once and then recoloured, for the same reason the tiles are: a
        county's graph runs to a couple of hundred thousand segments, and
        rebuilding the collection every time a checkbox is toggled is visibly
        slow. Hiding an edge is an alpha of zero, not a rebuild.
        """
        if self.edges is not None:
            self.edges.remove()
            self.edges = None
        segments = np.asarray(segments, dtype=np.float64)
        if not len(segments):
            self.draw_idle()
            return
        self.edges = LineCollection(
            segments, linewidths=linewidth, zorder=6, antialiaseds=True
        )
        self.edges.set_colors(np.zeros((len(segments), 4)))
        self.ax.add_collection(self.edges)
        self.draw_idle()

    def color_edges(self, rgba: np.ndarray) -> None:
        """Recolour the overlay. An alpha of 0 hides that edge."""
        if self.edges is None:
            return
        self.edges.set_colors(np.asarray(rgba, dtype=np.float64))
        self.draw_idle()

    def clear_edges(self) -> None:
        if self.edges is not None:
            self.edges.remove()
            self.edges = None
            self.draw_idle()

    # ------------------------------------------------------------------
    def save_png(self, path: str, dpi: int = 200) -> None:
        self.fig.savefig(path, dpi=dpi, bbox_inches="tight")


class DiagnosticCanvas(MapCanvas):
    """A MapCanvas with the extras the tile-diagnostics tab needs.

    Kept apart from ``MapCanvas`` so the live optimization map -- which repaints
    every few hundred iterations and wants to stay as cheap as possible -- does
    not carry any of this.

    Three overlays sit on top of the tiles (borders, node dots, and the faint
    underlay of tiles the optimizer does not own), and all three are useless
    when the whole county is in view: 64,000 outlined tiles render as a solid
    dark mass. So each is *requested* by the caller and *granted* by
    ``_apply_density``, which counts what is actually on screen after every pan
    or zoom and withholds the overlay until it would be readable.
    """

    #: More visible tiles than this and the overlays are noise, not information.
    DENSITY_LIMIT = 4000
    #: Pan/zoom fires xlim_changed continuously; recount only once it settles.
    SETTLE_MS = 120

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.nodes = None
        self.underlay: Optional[PolyCollection] = None
        self.points: Optional[np.ndarray] = None

        self._want = {"borders": False, "nodes": False, "underlay": False}
        self._granted = dict(self._want)
        self._visible_tiles = 0
        self.on_density_change: Optional[Callable[[int, dict], None]] = None
        self.on_tile_clicked: Optional[Callable[[Optional[int]], None]] = None

        self._settle = QtCore.QTimer(self)
        self._settle.setSingleShot(True)
        self._settle.setInterval(self.SETTLE_MS)
        self._settle.timeout.connect(self._apply_density)
        self.ax.callbacks.connect("xlim_changed", self._on_limits_changed)
        self.mpl_connect("button_press_event", self._on_click)

    # ------------------------------------------------------------------
    def set_tiles(self, geometries, simplify_tolerance=10.0, boundary=None) -> None:
        # ax.clear() in the base drops every artist, so forget them all here
        # rather than leaving dangling references that would fail to remove().
        self.nodes = None
        self.underlay = None
        super().set_tiles(geometries, simplify_tolerance, boundary)

    def set_nodes(self, points: np.ndarray) -> None:
        """One dot per tile, at the exact point the edges are drawn from."""
        self.points = np.asarray(points, dtype=np.float64)
        if self.nodes is not None:
            self.nodes.remove()
            self.nodes = None
        if not len(self.points):
            return
        self.nodes = self.ax.scatter(
            self.points[:, 0], self.points[:, 1],
            s=3.0, c="#222222", marker="o", linewidths=0, zorder=7,
        )
        self.nodes.set_visible(False)
        self._apply_density()

    def set_underlay(self, geometries, simplify_tolerance: float = 10.0) -> None:
        """Tiles that exist but hold no modeled parcel, drawn beneath the rest.

        These are what the gap-bridging edges hop over, and without them a gap
        in the map is indistinguishable from a road, a lake, or the edge of the
        study area.
        """
        if self.underlay is not None:
            self.underlay.remove()
            self.underlay = None
        verts, _ = build_polygons(geometries, simplify_tolerance)
        if not verts:
            self.draw_idle()
            return
        self.underlay = PolyCollection(
            verts, closed=True, facecolors="#f2ede3", edgecolors="#d8cdb8",
            linewidths=0.25, antialiaseds=False, zorder=0,
        )
        self.underlay.set_visible(False)
        self.ax.add_collection(self.underlay)
        self._apply_density()

    # ------------------------------------------------------------------
    def request(self, **wanted: bool) -> None:
        """Ask for overlays. Density decides whether they are actually shown."""
        self._want.update({k: bool(v) for k, v in wanted.items() if k in self._want})
        self._apply_density()

    def _on_limits_changed(self, _ax) -> None:
        self._settle.start()

    def _apply_density(self) -> None:
        """Grant each requested overlay only if the view is sparse enough."""
        self._visible_tiles = self._count_visible()
        sparse = self._visible_tiles <= self.DENSITY_LIMIT
        granted = {k: (v and sparse) for k, v in self._want.items()}

        if self.collection is not None:
            if granted["borders"]:
                self.collection.set_linewidths(0.3)
                self.collection.set_edgecolors((0.25, 0.25, 0.25, 0.85))
            else:
                self.collection.set_linewidths(0.0)
                self.collection.set_edgecolors("none")
        if self.nodes is not None:
            self.nodes.set_visible(granted["nodes"])
        if self.underlay is not None:
            self.underlay.set_visible(granted["underlay"])

        self._granted = granted
        if self.on_density_change is not None:
            self.on_density_change(self._visible_tiles, dict(granted))
        self.draw_idle()

    def _count_visible(self) -> int:
        """Tiles whose node lies in the current view.

        Counting points rather than querying polygon geometry keeps this to one
        NumPy pass, which matters because it runs after every zoom.
        """
        if self.points is None or not len(self.points):
            return 0
        x0, x1 = sorted(self.ax.get_xlim())
        y0, y1 = sorted(self.ax.get_ylim())
        px, py = self.points[:, 0], self.points[:, 1]
        return int(np.count_nonzero((px >= x0) & (px <= x1) & (py >= y0) & (py <= y1)))

    @property
    def suppressed(self) -> bool:
        return any(self._want[k] and not self._granted[k] for k in self._want)

    @property
    def visible_tiles(self) -> int:
        return self._visible_tiles

    # ------------------------------------------------------------------
    def _on_click(self, event) -> None:
        # Only a plain left click in the axes selects. While the toolbar is in
        # pan or zoom mode the drag belongs to it, not to us.
        if self.on_tile_clicked is None or event.inaxes is not self.ax:
            return
        if event.button != 1:
            return
        if getattr(getattr(self, "toolbar", None), "mode", ""):
            return
        if event.xdata is None or event.ydata is None:
            return
        self.on_tile_clicked((float(event.xdata), float(event.ydata)))


def make_toolbar(canvas: MapCanvas, parent=None) -> NavigationToolbar2QT:
    return NavigationToolbar2QT(canvas, parent)
