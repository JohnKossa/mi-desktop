"""PySide6 desktop front end.

Layout: a setup/control column on the left, the live tile map on the right.
All slow work (TIGERweb + Overpass downloads, the shatter, and the annealing
loop itself) runs on background QThreads; the map is repainted from the GUI
thread whenever the optimizer emits a snapshot.
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

import diagnostics
import fields
import pipeline
import pyarrow
import render
import sources
from checkpoints import Checkpoint, CheckpointStore, find_runs
from config import (
    CONTINUOUS_VARIABLES,
    DEFAULT_WEIGHTS,
    RunConfig,
    runs_dir,
)
from engine import OptimizerStats
from mapview import MapCanvas, make_toolbar, neighborhood_colors
from sources import Jurisdiction


# ==========================================================================
# Background threads
# ==========================================================================


class Task(QtCore.QThread):
    """Runs ``fn(log)`` off the GUI thread and emits its return value."""

    log = QtCore.Signal(str)
    failed = QtCore.Signal(str)
    done = QtCore.Signal(object)

    def __init__(self, fn: Callable[[Callable[[str], None]], object], parent=None):
        super().__init__(parent)
        self._fn = fn

    def run(self) -> None:  # noqa: D102
        try:
            result = self._fn(self.log.emit)
        except Exception:  # noqa: BLE001 - surfaced in the UI log
            self.failed.emit(traceback.format_exc())
            return
        self.done.emit(result)


class OptimizeThread(QtCore.QThread):
    log = QtCore.Signal(str)
    failed = QtCore.Signal(str)
    stats = QtCore.Signal(object)
    snapshot = QtCore.Signal(object, object)
    finished_run = QtCore.Signal(object, object)

    def __init__(self, prep: "pipeline.PreparedRun", parent=None):
        super().__init__(parent)
        self.prep = prep
        self.runner = None  # set when annealing in parallel

    def run(self) -> None:  # noqa: D102
        opt = self.prep.optimizer
        opt.progress = self.log.emit
        try:
            if self.prep.worker_count() > 1:
                self._run_parallel()
            else:
                opt.run(
                    store=self.prep.store,
                    stats_cb=self.stats.emit,
                    snapshot_cb=lambda ids, s: self.snapshot.emit(ids, s),
                )
        except Exception:  # noqa: BLE001
            self.failed.emit(traceback.format_exc())
            return

        self.runner = None
        # Written here rather than in the finished slot: on a county-sized
        # parcel set this is seconds of IO that would otherwise freeze the UI.
        out = self.prep.run_dir / "optimized_neighborhoods_tiled.parquet"
        try:
            opt.result_frame(self.prep.parcels).to_parquet(out)
            self.log.emit(f"Wrote {out}")
        except Exception as exc:  # noqa: BLE001
            self.log.emit(f"Could not write result: {exc}")
            out = None
        self.finished_run.emit(self.prep, out)

    # ------------------------------------------------------------------

    def _run_parallel(self) -> None:
        """Anneal each severed component group in its own process."""
        prep = self.prep
        runner = prep.make_parallel_runner(progress=self.log.emit)
        self.runner = runner

        def on_checkpoint(tile_n_ids, group_state) -> None:
            try:
                ids = prep.parcel_ids_from_tiles(tile_n_ids)
                states = list(group_state.values()) or [{}]
                cp = Checkpoint(
                    iteration=max(int(s.get("iteration", 0)) for s in states),
                    temperature=max(float(s.get("temperature", 0.0)) for s in states),
                    stability_counter=0,
                    accepted=sum(int(s.get("accepted", 0)) for s in states),
                    rejected=sum(int(s.get("rejected", 0)) for s in states),
                    mean_score=0.0,
                    n_neighborhoods=prep.optimizer.n_neighborhoods,
                    parcel_n_ids=ids,
                    rng_state=None,
                    # Per-group annealing state: each worker owns its own
                    # cooling schedule, so one global temperature won't do.
                    extra={"groups": {str(k): v for k, v in group_state.items()}},
                )
                self.log.emit(f"Checkpoint saved: {prep.store.save(cp)}")
            except Exception as exc:  # noqa: BLE001
                self.log.emit(f"Parallel checkpoint failed: {exc}")

        merged = runner.run(
            stats_cb=self.stats.emit,
            snapshot_cb=lambda ids, s: self.snapshot.emit(ids, s),
            checkpoint_cb=on_checkpoint,
        )
        # Fold the workers' result back into the parent optimizer so the export
        # and result-writing paths stay identical to a serial run.
        prep.optimizer.parcel_n_ids = np.asarray(merged, dtype=np.int64)


# ==========================================================================
# Weights editor
# ==========================================================================


class WeightsTable(QtWidgets.QTableWidget):
    """Scored fields and their weights.

    Rows are *derived* from the binning selection rather than typed, so a weight
    can no longer name a field that never gets binned -- which was the easiest
    mistake to make with free text and the last one to surface.

    Displayed names are source columns; the config still stores the ``*_binned``
    form, so run_config.json and the notebook's WEIGHTS dict stay compatible.
    """

    def __init__(self, parent=None):
        super().__init__(0, 2, parent)
        self.setHorizontalHeaderLabels(["Scored field", "Weight"])
        self.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        self.horizontalHeader().setSectionResizeMode(
            1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked
            | QtWidgets.QAbstractItemView.EditTrigger.SelectedClicked
        )
        self._orphans: set = set()

    # ------------------------------------------------------------------
    def set_candidates(self, sources, weights: dict) -> None:
        """Rebuild rows from ``sources``, keeping any weights already set.

        ``weights`` is keyed on the stored (``*_binned``) name. Entries that do
        not correspond to a candidate are kept as flagged rows rather than
        dropped -- a config loaded from an older run should never silently lose
        settings.
        """
        current = self.weights()
        current.update(weights or {})

        ordered = list(dict.fromkeys(sources))
        self._orphans = {
            fields.source_name(k) for k in current
            if fields.source_name(k) not in ordered
        }
        rows = ordered + sorted(self._orphans)

        self.blockSignals(True)
        self.setRowCount(0)
        for name in rows:
            r = self.rowCount()
            self.insertRow(r)
            item = QtWidgets.QTableWidgetItem(name)
            item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            if name in self._orphans:
                item.setForeground(QtGui.QColor("#c0392b"))
                item.setToolTip(
                    "Not in the binning list, so this field is never created. "
                    "Tick it under 'Bin these fields', or set its weight to 0."
                )
            self.setItem(r, 0, item)
            w = current.get(fields.scored_name(name), 0.0)
            self.setItem(r, 1, QtWidgets.QTableWidgetItem(f"{float(w):g}"))
        self.blockSignals(False)

    def weights(self) -> dict:
        """``{binned_name: weight}`` for every row with a non-zero weight."""
        out = {}
        for r in range(self.rowCount()):
            name_item, w_item = self.item(r, 0), self.item(r, 1)
            if not name_item or not name_item.text().strip():
                continue
            try:
                w = float(w_item.text()) if w_item else 0.0
            except ValueError:
                w = 0.0
            if w != 0.0:
                out[fields.scored_name(name_item.text().strip())] = w
        return out


# ==========================================================================
# Main window
# ==========================================================================


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("MI Neighborhoods — desktop")
        self.resize(1500, 950)

        self.cfg = RunConfig()
        self.matches: List[Jurisdiction] = []
        self.universe: Optional[fields.FieldUniverse] = None
        self.field_problems: List[fields.Problem] = []
        self._inspected_path: Optional[str] = None
        self.prep: Optional[pipeline.PreparedRun] = None
        self.task: Optional[Task] = None
        self.opt_thread: Optional[OptimizeThread] = None

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)

        # ---------------- left column ----------------
        left = QtWidgets.QWidget()
        left.setMinimumWidth(430)
        left.setMaximumWidth(560)
        lv = QtWidgets.QVBoxLayout(left)
        lv.setContentsMargins(10, 10, 10, 10)

        lv.addWidget(self._jurisdiction_box())
        lv.addWidget(self._parcels_box())
        lv.addWidget(self._fields_box(), 1)
        lv.addWidget(self._tiling_box())
        lv.addWidget(self._scoring_box())
        lv.addWidget(self._control_box())

        # The setup column has outgrown a fixed height now that field pickers
        # live in it, so let it scroll rather than squashing the weights table.
        scroller = QtWidgets.QScrollArea()
        scroller.setWidget(left)
        scroller.setWidgetResizable(True)
        scroller.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroller.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        scroller.setMinimumWidth(460)
        scroller.setMaximumWidth(580)

        # ---------------- right column ----------------
        right = QtWidgets.QWidget()
        rv = QtWidgets.QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self._map_tab(), "Map")
        self.tabs.addTab(self._diagnostics_tab(), "Tile diagnostics")
        rv.addWidget(self.tabs, 1)

        self.stats_label = QtWidgets.QLabel("Idle")
        self.stats_label.setStyleSheet(
            "font-family: Consolas, monospace; padding: 4px; color: #333;"
        )
        rv.addWidget(self.stats_label)

        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(3000)
        self.log_view.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
        self.log_view.setFixedHeight(170)
        rv.addWidget(self.log_view)

        splitter.addWidget(scroller)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        self.progress.setMaximumWidth(180)
        self.statusBar().addPermanentWidget(self.progress)
        self.statusBar().showMessage("Ready")

        # Only now that ckpt_combo and log_view exist is it safe to scan ./runs.
        self.refresh_runs()

    # ---------------- tabs ----------------

    def _map_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(page)
        v.setContentsMargins(0, 0, 0, 0)
        self.canvas = MapCanvas()
        v.addWidget(make_toolbar(self.canvas, self))
        v.addWidget(self.canvas, 1)
        return page

    #: (label, view key) for the diagnostics view picker.
    DIAG_VIEWS = (
        ("Adjacency edges by kind", "edges"),
        ("Neighborhoods + weak joints", "joints"),
        ("Contiguous fragments", "fragments"),
        ("Tiles with no shared border", "orphans"),
    )

    def _diagnostics_tab(self) -> QtWidgets.QWidget:
        """Inspect the graph the optimizer treats as 'contiguous'.

        The main map draws neighborhoods but not the edges holding them
        together, which is exactly the information needed to tell a genuine
        neighborhood from one that only looks joined because two tiles touch at
        a corner. This tab draws the graph itself.
        """
        page = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(page)
        v.setContentsMargins(6, 6, 6, 6)

        # ---- controls ----
        row = QtWidgets.QHBoxLayout()
        self.diag_btn = QtWidgets.QPushButton("Analyse tiles")
        self.diag_btn.setToolTip(
            "Classify every edge of the current run's adjacency graph.\n"
            "Works on a prepared run, including one that is still optimizing."
        )
        self.diag_btn.clicked.connect(self.on_analyse_tiles)
        row.addWidget(self.diag_btn)

        self.diag_view = QtWidgets.QComboBox()
        for label, key in self.DIAG_VIEWS:
            self.diag_view.addItem(label, key)
        self.diag_view.currentIndexChanged.connect(self._on_diag_view_changed)
        self.diag_view.setEnabled(False)
        row.addWidget(QtWidgets.QLabel("View"))
        row.addWidget(self.diag_view, 1)

        self.diag_refresh = QtWidgets.QPushButton("Re-read assignment")
        self.diag_refresh.setToolTip(
            "Re-colour against the optimizer's current state without "
            "reclassifying the geometry (which is the slow part)."
        )
        self.diag_refresh.clicked.connect(self.on_diag_refresh)
        self.diag_refresh.setEnabled(False)
        row.addWidget(self.diag_refresh)
        v.addLayout(row)

        # ---- edge class toggles ----
        row2 = QtWidgets.QHBoxLayout()
        row2.addWidget(QtWidgets.QLabel("Show edges:"))
        self.diag_class_boxes = []
        for cls, name in enumerate(diagnostics.CLASS_NAMES):
            colour = diagnostics.CLASS_COLORS[cls]
            box = QtWidgets.QCheckBox(name)
            box.setChecked(cls != diagnostics.ROOK)  # the defects, by default
            box.setStyleSheet(
                "color: rgb({}, {}, {}); font-weight: bold;".format(
                    *[int(255 * c) for c in colour]
                )
            )
            box.stateChanged.connect(self._redraw_diagnostics)
            self.diag_class_boxes.append(box)
            row2.addWidget(box)
        row2.addStretch(1)
        v.addLayout(row2)

        # ---- canvas + report ----
        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        holder = QtWidgets.QWidget()
        hv = QtWidgets.QVBoxLayout(holder)
        hv.setContentsMargins(0, 0, 0, 0)
        self.diag_canvas = MapCanvas()
        hv.addWidget(make_toolbar(self.diag_canvas, self))
        hv.addWidget(self.diag_canvas, 1)
        split.addWidget(holder)

        self.diag_report = QtWidgets.QPlainTextEdit()
        self.diag_report.setReadOnly(True)
        self.diag_report.setStyleSheet(
            "font-family: Consolas, monospace; font-size: 11px;"
        )
        self.diag_report.setPlainText(
            "Prepare or resume a run, then press “Analyse tiles”.\n\n"
            "Every edge of the optimizer's adjacency graph is classified by the "
            "geometry behind it:\n"
            "  shared border  the tiles run along each other. A real neighbour.\n"
            "  corner only    they meet at a point. Trading across one of these "
            "is what produces\n"
            "                 the checkerboard -- the optimizer thinks the pieces "
            "are joined.\n"
            "  gap bridged    they never touch; only the adjacency threshold "
            "joins them.\n\n"
            "enforce_contiguity and the severed-component split both consult this "
            "same graph, so\nanything it calls connected is invisible to them."
        )
        split.addWidget(self.diag_report)
        split.setStretchFactor(0, 1)
        split.setSizes([700, 190])
        v.addWidget(split, 1)

        self.diag: Optional[diagnostics.TileDiagnostics] = None
        self.diag_tile_n_ids: Optional[np.ndarray] = None
        return page

    # ---------------- boxes ----------------

    def _jurisdiction_box(self) -> QtWidgets.QWidget:
        box = QtWidgets.QGroupBox("1. Jurisdiction")
        g = QtWidgets.QGridLayout(box)

        self.jur_edit = QtWidgets.QLineEdit()
        self.jur_edit.setPlaceholderText("e.g. Fort Myers, FL  ·  Lee County, FL")
        self.jur_edit.returnPressed.connect(self.on_search)
        self.search_btn = QtWidgets.QPushButton("Search")
        self.search_btn.clicked.connect(self.on_search)

        self.jur_combo = QtWidgets.QComboBox()
        self.jur_combo.setEnabled(False)

        g.addWidget(QtWidgets.QLabel("Place"), 0, 0)
        g.addWidget(self.jur_edit, 0, 1)
        g.addWidget(self.search_btn, 0, 2)
        g.addWidget(QtWidgets.QLabel("Match"), 1, 0)
        g.addWidget(self.jur_combo, 1, 1, 1, 2)
        return box

    def _parcels_box(self) -> QtWidgets.QWidget:
        box = QtWidgets.QGroupBox("2. Parcels")
        g = QtWidgets.QGridLayout(box)

        self.parcel_edit = QtWidgets.QLineEdit()
        self.parcel_edit.setPlaceholderText("parcel .parquet / .gpkg / .shp / .geojson")
        self.parcel_edit.editingFinished.connect(self.on_parcel_path_changed)
        browse = QtWidgets.QPushButton("Browse\u2026")
        browse.clicked.connect(self.on_browse_parcels)

        # Editable combos, not plain dropdowns: they must still work before a
        # file is chosen, and must not discard a value loaded from an older
        # run_config.json that names a column this file happens to lack.
        self.filter_col = QtWidgets.QComboBox()
        self.filter_col.setEditable(True)
        self.filter_col.setCurrentText(self.cfg.parcel_filter_column)
        self.filter_col.currentTextChanged.connect(self.on_filter_column_changed)
        self.filter_col.setToolTip(
            "Column that says which parcels to model. Populated from the file "
            "once one is selected; you can still type a name."
        )

        self.filter_val = QtWidgets.QComboBox()
        self.filter_val.setEditable(True)
        self.filter_val.setCurrentText(self.cfg.parcel_filter_value)
        self.filter_val.setToolTip(
            "Value to keep. Populated with the distinct values actually present "
            "in the chosen column, most common first."
        )

        self.field_status = QtWidgets.QLabel("No parcel file selected.")
        self.field_status.setWordWrap(True)
        self.field_status.setStyleSheet("color: #777; font-size: 10px;")

        g.addWidget(QtWidgets.QLabel("File"), 0, 0)
        g.addWidget(self.parcel_edit, 0, 1, 1, 2)
        g.addWidget(browse, 0, 3)
        g.addWidget(QtWidgets.QLabel("Filter"), 1, 0)
        g.addWidget(self.filter_col, 1, 1)
        g.addWidget(QtWidgets.QLabel("=="), 1, 2)
        g.addWidget(self.filter_val, 1, 3)
        g.addWidget(self.field_status, 2, 0, 1, 4)
        return box

    def _fields_box(self) -> QtWidgets.QWidget:
        """Everything that names a column, driven by the file's own schema."""
        box = QtWidgets.QGroupBox("3. Fields")
        v = QtWidgets.QVBoxLayout(box)

        row = QtWidgets.QHBoxLayout()
        self.land_class_combo = QtWidgets.QComboBox()
        self.land_class_combo.setEditable(True)
        self.land_class_combo.setCurrentText(self.cfg.land_class_column)
        self.land_class_combo.setToolTip(
            "Column naming each parcel's land class. Only used when sightlines "
            "are blocked by 'all except in-gap classes'. Optional."
        )
        row.addWidget(QtWidgets.QLabel("Land class"))
        row.addWidget(self.land_class_combo, 1)
        v.addLayout(row)

        v.addWidget(QtWidgets.QLabel("Bin these fields (scoring candidates)"))
        self.continuous_list = self._make_checklist(
            "Numeric or derivable columns to bin. Only binned fields can be "
            "scored, so this list drives the weights table below."
        )
        self.continuous_list.itemChanged.connect(self.on_continuous_changed)
        v.addWidget(self.continuous_list)

        v.addWidget(QtWidgets.QLabel("Weights (0 = not scored)"))
        self.weights_table = WeightsTable()
        v.addWidget(self.weights_table, 1)

        v.addWidget(QtWidgets.QLabel("Seed on these fields (KMeans)"))
        self.seed_list = self._make_checklist(
            "Fields the initial KMeans seeding clusters on. Position columns "
            "plus whatever separates submarkets in this jurisdiction."
        )
        v.addWidget(self.seed_list)
        return box

    def _tiling_box(self) -> QtWidgets.QWidget:
        box = QtWidgets.QGroupBox("3. Tiling")
        g = QtWidgets.QGridLayout(box)

        self.grid_spin = QtWidgets.QDoubleSpinBox()
        self.grid_spin.setRange(50, 5000)
        self.grid_spin.setSingleStep(50)
        self.grid_spin.setValue(self.cfg.grid_size_ft)
        self.grid_spin.setSuffix(" ft")

        self.adj_spin = QtWidgets.QDoubleSpinBox()
        self.adj_spin.setRange(0, 2000)
        self.adj_spin.setSingleStep(10)
        self.adj_spin.setValue(self.cfg.adjacency_threshold_ft)
        self.adj_spin.setSuffix(" ft")

        self.blocks_chk = QtWidgets.QCheckBox("Census blocks")
        self.blocks_chk.setChecked(True)
        self.roads_chk = QtWidgets.QCheckBox("OSM roads")
        self.roads_chk.setChecked(True)
        self.waterways_chk = QtWidgets.QCheckBox("OSM waterways")
        self.waterways_chk.setChecked(True)
        self.waterways_chk.setToolTip(
            "River, stream, canal and ditch centrelines as cut lines.\n\n"
            "Switch off alongside 'OSM roads' if Overpass refuses this "
            "jurisdiction; tiles then come from census blocks plus the grid."
        )
        self.water_chk = QtWidgets.QCheckBox("Clip OSM water")
        self.water_chk.setChecked(True)

        g.addWidget(QtWidgets.QLabel("Grid"), 0, 0)
        g.addWidget(self.grid_spin, 0, 1)
        g.addWidget(QtWidgets.QLabel("Adjacency"), 0, 2)
        g.addWidget(self.adj_spin, 0, 3)
        g.addWidget(self.blocks_chk, 1, 0, 1, 2)
        g.addWidget(self.roads_chk, 1, 2, 1, 2)
        g.addWidget(self.waterways_chk, 2, 0, 1, 2)
        g.addWidget(self.water_chk, 2, 2, 1, 2)
        return box

    def _scoring_box(self) -> QtWidgets.QWidget:
        box = QtWidgets.QGroupBox("4. Scoring & annealing")
        v = QtWidgets.QVBoxLayout(box)

        form = QtWidgets.QGridLayout()
        self.k_spin = QtWidgets.QSpinBox()
        self.k_spin.setRange(2, 100000)
        self.k_spin.setValue(self.cfg.n_neighborhoods)

        self.bins_spin = QtWidgets.QSpinBox()
        self.bins_spin.setRange(2, 200)
        self.bins_spin.setValue(self.cfg.max_bins)

        self.temp_spin = QtWidgets.QDoubleSpinBox()
        self.temp_spin.setDecimals(3)
        self.temp_spin.setRange(0.0, 100.0)
        self.temp_spin.setValue(self.cfg.initial_temp)

        self.cool_spin = QtWidgets.QDoubleSpinBox()
        self.cool_spin.setDecimals(5)
        self.cool_spin.setRange(0.5, 0.99999)
        self.cool_spin.setSingleStep(0.001)
        self.cool_spin.setValue(self.cfg.cooling_rate)

        self.refresh_spin = QtWidgets.QSpinBox()
        self.refresh_spin.setRange(10, 1000000)
        self.refresh_spin.setValue(self.cfg.refresh_every)

        self.ckpt_spin = QtWidgets.QSpinBox()
        self.ckpt_spin.setRange(100, 10000000)
        self.ckpt_spin.setValue(self.cfg.checkpoint_every)

        self.workers_spin = QtWidgets.QSpinBox()
        self.workers_spin.setRange(0, 32)
        self.workers_spin.setValue(self.cfg.workers)
        self.workers_spin.setSpecialValueText("auto")
        self.workers_spin.setToolTip(
            "Annealing processes. Severed components (islands, opposite banks of "
            "a river) never interact, so they can be optimized concurrently.\n"
            "1 = serial. 0 = auto, clamped to the speedup the component "
            "structure actually allows."
        )

        self.adjacency_combo = QtWidgets.QComboBox()
        self.adjacency_combo.addItem("tile geometry (original)", "tile")
        self.adjacency_combo.addItem("parcel + line of sight", "parcel")
        self.adjacency_combo.setToolTip(
            "How two tiles come to be considered neighbours.\n\n"
            "tile geometry: tiles within the threshold of each other. Cheap, but "
            "two lots with a third lot between them are 'adjacent' whenever lots "
            "are narrower than the threshold.\n\n"
            "parcel + line of sight: tiles are neighbours if their parcels are, "
            "and a pair is dropped when the shortest line between them is blocked "
            "by another parcel. Streets and canals are usually unparceled so they "
            "stay crossable; an intervening lot does not. Costs a few seconds once."
        )

        self.obstacle_combo = QtWidgets.QComboBox()
        self.obstacle_combo.addItem("modeled parcels only", "modeled")
        self.obstacle_combo.addItem("all parcels", "all")
        self.obstacle_combo.addItem("all except in-gap classes", "all_except")
        self.obstacle_combo.setToolTip(
            "What blocks a sightline (only used by parcel adjacency).\n\n"
            "modeled parcels only: farmland, condos and vacant land are "
            "transparent, so single-family pockets separated by a 'sea' of other "
            "use classes reconnect.\n\n"
            "all parcels: anything with a parcel record blocks.\n\n"
            "all except in-gap classes: everything blocks except right-of-way, "
            "submerged land, utility strips, common elements and similar. Matched "
            "by keyword, so it transfers between jurisdictions -- check the log "
            "for which classes it actually treated as transparent."
        )

        self.contiguity_chk = QtWidgets.QCheckBox("Keep neighborhoods contiguous")
        self.contiguity_chk.setChecked(self.cfg.enforce_contiguity)
        self.contiguity_chk.setToolTip(
            "Reject any move that would split a neighborhood into disconnected "
            "pieces (enforce_contiguity).\n\n"
            "On: boundaries stay mappable, at a measured cost of roughly a "
            "quarter of the score improvement — the best-scoring move is often "
            "exactly the one that would carve a neighborhood.\n\n"
            "Off: neighborhoods may form exclaves. On Lee County that produced "
            "806 of them; about 40% were splinters of 5 parcels or fewer, but "
            "80% of the affected parcels sat in exclaves of 21+, which can be "
            "genuine submarkets.\n\n"
            "Either way this only prevents NEW disconnection. Neighborhoods that "
            "arrive disconnected from the KMeans seeding stay that way."
        )

        self.split_chk = QtWidgets.QCheckBox("Split severed neighborhoods")
        self.split_chk.setChecked(self.cfg.split_severed_neighborhoods)
        self.split_chk.setToolTip(
            "KMeans seeds on position without knowing tile adjacency, so it can "
            "put one neighborhood on both banks of a river. Trading can never "
            "separate those, so they are split before optimizing.\n"
            "Required for parallel annealing. Uncheck only to reproduce an "
            "older run."
        )

        for col, (label, widget) in enumerate(
            [
                ("Neighborhoods", self.k_spin),
                ("Bins", self.bins_spin),
            ]
        ):
            form.addWidget(QtWidgets.QLabel(label), 0, col * 2)
            form.addWidget(widget, 0, col * 2 + 1)
        for col, (label, widget) in enumerate(
            [("Initial T", self.temp_spin), ("Cooling", self.cool_spin)]
        ):
            form.addWidget(QtWidgets.QLabel(label), 1, col * 2)
            form.addWidget(widget, 1, col * 2 + 1)
        for col, (label, widget) in enumerate(
            [("Redraw every", self.refresh_spin), ("Checkpoint every", self.ckpt_spin)]
        ):
            form.addWidget(QtWidgets.QLabel(label), 2, col * 2)
            form.addWidget(widget, 2, col * 2 + 1)
        form.addWidget(QtWidgets.QLabel("Workers"), 3, 0)
        form.addWidget(self.workers_spin, 3, 1)
        form.addWidget(self.split_chk, 3, 2, 1, 2)
        form.addWidget(self.contiguity_chk, 4, 0, 1, 4)
        form.addWidget(QtWidgets.QLabel("Adjacency"), 5, 0)
        form.addWidget(self.adjacency_combo, 5, 1, 1, 3)
        form.addWidget(QtWidgets.QLabel("Blocked by"), 6, 0)
        form.addWidget(self.obstacle_combo, 6, 1, 1, 3)
        v.addLayout(form)
        return box

    def _control_box(self) -> QtWidgets.QWidget:
        box = QtWidgets.QGroupBox("5. Run")
        v = QtWidgets.QVBoxLayout(box)

        resume_row = QtWidgets.QHBoxLayout()
        self.run_combo = QtWidgets.QComboBox()
        self.run_combo.setToolTip("Existing runs found in ./runs")
        reload_btn = QtWidgets.QPushButton("↻")
        reload_btn.setFixedWidth(30)
        reload_btn.clicked.connect(self.refresh_runs)
        resume_row.addWidget(QtWidgets.QLabel("Resume"))
        resume_row.addWidget(self.run_combo, 1)
        resume_row.addWidget(reload_btn)
        v.addLayout(resume_row)

        self.ckpt_combo = QtWidgets.QComboBox()
        self.ckpt_combo.setToolTip("Checkpoint to restart from")
        self.run_combo.currentIndexChanged.connect(self.refresh_checkpoints)
        v.addWidget(self.ckpt_combo)

        btns = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("Start new run")
        self.start_btn.clicked.connect(self.on_start)
        self.resume_btn = QtWidgets.QPushButton("Resume selected")
        self.resume_btn.clicked.connect(self.on_resume)
        btns.addWidget(self.start_btn)
        btns.addWidget(self.resume_btn)
        v.addLayout(btns)

        btns2 = QtWidgets.QHBoxLayout()
        self.pause_btn = QtWidgets.QPushButton("Pause")
        self.pause_btn.setEnabled(False)
        self.pause_btn.clicked.connect(self.on_pause)
        self.stop_btn = QtWidgets.QPushButton("Stop && save")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.on_stop)
        self.export_btn = QtWidgets.QPushButton("Export…")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.on_export)
        btns2.addWidget(self.pause_btn)
        btns2.addWidget(self.stop_btn)
        btns2.addWidget(self.export_btn)
        v.addLayout(btns2)
        return box

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def log(self, text: str) -> None:
        self.log_view.appendPlainText(text)

    def running(self) -> bool:
        return bool(self.opt_thread and self.opt_thread.isRunning())

    def busy(self, on: bool, message: str = "") -> None:
        self.progress.setVisible(on)
        self.statusBar().showMessage(message or ("Working…" if on else "Ready"))
        self.search_btn.setEnabled(not on)
        # Never re-arm Start/Resume while an optimization is still going.
        can_start = (not on) and not self.running()
        self.start_btn.setEnabled(can_start)
        self.resume_btn.setEnabled(can_start)
        # The diagnostics only need a prepared run, not an idle one -- they are
        # useful mid-optimization -- but they must not contend for self.task.
        self.diag_btn.setEnabled((not on) and self.prep is not None)

    def _error_box(self, tb: str) -> None:
        self.log(tb)
        QtWidgets.QMessageBox.critical(
            self, "Error", tb.strip().splitlines()[-1] if tb.strip() else "Unknown error"
        )

    def on_failed(self, tb: str) -> None:
        """Preparation or search failed: nothing is running afterwards."""
        self.busy(False, "Failed")
        self._error_box(tb)

    def on_run_failed(self, tb: str) -> None:
        """The optimizer thread died: tear the run controls down too."""
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        # The slot runs while the thread is still technically alive (it emitted
        # `failed` and is on its way out), so drop the reference first --
        # otherwise running() vetoes re-enabling Start and nothing else will.
        self.opt_thread = None
        self.busy(False, "Run failed")
        self._error_box(tb)

    def collect_config(self) -> RunConfig:
        cfg = RunConfig()
        cfg.jurisdiction_query = self.jur_edit.text().strip()
        cfg.parcel_path = self.parcel_edit.text().strip()
        cfg.parcel_filter_column = self.filter_col.currentText().strip()
        cfg.parcel_filter_value = self.filter_val.currentText().strip()
        cfg.grid_size_ft = float(self.grid_spin.value())
        cfg.adjacency_threshold_ft = float(self.adj_spin.value())
        cfg.use_census_blocks = self.blocks_chk.isChecked()
        cfg.use_osm_roads = self.roads_chk.isChecked()
        cfg.use_osm_waterways = self.waterways_chk.isChecked()
        cfg.clip_water = self.water_chk.isChecked()
        cfg.n_neighborhoods = int(self.k_spin.value())
        cfg.max_bins = int(self.bins_spin.value())
        cfg.initial_temp = float(self.temp_spin.value())
        cfg.cooling_rate = float(self.cool_spin.value())
        cfg.refresh_every = int(self.refresh_spin.value())
        cfg.checkpoint_every = int(self.ckpt_spin.value())
        cfg.workers = int(self.workers_spin.value())
        cfg.split_severed_neighborhoods = self.split_chk.isChecked()
        cfg.enforce_contiguity = self.contiguity_chk.isChecked()
        cfg.adjacency_mode = self.adjacency_combo.currentData()
        cfg.obstacle_mode = self.obstacle_combo.currentData()
        cfg.land_class_column = self.land_class_combo.currentText().strip()
        cfg.continuous_variables = self._checked(self.continuous_list)
        cfg.seed_fields = self._checked(self.seed_list)
        cfg.weights = self.weights_table.weights()
        return cfg

    def apply_config(self, cfg: RunConfig) -> None:
        self.jur_edit.setText(cfg.jurisdiction_query)
        self.parcel_edit.setText(cfg.parcel_path)
        self.filter_col.setCurrentText(cfg.parcel_filter_column)
        self.filter_val.setCurrentText(cfg.parcel_filter_value)
        self.grid_spin.setValue(cfg.grid_size_ft)
        self.adj_spin.setValue(cfg.adjacency_threshold_ft)
        self.blocks_chk.setChecked(cfg.use_census_blocks)
        self.roads_chk.setChecked(cfg.use_osm_roads)
        self.waterways_chk.setChecked(cfg.use_osm_waterways)
        self.water_chk.setChecked(cfg.clip_water)
        self.k_spin.setValue(cfg.n_neighborhoods)
        self.bins_spin.setValue(cfg.max_bins)
        self.temp_spin.setValue(cfg.initial_temp)
        self.cool_spin.setValue(cfg.cooling_rate)
        self.refresh_spin.setValue(cfg.refresh_every)
        self.ckpt_spin.setValue(cfg.checkpoint_every)
        self.workers_spin.setValue(cfg.workers)
        self.split_chk.setChecked(cfg.split_severed_neighborhoods)
        self.contiguity_chk.setChecked(cfg.enforce_contiguity)
        for combo, value in ((self.adjacency_combo, cfg.adjacency_mode),
                             (self.obstacle_combo, cfg.obstacle_mode)):
            idx = combo.findData(value)
            if idx >= 0:
                combo.setCurrentIndex(idx)
        self.land_class_combo.setCurrentText(cfg.land_class_column)
        opts = self.universe.binnable if self.universe else []
        self._fill_checklist(self.continuous_list, opts, cfg.continuous_variables)
        self._fill_checklist(self.seed_list, opts, cfg.seed_fields)
        self.weights_table.set_candidates(cfg.continuous_variables, cfg.weights)
        self.revalidate_fields()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _make_checklist(self, tooltip: str) -> QtWidgets.QListWidget:
        w = QtWidgets.QListWidget()
        w.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        w.setFixedHeight(96)
        w.setToolTip(tooltip)
        return w

    # ------------------------------------------------------------------
    # Field introspection
    # ------------------------------------------------------------------

    def _fill_checklist(self, widget, options, checked) -> None:
        want = set(checked)
        # Anything configured but absent from the file is still listed, flagged,
        # so a loaded config never loses settings silently.
        missing = [c for c in checked if c not in set(options)]
        widget.blockSignals(True)
        widget.clear()
        for name in list(options) + missing:
            item = QtWidgets.QListWidgetItem(name)
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                QtCore.Qt.CheckState.Checked if name in want
                else QtCore.Qt.CheckState.Unchecked
            )
            if name in missing:
                item.setForeground(QtGui.QColor("#c0392b"))
                item.setToolTip("Not present in this parcel file")
            widget.addItem(item)
        widget.blockSignals(False)

    @staticmethod
    def _checked(widget) -> List[str]:
        return [
            widget.item(i).text() for i in range(widget.count())
            if widget.item(i).checkState() == QtCore.Qt.CheckState.Checked
        ]

    def on_browse_parcels(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Choose a parcel file",
            str(Path.cwd()),
            "Spatial data (*.parquet *.pq *.gpkg *.shp *.geojson *.json);;All files (*)",
        )
        if path:
            self.parcel_edit.setText(path)
            self.on_parcel_path_changed()

    def on_parcel_path_changed(self) -> None:
        """Introspect the file as soon as it is named.

        Cheap enough to do eagerly -- ~11 ms for the schema of a 158 MB parquet --
        and it moves field validation from step 4 of prepare(), after the census
        download and the shatter, to the moment of choosing.
        """
        path = self.parcel_edit.text().strip()
        if not path or path == getattr(self, "_inspected_path", None):
            return
        self._inspected_path = path
        self.field_status.setText("Reading fields\u2026")

        def work(log):
            uni = fields.inspect_parcel_file(path, progress=log)
            values = fields.distinct_values(
                path, self.cfg.parcel_filter_column, progress=log
            ) if uni.ok else []
            return uni, values

        self.task = Task(work, self)
        self.task.log.connect(self.log)
        self.task.failed.connect(self.on_failed)
        self.task.done.connect(self.on_fields_read)
        self.task.start()

    def on_fields_read(self, payload) -> None:
        uni, values = payload
        self.universe = uni
        if not uni.ok:
            self.field_status.setStyleSheet("color: #c0392b; font-size: 10px;")
            self.field_status.setText(f"Could not read fields: {uni.error}")
            return
        self.log(f"{uni.path.name}: {uni.summary()}")

        cfg = self.collect_config()
        for combo, options, keep in (
            (self.filter_col, uni.text + uni.numeric, cfg.parcel_filter_column),
            (self.land_class_combo, uni.text, cfg.land_class_column),
        ):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(sorted(options))
            combo.setCurrentText(keep)
            combo.blockSignals(False)

        self._set_filter_values(values, cfg.parcel_filter_value)
        self._fill_checklist(self.continuous_list, uni.binnable,
                             cfg.continuous_variables)
        self._fill_checklist(self.seed_list, uni.seedable, cfg.seed_fields)
        self.on_continuous_changed()
        self.revalidate_fields()

    def _set_filter_values(self, values, keep: str) -> None:
        self.filter_val.blockSignals(True)
        self.filter_val.clear()
        self.filter_val.addItems(values)
        self.filter_val.setCurrentText(keep)
        self.filter_val.blockSignals(False)

    def on_filter_column_changed(self, _text: str = "") -> None:
        """Reload the value list, since it depends on the chosen column."""
        path = self.parcel_edit.text().strip()
        column = self.filter_col.currentText().strip()
        if not path or not column or self.universe is None:
            return
        if column not in set(self.universe.columns):
            self.revalidate_fields()
            return
        keep = self.filter_val.currentText()
        values = fields.distinct_values(path, column, progress=self.log)
        self._set_filter_values(values, keep)
        self.revalidate_fields()

    def on_continuous_changed(self, _item=None) -> None:
        """Weights follow the binning selection, so the two cannot disagree."""
        chosen = self._checked(self.continuous_list)
        existing = self.weights_table.weights()
        if not existing:
            existing = {
                fields.scored_name(c): fields.default_weight_for(c, DEFAULT_WEIGHTS)
                for c in chosen
            }
        prebinned = list(self.universe.prebinned) if self.universe else []
        candidates = chosen + [fields.source_name(c) for c in prebinned]
        self.weights_table.set_candidates(candidates, existing)
        self.revalidate_fields()

    def revalidate_fields(self) -> List["fields.Problem"]:
        """Check the whole field configuration against the loaded file."""
        if self.universe is None or not self.universe.ok:
            self.field_problems = []
            return self.field_problems

        values = [self.filter_val.itemText(i)
                  for i in range(self.filter_val.count())] or None
        problems = fields.validate(self.collect_config(), self.universe, values)
        self.field_problems = problems

        fatal = fields.fatal_problems(problems)
        if not problems:
            self.field_status.setStyleSheet("color: #2c7a3f; font-size: 10px;")
            self.field_status.setText(
                f"{self.universe.summary()} \u2014 all configured fields present."
            )
        else:
            colour = "#c0392b" if fatal else "#b8860b"
            self.field_status.setStyleSheet(f"color: {colour}; font-size: 10px;")
            self.field_status.setText(
                f"{len(fatal)} blocking, {len(problems) - len(fatal)} advisory: "
                + "; ".join(f"{p.setting}={p.value}" for p in problems[:4])
                + (" \u2026" if len(problems) > 4 else "")
            )
        return problems

    def on_search(self) -> None:
        query = self.jur_edit.text().strip()
        if not query:
            return
        self.busy(True, f"Searching for '{query}'…")
        self.jur_combo.clear()
        self.jur_combo.setEnabled(False)
        # Drop the previous results too, so a failed search can't leave a stale
        # selection that on_start would happily run against.
        self.matches = []

        def work(log):
            return sources.search_jurisdictions(query, progress=log)

        self.task = Task(work, self)
        self.task.log.connect(self.log)
        self.task.failed.connect(self.on_failed)
        self.task.done.connect(self.on_search_done)
        self.task.start()

    def on_search_done(self, matches) -> None:
        self.busy(False)
        self.matches = list(matches or [])
        if not self.matches:
            self.log("No matching jurisdiction found.")
            QtWidgets.QMessageBox.information(
                self, "No match",
                "Nothing matched. Try 'City, ST' or 'Something County, ST'.",
            )
            return
        for j in self.matches:
            self.jur_combo.addItem(j.label)
        self.jur_combo.setEnabled(True)
        self.log(f"{len(self.matches)} candidate jurisdiction(s); best guess first.")

    # ------------------------------------------------------------------

    def refresh_runs(self) -> None:
        # Block signals while filling: each addItem would otherwise fire
        # currentIndexChanged and rescan a checkpoint directory.
        self.run_combo.blockSignals(True)
        try:
            self.run_combo.clear()
            for d in find_runs(runs_dir(self.cfg)):
                self.run_combo.addItem(d.name, str(d))
        finally:
            self.run_combo.blockSignals(False)
        self.refresh_checkpoints()

    def refresh_checkpoints(self) -> None:
        self.ckpt_combo.clear()
        path = self.run_combo.currentData()
        if not path:
            return
        try:
            for cp in CheckpointStore(Path(path)).list():
                self.ckpt_combo.addItem(cp["label"], cp["path"])
        except Exception as exc:  # noqa: BLE001
            self.log(f"Could not read checkpoints: {exc}")
        if self.ckpt_combo.count():
            self.ckpt_combo.setCurrentIndex(self.ckpt_combo.count() - 1)

    # ------------------------------------------------------------------

    def on_start(self) -> None:
        cfg = self.collect_config()
        if not cfg.parcel_path or not Path(cfg.parcel_path).exists():
            QtWidgets.QMessageBox.warning(
                self, "Parcels required", "Choose a parcel file first."
            )
            return
        idx = self.jur_combo.currentIndex()
        if not self.matches or idx < 0 or idx >= len(self.matches):
            QtWidgets.QMessageBox.warning(
                self, "Jurisdiction required",
                "Search for a jurisdiction and pick a match first.",
            )
            return
        if not self._fields_ok(cfg):
            return
        jur = self.matches[idx]
        self._prepare_and_run(cfg, jurisdiction=jur, run_dir=None, checkpoint=None)

    def _fields_ok(self, cfg: RunConfig) -> bool:
        """Refuse to start on a field problem, before anything expensive.

        Field validation used to happen inside load_parcels, which runs after the
        census download and the shatter -- so a misspelled column cost minutes
        before it surfaced. Checking here makes it cost nothing.
        """
        problems = self.revalidate_fields()
        fatal = fields.fatal_problems(problems)
        if not fatal:
            for p in problems:
                self.log(f"Field warning: {p}")
            return True

        detail = "\n".join(f"\u2022 {p.setting}: {p.value} \u2014 {p.detail}"
                            for p in fatal)
        self.log(fields.describe(problems))
        QtWidgets.QMessageBox.warning(
            self, "Fields don't match the parcel file",
            f"{len(fatal)} setting(s) name something this file does not have, so "
            "the run would fail after the downloads and tiling:\n\n" + detail,
        )
        return False

    def on_resume(self) -> None:
        run_path = self.run_combo.currentData()
        if not run_path:
            QtWidgets.QMessageBox.warning(
                self, "No run selected", "There are no runs in ./runs yet."
            )
            return
        run_dir = Path(run_path)
        cfg_path = run_dir / "run_config.json"
        cfg = RunConfig.load(cfg_path) if cfg_path.exists() else self.collect_config()
        # Live-editable knobs still come from the UI.
        cfg.refresh_every = int(self.refresh_spin.value())
        cfg.checkpoint_every = int(self.ckpt_spin.value())
        self.apply_config(cfg)
        self._prepare_and_run(
            cfg, jurisdiction=None, run_dir=run_dir,
            checkpoint=self.ckpt_combo.currentData(),
        )

    def _prepare_and_run(self, cfg, jurisdiction, run_dir, checkpoint) -> None:
        # Forget the previous run before preparing: if preparation fails, Export
        # must not silently write the old run's data.
        self.prep = None
        self.export_btn.setEnabled(False)
        # Same reasoning for the diagnostics: they describe a tileset that is
        # about to be replaced, and their tile count may not even match the new
        # one.
        self.diag = None
        self.diag_tile_n_ids = None
        self.diag_view.setEnabled(False)
        self.diag_refresh.setEnabled(False)
        self.diag_canvas.clear_edges()
        self.busy(True, "Preparing…")
        self.log("=" * 60)

        def work(log):
            prep = pipeline.prepare(
                cfg, jurisdiction=jurisdiction, run_dir=run_dir, progress=log
            )
            opt = prep.optimizer
            if checkpoint:
                log(f"Loading checkpoint {Path(checkpoint).name}")
                opt.load_checkpoint(prep.store.load(checkpoint))
            else:
                latest = prep.store.latest()
                if latest is not None:
                    log(f"Found an existing checkpoint at iteration {latest.iteration:,}")
                    opt.load_checkpoint(latest)
                else:
                    opt.consolidate_mixed_tiles()
            return prep

        self.task = Task(work, self)
        self.task.log.connect(self.log)
        self.task.failed.connect(self.on_failed)
        self.task.done.connect(self.on_prepared)
        self.task.start()

    def on_prepared(self, prep) -> None:
        self.prep = prep
        self.busy(False, f"Ready — {prep.run_dir.name}")
        self.log(f"Run directory: {prep.run_dir}")

        geoms = prep.tile_geometries()
        self.log(f"Drawing {len(geoms):,} tiles…")
        self.canvas.set_tiles(
            geoms,
            simplify_tolerance=max(prep.cfg.grid_size_ft / 50.0, 1.0),
            boundary=prep.jurisdiction_gdf.to_crs(prep.working_crs),
        )
        self.canvas.update_colors(
            prep.optimizer.tile_n_ids, f"{prep.cfg.jurisdiction_name} — iteration 0"
        )

        self.opt_thread = OptimizeThread(prep, self)
        self.opt_thread.log.connect(self.log)
        self.opt_thread.failed.connect(self.on_run_failed)
        self.opt_thread.stats.connect(self.on_stats)
        self.opt_thread.snapshot.connect(self.on_snapshot)
        self.opt_thread.finished_run.connect(self.on_run_finished)
        self.opt_thread.start()

        self.pause_btn.setEnabled(True)
        self.pause_btn.setText("Pause")
        self.stop_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
        self.start_btn.setEnabled(False)
        self.resume_btn.setEnabled(False)

    # ------------------------------------------------------------------
    # Tile diagnostics
    # ------------------------------------------------------------------

    def on_analyse_tiles(self) -> None:
        if not self.prep:
            QtWidgets.QMessageBox.information(
                self, "Nothing to analyse",
                "Start or resume a run first. The diagnostics read the "
                "adjacency graph that run built.",
            )
            return
        if self.task is not None and self.task.isRunning():
            QtWidgets.QMessageBox.information(
                self, "Busy", "Something else is still working; try again in a moment."
            )
            return

        prep = self.prep
        adj = prep.tile_adjacency or {}
        if not adj:
            QtWidgets.QMessageBox.warning(
                self, "No adjacency graph",
                "This run has no tile adjacency recorded, so there is nothing "
                "to classify.",
            )
            return

        geoms = prep.tile_geometries()
        tile_ids = prep.optimizer.tile_ids
        # Snapshot the assignment on this thread. A run in progress will have
        # moved on by the time the classification finishes, which is fine -- the
        # report is explicitly about the state at the moment you pressed the
        # button, and the geometry it classifies does not change at all.
        self.diag_tile_n_ids = prep.optimizer.tile_n_ids.copy()

        self.busy(True, "Classifying the adjacency graph…")
        self.diag_btn.setEnabled(False)
        self.log("=" * 60)

        def work(log):
            return diagnostics.analyse(geoms, adj, tile_ids, progress=log)

        self.task = Task(work, self)
        self.task.log.connect(self.log)
        self.task.failed.connect(self._on_diag_failed)
        self.task.done.connect(self._on_diag_ready)
        self.task.start()

    def _on_diag_failed(self, tb: str) -> None:
        self.diag_btn.setEnabled(True)
        self.on_failed(tb)

    def _on_diag_ready(self, diag) -> None:
        self.diag = diag
        self.busy(False, "Diagnostics ready")
        self.diag_btn.setEnabled(True)
        self.diag_view.setEnabled(True)
        self.diag_refresh.setEnabled(True)

        prep = self.prep
        self.diag_canvas.set_tiles(
            prep.tile_geometries(),
            simplify_tolerance=max(prep.cfg.grid_size_ft / 50.0, 1.0),
            boundary=prep.jurisdiction_gdf.to_crs(prep.working_crs),
        )
        self.diag_canvas.set_edges(diag.segments())
        # Via the view handler, not straight to the redraw, so the current view
        # gets its intended class defaults on first paint as well as on a change.
        self._on_diag_view_changed()
        self._write_diag_report()
        self.tabs.setCurrentIndex(1)

    def on_diag_refresh(self) -> None:
        """Re-colour against the optimizer's current assignment."""
        if self.diag is None or not self.prep:
            return
        tn = self.prep.optimizer.tile_n_ids
        if len(tn) != self.diag.n_tiles:
            QtWidgets.QMessageBox.warning(
                self, "Tileset changed",
                "The prepared run no longer matches these diagnostics. "
                "Press “Analyse tiles” again.",
            )
            return
        self.diag_tile_n_ids = np.asarray(tn).copy()
        self._redraw_diagnostics()
        self._write_diag_report()

    def _on_diag_view_changed(self, *_) -> None:
        # "Adjacency edges by kind" is about the fabric, so it wants every class
        # on; the other views are about defects inside a neighborhood, where the
        # shared borders are the uninteresting majority.
        key = self.diag_view.currentData()
        want_rook = key == "edges"
        rook_box = self.diag_class_boxes[diagnostics.ROOK]
        if rook_box.isChecked() != want_rook:
            rook_box.blockSignals(True)
            rook_box.setChecked(want_rook)
            rook_box.blockSignals(False)
        self._redraw_diagnostics()

    def _redraw_diagnostics(self, *_) -> None:
        d = self.diag
        tn = self.diag_tile_n_ids
        if d is None or tn is None:
            return

        key = self.diag_view.currentData()
        if key == "edges":
            colors = render.flat_fill(d.n_tiles)
            title = f"{d.n_edges:,} adjacency edges by kind"
        elif key == "joints":
            colors = neighborhood_colors(tn)
            title = "Neighborhoods, with their weak internal joints drawn"
        elif key == "fragments":
            labels, count = d.components(tn, (diagnostics.ROOK,))
            colors = render.fragment_colors(labels)
            title = (
                f"{count:,} shared-border fragments across "
                f"{len(np.unique(tn)):,} neighborhoods"
            )
        else:
            weak = d.weak_tiles(tn)
            colors = render.defect_colors(weak)
            title = (
                f"{int(weak.sum()):,} tiles have no shared border with their "
                "own neighborhood"
            )
        self.diag_canvas.set_face_colors(colors, title)

        chosen = [i for i, b in enumerate(self.diag_class_boxes) if b.isChecked()]
        visible = (
            d.class_mask(*chosen) if chosen
            else np.zeros(d.n_edges, dtype=bool)
        )
        if key != "edges":
            # Only edges *within* one neighborhood are joints. An edge between
            # two different neighborhoods is a boundary the optimizer trades
            # across, which is a different question entirely.
            visible = visible & (tn[d.left] == tn[d.right])
        self.diag_canvas.color_edges(
            render.edge_rgba(d.edge_class, diagnostics.CLASS_COLORS, visible)
        )

    def _write_diag_report(self) -> None:
        if self.diag is None or self.diag_tile_n_ids is None:
            return
        name = self.prep.cfg.jurisdiction_name if self.prep else ""
        mode = self.prep.cfg.adjacency_mode if self.prep else "?"
        thresh = self.prep.cfg.adjacency_threshold_ft if self.prep else 0.0
        header = (
            f"{name}   adjacency_mode={mode}   threshold={thresh:.0f} ft\n"
            + "-" * 68 + "\n"
        )
        text = header + self.diag.summary(self.diag_tile_n_ids)
        self.diag_report.setPlainText(text)
        for line in text.splitlines():
            self.log(line)

    # ------------------------------------------------------------------

    def on_stats(self, s: OptimizerStats) -> None:
        rate = s.rate
        # Parcel-weighted, not the plain mean: after severed neighborhoods are
        # split apart, hundreds of one-parcel island neighborhoods score 1.0 by
        # design and inflate a plain mean by ~60x.
        groups = getattr(s, "groups_running", None)
        suffix = f"   workers {groups}" if groups else ""
        self.stats_label.setText(
            f"iter {s.iteration:>9,}   T {s.temperature:>8.5f}   "
            f"score {getattr(s, 'weighted_score', s.mean_score):>10.6f}   "
            f"boundary {s.boundary_size:>8,}   "
            f"accepted {s.accepted:>8,}   hoods {s.active_neighborhoods:>5,}   "
            f"{rate:6.1f} it/s{suffix}"
        )

    def on_snapshot(self, tile_n_ids, s: OptimizerStats) -> None:
        name = self.prep.cfg.jurisdiction_name if self.prep else ""
        self.canvas.update_colors(
            np.asarray(tile_n_ids),
            f"{name} — iteration {s.iteration:,}  ·  score {s.mean_score:.6f}",
        )

    def on_run_finished(self, prep, out) -> None:
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.start_btn.setEnabled(True)
        self.resume_btn.setEnabled(True)
        self.statusBar().showMessage(
            f"Run finished — {Path(out).name}" if out else "Run finished"
        )
        self.refresh_runs()

    # ------------------------------------------------------------------

    def on_pause(self) -> None:
        if not self.prep:
            return
        runner = getattr(self.opt_thread, "runner", None)
        ev = self.prep.optimizer.pause_event
        if ev.is_set() or (runner is not None and runner.pause_event.is_set()):
            ev.clear()
            if runner is not None:
                runner.set_paused(False)
            self.pause_btn.setText("Pause")
            self.statusBar().showMessage("Running")
        else:
            # The worker writes the checkpoint itself once it parks, so the
            # state it captures is consistent.
            ev.set()
            if runner is not None:
                runner.set_paused(True)
            self.pause_btn.setText("Continue")
            self.statusBar().showMessage("Paused — a checkpoint is being written")

    def on_stop(self) -> None:
        if not self.prep:
            return
        self.log("Stopping — a final checkpoint will be written.")
        self.prep.optimizer.pause_event.clear()
        self.prep.optimizer.stop_event.set()
        runner = getattr(self.opt_thread, "runner", None)
        if runner is not None:
            runner.set_paused(False)
            runner.stop()
        self.stop_btn.setEnabled(False)
        self.pause_btn.setEnabled(False)

    def on_export(self) -> None:
        if not self.prep:
            return
        # Exporting the parcel table mid-run would read parcel_n_ids while the
        # worker is rewriting it, so the file could be internally inconsistent.
        # A PNG of the last painted frame is always fine.
        if self.running() and not self.prep.optimizer.pause_event.is_set():
            answer = QtWidgets.QMessageBox.question(
                self,
                "Run in progress",
                "Pause the run first so the export is a consistent snapshot?\n\n"
                "Yes — pause, then export.\n"
                "No — export the map image only.",
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No
                | QtWidgets.QMessageBox.StandardButton.Cancel,
            )
            if answer == QtWidgets.QMessageBox.StandardButton.Cancel:
                return
            if answer == QtWidgets.QMessageBox.StandardButton.Yes:
                self.on_pause()
                path, _ = QtWidgets.QFileDialog.getSaveFileName(
                    self, "Export parcels with neighborhood_id",
                    str(self.prep.run_dir / "optimized_neighborhoods_tiled.parquet"),
                    "Parquet (*.parquet);;GeoPackage (*.gpkg);;PNG map (*.png)",
                )
            else:
                path, _ = QtWidgets.QFileDialog.getSaveFileName(
                    self, "Export map image",
                    str(self.prep.run_dir / "map.png"), "PNG map (*.png)",
                )
            if path:
                self._write_export(Path(path))
            return

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export parcels with neighborhood_id",
            str(self.prep.run_dir / "optimized_neighborhoods_tiled.parquet"),
            "Parquet (*.parquet);;GeoPackage (*.gpkg);;PNG map (*.png)",
        )
        if path:
            self._write_export(Path(path))

    def _write_export(self, p: Path) -> None:
        if not self.prep:
            return
        try:
            if p.suffix.lower() == ".png":
                self.canvas.save_png(str(p))
            elif p.suffix.lower() == ".gpkg":
                self.prep.optimizer.result_frame(self.prep.parcels).to_file(
                    p, driver="GPKG"
                )
            else:
                self.prep.optimizer.result_frame(self.prep.parcels).to_parquet(p)
            self.log(f"Exported {p}")
        except Exception as exc:  # noqa: BLE001
            # Deliberately not on_failed(): a failed export must not re-enable
            # Start while the optimizer is still running.
            self._error_box(f"Export failed: {exc}")

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    KEEP, GRACEFUL, FORCE = "keep", "graceful", "force"

    def _ask_close(
        self, title: str, text: str, detail: str, graceful_label: Optional[str] = None
    ) -> str:
        """Warn about work in flight, but never without an exit.

        Advising someone against closing is fine; refusing to let them is not.
        There is always an "Exit Anyway" button, and "Keep Running" is the
        default so Enter/Escape do the safe thing.
        """
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        box.setWindowTitle(title)
        box.setText(text)
        box.setInformativeText(detail)

        keep = box.addButton(
            "Keep Running", QtWidgets.QMessageBox.ButtonRole.RejectRole
        )
        graceful = (
            box.addButton(graceful_label, QtWidgets.QMessageBox.ButtonRole.AcceptRole)
            if graceful_label else None
        )
        force = box.addButton(
            "Exit Anyway", QtWidgets.QMessageBox.ButtonRole.DestructiveRole
        )
        box.setDefaultButton(keep)
        box.setEscapeButton(keep)
        box.exec()

        clicked = box.clickedButton()
        if clicked is force:
            return self.FORCE
        if graceful is not None and clicked is graceful:
            return self.GRACEFUL
        return self.KEEP

    def _wait_for_stop(self, seconds: float) -> bool:
        """Wait for the optimizer thread, keeping the UI painting.

        A bare ``QThread.wait(60000)`` freezes the window for up to a minute,
        which is indistinguishable from the hang the user was trying to escape.
        """
        deadline = time.monotonic() + seconds
        app = QtWidgets.QApplication.instance()
        while self.opt_thread and self.opt_thread.isRunning():
            if time.monotonic() > deadline:
                return False
            self.opt_thread.wait(100)
            if app is not None:
                app.processEvents(
                    QtCore.QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents
                )
        return True

    def _force_quit(self) -> None:
        """Leave now, whatever is in flight. Does not return."""
        self.log("Force exit requested — abandoning work in progress.")
        self.statusBar().showMessage("Exiting…")

        # Worker processes first. os._exit skips atexit handlers, so spawned
        # children would otherwise outlive the parent and keep burning CPU with
        # no window to show for it.
        runner = getattr(self.opt_thread, "runner", None)
        if runner is not None:
            try:
                runner.stop()
                runner.kill_workers()
            except Exception:  # noqa: BLE001 - we are leaving regardless
                pass
        if self.prep is not None:
            try:
                self.prep.optimizer.pause_event.clear()
                self.prep.optimizer.stop_event.set()
            except Exception:  # noqa: BLE001
                pass

        # A short grace period in case a cooperative stop is about to land a
        # checkpoint anyway; then go, without waiting on the uninterruptible
        # geometry work that prompted this.
        if self.opt_thread is not None and self.opt_thread.isRunning():
            self._wait_for_stop(2.0)

        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except Exception:  # noqa: BLE001
                pass
        # Not event.accept(): destroying the window would take its still-running
        # QThread children with it and abort, which on Windows means a crash
        # dialog. _exit is the quiet door. Cache writes are atomic (see
        # pipeline._atomic_write), so nothing on disk can be left half-written.
        os._exit(0)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        if getattr(self, "_closing", False):
            event.ignore()
            return

        if self.task is not None and self.task.isRunning():
            self._closing = True
            try:
                choice = self._ask_close(
                    "Still preparing",
                    "Downloads and tiling are still running.",
                    "This only happens once per jurisdiction — the results are "
                    "cached, so letting it finish saves repeating the work. "
                    "Tiling a large county can take several minutes with no "
                    "visible movement.\n\n"
                    "Exiting now is safe: cache files are written atomically, so "
                    "nothing on disk will be left corrupted. You will just have "
                    "to redo the tiling next time.",
                )
            finally:
                self._closing = False
            if choice == self.KEEP:
                event.ignore()
                return
            self._force_quit()  # does not return

        if self.running():
            self._closing = True
            try:
                choice = self._ask_close(
                    "Optimization running",
                    "The optimizer is still running.",
                    "Stopping saves a final checkpoint, so you can resume exactly "
                    "where you left off.\n\n"
                    "Exiting immediately abandons any progress since the last "
                    "checkpoint.",
                    graceful_label="Stop && Save",
                )
            finally:
                self._closing = False

            if choice == self.KEEP:
                event.ignore()
                return
            if choice == self.FORCE:
                self._force_quit()  # does not return

            # Graceful: ask it to stop, then wait a bounded while.
            self.on_stop()
            self.statusBar().showMessage("Stopping and saving a checkpoint…")
            if not self._wait_for_stop(30.0):
                followup = self._ask_close(
                    "Still shutting down",
                    "The optimizer hasn't stopped yet.",
                    "It may be writing a large checkpoint, or be inside a long "
                    "geometry operation that can't be interrupted.\n\n"
                    "Give it longer, or exit now and lose progress since the "
                    "last checkpoint.",
                    graceful_label="Wait Longer",
                )
                if followup == self.FORCE:
                    self._force_quit()  # does not return
                if followup == self.GRACEFUL and self._wait_for_stop(60.0):
                    pass
                elif self.opt_thread is not None and self.opt_thread.isRunning():
                    # Still going and they didn't ask to force: leaving the
                    # window open beats aborting the process under them.
                    event.ignore()
                    return
        event.accept()


def main(argv: Optional[List[str]] = None) -> int:
    app = QtWidgets.QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("MI Neighborhoods")
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
