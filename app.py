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

import pipeline
import pyarrow
import sources
from checkpoints import Checkpoint, CheckpointStore, find_runs
from config import (
    CONTINUOUS_VARIABLES,
    DEFAULT_WEIGHTS,
    RunConfig,
    runs_dir,
)
from engine import OptimizerStats
from mapview import MapCanvas, make_toolbar
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
    def __init__(self, parent=None):
        super().__init__(0, 2, parent)
        self.setHorizontalHeaderLabels(["Binned field", "Weight"])
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
        self.set_weights(DEFAULT_WEIGHTS)

    def set_weights(self, weights: dict) -> None:
        self.setRowCount(0)
        for name, w in weights.items():
            self.add_row(name, w)

    def add_row(self, name: str = "", weight: float = 1.0) -> None:
        r = self.rowCount()
        self.insertRow(r)
        self.setItem(r, 0, QtWidgets.QTableWidgetItem(name))
        self.setItem(r, 1, QtWidgets.QTableWidgetItem(f"{weight:g}"))

    def remove_selected(self) -> None:
        for r in sorted({i.row() for i in self.selectedIndexes()}, reverse=True):
            self.removeRow(r)

    def weights(self) -> dict:
        out = {}
        for r in range(self.rowCount()):
            name_item = self.item(r, 0)
            w_item = self.item(r, 1)
            if not name_item or not name_item.text().strip():
                continue
            try:
                out[name_item.text().strip()] = float(w_item.text()) if w_item else 1.0
            except ValueError:
                out[name_item.text().strip()] = 1.0
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
        lv.addWidget(self._tiling_box())
        lv.addWidget(self._scoring_box(), 1)
        lv.addWidget(self._control_box())

        # ---------------- right column ----------------
        right = QtWidgets.QWidget()
        rv = QtWidgets.QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)

        self.canvas = MapCanvas()
        rv.addWidget(make_toolbar(self.canvas, self))
        rv.addWidget(self.canvas, 1)

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

        splitter.addWidget(left)
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
        browse = QtWidgets.QPushButton("Browse…")
        browse.clicked.connect(self.on_browse_parcels)

        self.filter_col = QtWidgets.QLineEdit(self.cfg.parcel_filter_column)
        self.filter_val = QtWidgets.QLineEdit(self.cfg.parcel_filter_value)

        g.addWidget(QtWidgets.QLabel("File"), 0, 0)
        g.addWidget(self.parcel_edit, 0, 1, 1, 2)
        g.addWidget(browse, 0, 3)
        g.addWidget(QtWidgets.QLabel("Filter"), 1, 0)
        g.addWidget(self.filter_col, 1, 1)
        g.addWidget(QtWidgets.QLabel("=="), 1, 2)
        g.addWidget(self.filter_val, 1, 3)
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
        self.water_chk = QtWidgets.QCheckBox("Clip OSM water")
        self.water_chk.setChecked(True)

        g.addWidget(QtWidgets.QLabel("Grid"), 0, 0)
        g.addWidget(self.grid_spin, 0, 1)
        g.addWidget(QtWidgets.QLabel("Adjacency"), 0, 2)
        g.addWidget(self.adj_spin, 0, 3)
        g.addWidget(self.blocks_chk, 1, 0, 1, 2)
        g.addWidget(self.roads_chk, 1, 2, 1, 2)
        g.addWidget(self.water_chk, 2, 0, 1, 2)
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
        v.addLayout(form)

        self.weights_table = WeightsTable()
        v.addWidget(self.weights_table, 1)

        row = QtWidgets.QHBoxLayout()
        add = QtWidgets.QPushButton("Add field")
        add.clicked.connect(lambda: self.weights_table.add_row("", 1.0))
        rem = QtWidgets.QPushButton("Remove")
        rem.clicked.connect(self.weights_table.remove_selected)
        reset = QtWidgets.QPushButton("Defaults")
        reset.clicked.connect(lambda: self.weights_table.set_weights(DEFAULT_WEIGHTS))
        row.addWidget(add)
        row.addWidget(rem)
        row.addWidget(reset)
        row.addStretch(1)
        v.addLayout(row)

        note = QtWidgets.QLabel(
            "Binned fields are created automatically from: "
            + ", ".join(CONTINUOUS_VARIABLES)
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #777; font-size: 10px;")
        v.addWidget(note)
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
        cfg.parcel_filter_column = self.filter_col.text().strip()
        cfg.parcel_filter_value = self.filter_val.text().strip()
        cfg.grid_size_ft = float(self.grid_spin.value())
        cfg.adjacency_threshold_ft = float(self.adj_spin.value())
        cfg.use_census_blocks = self.blocks_chk.isChecked()
        cfg.use_osm_roads = self.roads_chk.isChecked()
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
        cfg.weights = self.weights_table.weights()
        return cfg

    def apply_config(self, cfg: RunConfig) -> None:
        self.jur_edit.setText(cfg.jurisdiction_query)
        self.parcel_edit.setText(cfg.parcel_path)
        self.filter_col.setText(cfg.parcel_filter_column)
        self.filter_val.setText(cfg.parcel_filter_value)
        self.grid_spin.setValue(cfg.grid_size_ft)
        self.adj_spin.setValue(cfg.adjacency_threshold_ft)
        self.blocks_chk.setChecked(cfg.use_census_blocks)
        self.roads_chk.setChecked(cfg.use_osm_roads)
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
        self.weights_table.set_weights(cfg.weights)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def on_browse_parcels(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Choose a parcel file",
            str(Path.cwd()),
            "Spatial data (*.parquet *.pq *.gpkg *.shp *.geojson *.json);;All files (*)",
        )
        if path:
            self.parcel_edit.setText(path)

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
        jur = self.matches[idx]
        self._prepare_and_run(cfg, jurisdiction=jur, run_dir=None, checkpoint=None)

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
