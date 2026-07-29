"""End-to-end orchestration: jurisdiction -> tiles -> parcels -> optimizer.

Everything here is UI-free so it can also be driven from a script or notebook.
Intermediate artefacts are written into the run directory, so re-opening a run
skips the downloads and the (expensive) shatter entirely.
"""

from __future__ import annotations

import datetime as _dt
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

import adjacency
import engine
import parallel
import partition
import sources
import tiger
import tiles as tiles_mod
from checkpoints import CheckpointStore
from config import RunConfig, repo_root, runs_dir
from geo import crs_is_feet, describe_crs, pick_feet_crs
from sources import Jurisdiction

Progress = Callable[[str], None]


def _noop(_: str) -> None:
    pass


def _atomic_write(write, path: Path) -> Path:
    """Run ``write(tmp)`` then rename into place.

    Cache files have to be all-or-nothing. The app now offers a force-quit while
    tiling is in flight, and a half-written ``tiles.parquet`` left on disk would
    be picked up as a valid cache on the next run and fail to parse.
    """
    path = Path(path)
    # The suffix has to come last: np.savez_compressed silently appends ".npz"
    # to any filename that doesn't already end in it, so "x.npz.part" would be
    # written as "x.npz.part.npz" and the rename below would miss it.
    tmp = path.with_name(f"{path.stem}.part{path.suffix}")
    tmp.unlink(missing_ok=True)
    write(tmp)
    if not tmp.exists():
        raise OSError(f"{write!r} did not produce {tmp}")
    os.replace(tmp, path)  # atomic on the same filesystem, Windows included
    return path


def _read_cached(read, path: Path, progress: Progress, what: str):
    """Read a cache file, treating damage as "not cached" rather than fatal."""
    try:
        return read(path)
    except Exception as exc:  # noqa: BLE001
        progress(
            f"Ignoring unreadable cached {what} ({exc}); it will be rebuilt."
        )
        try:
            Path(path).unlink(missing_ok=True)
        except OSError:
            pass
        return None


def slugify(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return s or "run"


def fetch_blocks(
    jurisdiction: Jurisdiction, cfg: RunConfig, progress: Progress = _noop
) -> gpd.GeoDataFrame:
    """Census blocks for the study area, bulk by default.

    The bulk TIGER/Line path has no feature cap, so it's the only one that
    survives a county the size of Cook. The REST fetcher stays as an explicit
    opt-out and as an automatic fallback, since it avoids a large download for a
    single small city.
    """
    if cfg.census_source == "rest":
        return sources.fetch_census_blocks(jurisdiction, progress=progress)

    try:
        return tiger.fetch_blocks(jurisdiction, progress=progress)
    except Exception as exc:  # noqa: BLE001
        progress(
            f"Bulk TIGER/Line fetch failed ({exc}). Falling back to the TIGERweb "
            "API, which truncates above 100k blocks -- check the block count "
            "below against the size of the study area."
        )
        return sources.fetch_census_blocks(jurisdiction, progress=progress)


@dataclass
class PreparedRun:
    cfg: RunConfig
    run_dir: Path
    jurisdiction_gdf: gpd.GeoDataFrame
    working_crs: object
    tiles: gpd.GeoDataFrame            # tile_id -> geometry (populated tiles only)
    parcels: gpd.GeoDataFrame          # prepared, reset index, has tile_id
    optimizer: engine.TiledOptimizer
    store: CheckpointStore
    components: Optional["partition.Components"] = None
    parcel_component: Optional[np.ndarray] = None
    # True when some neighborhood straddles a severed component, which makes the
    # parallel runner unsafe: two workers would write the same count-table rows.
    neighborhoods_span_components: bool = False
    tile_to_parcels: Optional[Dict[int, np.ndarray]] = None
    tile_adjacency: Optional[Dict[int, set]] = None

    def tile_geometries(self) -> gpd.GeoSeries:
        """Tile geometries in the optimizer's tile order."""
        return self.tiles.geometry.reindex(self.optimizer.tile_ids)

    # ------------------------------------------------------------------
    def worker_count(self) -> int:
        """How many processes are worth starting for this tileset.

        Forced to 1 if any neighborhood spans a severed component. The whole
        no-shared-state argument for parallelism rests on neighborhoods being
        confined to one component; without that, two workers write the same
        count-table rows and the scores are quietly wrong. That is the case
        whenever `split_severed_neighborhoods` is off.
        """
        if self.components is None:
            return 1
        if self.neighborhoods_span_components:
            return 1
        return partition.useful_worker_count(
            self.components.parcel_counts, self.cfg.workers
        )

    def make_parallel_runner(
        self, progress: Callable[[str], None] = _noop
    ) -> "parallel.ParallelRunner":
        """A runner that anneals each severed component group in its own process."""
        if self.components is None or self.tile_to_parcels is None:
            raise RuntimeError("prepare() did not compute components")

        if self.neighborhoods_span_components:
            raise RuntimeError(
                "Refusing to anneal in parallel: some neighborhood spans a "
                "severed component, so workers would write the same count-table "
                "rows. Enable split_severed_neighborhoods, or set workers = 1."
            )
        n = self.worker_count()
        groups = partition.group_components(self.components.parcel_counts, n)
        progress(partition.describe_grouping(groups, self.components.parcel_counts))

        tasks = parallel.build_group_tasks(
            groups, self.components, self.tile_to_parcels, self.tile_adjacency or {}
        )
        return parallel.ParallelRunner(
            cfg=self.cfg,
            parcels_path=str(self.run_dir / "parcels_prepared.parquet"),
            tasks=tasks,
            parcel_n_ids=self.optimizer.parcel_n_ids,
            n_neighborhoods=self.optimizer.n_neighborhoods,
            progress=progress,
        )

    def parcel_ids_from_tiles(self, tile_n_ids: np.ndarray) -> np.ndarray:
        """Expand a merged tile-level assignment back to parcels."""
        out = self.optimizer.parcel_n_ids.copy()
        t2p = self.tile_to_parcels or {}
        for t, value in zip(self.optimizer.tile_ids, tile_n_ids):
            idx = t2p.get(int(t))
            if idx is not None:
                out[idx] = value
        return out


# ==========================================================================
# Parcels
# ==========================================================================


def load_parcels(
    path: str | Path,
    cfg: RunConfig,
    working_crs,
    clip_to: Optional[gpd.GeoDataFrame] = None,
    progress: Progress = _noop,
) -> gpd.GeoDataFrame:
    path = Path(path)
    progress(f"Reading parcels from {path.name}...")
    suffix = path.suffix.lower()
    if suffix in (".parquet", ".pq"):
        parcels = gpd.read_parquet(path)
    else:
        parcels = gpd.read_file(path)

    if parcels.crs is None:
        raise ValueError(
            f"{path.name} has no CRS. Set one (or export with a CRS) before loading."
        )
    progress(f"Loaded {len(parcels):,} parcels ({describe_crs(parcels.crs)})")

    col, val = cfg.parcel_filter_column, cfg.parcel_filter_value
    if col and val and col in parcels.columns:
        before = len(parcels)
        parcels = parcels[parcels[col].astype(str).eq(val)]
        progress(f"Filter {col}=={val}: {before:,} -> {len(parcels):,}")

    parcels = parcels[~parcels.geometry.is_empty & parcels.geometry.notna()]

    if parcels.crs != working_crs:
        progress(f"Reprojecting parcels to {describe_crs(working_crs)}...")
        parcels = parcels.to_crs(working_crs)

    if clip_to is not None and len(clip_to):
        area = clip_to.to_crs(working_crs).geometry.union_all()
        before = len(parcels)
        pts = parcels.geometry.representative_point()
        parcels = parcels[pts.within(area)]
        progress(f"Clipped to jurisdiction: {before:,} -> {len(parcels):,} parcels")

    if parcels.empty:
        raise ValueError(
            "No parcels fall inside the chosen jurisdiction. Check the parcel "
            "file's extent or pick a different jurisdiction."
        )

    # latitude/longitude are used for KMeans seeding; derive if absent.
    if "latitude" not in parcels.columns or "longitude" not in parcels.columns:
        progress("Deriving latitude/longitude from geometry for seeding...")
        cent = parcels.geometry.representative_point().to_crs(4326)
        parcels = parcels.assign(longitude=cent.x.values, latitude=cent.y.values)

    parcels = engine.add_derived_columns(parcels, progress=progress)
    parcels = engine.bin_continuous_fields(
        parcels, cfg.continuous_variables, cfg.max_bins, progress=progress
    )

    missing = [f for f in cfg.weights if f not in parcels.columns]
    if missing:
        raise ValueError(
            "These scored fields are missing from the parcel data: "
            + ", ".join(missing)
            + ". Adjust the weights, or make sure the source columns exist "
              "(binned columns are created automatically from "
            + ", ".join(cfg.continuous_variables) + ")."
        )

    return parcels.reset_index(drop=True)


def load_obstacle_parcels(
    cfg: RunConfig,
    working_crs,
    clip_to: Optional[gpd.GeoDataFrame] = None,
    progress: Progress = _noop,
) -> Tuple[Optional[np.ndarray], Optional[pd.Series]]:
    """The *unfiltered* parcels, for the sightline obstacle set.

    ``load_parcels`` throws away everything outside the model group, but
    ``obstacle_mode`` "all" / "all_except" need those rows back -- an agricultural
    parcel blocks a sightline even though it is never optimized. Only geometry and
    the land-class column are read, so this is cheap next to the main load.
    """
    path = Path(cfg.parcel_path)
    if not path.exists():
        progress(f"Cannot read obstacles: {path} is missing")
        return None, None

    want = ["geometry"]
    land_col = cfg.land_class_column
    try:
        if path.suffix.lower() in (".parquet", ".pq"):
            import pyarrow.parquet as pq

            available = set(pq.ParquetFile(path).schema_arrow.names)
            if land_col in available:
                want.append(land_col)
            frame = gpd.read_parquet(path, columns=want)
        else:
            frame = gpd.read_file(path)
            if land_col not in frame.columns:
                land_col = None
    except Exception as exc:  # noqa: BLE001
        progress(f"Cannot read obstacles ({exc}); sightlines will use the "
                 "modeled parcels only.")
        return None, None

    frame = frame[~frame.geometry.is_empty & frame.geometry.notna()]
    if frame.crs is not None and frame.crs != working_crs:
        frame = frame.to_crs(working_crs)
    if clip_to is not None and len(clip_to):
        area = clip_to.to_crs(working_crs).geometry.union_all()
        shapely.prepare(area)
        inside = shapely.intersects(area, frame.geometry.to_numpy())
        frame = frame[inside]

    progress(f"Obstacle candidates: {len(frame):,} parcels of every class")
    land = frame[land_col] if land_col and land_col in frame.columns else None
    if land is None:
        progress(
            f"No '{cfg.land_class_column}' column, so in-gap classes cannot be "
            "identified; 'all_except' will behave like 'all'."
        )
    return frame.geometry.to_numpy(), land


def build_tile_adjacency(
    parcels: gpd.GeoDataFrame,
    adj_tiles: gpd.GeoDataFrame,
    cfg: RunConfig,
    jur_gdf: Optional[gpd.GeoDataFrame] = None,
    progress: Progress = _noop,
) -> Dict[int, set]:
    """Tile adjacency, by whichever rule ``cfg.adjacency_mode`` selects."""
    if cfg.adjacency_mode != "parcel":
        return tiles_mod.calculate_adjacency(
            adj_tiles, cfg.adjacency_threshold_ft, progress=progress
        )

    modeled = parcels.geometry.to_numpy()
    all_geoms = land = None
    if cfg.obstacle_mode != "modeled" and cfg.require_line_of_sight:
        all_geoms, land = load_obstacle_parcels(
            cfg, parcels.crs, clip_to=jur_gdf, progress=progress
        )

    obstacles = adjacency.select_obstacles(
        modeled, mode=cfg.obstacle_mode, all_geoms=all_geoms,
        all_land_class=land,
        keywords=(cfg.transparent_land_class_keywords
                  or adjacency.DEFAULT_TRANSPARENT_KEYWORDS),
        progress=progress,
    )
    result = adjacency.build_parcel_adjacency(
        modeled,
        threshold_ft=cfg.adjacency_threshold_ft,
        obstacle_geoms=obstacles,
        require_line_of_sight=cfg.require_line_of_sight,
        epsilon_ft=cfg.sightline_epsilon_ft,
        progress=progress,
    )
    adj = result.to_tile_adjacency(
        parcels["tile_id"].to_numpy(),
        all_tiles=np.asarray(adj_tiles.index, dtype=np.int64),
    )
    edges = sum(len(v) for v in adj.values()) // 2
    progress(
        f"Adjacency: {len(adj):,} tiles, {edges:,} tile edges "
        f"({edges / max(len(adj), 1):.2f} avg)"
    )
    return adj


def parcel_field_candidates(path: str | Path) -> List[str]:
    """Numeric columns in a parcel file, for populating the weights editor."""
    path = Path(path)
    if path.suffix.lower() in (".parquet", ".pq"):
        import pyarrow as pa
        import pyarrow.parquet as pq

        schema = pq.ParquetFile(path).schema_arrow
        return [
            name
            for name, dtype in zip(schema.names, schema.types)
            if pa.types.is_integer(dtype) or pa.types.is_floating(dtype)
        ]
    head = gpd.read_file(path, rows=50)
    return [c for c in head.columns if pd.api.types.is_numeric_dtype(head[c])]


# ==========================================================================
# Run directory
# ==========================================================================


def create_run_dir(cfg: RunConfig, label: str) -> Path:
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    d = runs_dir(cfg) / f"{slugify(label)}_{stamp}"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ==========================================================================
# Preparation
# ==========================================================================


def prepare(
    cfg: RunConfig,
    jurisdiction: Optional[Jurisdiction] = None,
    run_dir: Optional[Path] = None,
    progress: Progress = _noop,
    use_cache: bool = True,
) -> PreparedRun:
    """Build (or reload) everything needed to start/resume an optimization."""

    # ---- 1. study area -------------------------------------------------
    if run_dir is not None and (Path(run_dir) / "jurisdiction.parquet").exists():
        run_dir = Path(run_dir)
        jur_gdf = gpd.read_parquet(run_dir / "jurisdiction.parquet")
        progress(f"Reusing study area from {run_dir.name}")
    else:
        if jurisdiction is None:
            progress(f"Looking up '{cfg.jurisdiction_query}'...")
            jurisdiction = sources.get_jurisdiction(
                cfg.jurisdiction_query, progress=progress
            )
        cfg.jurisdiction_geoid = jurisdiction.geoid
        cfg.jurisdiction_name = jurisdiction.name
        cfg.jurisdiction_layer = jurisdiction.layer_name
        progress(f"Study area: {jurisdiction.label}")
        jur_gdf = jurisdiction.to_gdf()
        run_dir = Path(run_dir) if run_dir else create_run_dir(cfg, jurisdiction.label)
        _atomic_write(jur_gdf.to_parquet, run_dir / "jurisdiction.parquet")

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- 2. working CRS -------------------------------------------------
    lonlat_bounds = tuple(jur_gdf.to_crs(4326).total_bounds)
    working_crs = pick_feet_crs(lonlat_bounds)  # type: ignore[arg-type]
    if not crs_is_feet(working_crs):
        raise RuntimeError("Failed to find a feet-based CRS for this study area.")
    progress(f"Working CRS: {describe_crs(working_crs)}")

    # ---- 3. tiles --------------------------------------------------------
    tiles_path = run_dir / "tiles.parquet"
    tileset = None
    if use_cache and tiles_path.exists():
        tileset = _read_cached(gpd.read_parquet, tiles_path, progress, "tileset")
        if tileset is not None:
            progress(f"Reusing {len(tileset):,} tiles from {tiles_path.name}")
    if tileset is None:
        if jurisdiction is None:
            jurisdiction = _jurisdiction_from_gdf(jur_gdf, cfg)
        blocks = (
            fetch_blocks(jurisdiction, cfg, progress=progress)
            if cfg.use_census_blocks else None
        )
        roads = (
            sources.fetch_osm_roads(jurisdiction, progress=progress)
            if cfg.use_osm_roads else None
        )
        water_lines = (
            sources.fetch_osm_waterway_lines(jurisdiction, progress=progress)
            if cfg.use_osm_waterways else None
        )
        water_areas = (
            sources.fetch_osm_water_areas(jurisdiction, progress=progress)
            if cfg.clip_water else None
        )
        if not (cfg.use_osm_roads or cfg.use_osm_waterways or cfg.clip_water):
            progress(
                "Every OSM layer is switched off, so Overpass was not contacted; "
                "tiles come from census blocks plus the grid alone."
            )
        tileset = tiles_mod.build_tileset(
            jur_gdf,
            blocks,
            roads,
            water_lines,
            water_areas,
            working_crs,
            grid_size_ft=cfg.grid_size_ft,
            clip_water=cfg.clip_water,
            progress=progress,
        )
        _atomic_write(tileset.to_parquet, tiles_path)

    if tileset.crs != working_crs:
        tileset = tileset.to_crs(working_crs)
    tileset = tileset.reset_index(drop=True)
    tileset.index.name = "tile_id"

    # ---- 4. parcels ------------------------------------------------------
    prepared_path = run_dir / "parcels_prepared.parquet"
    parcels = None
    if use_cache and prepared_path.exists():
        parcels = _read_cached(
            gpd.read_parquet, prepared_path, progress, "prepared parcels"
        )
    if parcels is not None:
        progress(f"Reusing {len(parcels):,} prepared parcels")
        all_tiles = _tiles_for(parcels, tileset, working_crs)
        tile_to_parcels = {
            int(t): np.asarray(idx, dtype=np.int64)
            for t, idx in parcels.groupby("tile_id").indices.items()
        }
    else:
        parcels = load_parcels(
            cfg.parcel_path, cfg, working_crs, clip_to=jur_gdf, progress=progress
        )
        parcels, all_tiles, tile_to_parcels = tiles_mod.assign_parcels_to_tiles(
            parcels, tileset, progress=progress
        )
        _atomic_write(parcels.to_parquet, prepared_path)

    # ---- 5. adjacency ----------------------------------------------------
    populated = sorted(tile_to_parcels.keys())
    adj_tiles = all_tiles.loc[all_tiles.index.intersection(populated)]
    tile_adj = build_tile_adjacency(
        parcels, adj_tiles, cfg, jur_gdf=jur_gdf, progress=progress
    )

    # ---- 6. seeding ------------------------------------------------------
    seeded_path = run_dir / "initial_neighborhoods.npz"
    if use_cache and seeded_path.exists():
        with np.load(seeded_path) as d:
            seed_ids = d["neighborhood_id"].astype(np.int64)
        if len(seed_ids) != len(parcels):
            seed_ids = None  # type: ignore[assignment]
        else:
            progress("Reusing KMeans seeding")
    else:
        seed_ids = None  # type: ignore[assignment]

    if seed_ids is None:  # type: ignore[comparison-overlap]
        seed_ids = engine.seed_neighborhoods(
            parcels, cfg.seed_fields, cfg.n_neighborhoods,
            random_state=cfg.random_seed, progress=progress,
        )
        _atomic_write(
            lambda p: np.savez_compressed(p, neighborhood_id=seed_ids.astype(np.int32)),
            seeded_path,
        )

    # ---- 6b. severed components -------------------------------------------
    comps = partition.find_components(tile_to_parcels, tile_adj, progress=progress)
    parcel_comp = partition.parcel_components(comps, tile_to_parcels, len(parcels))

    if cfg.split_severed_neighborhoods:
        split_path = run_dir / "split_neighborhoods.npz"
        if use_cache and split_path.exists():
            with np.load(split_path) as d:
                cached = d["neighborhood_id"].astype(np.int64)
            if len(cached) == len(parcels):
                seed_ids = cached
                progress("Reusing split neighborhood labels")
            else:
                seed_ids = None  # type: ignore[assignment]
        if seed_ids is None or not split_path.exists():
            seed_ids, _ = partition.split_neighborhoods(
                seed_ids, parcel_comp, progress=progress
            )
            _atomic_write(
                lambda p: np.savez_compressed(
                    p, neighborhood_id=seed_ids.astype(np.int32)
                ),
                split_path,
            )
        # An unsplit neighborhood spanning components is unfixable, and would
        # also make the parallel runner race on shared count-table rows.
        leftover = partition.spanning_neighborhoods(seed_ids, parcel_comp)
        if leftover:
            raise RuntimeError(
                f"{len(leftover)} neighborhoods still span severed components "
                "after splitting; refusing to continue."
            )

    spans = partition.any_spanning(seed_ids, parcel_comp)
    if spans and cfg.workers != 1:
        progress(
            "Some neighborhoods straddle severed components (splitting is off), "
            "so parallel annealing would race on shared count-table rows. "
            "Falling back to a single worker."
        )

    # ---- 7. optimizer ----------------------------------------------------
    progress("Building count tables...")
    opt = engine.TiledOptimizer(
        parcels, tile_to_parcels, tile_adj, cfg,
        neighborhood_ids=seed_ids, progress=progress,
    )

    store = CheckpointStore(run_dir, keep=cfg.keep_checkpoints)
    cfg.save(run_dir / "run_config.json")

    return PreparedRun(
        cfg=cfg,
        run_dir=run_dir,
        jurisdiction_gdf=jur_gdf,
        working_crs=working_crs,
        tiles=adj_tiles,
        parcels=parcels,
        optimizer=opt,
        store=store,
        components=comps,
        parcel_component=parcel_comp,
        neighborhoods_span_components=spans,
        tile_to_parcels=tile_to_parcels,
        tile_adjacency=tile_adj,
    )


def _tiles_for(
    parcels: gpd.GeoDataFrame, tileset: gpd.GeoDataFrame, working_crs
) -> gpd.GeoDataFrame:
    """Rebuild the tile geometry table (incl. virtual tiles) for a cached run."""
    base = gpd.GeoDataFrame(
        geometry=tileset.geometry.to_numpy(),
        index=pd.Index(np.asarray(tileset.index, dtype=np.int64), name="tile_id"),
        crs=working_crs,
    )
    tid = parcels["tile_id"].to_numpy()
    orphan = ~np.isin(tid, base.index.to_numpy())
    if not orphan.any():
        return base.sort_index()

    # Virtual tiles are one-per-orphan-parcel, so first occurrence wins.
    ids, first = np.unique(tid[orphan], return_index=True)
    virtual = gpd.GeoDataFrame(
        geometry=parcels.geometry.to_numpy()[orphan][first],
        index=pd.Index(ids.astype(np.int64), name="tile_id"),
        crs=working_crs,
    )
    return gpd.GeoDataFrame(pd.concat([base, virtual]), crs=working_crs).sort_index()


def _jurisdiction_from_gdf(gdf: gpd.GeoDataFrame, cfg: RunConfig) -> Jurisdiction:
    """Rebuild a Jurisdiction from a saved jurisdiction.parquet."""
    g = gdf.to_crs(4326)
    row = g.iloc[0]

    def field(col: str, fallback: str = "") -> str:
        val = row.get(col) if col in g.columns else None
        return "" if val is None or pd.isna(val) else str(val)

    layer_id = field("layer_id")
    return Jurisdiction(
        name=field("name") or cfg.jurisdiction_name or "study area",
        layer_id=int(layer_id) if layer_id.isdigit() else 0,
        layer_name=field("layer_name") or cfg.jurisdiction_layer or "custom",
        geoid=field("geoid") or cfg.jurisdiction_geoid,
        # Needed by the bulk block fetcher to pick the right state file. Older
        # run directories predate the column, so fall back to the GEOID prefix
        # (every Census GEOID starts with the 2-digit state code).
        state_fips=field("state_fips") or (field("geoid") or cfg.jurisdiction_geoid)[:2],
        geometry=g.geometry.union_all(),
        basename=field("basename"),
    )


# ==========================================================================
# Headless driver
# ==========================================================================


def run_headless(
    cfg: RunConfig,
    resume_from: Optional[str] = None,
    progress: Progress = print,
) -> Path:
    """Prepare + optimize with no GUI. Returns the output parquet path."""
    run_dir = Path(resume_from) if resume_from else None
    prep = prepare(cfg, run_dir=run_dir, progress=progress)
    opt = prep.optimizer

    latest = prep.store.latest()
    if latest is not None:
        opt.load_checkpoint(latest)
    else:
        opt.consolidate_mixed_tiles()

    def on_stats(s: engine.OptimizerStats) -> None:
        if s.iteration % 1000 == 0:
            progress(
                f"iter {s.iteration:,} T={s.temperature:.5f} "
                f"score={s.mean_score:.6f} boundary={s.boundary_size:,} "
                f"acc={s.accepted:,}"
            )

    opt.run(store=prep.store, stats_cb=on_stats)

    out = prep.run_dir / "optimized_neighborhoods_tiled.parquet"
    opt.result_frame(prep.parcels).to_parquet(out)
    progress(f"Wrote {out}")
    return out
