"""Configuration objects and defaults for the desktop optimizer.

Defaults mirror the constants at the top of ``main_tiled.ipynb`` so a desktop
run reproduces the notebook's behaviour when pointed at the same parcel file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

# --------------------------------------------------------------------------
# Notebook-equivalent constants
# --------------------------------------------------------------------------

MAX_BINS = 30
ADJACENCY_THRESHOLD_FT = 100.0
GRID_SIZE_FT = 500.0

CONTINUOUS_VARIABLES: List[str] = [
    "bldg_area_finished_sqft",
    "land_area_sqft",
    "bldg_age_years",
    "dist_to_open_water",
    "assr_land_ppsf",
    "assr_impr_ppsf",
]

SEED_NEIGHBORHOOD_FIELDS: List[str] = [
    "latitude",
    "longitude",
    "dist_to_open_water",
]

DEFAULT_WEIGHTS: Dict[str, float] = {
    "bldg_age_years_binned": 0.10,
    "bldg_area_finished_sqft_binned": 0.25,
    "land_area_sqft_binned": 0.25,
    "dist_to_open_water_binned": 0.50,
    "assr_land_ppsf_binned": 1.50,
    "assr_impr_ppsf_binned": 1.50,
}

# Derived columns the notebook computes from assr_market_value. Each entry is
# (new_column, numerator, denominator); applied only when all inputs exist.
DERIVED_RATIOS = [
    ("assr_impr_ppsf", "assr_market_value", "bldg_area_finished_sqft"),
    ("assr_land_ppsf", "assr_market_value", "land_area_sqft"),
]


# --------------------------------------------------------------------------
# Run configuration
# --------------------------------------------------------------------------


@dataclass
class RunConfig:
    """Everything needed to reproduce (or resume) a run."""

    # --- study area -------------------------------------------------------
    jurisdiction_query: str = ""
    jurisdiction_geoid: str = ""
    jurisdiction_name: str = ""
    jurisdiction_layer: str = ""

    # --- parcels ----------------------------------------------------------
    parcel_path: str = ""
    parcel_filter_column: str = "model_group"
    parcel_filter_value: str = "single_family"

    # --- tiling -----------------------------------------------------------
    grid_size_ft: float = GRID_SIZE_FT
    use_census_blocks: bool = True
    # "tiger" downloads one bulk per-state block shapefile (no feature cap, so
    # it scales to any county); "rest" uses the TIGERweb API, which truncates
    # above 100k features but avoids a large download for a single small city.
    census_source: str = "tiger"
    use_osm_roads: bool = True
    # Waterway centrelines (rivers, streams, canals, ditches) as cut lines. This
    # had no toggle at all and fetched unconditionally, so turning roads and water
    # off still fired an Overpass query -- and still failed on any jurisdiction
    # Overpass refused.
    use_osm_waterways: bool = True
    clip_water: bool = True
    adjacency_threshold_ft: float = ADJACENCY_THRESHOLD_FT

    # --- adjacency ---------------------------------------------------------
    # "tile"   : tiles adjacent if their geometry is within the threshold. The
    #            original rule. Cheap, but a tile whose parcels sit far from the
    #            shared edge still counts as adjacent.
    # "parcel" : tiles adjacent if any of their *parcels* are adjacent, with an
    #            optional line-of-sight test that drops pairs with someone else's
    #            lot in between. See adjacency.py.
    adjacency_mode: str = "tile"
    require_line_of_sight: bool = True
    # What blocks a sightline: "modeled" (only the parcels being optimized, so
    # farmland and vacant land are transparent), "all" (every parcel), or
    # "all_except" (every parcel bar the in-gap classes below).
    obstacle_mode: str = "modeled"
    transparent_land_class_keywords: List[str] = field(default_factory=list)
    land_class_column: str = "land_class"
    # Endpoint shrink for the sightline segment. Results are insensitive to it
    # (0.1 ft and 0.5 ft differ by 0.1% on Lee County); it only has to exceed
    # floating-point noise.
    sightline_epsilon_ft: float = 0.5

    # --- scoring ----------------------------------------------------------
    max_bins: int = MAX_BINS
    continuous_variables: List[str] = field(
        default_factory=lambda: list(CONTINUOUS_VARIABLES)
    )
    seed_fields: List[str] = field(
        default_factory=lambda: list(SEED_NEIGHBORHOOD_FIELDS)
    )
    weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    # The notebook's MI loop skips bins the neighborhood has zero parcels in,
    # which drops those bins' "outside" contribution. Leave False to reproduce
    # the notebook exactly; set True for textbook MI.
    exact_mi: bool = False

    # --- optimizer --------------------------------------------------------
    n_neighborhoods: int = 100
    initial_temp: float = 1.0
    cooling_rate: float = 0.99
    max_iterations: int = 1_000_000
    # Backstop: consecutive rejected batches at T~0 before giving up.
    max_stability: int = 1000

    # --- assignment-stability convergence ---------------------------------
    # A rejected-move count is a poor convergence signal: it means different
    # things on a 500-edge boundary than on a 20,000-edge one, and it never
    # fires while simulated annealing keeps accepting marginal moves that don't
    # actually change the map. Tracking how many parcels changed label recently
    # is scale-free and measures the thing we care about directly.
    #
    # The measure is *novelty*, not raw volume: over the last
    # `assignment_stability_iters` iterations, how many distinct parcels changed
    # label, divided by how many relabel events occurred? Every move touching
    # fresh ground gives ~1.0; the same few tiles flipping back and forth drives
    # it toward 0, which is what convergence actually looks like under annealing.
    #
    # Do NOT normalise by total parcels. That was the first attempt and it is
    # backwards: a move relabels a fixed handful of parcels regardless of dataset
    # size, so the achievable fraction *shrinks* as the data grows. At Lee County
    # scale (276k parcels, 5 per tile) a 500-iteration window can never exceed
    # ~1.8% even if every single iteration accepts -- so a 1% threshold fired
    # during healthy operation and stopped the run near its KMeans seeding. Small
    # test fixtures sit in a regime where the same threshold looks fine, which is
    # why it took a county-sized run to surface.
    assignment_stability_iters: int = 500
    assignment_progress_ratio: float = 0.20
    assignment_stability_streak: int = 5

    # --- contiguity -------------------------------------------------------
    # Reject moves that would newly disconnect a neighborhood in the tile
    # adjacency graph. Pre-existing disconnection is preserved, not repaired --
    # the gate only prevents things getting worse. Changes results relative to
    # runs made without it.
    enforce_contiguity: bool = True
    batch_divisor: int = 10  # subsample = len(boundary) // batch_divisor
    max_batch: int = 256  # ...but never evaluate more than this many edges/iter
    min_batch: int = 8
    random_seed: int = 42

    # --- decomposition ----------------------------------------------------
    # KMeans seeds on position without knowing tile adjacency, so it happily
    # puts one neighborhood on both banks of a river. Those can never be
    # separated by trading (there is no boundary between severed components),
    # so they are split apart before optimizing. Turn off only to reproduce a
    # pre-existing run.
    split_severed_neighborhoods: bool = True
    # Annealing processes. 1 = serial, in-process. 0 = auto, clamped to the
    # speedup actually available from the component structure.
    workers: int = 1

    # --- runtime ----------------------------------------------------------
    refresh_every: int = 1000  # repaint the map every N iterations
    checkpoint_every: int = 5000  # write a checkpoint every N iterations
    keep_checkpoints: int = 20  # 0 = keep all
    work_dir: str = "runs"

    # ------------------------------------------------------------------
    def crs_hint(self) -> Optional[int]:
        """Optional EPSG override for the working (feet-based) CRS."""
        return None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    def save(self, path: Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "RunConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------


def repo_root() -> Path:
    # This module sits at the top of the project, so the project root is its own
    # directory. (It used to be one level deeper, hence the previous .parent.parent
    # -- which after flattening pointed at the parent of the project.)
    return Path(__file__).resolve().parent


def cache_dir() -> Path:
    d = repo_root() / ".mi_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def runs_dir(cfg: RunConfig) -> Path:
    d = repo_root() / cfg.work_dir
    d.mkdir(parents=True, exist_ok=True)
    return d
