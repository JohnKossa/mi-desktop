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
    clip_water: bool = True
    adjacency_threshold_ft: float = ADJACENCY_THRESHOLD_FT

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
    max_stability: int = 1000
    # A fixed miss count means different things at different scales. 1,000 misses
    # against a 500-edge boundary is ~100 full sweeps of it; against a
    # 20,000-edge boundary it is barely 12. When components are annealed
    # separately those subproblems differ by orders of magnitude, so a flat
    # threshold makes "converged" mean something different in each one.
    #
    # Set > 0 to express the limit as "this many complete sweeps of my own
    # boundary found nothing", which is scale-invariant. 0 keeps the original
    # fixed-count behaviour exactly. Recommended for parallel runs.
    stability_sweeps: float = 0.0
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
