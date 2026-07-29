"""What fields a parcel file offers, and whether a config actually fits it.

Two problems this solves.

**Typing field names blind.** The UI used free text for the filter column, its
value, and every scored field. A typo is indistinguishable from a legitimate
column name until something downstream fails.

**Failing late.** Field validation used to live in ``pipeline.load_parcels``,
which runs at step 4 of ``prepare()`` -- *after* the census download and the
shatter. So a misspelled weight cost a full download plus tiling before it
surfaced. Introspection is cheap enough to do the moment a file is chosen:
schema is ~11 ms on a 158 MB / 126-column parquet, and pulling a whole column for
its distinct values is ~75-130 ms.

The subtlety is that scored fields are **not** file columns. ``bldg_area_finished_sqft_binned``
is created later by ``bin_continuous_fields``, and ``assr_impr_ppsf`` is derived
from ``assr_market_value``. So the useful question is not "what is in the file"
but "what will exist by the time scoring happens", which is what
``FieldUniverse`` computes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from config import DERIVED_RATIOS, RunConfig

Progress = Callable[[str], None]


def _noop(_: str) -> None:
    pass


BINNED_SUFFIX = "_binned"

#: Never offered as a scoring/seeding field.
EXCLUDED_COLUMNS = frozenset({"geometry", "tile_id", "neighborhood_id"})


def source_name(scored: str) -> str:
    """``bldg_age_years_binned`` -> ``bldg_age_years``."""
    return scored[: -len(BINNED_SUFFIX)] if scored.endswith(BINNED_SUFFIX) else scored


def scored_name(source: str) -> str:
    """``bldg_age_years`` -> ``bldg_age_years_binned`` (idempotent)."""
    return source if source.endswith(BINNED_SUFFIX) else source + BINNED_SUFFIX


# ==========================================================================
# Problems
# ==========================================================================


@dataclass(frozen=True)
class Problem:
    setting: str      # which config field is at fault
    value: str        # the offending name
    detail: str
    fatal: bool = True

    def __str__(self) -> str:
        kind = "ERROR" if self.fatal else "warning"
        return f"{kind}: {self.setting} = {self.value!r} -- {self.detail}"


# ==========================================================================
# Universe
# ==========================================================================


@dataclass
class FieldUniverse:
    """The columns a parcel file has, and the ones it will have after prep."""

    path: Path
    columns: List[str] = field(default_factory=list)
    numeric: List[str] = field(default_factory=list)
    text: List[str] = field(default_factory=list)
    derived: List[str] = field(default_factory=list)   # ratios we can compute
    prebinned: List[str] = field(default_factory=list)  # *_binned already present
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.columns)

    @property
    def binnable(self) -> List[str]:
        """Columns that can be binned, so eligible for scoring or seeding."""
        return sorted(set(self.numeric) | set(self.derived))

    @property
    def scoreable(self) -> List[str]:
        """Source names whose binned form will exist by scoring time."""
        return sorted(set(self.binnable) | {source_name(c) for c in self.prebinned})

    @property
    def seedable(self) -> List[str]:
        return self.binnable

    def has(self, name: str) -> bool:
        return name in self.columns or name in self.derived

    def summary(self) -> str:
        if not self.ok:
            return f"could not read fields: {self.error}"
        return (
            f"{len(self.columns)} columns ({len(self.numeric)} numeric, "
            f"{len(self.text)} text), {len(self.derived)} derivable, "
            f"{len(self.prebinned)} already binned"
        )


def inspect_parcel_file(
    path: str | Path, progress: Progress = _noop
) -> FieldUniverse:
    """Read just enough of a parcel file to know what it offers."""
    path = Path(path)
    uni = FieldUniverse(path=path)
    if not path.exists():
        uni.error = "file not found"
        return uni

    try:
        if path.suffix.lower() in (".parquet", ".pq"):
            import pyarrow as pa
            import pyarrow.parquet as pq

            schema = pq.ParquetFile(path).schema_arrow
            for name, dtype in zip(schema.names, schema.types):
                if name in EXCLUDED_COLUMNS:
                    continue
                uni.columns.append(name)
                if pa.types.is_integer(dtype) or pa.types.is_floating(dtype):
                    uni.numeric.append(name)
                elif (pa.types.is_string(dtype) or pa.types.is_large_string(dtype)
                      or pa.types.is_dictionary(dtype) or pa.types.is_boolean(dtype)):
                    uni.text.append(name)
        else:
            import geopandas as gpd

            head = gpd.read_file(path, rows=200)
            for name in head.columns:
                if name in EXCLUDED_COLUMNS:
                    continue
                uni.columns.append(name)
                if pd.api.types.is_numeric_dtype(head[name]):
                    uni.numeric.append(name)
                else:
                    uni.text.append(name)
    except Exception as exc:  # noqa: BLE001
        uni.error = str(exc)
        return uni

    present = set(uni.columns)
    # Ratios pipeline.add_derived_columns will manufacture, given their inputs.
    for name, num, den in DERIVED_RATIOS:
        if name not in present and num in present and den in present:
            uni.derived.append(name)
    uni.prebinned = [c for c in uni.columns if c.endswith(BINNED_SUFFIX)]

    progress(f"{path.name}: {uni.summary()}")
    return uni


def distinct_values(
    path: str | Path,
    column: str,
    limit: int = 200,
    progress: Progress = _noop,
) -> List[str]:
    """Distinct values of one column, for the filter-value picker.

    Columnar formats make this cheap -- ~75-130 ms for a full column of 558k
    rows -- so it can run whenever the chosen column changes.
    """
    path = Path(path)
    if not path.exists() or not column:
        return []
    try:
        if path.suffix.lower() in (".parquet", ".pq"):
            series = pd.read_parquet(path, columns=[column])[column]
        else:
            import geopandas as gpd

            series = gpd.read_file(path, columns=[column])[column]
    except Exception as exc:  # noqa: BLE001
        progress(f"Could not list values of {column!r}: {exc}")
        return []

    counts = series.astype(str).value_counts()
    values = [str(v) for v in counts.index[:limit]]
    progress(
        f"{column}: {counts.size} distinct value(s)"
        + (f", showing the {limit} most common" if counts.size > limit else "")
    )
    return values


# ==========================================================================
# Validation
# ==========================================================================


def validate(
    cfg: RunConfig, uni: FieldUniverse, filter_values: Optional[Sequence[str]] = None
) -> List[Problem]:
    """Every way the configured fields can fail against this file.

    Returned rather than raised, so the UI can flag each setting individually and
    keep the user's value on screen instead of discarding it.
    """
    problems: List[Problem] = []
    if not uni.ok:
        return [Problem("parcel_path", str(uni.path), uni.error or "unreadable")]

    present = set(uni.columns)

    if cfg.parcel_filter_column:
        if cfg.parcel_filter_column not in present:
            problems.append(Problem(
                "parcel_filter_column", cfg.parcel_filter_column,
                "no such column in the parcel file",
            ))
        elif filter_values is not None and cfg.parcel_filter_value and \
                cfg.parcel_filter_value not in filter_values:
            problems.append(Problem(
                "parcel_filter_value", cfg.parcel_filter_value,
                f"never occurs in {cfg.parcel_filter_column}; the filter would "
                "match zero parcels",
            ))

    # Only consulted when sightlines block on land class.
    if cfg.adjacency_mode == "parcel" and cfg.obstacle_mode == "all_except":
        if cfg.land_class_column not in present:
            problems.append(Problem(
                "land_class_column", cfg.land_class_column,
                "no such column, so 'all except in-gap classes' will behave "
                "like 'all parcels'", fatal=False,
            ))

    binnable = set(uni.binnable)
    for name in cfg.continuous_variables:
        if name not in binnable:
            problems.append(Problem(
                "continuous_variables", name,
                "not a numeric or derivable column, so it cannot be binned",
            ))
    for name in cfg.seed_fields:
        if name not in binnable:
            problems.append(Problem(
                "seed_fields", name, "not a numeric or derivable column",
            ))

    # A weight is only meaningful if its binned column will exist: either the
    # source is being binned, or the file already carries the binned form. This
    # cross-check is the one that free-text entry made easy to get wrong.
    will_exist = {scored_name(c) for c in cfg.continuous_variables}
    will_exist |= set(uni.prebinned)
    for scored, weight in cfg.weights.items():
        if scored in will_exist:
            continue
        src = source_name(scored)
        if src in binnable:
            problems.append(Problem(
                "weights", scored,
                f"{src!r} exists but is not in the binning list, so "
                f"{scored!r} is never created",
            ))
        else:
            problems.append(Problem(
                "weights", scored, "no such column, and it cannot be derived",
            ))

    if not cfg.weights:
        problems.append(Problem(
            "weights", "(empty)", "nothing is being scored", fatal=True,
        ))
    return problems


def fatal_problems(problems: Sequence[Problem]) -> List[Problem]:
    return [p for p in problems if p.fatal]


def describe(problems: Sequence[Problem]) -> str:
    if not problems:
        return "All configured fields are present in the parcel file."
    lines = [str(p) for p in problems]
    n_fatal = len(fatal_problems(problems))
    head = (f"{n_fatal} problem(s) would stop the run"
            if n_fatal else "No blocking problems")
    return head + ":\n  " + "\n  ".join(lines)


def default_weight_for(source: str, defaults: Dict[str, float]) -> float:
    """Look a source name up in a ``*_binned``-keyed defaults dict."""
    return float(defaults.get(scored_name(source), 0.0))
