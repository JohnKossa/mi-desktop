# mi_desktop

Qt app that runs the tiled MI boundary-swapping algorithm end to end: name a
jurisdiction, point it at a parcel file, and it fetches the geography, builds the
tileset, and anneals — showing the map live and checkpointing as it goes.

```
uv sync
python app.py                     # GUI
python tests/test_desktop.py      # offline, ~30s
python tests/test_sources.py      # offline, ~5s
python tests/test_parallel.py     # offline, spawns real processes, ~30s
python tests/test_contiguity.py   # offline, ~25s
python tests/test_water_clip.py   # offline, ~10s
python tests/test_shutdown.py     # offline, ~5s
```

---

## Pipeline

1. **Jurisdiction.** Free text (`Lee County, FL`, `Chicago Illinois`, `Cape Coral`).
   TIGERweb is searched across Incorporated Places, CDPs, County Subdivisions and
   Counties; matches are ranked best-guess first and you pick one. Its boundary
   is the study area, and everything is clipped to it.
2. **Working CRS.** A State Plane zone in US survey feet, chosen automatically
   (Lee County → EPSG:2237). Falls back to a locally-centred transverse Mercator
   in feet where no zone fits. The adjacency threshold is in feet, so this
   matters.
3. **Census blocks.** One bulk per-state TIGER/Line shapefile, cached on disk,
   read through `/vsizip/` with a bbox prefilter. See below.
4. **OSM roads, waterways, water.** Overpass, subdivided adaptively. See below.
5. **Tiles.** Block boundaries + road centrelines + waterway centrelines + a
   500 ft grid + the jurisdiction outline are noded into one linework graph and
   polygonized. Faces outside the boundary are dropped, water is clipped out,
   slivers under 100 sq ft discarded.
6. **Parcels.** Filtered (`model_group == single_family` by default),
   reprojected, clipped to the boundary, given derived `assr_impr_ppsf` /
   `assr_land_ppsf`, binned, joined to tiles on representative point. Parcels
   landing in no tile become single-parcel "virtual" tiles.
7. **Optimize.** KMeans seed → MI consolidation pass → tile-level simulated
   annealing with donations and swaps. The map repaints every N iterations, a
   checkpoint lands every M.

---

## Scaling to arbitrary counties

Three things had to change to go from "one city" to "any US county".

**Jurisdiction lookup.** TIGER stores `BASENAME` without the legal/statistical
suffix and `NAME` with it: the county is `BASENAME="Lee"`, `NAME="Lee County"`.
The original search only matched `BASENAME LIKE 'Lee County%'`, so typing
"Lee County, FL" returned *nothing*. Now both columns are matched, against both
the query as typed and the query with its LSAD suffix stripped — so
`Lee County, FL`, `Lee, FL`, `East Baton Rouge Parish, LA` and
`Matanuska-Susitna Borough, AK` all resolve. Suffixes are stripped longest-first,
so `Iosco charter township` becomes `Iosco` rather than `Iosco charter`. The
county layer is also searched unconditionally — it used to be possible for
like-named places to fill the result limit and short-circuit it.

**Bulk census blocks (`tiger.py`).** TIGERweb caps a single `returnIdsOnly`
response at 100,000 features and serves geometry 1,000 at a time. Lee County
(~30k blocks) squeaks under; Cook County does not, and would silently truncate.
So blocks now come from the source Census publishes for bulk use:

```
https://www2.census.gov/geo/tiger/TIGER<year>/TABBLOCK20/tl_<year>_<ss>_tabblock20.zip
```

One download per state, cached indefinitely, read via GDAL `/vsizip/` with a
bounding-box prefilter and `columns=[]` (blocks are only cut lines here, so their
attributes are dead weight). No caps, no pagination, and every later jurisdiction
in that state is fully offline. The vintage year is **probed at runtime**,
newest-first, and cached — a new Census release needs no code change. The REST
fetcher survives as `census_source = "rest"` and as an automatic fallback.

**Adaptive Overpass (`sources._fetch_osm_tiled`).** A single county-wide road
query blows Overpass's time and memory budget. Query cost tracks feature density,
not area, so rather than guess a cell size the whole area is tried first and split
into quadrants only where the server pushes back — rural counties resolve in one
request, dense metros subdivide until each piece fits (up to
`MAX_OVERPASS_DEPTH = 4`, i.e. 256 cells). Each leaf is cached separately, so a
download interrupted three quarters through resumes rather than restarting.
Requests stay sequential with a courtesy pause, because the endpoints publish
fair-use limits.

The subtle part: an over-budget Overpass query returns **HTTP 200 with partial
data** and an explanation in a `remark` field. That's checked explicitly —
otherwise the study area quietly loses roads.

### Measured at full Lee County scale

276,617 single-family parcels over a 44 × 33 mile extent, 500 ft grid, 30k road
segments, 400 seed neighborhoods:

| stage | time |
|---|---|
| shatter → 380,024 tiles | 11.5 s |
| parcel → tile join | 0.5 s |
| adjacency (55,304 populated tiles, 180,319 edges) | 0.4 s |
| KMeans seeding | 4.7 s |
| optimizer init (153 flat bins) | 1.0 s |
| consolidation pass | 0.6 s |
| annealing | 48.5 it/s |
| map redraw (55,202 paths) | 60 ms |

Two things make this hold up. Only *populated* tiles are drawn — 55k of 380k —
so the live map stays cheap. And the candidate batch is capped at `max_batch`
(256) rather than `len(boundary) // 10`, so per-iteration cost is roughly
independent of study-area size; a bigger county means more iterations to
converge, not slower ones.

Cook County should be broadly similar in area but several times denser, so expect
the shatter and the render to scale up by that factor and the iteration rate to
hold. The untested risks there are peak memory during `polygonize` and the
initial ~100 MB Illinois download.

---

## Water clipping: select before you join

Clipping water out of the *finished* tiles is the obvious way round and it is a
trap. `base.geometry.difference(mask.union_all())` is an elementwise difference
against one giant MultiPolygon, so every tile pays the cost of all the water in
the county — and Cape Coral's canal network alone runs to thousands of polygons.
Measured on the Lee County tileset with 4,800 water polygons (4,556 union parts,
4.3 MB of WKB):

| | |
|---|---|
| difference, 380,024 tiles vs whole union | **~11–12 min** |
| subtract water from the study area first | **0.6 s** (union 0.5 s + one difference 0.1 s) |
| prepared study-area trim, 380,024 tiles | 0.2 s |
| whole `build_tileset`, 30k roads + water, 393,481 tiles | 18.9 s |

So water is now removed from the study-area polygon *before* the shatter. Two
things fall out for free: the water outlines join the linework via the area
boundary, so tiles stop at the shoreline by construction, and the existing
jurisdiction trim drops water faces without a separate pass.

The trim needs `shapely.prepare()` on the water-perforated area polygon — without
it, each of the ~380k point-in-polygon tests walks every ring, which is the same
quadratic shape as the bug being fixed (0.2 s prepared vs ~6 s not).

`clip_out` is retained for clipping an already-built tileset, and rewritten to
index the mask and touch only the tiles that actually meet it.
`test_clip_out_helper_matches_the_naive_union_difference` pins it against the
slow one-liner; `test_clipping_is_not_quadratic_in_tile_count` fails if per-tile
clipping ever returns.

---

## Parallel annealing over severed components

Adjacency is only computed between tiles that hold parcels, and a real county is
nothing like one connected blob. Lee County splits into **1,012 connected
components**, the largest holding 27.9% of tiles and 23.0% of parcels — barrier
islands, the Caloosahatchee, and undeveloped stretches sever the graph outright.
Two moves in different components cannot interact, so they can be optimized
concurrently.

**The precondition: split severed neighborhoods.** KMeans seeds on
lat/long/distance-to-water with no notion of adjacency, so it happily puts one
neighborhood on both banks of a river. On Lee County, **211 of 400** seeded
neighborhoods straddle a severance. Each is permanently unfixable — trades only
happen along boundaries and there is no boundary between components — so those
parcels stay welded together however long it anneals. `partition.split_neighborhoods`
gives each (neighborhood, component) pair its own id. Worth doing on its own
merits; it also happens to be what makes the count-table rows disjoint per
worker. Controlled by `split_severed_neighborhoods` (default on).

**Processes, not threads.** The inner loop is NumPy on ~150-element arrays plus
Python set work on the boundary — squarely GIL-bound. Processes also give a
cleaner story: after splitting there is no shared mutable state. Workers hold the
*full* parcel arrays (a few MB) rather than a slice, which keeps bin codes,
global counts and neighborhood numbering identical to serial by construction —
slicing the frame first would make `build_count_tables` re-derive
subset-dependent bin codes and silently change the objective. The spawn start
method is used on every platform, since forking a process with Qt loaded is
unsafe and spawn is what Windows does anyway.

### Measured on Lee County (276,617 parcels, 400 seed neighborhoods)

| | |
|---|---|
| components | 1,012 |
| largest component | 63,603 parcels (23.0%) |
| structural ceiling | 4.35x |
| 4-worker load balance | 69,155 / 69,154 / 69,154 / 69,154 parcels |
| serial | 48.3 it/s |
| 2 workers, steady state | 108.6 it/s aggregate = **2.25x** |
| per-worker rate | 54.3 it/s (slightly above serial: smaller working set) |
| worker startup | 3.2 s fixed (spawn + sklearn import + parquet read) |

**My sandbox has only 2 cores**, so 2 workers is all I could measure honestly —
a 4-worker run came out at 1.50x purely from oversubscription, not from any
design problem. On ≥4 cores the structure allows ~4x, but I have not verified
that. Startup is a fixed 3.2 s, which was 37% of wall time on a deliberately
tiny 300-iteration test and is noise on a real run.

### Two ways parallel output differs from serial

Neither is a bug, but neither is ignorable.

**1. Different Markov chain.** Serial samples one global batch and commits the
single best move across the whole county. Parallel commits the best move *per
component*. A one-group parallel run is bit-identical to serial
(`test_single_group_parallel_matches_serial_exactly`); a multi-group run is not.
Arguably parallel is better: with a 256-edge batch drawn from 20,000+ boundary
edges, a 30-tile island is sampled roughly never, so small components are
starved today.

**2. The convergence test is scale-dependent.** The optimizer stops after
`max_stability` consecutive misses at low temperature. That flat count means very
different things at different scales — 1,000 misses against a 500-edge boundary
is ~100 full sweeps of it, but against a 20,000-edge boundary it is barely 12. So
identical thresholds stop large and small subproblems at very different depths,
and a smaller group is sampled far more thoroughly per iteration, making each
miss a stronger signal. There is also an effect pulling the other way: serial
shares *one* counter across all 1,012 components, so it is constantly reset by
activity in unrelated parts of the county and effectively never converges,
whereas per-group counters can.

`stability_sweeps` expresses the limit as "this many complete sweeps of my own
boundary found nothing", which is scale-invariant. It defaults to `0.0`
(the original fixed-count behaviour, exactly) — **recommended to set for
parallel runs**, e.g. `4.0`.

### The score readout had to change

Splitting takes Lee County from 400 to 1,518 neighborhoods, 477 of them
singletons. Singletons score 1.0 and empties 2.0 by design (`SPEC.md`,
absorption), so the plain mean score jumps from **0.0047 to 0.3155** — a 67x
inflation with no change in actual quality. The parcel-weighted mean moves
0.0056 → 0.0070, a real but modest improvement.

The status line therefore reports the **parcel-weighted** score. Workers send
numerator and denominator separately so the parent forms the exact global figure
rather than an average of averages. Scores from before this change are not
comparable.

---

## Contiguity gate

Adapted from the `kossa_neighborhoods` approach, with one change of position.

Before a move is committed, `_removal_disconnects` asks whether taking a tile out
of a neighborhood would split what remains — a local articulation test, since
every same-neighborhood tile adjacent to the departing one was reachable through
it. `_addition_creates_island` covers the swap-only case where the arriving
tile's sole anchor leaves in the same move; donations always travel along a
boundary edge, so their recipient is adjacent by construction.

**The gate runs after ranking, not per candidate.** Filtering every candidate and
then taking the best is identical to taking the best that passes, but only the
second form can stop checking once it finds a legal move. At the measured 5%
block rate that's ~1.05 checks per iteration instead of ~1,500:

| | Lee County |
|---|---|
| no gate | 47.1 it/s |
| gate after ranking (this) | **47.5 it/s** |
| gate per candidate (as ported) | ~25.9 it/s (18 ms/iteration) |

So it's free rather than costing half the throughput. `test_ranked_gating_equals_prefiltering`
asserts the two orderings pick the same move.

**It prevents, it does not repair.** Pre-existing disconnection is preserved
deliberately, and there's a lot of it: on Lee County, 753 of 1,518 neighborhoods
are *already* disconnected after seeding — KMeans fragments neighborhoods
*within* a component, which `partition.split_neighborhoods` doesn't touch because
it only splits *across* components. If you want those repaired too, that's a
seed-time split on the neighborhood's own component structure; say the word.

Effect, on a fixture with enough tiles per neighborhood for fragmentation to be
reachable (excess components = how many neighborhoods are in more pieces than
one):

```
gate on : 6 -> 3     disconnected  6 -> 3
gate off: 6 -> 76    disconnected  6 -> 44
```

`enforce_contiguity` defaults on, which **changes results** relative to earlier
runs.

## Assignment-stability convergence

The old rule — N consecutive rejected batches at low temperature — never fires
while annealing keeps accepting marginal moves that don't move the map. The
replacement asks the direct question, but **how you normalise it matters more
than it looks**.

### The regression, and the fix

The first version measured *fraction of parcels relabelled* in the last window
and stopped below 1%. That is backwards. A move relabels a fixed handful of
parcels no matter how large the dataset is, so the achievable fraction *shrinks*
as parcels grow:

| dataset | ceiling on "% of parcels changed", 500-iteration window |
|---|---|
| 5,000 parcels (test fixture) | ~63% — 1% threshold is meaningful |
| 276,617 parcels (Lee County) | **~1.8%** — 1% threshold is inside the noise |

At county scale the statistic could not exceed ~1.8% even if *every* iteration
accepted a move; measured on a real run it sat at 1.44–1.85% and was declining.
So the rule fired whenever the accept rate merely dipped, stopping the run a
couple of thousand iterations in and leaving boundaries near their KMeans
seeding. Every test passed because the fixtures live in the regime where the
threshold works.

The metric is now **novelty**: distinct parcels changed in the window, divided by
the number of relabel *events* in the window. Both terms scale together, so the
ratio doesn't. ~1.0 while the optimizer keeps reaching fresh ground; toward 0 once
it is shuffling the same tiles between the same neighborhoods, which is what
convergence looks like under annealing. Measured over a Lee County run it held at
**0.987–0.998**, so the 0.20 default has a 5x margin.

`test_progress_ratio_does_not_depend_on_dataset_size` and
`test_fraction_of_parcels_has_a_size_dependent_ceiling` pin the property that was
missing.

### The array must be checkpointed, and omitting it fails unsafely

Resuming at iteration 50,000 with a zeroed `last_change_iter` makes every parcel
look 50,000 iterations stale, so the window reads empty and the run declares
convergence and exits the moment it resumes. Not a delay, a wrong answer. It is
stored in the checkpoint `.npz`, and `Checkpoint.restore_last_change` fills a
missing array with the checkpoint's iteration ("everything just changed") rather
than zeros. Parallel workers do the same on resume, since per-group state carries
scalars only. Checkpoints grow roughly 2x.

### What the gate costs

Measured over 700 iterations on Lee County, gated vs not:

| | |
|---|---|
| parcels moved | 98.5% as many — the gate is *not* throttling swap volume |
| score improvement captured | **74.6%** of ungated |
| candidates blocked | 0.196% of all candidates |
| iterations where the **top-ranked** move was blocked | **55.9%** |
| mean gain: blocked #1 vs substitute taken | 1.62e-4 vs 8.93e-5 |

So the blocked candidates are disproportionately the *best* ones, and the
substitute carries about half the gain. That is the real price of contiguity on
this data, and it is worth knowing before assuming a bad map is the gate's fault.
Note also that gate rejections cannot inflate the `max_stability` counter: when
every candidate is blocked, `_best_move` returns `None` and the loop `continue`s
before that counter is touched (`blocked_batches` records it separately, and
measured 0 on Lee County).

## What was deliberately left out

From `kossa_neighborhoods`, after measuring:

- **Empty-tile bridging** — connecting neighborhoods through parcel-free tiles.
  On this tileset it collapses 1,012 components to **1**, taking the parallel
  ceiling from 4.35x to 1.00x. The two features are in definitional opposition:
  the components exist precisely because adjacency is restricted to populated
  tiles.
- **Reactive exclave splitting** — it creates tiny neighborhoods, which makes the
  `SCORE_EMPTY = 2.0` absorption bonus exploitable (donate a one-parcel exclave,
  donor empties, +2 dominates every real gain, splitter fragments the recipient,
  repeat). Adopting it forces replacing the objective with HHI + shrinkage, which
  would break comparability with existing runs. The gate already guarantees
  topological contiguity; the splitter only adds the stricter parcel-proximity
  notion.

---

## Closing the app

Warning someone off an action is fine; refusing to let them do it is not. The
"still preparing" dialog used to have a single OK button and then called
`event.ignore()` unconditionally — there was no way out of the app short of a
kill signal, which is a worse trap than the slow operation it was guarding.

Every shutdown dialog now has **Exit Anyway**, with **Keep Running** as the
default and escape action. Mid-optimization there is a third option, **Stop &
Save**, which lands a final checkpoint first so the run can resume exactly where
it left off; Exit Anyway abandons progress since the last checkpoint. If a
graceful stop doesn't complete in 30 s the dialog comes back offering to wait
longer or leave — it never silently traps.

Force-quit goes through `os._exit(0)` rather than `event.accept()`. Destroying
the window while its QThread children are running aborts the process, which on
Windows means a crash dialog; `_exit` is the quiet door. Two things make that
safe rather than reckless:

- **Worker processes are killed first.** `os._exit` skips atexit handlers, so
  spawned annealing children would otherwise outlive the parent and keep burning
  CPU with no window attached.
- **Every cache write is atomic** (`pipeline._atomic_write`, `CheckpointStore.save`):
  write to a temp name, then rename. A kill can therefore never leave a
  half-written `tiles.parquet` that the next run would pick up as valid and choke
  on. Damaged caches are also treated as "not cached" rather than fatal, so an
  older corrupt file just triggers a rebuild.

Two traps worth remembering, both caught by tests rather than by reading:

- `np.savez_compressed` **silently appends `.npz`** to any filename that lacks
  it. A temp name of `x.npz.part` gets written as `x.npz.part.npz`, so the rename
  targets a file that was never created. The first version of the atomic-write
  helper had exactly this bug and would have broken every checkpoint write.
  Temp names now keep the real suffix last.
- Checkpoint temp files are named `_writing_*` rather than `checkpoint_*.part`,
  because `list()` and `_prune()` glob `checkpoint_*` and would otherwise pick up
  a half-written file as a real checkpoint.

The blocking `QThread.wait(60000)` is gone too — it froze the window for up to a
minute, which is indistinguishable from the hang the user was trying to escape.
`_wait_for_stop` waits in 100 ms slices and pumps the event loop.

---

## Benchmark correction

An earlier note in this file claimed the flat-bin refactor made things "6x faster
than the notebook", with a `notebook | desktop` table next to the figure. That
was wrong: **15.8 → 97.7 it/s was measured against the first version of the
refactor, not against `main_tiled.ipynb`**, which was never benchmarked. The
notebook is very likely slower still — it looped over bins in Python and rebuilt
the boundary list on every accepted move — but that factor is unmeasured.

What the 6x *is* attributable to, on a 26k-parcel slice:

- **One count vector instead of six.** `p_in`/`p_out` depend only on neighborhood
  size, not on which field is being scored, so the whole weighted MI sum
  collapses to a single dot product over ~150 flat bins instead of one NumPy pass
  per field.
- **Cached logs.** Expanding `p·log(p/(p_in·p_val))` in raw counts leaves
  `log c_total` and `log G` constant across calls, so only two elementwise logs
  remain per candidate.
- **Scratch buffer.** Candidates are scored into one reusable array; the winning
  delta is applied in place on accept, instead of allocating fresh count rows for
  every candidate.

Verified equivalent: the synthetic test lands on score 1.217697 with identical
assignment vectors before and after the rewrite.

---

## Pausing, stopping, resuming

Each run gets a directory under `runs/`:

```
runs/lee_county_fl_county_20260728_.../
    run_config.json                     every parameter used
    jurisdiction.parquet                boundary + geoid + state_fips + layer
    tiles.parquet                       the shattered tileset
    parcels_prepared.parquet            filtered, binned, tile-joined parcels
    initial_neighborhoods.npz           the KMeans seeding
    checkpoints/                        checkpoint_<iter>.npz + .json
    optimized_neighborhoods_tiled.parquet
```

A checkpoint is the assignment vector plus iteration, temperature, stability
counter, accept/reject counts and the NumPy RNG state — a couple of megabytes.
Count tables, scores and the boundary set are all derived from it on load.

Resuming continues the *same* trajectory: stopping at iteration N and resuming is
bit-identical to never having stopped (`test_checkpoint_resume_is_bit_identical`).
That required making two things independent of run history — the boundary
sampling pool is sorted rather than set-iteration-ordered, and its refresh
schedule keys off the absolute iteration number.

Pause checkpoints are written by the worker thread when it parks, not by the GUI
thread when the button is pressed, so they can't catch the loop mid-commit.

Reopening a run reuses `tiles.parquet` and `parcels_prepared.parquet`, so neither
the downloads nor the shatter happen twice. Delete the run directory to rebuild;
delete `.mi_cache/` to re-download.

---

## Module map

| file | role |
|---|---|
| `config.py` | `RunConfig` + the notebook's constants; `repo_root`/`cache_dir`/`runs_dir` |
| `geo.py` | picks a feet-based CRS for a study area |
| `sources.py` | TIGERweb search, REST blocks, adaptive Overpass, disk cache |
| `tiger.py` | bulk TIGER/Line block shapefiles: vintage probe, download, `/vsizip/` read |
| `tiles.py` | 500 ft grid, shatter, water clip, parcel↔tile join, adjacency |
| `mi.py` | flat-bin count tables and weighted MI |
| `engine.py` | `TiledOptimizer`: consolidation pass, annealing loop, pause/stop |
| `checkpoints.py` | checkpoint format, store, pruning, run discovery |
| `partition.py` | connected components, severed-neighborhood splitting, worker grouping |
| `parallel.py` | spawn-based process pool: one optimizer per component group |
| `pipeline.py` | wires it together; `prepare()`, `fetch_blocks()`, `run_headless()` |
| `render.py` | colours and polygon flattening (no Qt) |
| `mapview.py` | the Qt canvas |
| `app.py` | the window; `python app.py` |

---

## Things worth knowing

- **One deliberate quirk is preserved.** The notebook's MI loop only visits bins
  a neighborhood occupies, dropping those bins' "outside" contribution. That's
  reproduced by default so scores stay comparable with existing runs. Set
  `exact_mi: true` in `run_config.json` for textbook MI.
- **Contiguity still isn't enforced.** Trading only along boundaries keeps
  neighborhoods mostly contiguous in practice, and tiles strengthen that.
- **Neighborhoods can be absorbed.** Empty ones score 2.0 and singletons 1.0, so
  the count drifts below the seed value over a long run. The status line's
  "hoods" figure is the live count.
- **Tests never touch the network.** Both suites pass with `socket.connect`
  hard-blocked. If you add a source, stub it in
  `test_pipeline_prepare_caches_and_resumes` — otherwise that test will happily
  download an entire state.
- **`run_headless` has no CLI in this layout.** Call it from Python, or add an
  entry point if you want the old `--headless` flags back.
