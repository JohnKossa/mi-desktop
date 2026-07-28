"""Desktop version of the MI boundary-swapping neighborhood optimizer.

Pipeline
--------
1. User types a jurisdiction (free text, e.g. "Fort Myers, FL").
2. TIGERweb is geocoded/searched for the matching place, CDP, county subdivision
   or county; its boundary becomes the study area.
3. 2020 Census blocks (TIGERweb layer 12), OSM roads and OSM waterways are
   downloaded for that study area and cached on disk.
4. A 500 ft grid is drawn over the study area; blocks + roads + grid are
   shattered (linework union -> polygonize) into tiles, and water is clipped out.
5. The user's parcel file is loaded, joined to tiles, KMeans-seeded, run through
   the MI consolidation pass, then optimized with tile-level simulated annealing.
6. A Qt window shows the tile map live, repainting every N accepted swaps, and
   checkpoints are written so a run can be paused/stopped and resumed.

Entry point: ``python run_desktop.py`` at the repo root.
"""

__version__ = "0.1.0"
