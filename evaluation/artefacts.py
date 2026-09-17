"""Where the current results live, named once.

The statewide evaluation has been re-run several times, and each run writes a
new set of files: shards, an aggregate, a bootstrap summary, an area-adjusted
estimate. Every script that reads them used to spell the version into its own
source, so a rerun meant editing five files and finding the one that was missed
by noticing a suspicious number in a figure days later.

The version is declared here instead. Bump `RESULTS_VERSION` when a rerun lands
and everything downstream follows, which also means a figure and a table can no
longer be built from different runs.

Kept separate from backend.config because these are outputs, not settings: the
backend serves a map without ever reading them.
"""

from pathlib import Path

ASSETS = Path(__file__).resolve().parents[1] / "doc" / "assets"

# The run every figure, table and benchmark in the current manuscript describes.
# v6 was the first under the matched-window pipeline; v7 adds the band stack
# and hyperparameters the development districts selected. A v8 under uniform
# 0.70 reference screening was attempted and abandoned: it empties the
# agriculture and open-land strata outright (evaluation/references.DW_SCREEN).
RESULTS_VERSION = "v7"


def versioned(stem, version=None):
    """Path of one versioned artefact, e.g. versioned("mp_spatial_uncertainty")."""
    return ASSETS / f"{stem}_{version or RESULTS_VERSION}.json"


def shard_files(version=None):
    """The per-worker shard files of one statewide run, in worker order."""
    version = version or RESULTS_VERSION
    return sorted(ASSETS.glob(f"mp_external_shards_{version}_worker_*.json"))


EXTERNAL_RESULTS = "mp_external_results"
SPATIAL_UNCERTAINTY = "mp_spatial_uncertainty"
AREA_ADJUSTED = "mp_area_adjusted"
