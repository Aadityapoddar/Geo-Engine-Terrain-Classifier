"""Spatially blocked development/test partition of the evaluation districts.

The audit's sharpest methodological objection is that the same statewide
reference population was used three times: to pick the 19-band feature stack,
to pick the winning classifier, and then to report the winner's accuracy. Any
choice made by looking at a test set stops that set being a test set, and the
final number is the maximum over the choices rather than an estimate of skill.

The repair is to split the evaluation districts once, before anything is
selected, and never let a development decision see the test half:

    DEVELOPMENT   feature-stack ablation, classifier choice, hyperparameter
                  search -- everything that involves looking at a score and
                  then changing the pipeline.
    TEST          the frozen pipeline, scored once.
    TRAINING      every district that physically contains labelled training
                  points. These are not external districts at all. They were
                  previously counted inside the statewide test population; a
                  100 m buffer keeps individual test points off individual
                  training points, but it does not make a district independent
                  of a model trained inside it.

The training set is described everywhere as "5,000 points from Jabalpur
district", and scripts/label_provenance_audit.py shows that 872 of them are
not: 868 fall in Katni, three in Dindori and one in Mandla, all within 23 km of
the Jabalpur boundary. Katni was carved out of Jabalpur in 1998, so a wider
historical outline of "Jabalpur" explains the placement, but it does not undo
the consequence -- under the FAO GAUL 2015 boundaries the evaluation actually
uses, those three districts held training data and were simultaneously being
reported as independent test districts. All four are therefore withheld, which
costs four districts and buys back the claim that the test set is external.

The split is by *spatial block*, not by district. Neighbouring districts share
soils, cropping calendar and reference-map error, so assigning them
independently would put near-duplicates on both sides and make the development
half a partial copy of the test half. Blocks are one-degree cells of the
district centroid, and a cell goes to development when (row + column) % 3 == 0.
That is a fixed, seedless, checkerboard-style rule: it needs no RNG to
reproduce, it spreads development blocks across the state's full longitude and
climate range rather than carving off one corner, and it keeps each block whole.

Deterministic given doc/assets/mp_district_centroids.json, so the partition can
be recomputed and audited without Earth Engine.
"""

import json
import math
from pathlib import Path

# Measured, not assumed: see scripts/label_provenance_audit.py and
# doc/assets/training_label_provenance.json for the point counts behind each.
TRAINING_DISTRICTS = ("Jabalpur", "Katni", "Dindori", "Mandla")
BLOCK_DEGREES = 1.0
DEVELOPMENT_MODULUS = 3
CENTROIDS_PATH = (
    Path(__file__).resolve().parents[1] / "doc" / "assets" /
    "mp_district_centroids.json"
)


def block_of(lon, lat, block_degrees=BLOCK_DEGREES):
    """Row/column of the spatial block a centroid falls in."""
    return (int(math.floor(lat / block_degrees)),
            int(math.floor(lon / block_degrees)))


def district_splits(centroids=None):
    """Return {"development": [...], "test": [...], "training": [...]}."""
    if centroids is None:
        centroids = json.loads(CENTROIDS_PATH.read_text())
    development, test = [], []
    for name in sorted(centroids):
        if name in TRAINING_DISTRICTS:
            continue
        row, column = block_of(centroids[name]["lon"], centroids[name]["lat"])
        target = development if (row + column) % DEVELOPMENT_MODULUS == 0 else test
        target.append(name)
    return {
        "development": development,
        "test": test,
        "training": list(TRAINING_DISTRICTS),
    }


def split_of(district, splits=None):
    """Which half a district belongs to."""
    splits = splits or district_splits()
    for name, members in splits.items():
        if district in members:
            return name
    raise KeyError(f"{district} is not an evaluation district")


def block_membership(centroids=None):
    """Blocks with their districts, for plotting and for the split manifest."""
    if centroids is None:
        centroids = json.loads(CENTROIDS_PATH.read_text())
    blocks = {}
    for name in sorted(centroids):
        row, column = block_of(centroids[name]["lon"], centroids[name]["lat"])
        key = f"{row},{column}"
        entry = blocks.setdefault(key, {
            "row": row,
            "column": column,
            "assignment": ("development"
                           if (row + column) % DEVELOPMENT_MODULUS == 0
                           else "test"),
            "districts": [],
        })
        entry["districts"].append(name)
    return blocks


if __name__ == "__main__":
    splits = district_splits()
    for name, members in splits.items():
        print(f"{name:12s} {len(members):2d}  {', '.join(members)}")
    blocks = block_membership()
    print(f"\n{len(blocks)} spatial blocks, "
          f"{sum(b['assignment'] == 'development' for b in blocks.values())} "
          "assigned to development")
