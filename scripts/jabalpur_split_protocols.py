#!/usr/bin/env python3
"""The random and spatially blocked splits inside the labelled area, under v4.

Table XI sets three protocols against each other: a random held-out split, a
spatially blocked split, and the external district consensus. Only the third
was re-measured when the pipeline changed, which would have left one table
quoting two different composites -- exactly the internal inconsistency this
revision exists to remove. This recomputes the first two on the v4 composite.

The blocked split is ten contiguous longitudinal bands across the labelled
extent, three of which are held out. It replaces the 0.1-degree lattice
scrambled by an integer hash that earlier versions used: that assignment leaked
at every cell boundary, so the "blocked" fold sat metres from its own training
points and measured little more than a random split does.

Three properties make the contrast a contrast rather than a confound. Both
protocols run on the production sel19 stack, not the weaker alt19 an earlier
version used. Both are given the same number of training rows in a realisation,
so partition geometry is what differs and not training size. And the whole thing
is repeated over rotations of the held-out window, so the protocol effect is
reported with its spread across realisations instead of as one number from one
arbitrary fold.

The random split is the thing being argued against, so it is drawn the naive
way on purpose: a uniform split over the same points, matched to the blocked
fold's size, with no spatial constraint at all.
"""

import argparse
import json
import sys
from pathlib import Path

import ee

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backend.config import (  # noqa: E402
    BANDS,
    SEASONS,
    TRAINING_SCHEMA_VERSION,
)
from backend.gee_classifier import (  # noqa: E402
    init_ee,
    make_classifier,
    sample_training_points,
)
from evaluation.assets import training_points  # noqa: E402
from evaluation.metrics import metrics_from_matrix  # noqa: E402
from evaluation.references import REFERENCE_LABELS  # noqa: E402
from evaluation.runner import MODEL_NAMES  # noqa: E402
from scripts.run_seasonal_evaluation import BLOCK_SEED, BLOCK_SPLIT  # noqa: E402

DEFAULT_OUTPUT = REPO / "doc" / "assets" / "jabalpur_split_protocols_v3.json"
LABELS = list(REFERENCE_LABELS.values())
RANDOM_TRAIN_FRACTION = 0.7


BLOCK_COUNT = 10


def _contiguous_blocks(points):
    """Ten contiguous longitudinal bands across the labelled extent.

    Through v7 the assignment was (31*bx + 17*by) mod 10 over 0.1-degree cells.
    Two things were wrong with that. The cells were ten times finer than the
    one degree the manuscript claimed, and the modulus is a hash, so adjacent
    cells landed in different folds and training and test points sat metres
    apart across every cell boundary. A blocked split that leaks at every
    boundary measures little more than a random one.

    Bands leak only at the nine edges between them. The width is the labelled
    extent divided by BLOCK_COUNT and is written into the output rather than
    assumed, since the extent is a property of the label set and not a constant.
    """
    coordinates = ee.List(points.geometry().bounds().coordinates().get(0))
    longitudes = coordinates.map(lambda c: ee.List(c).get(0))
    lon_min = ee.Number(longitudes.reduce(ee.Reducer.min()))
    lon_max = ee.Number(longitudes.reduce(ee.Reducer.max()))
    width = lon_max.subtract(lon_min).divide(BLOCK_COUNT)

    def assign(feature):
        lon = ee.Number(feature.geometry().coordinates().get(0))
        band = lon.subtract(lon_min).divide(width).floor().min(BLOCK_COUNT - 1)
        return feature.set("block", band)

    return points.map(assign), width


def _metrics(matrix):
    return metrics_from_matrix([[int(v) for v in row] for row in matrix], LABELS)


def _score(train_table, test_table, bands):
    """Train every model on one table and score it on the other."""
    payload = {"sample_count": test_table.size()}
    for model in MODEL_NAMES:
        classifier = make_classifier(model).train(
            features=train_table, classProperty="label", inputProperties=bands)
        payload[model] = (
            test_table.classify(classifier)
            .errorMatrix("label", "classification", list(REFERENCE_LABELS))
            .array().toList()
        )
    return ee.Dictionary(payload).getInfo()


def _budgeted(table, budget, seed):
    """Cut a training table down to `budget` rows, deterministically.

    The two protocols do not hand their models the same amount of data -- a
    contiguous band split leaves whatever the bands leave -- so a protocol
    contrast run without this measures training size as well as partition
    geometry. Every fit in one realisation is given the same number of rows.
    """
    ordered = table.randomColumn("budget", seed).sort("budget")
    return ee.FeatureCollection(ordered.toList(budget))


def _entry(raw, train_size):
    sample_count = raw.pop("sample_count")
    entry = {"sample_count": sample_count, "train_count": train_size,
             "models": {}}
    for model, matrix in raw.items():
        metrics = _metrics(matrix)
        entry["models"][model] = {
            # The matrix itself, not only the three numbers derived from it.
            # Without it the protocol figure and the protocol table were built
            # from separate records and drifted apart: the published figure
            # implied n = 1,545 and n = 1,676 against a table reporting 1,509
            # and 1,616, and nothing on disk could say which was right. Every
            # count in both now comes from here.
            "confusion_matrix": [[int(v) for v in row] for row in matrix],
            "sample_count": int(sum(sum(row) for row in matrix)),
            "reference_totals": [int(sum(row)) for row in matrix],
            "predicted_totals": [int(sum(row[i] for row in matrix))
                                 for i in range(len(matrix))],
            "overall_accuracy": metrics["overall_accuracy"],
            "kappa": metrics["kappa"],
            "macro_f1": metrics["macro_f1"],
            "per_class_f1": {label: metrics["per_class"][label]["f1"]
                             for label in LABELS},
        }
    values = [value["overall_accuracy"] for value in entry["models"].values()]
    entry["overall_accuracy_range"] = [min(values), max(values)]
    return entry


def _held_out_bands(index, width):
    """The `width` contiguous bands held out by realisation `index`."""
    return [(index + offset) % BLOCK_COUNT for offset in range(width)]


def run(args):
    init_ee()
    # BANDS is the production sel19 stack the rest of the paper reports.
    # Through v7 this read BAND_STACKS["b19"], which is alt19: the two share
    # thirteen bands of nineteen, and alt19 is the lowest-scoring of the four
    # candidates. The protocol contrast was internally valid but measured on
    # the weakest configuration in the paper, and it was captioned as sel19.
    bands = list(BANDS)
    dates = SEASONS[args.season]
    points = training_points("after")
    sampled = sample_training_points(
        points, start_date=dates["start"], end_date=dates["end"], bands=bands)
    blocked, band_width = _contiguous_blocks(sampled)

    held_out_width = BLOCK_COUNT - BLOCK_SPLIT
    output = {
        "training_schema_version": TRAINING_SCHEMA_VERSION,
        "season": args.season,
        "band_stack": "sel19 (production)",
        "block_geometry": "contiguous longitudinal bands",
        "block_count": BLOCK_COUNT,
        "block_width_degrees": band_width.getInfo(),
        "held_out_bands_per_realisation": held_out_width,
        "realisation_count": args.realisations,
        "training_budget": args.budget,
        "random_train_fraction": RANDOM_TRAIN_FRACTION,
        "realisations": [],
        "protocols": {},
    }

    for index in range(args.realisations):
        held_out = _held_out_bands(index, held_out_width)
        keep = ee.Filter.inList("block", held_out)
        blocked_test = blocked.filter(keep)
        blocked_train = blocked.filter(ee.Filter.Not(keep))
        # The random protocol is drawn afresh per realisation and matched to
        # this realisation's held-out size, so the two differ in partition
        # geometry and in nothing else that is under our control.
        test_size = blocked_test.size().getInfo()
        total = blocked.size().getInfo()
        randomised = sampled.randomColumn("split", BLOCK_SEED + index)
        threshold = 1.0 - test_size / total
        random_train = randomised.filter(ee.Filter.lt("split", threshold))
        random_test = randomised.filter(ee.Filter.gte("split", threshold))

        budget = args.budget or min(blocked_train.size().getInfo(),
                                    random_train.size().getInfo())
        realisation = {"realisation": index, "held_out_blocks": held_out,
                       "training_budget": budget, "protocols": {}}
        for name, (train_table, test_table) in {
            "random": (random_train, random_test),
            "spatially_blocked": (blocked_train, blocked_test),
        }.items():
            entry = _entry(_score(_budgeted(train_table, budget,
                                            BLOCK_SEED + index),
                                  test_table, bands), budget)
            realisation["protocols"][name] = entry
            print(f"r{index} {name:18s} train={budget} n={entry['sample_count']:5d}  "
                  + "  ".join(f"{m} {v['overall_accuracy'] * 100:.1f}"
                              for m, v in sorted(entry["models"].items())),
                  flush=True)
        realisation["protocol_effect_pp"] = {
            model: round((realisation["protocols"]["random"]["models"][model]
                          ["overall_accuracy"]
                          - realisation["protocols"]["spatially_blocked"]["models"]
                          [model]["overall_accuracy"]) * 100, 3)
            for model in MODEL_NAMES
        }
        output["realisations"].append(realisation)
        if index == 0:
            output["protocols"] = realisation["protocols"]

    effects = {model: [r["protocol_effect_pp"][model] for r in output["realisations"]]
               for model in MODEL_NAMES}
    output["protocol_effect_summary_pp"] = {
        model: {
            "mean": round(sum(values) / len(values), 3),
            "min": min(values),
            "max": max(values),
            "sd": round((sum((v - sum(values) / len(values)) ** 2
                             for v in values) / max(len(values) - 1, 1)) ** 0.5, 3),
            "realisations": values,
        }
        for model, values in effects.items()
    }
    print("\nprotocol effect (random minus blocked), percentage points:")
    for model, summary in output["protocol_effect_summary_pp"].items():
        print(f"  {model:5} mean {summary['mean']:+6.2f}  sd {summary['sd']:5.2f}  "
              f"range [{summary['min']:+.2f}, {summary['max']:+.2f}] over "
              f"{len(summary['realisations'])} realisations")

    existing = (json.loads(args.output.read_text())
                if args.output.exists() else {"seasons": {}})
    existing.setdefault("seasons", {})[args.season] = output
    existing["training_schema_version"] = TRAINING_SCHEMA_VERSION
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(existing, indent=2) + "\n")
    print(f"wrote {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", default="winter", choices=list(SEASONS))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--realisations", type=int, default=8,
                        help="contiguous held-out band windows to repeat over")
    parser.add_argument("--budget", type=int, default=None,
                        help="training rows per fit; default is the largest "
                             "budget both protocols can meet in a realisation")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
