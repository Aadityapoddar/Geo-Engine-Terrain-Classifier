#!/usr/bin/env python3
"""Design-based, area-adjusted accuracy and class areas (Olofsson et al. 2014).

The statewide sample takes 20 points per class per district. That is a good way
to measure a rare class and a bad way to measure a map: Water gets the same
number of points as Agriculture while covering a tiny fraction of the ground, so
the pooled confusion matrix describes a landscape that is one-fifth water. Every
overall accuracy reported off that matrix is the accuracy of a population that
does not exist, and no class area can be read from it at all.

The repair is the standard stratified estimator. Each stratum is one
(district, reference class) cell -- exactly the unit the sample was drawn by --
carrying a weight W_h equal to its share of the reporting domain's area, as
measured by scripts/reference_stratum_weights.py. Then, with p_hj the share of
stratum h's sample that the map assigned to class j:

    area proportion of cell (h, j)   p_hj_hat = W_h * n_hj / n_h
    area-adjusted overall accuracy   sum_h W_h * n_hh / n_h
    estimated area of map class j    sum_h p_hj_hat, times the domain area
    user's accuracy of class j       p_jj_hat / sum_h p_hj_hat
    producer's accuracy of class c   p_cc_hat / sum_over_districts W_(d,c)

Variances follow Olofsson's equations for a stratified random sample, so the
confidence intervals here are design-based rather than bootstrapped. They answer
a different question from scripts/spatial_uncertainty.py, which resamples whole
districts to capture spatial correlation; both belong in the paper. This one
says how precisely the sample pins down the map's area-weighted accuracy, that
one says how much the answer depends on which districts happened to be sampled.

A stratum with fewer than two sampled points contributes its point estimate but
no variance, and the share of domain area in such strata is reported so the
reader can see how much of the map that covers.
"""

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from evaluation.references import (  # noqa: E402
    REFERENCE_LABELS,
    canonical_label,
)
from evaluation.splits import district_splits  # noqa: E402

LABELS = list(REFERENCE_LABELS.values())
Z95 = 1.959964


def load_shards(paths):
    seasons = {}
    for path in paths:
        payload = json.loads(Path(path).read_text())
        for shard in payload["shards"].values():
            seasons.setdefault(shard["season"], {})[shard["district"]] = \
                shard["matrices"]
    return seasons


def load_weights(path):
    payload = json.loads(Path(path).read_text())
    weights, district_area = {}, {}
    for record in payload["strata"].values():
        key = (record["season"], record["district"])
        weights[key] = {canonical_label(label): area
                        for label, area in record["class_area_m2"].items()}
        district_area[key] = record["district_area_m2"]
    return weights, district_area, payload.get("scale_m")


def domain_coverage(weights, district_area, season, districts):
    """How much ground the consensus actually labels, and what it labels it as.

    This is the caveat that decides how the area-adjusted number may be read.
    The reference is a consensus of five products under a triple gate for some
    classes and a double gate for others -- Forest needs WorldCover trees AND a
    Dynamic World tree label AND 0.70 confidence, while Agriculture needs only
    WorldCover cropland AND WorldCereal -- so the surviving domain is not a
    scale model of the state. Anything weighted by these areas is an estimate
    for that domain and not for Madhya Pradesh.
    """
    masked, total, per_class = 0.0, 0.0, {}
    for district in districts:
        key = (season, district)
        if key not in weights:
            continue
        total += district_area[key]
        for label, area in weights[key].items():
            per_class[label] = per_class.get(label, 0.0) + (area or 0.0)
            masked += area or 0.0
    return {
        "district_area_km2": total / 1e6,
        "labelled_area_km2": masked / 1e6,
        "labelled_share_of_districts": masked / total if total else None,
        "reference_class_area_share": {
            label: (area / masked if masked else None)
            for label, area in sorted(per_class.items(),
                                      key=lambda item: -item[1])
        },
    }


def estimate(matrices, areas, districts):
    """Olofsson stratified estimates over the given districts."""
    size = len(LABELS)
    strata = []
    for district in districts:
        if district not in matrices or district not in areas:
            continue
        matrix = matrices[district]
        class_area = areas[district]
        for index, label in enumerate(LABELS):
            counts = matrix[index]
            n_h = sum(counts)
            area = class_area.get(label) or 0.0
            if area <= 0:
                continue
            strata.append({
                "district": district,
                "reference": label,
                "reference_index": index,
                "area_m2": area,
                "n": n_h,
                "counts": counts,
            })
    total_area = sum(stratum["area_m2"] for stratum in strata)
    if not total_area:
        raise ValueError("No stratum area for the requested districts")
    sampled_area = sum(stratum["area_m2"] for stratum in strata if stratum["n"])
    for stratum in strata:
        stratum["weight"] = stratum["area_m2"] / total_area

    # Area-proportion confusion matrix.
    cells = [[0.0] * size for _ in range(size)]
    overall = 0.0
    overall_variance = 0.0
    low_sample_area = 0.0
    for stratum in strata:
        n_h = stratum["n"]
        if not n_h:
            continue
        weight = stratum["weight"]
        row = stratum["reference_index"]
        for column in range(size):
            share = stratum["counts"][column] / n_h
            cells[row][column] += weight * share
        correct = stratum["counts"][row] / n_h
        overall += weight * correct
        if n_h > 1:
            overall_variance += (
                weight ** 2 * correct * (1 - correct) / (n_h - 1))
        else:
            low_sample_area += stratum["area_m2"]

    # Renormalise onto the area actually represented by sampled strata, so the
    # estimate describes the domain it can speak for rather than silently
    # counting unsampled strata as zero accuracy.
    represented = sum(stratum["weight"] for stratum in strata if stratum["n"])
    if represented:
        overall /= represented
        overall_variance /= represented ** 2
        cells = [[value / represented for value in row] for row in cells]

    mapped_proportion = [sum(cells[row][column] for row in range(size))
                         for column in range(size)]
    reference_proportion = [sum(cells[row]) for row in range(size)]

    per_class = {}
    for index, label in enumerate(LABELS):
        users = (cells[index][index] / mapped_proportion[index]
                 if mapped_proportion[index] else None)
        producers = (cells[index][index] / reference_proportion[index]
                     if reference_proportion[index] else None)
        # Olofsson eq. 10: standard error of an estimated area proportion.
        variance = 0.0
        for stratum in strata:
            n_h = stratum["n"]
            if n_h < 2:
                continue
            share = stratum["counts"][index] / n_h
            variance += (stratum["weight"] ** 2 * share * (1 - share)
                         / (n_h - 1))
        if represented:
            variance /= represented ** 2
        error = math.sqrt(variance)
        per_class[label] = {
            "users_accuracy": users,
            "producers_accuracy": producers,
            "f1": (2 * users * producers / (users + producers)
                   if users and producers else None),
            "mapped_area_proportion": mapped_proportion[index],
            "mapped_area_km2": mapped_proportion[index] * sampled_area / 1e6,
            "mapped_area_ci95_km2": Z95 * error * sampled_area / 1e6,
            "reference_area_proportion": reference_proportion[index],
        }

    error = math.sqrt(overall_variance)
    return {
        "districts": len(set(stratum["district"] for stratum in strata)),
        "strata": len(strata),
        "strata_with_samples": sum(1 for s in strata if s["n"]),
        "sample_count": sum(stratum["n"] for stratum in strata),
        "domain_area_km2": total_area / 1e6,
        "sampled_domain_area_km2": sampled_area / 1e6,
        "area_adjusted_overall_accuracy": overall,
        "area_adjusted_overall_accuracy_ci95": Z95 * error,
        "area_proportion_matrix": cells,
        "per_class": per_class,
        "area_share_in_strata_below_two_points": (
            low_sample_area / total_area if total_area else 0.0),
    }


def run(args):
    seasons = load_shards(args.shards)
    weights, district_area, scale = load_weights(args.weights)
    splits = district_splits()
    output = {
        "weights_source": str(args.weights),
        "weight_scale_m": scale,
        "estimator": "Olofsson et al. (2014) stratified, strata = district x reference class",
        "classifier": args.classifier,
        "district_splits": splits,
        "seasons": {},
    }
    for season, districts in sorted(seasons.items()):
        matrices = {
            name: value[args.classifier]
            for name, value in districts.items()
            if args.classifier in value
        }
        areas = {name: weights[(season, name)] for name in matrices
                 if (season, name) in weights}
        payload = {}
        for split in ("development", "test"):
            members = [name for name in splits[split] if name in areas]
            if members:
                payload[split] = estimate(matrices, areas, members)
        for split in ("development", "test"):
            if split in payload:
                payload[split]["domain"] = domain_coverage(
                    weights, district_area, season, splits[split])
        payload["external"] = estimate(
            matrices, areas,
            [name for name in splits["development"] + splits["test"]
             if name in areas])
        output["seasons"][season] = payload

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n")

    for season, payload in output["seasons"].items():
        print(f"\n== {season} ({args.classifier}) ==")
        for split, value in payload.items():
            print(f"  {split:12s} {value['districts']:2d} districts, "
                  f"n={value['sample_count']}, "
                  f"area {value['sampled_domain_area_km2']:9.0f} km2")
            print(f"    area-adjusted OA "
                  f"{value['area_adjusted_overall_accuracy'] * 100:5.2f}% "
                  f"+/- {value['area_adjusted_overall_accuracy_ci95'] * 100:.2f}")
            domain = value.get("domain")
            if domain:
                shares = ", ".join(
                    f"{label} {share * 100:.1f}%"
                    for label, share in
                    list(domain["reference_class_area_share"].items())[:3])
                print(f"    domain: consensus labels "
                      f"{domain['labelled_share_of_districts'] * 100:.1f}% of "
                      f"the districts' area; composition {shares}")
        value = payload["test"]
        print("    class            UA      PA   mapped area km2 (+/-95%)")
        for label in LABELS:
            per = value["per_class"][label]
            users = per["users_accuracy"]
            producers = per["producers_accuracy"]
            print(f"    {label:12s} "
                  f"{'   n/a' if users is None else f'{users * 100:6.1f}'} "
                  f"{'   n/a' if producers is None else f'{producers * 100:6.1f}'}   "
                  f"{per['mapped_area_km2']:10.0f} "
                  f"+/-{per['mapped_area_ci95_km2']:8.0f}")
    print(f"\nwrote {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shards", nargs="+", type=Path)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--classifier", default="after-gtb")
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
