#!/usr/bin/env python3
"""Per-class area-weighted agreement over the covered reference domain.

The manuscript reports the weighted overall figure and, until now, said the
per-class version was not computed. It is the informative one: a weighted OA of
75.5% over a domain that is 95.9% agriculture says almost nothing, whereas the
weighted producer's and user's accuracies say which classes the map actually
gets right on the ground the reference covers.

Strata are (district, consensus class), so every point in a stratum shares its
reference class and the weighted error matrix is

    p[a][b] = sum_d  W[d,a] * C_d[a][b] / n[d,a]

with W the stratum area share. Reads the frozen archive only; touches no network.
"""
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "doc" / "assets" / "result_archive"
MATRIX_CLASSES = ["vegetation", "water", "builtarea", "openland", "agriculture"]
AREA_CLASSES = ["Vegetation", "Water", "Built Area", "Open Land", "Agriculture"]


def _rows(name):
    with (ARCHIVE / f"{name}.csv").open() as handle:
        return list(csv.DictReader(handle))


def weighted_matrix(season, model="after-gtb", role="test"):
    areas = {}
    for row in _rows("stratum_areas"):
        if row["season"] != season or row["role"] != role:
            continue
        areas[(row["district"], row["consensus_class"])] = float(row["eligible_area_km2"])
    total = sum(areas.values())

    p = np.zeros((5, 5))
    covered = 0.0
    for row in _rows("district_confusion_matrices"):
        if row["season"] != season or row["model_key"] != model or row["role"] != role:
            continue
        for a, (mcls, acls) in enumerate(zip(MATRIX_CLASSES, AREA_CLASSES)):
            counts = np.array([int(row[f"c_{mcls}_{b}"]) for b in MATRIX_CLASSES], float)
            n = counts.sum()
            area = areas.get((row["district"], acls), 0.0)
            if n == 0 or area == 0:
                continue
            p[a] += (area / total) * counts / n
            covered += area / total
    # two summer strata carry area and no sample: renormalise onto what is
    # represented rather than silently crediting the gap to the diagonal
    return p / p.sum(), p.sum()


def report():
    out = {}
    for season in ("winter", "summer"):
        p, represented = weighted_matrix(season)
        oa = p.diagonal().sum()
        producer = np.divide(p.diagonal(), p.sum(axis=1),
                             out=np.zeros(5), where=p.sum(axis=1) > 0)
        user = np.divide(p.diagonal(), p.sum(axis=0),
                         out=np.zeros(5), where=p.sum(axis=0) > 0)
        prevalence = p.sum(axis=1)
        out[season] = {
            "weighted_overall_accuracy": round(oa * 100, 2),
            "represented_share_of_domain": round(represented * 100, 4),
            "per_class": {
                AREA_CLASSES[i]: {
                    "reference_area_share_pct": round(prevalence[i] * 100, 3),
                    "weighted_producers_accuracy_pct": round(producer[i] * 100, 2),
                    "weighted_users_accuracy_pct": round(user[i] * 100, 2),
                }
                for i in range(5)
            },
        }
        print(f"\n{season.upper()}  weighted OA = {oa * 100:.2f}%")
        print(f"  {'class':12} {'area %':>8} {'producer %':>11} {'user %':>8}")
        for i, name in enumerate(AREA_CLASSES):
            print(f"  {name:12} {prevalence[i] * 100:8.3f} "
                  f"{producer[i] * 100:11.2f} {user[i] * 100:8.2f}")
    path = ROOT / "doc" / "assets" / "weighted_per_class.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    report()
