#!/usr/bin/env python3
"""Every pairwise classifier contrast, not only the ones against the leader.

The paper claimed CART was "separated from every other classifier" while only
the four leader-versus-rest contrasts had been tested. Ten unordered pairs exist
per season; this runs all ten under the same paired procedure the leader
contrasts use -- mean per-district overall-accuracy difference, a percentile
interval from resampling the 29 district identifiers jointly for both models,
and a two-sided Wilcoxon signed-rank p adjusted by Holm within the season.

The multiplicity family is the whole set of ten, which is stricter than the
four-contrast family the leader table declares, so the two tables report
different adjusted p for the same raw p and both say which family they used.

Reads the frozen archive only; touches no network.
"""
import csv
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "doc" / "assets" / "result_archive"
CLASSES = ["vegetation", "water", "builtarea", "openland", "agriculture"]
MODELS = ["gtb", "rf", "svm", "knn", "cart"]
NAMES = {"gtb": "Smile GTB", "rf": "RF", "svm": "SVM", "knn": "KNN", "cart": "CART"}
BOOTSTRAP_SEED = 20260901
BOOTSTRAP_REPLICATES = 5000


def district_accuracy(role="test"):
    """{season: {model: {district: overall accuracy}}} from the stored matrices."""
    out = {}
    with (ARCHIVE / "district_confusion_matrices.csv").open() as handle:
        for row in csv.DictReader(handle):
            if row["role"] != role or not row["model_key"].startswith("after-"):
                continue
            model = row["model_key"].removeprefix("after-")
            if model not in MODELS:
                continue
            counts = np.array([[float(row[f"c_{a}_{b}"]) for b in CLASSES]
                               for a in CLASSES])
            out.setdefault(row["season"], {}).setdefault(model, {})[row["district"]] = (
                counts.trace() / counts.sum())
    return out


def holm(p_values):
    """Holm-Bonferroni step-down adjusted p-values, monotonised."""
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def contrasts(accuracy, season):
    per_model = accuracy[season]
    districts = sorted(set.intersection(*(set(per_model[m]) for m in MODELS)))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(districts), size=(BOOTSTRAP_REPLICATES, len(districts)))

    rows = []
    for i, left in enumerate(MODELS):
        for right in MODELS[i + 1:]:
            differences = np.array([per_model[left][d] - per_model[right][d]
                                    for d in districts])
            means = differences[draws].mean(axis=1)
            low, high = np.percentile(means, [2.5, 97.5])
            nonzero = differences[differences != 0]
            magnitudes = np.abs(nonzero)
            rows.append({
                "season": season,
                "contrast": f"{NAMES[left]} - {NAMES[right]}",
                "districts": len(districts),
                "mean_district_oa_difference_pp": round(differences.mean() * 100, 3),
                "ci95_low_pp": round(low * 100, 3),
                "ci95_high_pp": round(high * 100, 3),
                "districts_tied": int(differences.size - nonzero.size),
                "abs_difference_ties": int(magnitudes.size - np.unique(magnitudes).size),
                "wilcoxon_p_raw": float(stats.wilcoxon(
                    differences, zero_method="wilcox", alternative="two-sided").pvalue),
                "family": f"{season}: all ten pairwise classifier contrasts",
            })
    adjusted = holm(np.array([row["wilcoxon_p_raw"] for row in rows]))
    for row, value in zip(rows, adjusted):
        row["wilcoxon_p_holm"] = float(value)
    return rows


def main():
    accuracy = district_accuracy()
    rows = []
    for season in ("winter", "summer"):
        rows.extend(contrasts(accuracy, season))

    fields = list(rows[0])
    path = ARCHIVE / "pairwise_contrasts_all.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    for season in ("winter", "summer"):
        print(f"\n{season.upper()}   {'contrast':22} {'diff pp':>8} "
              f"{'CI95':>18} {'raw p':>9} {'Holm p':>9}")
        for row in rows:
            if row["season"] != season:
                continue
            interval = f"[{row['ci95_low_pp']:+.2f}, {row['ci95_high_pp']:+.2f}]"
            print(f"         {row['contrast']:22} {row['mean_district_oa_difference_pp']:+8.2f} "
                  f"{interval:>18} {row['wilcoxon_p_raw']:9.4f} {row['wilcoxon_p_holm']:9.4f}")
    print(f"\nwrote {path.relative_to(ROOT)}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
