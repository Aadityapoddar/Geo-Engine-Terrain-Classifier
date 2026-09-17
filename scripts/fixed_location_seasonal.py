#!/usr/bin/env python3
"""The seasonal comparison at fixed locations and a fixed reference label.

The headline seasonal comparison changes two things at once: the imagery and the
reference population. Agriculture's consensus mask is static by construction
(WorldCover 40 AND WorldCereal temporary crops, with Dynamic World taking no
part), and the stratified sample is seeded, so a large subset of reference
locations is drawn in both seasons and carries the same reference label in both.
That subset is the control the paper was missing: same pixel, same reference
label, different composite.

What it still does not hold fixed is the fitted model -- each season's model is
fitted on that season's composite under the same frozen configuration -- so the
contrast is imagery-plus-refit at fixed reference, not imagery alone.

Reads the frozen archive only; touches no network.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "doc" / "assets" / "result_archive"
CLASSES = ["Vegetation", "Water", "Built Area", "Open Land", "Agriculture"]
MODELS = {"gtb": "Smile GTB", "rf": "RF", "svm": "SVM", "knn": "KNN", "cart": "CART"}


def shared_locations(role="test"):
    """Locations sampled in both seasons that carry the same reference label."""
    by_point = defaultdict(dict)
    with (ARCHIVE / "predictions.csv").open() as handle:
        for row in csv.DictReader(handle):
            if row["role"] != role:
                continue
            key = (row["district"], row["lon"], row["lat"])
            by_point[key][row["season"]] = row
    shared, relabelled = [], 0
    for pair in by_point.values():
        if len(pair) != 2:
            continue
        if pair["winter"]["reference_label"] != pair["summer"]["reference_label"]:
            relabelled += 1
            continue
        shared.append(pair)
    return shared, relabelled


def accuracy(pairs, season, model, reference=None):
    column = f"predicted_after-{model}"
    rows = [p[season] for p in pairs
            if reference is None or p[season]["reference_label"] == reference]
    if not rows:
        return None, 0
    hits = sum(row[column] == row["reference_label"] for row in rows)
    return hits / len(rows), len(rows)


def main():
    pairs, relabelled = shared_locations()
    counts = {c: sum(p["winter"]["reference_label"] == c for p in pairs)
              for c in CLASSES}
    report = {
        "shared_locations": len(pairs),
        "locations_dropped_for_label_change": relabelled,
        "per_class_counts": counts,
        "models": {},
    }
    print(f"shared test locations with an unchanged reference label: {len(pairs)}")
    print(f"dropped because the reference label itself changed: {relabelled}")
    print("per class: " + ", ".join(f"{c} {n}" for c, n in counts.items()))

    for model, name in MODELS.items():
        winter, n_w = accuracy(pairs, "winter", model)
        summer, n_s = accuracy(pairs, "summer", model)
        entry = {
            "winter_agreement_pct": round(winter * 100, 2),
            "summer_agreement_pct": round(summer * 100, 2),
            "seasonal_drop_pp": round((winter - summer) * 100, 2),
            "n": n_w,
            "per_class": {},
        }
        for reference in CLASSES:
            w, n = accuracy(pairs, "winter", model, reference)
            s, _ = accuracy(pairs, "summer", model, reference)
            if n == 0:
                continue
            entry["per_class"][reference] = {
                "n": n,
                "winter_recall_pct": round(w * 100, 2),
                "summer_recall_pct": round(s * 100, 2),
                "drop_pp": round((w - s) * 100, 2),
            }
        report["models"][name] = entry
        print(f"\n{name}: winter {winter * 100:.2f}%  summer {summer * 100:.2f}%  "
              f"drop {(winter - summer) * 100:.2f} pp on {n_w} fixed locations")
        for reference, values in entry["per_class"].items():
            print(f"   {reference:12} n={values['n']:4}  "
                  f"{values['winter_recall_pct']:6.2f} -> {values['summer_recall_pct']:6.2f}  "
                  f"({values['drop_pp']:+.2f})")

    path = ROOT / "doc" / "assets" / "fixed_location_seasonal.json"
    path.write_text(json.dumps(report, indent=1) + "\n")
    print(f"\nwrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
