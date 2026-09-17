#!/usr/bin/env python3
"""Rebuild the validation-protocol figure and its table from one record.

The published figure and the published table were two separate records of the
same experiment and did not agree: the embedded confusion matrices summed to
1,545 and 1,676 points at 89.26% and 76.43% overall, while the table beside
them reported 1,509 and 1,616 at 88.5% and 84.3%. Nothing on disk could say
which was right, because the script that produced the table discarded the
matrices it had already computed.

It no longer does. Every count and every percentage on this figure, and every
number in the protocol table, is read from
doc/assets/jabalpur_split_protocols_v2.json, which stores the matrix alongside
the metrics derived from it.

The figure also shows the thing the old one hid. A random split and a spatially
blocked split of the same labelled points do not produce test sets of the same
composition: the random fold inherits the training set's 1,000-per-class
balance, while the blocked fold assigns whole one-degree cells and lands on a
fold that is a third open land. Part of the accuracy difference between the two
protocols is therefore a difference in what is being classified, not only in
how the partition was drawn, so the panel reports class-balanced accuracy --
the mean of the five per-class recalls, which is invariant to that composition
-- next to overall accuracy.
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from evaluation.references import REFERENCE_LABELS  # noqa: E402

ASSETS = REPO / "doc" / "assets"
RECORD = ASSETS / "jabalpur_split_protocols_v3.json"
FIGURE = REPO / "figures" / "fig_protocols.png"
TABLE = ASSETS / "result_archive" / "protocol_comparison.csv"

LABELS = list(REFERENCE_LABELS.values())
SHORT = ["Veg.", "Water", "Built", "Open", "Agri."]
MODEL_LABEL = {"gtb": "Smile GTB", "rf": "Random Forest", "svm": "SVM",
               "knn": "KNN", "cart": "CART"}
PROTOCOL_LABEL = {"random": "Random held-out split",
                  "spatially_blocked": "Spatially blocked split"}
ORDER = ("gtb", "rf", "svm", "knn", "cart")


def class_balanced(matrix):
    """Mean per-class recall: the same metric under either class composition."""
    array = np.asarray(matrix, dtype=float)
    rows = array.sum(axis=1)
    recalls = [array[i, i] / rows[i] for i in range(len(LABELS)) if rows[i]]
    return float(np.mean(recalls)) if recalls else float("nan")


def rows_for(record):
    """One row per (season, realisation, protocol, model).

    A record written before the repeated-realisation design carries a single
    `protocols` block and no `realisations`; it is read as realisation 0 so the
    same figure code draws either.
    """
    rows = []
    for season, payload in record["seasons"].items():
        realisations = payload.get("realisations") or [
            {"realisation": 0, "held_out_blocks": None,
             "protocols": payload["protocols"]}]
        for realisation in realisations:
            for protocol, entry in realisation["protocols"].items():
                for model, values in entry["models"].items():
                    matrix = np.asarray(values["confusion_matrix"], dtype=int)
                    rows.append({
                        "season": season,
                        "realisation": realisation["realisation"],
                        "held_out_blocks": "|".join(
                            str(b) for b in realisation["held_out_blocks"] or []),
                        "protocol": protocol,
                        "model": model,
                        "train_n": entry["train_count"],
                        "test_n": int(matrix.sum()),
                        "overall_accuracy": float(np.trace(matrix) / matrix.sum()),
                        "class_balanced_accuracy": class_balanced(matrix),
                        "kappa": values["kappa"],
                        **{f"reference_{slug}": int(total) for slug, total
                           in zip(SHORT, matrix.sum(axis=1))},
                    })
    return rows


def write_table(rows):
    import csv
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    with TABLE.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {TABLE.relative_to(REPO)} ({len(rows)} rows)")


def draw(record, rows):
    seasons = [s for s in ("winter", "summer") if s in record["seasons"]]
    figure, axes = plt.subplots(1, 3, figsize=(13.4, 4.1), dpi=300,
                               gridspec_kw={"width_ratios": [1.25, 1.25, 1.0]})

    # (a) and (b): overall and class-balanced accuracy, per model, per protocol.
    for panel, (metric, title) in enumerate((
            ("overall_accuracy", "(a) Overall accuracy"),
            ("class_balanced_accuracy", "(b) Class-balanced accuracy"))):
        axis = axes[panel]
        # Four bars per model: two protocols by two seasons, side by side. An
        # earlier version overlaid the seasons with alpha, which read as one
        # bar with a line across it.
        series = [(protocol, season)
                  for protocol in ("random", "spatially_blocked")
                  for season in seasons]
        width = 0.82 / len(series)
        positions = np.arange(len(ORDER))
        for index, (protocol, season) in enumerate(series):
            values, lows, highs = [], [], []
            for model in ORDER:
                match = [r[metric] * 100 for r in rows if r["season"] == season
                         and r["protocol"] == protocol and r["model"] == model]
                values.append(float(np.mean(match)) if match else np.nan)
                lows.append(values[-1] - min(match) if match else 0.0)
                highs.append(max(match) - values[-1] if match else 0.0)
            offset = (index - (len(series) - 1) / 2) * width
            # The bar is the mean over realisations and the whisker their full
            # range. One realisation of a blocked split is one arbitrary choice
            # of which ground to hold out, and the range is how much that
            # choice is worth.
            axis.bar(positions + offset, values, width * 0.9,
                     color=("#5bc0de" if protocol == "random" else "#f0ad4e"),
                     alpha=1.0 if season == "winter" else 0.55,
                     edgecolor="white", linewidth=0.5,
                     yerr=[lows, highs], error_kw={"elinewidth": 0.6,
                                                   "capsize": 1.6,
                                                   "ecolor": "#444444"},
                     label=(f"{PROTOCOL_LABEL[protocol]}, {season}"
                            if panel == 0 else None))
        axis.set_xticks(positions)
        axis.set_xticklabels([MODEL_LABEL[m] for m in ORDER], fontsize=7.5,
                             rotation=20, ha="right")
        axis.set_ylim(50, 100)
        axis.set_ylabel("%", fontsize=8)
        axis.set_title(title, fontsize=9.5)
        axis.tick_params(labelsize=7.5)
        axis.grid(axis="y", linewidth=0.4, alpha=0.35)
        axis.set_axisbelow(True)
    axes[0].legend(fontsize=6.6, frameon=False, loc="lower left", ncol=1)

    # (c) the composition difference that panel (b) removes.
    axis = axes[2]
    season = seasons[0]
    positions = np.arange(len(LABELS))
    width = 0.38
    for offset, protocol in enumerate(("random", "spatially_blocked")):
        match = [r for r in rows if r["season"] == season
                 and r["protocol"] == protocol and r["model"] == "gtb"
                 and r["realisation"] == 0][0]
        counts = [match[f"reference_{slug}"] for slug in SHORT]
        axis.bar(positions + (offset - 0.5) * width, counts, width * 0.92,
                 color=("#5bc0de" if protocol == "random" else "#f0ad4e"),
                 edgecolor="white", linewidth=0.6,
                 label=f"{PROTOCOL_LABEL[protocol]} (n={match['test_n']:,})")
    axis.set_xticks(positions)
    axis.set_xticklabels(SHORT, fontsize=7.5)
    axis.set_ylabel("reference points in the held-out fold", fontsize=8)
    axis.set_title(f"(c) Class composition of the fold, {season}, "
                   f"realisation 0", fontsize=9.5)
    axis.tick_params(labelsize=7.5)
    axis.legend(fontsize=6.6, frameon=False)
    axis.grid(axis="y", linewidth=0.4, alpha=0.35)
    axis.set_axisbelow(True)

    figure.tight_layout()
    figure.savefig(FIGURE, dpi=300, facecolor="white", bbox_inches="tight")
    print(f"wrote {FIGURE.relative_to(REPO)}")


def summarise(rows):
    for season in sorted({r["season"] for r in rows}):
        for metric in ("overall_accuracy", "class_balanced_accuracy"):
            gaps = []
            for model in ORDER:
                for realisation in sorted({r["realisation"] for r in rows}):
                    pair = {r["protocol"]: r[metric] for r in rows
                            if r["season"] == season and r["model"] == model
                            and r["realisation"] == realisation}
                    if len(pair) == 2:
                        gaps.append(
                            (pair["random"] - pair["spatially_blocked"]) * 100)
            if gaps:
                print(f"  {season:6s} {metric:26s} random minus blocked: "
                      f"{min(gaps):+.2f} to {max(gaps):+.2f} pp, "
                      f"mean {np.mean(gaps):+.2f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, default=RECORD)
    args = parser.parse_args()
    record = json.loads(args.record.read_text())
    rows = rows_for(record)
    write_table(rows)
    draw(record, rows)
    summarise(rows)


if __name__ == "__main__":
    main()
