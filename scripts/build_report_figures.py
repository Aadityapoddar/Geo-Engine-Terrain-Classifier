#!/usr/bin/env python3
"""Build the quantitative figures for the LaTeX report, from the stored results.

Every number plotted here is read out of a results artefact in doc/assets -- none
is transcribed from the paper. Each figure carries a provenance line naming the
file, the sample count and the schema the run was computed under, because two of
these studies predate the current class inventory and a reader has to be able to
tell which is which:

    F1, F2  four-class soil-era feature studies (Forest/Water/Buildings/Soil)
    F3-F6   five-class-19-band-v3-no-sand, the shipped inventory

Output is PDF (vector, for \\includegraphics) and PNG at 300 dpi (for preview),
sized to IEEEtran columns: 3.5 in for a `figure`, 7.16 in for a `figure*`.

    python scripts/build_report_figures.py
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from backend.config import BANDS, TRAINING_SCHEMA_VERSION  # noqa: E402
from evaluation.artefacts import (  # noqa: E402
    AREA_ADJUSTED,
    EXTERNAL_RESULTS,
    SPATIAL_UNCERTAINTY,
    versioned,
)

ASSETS = os.path.join(REPO, "doc", "assets")
OUT = os.path.join(ASSETS, "report_figs")

COL, WIDE = 3.5, 7.16          # IEEEtran \columnwidth and \textwidth, inches

# Two-series categorical pair, validated for CVD separation against a white
# surface (worst adjacent dE 21.9 protan, 31.2 normal). Winter/Before keep the
# blue and Summer/After the vermillion everywhere they appear, so a colour means
# the same thing in every figure of the report.
BLUE, VERM = "#0072B2", "#D55E00"
GREY, INK, MUTED = "#8a8a86", "#1a1a1a", "#5a5a56"
SEQ = ["#deebf7", "#9ecae1", "#4292c6", "#08519c"]   # single-hue, light to dark

# Row/column order of every confusion matrix this project writes.
CLASS_ORDER = ["Vegetation", "Water", "Built Area", "Open Land",
               "Agriculture"]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.edgecolor": "#555555",
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,          # embed TrueType; keeps text selectable
})


def load(name):
    with open(os.path.join(ASSETS, name)) as handle:
        return json.load(handle)


def tidy(ax, grid_axis="y"):
    """Recessive grid, no top/right spines -- the marks carry the chart."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, axis=grid_axis, color="#d8d8d4", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, colors=MUTED)


def provenance(fig, text):
    fig.text(0.5, -0.012, text, ha="center", va="top", fontsize=6, color=MUTED,
             style="italic")


def save(fig, stem):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{stem}.{ext}"))
    plt.close(fig)
    print(f"  wrote {stem}.pdf / .png")



# ── F2a ────────────────────────────────────────────────────────────────────
def fig_family_removal():
    """What each feature family is worth, and which error it exists to fix.

    The same two questions the retired version asked, on evidence that belongs
    to this paper: leave-one-family-out against the 27-band stack, scored
    district by district on the development half under the five-class
    consensus. The old figure came from a four-class WorldCover tile study with
    a soil-era class inventory and a composite belonging to neither season, so
    none of its numbers could be quoted beside the rest of the results.

    Panel (b) is the point of the whole stack. Open land called built area is
    the confusion the texture and radar families were added to remove, so the
    honest way to show a family's worth is the error it leaves behind, not only
    the accuracy it moves.
    """
    data = load("band_family_ablation_v4.json")
    labels = json.load(open(os.path.join(ASSETS, "band_family_stacks_v4.json")))["labels"]
    scores = data["scores"]

    open_index = CLASS_ORDER.index("Open Land")
    built_index = CLASS_ORDER.index("Built Area")

    def confusion(season, key):
        matrix = data["pooled_matrices"][season][key]
        return matrix[open_index][built_index]

    families = list(labels)
    baseline = {season: scores[season]["gtb:all27"]["overall_accuracy"]
                for season in scores}
    deltas = {
        season: {slug: (scores[season][f"gtb:{slug}"]["overall_accuracy"]
                        - baseline[season]) * 100
                 for slug in families}
        for season in scores
    }
    # Ordered by winter cost, most damaging family at the bottom, matching the
    # reading order of the table it sits beside.
    order = sorted(families, key=lambda slug: -deltas["winter"][slug])

    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 3.15))
    y = np.arange(len(order))[::-1]
    height = 0.36
    for index, (season, colour) in enumerate((("winter", BLUE),
                                              ("summer", VERM))):
        if season not in scores:
            continue
        values = [deltas[season][slug] for slug in order]
        axes[0].barh(y + (index - 0.5) * (height + 0.02), values, height,
                     color=colour, label=season.capitalize(), zorder=3)
        for value, yi in zip(values, y + (index - 0.5) * (height + 0.02)):
            axes[0].text(value + (0.06 if value >= 0 else -0.06), yi,
                         f"{value:+.2f}", va="center",
                         ha="left" if value >= 0 else "right",
                         fontsize=5.8, color=INK)
        counts = [confusion(season, f"gtb:{slug}") for slug in order]
        axes[1].barh(y + (index - 0.5) * (height + 0.02), counts, height,
                     color=colour, zorder=3)
        for value, yi in zip(counts, y + (index - 0.5) * (height + 0.02)):
            axes[1].text(value + 2, yi, f"{value}", va="center", ha="left",
                         fontsize=5.8, color=INK)
        axes[1].axvline(confusion(season, "gtb:all27"), color=colour,
                        linestyle="--", linewidth=0.9, zorder=4)

    axes[0].axvline(0, color=INK, linewidth=0.8, zorder=4)
    left, right = axes[0].get_xlim()
    axes[0].set_xlim(left - (right - left) * 0.10, right + (right - left) * 0.06)
    axes[0].set_yticks(y, [labels[slug] for slug in order], fontsize=6.4)
    axes[0].set_xlabel("change in overall accuracy when the family is removed "
                       "(points)")
    axes[0].set_title("(a) accuracy cost of removing a feature family",
                      loc="left", fontsize=7.4, color=INK, pad=6)
    # Upper right of panel (a): every bar but one runs left of zero, so that
    # corner is the only reliably empty space. Below-left sat on the longest
    # bar's value label and above-centre sat on the panel title.
    axes[0].legend(frameon=False, ncol=1, loc="upper right",
                   handlelength=0.9, fontsize=6.6)
    tidy(axes[0], grid_axis="x")

    axes[1].set_yticks(y, ["" for _ in order])
    axes[1].set_xlabel("open-land samples called built area")
    axes[1].set_title("(b) the error the features exist to fix", loc="left",
                      fontsize=7.4, color=INK, pad=14)
    tidy(axes[1], grid_axis="x")

    fig.tight_layout()
    provenance(
        fig,
        f"Smile GTB on the {len(data['districts'])} development districts, "
        f"five-class public-map consensus. Dashed lines are the full 27-band "
        f"stack.\n"
        f"Source: doc/assets/band_family_ablation_v4.json "
        f"({data['training_schema_version']}).")
    save(fig, "fig_family_removal")


# ── F2c ────────────────────────────────────────────────────────────────────
def fig_band_selection():
    """Leave-one-band-out, and what the candidate stacks do on held-out ground.

    Two questions, two panels. The left one is diagnostic and lives entirely on
    the development districts: which single bands can be removed without loss.
    The right one is the actual result -- the stacks those removals suggest,
    scored once on districts that took no part in suggesting them.
    """
    ablation = load("band_ablation_v4.json")["ablation"]
    stacks = load("band_stack_comparison_v4.json")

    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 3.1),
                             gridspec_kw={"width_ratios": [1.25, 1.0]})

    # (a) per-band deltas, winter, sorted
    rows = sorted(ablation["winter"]["bands"], key=lambda row: row["delta_points"])
    y = np.arange(len(rows))
    colours = [VERM if row["delta_points"] < 0 else BLUE for row in rows]
    axes[0].barh(y, [row["delta_points"] for row in rows], color=colours,
                 height=0.72, zorder=3)
    axes[0].set_yticks(y, [row["band"] for row in rows], fontsize=5.8)
    axes[0].axvline(0, color=INK, linewidth=0.8, zorder=4)
    axes[0].set_xlabel("change in overall accuracy when the band is removed "
                       "(points)")
    axes[0].set_title("(a) leave-one-band-out, winter, development districts",
                      loc="left", fontsize=7.2, color=INK)
    tidy(axes[0], grid_axis="x")

    # (b) candidate stacks, development against test
    names = [name for name in stacks["stacks_order"]]
    x = np.arange(len(names))
    width = 0.38
    for index, (split, colour) in enumerate((("development", SEQ[1]),
                                             ("test", SEQ[3]))):
        values, errs = [], [[], []]
        for name in names:
            entry = stacks["seasons"]["winter"][split][name]
            values.append(entry["overall_accuracy"] * 100)
        axes[1].bar(x + (index - 0.5) * (width + 0.02), values, width,
                    color=colour, label=f"{split} districts", zorder=3)
        for xi, value in zip(x, values):
            axes[1].text(xi + (index - 0.5) * (width + 0.02), value + 0.25,
                         f"{value:.1f}", ha="center", fontsize=5.8, color=INK)
    axes[1].set_xticks(
        x, [f"{name}\n({len(stacks['stacks'][name])})" for name in names],
        fontsize=6.0)
    axes[1].set_ylabel("overall accuracy (%)")
    low = min(stacks["seasons"]["winter"][s][n]["overall_accuracy"] * 100
              for s in ("development", "test") for n in names)
    axes[1].set_ylim(low - 3, None)
    axes[1].legend(frameon=False, ncol=2, loc="upper center",
                   bbox_to_anchor=(0.5, 1.16), handlelength=0.9, fontsize=6.6)
    axes[1].set_title("(b) candidate stacks, winter", loc="left",
                      fontsize=7.2, color=INK)
    tidy(axes[1])

    fig.tight_layout()
    provenance(
        fig,
        "Smile GTB throughout. The stack is chosen on the development "
        "districts and scored once on the test districts.\n"
        "Sources: doc/assets/band_ablation_v4.json, "
        "doc/assets/band_stack_comparison_v4.json.")
    save(fig, "fig_band_selection")


# ── shared: the current statewide external results ─────────────────────────
def v5_runs():
    data = load(versioned(EXTERNAL_RESULTS).name)
    runs = {r["run_id"]: r["metrics"]["external_five_class"] for r in data["runs"]}
    return data, runs


def test_runs():
    """Per-model metrics on the held-out test districts, with district CIs.

    The aggregate above pools all 48 districts, four of which hold training
    points. Figures that sit beside the paper's accuracy tables have to use the
    same population those tables report, or the two disagree on the page.
    """
    data = load(versioned(SPATIAL_UNCERTAINTY).name)
    runs = {}
    for season, payload in data["seasons"].items():
        for key, value in payload["test"]["pooled"].items():
            runs[f"{season}-{key}"] = value
    return data, runs


MODEL_LABEL = {"gtb": "Smile GTB", "svm": "SVM", "knn": "KNN",
               "rf": "Random Forest", "cart": "CART"}


# ── F3 ─────────────────────────────────────────────────────────────────────
def fig_model_comparison():
    """Held-out district accuracy per classifier, both seasons, Kappa labelled."""
    data, runs = test_runs()
    models = sorted(MODEL_LABEL, key=lambda m: -runs[f"winter-after-{m}"]["overall_accuracy"])

    fig, ax = plt.subplots(figsize=(COL, 2.6))
    x = np.arange(len(models))
    width = 0.38
    for i, (season, colour) in enumerate((("winter", BLUE), ("summer", VERM))):
        vals = [runs[f"{season}-after-{m}"]["overall_accuracy"] * 100 for m in models]
        kappa = [runs[f"{season}-after-{m}"]["kappa"] for m in models]
        errs = [[v - runs[f"{season}-after-{m}"]["overall_accuracy_ci95"][0] * 100
                 for v, m in zip(vals, models)],
                [runs[f"{season}-after-{m}"]["overall_accuracy_ci95"][1] * 100 - v
                 for v, m in zip(vals, models)]]
        ax.bar(x + (i - 0.5) * (width + 0.02), vals, width, color=colour,
               label=season.capitalize(), zorder=3,
               yerr=errs, capsize=1.8,
               error_kw={"elinewidth": 0.7, "ecolor": "#3a3a38", "zorder": 5})
        for xi, (v, k) in enumerate(zip(vals, kappa)):
            ax.text(xi + (i - 0.5) * (width + 0.02), v + 3.2, f"{v:.1f}",
                    ha="center", fontsize=6.6, color=INK)
            ax.text(xi + (i - 0.5) * (width + 0.02), v - 7.0, f"$\\kappa$ {k:.2f}",
                    ha="center", fontsize=6.1, color="white")

    ax.set_xticks(x, [MODEL_LABEL[m].replace(" ", "\n") for m in models])
    ax.set_ylabel("overall accuracy (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, 1.13), handlelength=1.1)
    tidy(ax)
    fig.tight_layout()
    winter_n = runs["winter-after-gtb"]["sample_count"]
    summer_n = runs["summer-after-gtb"]["sample_count"]
    provenance(
        fig,
        f"{len(data['district_splits']['test'])} held-out test districts; "
        f"n = {winter_n:,} winter and {summer_n:,} summer.\n"
        f"Whiskers are 95% intervals from resampling whole districts.\n"
        f"Source: {versioned(SPATIAL_UNCERTAINTY).name} "
        f"({data['training_schema_version']}).")
    save(fig, "fig_model_comparison")


# ── F4 ─────────────────────────────────────────────────────────────────────
def fig_per_class_f1():
    """Per-class F1 for every classifier, both seasons, as two heat panels."""
    data, runs = test_runs()
    models = sorted(MODEL_LABEL, key=lambda m: -runs[f"winter-after-{m}"]["overall_accuracy"])
    classes = ["Vegetation", "Water", "Built Area", "Open Land", "Agriculture"]

    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 2.35))
    for ax, season, tag in zip(axes, ("winter", "summer"), ("(a)", "(b)")):
        grid = np.array([[runs[f"{season}-after-{m}"]["per_class_f1"][c] or 0.0
                          for c in classes] for m in models])
        ax.imshow(grid, cmap="Blues", vmin=0.35, vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(classes)), classes)
        ax.set_yticks(range(len(models)), [MODEL_LABEL[m] for m in models])
        ax.set_title(f"{tag} {season.capitalize()}", loc="left", color=INK)
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                ax.text(c, r, f"{grid[r, c]:.2f}", ha="center", va="center",
                        fontsize=7,
                        color="white" if grid[r, c] > 0.78 else INK)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0, colors=MUTED)
        ax.set_xticks(np.arange(-0.5, len(classes), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(models), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.4)
        ax.tick_params(which="minor", length=0)

    fig.tight_layout(w_pad=2.0)
    provenance(
        fig,
        f"F1 against the public-map consensus over "
        f"{len(data['district_splits']['test'])} held-out test districts. "
        f"Darker is better. Source: {versioned(SPATIAL_UNCERTAINTY).name} "
        f"({data['training_schema_version']}).")
    save(fig, "fig_per_class_f1")


# ── F5 ─────────────────────────────────────────────────────────────────────
def fig_before_after():
    """What the v3 inventory bought, per class, on the same reference.

    Reads the pooled aggregate rather than the test-district split: the Before
    condition is a property of the training inventory, not a claim about
    held-out skill, and the split file only carries the After runs.
    """
    data, runs = v5_runs()
    classes = ["Vegetation", "Water", "Built Area", "Open Land", "Agriculture"]
    before = runs["winter-before-gtb"]["per_class"]
    after = runs["winter-after-gtb"]["per_class"]

    # The two conditions are dodged onto their own rows rather than sharing one.
    # Vegetation, Water and Built Area move by at most 0.008, so a shared row
    # draws the marks concentric and the second one simply hides the first.
    fig, ax = plt.subplots(figsize=(COL, 2.35))
    y = np.arange(len(classes))[::-1]
    dodge = 0.17
    for yi, name in zip(y, classes):
        b = before[name]["f1"]
        a = after[name]["f1"]
        if b is None:                      # never predicted under Before
            ax.annotate("", xy=(a - 0.015, yi - dodge), xytext=(0.40, yi + dodge),
                        arrowprops={"arrowstyle": "-|>", "color": GREY,
                                    "linewidth": 0.9, "shrinkA": 0, "shrinkB": 0})
            ax.text(0.40, yi + dodge + 0.20, "never predicted before",
                    ha="left", va="bottom", fontsize=6.2, color=MUTED)
        else:
            ax.plot([b, a], [yi + dodge, yi - dodge], color="#d0d0cb",
                    linewidth=1.4, zorder=2, solid_capstyle="round")
            ax.scatter([b], [yi + dodge], s=24, color=BLUE, zorder=4,
                       edgecolors="white", linewidths=0.8)
            ax.text(b, yi + dodge + 0.20, f"{b:.2f}", ha="center", fontsize=6.2,
                    color=MUTED)
        ax.scatter([a], [yi - dodge], s=24, color=VERM, zorder=5,
                   edgecolors="white", linewidths=0.8)
        ax.text(a, yi - dodge - 0.36, f"{a:.2f}", ha="center", fontsize=6.2,
                color=INK)

    ax.scatter([], [], s=26, color=BLUE, label="Before (soil-era)")
    ax.scatter([], [], s=26, color=VERM, label="After (five-class v3)")
    ax.set_yticks(y, classes)
    ax.set_xlabel("per-class F1, Smile GTB, winter")
    ax.set_xlim(0.30, 1.08)
    ax.set_ylim(-0.75, len(classes) - 0.05)
    ax.legend(frameon=False, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, 1.2), handletextpad=0.2, columnspacing=1.0)
    tidy(ax, grid_axis="x")
    fig.tight_layout()
    provenance(
        fig,
        "Both conditions scored on the identical statewide reference.\n"
        f"Source: {versioned(EXTERNAL_RESULTS).name} "
        f"({data['training_schema_version']}).")
    save(fig, "fig_before_after")


# ── F6 ─────────────────────────────────────────────────────────────────────
def fig_confusion():
    """Row-normalised confusion matrices for the served model, both seasons."""
    data, runs = test_runs()
    classes = ["Vegetation", "Water", "Built Area", "Open Land", "Agri."]

    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 2.45))
    for ax, season, tag in zip(axes, ("winter", "summer"), ("(a)", "(b)")):
        run = runs[f"{season}-after-gtb"]
        matrix = np.array(run["confusion_matrix"], dtype=float)
        pct = matrix / matrix.sum(axis=1, keepdims=True) * 100
        ax.imshow(pct, cmap="Blues", vmin=0, vmax=100, aspect="equal")
        ax.set_xticks(range(5), classes)
        ax.set_yticks(range(5), classes)
        ax.set_xlabel("Predicted")
        if tag == "(a)":
            ax.set_ylabel("Reference")
        ax.set_title(
            f"{tag} {season.capitalize()}   OA {run['overall_accuracy']*100:.1f}%, "
            f"$\\kappa$ {run['kappa']:.3f}", loc="left", color=INK)
        for r in range(5):
            for c in range(5):
                ax.text(c, r, f"{pct[r, c]:.0f}", ha="center", va="center",
                        fontsize=7.5, color="white" if pct[r, c] > 55 else INK)
        for spine in ax.spines.values():
            spine.set_edgecolor("#555555")
        ax.tick_params(length=0, colors=MUTED)

    fig.tight_layout(w_pad=2.4)
    provenance(
        fig,
        "Cells are percentages of the reference row. Source: "
        f"{versioned(EXTERNAL_RESULTS).name} ({data['training_schema_version']}).")
    save(fig, "fig_confusion_seasonal")


# ── F0 ─────────────────────────────────────────────────────────────────────
def fig_pipeline():
    """The method as a picture: two lanes sharing one feature engine.

    Drawn here rather than in TikZ so it is built by the same command, in the
    same face, as every other figure -- and so it cannot silently fail to
    compile on a machine without the paper's TikZ libraries.
    """
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(WIDE, 3.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis("off")

    def box(x, y, w, h, title, lines, accent=False):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.6",
            linewidth=0.7, edgecolor=BLUE if accent else "#8a8a86",
            facecolor="#eef5fa" if accent else "#f6f6f4", zorder=2))
        ax.text(x + 1.4, y + h - 2.4, title, fontsize=7.6, fontweight="bold",
                color=INK, zorder=3, va="top")
        for i, line in enumerate(lines):
            ax.text(x + 1.4, y + h - 5.4 - i * 2.7, line, fontsize=6.2,
                    color=MUTED, zorder=3, va="top")

    def arrow(x0, y0, x1, y1, colour="#6a6a66", label=None, style="-|>"):
        ax.add_patch(FancyArrowPatch(
            (x0, y0), (x1, y1), arrowstyle=style, mutation_scale=7,
            linewidth=0.8, color=colour, shrinkA=0, shrinkB=0, zorder=4))
        if label:
            ax.text((x0 + x1) / 2, max(y0, y1) + 0.9, label, fontsize=5.8,
                    color=MUTED, ha="center", va="bottom", zorder=4)

    train_y, apply_y, valid_y = 37.5, 20.0, 2.0
    height = 10.0

    for y, lane in ((train_y, "TRAIN"), (apply_y, "APPLY"), (valid_y, "VALIDATE")):
        ax.text(1.0, y + height / 2, lane, fontsize=6.2, color=MUTED,
                rotation=90, va="center", ha="center")

    box(4, train_y, 16, height, "5,000 points",
        ["1,000 per class,", "each carrying a label"])
    box(24, train_y, 17, height, "Training composite",
        ["median over the points'", "own bounding box"])
    box(67, train_y, 15, height, "sampleRegions",
        ["scale = 10 m", "null rows dropped"])
    box(86, train_y, 14, height, "Five classifiers",
        ["RF, SVM, Smile GTB,", "CART, KNN"])

    box(4, apply_y, 16, height, "Drawn AOI",
        ["any polygon,", "anywhere"])
    box(24, apply_y, 17, height, "AOI composite",
        ["same mask and median,", "clipped to the polygon"])
    box(67, apply_y, 15, height, "Classify",
        ["+ 3x3 focal mode", "majority smoothing"])
    box(86, apply_y, 14, height, "10 m class raster",
        ["areas, tiled PNG,", "GeoTIFF"])

    # the shared engine, spanning both lanes
    box(45, apply_y, 18, train_y - apply_y + height, "", [], accent=True)
    ax.text(54, train_y + height - 2.3, "Shared feature engine", fontsize=7.4,
            fontweight="bold", color=BLUE, ha="center", va="top", zorder=3)
    ax.text(54, train_y + height - 5.2, "one code path, both lanes",
            fontsize=6.0, color=MUTED, ha="center", va="top", zorder=3)
    groups = [("S2 reflectance", "+3"), ("Bareness indices", "+5"),
              ("GLCM texture", "+6"), ("Sentinel-1 SAR", "+5")]
    for i, (name, count) in enumerate(groups):
        y = train_y + height - 10.0 - i * 3.6
        ax.text(46.4, y, name, fontsize=6.3, color=INK, va="center", zorder=3)
        ax.text(61.6, y, count, fontsize=6.3, color=BLUE, va="center",
                ha="right", zorder=3)
        ax.plot([46.4, 61.6], [y - 1.8, y - 1.8], color=BLUE, linewidth=0.4,
                alpha=0.35, zorder=3)
    ax.text(54, apply_y + 3.6, "19 bands, rescaled 0-1", fontsize=6.5,
            fontweight="bold", color=INK, ha="center", zorder=3)
    ax.text(54, apply_y + 1.6, "on the 10 m grid", fontsize=6.0, color=MUTED,
            ha="center", zorder=3)

    # No labels on these arrows: the gap between boxes is narrower than the
    # words would be, and the engine already names what flows out of it.
    mid_t, mid_a = train_y + height / 2, apply_y + height / 2
    for y in (mid_t, mid_a):
        arrow(20, y, 23.4, y)
        arrow(41, y, 44.4, y)
        arrow(63, y, 66.4, y)
        arrow(82, y, 85.4, y)

    # the trained model is the only thing that crosses from TRAIN to APPLY
    ax.plot([93, 93, 74.5, 74.5], [train_y, 34.2, 34.2, apply_y + height + 1.2],
            color=BLUE, linewidth=0.8, zorder=4, solid_joinstyle="round")
    arrow(74.5, apply_y + height + 1.6, 74.5, apply_y + height + 0.2, colour=BLUE)
    ax.text(83.7, 35.0, "fitted once, applied to any AOI",
            fontsize=5.8, color=BLUE, ha="center", va="bottom")

    # the barrier
    ax.plot([4, 100], [15.4, 15.4], linestyle=(0, (4, 3)), linewidth=0.8,
            color="#9a6a10", zorder=3)
    ax.text(52, 13.6, "the labelled points never cross this line - below it the "
            "model meets only independent data",
            fontsize=6.0, color="#9a6a10", ha="center", va="top", style="italic")

    box(4, valid_y, 25, height, "Reference consensus",
        ["WorldCover and Dynamic", "World agree, p >= 0.70"])
    box(33, valid_y, 22, height, "Specialists confirm",
        ["OPERA DSWx, GHSL,", "WorldCereal"])
    box(59, valid_y, 20, height, "Stratified sample",
        ["20 / class / district,", "minus 100 m buffer"])
    box(83, valid_y, 17, height, "Confusion matrix",
        ["OA, Kappa, F1,", "96 stamped shards"])
    mid_v = valid_y + height / 2
    arrow(29, mid_v, 32.4, mid_v)
    arrow(55, mid_v, 58.4, mid_v)
    arrow(79, mid_v, 82.4, mid_v)
    arrow(93, apply_y, 93, valid_y + height + 0.4)
    ax.text(92, 17.4, "predictions", fontsize=5.8, color=MUTED, ha="right")

    fig.tight_layout(pad=0.1)
    provenance(fig, "Both lanes call the same band-construction code; only the "
                    "trained classifier crosses between them.")
    save(fig, "fig_pipeline")


def main():
    print(f"building report figures into {OUT}")
    fig_pipeline()
    fig_family_removal()
    fig_band_selection()
    fig_model_comparison()
    fig_per_class_f1()
    fig_before_after()
    fig_confusion()


def _self_check():
    """The plots are only as good as the read: assert the numbers behind them."""
    data, runs = v5_runs()
    # Pinning magic numbers here only survives until the next rerun, and then
    # it fails for the one reason that is not a bug. Assert the invariants a
    # misread would actually break instead.
    assert data["training_schema_version"] == TRAINING_SCHEMA_VERSION, (
        f"figures would be built from {data['training_schema_version']} "
        f"results while the code is {TRAINING_SCHEMA_VERSION}")
    assert len(runs) == 20, sorted(runs)
    winter = runs["winter-after-gtb"]
    assert 0.5 < winter["overall_accuracy"] < 1.0, winter["overall_accuracy"]
    assert winter["sample_count"] == runs["winter-after-svm"]["sample_count"]
    assert winter["overall_accuracy"] > runs["winter-before-gtb"]["overall_accuracy"]
    # Agriculture is unpredictable under Before -- the whole point of F5.
    assert runs["winter-before-gtb"]["per_class"]["Agriculture"]["f1"] is None
    # The band artefacts are deliberately *not* checked against the current
    # schema. They are upstream of it: the ablation and the stack comparison
    # ran on the previous stack and are what selected the current one, so their
    # stamp is the older version by construction. Requiring equality here would
    # reject the evidence for the change because of the change. Structure is
    # what matters instead.
    ablation = load("band_ablation_v4.json")
    for season, payload in ablation["ablation"].items():
        assert len(payload["bands"]) == 27, (season, len(payload["bands"]))
    comparison = load("band_stack_comparison_v4.json")
    from scripts.adopt_band_stack import select_stack
    chosen, _ = select_stack(comparison)
    assert comparison["stacks"][chosen] == list(BANDS), (
        f"figures would show {chosen} as selected while config ships a "
        f"different {len(BANDS)}-band stack")
    families = load("band_family_ablation_v4.json")
    assert families["training_schema_version"] == TRAINING_SCHEMA_VERSION, (
        f"the family figure would be built from "
        f"{families['training_schema_version']} while the code is "
        f"{TRAINING_SCHEMA_VERSION}")
    assert "pooled_matrices" in families, "no confusion matrices for panel (b)"
    print("self-check ok")


if __name__ == "__main__":
    import sys
    if "--self-check" in sys.argv:
        _self_check()
    else:
        main()
