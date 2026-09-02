#!/usr/bin/env python3
"""Rewrite new.tex from the measured artefacts, then it can be compiled.

The LaTeX source is the manuscript proper; 4.docx is a conversion of an earlier
render of it. Editing the source is the right place to make these corrections,
because a table here is a table rather than a scatter of text-box fragments.

Every table this touches is regenerated whole from doc/assets/, and every
sentence that quotes a number is replaced with one that quotes the current
number. Anything the artefacts do not cover is left exactly as written.

    python scripts/update_paper_tex.py --tex new.tex
    tectonic new.tex

Idempotent: the second run finds nothing left to match.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from evaluation.artefacts import (  # noqa: E402
    AREA_ADJUSTED,
    SPATIAL_UNCERTAINTY,
    versioned,
)
from evaluation.splits import district_splits  # noqa: E402

ASSETS = REPO / "doc" / "assets"
FIGURES = REPO / "figures"

MODEL_ROW = [
    ("after-gtb", "Smile GTB"),
    ("after-svm", "SVM (RBF)"),
    ("after-knn", "KNN"),
    ("after-rf", "RF"),
    ("after-cart", "CART"),
]
F1_ROW = [
    ("after-gtb", "Smile GTB"),
    ("after-rf", "Random Forest"),
    ("after-knn", "KNN"),
    ("after-cart", "CART"),
    ("after-svm", "SVM"),
]
CLASSES = ["Vegetation", "Water", "Built Area", "Open Land", "Agriculture"]

# Figures superseded by regenerated versions. The captions in new.tex identify
# each one unambiguously, so the mapping is by content and not by guesswork.
FIGURE_UPDATES = {
    "fig5_statewide_accuracy.png": "report_figs/fig_model_comparison.png",
    "fig6.png": "report_figs/fig_per_class_f1.png",
    "fig7.png": "report_figs/fig_confusion_seasonal.png",
    "fig8.png": "report_figs/fig_before_after.png",
    # Both band figures came from the retired four-class WorldCover tile study
    # and are replaced wholesale by the v4 district ablation.
    "fig4_feature_ablation.png": "report_figs/fig_family_removal.png",
    "fig10_band_reduction.png": "report_figs/fig_band_selection.png",
}

GROUP_LABEL = [
    ("g10_baseline", "Optical + core indices", 10),
    ("g15_bareness", "+ bareness/built indices", 15),
    ("g21_texture", "+ optical GLCM texture", 21),
    ("g27_sar", "+ Sentinel-1 and SAR texture", 27),
]


def load(name, required=True):
    path = ASSETS / name
    if not path.exists():
        if required:
            raise SystemExit(f"missing {path}")
        return None
    return json.loads(path.read_text())


def pct(value, digits=2):
    return "n/a" if value is None else f"{value * 100:.{digits}f}\\%"


def _sorted_models(pooled):
    return sorted(
        (key for key, _ in MODEL_ROW),
        key=lambda key: -pooled[key]["overall_accuracy"],
    )


def accuracy_table(uncertainty, splits):
    label = dict(MODEL_ROW)
    lines = []
    for season in ("winter", "summer"):
        pooled = uncertainty["seasons"][season]["test"]["pooled"]
        head = season.capitalize()
        for key in _sorted_models(pooled):
            value = pooled[key]
            low, high = value["overall_accuracy_ci95"]
            lines.append(
                f"{head:<7}& {label[key]:<10}& {value['overall_accuracy'] * 100:.2f}\\% "
                f"& [{low * 100:.2f}, {high * 100:.2f}] & {value['kappa']:.3f} \\\\")
            head = ""
        if season == "winter":
            lines.append("\\midrule")
    winter = uncertainty["seasons"]["winter"]["test"]["pooled"]["after-gtb"]
    summer = uncertainty["seasons"]["summer"]["test"]["pooled"]["after-gtb"]
    body = "\n".join(lines)
    return f"""\\begin{{table}}[!htbp]
\\caption{{External Accuracy over the {len(splits['test'])} Held-Out Test Districts of Madhya Pradesh ({winter['sample_count']:,} Winter and {summer['sample_count']:,} Summer Reference Samples)}}
\\label{{tab:statewide_accuracy}}
\\centering
\\setlength{{\\tabcolsep}}{{3.5pt}}
\\begin{{tabular}}{{llrlr}}
\\toprule
\\textbf{{Season}} & \\textbf{{Model}} & \\textbf{{OA}} & \\textbf{{95\\% CI\\textsuperscript{{*}}}} & \\textbf{{Kappa}} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}

\\vspace{{2pt}}
\\parbox{{\\columnwidth}}{{\\textsuperscript{{*}}\\footnotesize Percentile interval over {uncertainty['bootstrap_replicates']:,} bootstrap replicates that resample whole districts, not points. Samples inside one district share landscape, phenology and reference-map error, so a pixel-level interval would be far too narrow.}}
\\end{{table}}"""


def f1_table(uncertainty, splits):
    lines = []
    for season in ("winter", "summer"):
        pooled = uncertainty["seasons"][season]["test"]["pooled"]
        head = season.capitalize()
        for key, name in F1_ROW:
            values = pooled[key]["per_class_f1"]
            cells = " & ".join(
                "n/a" if values.get(c) is None else f"{values[c]:.3f}"
                for c in CLASSES)
            lines.append(f"{head:<7}& {name:<14}& {cells} \\\\")
            head = ""
        if season == "winter":
            lines.append("\\midrule")
    body = "\n".join(lines)
    return f"""\\begin{{table}}[!htbp]
\\caption{{Per-Class F1 over the {len(splits['test'])} Held-Out Test Districts}}
\\begin{{center}}
\\begin{{tabular}}{{lllllll}}
\\toprule
\\textbf{{Season}} & \\textbf{{Model}} & \\textbf{{Veg.}} & \\textbf{{Water}} & \\textbf{{Built}} & \\textbf{{Open}} & \\textbf{{Agri.}} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{center}}
\\end{{table}}"""


def family_table(ablation, splits):
    """Table VIII: what each feature family adds, under the v4 protocol."""
    scores = ablation["scores"]
    rows = []
    for season in ("winter", "summer"):
        if season not in scores:
            continue
        table = scores[season]
        head = season.capitalize()
        for key, label, count in GROUP_LABEL:
            entry = table.get(f"gtb:{key}")
            if not entry:
                continue
            recall = entry["per_class_recall"].get("Open Land")
            precision = entry["per_class_precision"].get("Built Area")
            rows.append(
                f"{head:<7}& {label:<28}& {count} & "
                f"{entry['overall_accuracy'] * 100:.1f}\\% & "
                f"{pct(recall, 1)} & {pct(precision, 1)} \\\\")
            head = ""
        if season == "winter":
            rows.append("\\midrule")
    body = "\n".join(rows)
    return f"""\\begin{{table}}[!htbp]
\\caption{{Feature-Family Attribution, Smile GTB on the {len(ablation['districts'])} Development Districts. Families are Cumulative.}}
\\begin{{center}}
\\setlength{{\\tabcolsep}}{{3pt}}
\\begin{{tabular}}{{llrrrr}}
\\toprule
\\textbf{{Season}} & \\textbf{{Feature set}} & \\textbf{{Bands}} & \\textbf{{OA}} & \\textbf{{Open rec.}} & \\textbf{{Built prec.}} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{center}}
\\end{{table}}"""


def protocol_table(uncertainty, protocols, splits):
    winter = uncertainty["seasons"]["winter"]["test"]["pooled"]
    values = [winter[key]["overall_accuracy"] for key, _ in MODEL_ROW]
    external = f"{min(values) * 100:.1f}--{max(values) * 100:.1f}\\%"
    rows = []
    if protocols:
        for name, label in (("random", "Random held-out split"),
                            ("spatially_blocked", "Spatially blocked split")):
            entry = protocols["seasons"]["winter"]["protocols"][name]
            low, high = entry["overall_accuracy_range"]
            rows.append(f"{label:<30} & Jabalpur labelled points"
                        f"{'':<14} & {low * 100:.1f}--{high * 100:.1f}\\% \\\\")
    rows.append(
        f"{'External public-map consensus':<30} & "
        f"{len(splits['test'])} held-out test districts & {external} \\\\")
    body = "\n".join(rows)
    return f"""\\begin{{table}}[!htbp]
\\caption{{Overall Accuracy under Three Validation Protocols, All Measured on the Same Winter Composite}}
\\centering
\\begin{{tabular}}{{p{{2.6cm}}p{{3.1cm}}r}}
\\toprule
\\textbf{{Protocol}} & \\textbf{{Scope}} & \\textbf{{Overall}} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}

\\vspace{{2pt}}
\\parbox{{\\columnwidth}}{{\\footnotesize Ranges span the five classifiers. Each step removes a way for the test set to resemble the training set, and each one costs accuracy.}}
\\end{{table}}"""


def stack_table(comparison, splits):
    """Table XII: candidate stacks, chosen on development and scored on test."""
    rows = []
    for season in ("winter", "summer"):
        payload = comparison["seasons"].get(season)
        if not payload:
            continue
        head = season.capitalize()
        for name in comparison["stacks_order"]:
            if name not in payload["development"]:
                continue
            development = payload["development"][name]
            test = payload["test"][name]
            low, high = test["overall_accuracy_ci95"]
            rows.append(
                f"{head:<7}& \\texttt{{{name.replace('_', chr(92) + '_')}}} & "
                f"{len(comparison['stacks'][name])} & "
                f"{development['overall_accuracy'] * 100:.2f}\\% & "
                f"{test['overall_accuracy'] * 100:.2f}\\% & "
                f"[{low * 100:.2f}, {high * 100:.2f}] \\\\")
            head = ""
        if season == "winter":
            rows.append("\\midrule")
    body = "\n".join(rows)
    return f"""\\begin{{table}}[!htbp]
\\caption{{Candidate Feature Stacks, Smile GTB. Each is Chosen on the {len(splits['development'])} Development Districts and Scored Once on the {len(splits['test'])} Test Districts.}}
\\begin{{center}}
\\setlength{{\\tabcolsep}}{{3.5pt}}
\\begin{{tabular}}{{llrrrl}}
\\toprule
\\textbf{{Season}} & \\textbf{{Stack}} & \\textbf{{Bands}} & \\textbf{{Development OA}} & \\textbf{{Test OA}} & \\textbf{{Test 95\\% CI}} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{center}}
\\end{{table}}"""


def manifest_table(uncertainty, provenance, splits):
    winter = uncertainty["seasons"]["winter"]
    summer = uncertainty["seasons"]["summer"]
    rows = [
        ("Jabalpur-region training",
         f"{provenance['total_points']:,}",
         "Model training; 1,000 per class; 100\\,m buffer from every test point. "
         "Influenced model choice by construction."),
        ("Development, winter",
         f"{winter['development']['pooled']['after-gtb']['sample_count']:,}",
         f"Feature stack, classifier and hyperparameter choice; "
         f"{len(splits['development'])} districts."),
        ("Development, summer",
         f"{summer['development']['pooled']['after-gtb']['sample_count']:,}",
         f"As above; {len(splits['development'])} districts."),
        ("Test, winter",
         f"{winter['test']['pooled']['after-gtb']['sample_count']:,}",
         f"Final winter reporting, scored once; {len(splits['test'])} districts. "
         "No influence on any choice."),
        ("Test, summer",
         f"{summer['test']['pooled']['after-gtb']['sample_count']:,}",
         f"Final summer reporting, scored once; {len(splits['test'])} districts. "
         "No influence on any choice."),
        ("Withheld districts", "--",
         "Jabalpur, Katni, Dindori and Mandla contain labelled training points "
         "and are excluded from all reported accuracy."),
    ]
    body = "\n".join(f"{name} & {count} & {note} \\\\" for name, count, note in rows)
    return f"""\\begin{{table}}[htbp]
\\caption{{Dataset and Split Manifest}}
\\begin{{center}}
\\begin{{tabular}}{{p{{2.4cm}}rp{{3.7cm}}}}
\\toprule
\\textbf{{Dataset}} & \\textbf{{Count}} & \\textbf{{Purpose / Notes}} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\label{{tab_splits}}
\\end{{center}}
\\end{{table}}"""


def replace_block(text, start_marker, end_marker, replacement, label):
    """Swap one \\begin{table}...\\end{table} block, matched by a unique line."""
    start = text.find(start_marker)
    if start == -1:
        print(f"  skip {label}: anchor not found")
        return text
    begin = text.rfind("\\begin{table}", 0, start)
    if begin == -1:
        begin = text.rfind("\\begin{table*}", 0, start)
    end = text.find(end_marker, start)
    if begin == -1 or end == -1:
        print(f"  skip {label}: block boundaries not found")
        return text
    end += len(end_marker)
    print(f"  replaced {label}")
    return text[:begin] + replacement + text[end:]


def update_figures():
    """Point the superseded figure files at the regenerated plots."""
    FIGURES.mkdir(exist_ok=True)
    for target, source in FIGURE_UPDATES.items():
        path = ASSETS / source
        if not path.exists():
            print(f"  skip figure {target}: {source} missing")
            continue
        shutil.copyfile(path, FIGURES / target)
        print(f"  figure {target} <- {source}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tex", type=Path, default=REPO / "new.tex")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    uncertainty = load(versioned(SPATIAL_UNCERTAINTY).name)
    area = load(versioned(AREA_ADJUSTED).name)
    depth = load("mp_composite_depth.json")
    provenance = load("training_label_provenance.json")
    arbiter = load("mp_consensus_arbiter_audit.json", required=False)
    protocols = load("jabalpur_split_protocols.json", required=False)
    ablation = load("band_ablation_v4.json", required=False)
    comparison = load("band_stack_comparison_v4.json", required=False)
    splits = district_splits()

    text = args.tex.read_text()
    if "\\graphicspath" not in text:
        # Figures are extracted from the manuscript into figures/ by this
        # script, so the source has to know to look there. tectonic does not
        # honour TEXINPUTS the way latexmk does.
        text = text.replace(
            "\\usepackage{graphicx}",
            "\\usepackage{graphicx}\n"
            "% Figures are collected into one directory by\n"
            "% scripts/update_paper_tex.py, so this compiles from a clean checkout.\n"
            "\\graphicspath{{figures/}}")
        print("added \\graphicspath")
    print("tables:")
    text = replace_block(
        text, "\\caption{Dataset and Split Manifest}", "\\end{table}",
        manifest_table(uncertainty, provenance, splits), "VII manifest")
    text = replace_block(
        text, "\\label{tab:statewide_accuracy}", "\\end{table}",
        accuracy_table(uncertainty, splits), "IX accuracy")
    text = replace_block(
        text, "\\caption{Per-Class F1 Against the Statewide Public-Map Consensus}",
        "\\end{table}", f1_table(uncertainty, splits), "X per-class F1")
    text = replace_block(
        text, "\\caption{Overall Accuracy under Three Validation Protocols}",
        "\\end{table}", protocol_table(uncertainty, protocols, splits),
        "XI protocols")
    if ablation and "scores" in ablation:
        text = replace_block(
            text, "\\caption{Feature-Group Attribution (Random Forest, 1,274 Statewide Points). Last Column: Open-Land Samples Called Built Area.}",
            "\\end{table}", family_table(ablation, splits),
            "VIII feature families")
    if comparison:
        text = replace_block(
            text, "\\caption{Effect of Reducing the Feature Stack on 5,760 Independent Evaluation Points (Mean over Seeds)}",
            "\\end{table}", stack_table(comparison, splits), "XII band stacks")

    from scripts.paper_tex_prose import tex_replacements
    pairs = tex_replacements(uncertainty, area, depth, provenance, arbiter,
                             protocols, ablation, comparison)
    print("prose:")
    applied = 0
    for old, new in sorted(pairs, key=lambda pair: -len(pair[0])):
        if old in text:
            text = text.replace(old, new)
            applied += 1
        else:
            print(f"  MISS {old[:70]!r}")
    print(f"  applied {applied}/{len(pairs)}")

    args.tex.write_text(text)
    if not args.no_figures:
        print("figures:")
        update_figures()
    print(f"wrote {args.tex}")


if __name__ == "__main__":
    main()
