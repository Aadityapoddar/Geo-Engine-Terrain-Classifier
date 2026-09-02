#!/usr/bin/env python3
"""District-block uncertainty, paired model tests and spatial-transfer decay.

Every statewide number this project reports is a sum over 48 district shards,
and samples inside one district share landscape, phenology and reference-map
error. Treating the pooled 4,787 rows as independent would give a pixel-level
confidence interval that is far too narrow. So the resampling unit here is the
*district*, not the point: a bootstrap replicate draws 48 districts with
replacement, re-sums their confusion matrices, and recomputes the metric.

Model comparison uses the same block structure. Two classifiers scored on the
same district see the same reference points, so the districts are paired and
the informative quantity is the per-district difference, not two independent
means.

Reads the per-district shard files written by scripts/run_sharded_mp_external.py.
No Earth Engine access is needed -- the confusion matrices are already on disk.
"""

import argparse
import json
import math
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from evaluation.metrics import metrics_from_matrix  # noqa: E402
from evaluation.references import REFERENCE_LABELS  # noqa: E402
from evaluation.splits import district_splits  # noqa: E402

CENTROIDS_PATH = REPO / "doc" / "assets" / "mp_district_centroids.json"
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 20260901
LABELS = list(REFERENCE_LABELS.values())
# The band-stack variants are named "<condition>-<model>-<stack>", so the
# 19-band baseline they are compared against is the plain "<condition>-<model>".
STACK_BASE_MODEL_SUFFIX = "-gtb"
STACK_BASELINE = "after-gtb"


def load_shards(paths):
    """Merge worker shard files into {season: {district: {key: matrix}}}."""
    seasons = {}
    schema = None
    for path in paths:
        payload = json.loads(Path(path).read_text())
        recorded = payload.get("training_schema_version")
        if schema is None:
            schema = recorded
        elif schema != recorded:
            raise ValueError(f"{path} holds {recorded}, expected {schema}")
        for shard in payload["shards"].values():
            season = seasons.setdefault(shard["season"], {})
            if shard["district"] in season:
                raise ValueError(f"Duplicate {shard['season']}:{shard['district']}")
            season[shard["district"]] = {
                key: np.array(matrix, dtype=float)
                for key, matrix in shard["matrices"].items()
            }
    return schema, seasons


def _oa(matrix):
    total = matrix.sum()
    return float(np.trace(matrix) / total) if total else math.nan


def _summarise(matrix):
    metrics = metrics_from_matrix(matrix.astype(int).tolist(), LABELS)
    return metrics["overall_accuracy"], metrics["kappa"], metrics["macro_f1"], metrics


def _percentile_ci(values, level=95.0):
    values = np.asarray([v for v in values if v is not None and not math.isnan(v)])
    if values.size == 0:
        return None
    half = (100.0 - level) / 2.0
    return [float(np.percentile(values, half)),
            float(np.percentile(values, 100.0 - half))]


def block_bootstrap(district_matrices, replicates, rng):
    """Resample whole districts and recompute pooled metrics on each replicate."""
    names = sorted(district_matrices)
    stack = np.array([district_matrices[name] for name in names])
    draws = rng.integers(0, len(names), size=(replicates, len(names)))
    oa, kappa, macro_f1 = [], [], []
    per_class_f1 = {label: [] for label in LABELS}
    for draw in draws:
        pooled = stack[draw].sum(axis=0)
        value_oa, value_kappa, value_macro, metrics = _summarise(pooled)
        oa.append(value_oa)
        kappa.append(value_kappa)
        macro_f1.append(value_macro)
        for label in LABELS:
            per_class_f1[label].append(metrics["per_class"][label]["f1"])
    return {
        "overall_accuracy": _percentile_ci(oa),
        "kappa": _percentile_ci(kappa),
        "macro_f1": _percentile_ci(macro_f1),
        "per_class_f1": {label: _percentile_ci(values)
                         for label, values in per_class_f1.items()},
    }


def paired_comparison(district_matrices, left, right, replicates, rng):
    """Compare two classifiers on the districts they both scored."""
    names = sorted(district_matrices)
    differences = np.array([
        _oa(district_matrices[name][left]) - _oa(district_matrices[name][right])
        for name in names
    ])
    draws = rng.integers(0, len(names), size=(replicates, len(names)))
    means = differences[draws].mean(axis=1)
    # Wilcoxon needs at least one non-zero difference; identical models tie
    # every district and the test is undefined rather than merely insignificant.
    if np.any(differences != 0):
        statistic, p_value = stats.wilcoxon(differences)
        p_value = float(p_value)
    else:
        p_value = None
    return {
        "left": left,
        "right": right,
        "mean_district_oa_difference": float(differences.mean()),
        "ci95": _percentile_ci(means),
        "districts_left_better": int((differences > 0).sum()),
        "districts_right_better": int((differences < 0).sum()),
        "districts_tied": int((differences == 0).sum()),
        "wilcoxon_p": p_value,
    }


def _haversine_km(lon1, lat1, lon2, lat2):
    radius = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = phi2 - phi1
    d_lambda = math.radians(lon2 - lon1)
    a = (math.sin(d_phi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2)
    return 2 * radius * math.asin(math.sqrt(a))


def transfer_decay(district_matrices, key, centroids, origin="Jabalpur"):
    """Regress per-district accuracy on distance from the training district."""
    home = centroids[origin]
    rows = []
    for name in sorted(district_matrices):
        if name not in centroids:
            continue
        point = centroids[name]
        rows.append({
            "district": name,
            "distance_km": _haversine_km(
                home["lon"], home["lat"], point["lon"], point["lat"]),
            "overall_accuracy": _oa(district_matrices[name][key]),
            "sample_count": int(district_matrices[name][key].sum()),
        })
    distance = np.array([row["distance_km"] for row in rows])
    accuracy = np.array([row["overall_accuracy"] for row in rows])
    regression = stats.linregress(distance, accuracy)
    spearman = stats.spearmanr(distance, accuracy)
    return {
        "origin": origin,
        "classifier": key,
        "districts": rows,
        "slope_oa_per_100km": float(regression.slope * 100),
        "slope_stderr_per_100km": float(regression.stderr * 100),
        "intercept": float(regression.intercept),
        "r_squared": float(regression.rvalue ** 2),
        "linregress_p": float(regression.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "nearest_quartile_mean_oa": float(
            np.mean(accuracy[np.argsort(distance)[: max(1, len(rows) // 4)]])),
        "farthest_quartile_mean_oa": float(
            np.mean(accuracy[np.argsort(distance)[-max(1, len(rows) // 4):]])),
    }


def class_transfer_decay(district_matrices, key, centroids, origin="Jabalpur"):
    """Which class fails first as distance from the training district grows."""
    home = centroids[origin]
    names = [name for name in sorted(district_matrices) if name in centroids]
    distance = np.array([
        _haversine_km(home["lon"], home["lat"],
                      centroids[name]["lon"], centroids[name]["lat"])
        for name in names
    ])
    output = {}
    for index, label in enumerate(LABELS):
        recall = []
        keep = []
        for position, name in enumerate(names):
            matrix = district_matrices[name][key]
            support = matrix[index].sum()
            if support:
                recall.append(matrix[index][index] / support)
                keep.append(position)
        if len(recall) < 5:
            continue
        subset = distance[keep]
        regression = stats.linregress(subset, np.array(recall))
        output[label] = {
            "districts_with_support": len(recall),
            "slope_recall_per_100km": float(regression.slope * 100),
            "p_value": float(regression.pvalue),
            "mean_recall": float(np.mean(recall)),
        }
    return output


def _pool(districts, keys, replicates, rng):
    """Point estimates plus district-block CIs for every classifier."""
    pooled = {}
    for key in keys:
        matrix = sum(districts[name][key] for name in districts)
        value_oa, value_kappa, value_macro, metrics = _summarise(matrix)
        block = block_bootstrap(districts_for(districts, key), replicates, rng)
        pooled[key] = {
            # Carried so figures drawn beside the accuracy table use the same
            # population it reports, rather than the all-48-district pool.
            "confusion_matrix": matrix.astype(int).tolist(),
            "sample_count": int(matrix.sum()),
            "district_count": len(districts),
            "overall_accuracy": value_oa,
            "overall_accuracy_ci95": block["overall_accuracy"],
            "kappa": value_kappa,
            "kappa_ci95": block["kappa"],
            "macro_f1": value_macro,
            "macro_f1_ci95": block["macro_f1"],
            "per_class_f1": {
                label: metrics["per_class"][label]["f1"] for label in LABELS},
            "per_class_f1_ci95": block["per_class_f1"],
        }
    return pooled


def districts_for(districts, key):
    return {name: matrices[key] for name, matrices in districts.items()}


def run(args):
    schema, seasons = load_shards(args.shards)
    centroids = json.loads(CENTROIDS_PATH.read_text())
    splits = district_splits()
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    output = {
        "training_schema_version": schema,
        "bootstrap_replicates": args.replicates,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "resampling_unit": "district",
        "district_splits": splits,
        "protocol": (
            "Classifier and feature-stack choices are read from the development "
            "districts. The test districts are scored once with whatever those "
            "choices produced and are never consulted before that. Districts "
            "holding training points are excluded from both."
        ),
        "seasons": {},
    }
    for season, all_districts in sorted(seasons.items()):
        keys = sorted(next(iter(all_districts.values())))
        after_keys = [key for key in keys
                      if key.startswith("after-") and key.count("-") == 1]
        stack_keys = [key for key in keys if key.count("-") == 2]
        payload = {}
        for split in ("development", "test"):
            members = [name for name in splits[split] if name in all_districts]
            if not members:
                continue
            districts = {name: all_districts[name] for name in members}
            # The feature-stack variants are compared with the same paired
            # machinery as the classifiers. Whether 19 bands beat 27 is a
            # claim about a difference, and a difference with a confidence
            # interval straddling zero is not a reduction that "improved
            # accuracy" -- it is a reduction that cost nothing measurable.
            payload[split] = {
                "districts": members,
                "pooled": _pool(districts, keys, args.replicates, rng),
                "paired_comparisons": [
                    paired_comparison(districts, left, right,
                                      args.replicates, rng)
                    for left, right in combinations(after_keys, 2)
                ],
                "stack_comparisons": [
                    paired_comparison(districts, left, right,
                                      args.replicates, rng)
                    for left, right in combinations(
                        sorted(stack_keys + [STACK_BASELINE]), 2)
                ],
            }
        # The one decision the development half is allowed to make.
        selected = max(
            after_keys,
            key=lambda key: payload["development"]["pooled"][key]["overall_accuracy"],
        )
        selected_stack = None
        if stack_keys:
            candidates = stack_keys + [
                key for key in after_keys
                if key.endswith(STACK_BASE_MODEL_SUFFIX)]
            selected_stack = max(
                candidates,
                key=lambda key: payload["development"]["pooled"][key]["overall_accuracy"],
            )
        external = {name: all_districts[name] for name in all_districts
                    if name in splits["development"] + splits["test"]}
        payload.update({
            "selected_on_development": selected,
            "selected_band_stack_on_development": selected_stack,
            "reported_on_test": payload["test"]["pooled"][selected],
            "transfer": transfer_decay(external, selected, centroids),
            "class_transfer": class_transfer_decay(external, selected, centroids),
        })
        # Kept for continuity with the pre-split reports, clearly labelled.
        payload["all_48_districts_pooled"] = _pool(
            all_districts, [selected], args.replicates, rng)[selected]
        output["seasons"][season] = payload
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n")

    for season, payload in output["seasons"].items():
        print(f"\n== {season} ==")
        for split in ("development", "test"):
            print(f"  -- {split} ({len(payload[split]['districts'])} districts)")
            for key, value in sorted(payload[split]["pooled"].items()):
                if not key.startswith("after-"):
                    continue
                low, high = value["overall_accuracy_ci95"]
                print(f"    {key:18s} OA {value['overall_accuracy']*100:5.2f}% "
                      f"[{low*100:5.2f}, {high*100:5.2f}]  "
                      f"kappa {value['kappa']:.3f}  "
                      f"macroF1 {value['macro_f1']:.3f}")
        print(f"  selected on development: {payload['selected_on_development']}"
              f"  stack: {payload['selected_band_stack_on_development']}")
        reported = payload["reported_on_test"]
        low, high = reported["overall_accuracy_ci95"]
        print(f"  REPORTED (test, {reported['district_count']} districts, "
              f"n={reported['sample_count']}): OA "
              f"{reported['overall_accuracy']*100:.2f}% "
              f"[{low*100:.2f}, {high*100:.2f}] kappa {reported['kappa']:.3f}")
        transfer = payload["transfer"]
        print(f"  transfer {transfer['classifier']}: "
              f"{transfer['slope_oa_per_100km']*100:+.2f} pp OA per 100 km "
              f"(R2={transfer['r_squared']:.3f}, p={transfer['linregress_p']:.4f}), "
              f"near {transfer['nearest_quartile_mean_oa']*100:.1f}% vs far "
              f"{transfer['farthest_quartile_mean_oa']*100:.1f}%")
        best = payload["selected_on_development"]
        for comparison in payload["test"]["paired_comparisons"]:
            if best not in (comparison["left"], comparison["right"]):
                continue
            # Always print the difference oriented as best-minus-other.
            sign = 1 if comparison["left"] == best else -1
            other = comparison["right"] if sign == 1 else comparison["left"]
            low, high = sorted(value * sign for value in comparison["ci95"])
            print(f"  {best} - {other}: "
                  f"{comparison['mean_district_oa_difference']*sign*100:+.2f} pp "
                  f"[{low*100:+.2f}, {high*100:+.2f}] "
                  f"wilcoxon p={comparison['wilcoxon_p']:.2e}")
        for comparison in payload["test"]["stack_comparisons"]:
            low, high = comparison["ci95"]
            print(f"  stack {comparison['left']} - {comparison['right']}: "
                  f"{comparison['mean_district_oa_difference']*100:+.2f} pp "
                  f"[{low*100:+.2f}, {high*100:+.2f}] "
                  f"wilcoxon p={comparison['wilcoxon_p']:.3f}")
        for label, value in payload["class_transfer"].items():
            print(f"  class {label:12s} recall slope "
                  f"{value['slope_recall_per_100km']*100:+.2f} pp/100km "
                  f"(p={value['p_value']:.3f}, mean {value['mean_recall']*100:.1f}%)")
    print(f"\nwrote {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shards", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=BOOTSTRAP_REPLICATES)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
