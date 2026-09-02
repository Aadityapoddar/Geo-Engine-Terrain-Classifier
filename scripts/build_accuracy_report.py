"""Bake district_eval results + sites into the maximalism report webpage.

    python scripts/build_accuracy_report.py

Reads doc/assets/district_eval_results.json (written by district_eval.py run)
and doc/assets/district_eval_sites.json (district_eval.py sites), injects them
into scripts/accuracy_report_template.html and writes a standalone
doc/mp_accuracy_report.html with every number inline - no fetch, opens offline.
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.config import LAND_COVER_CLASSES, MODEL_METADATA

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ASSETS = os.path.join(REPO, "doc", "assets")
RESULTS_PATH = os.path.join(ASSETS, "district_eval_results.json")
SITES_PATH = os.path.join(ASSETS, "district_eval_sites.json")
TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "accuracy_report_template.html")
OUTPUT_PATH = os.path.join(REPO, "doc", "mp_accuracy_report.html")

TRAIN_COMPOSITION = [
    {"name": "Vegetation", "n": 1000},
    {"name": "Water", "n": 1000},
    {"name": "Built Area", "n": 1000},
    {"name": "Open Land", "n": 2000},
    {"name": "Sand", "n": 500},
]


def main():
    if not os.path.exists(RESULTS_PATH):
        raise SystemExit(
            f"{RESULTS_PATH} not found - run `python scripts/district_eval.py run` first.")

    with open(RESULTS_PATH) as fh:
        results = json.load(fh)

    sites = None
    if os.path.exists(SITES_PATH):
        with open(SITES_PATH) as fh:
            sites = json.load(fh)

    class_colors = [LAND_COVER_CLASSES[i]["color"] for i in results["class_ids"]]
    model_names = {m: MODEL_METADATA[m]["name"] for m in results["models"]
                   if m in MODEL_METADATA}

    payload = {
        "results": results,
        "sites": sites,
        "class_colors": class_colors,
        "model_names": model_names,
        "train_composition": TRAIN_COMPOSITION,
    }

    with open(TEMPLATE_PATH) as fh:
        html = fh.read()

    marker = "window.REPORT_DATA = null;"
    if marker not in html:
        raise SystemExit("template marker not found")
    html = html.replace(marker, "window.REPORT_DATA = " + json.dumps(payload) + ";")

    with open(OUTPUT_PATH, "w") as fh:
        fh.write(html)

    n = results.get("n_train")
    summary = {
        "models": results["models"],
        "districts": len(results["districts"]),
        "n_train": n,
        "n_eval": results.get("n_eval"),
        "wrote": OUTPUT_PATH,
    }
    print(json.dumps(summary, indent=2))
    if sites:
        print(json.dumps({"sites": list(sites.get("sites", {}).keys())}, indent=2))


if __name__ == "__main__":
    main()
