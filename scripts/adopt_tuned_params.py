#!/usr/bin/env python3
"""Promote the settings the development districts chose into production config.

The hyperparameter search writes what it found to doc/assets/; the dashboard,
the notebooks and the statewide comparison all read
backend.config.MODEL_METADATA. Copying between the two by hand is how a paper
ends up reporting settings the code does not use, so this does the copy and
rewrites the human-readable description to match.

Selection is on the development districts only, and by design the search never
saw the test half, so the comparison that follows is still an estimate rather
than a maximum over a search.

    python scripts/adopt_tuned_params.py --dry-run
    python scripts/adopt_tuned_params.py --season winter
"""

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

CONFIG = REPO / "backend" / "config.py"
DEFAULT_TUNING = REPO / "doc" / "assets" / "development_tuning.json"

# How each model's settings read in the served description.
DESCRIPTIONS = {
    "rf": lambda p: (f"{p['numberOfTrees']} decision trees with bagging "
                     f"fraction {p['bagFraction']}"
                     + (f" and max nodes {p['maxNodes']}."
                        if p.get("maxNodes") else " and no node cap.")),
    "svm": lambda p: (f"{p['kernelType']} kernel with C={p['cost']} and "
                      f"gamma={p['gamma']}."),
    "gtb": lambda p: (f"{p['numberOfTrees']} gradient-boosted trees with "
                      f"shrinkage rate {p['shrinkage']}"
                      + (f" and max nodes {p['maxNodes']}."
                         if p.get("maxNodes") else " and no node cap.")
                      + " Via ee.Classifier.smileGradientTreeBoost."),
    "cart": lambda p: ("Classification and Regression Tree with "
                       + (f"max nodes {p['maxNodes']}."
                          if p.get("maxNodes") else "no node cap.")),
    "knn": lambda p: f"K={p['k']} nearest neighbours over the feature space.",
}

# Known parameter names, longest first so "kernelType" is matched before any
# shorter name that happens to be a prefix. A greedy [A-Za-z]+ would swallow
# the value of a string parameter and yield {"kernelTypeRBF": ""}.
TYPES = {"numberOfTrees": int, "minLeafPopulation": int, "maxNodes": int,
         "bagFraction": float, "shrinkage": float, "kernelType": str,
         "gamma": float, "cost": float, "k": int}
NAMES = sorted(TYPES, key=len, reverse=True)


def parse_config_id(config_id):
    """"gtb:maxNodes50_numberOfTrees300_shrinkage0.05" -> dict of settings."""
    _, tail = config_id.split(":", 1)
    settings = {}
    for part in tail.split("_"):
        name = next((known for known in NAMES if part.startswith(known)), None)
        if name is None:
            continue
        value = part[len(name):]
        if value in ("", "None"):
            # The grid writes an unset node cap as None, which Earth Engine
            # must never see as an argument; it means "no cap".
            settings[name] = None
            continue
        settings[name] = TYPES[name](value)
    return settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tuning", type=Path, default=DEFAULT_TUNING)
    parser.add_argument("--season", default="winter")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    payload = json.loads(args.tuning.read_text())
    best = payload.get("best_per_model", {}).get(args.season)
    if not best:
        raise SystemExit(f"{args.tuning} has no best_per_model for "
                         f"{args.season}; run the search first")

    from backend.config import MODEL_METADATA
    source = CONFIG.read_text()
    # Anchor inside MODEL_METADATA. Searching the whole file for '"rf": {'
    # finds MODEL_BENCHMARKS first, and the non-greedy run to the next
    # '"params": {' then lands in whichever model happens to come first in the
    # metadata -- which is how every model's tuned settings ended up written
    # into Random Forest's block.
    head, marker, tail = source.partition("MODEL_METADATA = {")
    if not marker:
        raise SystemExit("could not find MODEL_METADATA in config.py")
    changed = 0
    for model, entry in sorted(best.items()):
        settings = parse_config_id(entry["config"])
        # kernelType is categorical and the grid never varies it away from RBF.
        current = dict(MODEL_METADATA[model]["params"])
        merged = {**current, **settings}
        if merged == current:
            print(f"  {model:5s} unchanged")
            continue
        print(f"  {model:5s} {current} -> {merged}  "
              f"(dev OA {entry['overall_accuracy'] * 100:.2f}%)")
        changed += 1
        if args.dry_run:
            continue

        block = "{\n" + "".join(
            f"            {key!r}: {value!r},\n" for key, value in merged.items()
        ) + "        }"
        pattern = (r'("' + re.escape(model) + r'": \{.*?"params": )\{[^}]*\}')
        tail, count = re.subn(pattern, lambda m: m.group(1) + block,
                              tail, count=1, flags=re.S)
        if count != 1:
            raise SystemExit(f"could not rewrite params for {model}")
        description = DESCRIPTIONS[model](merged)
        pattern = (r'("' + re.escape(model) +
                   r'": \{\s*"name": "[^"]*",\s*"type": "[^"]*",\s*'
                   r'"description": )\(?[^,]*?\)?(,\s*"benchmark")')
        tail, count = re.subn(
            pattern, lambda m: m.group(1) + repr(description) + m.group(2),
            tail, count=1, flags=re.S)
        if count != 1:
            print(f"    note: left {model}'s description text unchanged")

    if args.dry_run or not changed:
        print("no file written" if args.dry_run else "config already current")
        return
    CONFIG.write_text(head + marker + tail)
    print(f"wrote {CONFIG}: {changed} model(s) retuned on the "
          f"{args.season} development districts")


if __name__ == "__main__":
    main()
