#!/usr/bin/env python3
"""Copy the measured test-district accuracies into the dashboard's badges.

MODEL_BENCHMARKS is what the served UI shows beside each model name, so it is
an accuracy claim made to a user. Typing it out by hand after a rerun is how
that claim goes stale while still looking authoritative, and
tests/test_model_benchmarks.py exists to catch exactly that. This does the copy
so the test has nothing to catch.

Only the held-out test districts are used. The development districts chose the
stack and the hyperparameters, so their scores are a maximum over a search and
have no business on a badge.

    python scripts/sync_model_benchmarks.py --dry-run
"""

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from evaluation.artefacts import SPATIAL_UNCERTAINTY, versioned  # noqa: E402
from evaluation.runner import MODEL_NAMES  # noqa: E402

CONFIG = REPO / "backend" / "config.py"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path,
                        default=versioned(SPATIAL_UNCERTAINTY))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    payload = json.loads(args.results.read_text())
    values = {}
    for model in MODEL_NAMES:
        values[model] = {
            season: (payload["seasons"][season]["test"]["pooled"]
                     [f"after-{model}"]["overall_accuracy"] * 100)
            for season in ("winter", "summer")
        }

    block = "MODEL_BENCHMARKS = {\n" + "".join(
        f'    "{model}": {{"winter": {value["winter"]!r}, '
        f'"summer": {value["summer"]!r}}},\n'
        for model, value in values.items()) + "}\n"

    for model, value in values.items():
        print(f"  {model:5s} winter {value['winter']:6.2f}%  "
              f"summer {value['summer']:6.2f}%")
    if args.dry_run:
        return

    source = CONFIG.read_text()
    source, count = re.subn(r"MODEL_BENCHMARKS = \{.*?\n\}\n", block, source,
                            count=1, flags=re.S)
    if count != 1:
        raise SystemExit("could not locate MODEL_BENCHMARKS in config.py")

    districts = len(payload["district_splits"]["test"])
    source = re.sub(
        r'"scope": \([^)]*\)',
        f'"scope": ("{districts} held-out MP test districts, FAO GAUL 2015 "\n'
        f'                  "level-2 vintage; public-map consensus, not field "\n'
        f'                  "ground truth")',
        source, count=1)
    CONFIG.write_text(source)
    print(f"wrote {CONFIG} from {args.results.name}")


if __name__ == "__main__":
    main()
