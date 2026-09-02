#!/usr/bin/env python3
"""Promote the stack the development districts chose into production config.

The selection lives in doc/assets/band_stack_comparison_v4.json and the shipped
stack lives in backend/config.py, and the two disagreeing is exactly the drift
this revision exists to remove. Rather than hand-editing the band list, this
reads the winner and rewrites BANDS and TRAINING_SCHEMA_VERSION to match.

Changing BANDS changes what every classifier is trained on, so every statewide
result measured against the old stack becomes incomparable. The schema version
is bumped for that reason: shard files refuse to merge across the boundary, and
the refusal is the point.

    python scripts/adopt_band_stack.py --dry-run
    python scripts/adopt_band_stack.py
"""

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

CONFIG = REPO / "backend" / "config.py"
DEFAULT_COMPARISON = REPO / "doc" / "assets" / "band_stack_comparison_v4.json"


def format_band_list(bands, indent=9):
    """Wrap the band list the way the rest of config.py is written."""
    lines, current = [], []
    for band in bands:
        entry = f'"{band}",'
        if sum(len(item) + 1 for item in current) + len(entry) + indent > 79:
            lines.append(" ".join(current))
            current = []
        current.append(entry)
    if current:
        lines.append(" ".join(current))
    pad = " " * indent
    body = ("\n" + pad).join(lines).rstrip(",")
    return f"BANDS = [{body}]"


def select_stack(comparison, rule="mean", season="winter"):
    """The stack the development districts choose, and why.

    Defined here so the adoption and the test that guards it cannot disagree
    about the rule -- which they did once, when the rule moved from one season
    to both and only one of the two places was updated.

    Returns (name, {stack: mean development OA}).
    """
    seasons = comparison["seasons"]
    if rule == "season":
        return seasons[season]["selected_on_development"], {}
    scores = {}
    for name in comparison["stacks_order"]:
        values = [payload["development"][name]["overall_accuracy"]
                  for payload in seasons.values()
                  if name in payload["development"]]
        if values:
            scores[name] = sum(values) / len(values)
    return max(scores, key=scores.get), scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument(
        "--rule", default="mean", choices=("mean", "season"),
        help="'mean' ranks by development OA averaged over both reported "
             "seasons; 'season' uses one season only")
    parser.add_argument("--season", default="winter",
                        help="which season, when --rule season")
    parser.add_argument("--schema", help="new TRAINING_SCHEMA_VERSION")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    comparison = json.loads(args.comparison.read_text())
    seasons = comparison["seasons"]

    # Selection is on development districts only, in both rules. The default
    # averages the two seasons because the paper reports both: picking on one
    # season alone is an arbitrary choice that the other season may not share,
    # and here it does not -- winter development prefers a 25-band stack while
    # summer development prefers a different 19-band one.
    chosen, scores = select_stack(comparison, args.rule, args.season)
    if scores:
        print(f"mean development OA across {len(seasons)} seasons:")
        for name, value in sorted(scores.items(), key=lambda item: -item[1]):
            mark = "  <-" if name == chosen else ""
            print(f"    {name:14s} {value * 100:6.2f}%{mark}")
        per_season = {season: payload["selected_on_development"]
                      for season, payload in seasons.items()}
        if len(set(per_season.values())) > 1:
            print(f"  note: the seasons disagree on their own "
                  f"({per_season}); the mean rule settles it")
    bands = comparison["stacks"][chosen]

    from backend.config import BANDS as current
    print(f"selected on the development districts by --rule {args.rule}: "
          f"{chosen} ({len(bands)} bands)")
    for season, payload in seasons.items():
        print(f"  {season:6s} development "
              f"{payload['development'][chosen]['overall_accuracy'] * 100:.2f}%"
              f"   test "
              f"{payload['test'][chosen]['overall_accuracy'] * 100:.2f}%")
    if list(current) == list(bands):
        print("config already ships this stack; nothing to do")
        return

    added = [band for band in bands if band not in current]
    removed = [band for band in current if band not in bands]
    print(f"  adds:    {', '.join(added) or 'none'}")
    print(f"  removes: {', '.join(removed) or 'none'}")

    schema = (args.schema
              or f"five-class-{len(bands)}-band-v5-{chosen.replace('_', '-')}")
    if args.dry_run:
        print(f"  would set TRAINING_SCHEMA_VERSION = {schema!r}")
        return

    source = CONFIG.read_text()
    source, count = re.subn(
        r"BANDS = \[[^\]]*\]", format_band_list(bands), source, count=1)
    if count != 1:
        raise SystemExit("could not locate the BANDS assignment in config.py")
    source, count = re.subn(
        r'TRAINING_SCHEMA_VERSION = "[^"]*"',
        f'TRAINING_SCHEMA_VERSION = "{schema}"', source, count=1)
    if count != 1:
        raise SystemExit("could not locate TRAINING_SCHEMA_VERSION")
    CONFIG.write_text(source)
    print(f"wrote {CONFIG}: {len(bands)} bands, schema {schema}")


if __name__ == "__main__":
    main()
