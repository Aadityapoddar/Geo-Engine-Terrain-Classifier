#!/usr/bin/env python3
"""Turn the leave-one-out ablation into candidate stacks worth testing.

The ablation says what each band is worth when removed on its own. It does not
say what happens when several are removed together, and it cannot: dropping two
correlated bands that each looked harmless can cost more than either did alone.
So the ablation proposes and a second pass disposes -- the candidates written
here are scored district by district, on development first and on the held-out
test districts only once the choice is frozen.

Candidates:

    all27         every band, the reference point
    drop_winter   drop the bands whose removal did not hurt winter
    drop_summer   the same for summer
    drop_both     drop only bands that were droppable in both seasons, which is
                  the conservative reading and the one most likely to survive
    b22, b19      the two historical cuts, carried along so the new protocol
                  can say whether the old choice was right

Written as {name: [bands]} so scripts/district_experiments.py --mode stacks can
read it directly.
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backend.config import BAND_STACKS, FULL_BANDS  # noqa: E402

DEFAULT_INPUT = REPO / "doc" / "assets" / "band_ablation_v4.json"
DEFAULT_OUTPUT = REPO / "doc" / "assets" / "band_selection_v4.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    payload = json.loads(args.ablation.read_text())
    ablation = payload.get("ablation")
    if not ablation:
        raise SystemExit(f"{args.ablation} carries no ablation summary yet")

    droppable = {season: set(value["droppable_bands"])
                 for season, value in ablation.items()}
    both = set.intersection(*droppable.values()) if len(droppable) > 1 \
        else set(next(iter(droppable.values())))

    stacks = {"all27": list(FULL_BANDS)}
    for season, bands in droppable.items():
        keep = [band for band in FULL_BANDS if band not in bands]
        if keep and len(keep) < len(FULL_BANDS):
            stacks[f"drop_{season}"] = keep
    keep_both = [band for band in FULL_BANDS if band not in both]
    if keep_both and len(keep_both) < len(FULL_BANDS):
        stacks["drop_both"] = keep_both
    stacks["b22"] = list(BAND_STACKS["b22"])
    stacks["b19"] = list(BAND_STACKS["b19"])

    # Two candidates that select the same bands are the same experiment.
    unique, seen = {}, {}
    for name, bands in stacks.items():
        signature = tuple(bands)
        if signature in seen:
            print(f"  {name} is identical to {seen[signature]}, dropped")
            continue
        seen[signature] = name
        unique[name] = bands

    output = {
        "source": str(args.ablation.name),
        "training_schema_version": payload.get("training_schema_version"),
        "droppable_by_season": {season: sorted(bands)
                                for season, bands in droppable.items()},
        "droppable_in_both": sorted(both),
        "stacks": unique,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    for name, bands in unique.items():
        print(f"  {name:14s} {len(bands):2d} bands")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
