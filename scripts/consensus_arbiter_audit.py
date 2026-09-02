#!/usr/bin/env python3
"""Audit the public-map consensus against a product that is not part of it.

The statewide reference is a consensus of WorldCover, Dynamic World, GHSL,
OPERA DSWx and WorldCereal. Calling agreement with it "accuracy" assumes the
consensus is right, and it cannot be checked using its own members: those five
products share sensors, and in places share training data, so their errors are
correlated and a majority of them can be confidently wrong together.

So bring in an arbiter that contributed nothing to the consensus: the ESRI 10 m
Annual Land Cover time series, an independent Sentinel-2 classification whose
2024 epoch is also the closest in time to the 2025 composites -- WorldCover is
2021, WorldCereal 2021, GHSL 2018.

What this measures is the consensus's own reliability, expressed as a confusion
matrix against the arbiter at exactly the points the accuracy assessment uses.
It is emphatically not field verification: ESRI's product is another satellite
classification with its own errors, and where the two disagree this cannot say
which one is wrong. What it can do is put a number on how much of the reported
model error is really reference error, which is currently assumed to be zero.
"""

import argparse
import json
import os
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import ee

SOCKET_TIMEOUT_SECONDS = 600
RETRYABLE_ERRORS = (ee.EEException, OSError)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backend.config import SEASONS, TRAINING_SCHEMA_VERSION  # noqa: E402
from backend.gee_classifier import init_ee  # noqa: E402
from evaluation.metrics import metrics_from_matrix  # noqa: E402
from evaluation.references import (  # noqa: E402
    REFERENCE_LABELS,
    build_reference_image,
    madhya_pradesh_districts,
)

ARBITER_ID = "projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS"
ARBITER_YEAR = 2024
# ESRI class codes -> the five-class reference taxonomy. Rangeland (11) and
# flooded vegetation (4) have no honest counterpart here -- rangeland spans
# grass, scrub and degraded open ground, which this ontology splits between
# Forest, Bare and Agriculture -- so those pixels are left unmapped and simply
# do not participate, rather than being forced into a class and counted as
# disagreement.
ARBITER_MAP = {1: 1, 2: 0, 5: 4, 7: 2, 8: 3}
BLOCK_SEED = 20260807
POINTS_PER_CLASS = 20
DEFAULT_OUTPUT = REPO / "doc" / "assets" / "mp_consensus_arbiter_audit.json"


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def _call_with_retry(label, function, retries):
    error = None
    for attempt in range(1, retries + 1):
        try:
            return function()
        except RETRYABLE_ERRORS as caught:
            error = caught
            print(f"retry {label} attempt={attempt}: {caught}", flush=True)
            time.sleep(5 * attempt)
    raise error


def arbiter_image(region):
    start = f"{ARBITER_YEAR}-01-01"
    end = f"{ARBITER_YEAR + 1}-01-01"
    mosaic = (
        ee.ImageCollection(ARBITER_ID)
        .filterBounds(region)
        .filterDate(start, end)
        .mosaic()
        .select("b1")
    )
    remapped = mosaic.remap(
        list(ARBITER_MAP), list(ARBITER_MAP.values())).rename("arbiter")
    # remap leaves unlisted codes unmasked-but-null in some builds; force the
    # mask so unmapped ESRI classes drop out instead of arriving as zeros and
    # masquerading as Forest.
    keep = mosaic.eq(ee.Image.constant(list(ARBITER_MAP))).reduce(ee.Reducer.max())
    return remapped.updateMask(keep)


def district_matrix(district, name, season):
    dates = SEASONS[season]
    region = district.geometry()
    reference = build_reference_image(
        region, season, dates["start"], dates["end"])
    samples = reference.addBands(arbiter_image(region)).stratifiedSample(
        numPoints=0,
        classBand="reference",
        region=region,
        scale=10,
        classValues=list(REFERENCE_LABELS),
        classPoints=[POINTS_PER_CLASS] * len(REFERENCE_LABELS),
        seed=BLOCK_SEED,
        geometries=False,
        tileScale=4,
    ).filter(ee.Filter.notNull(["reference", "arbiter"]))
    matrix = samples.errorMatrix(
        "reference", "arbiter", list(REFERENCE_LABELS)).array().toList()
    return ee.Dictionary({
        "matrix": matrix,
        "sample_count": samples.size(),
    }).getInfo()


def summarise(state):
    labels = list(REFERENCE_LABELS.values())
    summary = {}
    for season in SEASONS:
        matrix = [[0] * len(labels) for _ in labels]
        rows = [record for record in state["records"].values()
                if record["season"] == season]
        for record in rows:
            for row in range(len(labels)):
                for column in range(len(labels)):
                    matrix[row][column] += record["matrix"][row][column]
        metrics = metrics_from_matrix(matrix, labels)
        metrics["district_count"] = len(rows)
        summary[season] = metrics
        print(f"\n{season}: consensus vs {ARBITER_YEAR} arbiter over "
              f"{len(rows)} districts, {metrics['sample_count']} points "
              f"with an arbiter label")
        print(f"  agreement {metrics['overall_accuracy'] * 100:.2f}%  "
              f"kappa {metrics['kappa']:.3f}")
        for label in labels:
            per = metrics["per_class"][label]
            recall = per["recall"]
            print(f"    {label:12s} consensus n={per['support']:5d}  "
                  f"arbiter agrees "
                  f"{'n/a' if recall is None else f'{recall * 100:5.1f}%'}")
    state["summary"] = summary
    return state


def run(args):
    socket.setdefaulttimeout(SOCKET_TIMEOUT_SECONDS)
    _call_with_retry("initialize", init_ee, args.retries)
    districts = madhya_pradesh_districts()
    names = _call_with_retry(
        "district-list",
        lambda: districts.aggregate_array("ADM2_NAME").sort().getInfo(),
        args.retries,
    )
    # Same striping as the other district runners, so this can share the
    # machine with them instead of queueing behind them.
    names = [name for index, name in enumerate(names)
             if index % args.worker_count == args.worker_index]
    output = args.output
    state = json.loads(output.read_text()) if output.exists() else {
        "training_schema_version": TRAINING_SCHEMA_VERSION,
        "arbiter": ARBITER_ID,
        "arbiter_year": ARBITER_YEAR,
        "arbiter_class_map": {str(k): v for k, v in ARBITER_MAP.items()},
        "records": {},
    }
    for season in SEASONS:
        for name in names:
            key = f"{season}:{name}"
            if key in state["records"]:
                continue
            district = districts.filter(ee.Filter.eq("ADM2_NAME", name)).first()
            payload = _call_with_retry(
                key, lambda: district_matrix(district, name, season),
                args.retries)
            payload.update({"season": season, "district": name})
            state["records"][key] = payload
            state["updated_at"] = datetime.now(timezone.utc).isoformat()
            _write_json(output, state)
            matrix = payload["matrix"]
            total = sum(sum(row) for row in matrix)
            agree = sum(matrix[i][i] for i in range(len(matrix)))
            print(f"landed {key} n={total:4d} agreement="
                  f"{(agree / total * 100) if total else float('nan'):5.1f}%",
                  flush=True)
    # A worker holds a quarter of the districts, so summarising here would
    # describe a quarter of the evidence. --merge does it once they are back
    # together.
    if args.worker_count == 1:
        state = summarise(state)
    _write_json(output, state)
    print(f"\nwrote {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summarise-only", action="store_true")
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--merge", nargs="+", type=Path,
                        help="worker files to union into --output, then summarise")
    args = parser.parse_args()
    if not 0 <= args.worker_index < args.worker_count:
        parser.error("worker-index must be in [0, worker-count)")
    if args.merge:
        merged = None
        for path in sorted(args.merge):
            payload = json.loads(path.read_text())
            if merged is None:
                merged = {key: value for key, value in payload.items()
                          if key != "records"}
                merged["records"] = {}
            if payload.get("arbiter_year") != merged.get("arbiter_year"):
                raise SystemExit(f"{path} audits a different arbiter epoch")
            merged["records"].update(payload["records"])
        _write_json(args.output, summarise(merged))
        print(f"\nwrote {args.output}: {len(merged['records'])} districts")
        return
    if args.summarise_only:
        _write_json(args.output, summarise(json.loads(args.output.read_text())))
        return
    run(args)


if __name__ == "__main__":
    main()
