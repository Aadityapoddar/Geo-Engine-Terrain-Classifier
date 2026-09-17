#!/usr/bin/env python3
"""What is in the part of the state the reference refuses, and how the map does there.

Every figure in the paper describes the consensus-covered domain, which is 56.5%
of the test districts in winter. The complement is not reported on at all, and
the obvious worry about a reference built from agreement is that the pixels it
drops are the difficult ones -- so the reported accuracy would be an accuracy on
the easy half by construction.

This samples the rejected half directly. At points drawn uniformly from pixels
the consensus assigns to no class, it reads the ESRI 2024 arbiter, which is not
part of the consensus, and the model's own prediction. Two things come out: what
the rejected domain is made of according to an independent product, and whether
the model agrees with that product less often there than on the accepted domain
where every reported figure lives.

The arbiter is another satellite classification, so this bounds nothing. It says
whether the excluded domain looks different, which is the question the paper
could not answer at all before.

Writes doc/assets/rejected_domain_audit.json. Needs Earth Engine.
"""
import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import ee

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backend import config  # noqa: E402
from backend.gee_classifier import (  # noqa: E402
    _add_sar,
    _add_spectral_indices,
    _build_collection,
    init_ee,
    make_classifier,
    merge_feature_collections,
    sample_training_points,
)
from evaluation.references import (  # noqa: E402
    REFERENCE_LABELS,
    build_reference_image,
    madhya_pradesh_districts,
)
from scripts.consensus_arbiter_audit import ARBITER_MAP, arbiter_image  # noqa: E402

ARCHIVE = REPO / "doc" / "assets" / "result_archive"
OUTPUT = REPO / "doc" / "assets" / "rejected_domain_audit.json"
SEED = 20260909


def with_retry(label, call, attempts=6):
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except ee.EEException as error:
            if attempt == attempts:
                raise
            print(f"  retry {label} ({attempt}/{attempts}): {error}", flush=True)
            time.sleep(20 * attempt)


def test_districts(season):
    with (ARCHIVE / "district_split.csv").open() as handle:
        return sorted(row["district"] for row in csv.DictReader(handle)
                      if row["role"] == "test")


def district_sample(district, season, classifier, points):
    """Points on both sides of the consensus mask, with arbiter and prediction.

    Both sides, because the comparison that matters is model-versus-arbiter in
    the rejected domain against model-versus-arbiter in the accepted one. Those
    two are measured the same way here; comparing the first against the stored
    consensus-versus-arbiter figure would compare two different quantities.
    """
    region = ee.Feature(district).geometry()
    dates = config.SEASONS[season]
    reference = build_reference_image(region, season, dates["start"], dates["end"])
    covered = reference.mask().toByte().rename("covered").clip(region)

    collection = _build_collection(region, dates["start"], dates["end"], 15)
    composite = _add_spectral_indices(collection.median())
    composite = _add_sar(composite, region, dates["start"], dates["end"])
    predicted = composite.select(config.BANDS).classify(classifier).rename("predicted")

    stack = covered.addBands(arbiter_image(region)).addBands(predicted)
    return stack.stratifiedSample(
        numPoints=0, classBand="covered", region=region, scale=10,
        classValues=[0, 1], classPoints=[points, points], seed=SEED,
        geometries=False, tileScale=4,
    ).filter(ee.Filter.notNull(["arbiter", "predicted", "covered"]))


def accepted_agreement(season):
    """Model-versus-arbiter agreement on the accepted domain, for comparison.

    Read from the frozen archive rather than re-sampled: these are the same
    points every reported figure uses, and their arbiter labels are already
    stored by the consensus audit.
    """
    audit = json.loads((REPO / "doc" / "assets"
                        / "mp_consensus_arbiter_audit.json").read_text())
    roles = {row["district"]: row["role"] for row in
             csv.DictReader((ARCHIVE / "district_split.csv").open())}
    labels = list(REFERENCE_LABELS.values())
    matrix = [[0] * 5 for _ in range(5)]
    for record in audit["records"].values():
        if record["season"] != season or roles.get(record["district"]) != "test":
            continue
        for i in range(5):
            for j in range(5):
                matrix[i][j] += record["matrix"][i][j]
    total = sum(map(sum, matrix))
    return {"labels": labels, "n": total,
            "consensus_vs_arbiter_agreement_pct":
                round(sum(matrix[i][i] for i in range(5)) / total * 100, 2)}


def run(args):
    init_ee()
    districts = madhya_pradesh_districts()
    report = {"seed": SEED, "points_per_district": args.points, "seasons": {}}

    for season in args.seasons:
        dates = config.SEASONS[season]
        classifier = make_classifier("gtb").train(
            features=sample_training_points(
                merge_feature_collections(config.FEATURE_COLLECTIONS.values()),
                start_date=dates["start"], end_date=dates["end"]),
            classProperty="label", inputProperties=config.BANDS)

        names = test_districts(season)
        if args.districts:
            names = [name for name in names if name in args.districts]
        rows = []
        for index, name in enumerate(names, start=1):
            district = districts.filter(ee.Filter.eq("ADM2_NAME", name)).first()
            sampled = with_retry(f"{season}:{name}", lambda: district_sample(
                district, season, classifier, args.points).getInfo())["features"]
            rows.extend(row["properties"] for row in sampled)
            print(f"  [{index}/{len(names)}] {season} {name}: "
                  f"{len(sampled)} rejected-domain points, running {len(rows)}",
                  flush=True)

        def side(covered):
            chosen = [row for row in rows if int(row["covered"]) == covered]
            arbiter = Counter(int(row["arbiter"]) for row in chosen)
            predicted = Counter(int(row["predicted"]) for row in chosen)
            agree = sum(int(row["arbiter"]) == int(row["predicted"])
                        for row in chosen)
            return {
                "points": len(chosen),
                "arbiter_class_share_pct": {
                    REFERENCE_LABELS[value]: round(count / len(chosen) * 100, 2)
                    for value, count in sorted(arbiter.items())},
                "model_class_share_pct": {
                    REFERENCE_LABELS[value]: round(count / len(chosen) * 100, 2)
                    for value, count in sorted(predicted.items())},
                "model_vs_arbiter_agreement_pct":
                    round(agree / len(chosen) * 100, 2) if chosen else None,
            }

        report["seasons"][season] = {
            "districts": len(names),
            "rejected": side(0),
            "accepted": side(1),
            "consensus_vs_arbiter_agreement_pct_accepted":
                accepted_agreement(season)["consensus_vs_arbiter_agreement_pct"],
        }
        entry = report["seasons"][season]
        entry["agreement_gap_pp"] = round(
            entry["accepted"]["model_vs_arbiter_agreement_pct"]
            - entry["rejected"]["model_vs_arbiter_agreement_pct"], 2)
        print(f"\n{season}: {len(names)} test districts, "
              f"{entry['rejected']['points']} points the consensus rejects and "
              f"{entry['accepted']['points']} it accepts")
        for name in ("rejected", "accepted"):
            print(f"  {name}: arbiter says " + ", ".join(
                f"{cls} {share}%" for cls, share in
                entry[name]["arbiter_class_share_pct"].items()))
            print(f"  {name}: the map says " + ", ".join(
                f"{cls} {share}%" for cls, share in
                entry[name]["model_class_share_pct"].items()))
        print(f"  map agrees with the arbiter on "
              f"{entry['rejected']['model_vs_arbiter_agreement_pct']}% of the "
              f"rejected domain against "
              f"{entry['accepted']['model_vs_arbiter_agreement_pct']}% of the "
              f"accepted one ({entry['agreement_gap_pp']:+} pp)", flush=True)

    OUTPUT.write_text(json.dumps(report, indent=1) + "\n")
    print(f"\nwrote {OUTPUT.relative_to(REPO)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", nargs="+", default=["winter", "summer"],
                        choices=sorted(config.SEASONS))
    parser.add_argument("--points", type=int, default=20)
    parser.add_argument("--districts", nargs="*", default=None)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
