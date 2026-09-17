#!/usr/bin/env python3
"""Which reference strata missed the 20-point quota, and whether area explains it.

The summer sample is 208 points short in the test half and every missing point
is vegetation. The paper attributed that to the collapse of the eligible
vegetation area, from 1,999 km2 in winter to 98 km2 in summer. A collapse to
98 km2 is still 980,000 eligible 10 m pixels across 29 districts, so area alone
does not explain a failure to draw twenty pixels per district: the question is
per district, and it separates districts where the consensus really holds fewer
than twenty eligible pixels from districts where it holds thousands and the
stratified sampler still returned fewer than twenty.

Prints and stores both, per district, so the shortfall is attributed rather than
asserted. Reads the frozen archive only; touches no network.
"""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "doc" / "assets" / "result_archive"
QUOTA = 20
PIXELS_PER_KM2 = 10_000  # a 10 m pixel is 100 m2


def strata(path, role="test"):
    with path.open() as handle:
        for row in csv.DictReader(handle):
            if row["role"] == role:
                yield row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--areas", type=Path,
                        default=ARCHIVE / "stratum_areas.csv")
    parser.add_argument("--role", default="test")
    args = parser.parse_args()

    report = {"source": args.areas.name, "role": args.role, "seasons": {}}
    for season in ("winter", "summer"):
        rows = [r for r in strata(args.areas, args.role) if r["season"] == season]
        short = [r for r in rows if int(r["sample_n"]) < QUOTA]
        missing = sum(QUOTA - int(r["sample_n"]) for r in short)
        by_class = {}
        for row in short:
            entry = by_class.setdefault(row["consensus_class"], [])
            area = float(row["eligible_area_km2"])
            entry.append({
                "district": row["district"],
                "eligible_area_km2": round(area, 4),
                "eligible_10m_pixels": int(round(area * PIXELS_PER_KM2)),
                "sampled": int(row["sample_n"]),
            })
        scarce = [d for entries in by_class.values() for d in entries
                  if d["eligible_10m_pixels"] < QUOTA]
        ample = [d for entries in by_class.values() for d in entries
                 if d["eligible_10m_pixels"] >= 1000]
        report["seasons"][season] = {
            "strata": len(rows),
            "strata_below_quota": len(short),
            "missing_points": missing,
            "classes_affected": sorted(by_class),
            "short_strata_with_fewer_than_20_eligible_pixels": len(scarce),
            "short_strata_with_at_least_1000_eligible_pixels": len(ample),
            "detail": by_class,
        }
        print(f"\n{season.upper()}  {len(short)} strata below the {QUOTA}-point "
              f"quota, {missing} points missing, classes "
              f"{sorted(by_class) or 'none'}")
        for name, entries in by_class.items():
            print(f"  {name}:")
            for entry in sorted(entries, key=lambda e: -e["eligible_10m_pixels"]):
                print(f"    {entry['district']:16} eligible "
                      f"{entry['eligible_area_km2']:9.4f} km2 "
                      f"({entry['eligible_10m_pixels']:>9,} pixels)  "
                      f"drawn {entry['sampled']:2}/{QUOTA}")
        if short:
            print(f"  of these, {len(scarce)} strata hold fewer than {QUOTA} "
                  f"eligible pixels and {len(ample)} hold at least 1,000")

    path = ROOT / "doc" / "assets" / "reference_shortfall_audit.json"
    path.write_text(json.dumps(report, indent=1) + "\n")
    print(f"\nwrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
