#!/usr/bin/env python3
"""Draw the district split and the reference domain it is assessed over.

A list of totals -- four withheld, fifteen development, twenty-nine test --
does not let a reader check the partition, see whether the development half is
a corner of the state, or judge how much of each district the consensus
reference actually covers. This draws all of it on one map from the same two
records the evaluation uses: evaluation.splits for the roles and
mp_reference_stratum_areas.json for the covered area, with the labelled
training points plotted where they physically are rather than where the text
says they are.

Panel (a) is the partition with training points overlaid; panel (b) shades each
district by the fraction of its area the winter consensus assigns to a class at
all, which is the domain every area-weighted number in the paper describes and
the complement of the domain no reported figure assesses.

Writes figures/fig_split_map.png and the machine-readable
doc/assets/result_archive/district_split.csv (via build_result_archive.py).
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Polygon as MplPolygon  # noqa: E402

from evaluation.splits import TRAINING_DISTRICTS, district_splits  # noqa: E402

ASSETS = REPO / "doc" / "assets"
GEOMETRY_PATH = ASSETS / "mp_district_geometries.json"
POINTS_PATH = ASSETS / "training_point_coordinates.json"
OUT = REPO / "figures" / "fig_split_map.png"

GAUL_LEVEL2 = "FAO/GAUL/2015/level2"
STATE = "Madhya Pradesh"
SIMPLIFY_M = 500

ROLE_STYLE = {
    "training-withheld": {"face": "#d9534f", "label": "withheld: holds training labels (4)"},
    "development": {"face": "#f0ad4e", "label": "development: every choice made here (15)"},
    "test": {"face": "#5bc0de", "label": "test: scored once, all reported figures (29)"},
}


def fetch_geometries():
    """Simplified GAUL ADM2 outlines for the state. Fetched once, then cached."""
    if GEOMETRY_PATH.exists():
        return json.loads(GEOMETRY_PATH.read_text())
    import ee
    from backend.gee_classifier import init_ee
    init_ee()
    collection = (ee.FeatureCollection(GAUL_LEVEL2)
                  .filter(ee.Filter.eq("ADM1_NAME", STATE))
                  .map(lambda f: ee.Feature(f.geometry().simplify(SIMPLIFY_M))
                       .copyProperties(f, ["ADM2_NAME", "ADM2_CODE"])))
    payload = collection.getInfo()
    geometries = {
        feature["properties"]["ADM2_NAME"]: {
            "adm2_code": feature["properties"].get("ADM2_CODE"),
            "geometry": feature["geometry"],
        }
        for feature in payload["features"]
    }
    GEOMETRY_PATH.write_text(json.dumps(geometries) + "\n")
    print(f"wrote {GEOMETRY_PATH.relative_to(REPO)} ({len(geometries)} districts)")
    return geometries


def fetch_points():
    """Coordinates of the labelled training points, per class."""
    if POINTS_PATH.exists():
        return json.loads(POINTS_PATH.read_text())
    import ee
    from backend.gee_classifier import init_ee
    from evaluation.assets import training_points
    init_ee()
    collection = training_points("after")
    coordinates = collection.geometry().coordinates().getInfo()
    POINTS_PATH.write_text(json.dumps(coordinates) + "\n")
    print(f"wrote {POINTS_PATH.relative_to(REPO)} ({len(coordinates)} points)")
    return coordinates


def rings(geometry):
    """Every exterior ring of a polygonal geometry, as coordinate lists.

    Simplifying a GAUL district can leave a GeometryCollection when the outline
    degenerates to a line or point somewhere, so recurse rather than assume.
    """
    kind = geometry["type"]
    if kind == "Polygon":
        return [geometry["coordinates"][0]]
    if kind == "MultiPolygon":
        return [polygon[0] for polygon in geometry["coordinates"]]
    if kind == "GeometryCollection":
        return [ring for part in geometry["geometries"] for ring in rings(part)]
    return []


def coverage_fraction(season="winter"):
    strata = json.loads((ASSETS / "mp_reference_stratum_areas.json").read_text())
    fractions = {}
    for record in strata["strata"].values():
        if record["season"] != season:
            continue
        covered = sum(value or 0.0 for value in record["class_area_m2"].values())
        fractions[record["district"]] = covered / record["district_area_m2"]
    return fractions


def draw(geometries, points, splits, fractions):
    role = {name: "test" for name in splits["test"]}
    role.update({name: "development" for name in splits["development"]})
    role.update({name: "training-withheld" for name in TRAINING_DISTRICTS})

    figure, axes = plt.subplots(1, 2, figsize=(13.2, 6.4), dpi=300)

    for name, record in sorted(geometries.items()):
        style = ROLE_STYLE.get(role.get(name), {"face": "#eeeeee"})
        for ring in rings(record["geometry"]):
            axes[0].add_patch(MplPolygon(ring, closed=True,
                                         facecolor=style["face"],
                                         edgecolor="white", linewidth=0.5))
    axes[0].scatter([p[0] for p in points], [p[1] for p in points],
                    s=0.6, color="#111111", alpha=0.55, linewidths=0,
                    zorder=5)
    axes[0].set_title("(a) Partition, fixed before any selection", fontsize=10)
    handles = [Patch(facecolor=style["face"], edgecolor="white", label=style["label"])
               for style in ROLE_STYLE.values()]
    # 5,000 labelled features, 4,998 distinct coordinates: two are exact
    # duplicates, and 4,988 distinct cells survive on the 10 m analysis grid.
    handles.append(Line2D([], [], marker="o", linestyle="none", color="#111111",
                          markersize=3,
                          label=f"labelled training points "
                                f"(5,000 labels, {len(points):,} distinct locations)"))
    axes[0].legend(handles=handles, loc="lower left", fontsize=7.2, frameon=False)

    normalise = Normalize(vmin=0.0, vmax=1.0)
    colours = plt.get_cmap("YlGnBu")
    for name, record in sorted(geometries.items()):
        fraction = fractions.get(name)
        face = colours(normalise(fraction)) if fraction is not None else "#eeeeee"
        for ring in rings(record["geometry"]):
            axes[1].add_patch(MplPolygon(ring, closed=True, facecolor=face,
                                         edgecolor="white", linewidth=0.5))
    axes[1].set_title("(b) Share of each district the winter consensus labels at all",
                      fontsize=10)
    bar = figure.colorbar(plt.cm.ScalarMappable(norm=normalise, cmap=colours),
                          ax=axes[1], fraction=0.035, pad=0.02)
    bar.set_label("consensus-covered fraction of district area", fontsize=8)
    bar.ax.tick_params(labelsize=7)

    for axis in axes:
        axis.set_aspect("equal")
        axis.autoscale_view()
        axis.set_xlabel("Longitude (deg E)", fontsize=8)
        axis.set_ylabel("Latitude (deg N)", fontsize=8)
        axis.tick_params(labelsize=7)

    figure.tight_layout()
    figure.savefig(OUT, dpi=300, facecolor="white", bbox_inches="tight")
    print(f"wrote {OUT.relative_to(REPO)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", default="winter")
    args = parser.parse_args()
    draw(fetch_geometries(), fetch_points(), district_splits(),
         coverage_fraction(args.season))


if __name__ == "__main__":
    main()
