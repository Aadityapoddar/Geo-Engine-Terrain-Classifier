#!/usr/bin/env python3
"""Regenerate the Jabalpur map plate and its analytics from the actual raster.

The published version of this figure carried three claims that its own pipeline
does not support. It named the classifier "Gradient Boosted Trees (XGBoost),
100 trees" when the model is Earth Engine's Smile gradient tree boosting with
300 trees; it printed a 556,920 ha total against a district polygon of roughly
402,000 ha, because class areas were counted as pixels times a nominal 100 m2
rather than measured with ee.Image.pixelArea in a projection where 10 m means
10 m; and it put the 29-district external accuracy in a corner of a
single-district map with nothing to say the two describe different ground.

So the plate is rebuilt rather than relabelled:

  * the AOI is the FAO GAUL 2015 level-2 feature actually used everywhere else,
    identified by its ADM2 code, and its geodesic area is measured, not quoted;
  * the classifier is the frozen production configuration read from
    backend.config, so the caption cannot drift from the model;
  * class areas come from ee.Image.pixelArea summed under the same mask that
    produced the map -- pixelArea gives each pixel's true ground area, so the
    total is right whatever grid the reduction runs on -- and are reported for
    the raw classification and for the majority-filtered raster the dashboard
    delivers, because only the raw one has a measured agreement figure;
  * the analytical grid and the web-display projection are stated separately;
  * no external accuracy appears on the plate.

Writes doc/assets/jabalpur_plate_metadata.json and figures/fig1_jabalpur_map.png.
"""

import argparse
import io
import json
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import ee  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

from scripts.build_paper_figure_plates import load_outline, locator  # noqa: E402

from backend.config import (  # noqa: E402
    BANDS,
    CLASS_PALETTE,
    LAND_COVER_CLASSES,
    MODEL_METADATA,
    SEASONS,
)
from backend.gee_classifier import (  # noqa: E402
    build_sentinel_composite,
    get_trained_classifier,
    init_ee,
)

ASSETS = REPO / "doc" / "assets"
FIGURES = REPO / "figures"
METADATA_PATH = ASSETS / "jabalpur_plate_metadata.json"
PLATE_PATH = FIGURES / "fig1_jabalpur_map.png"

GAUL_LEVEL2 = "FAO/GAUL/2015/level2"
DISTRICT = "Jabalpur"
STATE = "Madhya Pradesh"
MODEL = "gtb"
SEASON = "winter"
CLOUD_THRESHOLD = 15
# The analytical grid is the imagery's own UTM projection, where 10 m is 10 m
# on the ground; the web overlay is served in Web Mercator, where a 10 m pixel
# at 23.2 N covers 18% more map area than ground area. Counting pixels and
# multiplying by a nominal 100 m2 in the display projection is what inflated
# the previously published class areas, so the two grids are kept apart and
# every area is measured with ee.Image.pixelArea on the analytical one.
DISPLAY_CRS = "EPSG:3857"  # Web Mercator, display only
# The map is produced on a 10 m grid and the areas belong there. No single
# reduction over the whole district returns at 10 m -- not the grouped
# pixelArea sum, not five masked sums, not a frequency histogram, with or
# without tileScale -- because evaluating a 19-band classifier carrying two
# GLCM stacks over roughly forty million pixels exceeds what one interactive
# request will do. That is a limit on the request, not on the computation, and
# this project already answers it the same way everywhere else: shard it. The
# district is cut into a grid, each cell is reduced on its own, and the counts
# are summed. Each cell is then about the size of an AOI the dashboard already
# serves at this scale.
SCALE_M = 10
# Cell size in degrees. At this latitude 0.1 deg is roughly 11 km, so a cell is
# of order 100 km2 and about a million pixels.
TILE_DEGREES = 0.1


def district_feature():
    """The exact GAUL feature every other number in the paper is computed on."""
    collection = (
        ee.FeatureCollection(GAUL_LEVEL2)
        .filter(ee.Filter.eq("ADM1_NAME", STATE))
        .filter(ee.Filter.eq("ADM2_NAME", DISTRICT))
    )
    count = collection.size().getInfo()
    if count != 1:
        raise SystemExit(f"expected exactly one GAUL feature, found {count}")
    return ee.Feature(collection.first())


def with_retry(label, call, attempts=10):
    """Retry past Earth Engine's Restricted Mode concurrency limit."""
    delay = 30
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except ee.ee_exception.EEException as error:
            if "Too Many Requests" not in str(error) and \
                    "concurrency" not in str(error):
                raise
            print(f"    {label}: rate limited, retry {attempt} in {delay}s",
                  flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 300)
    raise RuntimeError(f"{label}: still rate limited after {attempts} attempts")


def tiles(geometry, west, south, east, north):
    """The district cut into a grid of cells, each small enough to reduce."""
    cells = []
    latitude = south
    while latitude < north:
        longitude = west
        while longitude < east:
            cell = ee.Geometry.Rectangle(
                [longitude, latitude,
                 min(longitude + TILE_DEGREES, east),
                 min(latitude + TILE_DEGREES, north)], None, False)
            cells.append(cell.intersection(geometry, maxError=1))
            longitude += TILE_DEGREES
        latitude += TILE_DEGREES
    return cells


def class_areas(classified, geometry, analysis_crs, bounds, label=""):
    """Class areas on the 10 m classification grid, accumulated over tiles.

    Two quantities per tile, each cheap where a whole-district version is not.
    A frequency histogram gives exact pixel counts per class; a pixelArea sum
    gives the true ground area those pixels cover, so the total is anchored to
    a measured area rather than to an assumed 100 square metres per pixel.
    Empty tiles -- the grid is rectangular and the district is not -- return
    nothing and are skipped.
    """
    cells = tiles(geometry, bounds["west"], bounds["south"],
                  bounds["east"], bounds["north"])
    counts = {index: 0.0 for index in LAND_COVER_CLASSES}
    total_area_m2 = 0.0
    done = 0
    for number, cell in enumerate(cells, start=1):
        def histogram_cell(cell=cell):
            return classified.reduceRegion(
                reducer=ee.Reducer.frequencyHistogram(),
                geometry=cell, scale=SCALE_M, crs=analysis_crs,
                maxPixels=1e9, tileScale=4).get("classification").getInfo()

        def area_cell(cell=cell):
            return (ee.Image.pixelArea().updateMask(classified.mask())
                    .reduceRegion(reducer=ee.Reducer.sum(), geometry=cell,
                                  scale=SCALE_M, crs=analysis_crs,
                                  maxPixels=1e9, tileScale=4)
                    .get("area").getInfo())

        histogram = with_retry(f"{label}tile {number}/{len(cells)} counts",
                               histogram_cell)
        if not histogram:
            continue
        area = with_retry(f"{label}tile {number}/{len(cells)} area", area_cell)
        for key, value in histogram.items():
            counts[int(float(key))] += value
        total_area_m2 += area or 0.0
        done += 1
        print(f"    {label}tile {number}/{len(cells)}: "
              f"{sum(histogram.values()):>9,.0f} px, "
              f"{(area or 0) / 1e6:7.2f} km2", flush=True)

    total_count = sum(counts.values())
    areas = {}
    print(f"    {label}{done} non-empty tiles, {total_count:,.0f} pixels, "
          f"{total_area_m2 / 1e6:.2f} km2", flush=True)
    for index, meta in sorted(LAND_COVER_CLASSES.items()):
        share = counts[index] / total_count if total_count else 0.0
        areas[meta["name"]] = share * total_area_m2 / 1e6
        print(f"      {meta['name']:12s} {areas[meta['name']]:9.2f} km2 "
              f"({100 * share:5.2f}%)", flush=True)
    return areas


def thumbnail(image, geometry, dimensions, vis):
    url = image.getThumbURL({
        "region": geometry,
        "dimensions": dimensions,
        "format": "png",
        "crs": DISPLAY_CRS,
        **vis,
    })
    with urllib.request.urlopen(url, timeout=600) as handle:
        return Image.open(io.BytesIO(handle.read())).convert("RGBA")


def compute():
    init_ee()
    feature = district_feature()
    geometry = feature.geometry()
    properties = feature.toDictionary().getInfo()

    dates = SEASONS[SEASON]
    classifier = get_trained_classifier(
        MODEL, dates["start"], dates["end"], CLOUD_THRESHOLD,
        allow_temporal_fallback=False)
    composite = build_sentinel_composite(
        geometry=geometry,
        start_date=dates["start"],
        end_date=dates["end"],
        cloud_threshold=CLOUD_THRESHOLD,
        allow_temporal_fallback=False,
    )
    raw = composite.select(BANDS).classify(classifier).clip(geometry)
    filtered = raw.focalMode(radius=1, kernelType="square", units="pixels") \
        .clip(geometry)

    # The imagery's own grid, read off the composite rather than assumed.
    analysis_crs = composite.select("B4").projection().crs().getInfo()
    geodesic_area_km2 = geometry.area(maxError=1).getInfo() / 1e6
    bounds = geometry.bounds(maxError=1).coordinates().getInfo()[0]
    longitudes = [point[0] for point in bounds]
    latitudes = [point[1] for point in bounds]

    box = {"west": min(longitudes), "east": max(longitudes),
           "south": min(latitudes), "north": max(latitudes)}
    metadata = {
        "aoi": {
            "source": GAUL_LEVEL2,
            "adm2_name": properties.get("ADM2_NAME"),
            "adm2_code": properties.get("ADM2_CODE"),
            "adm1_name": properties.get("ADM1_NAME"),
            "adm0_name": properties.get("ADM0_NAME"),
            "geodesic_area_km2": round(geodesic_area_km2, 2),
            "bounds_west": min(longitudes), "bounds_east": max(longitudes),
            "bounds_south": min(latitudes), "bounds_north": max(latitudes),
        },
        "composite": {
            "season": SEASON,
            "date_filter": f"{dates['start']} to {dates['end']} (half-open)",
            "cloud_threshold_percent": CLOUD_THRESHOLD,
            "optical": "COPERNICUS/S2_SR_HARMONIZED, per-pixel median",
            "radar": "COPERNICUS/S1_GRD IW descending, per-pixel median",
            "temporal_fallback": False,
        },
        "model": {
            "constructor": "ee.Classifier.smileGradientTreeBoost",
            "parameters": MODEL_METADATA[MODEL]["params"],
            "unset_parameters_use_earth_engine_defaults":
                "samplingRate 0.7, seed 0",
            "feature_stack": f"sel19 ({len(BANDS)} bands)",
            "bands": list(BANDS),
        },
        "grids": {
            "analysis_crs": analysis_crs,
            "analysis_scale_m": SCALE_M,
            "tiling_degrees": TILE_DEGREES,
            "display_crs": DISPLAY_CRS,
            "area_method": "ee.Image.pixelArea summed by class group",
        },
        "class_areas_km2": {
            "raw": class_areas(raw, geometry, analysis_crs, box, "raw "),
            "majority_filtered": class_areas(filtered, geometry, analysis_crs,
                                             box, "filtered "),
        },
    }
    for key, areas in metadata["class_areas_km2"].items():
        metadata.setdefault("class_area_totals_km2", {})[key] = round(
            sum(areas.values()), 2)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"wrote {METADATA_PATH.relative_to(REPO)}")

    print(f"  GAUL geodesic area      {geodesic_area_km2:10.2f} km2")
    for key, total in metadata["class_area_totals_km2"].items():
        print(f"  pixelArea total ({key:17s}) {total:10.2f} km2")

    print("  rendering plate ...", flush=True)
    rgb = thumbnail(composite, geometry, 1800,
                    {"bands": ["B4", "B3", "B2"], "min": 0, "max": 0.3})
    overlay = thumbnail(raw, geometry, 1800,
                        {"min": 0, "max": 4, "palette": CLASS_PALETTE})
    rgb.save(ASSETS / "jabalpur_plate_rgb.png")
    overlay.save(ASSETS / "jabalpur_plate_class.png")
    return metadata


def draw(metadata):
    rgb = Image.open(ASSETS / "jabalpur_plate_rgb.png").convert("RGBA")
    overlay = Image.open(ASSETS / "jabalpur_plate_class.png").convert("RGBA")
    aoi = metadata["aoi"]
    extent = [aoi["bounds_west"], aoi["bounds_east"],
              aoi["bounds_south"], aoi["bounds_north"]]

    figure = plt.figure(figsize=(13.5, 7.2), dpi=300)
    grid = figure.add_gridspec(1, 2, width_ratios=[1.95, 1.0], wspace=0.10,
                               left=0.055, right=0.985, top=0.93, bottom=0.06)
    axis = figure.add_subplot(grid[0, 0])
    axis.imshow(rgb, extent=extent, aspect="auto")
    axis.imshow(overlay, extent=extent, aspect="auto")
    axis.set_xlabel("Longitude (deg E)")
    axis.set_ylabel("Latitude (deg N)")
    axis.set_title(
        f"{aoi['adm2_name']} district, FAO GAUL 2015 ADM2 {aoi['adm2_code']}",
        fontsize=10.5)
    axis.tick_params(labelsize=8)

    india = load_outline("india_outline.json")
    madhya_pradesh = load_outline("mp_outline.json")
    map_box = axis.get_position()
    locator_size = min(map_box.width * 0.23, map_box.height * 0.28)
    locator_axis = figure.add_axes([
        map_box.x1 - locator_size - 0.02,
        map_box.y0 + 0.02,
        locator_size,
        locator_size,
    ], zorder=8)
    locator(locator_axis, (extent[0] + extent[1]) / 2,
            (extent[2] + extent[3]) / 2, india, madhya_pradesh)

    panel = figure.add_subplot(grid[0, 1])
    panel.axis("off")
    raw_areas = metadata["class_areas_km2"]["raw"]
    filtered_areas = metadata["class_areas_km2"]["majority_filtered"]
    raw_total = sum(raw_areas.values())

    y = 0.985
    panel.text(0, y, "Class areas from the delivered raster", fontsize=10,
               weight="bold", va="top", transform=panel.transAxes)
    y -= 0.055
    panel.text(0.40, y, "raw", fontsize=8, ha="right", style="italic", va="top", transform=panel.transAxes)
    panel.text(0.66, y, "filtered", fontsize=8, ha="right", style="italic", va="top", transform=panel.transAxes)
    panel.text(0.93, y, "% of raw", fontsize=8, ha="right", style="italic", va="top", transform=panel.transAxes)
    y -= 0.045
    for index in sorted(LAND_COVER_CLASSES):
        name = LAND_COVER_CLASSES[index]["name"]
        panel.add_patch(Rectangle((0.0, y - 0.026), 0.030, 0.026,
                                  color=LAND_COVER_CLASSES[index]["color"],
                                  transform=panel.transAxes, clip_on=False))
        panel.text(0.045, y - 0.006, name, fontsize=8.5, va="top", transform=panel.transAxes)
        panel.text(0.40, y - 0.006, f"{raw_areas.get(name, 0.0):,.1f}",
                   fontsize=8.5, ha="right", va="top", transform=panel.transAxes)
        panel.text(0.66, y - 0.006, f"{filtered_areas.get(name, 0.0):,.1f}",
                   fontsize=8.5, ha="right", va="top", transform=panel.transAxes)
        panel.text(0.93, y - 0.006,
                   f"{100 * raw_areas.get(name, 0.0) / raw_total:.2f}%",
                   fontsize=8.5, ha="right", va="top", transform=panel.transAxes)
        y -= 0.050
    panel.plot([0, 0.93], [y + 0.020, y + 0.020], color="0.3", lw=0.8,
               transform=panel.transAxes, clip_on=False)
    y -= 0.012
    panel.text(0.045, y, "Mapped total", fontsize=8.5, weight="bold",
               va="top", transform=panel.transAxes)
    panel.text(0.40, y, f"{raw_total:,.1f}", fontsize=8.5, weight="bold",
               ha="right", va="top", transform=panel.transAxes)
    panel.text(0.66, y, f"{sum(filtered_areas.values()):,.1f}", fontsize=8.5,
               weight="bold", ha="right", va="top", transform=panel.transAxes)
    y -= 0.045
    panel.text(0.045, y, f"GAUL geodesic area  {aoi['geodesic_area_km2']:,.1f} km$^2$",
               fontsize=8, va="top", color="0.25", transform=panel.transAxes)

    y -= 0.075
    model = metadata["model"]
    parameters = model["parameters"]
    rows = [
        ("Classifier", model["constructor"].split(".")[-1]),
        ("Settings", f"{parameters['numberOfTrees']} trees, shrinkage "
                     f"{parameters['shrinkage']}, max nodes {parameters['maxNodes']}"),
        ("Unset settings", model["unset_parameters_use_earth_engine_defaults"]),
        ("Features", model["feature_stack"]),
        ("Composite", metadata["composite"]["date_filter"]),
        ("Cloud filter", f"scenes < {metadata['composite']['cloud_threshold_percent']}% cloud, QA60 bits 10 and 11"),
        ("Sensors", "Sentinel-2 L2A + Sentinel-1 IW descending, medians"),
        ("Displayed raster", "raw per-pixel classification"),
        ("Area measurement", f"pixelArea sum at "
                             f"{metadata['grids']['analysis_scale_m']} m on "
                             f"{metadata['grids']['analysis_crs']}"),
        ("Display projection", metadata["grids"]["display_crs"]),
    ]
    for label, value in rows:
        panel.text(0.0, y, label, fontsize=7.6, weight="bold", va="top", transform=panel.transAxes)
        panel.text(0.30, y, value, fontsize=7.6, va="top", wrap=True, transform=panel.transAxes)
        y -= 0.042

    y -= 0.010
    panel.text(0.0, y,
               "Class areas are raster-derived estimates, not independently\n"
               "validated land-use statistics. No accuracy figure in this paper\n"
               "is measured on this district: it holds training labels and is\n"
               "withheld from every reported score.",
               fontsize=7.2, va="top", color="0.25", transform=panel.transAxes)

    # No bbox_inches="tight" here. The right-hand panel draws its text with
    # axes coordinates that run past the axes box, and a tight bounding box
    # grows to enclose all of it -- the first render came out at 245 megapixels.
    # A fixed canvas at a sane dpi is what a figure at column width needs.
    figure.savefig(PLATE_PATH, dpi=220, facecolor="white")
    print(f"wrote {PLATE_PATH.relative_to(REPO)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draw-only", action="store_true",
                        help="redraw from the stored metadata, no Earth Engine")
    args = parser.parse_args()
    metadata = (json.loads(METADATA_PATH.read_text()) if args.draw_only
                else compute())
    draw(metadata)


if __name__ == "__main__":
    main()
