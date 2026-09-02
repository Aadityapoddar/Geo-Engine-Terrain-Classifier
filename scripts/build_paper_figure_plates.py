#!/usr/bin/env python3
"""Build the five publication plates of classified areas of interest.

Every pixel on these maps comes from the deployed classifier, not from a local
re-run: the script posts each area of interest to the live `/api/classify`, waits
for the tiled 10 m overlay to render, and draws the plate around the PNG that
comes back. What the reader sees is therefore exactly what the service serves.

The Sentinel-2 rasters are cached under doc/assets/paper_figs/sources/ (ignored
by git, ~45 MB) so re-running to adjust the cartography costs nothing.

    python scripts/build_paper_figure_plates.py
"""

import json
import math
import os
import sys
import textwrap
import time
import urllib.error
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from matplotlib.patches import Polygon as MplPolygon, Rectangle  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from backend.config import (  # noqa: E402
    LAND_COVER_CLASSES,
    MODEL_METADATA,
    SEASONS,
)

PLATES = os.path.join(REPO, "doc", "assets", "paper_figs")
SOURCES = os.path.join(PLATES, "sources")


# GAUL carves the disputed units out of "India" and files them as separate ADM0
# entries, so filtering on the name alone draws an India missing Jammu and
# Kashmir, Ladakh and Arunachal Pradesh. That map may not be published in India.
# The locator therefore unions the claimed territory back together.
INDIA_CLAIMED = ["India", "Jammu and Kashmir", "Aksai Chin", "Arunachal Pradesh",
                 "China/India"]


def outlines():
    """India and Madhya Pradesh, for the locator inset. Fetched once, then cached."""
    wanted = {
        "india_outline.json": ("FAO/GAUL/2015/level0", "ADM0_NAME", INDIA_CLAIMED, 2000),
        "mp_outline.json": ("FAO/GAUL/2015/level1", "ADM1_NAME", ["Madhya Pradesh"], 500),
    }
    missing = {n: v for n, v in wanted.items()
               if not os.path.exists(os.path.join(SOURCES, n))}
    if not missing:
        return
    import ee
    from backend.gee_classifier import init_ee
    init_ee()
    for name, (table, field, values, tol) in missing.items():
        fc = ee.FeatureCollection(table).filter(ee.Filter.inList(field, values))
        json.dump(fc.geometry().dissolve(maxError=500).simplify(tol).getInfo(),
                  open(os.path.join(SOURCES, name), "w"))


BASE = os.getenv("GEO_ENGINE_URL", "https://geo-engine-terrain-classifier.onrender.com")
# Taken from config so the plate can never drift from the window the
# accuracy figures were measured in.
SEASON = (SEASONS["winter"]["start"], SEASONS["winter"]["end"])

AOIS = [
    ("jabalpur_city",  "Jabalpur City and the Narmada Corridor", 79.830, 23.080, 80.030, 23.245),
    ("bargi_reservoir","Bargi Reservoir, Narmada Basin",         79.760, 22.855, 79.985, 23.020),
    ("bhopal_lake",    "Bhoj Wetland and Bhopal Urban Fabric",   77.215, 23.170, 77.445, 23.335),
    ("indore_fringe",  "Indore Urban-Agricultural Fringe",       75.755, 22.630, 75.975, 22.795),
    ("kanha_forest",   "Kanha Deciduous Forest Belt",            80.540, 22.245, 80.760, 22.410),
]


def post(path, body, timeout=900):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def get(path, timeout=120):
    return json.load(urllib.request.urlopen(BASE + path, timeout=timeout))


def bbox_polygon(w, s, e, n):
    return {"type": "Polygon", "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]}


def download(url, path):
    for attempt in range(4):
        try:
            urllib.request.urlretrieve(url, path)
            return True
        except Exception as ex:
            print(f"   retry {attempt} on {path}: {ex}", flush=True)
            time.sleep(10)
    return False


def fetch_all():
    os.makedirs(SOURCES, exist_ok=True)
    outlines()
    for key, title, w, s, e, n in AOIS:
        meta_path = os.path.join(SOURCES, f"{key}.json")
        png_path = os.path.join(SOURCES, f"{key}_class.png")
        if os.path.exists(meta_path) and os.path.exists(png_path):
            print(f"[{key}] already done", flush=True)
            continue

        print(f"[{key}] classifying ...", flush=True)
        t0 = time.time()
        res = post("/api/classify", {
            "geometry": bbox_polygon(w, s, e, n),
            "model_type": "gtb",
            "start_date": SEASON[0], "end_date": SEASON[1],
            "cloud_threshold": 15, "smoothing": True,
        })
        res["aoi"] = {"key": key, "title": title,
                      "west": w, "south": s, "east": e, "north": n,
                      "start_date": SEASON[0], "end_date": SEASON[1]}
        ov = res["terrain_overlay"]
        print(f"[{key}] stats in {time.time()-t0:.0f}s, "
              f"overlay {ov['width_px']}x{ov['height_px']} key={ov['key']}", flush=True)

        # true-colour composite (expires, so fetch straight away)
        download(res["rgb_overlay"]["url"], os.path.join(SOURCES, f"{key}_rgb.png"))

        # classified overlay renders tile by tile in the background
        deadline = time.time() + 3600
        while time.time() < deadline:
            st = get(f"/api/overlay/{ov['key']}")
            if st.get("status") == "ready":
                break
            print(f"   {st.get('status')} {st.get('done')}/{st.get('total')}", flush=True)
            time.sleep(20)
        else:
            print(f"[{key}] TIMED OUT", flush=True)
            continue

        download(f"{BASE}/overlays/{ov['key']}.png", png_path)
        json.dump(res, open(meta_path, "w"), indent=1)
        print(f"[{key}] done in {time.time()-t0:.0f}s", flush=True)


R = 6378137.0
merc_x = lambda lon: math.radians(lon) * R
merc_y = lambda lat: math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)) * R

# Straight from the deployed inventory, so the key can never drift from the
# palette the overlay was rendered with.
CLASSES = [(c["name"], c["color"]) for c in LAND_COVER_CLASSES.values()]

CAPTIONS = {
    "jabalpur_city": "Five-class terrain classification of Jabalpur city and the Narmada corridor, "
                     "inside the district the labelled training set was collected in. The built-up "
                     "core is recovered as one contiguous unit, the Narmada is tracked as an "
                     "unbroken channel across the southern third of the scene, and the cultivated "
                     "plain to the north and west separates from the forested ridges on the eastern "
                     "margin.",
    "bargi_reservoir": "Five-class terrain classification over the Bargi reservoir on the Narmada. "
                       "The impounded water body on the eastern side of the scene and the meandering "
                       "channel upstream of it are mapped as a single connected water class, and the "
                       "cultivated land of the valley floor is separated from the bare rocky ground "
                       "of the surrounding plateau.",
    "bhopal_lake": "Five-class terrain classification of the Bhoj Wetland and the Bhopal urban "
                   "fabric, roughly 300 km west of the labelled area. The lake margin holds against "
                   "dense built-up land pressing on it from the east, which is the hardest boundary "
                   "in the scene for a spectral classifier.",
    "indore_fringe": "Five-class terrain classification of the Indore urban-agricultural fringe on "
                     "the Malwa plateau. The built-up core is compact, but the transition into "
                     "black-cotton-soil cropland is where the residual bare-versus-built confusion "
                     "reported for this model is most visible at map scale.",
    "kanha_forest": "Five-class terrain classification of the Kanha deciduous forest belt in the "
                    "Satpura range, the far end of the generalisation range from the labelled area. "
                    "Closed canopy covers 76 % of the scene and the cultivated and bare clearings "
                    "inside and along the eastern boundary of the reserve are resolved at the 10 m "
                    "grid.",
}

MODEL_LINE = ("Smile Gradient Tree Boosting (GTB), 100 trees, shrinkage 0.1")


def stretch(img, low=1.5, high=99.0, gamma=0.92):
    """Percentile stretch. The winter composite is dark at min 0 / max 0.3."""
    a = np.asarray(img).astype(np.float32)
    out = np.empty_like(a)
    for b in range(a.shape[2]):
        lo, hi = np.percentile(a[..., b], [low, high])
        out[..., b] = np.clip((a[..., b] - lo) / max(hi - lo, 1e-6), 0, 1) ** gamma
    return out


def load_outline(name, min_area=0.01):
    """Outer rings of a GeoJSON geometry, in square degrees of `min_area` and up.

    Dissolving the claimed territory leaves a tail of sub-pixel slivers behind;
    the threshold drops those while keeping the Andaman and Nicobar chain, the
    Kutch islands and the Sundarbans delta, which belong on a map of India.
    """
    def rings(g):
        t = g["type"]
        if t == "GeometryCollection":
            return [r for sub in g["geometries"] for r in rings(sub)]
        if t == "MultiPolygon":
            return [p[0] for p in g["coordinates"]]
        if t == "Polygon":
            return [g["coordinates"][0]]
        return []
    def area(r):
        return 0.5 * abs(np.dot(r[:-1, 0], r[1:, 1]) - np.dot(r[1:, 0], r[:-1, 1]))

    out = [np.array(r) for r in rings(json.load(open(os.path.join(SOURCES, name))))]
    return [r for r in out if len(r) >= 4 and area(r) >= min_area]





def dms(value, positive, negative):
    hemi = positive if value >= 0 else negative
    value = abs(value)
    deg = int(value)
    minutes = (value - deg) * 60.0
    if abs(minutes - round(minutes)) < 1e-6:
        return f"{deg}°{int(round(minutes)):02d}'{hemi}"
    return f"{deg}°{minutes:04.1f}'{hemi}"


def nice_ticks(lo, hi, target=5):
    """Graticule interval that lands on round minutes of arc."""
    span = hi - lo
    for step in (0.01, 0.02, 0.025, 0.05, 0.1, 0.2, 0.25, 0.5, 1.0):
        if span / step <= target:
            break
    first = math.ceil(lo / step) * step
    ticks, v = [], first
    while v <= hi + 1e-9:
        ticks.append(round(v, 6))
        v += step
    return ticks


def north_arrow(ax, x, y, size=0.055):
    """Filled compass needle in axes coordinates."""
    ax.add_patch(Rectangle((x - size * 0.62, y - size * 0.62), size * 1.24, size * 2.05,
                           transform=ax.transAxes, facecolor="white", edgecolor="black",
                           linewidth=0.5, zorder=5))
    h = size
    w = size * 0.42
    ax.add_patch(MplPolygon([[x, y + h], [x - w / 2, y - h * 0.45], [x, y - h * 0.18]],
                            closed=True, transform=ax.transAxes, facecolor="white",
                            edgecolor="black", linewidth=0.7, zorder=6))
    ax.add_patch(MplPolygon([[x, y + h], [x + w / 2, y - h * 0.45], [x, y - h * 0.18]],
                            closed=True, transform=ax.transAxes, facecolor="black",
                            edgecolor="black", linewidth=0.7, zorder=6))
    ax.text(x, y + h + 0.012, "N", transform=ax.transAxes, ha="center", va="bottom",
            fontsize=9.5, fontweight="bold", zorder=6)


def scale_bar(ax, lat_mid, x=0.045, y=0.045):
    """Alternating-segment bar. Ground km, corrected for the Mercator inflation."""
    x0, x1 = ax.get_xlim()
    ground_span_km = (x1 - x0) * math.cos(math.radians(lat_mid)) / 1000.0
    for total in (1, 2, 2.5, 5, 10, 20, 25, 50):
        if total >= ground_span_km * 0.22:
            break
    n_seg = 4
    seg_m = (total / n_seg) * 1000.0 / math.cos(math.radians(lat_mid))
    xb = x0 + (x1 - x0) * x
    y0, y1 = ax.get_ylim()
    yb = y0 + (y1 - y0) * y
    hb = (y1 - y0) * 0.011

    ax.add_patch(Rectangle((xb - seg_m * 0.35, yb - hb * 2.6),
                           seg_m * n_seg + seg_m * 0.7, hb * 6.2,
                           facecolor="white", edgecolor="black", linewidth=0.5, zorder=5))
    for i in range(n_seg):
        ax.add_patch(Rectangle((xb + i * seg_m, yb), seg_m, hb,
                               facecolor="black" if i % 2 == 0 else "white",
                               edgecolor="black", linewidth=0.6, zorder=6))
    for i in (0, n_seg // 2, n_seg):
        label = f"{total * i / n_seg:g}"
        ax.text(xb + i * seg_m, yb + hb * 1.25, label, ha="center", va="bottom",
                fontsize=7.2, zorder=6)
    ax.text(xb + n_seg * seg_m + seg_m * 0.18, yb + hb * 0.1, "km",
            ha="left", va="bottom", fontsize=7.2, zorder=6)


def locator(ax, lon, lat, india, mp):
    ax.set_facecolor("#eef2f6")
    for ring in india:
        ax.add_patch(MplPolygon(ring, closed=True, facecolor="#ffffff",
                                edgecolor="#7a8896", linewidth=0.5, zorder=1))
    for ring in mp:
        ax.add_patch(MplPolygon(ring, closed=True, facecolor="#b9c8d6",
                                edgecolor="#41505f", linewidth=0.55, zorder=2))
    ax.plot([lon], [lat], marker="*", markersize=8.5, markerfacecolor="#d81515",
            markeredgecolor="black", markeredgewidth=0.5, zorder=3, linestyle="none")
    ax.set_xlim(67.0, 98.5)
    ax.set_ylim(6.0, 37.5)
    ax.set_aspect(1 / math.cos(math.radians(22.0)))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.8)
    ax.text(0.5, 0.022, "Madhya Pradesh, India", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=6.6,
            bbox=dict(boxstyle="square,pad=0.18", facecolor="white", edgecolor="none",
                      alpha=0.85))


def build(key):
    india, mp = load_outline("india_outline.json"), load_outline("mp_outline.json")
    meta = json.load(open(os.path.join(SOURCES, f"{key}.json")))
    aoi = meta["aoi"]
    ov = meta["terrain_overlay"]["bounds"]
    cls_img = Image.open(os.path.join(SOURCES, f"{key}_class.png")).convert("RGBA")
    rgb_path = os.path.join(SOURCES, f"{key}_rgb.png")
    rgb_img = Image.open(rgb_path).convert("RGB") if os.path.exists(rgb_path) else None

    x0, x1 = merc_x(ov["west"]), merc_x(ov["east"])
    y0, y1 = merc_y(ov["south"]), merc_y(ov["north"])
    lat_mid = (aoi["north"] + aoi["south"]) / 2.0

    # ── geometry of the plate, in inches ──────────────────────────────────────
    map_h = 6.0
    map_w = map_h * (x1 - x0) / (y1 - y0)
    side_w = 3.05
    pad_l, pad_r, gap = 0.78, 0.22, 0.42
    pad_b, pad_t = 1.02, 0.66
    fig_w = pad_l + map_w + gap + side_w + pad_r
    fig_h = pad_b + map_h + pad_t

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=400, facecolor="white")
    ax = fig.add_axes([pad_l / fig_w, pad_b / fig_h, map_w / fig_w, map_h / fig_h])

    ax.imshow(np.asarray(cls_img), extent=[x0, x1, y0, y1],
              interpolation="nearest", zorder=1)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)

    lon_ticks = nice_ticks(ov["west"], ov["east"])
    lat_ticks = nice_ticks(ov["south"], ov["north"])
    for lon in lon_ticks:
        ax.axvline(merc_x(lon), color="white", linewidth=0.45, alpha=0.45,
                   linestyle=(0, (4, 4)), zorder=3)
    for lat in lat_ticks:
        ax.axhline(merc_y(lat), color="white", linewidth=0.45, alpha=0.45,
                   linestyle=(0, (4, 4)), zorder=3)

    ax.set_xticks([merc_x(v) for v in lon_ticks])
    ax.set_xticklabels([dms(v, "E", "W") for v in lon_ticks], fontsize=8)
    ax.set_yticks([merc_y(v) for v in lat_ticks])
    ax.set_yticklabels([dms(v, "N", "S") for v in lat_ticks], fontsize=8, rotation=90,
                       va="center")
    ax.tick_params(direction="out", length=3.2, width=0.7, top=True, right=True,
                   labeltop=False, labelright=False, pad=2.5)
    for s in ax.spines.values():
        s.set_linewidth(1.1)
        s.set_zorder(7)

    north_arrow(ax, 0.935, 0.90)
    scale_bar(ax, lat_mid)

    ax.text(0.5, 1.012, aoi["title"], transform=ax.transAxes, ha="center", va="bottom",
            fontsize=11.5, fontweight="bold")

    # ── side column ───────────────────────────────────────────────────────────
    sx = (pad_l + map_w + gap) / fig_w
    sw = side_w / fig_w
    top_in = pad_b + map_h          # inches from the figure bottom
    gap_in = 0.26
    leg_h_in = 1.62
    info_h_in = 1.66

    # (a) Sentinel-2 true colour, the same footprint as the map. Sized to
    # whatever height the key and the information panel leave behind, so the
    # column always ends flush with the map.
    rgb_h_in = 0.0
    if rgb_img is not None:
        budget = map_h - leg_h_in - info_h_in - 2 * gap_in
        rgb_w_in = min(side_w, budget * rgb_img.width / rgb_img.height)
        rgb_h_in = rgb_w_in * rgb_img.height / rgb_img.width
        ax_rgb = fig.add_axes([sx + (sw - rgb_w_in / fig_w) / 2,
                               (top_in - rgb_h_in) / fig_h,
                               rgb_w_in / fig_w, rgb_h_in / fig_h])
        ax_rgb.imshow(stretch(rgb_img), interpolation="bilinear")
        ax_rgb.set_xticks([]); ax_rgb.set_yticks([])
        for sp in ax_rgb.spines.values():
            sp.set_linewidth(1.0)
        ax_rgb.text(0.5, 1.012, "Sentinel-2 true colour (B4/B3/B2)",
                    transform=ax_rgb.transAxes, ha="center", va="bottom", fontsize=8.4)

    # (b) class key with the mapped areas
    areas = {a["name"]: a for a in meta["individual_class_areas"]}
    rows = len(CLASSES)
    leg_b = top_in - rgb_h_in - gap_in - leg_h_in
    ax_leg = fig.add_axes([sx, leg_b / fig_h, sw, leg_h_in / fig_h])
    ax_leg.set_xlim(0, 1); ax_leg.set_ylim(0, 1)
    ax_leg.axis("off")
    ax_leg.add_patch(Rectangle((0, 0), 1, 1, transform=ax_leg.transAxes, fill=False,
                               edgecolor="black", linewidth=0.9))
    ax_leg.text(0.5, 0.955, "T E R R A I N   C L A S S", ha="center", va="top",
                fontsize=8.4, fontweight="bold")
    ax_leg.plot([0.045, 0.955], [0.868, 0.868], color="black", linewidth=0.7)
    ax_leg.text(0.635, 0.845, "Area", ha="center", va="top", fontsize=7.4, style="italic")
    ax_leg.text(0.895, 0.845, "Cover", ha="center", va="top", fontsize=7.4, style="italic")
    ax_leg.text(0.635, 0.775, "(km$^2$)", ha="center", va="top", fontsize=6.9)
    ax_leg.text(0.895, 0.775, "(%)", ha="center", va="top", fontsize=6.9)
    ax_leg.plot([0.045, 0.955], [0.712, 0.712], color="black", linewidth=0.5)

    top, step = 0.632, 0.104
    for i, (name, colour) in enumerate(CLASSES):
        y = top - i * step
        ax_leg.add_patch(Rectangle((0.055, y - 0.030), 0.080, 0.058, facecolor=colour,
                                   edgecolor="black", linewidth=0.5))
        ax_leg.text(0.163, y, name, ha="left", va="center", fontsize=8.4)
        a = areas.get(name, {})
        ax_leg.text(0.712, y, f"{a.get('area_ha', 0) / 100.0:,.1f}",
                    ha="right", va="center", fontsize=8.4)
        ax_leg.text(0.955, y, f"{a.get('percentage', 0):.1f}",
                    ha="right", va="center", fontsize=8.4)
    rule = top - (rows - 1) * step - 0.058
    ax_leg.plot([0.045, 0.955], [rule, rule], color="black", linewidth=0.7)
    total_km2 = meta["summary"]["total_area_ha"] / 100.0
    ty = rule - 0.052
    ax_leg.text(0.163, ty, "Total mapped", ha="left", va="center", fontsize=8.4,
                fontweight="bold")
    ax_leg.text(0.712, ty, f"{total_km2:,.1f}", ha="right", va="center", fontsize=8.4,
                fontweight="bold")
    ax_leg.text(0.955, ty, "100.0", ha="right", va="center", fontsize=8.4,
                fontweight="bold")

    # (c) what produced the map.
    #
    # The benchmark is read from config rather than from the cached API
    # response beside the raster. The raster is expensive and worth reusing;
    # the accuracy printed next to it is not, and a stale response would put a
    # retired number on the figure while the paper carried the current one.
    bench = MODEL_METADATA["gtb"]["benchmark"]
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    def pretty(d):
        return f"{int(d[8:10])} {months[int(d[5:7]) - 1]} {d[:4]}"

    lines = [
        ("Sensor", "Sentinel-2 MSI, 10 m"),
        ("Composite", f"{pretty(aoi['start_date'])} to {pretty(aoi['end_date'])}"),
        ("Cloud filter", "scenes < 15 % cloud"),
        ("Features", "19: spectral, indices,"),
        ("", "GLCM texture, S-1 SAR"),
        ("Classifier", "Smile Gradient Tree"),
        ("", "Boosting, 100 trees"),
        ("External OA", f"{bench['winter']:.1f} % winter,"),
        ("", "held-out MP districts"),
        ("Projection", "WGS 84 / Web Mercator"),
        ("Platform", "Google Earth Engine"),
    ]
    info_b = leg_b - gap_in - info_h_in
    ax_info = fig.add_axes([sx, info_b / fig_h, sw, info_h_in / fig_h])
    ax_info.set_xlim(0, 1); ax_info.set_ylim(0, 1)
    ax_info.axis("off")
    ax_info.add_patch(Rectangle((0, 0), 1, 1, transform=ax_info.transAxes, fill=False,
                                edgecolor="black", linewidth=0.9))
    yy = 1.0 - 0.20 / info_h_in
    dy = 0.145 / info_h_in
    for label, value in lines:
        if label:
            ax_info.text(0.045, yy, f"{label}", fontsize=7.3, va="center",
                         fontweight="bold")
        ax_info.text(0.44, yy, value, fontsize=7.3, va="center")
        yy -= dy

    # (d) where in India this is, dropped into the map itself
    loc_in = 1.22
    ax_loc = fig.add_axes([(pad_l + map_w - loc_in - 0.10) / fig_w,
                           (pad_b + 0.10) / fig_h,
                           loc_in / fig_w, loc_in / fig_h], zorder=8)
    locator(ax_loc, (aoi["west"] + aoi["east"]) / 2, lat_mid, india, mp)

    # ── caption ───────────────────────────────────────────────────────────────
    caption = CAPTIONS[key]
    wrapped = textwrap.fill(caption, width=int(fig_w * 15.5))
    fig.text(pad_l / fig_w, (pad_b - 0.34) / fig_h,
             f"Fig. {ORDER.index(key) + 1}.  {wrapped}",
             fontsize=8.6, va="top", ha="left", linespacing=1.45)

    out = os.path.join(PLATES, f"fig{ORDER.index(key) + 1}_{key}.png")
    fig.savefig(out, dpi=400, facecolor="white")
    plt.close(fig)
    print("wrote", out, f"{os.path.getsize(out)/1e6:.1f} MB")


ORDER = ["jabalpur_city", "bargi_reservoir", "bhopal_lake", "indore_fringe", "kanha_forest"]


def main():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.linewidth": 1.0,
    })
    os.makedirs(PLATES, exist_ok=True)
    fetch_all()
    for key in ORDER:
        build(key)


if __name__ == "__main__":
    main()
