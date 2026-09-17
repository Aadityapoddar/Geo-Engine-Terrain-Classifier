"""Eligible area for every Dynamic World confidence threshold, in one pass.

For each class the candidate set is WorldCover + specialist + the matching DW
label. A histogram of DW probability over that set gives the surviving area at
every threshold at once, so the whole sensitivity curve costs one reduction.
"""
import os, sys, json
sys.path.insert(0, '/Users/ashutoshsaxena/Desktop/CODE-SEM5/Geo-Engine-Terrain-Classifier')
os.chdir('/Users/ashutoshsaxena/Desktop/CODE-SEM5/Geo-Engine-Terrain-Classifier')
from dotenv import load_dotenv
load_dotenv('/Users/ashutoshsaxena/Desktop/CODE-SEM5/Geo-Engine-Terrain-Classifier/.env')
import ee
ee.Initialize(project=os.getenv('EE_PROJECT_ID'))

from evaluation import references as R
from backend.config import SEASONS
import csv

test = {r['district'] for r in csv.DictReader(
    open('doc/assets/result_archive/district_split.csv')) if r['role'] == 'test'}
districts = R.madhya_pradesh_districts().filter(
    ee.Filter.inList('ADM2_NAME', sorted(test)))
region = districts.geometry()
print(f"test districts: {len(test)}", flush=True)

STEPS = [0.0, 0.3, 0.5, 0.7]  # coarse: the shape, not the curve
out = {}

for season in ("winter", "summer"):
    d = SEASONS[season]
    dw = R._dynamic_world(region, d["start"], d["end"])
    dw_label, prob = dw.select("dw_label"), dw.select("dw_probability")
    wc = ee.ImageCollection(R.WORLD_COVER_ID).first().select("Map")
    ghsl = ee.Image(R.GHSL_ID).select("built_surface").gte(R.GHSL_MIN_BUILT_SQM)
    crops = R._world_cereal(region, season)
    opera = R._opera_water(region, d["start"], d["end"])
    conflict = opera.Or(ghsl).Or(crops)

    candidates = {
        "Vegetation":  wc.eq(10).And(dw_label.eq(1)),
        "Water":       wc.eq(80).And(dw_label.eq(0)).And(opera),
        "Built Area":  wc.eq(50).And(dw_label.eq(6)).And(ghsl),
        "Open Land":   wc.eq(60).And(dw_label.eq(7)).And(conflict.Not()),
        "Agriculture": wc.eq(40).And(dw_label.eq(4)).And(crops),
    }
    for name, mask in candidates.items():
        area = ee.Image.pixelArea().divide(1e6).updateMask(mask)
        # area surviving each threshold, one server-side list
        curve = ee.List(STEPS).map(lambda t: area.updateMask(prob.gte(ee.Number(t)))
                                   .reduceRegion(ee.Reducer.sum(), region, 300,
                                                 maxPixels=1e12, bestEffort=True)
                                   .get("area"))
        vals = curve.getInfo()
        out[f"{season}/{name}"] = {str(t): (v or 0.0) for t, v in zip(STEPS, vals)}
        at = out[f"{season}/{name}"]
        print(f"{season:7} {name:12} t=0.00 {at['0.0']:9,.0f}  t=0.30 {at['0.3']:9,.0f}  "
              f"t=0.50 {at['0.5']:9,.0f}  t=0.70 {at['0.7']:9,.0f} km2", flush=True)

json.dump(out, open('/Users/ashutoshsaxena/Desktop/CODE-SEM5/Geo-Engine-Terrain-Classifier/doc/assets/reference_threshold_sweep.json', 'w'), indent=1)
print("saved sweep.json", flush=True)
