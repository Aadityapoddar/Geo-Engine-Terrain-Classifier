#!/bin/sh
# Rebuild every weighted figure on the 10 m sampling grid.
#
# The stratified sample is drawn at 10 m and was weighted by stratum areas
# measured at 30 m, which under-measures fragmented classes and left sampled
# points in strata of zero measured area. This measures the areas on the grid
# the sample was drawn on and regenerates everything downstream of them.
#
# The 30 m record is kept rather than overwritten: the comparison between the
# two is itself a reported result.
#
#     sh scripts/refresh_weights_10m.sh
set -e
cd "$(dirname "$0")/.."

ASSETS=doc/assets
TEN=$ASSETS/mp_reference_stratum_areas_10m.json

echo "-- merge the shard measurements"
python3 -W ignore scripts/reference_stratum_weights.py \
  --scale 10 --output "$TEN" \
  --merge $ASSETS/mp_reference_stratum_areas_10m_s0.json \
          $ASSETS/mp_reference_stratum_areas_10m_s1.json \
          $ASSETS/mp_reference_stratum_areas_10m_s2.json

echo "-- area-adjusted accuracy on the 10 m weights"
python3 -W ignore scripts/area_adjusted_accuracy.py \
  $ASSETS/mp_external_shards_v7_worker_*.json \
  --weights "$TEN" \
  --classifier after-gtb \
  --output $ASSETS/mp_area_adjusted_v7_10m.json

echo "-- rebuild the archive against the 10 m weights"
python3 -W ignore scripts/build_result_archive.py \
  --stratum-areas mp_reference_stratum_areas_10m.json \
  --area-adjusted mp_area_adjusted_v7_10m.json

echo "-- per-class weighted table and the shortfall audit that reads the same areas"
python3 -W ignore scripts/weighted_per_class.py
python3 -W ignore scripts/reference_shortfall_audit.py

echo "== done =="
