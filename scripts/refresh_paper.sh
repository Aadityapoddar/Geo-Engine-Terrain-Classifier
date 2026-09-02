#!/bin/sh
# Rebuild every derived artefact from the shards on disk, in dependency order.
#
# Nothing here touches Earth Engine: the expensive part is the statewide run
# (scripts/run_sharded_mp_external.py) and the district experiments
# (scripts/district_experiments.py), and this is what turns their output into
# the numbers, figures and documents the paper ships.
#
# The order matters. Config is rewritten from the results before the figures
# and the manuscript are built, so the dashboard badge, the tables and the plots
# all describe one run. Bump evaluation/artefacts.RESULTS_VERSION first when a
# new statewide run lands.
#
#     sh scripts/refresh_paper.sh
set -e
cd "$(dirname "$0")/.."

VERSION=$(python3 -W ignore -c 'from evaluation.artefacts import RESULTS_VERSION; print(RESULTS_VERSION)')
echo "== refreshing from ${VERSION} shards =="

echo "-- aggregate"
python3 -W ignore scripts/aggregate_mp_external_shards.py \
  doc/assets/mp_external_shards_${VERSION}_worker_*.json \
  --output doc/assets/mp_external_results_${VERSION}.json

echo "-- district-block uncertainty, paired tests, transfer decay"
python3 -W ignore scripts/spatial_uncertainty.py \
  doc/assets/mp_external_shards_${VERSION}_worker_*.json \
  --output doc/assets/mp_spatial_uncertainty_${VERSION}.json

echo "-- area-adjusted accuracy (Olofsson)"
python3 -W ignore scripts/area_adjusted_accuracy.py \
  doc/assets/mp_external_shards_${VERSION}_worker_*.json \
  --weights doc/assets/mp_reference_stratum_areas.json \
  --classifier after-gtb \
  --output doc/assets/mp_area_adjusted_${VERSION}.json

echo "-- sync the dashboard benchmarks to the measured test-district results"
python3 -W ignore scripts/sync_model_benchmarks.py

echo "-- figures"
python3 -W ignore scripts/build_report_figures.py --self-check
python3 -W ignore scripts/build_report_figures.py

echo "-- manuscript"
python3 -W ignore scripts/update_paper_tex.py
tectonic -X compile new.tex --outdir . >/dev/null 2>&1 || tectonic -X compile new.tex --outdir .
python3 -W ignore scripts/build_paper_docx.py

echo "-- audit response"
python3 -W ignore scripts/build_audit_response.py

echo "-- tests"
python3 -W ignore -m pytest tests/ -q --ignore=tests/test_api.py \
  --ignore=tests/test_asset_audit.py

echo "== done =="
