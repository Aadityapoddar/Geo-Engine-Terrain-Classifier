# Seasonal Six-Class Madhya Pradesh Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export Agriculture label 5, ship the six-class 19-band model, and report Winter/Summer before-versus-after accuracy for RF, SVM, XGB, CART, and KNN across Madhya Pradesh.

**Architecture:** `backend.config` is the schema source; `backend.gee_classifier` owns feature/classifier construction; a new `evaluation` package owns audits, external references, metrics, runs, and reporting. A phase-based CLI exports reusable Earth Engine tables, records task state, evaluates 20 conditions, and renders JSON/Markdown/figures.

**Tech Stack:** Python, Earth Engine, FastAPI, NumPy, Seaborn, Jupyter, JavaScript/Leaflet, TypeScript/Vite/MapLibre, pytest-compatible unittest.

## Global Constraints

- Labels: Forest `0`, Water `1`, Buildings `2`, Soil `3`, Sand `4`, Agriculture `5`.
- Never use 100-point `water_points`.
- Before uses `district_train_table`; After uses every current source asset.
- Winter: `2025-01-01` to exclusive `2025-02-28`; Summer: `2025-03-31` to exclusive `2025-04-30`.
- Use 19 inputs; exclude `B2`, `B4`, `B8`, `NDVI`, `NDWI`, `NDBI`, `SAVI`, `s_ent`.
- Keep existing model parameters.
- External MP accuracy collapses Soil+Sand to Bare; six-class accuracy uses held-out project labels.
- Preserve unrelated uncommitted work and old artifacts.

## File Map

- Create `evaluation/{__init__,assets,references,metrics,runner,report}.py`.
- Create `scripts/audit_training_assets.py`, `scripts/run_seasonal_evaluation.py`.
- Create focused tests for schema, assets, references, metrics, runner, notebooks, report.
- Create consolidated seasonal notebook, result JSON, report Markdown, manifest, audit, matrices.
- Modify backend config/classifier/API, both frontends, README/CHANGES, five training and five Jabalpur notebooks.

---

### Task 1: Lock Canonical Schema

**Files:** Create `tests/test_config_schema.py`; modify `backend/config.py:23-59`.

**Interfaces:** Produce `TRAINING_SCHEMA_VERSION`, `SEASONS`, `BEFORE_TRAINING_TABLE`, `FEATURE_COLLECTIONS`, `EXPECTED_ASSET_LABELS`, `BANDS`.

- [ ] **Step 1: Write failing test**

```python
from backend.config import BANDS, EXPECTED_ASSET_LABELS, LAND_COVER_CLASSES, SEASONS
def test_schema():
    assert {k:v["name"] for k,v in LAND_COVER_CLASSES.items()} == {
        0:"Forest",1:"Water",2:"Buildings",3:"Soil",4:"Sand",5:"Agriculture"}
    assert BANDS == ["B3","B11","B12","BSI","UI","IBI","SWIRratio","BAEI",
        "g_contrast","g_ent","g_var","g_idm","g_diss","g_asm","VV","VH","VVVH","s_contrast","s_var"]
    assert EXPECTED_ASSET_LABELS == {"forest":0,"water":1,"buildings":2,"soil":3,"soil_mp":3,"sand_mp":4,"agriculture":5}
    assert SEASONS["winter"] == {"start":"2025-01-01","end":"2025-02-28"}
```

- [ ] **Step 2: Run red test**

Run `python3 -m pytest tests/test_config_schema.py -q`; expect missing six-class/19-band schema.

- [ ] **Step 3: Implement configuration**

Add Agriculture `#F59E0B`, schema version `six-class-19-band-v1`, exact seasons, `district_train_table`, seven After assets, labels, exact 19 bands. Keep raw B2/B4/B8 for derived-band computation.

- [ ] **Step 4: Verify and commit**

```bash
python3 -m pytest tests/test_config_schema.py -q
git add backend/config.py tests/test_config_schema.py
git commit -m "feat: define six-class 19-band schema"
```

---

### Task 2: Extract Classifier and Asset Boundaries

**Files:** Modify `backend/gee_classifier.py:169-235`; create `evaluation/__init__.py`, `evaluation/assets.py`, `scripts/audit_training_assets.py`, `tests/test_classifier_schema.py`, `tests/test_asset_audit.py`.

**Interfaces:** Produce `_classifier_cache_key`, `make_classifier`, `merge_feature_collections`, `sample_training_points`, `training_points(condition)`, `audit_collection`, `validate_audit_record`.

- [ ] **Step 1: Write failing tests**

```python
from backend import gee_classifier as gc
from evaluation.assets import validate_audit_record
def test_helpers():
    assert gc._classifier_cache_key("RF","a","b",15)[-1] == "six-class-19-band-v1"
    try: gc.make_classifier("bad")
    except ValueError: pass
    else: raise AssertionError("unknown model accepted")
    validate_audit_record({"size":1000,"geometry_types":["Point"],
        "label_histogram":{"5":1000},"valid_sample_count":990}, 5)
```

- [ ] **Step 2: Run red tests**

Run `python3 -m pytest tests/test_classifier_schema.py tests/test_asset_audit.py -q`; expect missing helpers.

- [ ] **Step 3: Implement minimal boundaries**

Factory returns unchanged RF/SVM/XGB/CART/KNN constructors and rejects unknown IDs. Cache includes schema version. Before loads `district_train_table`; After merges all configured assets. Audit size, labels, point geometry, bounds, and both seasonal valid-sample counts; reject invalid records.

- [ ] **Step 4: Verify and commit**

```bash
python3 -m pytest tests/test_classifier_schema.py tests/test_asset_audit.py -q
git add backend/gee_classifier.py evaluation/__init__.py evaluation/assets.py scripts/audit_training_assets.py tests/test_classifier_schema.py tests/test_asset_audit.py
git commit -m "refactor: share classifier and asset validation"
```

---

### Task 3: Export Agriculture and Audit All Assets

**Files:** Create `doc/assets/training_asset_audit.json`.

**Interfaces:** Produce live `users/ashutoshsaxena703/jabalpur_agriculture_points` and audit evidence.

- [ ] **Step 1: Open saved Code Editor script**

Use the logged-in `jabalpur_agriculture_points` geometry import; never reconstruct screenshot points.

- [ ] **Step 2: Label/export once**

```javascript
var agricultureLabelled = jabalpur_agriculture_points.map(function(f) {
  return ee.Feature(f.geometry(), {label: 5});
});
Export.table.toAsset({collection: agricultureLabelled,
  description: 'export_jabalpur_agriculture_points',
  assetId: 'users/ashutoshsaxena703/jabalpur_agriculture_points'});
```

Retain task ID; use bounded waits; never overwrite other assets.

- [ ] **Step 3: Audit**

Run `python3 scripts/audit_training_assets.py --output doc/assets/training_asset_audit.json`. Expect Before `0:1000,1:1000,2:1000,3:2000,4:500`; After labels 0-5; no `water_points`; nonzero seasonal samples.

- [ ] **Step 4: Commit evidence**

```bash
git add doc/assets/training_asset_audit.json
git commit -m "docs: record complete training asset audit"
```

---

### Task 4: Build MP References, Metrics, Run Matrix

**Files:** Create `evaluation/references.py`, `evaluation/metrics.py`, `evaluation/runner.py`, `tests/test_reference_schema.py`, `tests/test_evaluation_metrics.py`, `tests/test_evaluation_runner.py`.

**Interfaces:** Produce `build_reference_image`, `sample_mp_reference`, `metrics_from_matrix`, `RunSpec`, `iter_run_specs`.

- [ ] **Step 1: Write failing tests**

```python
from evaluation.references import DW_MIN_PROBABILITY, GHSL_MIN_BUILT_SQM, REFERENCE_LABELS
from evaluation.metrics import collapse_project_prediction, metrics_from_matrix
from evaluation.runner import iter_run_specs
def test_contract():
    assert (DW_MIN_PROBABILITY, GHSL_MIN_BUILT_SQM) == (0.70, 50)
    assert REFERENCE_LABELS == {0:"Forest",1:"Water",2:"Buildings",3:"Bare",4:"Agriculture"}
    assert [collapse_project_prediction(i) for i in range(6)] == [0,1,2,3,3,4]
    assert metrics_from_matrix([[2,0],[0,3]],["A","B"])["overall_accuracy"] == 1
    assert len(iter_run_specs()) == 20
```

- [ ] **Step 2: Run red tests**

Run `python3 -m pytest tests/test_reference_schema.py tests/test_evaluation_metrics.py tests/test_evaluation_runner.py -q`.

- [ ] **Step 3: Implement consensus masks**

Require WorldCover and season-matched Dynamic World agreement with probability >=0.70. Refine Buildings with GHSL built surface >=50 m2, Agriculture with applicable WorldCereal active marker, Water with OPERA DSWx open/partial water. Bare excludes crop/built/water. Mask conflicts.

- [ ] **Step 4: Implement statewide sampling**

Use GAUL MP districts; sample up to 20 points/class/district at 10 m, seed `20260807`, attach district/season, exclude points within 100 m of either training population.

- [ ] **Step 5: Implement metrics/runs**

Return fixed matrices, OA, kappa, macro precision/recall/F1, per-class metrics, `None` for zero support. Generate 2 seasons x 2 conditions x 5 models.

- [ ] **Step 6: Verify and commit**

```bash
python3 -m pytest tests/test_reference_schema.py tests/test_evaluation_metrics.py tests/test_evaluation_runner.py -q
git add evaluation/references.py evaluation/metrics.py evaluation/runner.py tests/test_reference_schema.py tests/test_evaluation_metrics.py tests/test_evaluation_runner.py
git commit -m "feat: add MP seasonal evaluation core"
```

---

### Task 5: Phase Runner and Evidence Report

**Files:** Create `scripts/run_seasonal_evaluation.py`, `evaluation/report.py`, `tests/test_report_consistency.py`; generate manifest, results JSON, Markdown report, matrices under `doc/assets/`.

**Interfaces:** CLI phases `prepare`, `status`, `evaluate`, `report`.

- [ ] **Step 1: Write failing result-shape test**

```python
import json
from pathlib import Path
def test_results():
    runs = json.loads(Path("doc/assets/seasonal_before_after_results.json").read_text())["runs"]
    assert len(runs) == 20
    assert all(set(r["metrics"]) == {"external_five_class","heldout_six_class"} for r in runs)
```

- [ ] **Step 2: Implement phases**

`prepare` exports four training, four blocked-heldout, two external tables to schema-versioned IDs and records tasks. `status` reads once and exits nonzero unless all succeed. `evaluate` checkpoints after each run. `report` reads JSON only.

- [ ] **Step 3: Execute bounded external work**

```bash
python3 scripts/run_seasonal_evaluation.py prepare
python3 scripts/run_seasonal_evaluation.py status
python3 scripts/run_seasonal_evaluation.py evaluate
python3 scripts/run_seasonal_evaluation.py report
```

Delay status checks appropriately. Quota/failed tasks remain blockers.

- [ ] **Step 4: Verify/commit**

Run `python3 -m pytest tests/test_report_consistency.py -q`. Confirm 20 runs, two metrics each, Winter/Summer deltas, fixed matrices, no invented Sand external score. Commit Task 5 files/evidence as `docs: report seasonal MP accuracy changes`.

---

### Task 6: Synchronize Eleven Notebooks

**Files:** Modify five training and five Jabalpur notebooks; create `model_testing/multi_classification/madhya_pradesh/seasonal_before_after_all_models.ipynb`, `tests/test_notebook_schema.py`.

**Interfaces:** Notebooks import canonical config/runner/report; no duplicated formulas.

- [ ] **Step 1: Write failing source audit**

```python
import json
from pathlib import Path
def test_notebooks_import_config():
    paths = list(Path("model_training/multi_classification").glob("*_multi.ipynb"))
    paths += list(Path("model_testing/multi_classification/jabalpur").glob("*_multi_jabalpur.ipynb"))
    for p in paths:
        nb=json.loads(p.read_text()); src="".join("".join(c.get("source",[])) for c in nb["cells"] if c["cell_type"]=="code")
        assert "from backend.config import" in src and "BANDS" in src
        assert "class_names = ['Forest', 'Water', 'Buildings', 'Soil']" not in src
```

- [ ] **Step 2: Update historical notebooks**

Import bands/classes/assets/seasons; derive palette/max/order; mark embedded old outputs pre-Agriculture until replaced.

- [ ] **Step 3: Create consolidated notebook**

Initialize EE, show audit/run matrix, load JSON, call `evaluation.report`, display both seasonal tables/matrices. Do not reimplement algorithms.

- [ ] **Step 4: Verify/commit**

Run notebook test and compile every Python code cell. Commit eleven notebooks/test as `docs: synchronize six-class evaluation notebooks`.

---

### Task 7: Backend, Frontends, Docs, Full Verification

**Files:** Modify `backend/app.py`, API/overlay tests, `frontend/js/{app,charts}.js`, `frontend-next/src/{api,main}.ts`, `README.md`, `doc/CHANGES.md`.

**Interfaces:** `/api/config` returns classes, bands, seasons, schema version; classification returns six area rows.

- [ ] **Step 1: Write failing API assertion**

```python
def test_config_exposes_schema(self):
    data=self.client.get("/api/config").json()
    self.assertEqual((len(data["classes"]),len(data["bands"])),(6,19))
    self.assertEqual(data["training_schema_version"],"six-class-19-band-v1")
```

- [ ] **Step 2: Implement API/frontends**

Expose canonical config; keep bounds dynamic; prefer API colors; Agriculture fallback `#F59E0B`; remove fixed counts; extend v2 types.

- [ ] **Step 3: Update README/CHANGES**

Document six classes, 19 bands, all assets, seasons, report, and external-reference limits.

- [ ] **Step 4: Run full checks**

```bash
python3 -m pytest -q
npm --prefix frontend-next run typecheck
npm --prefix frontend-next run build
git diff --check
```

- [ ] **Step 5: Run/inspect frontend**

Start `python3 run_app.py` with logs at `/tmp/geo-engine-six-class.log`. Verify `/api/health`, `/api/config`, five models, six cards, Agriculture, and one small Winter and Summer Jabalpur classification; inspect console.

- [ ] **Step 6: Commit and scope audit**

Commit Task 7 files as `feat: serve six-class seasonal terrain model`. Run `git status --short`; preserve unrelated changes. Claim completion only after asset export/audit, 20 runs/report, tests/builds, and visual checks all succeed.
