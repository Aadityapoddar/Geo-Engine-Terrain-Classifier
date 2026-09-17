"""The manuscript's headline numbers, checked against the frozen archive.

Every figure asserted here was wrong in an earlier draft of new.tex and was
corrected against doc/assets/result_archive. The point of the file is that the
next edit which reintroduces one of those errors fails a test instead of
reaching a reviewer.
"""
import csv
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "doc" / "assets" / "result_archive"
TEX = ROOT / "new.tex"
CLASSES = ["vegetation", "water", "builtarea", "openland", "agriculture"]


def _rows(name):
    with (ARCHIVE / f"{name}.csv").open() as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def pooled():
    """Pooled test-district confusion matrix per season for the reported model."""
    out = {}
    for row in _rows("district_confusion_matrices"):
        if row["model_key"] != "after-gtb" or row["role"] != "test":
            continue
        matrix = out.setdefault(row["season"], np.zeros((5, 5), dtype=int))
        for i, ref in enumerate(CLASSES):
            for j, pred in enumerate(CLASSES):
                matrix[i, j] += int(row[f"c_{ref}_{pred}"])
    return out


@pytest.fixture(scope="module")
def tex():
    return TEX.read_text()


def _balanced(matrix):
    support = matrix.sum(axis=1)
    return (matrix.diagonal()[support > 0] / support[support > 0]).mean()


def test_summer_sample_is_not_class_balanced(pooled):
    """The 208 missing summer points are all vegetation, not agriculture/open land."""
    support = dict(zip(CLASSES, pooled["summer"].sum(axis=1).tolist()))
    assert support["vegetation"] == 372
    assert {c: n for c, n in support.items() if c != "vegetation"} == {
        "water": 580, "builtarea": 580, "openland": 580, "agriculture": 580
    }
    assert sum(support.values()) == 2692
    assert dict(zip(CLASSES, pooled["winter"].sum(axis=1).tolist())) == {
        c: 580 for c in CLASSES
    }


def test_pooled_and_balanced_summer_accuracy_differ(pooled, tex):
    summer, winter = pooled["summer"], pooled["winter"]
    assert summer.diagonal().sum() / summer.sum() == pytest.approx(0.738484, abs=5e-7)
    assert _balanced(summer) == pytest.approx(0.717138, abs=5e-7)
    # winter is exactly balanced, so its two readings must coincide
    assert _balanced(winter) == pytest.approx(winter.diagonal().sum() / winter.sum())
    assert "71.71" in tex, "class-balanced summer accuracy must be reported"
    assert "class-balanced sample agreement of 87.41" not in tex


def test_weighted_domain_is_dominated_by_agriculture(tex):
    """The area-weighted estimator admits a constant-agriculture baseline."""
    shares = {}
    for season in ("winter", "summer"):
        areas = {}
        for row in _rows("stratum_areas"):
            if row["season"] != season or row["role"] != "test":
                continue
            areas[row["consensus_class"]] = areas.get(row["consensus_class"], 0.0) + float(
                row["eligible_area_km2"]
            )
        shares[season] = areas["Agriculture"] / sum(areas.values()) * 100
    # measured on the 10 m sampling grid, not the 30 m grid an earlier
    # version weighted with: the coarse grid read 95.92 and 98.40
    assert shares["winter"] == pytest.approx(95.86, abs=0.01)
    assert shares["summer"] == pytest.approx(98.39, abs=0.01)
    # both must appear, since the trained pipeline scores 75.50 and 58.67 here
    assert "95.86" in tex and "98.39" in tex


def test_protocol_effect_is_smaller_than_the_classifier_spread():
    """The conclusion may not claim protocol choice outweighs classifier choice.

    The record now carries eight contiguous-block realisations per model, so the
    effect is a mean over realisations and the spread is a mean over the same
    realisations. Reading one row per model, as this test used to, silently
    reported whichever realisation happened to be written last.
    """
    accuracy = {}
    for row in _rows("protocol_comparison"):
        if row["season"] != "winter":
            continue
        accuracy.setdefault(row["protocol"], {}).setdefault(
            int(row["realisation"]), {})[row["model"]] = float(
                row["overall_accuracy"])
    random_, blocked = accuracy["random"], accuracy["spatially_blocked"]
    realisations = sorted(set(random_) & set(blocked))
    assert len(realisations) == 8, realisations

    effects = [random_[r][m] - blocked[r][m]
               for r in realisations for m in random_[r]]
    protocol_effect = np.mean(effects) * 100
    assert protocol_effect == pytest.approx(5.67, abs=0.01)

    for fold in (random_, blocked):
        spread = np.mean([max(fold[r].values()) - min(fold[r].values())
                          for r in realisations]) * 100
        assert spread > protocol_effect


def test_per_class_f1_ordering_is_season_dependent(pooled):
    """Summer vegetation sits fourth, not second: the ordering is not constant."""
    def f1(matrix):
        return {
            c: 2 * matrix[i, i] / (matrix[i].sum() + matrix[:, i].sum())
            for i, c in enumerate(CLASSES)
        }
    winter = sorted(f1(pooled["winter"]), key=f1(pooled["winter"]).get, reverse=True)
    summer = sorted(f1(pooled["summer"]), key=f1(pooled["summer"]).get, reverse=True)
    assert winter == ["water", "vegetation", "builtarea", "openland", "agriculture"]
    assert summer == ["water", "builtarea", "openland", "vegetation", "agriculture"]
    # the summer vegetation row carries 208 of 704 errors, so "almost entirely
    # open land and agriculture" is false for that season
    errors = pooled["summer"].sum(axis=1) - pooled["summer"].diagonal()
    assert errors[CLASSES.index("vegetation")] == 208
    assert errors.sum() == 704


def test_protocol_experiment_used_the_alt19_stack():
    """jabalpur_split_protocols.py selects b19, and b19 is alt19, not sel19."""
    from backend.config import BAND_STACKS, BANDS

    assert set(BAND_STACKS["b19"]) != set(BANDS)
    assert set(BANDS) - set(BAND_STACKS["b19"]) == {
        "B4", "B8", "NDVI", "NDWI", "SAVI", "s_ent"
    }
    source = (ROOT / "scripts" / "jabalpur_split_protocols.py").read_text()
    assert 'BAND_STACKS["b19"]' in source


def test_reference_screening_is_not_uniform_across_classes():
    """Agriculture skips Dynamic World; open land skips the confidence threshold."""
    from evaluation.references import DW_MIN_PROBABILITY, DW_SCREEN

    rules = {wc: (dw, threshold) for wc, dw, threshold in DW_SCREEN}
    assert set(rules) == {10, 80, 50, 60, 40}
    # tree cover, water and built surface clear the full condition
    for code, label in ((10, 1), (80, 0), (50, 6)):
        assert rules[code] == (label, DW_MIN_PROBABILITY)
    # open land carries the label but no threshold; agriculture neither
    assert rules[60] == (7, None)
    assert rules[40] == (None, None)


def test_worldcereal_is_the_static_annual_product():
    """No seasonal selection happens, whatever the season argument says."""
    from evaluation.references import WORLD_CEREAL_PRODUCT

    assert WORLD_CEREAL_PRODUCT == "temporarycrops"
    body = (ROOT / "evaluation" / "references.py").read_text().split(
        "def _world_cereal", 1
    )[1].split("def ", 1)[0]
    assert "season" not in body.split("return", 1)[1]


def test_corrected_claims_are_absent_from_the_manuscript(tex):
    """Strings that were factually wrong must not reappear."""
    for wrong in (
        "winter-cereals",
        "main-maize",
        "33 development",
        "RF, SVM, GTB",
        "535 point-level",
        "the full stack should be the worst",
    ):
        assert wrong not in tex, f"corrected claim reintroduced: {wrong!r}"


# ── claims added by the second-round audit ────────────────────────────────
def test_cart_is_separated_from_every_classifier(tex):
    """The CART claim needs the six contrasts the leader family does not contain."""
    rows = _rows("pairwise_contrasts_all")
    assert len(rows) == 20, len(rows)
    for row in rows:
        if "CART" not in row["contrast"]:
            continue
        assert float(row["wilcoxon_p_holm"]) < 0.001, row
    # and the summer leader-versus-RF margin does not survive the larger family
    summer_rf = [r for r in rows if r["season"] == "summer"
                 and r["contrast"] == "Smile GTB - RF"][0]
    assert float(summer_rf["wilcoxon_p_holm"]) > 0.05
    assert "0.061" in tex, "the weaker adjusted p must be reported"


def test_agriculture_reference_is_static_between_seasons(tex):
    """Eq. (6) reads no seasonal layer for agriculture, so its strata cannot move."""
    areas = {}
    for row in _rows("stratum_areas"):
        if row["consensus_class"] != "Agriculture":
            continue
        areas.setdefault((row["district"]), {})[row["season"]] = float(
            row["eligible_area_km2"])
    assert areas, "no agriculture strata found"
    for district, seasons in areas.items():
        assert seasons["winter"] == seasons["summer"], district
    assert "static in its entirety" in tex


def test_fixed_location_seasonal_control_is_reported(tex):
    """A seasonal fall measured at fixed reference locations, not just pooled."""
    import json
    record = json.loads((ROOT / "doc" / "assets"
                         / "fixed_location_seasonal.json").read_text())
    assert record["shared_locations"] == 1350
    assert record["locations_dropped_for_label_change"] == 0
    gtb = record["models"]["Smile GTB"]
    assert gtb["seasonal_drop_pp"] == pytest.approx(8.67, abs=0.01)
    # smaller than the pooled 13.56-point fall, which is the whole point
    assert gtb["seasonal_drop_pp"] < 13.56
    assert "1{,}350 locations" in tex or "1,350 locations" in tex


def test_summer_shortfall_is_attributed_per_district(tex):
    """Scarcity explains most of the missing vegetation points, but not all."""
    import json
    record = json.loads((ROOT / "doc" / "assets"
                         / "reference_shortfall_audit.json").read_text())
    summer = record["seasons"]["summer"]
    assert summer["missing_points"] == 208
    assert summer["classes_affected"] == ["Vegetation"]
    assert record["seasons"]["winter"]["strata_below_quota"] == 0
    # on the sampling grid every short stratum is genuinely scarce: no district
    # holds the twenty eligible pixels the quota asks for. The 30 m measurement
    # said otherwise, which is the artefact the manuscript now records.
    assert summer["short_strata_with_at_least_1000_eligible_pixels"] == 0
    assert summer["short_strata_with_fewer_than_20_eligible_pixels"] == 12
    assert "Bhind" in tex
