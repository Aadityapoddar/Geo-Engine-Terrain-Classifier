from evaluation.runner import MODEL_NAMES, iter_run_specs
from scripts.run_seasonal_evaluation import asset_ids


def test_runner_defines_exactly_twenty_comparable_runs():
    specs = list(iter_run_specs())

    assert len(specs) == 20
    assert {spec.season for spec in specs} == {"winter", "summer"}
    assert {spec.condition for spec in specs} == {"before", "after"}
    assert {spec.model for spec in specs} == set(MODEL_NAMES)
    assert len({spec.run_id for spec in specs}) == 20


def test_run_ids_are_stable_and_readable():
    assert next(iter(iter_run_specs())).run_id == "winter-before-rf"


def test_external_evaluation_has_full_population_training_tables():
    assets = asset_ids()

    assert len(assets) == 14
    assert "winter-before-full" in assets
    assert "winter-after-full" in assets
    assert "summer-before-full" in assets
    assert "summer-after-full" in assets
