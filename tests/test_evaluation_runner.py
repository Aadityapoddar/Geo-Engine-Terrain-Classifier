from evaluation.runner import MODEL_NAMES, iter_run_specs


def test_runner_defines_exactly_twenty_comparable_runs():
    specs = list(iter_run_specs())

    assert len(specs) == 20
    assert {spec.season for spec in specs} == {"winter", "summer"}
    assert {spec.condition for spec in specs} == {"before", "after"}
    assert {spec.model for spec in specs} == set(MODEL_NAMES)
    assert len({spec.run_id for spec in specs}) == 20


def test_run_ids_are_stable_and_readable():
    assert next(iter(iter_run_specs())).run_id == "winter-before-rf"
