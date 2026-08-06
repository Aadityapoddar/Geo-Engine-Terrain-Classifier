"""Canonical evaluation run matrix."""

from dataclasses import dataclass


MODEL_NAMES = ("rf", "svm", "xgb", "cart", "knn")
SEASON_NAMES = ("winter", "summer")
CONDITIONS = ("before", "after")


@dataclass(frozen=True)
class RunSpec:
    season: str
    condition: str
    model: str

    @property
    def run_id(self):
        return f"{self.season}-{self.condition}-{self.model}"


def iter_run_specs():
    """Yield the fixed 20-run matrix in stable report order."""
    for season in SEASON_NAMES:
        for condition in CONDITIONS:
            for model in MODEL_NAMES:
                yield RunSpec(season=season, condition=condition, model=model)

