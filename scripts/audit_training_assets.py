#!/usr/bin/env python3
"""Write a strict live audit of Before and After Earth Engine training assets."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.config import TRAINING_SCHEMA_VERSION  # noqa: E402
from evaluation.assets import audit_all_training_assets  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    payload = {
        "training_schema_version": TRAINING_SCHEMA_VERSION,
        "assets": audit_all_training_assets(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    for asset in payload["assets"]:
        print(
            asset["name"],
            f"size={asset['size']}",
            f"labels={asset['label_histogram']}",
            f"valid={asset['seasonal_valid_samples']}",
        )


if __name__ == "__main__":
    main()
