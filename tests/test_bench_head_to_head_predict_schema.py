import runpy
import sys
from pathlib import Path

import pytest


def _load_mod():
    scripts_dir = (Path(__file__).resolve().parents[1] / "scripts").as_posix()
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    return runpy.run_path("scripts/bench_h2h_predict.py")


def test_predictions_schema_rejects_duplicate_task_budget_trial_rows():
    mod = _load_mod()
    validate_duplicates = mod["_validate_no_duplicate_prediction_rows"]

    rows = [
        {"task_id": "retrieval:R01", "budget_tokens": 2000, "trial": 1},
        {"task_id": "retrieval:R01", "budget_tokens": 2000, "trial": 1},
    ]

    with pytest.raises(ValueError, match="duplicate prediction row"):
        validate_duplicates(rows)
