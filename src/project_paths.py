"""Project paths shared by the application and analysis notebook."""

from __future__ import annotations

import os
from pathlib import Path

SOURCE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = (
    Path(os.environ.get("DIABETES_CLASSIFIER_ROOT", SOURCE_DIR.parent))
    .expanduser()
    .resolve()
)

DATASET_PATH = PROJECT_ROOT / "dataset" / "pima_indians_diabetes.csv"
MODEL_DIR = (
    Path(
        os.environ.get("DIABETES_CLASSIFIER_MODEL_DIR", PROJECT_ROOT / "out" / "models")
    )
    .expanduser()
    .resolve()
)
PIPELINE_PATH = MODEL_DIR / "diabetes_pipeline.joblib"
