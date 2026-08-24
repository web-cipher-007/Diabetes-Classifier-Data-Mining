"""Load the trained pipeline and prepare patient data for inference."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import joblib
import pandas as pd

from project_paths import PIPELINE_PATH

FEATURE_NAMES = (
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
)


@dataclass(frozen=True)
class Prediction:
    """A binary prediction and its diabetes probability."""

    is_diabetic: bool
    diabetes_probability: float

    @property
    def non_diabetes_probability(self) -> float:
        return 1.0 - self.diabetes_probability


def load_pipeline() -> Any:
    """Load the fitted preprocessing and classification pipeline."""

    if not PIPELINE_PATH.is_file():
        raise FileNotFoundError(f"Missing model artifact: {PIPELINE_PATH}")

    return joblib.load(PIPELINE_PATH)


def prepare_features(values: Mapping[str, float]) -> pd.DataFrame:
    """Return one model-ready row in the feature order used during training."""

    missing_features = [name for name in FEATURE_NAMES if name not in values]
    if missing_features:
        raise ValueError(f"Missing feature(s): {', '.join(missing_features)}")

    processed: dict[str, float] = {}
    for name in FEATURE_NAMES:
        processed[name] = float(values[name])

    return pd.DataFrame([processed], columns=FEATURE_NAMES)


def predict(values: Mapping[str, float], pipeline: Any) -> Prediction:
    """Prepare patient values and return the pipeline prediction."""

    features = prepare_features(values)
    predicted_class = int(pipeline.predict(features)[0])
    diabetes_probability = float(pipeline.predict_proba(features)[0, 1])

    return Prediction(
        is_diabetic=predicted_class == 1,
        diabetes_probability=diabetes_probability,
    )
