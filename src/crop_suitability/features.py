"""Feature construction shared by training and inference.

Training and serving must one-hot encode the crop column identically, so both
paths go through :func:`encode_crop` / :func:`align_to_schema` rather than
calling ``pd.get_dummies`` ad hoc.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import config


def encode_crop(df: pd.DataFrame, crop_column: str = config.CROP_COLUMN) -> pd.DataFrame:
    """One-hot encode the crop column, producing ``crop_<name>`` indicators."""
    return pd.get_dummies(df, columns=[crop_column])


def align_to_schema(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return ``df`` with exactly ``columns``, in order.

    Columns absent from ``df`` are added as zeros (the normal case for one-hot
    crop indicators at inference time); extra columns are dropped.
    """
    aligned = df.copy()
    for column in columns:
        if column not in aligned.columns:
            aligned[column] = 0
    return aligned[columns]


def build_suitability_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the suitability model's design matrix from the merged dataset."""
    subset = df[config.BASE_FEATURES + [config.CROP_COLUMN]].copy()
    encoded = encode_crop(subset)
    return align_to_schema(encoded, config.SUITABILITY_FEATURES)


def build_recommendation_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the crop recommendation model's design matrix (soil/weather only)."""
    return df[config.BASE_FEATURES].copy()


def prepare_suitability_input(
    conditions: dict[str, float | str],
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Turn a single dict of conditions into a one-row suitability design matrix."""
    columns = columns or config.SUITABILITY_FEATURES
    encoded = encode_crop(pd.DataFrame([conditions]))
    return align_to_schema(encoded, columns)


def prepare_recommendation_input(conditions: dict[str, float | str]) -> pd.DataFrame:
    """Turn a single dict of conditions into a one-row recommendation matrix."""
    row = {feature: conditions[feature] for feature in config.BASE_FEATURES}
    return pd.DataFrame([row])


# --------------------------------------------------------------------------- #
# Schema persistence
# --------------------------------------------------------------------------- #


def save_feature_schema(
    suitability_columns: list[str],
    recommendation_columns: list[str],
    crop_categories: list[str],
    path: Path | None = None,
) -> Path:
    """Persist the exact column layout the models were trained on."""
    path = path or config.FEATURE_SCHEMA_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "suitability_features": list(suitability_columns),
        "recommendation_features": list(recommendation_columns),
        "crop_categories": list(crop_categories),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_feature_schema(path: Path | None = None) -> dict[str, list[str]]:
    """Load the persisted schema, falling back to the configured defaults.

    The fallback keeps pre-packaged model files usable even though they were
    trained before the schema was written out.
    """
    path = path or config.FEATURE_SCHEMA_PATH
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "suitability_features": list(config.SUITABILITY_FEATURES),
        "recommendation_features": list(config.BASE_FEATURES),
        "crop_categories": list(config.CROP_CATEGORIES),
    }
