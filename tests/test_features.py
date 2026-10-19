"""Tests for feature construction and schema alignment."""

from __future__ import annotations

import pandas as pd
import pytest

from crop_suitability import config, features
from crop_suitability.data import add_suitability_label


@pytest.fixture
def sample_conditions() -> dict:
    return {
        "N": 90,
        "P": 42,
        "K": 43,
        "temperature": 25,
        "humidity": 80,
        "ph": 6.5,
        "rainfall": 210,
        "crop": "rice",
    }


def test_prepare_suitability_input_matches_schema(sample_conditions):
    X = features.prepare_suitability_input(sample_conditions)
    assert list(X.columns) == config.SUITABILITY_FEATURES
    assert len(X) == 1


def test_prepare_suitability_input_sets_only_the_chosen_crop(sample_conditions):
    X = features.prepare_suitability_input(sample_conditions)
    assert X["crop_rice"].iloc[0] == 1
    for crop in config.CROP_CATEGORIES:
        if crop != "rice":
            assert X[f"crop_{crop}"].iloc[0] == 0


def test_prepare_recommendation_input_drops_the_crop_column(sample_conditions):
    X = features.prepare_recommendation_input(sample_conditions)
    assert list(X.columns) == config.BASE_FEATURES
    assert "crop" not in X.columns


def test_align_to_schema_adds_missing_and_drops_extra():
    df = pd.DataFrame([{"a": 1, "b": 2, "unexpected": 3}])
    aligned = features.align_to_schema(df, ["a", "b", "c"])
    assert list(aligned.columns) == ["a", "b", "c"]
    assert aligned["c"].iloc[0] == 0


def test_suitability_label_respects_thresholds():
    df = pd.DataFrame(
        {
            "crop": ["rice", "rice", "maize", "unknown_crop"],
            "Yield": [3.0, 1.0, 5.0, 9_999.0],
        }
    )
    labelled = add_suitability_label(df)
    # rice threshold 2.5, maize 4.0, unconfigured crops never count as suitable.
    assert labelled[config.SUITABILITY_COLUMN].tolist() == [1, 0, 1, 0]
