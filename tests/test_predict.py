"""Tests for the inference API. Skipped when the model files are absent."""

from __future__ import annotations

import pytest

from crop_suitability import config
from crop_suitability.predict import CropAdvisor

pytestmark = pytest.mark.skipif(
    not (config.SUITABILITY_MODEL_PATH.exists() and config.CROP_MODEL_PATH.exists()),
    reason="trained models not present; run `python -m crop_suitability.train` first",
)

CONDITIONS = {
    "N": 90,
    "P": 42,
    "K": 43,
    "temperature": 25,
    "humidity": 80,
    "ph": 6.5,
    "rainfall": 210,
}


@pytest.fixture(scope="module")
def advisor() -> CropAdvisor:
    return CropAdvisor()


def test_check_suitability_returns_bool_and_confidence(advisor):
    suitable, confidence = advisor.check_suitability("rice", **CONDITIONS)
    assert isinstance(suitable, bool)
    assert confidence is None or 0.0 <= confidence <= 1.0


def test_unknown_crop_raises(advisor):
    with pytest.raises(ValueError, match="Unknown crop"):
        advisor.check_suitability("dragonfruit", **CONDITIONS)


def test_missing_condition_raises(advisor):
    partial = {k: v for k, v in CONDITIONS.items() if k != "ph"}
    with pytest.raises(ValueError, match="Missing required conditions"):
        advisor.recommend_crop(**partial)


def test_recommend_crop_returns_known_crop(advisor):
    crop, _ = advisor.recommend_crop(**CONDITIONS)
    assert crop in advisor.crop_categories


def test_rank_crops_is_sorted_and_normalised(advisor):
    ranked = advisor.rank_crops(**CONDITIONS)
    scores = [score for _, score in ranked]
    assert scores == sorted(scores, reverse=True)
    assert sum(scores) == pytest.approx(1.0, abs=1e-6)


def test_analyse_combines_both_models(advisor):
    prediction = advisor.analyse("rice", **CONDITIONS)
    assert prediction.crop == "rice"
    assert prediction.recommended_crop in advisor.crop_categories
