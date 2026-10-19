"""Inference API over the two trained models.

    python -m crop_suitability.predict --crop rice --N 90 --P 42 --K 43 \
        --temperature 25 --humidity 80 --ph 6.5 --rainfall 210
    python -m crop_suitability.predict --self-test
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import joblib

from . import config, features

logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    """Result of a combined suitability + recommendation query."""

    crop: str
    suitable: bool
    suitability_confidence: float | None
    recommended_crop: str
    recommendation_confidence: float | None

    def to_dict(self) -> dict:
        return {
            "crop": self.crop,
            "suitable": self.suitable,
            "suitability_confidence": self.suitability_confidence,
            "recommended_crop": self.recommended_crop,
            "recommendation_confidence": self.recommendation_confidence,
        }


@dataclass
class CropAdvisor:
    """Loads both models once and answers queries against them."""

    suitability_model_path: Path = config.SUITABILITY_MODEL_PATH
    crop_model_path: Path = config.CROP_MODEL_PATH
    suitability_model: object = field(init=False, repr=False)
    crop_model: object = field(init=False, repr=False)
    suitability_features: list[str] = field(init=False, repr=False)
    crop_categories: list[str] = field(init=False)

    def __post_init__(self) -> None:
        for path in (self.suitability_model_path, self.crop_model_path):
            if not path.exists():
                raise FileNotFoundError(
                    f"Model file not found: {path}\n"
                    "Train the models first with: python -m crop_suitability.train"
                )
        self.suitability_model = joblib.load(self.suitability_model_path)
        self.crop_model = joblib.load(self.crop_model_path)

        schema = features.load_feature_schema()
        self.suitability_features = schema["suitability_features"]
        self.crop_categories = schema["crop_categories"]
        logger.info("Loaded models for crops: %s", ", ".join(self.crop_categories))

    # ------------------------------------------------------------------ #

    def _validate(self, conditions: dict) -> dict:
        missing = [f for f in config.BASE_FEATURES if f not in conditions]
        if missing:
            raise ValueError(f"Missing required conditions: {', '.join(missing)}")
        return {k: conditions[k] for k in config.BASE_FEATURES}

    def check_suitability(self, crop: str, **conditions: float) -> tuple[bool, float | None]:
        """Is ``crop`` suitable under these conditions? Returns (verdict, confidence)."""
        crop = crop.lower().strip()
        if crop not in self.crop_categories:
            raise ValueError(
                f"Unknown crop {crop!r}. Supported: {', '.join(self.crop_categories)}"
            )
        payload = self._validate(conditions) | {config.CROP_COLUMN: crop}
        X = features.prepare_suitability_input(payload, self.suitability_features)

        verdict = bool(self.suitability_model.predict(X)[0])
        confidence = None
        if hasattr(self.suitability_model, "predict_proba"):
            proba = self.suitability_model.predict_proba(X)[0]
            confidence = float(proba[1] if verdict else proba[0])
        return verdict, confidence

    def recommend_crop(self, **conditions: float) -> tuple[str, float | None]:
        """Best-matching crop for these conditions. Returns (crop, confidence)."""
        X = features.prepare_recommendation_input(self._validate(conditions))
        crop = str(self.crop_model.predict(X)[0])
        confidence = None
        if hasattr(self.crop_model, "predict_proba"):
            confidence = float(self.crop_model.predict_proba(X)[0].max())
        return crop, confidence

    def rank_crops(self, **conditions: float) -> list[tuple[str, float]]:
        """All crops ranked by recommendation probability, highest first."""
        X = features.prepare_recommendation_input(self._validate(conditions))
        if not hasattr(self.crop_model, "predict_proba"):
            crop, _ = self.recommend_crop(**conditions)
            return [(crop, 1.0)]
        proba = self.crop_model.predict_proba(X)[0]
        pairs = zip(
            (str(c) for c in self.crop_model.classes_),
            (float(p) for p in proba),
            strict=True,
        )
        return sorted(pairs, key=lambda item: item[1], reverse=True)

    def analyse(self, crop: str, **conditions: float) -> Prediction:
        """Run both models over the same conditions."""
        suitable, suit_conf = self.check_suitability(crop, **conditions)
        recommended, rec_conf = self.recommend_crop(**conditions)
        return Prediction(
            crop=crop.lower().strip(),
            suitable=suitable,
            suitability_confidence=suit_conf,
            recommended_crop=recommended,
            recommendation_confidence=rec_conf,
        )


# --------------------------------------------------------------------------- #
# Smoke-test cases carried over from the original notebook
# --------------------------------------------------------------------------- #

SELF_TEST_CASES: list[dict] = [
    {
        "description": "Favourable for rice",
        "data": {"N": 90, "P": 42, "K": 43, "temperature": 25, "humidity": 80,
                 "ph": 6.5, "rainfall": 210, "crop": "rice"},
    },
    {
        "description": "Favourable for jute",
        "data": {"N": 80, "P": 45, "K": 40, "temperature": 26, "humidity": 85,
                 "ph": 6.8, "rainfall": 180, "crop": "jute"},
    },
    {
        "description": "Poor conditions, testing banana",
        "data": {"N": 10, "P": 10, "K": 10, "temperature": 15, "humidity": 40,
                 "ph": 5.0, "rainfall": 50, "crop": "banana"},
    },
    {
        "description": "Maize-like conditions, testing rice",
        "data": {"N": 75, "P": 50, "K": 20, "temperature": 23, "humidity": 65,
                 "ph": 6.2, "rainfall": 80, "crop": "rice"},
    },
    {
        "description": "Favourable for coconut",
        "data": {"N": 20, "P": 30, "K": 35, "temperature": 28, "humidity": 90,
                 "ph": 5.8, "rainfall": 1600, "crop": "coconut"},
    },
]


def run_self_test(advisor: CropAdvisor | None = None) -> list[Prediction]:
    """Run the reference cases and print a short report."""
    advisor = advisor or CropAdvisor()
    results: list[Prediction] = []
    for case in SELF_TEST_CASES:
        data = dict(case["data"])
        crop = data.pop("crop")
        prediction = advisor.analyse(crop, **data)
        results.append(prediction)
        verdict = "suitable" if prediction.suitable else "not suitable"
        print(f"\n{case['description']}")
        print(f"  {crop} -> {verdict}")
        print(f"  best match -> {prediction.recommended_crop}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the trained crop models.")
    parser.add_argument("--crop", help="crop to test for suitability")
    for feature in config.BASE_FEATURES:
        low, high, default = config.INPUT_RANGES[feature]
        parser.add_argument(
            f"--{feature}",
            type=float,
            default=default,
            help=f"{feature} (typical range {low}-{high}, default {default})",
        )
    parser.add_argument("--rank", action="store_true", help="show all crops ranked")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    parser.add_argument("--self-test", action="store_true", help="run the reference cases")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    if args.self_test:
        run_self_test()
        return

    advisor = CropAdvisor()
    conditions = {feature: getattr(args, feature) for feature in config.BASE_FEATURES}

    if args.crop:
        prediction = advisor.analyse(args.crop, **conditions)
        if args.json:
            print(json.dumps(prediction.to_dict(), indent=2))
        else:
            verdict = "SUITABLE" if prediction.suitable else "NOT SUITABLE"
            print(f"{prediction.crop.title()}: {verdict}")
            print(f"Best match for these conditions: {prediction.recommended_crop.title()}")
    else:
        crop, confidence = advisor.recommend_crop(**conditions)
        if args.json:
            print(json.dumps({"recommended_crop": crop, "confidence": confidence}, indent=2))
        else:
            print(f"Recommended crop: {crop.title()}")

    if args.rank:
        print("\nAll crops ranked:")
        for crop, score in advisor.rank_crops(**conditions):
            print(f"  {crop:<10} {score:6.1%}")


if __name__ == "__main__":
    main()
