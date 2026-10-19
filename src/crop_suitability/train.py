"""Train the suitability and crop-recommendation models.

    python -m crop_suitability.train                # both models
    python -m crop_suitability.train --task suitability
    python -m crop_suitability.train --cv           # add cross-validated scores
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from . import config, features
from .data import load_dataset

logger = logging.getLogger(__name__)


def train_lgbm(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    params: dict | None = None,
) -> LGBMClassifier:
    """Fit a LightGBM classifier with early stopping on the held-out split."""
    model = LGBMClassifier(**(params or config.LGBM_PARAMS))
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="logloss",
        callbacks=[early_stopping(config.EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    return model


def _split(X: pd.DataFrame, y: pd.Series):
    return train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )


def train_suitability_model(df: pd.DataFrame) -> tuple[LGBMClassifier, dict]:
    """Train the binary "is this crop suitable here?" classifier."""
    X = features.build_suitability_features(df)
    y = df[config.SUITABILITY_COLUMN]
    X_train, X_test, y_train, y_test = _split(X, y)

    logger.info("Training suitability model on %s rows", len(X_train))
    model = train_lgbm(X_train, X_test, y_train, y_test)

    from .evaluate import binary_metrics  # local import avoids a cycle

    metrics = binary_metrics(model, X_test, y_test)
    logger.info("Suitability test accuracy: %.4f", metrics["accuracy"])
    return model, metrics


def train_recommendation_model(df: pd.DataFrame) -> tuple[LGBMClassifier, dict]:
    """Train the multi-class "which crop fits these conditions?" classifier."""
    X = features.build_recommendation_features(df)
    y = df[config.CROP_COLUMN]
    X_train, X_test, y_train, y_test = _split(X, y)

    logger.info("Training recommendation model on %s rows", len(X_train))
    model = train_lgbm(X_train, X_test, y_train, y_test)

    from .evaluate import multiclass_metrics

    metrics = multiclass_metrics(model, X_test, y_test)
    logger.info("Recommendation test accuracy: %.4f", metrics["accuracy"])
    return model, metrics


def cross_validate_recommendation(df: pd.DataFrame, n_splits: int = 5) -> dict:
    """Stratified k-fold macro-F1 for the recommendation model."""
    X = features.build_recommendation_features(df)
    y = df[config.CROP_COLUMN]
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)
    model = LGBMClassifier(**{**config.LGBM_PARAMS, "n_estimators": 300})
    scores = cross_val_score(model, X, y, cv=cv, scoring="f1_macro")
    logger.info("CV macro-F1: %.4f (+/- %.4f)", scores.mean(), scores.std())
    return {"f1_macro_folds": scores.tolist(), "f1_macro_mean": float(np.mean(scores))}


def save_model(model: LGBMClassifier, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Saved %s", path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the crop models.")
    parser.add_argument(
        "--task",
        choices=["suitability", "recommendation", "both"],
        default="both",
        help="which model to train (default: both)",
    )
    parser.add_argument(
        "--rebuild-data",
        action="store_true",
        help="regenerate the merged dataset before training",
    )
    parser.add_argument(
        "--cv",
        action="store_true",
        help="also report cross-validated macro-F1 for the recommendation model",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="train on a random sample of N rows (useful for a quick smoke test)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config.ensure_directories()

    df = load_dataset(rebuild=args.rebuild_data)
    if args.sample:
        df = df.sample(n=min(args.sample, len(df)), random_state=config.RANDOM_STATE)
        logger.info("Sampled %s rows for a quick run", len(df))

    metrics: dict[str, dict] = {}

    if args.task in ("suitability", "both"):
        model, metrics["suitability"] = train_suitability_model(df)
        save_model(model, config.SUITABILITY_MODEL_PATH)

    if args.task in ("recommendation", "both"):
        model, metrics["recommendation"] = train_recommendation_model(df)
        save_model(model, config.CROP_MODEL_PATH)
        if args.cv:
            metrics["recommendation"]["cross_validation"] = cross_validate_recommendation(df)

    crops = sorted(df[config.CROP_COLUMN].unique())
    features.save_feature_schema(
        suitability_columns=config.BASE_FEATURES + [f"crop_{c}" for c in crops],
        recommendation_columns=config.BASE_FEATURES,
        crop_categories=crops,
    )

    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote metrics to %s", config.METRICS_PATH)


if __name__ == "__main__":
    main()
