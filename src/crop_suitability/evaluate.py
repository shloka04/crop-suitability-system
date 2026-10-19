"""Evaluation: metrics, confusion matrices and SHAP feature attributions.

    python -m crop_suitability.evaluate            # metrics + saved figures
    python -m crop_suitability.evaluate --shap     # also compute SHAP summaries
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # figures are written to disk, never shown interactively

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

from . import config, features
from .data import load_dataset

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def binary_metrics(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
        "report": classification_report(y_test, y_pred, zero_division=0, output_dict=True),
    }


def multiclass_metrics(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "report": classification_report(y_test, y_pred, zero_division=0, output_dict=True),
    }


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def plot_confusion_matrix(
    y_true,
    y_pred,
    labels: list | None = None,
    title: str = "Confusion matrix",
    output_path: Path | None = None,
) -> Path | None:
    """Draw a confusion matrix heatmap and save it under reports/figures."""
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels or "auto",
        yticklabels=labels or "auto",
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        logger.info("Saved %s", output_path)
    plt.close(fig)
    return output_path


def plot_yield_distribution(df: pd.DataFrame, output_path: Path | None = None) -> Path | None:
    """Box plot of yield per crop, with the labelling thresholds overlaid."""
    output_path = output_path or config.FIGURES_DIR / "yield_distribution.png"
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(
        data=df,
        x=config.CROP_COLUMN,
        y=config.YIELD_COLUMN,
        hue=config.SUITABILITY_COLUMN,
        showfliers=False,
        ax=ax,
    )
    ax.set_yscale("symlog")  # coconut yields are orders of magnitude larger
    ax.set_title("Yield distribution by crop and suitability label")
    ax.set_ylabel("Yield (t/ha; nuts/ha for coconut, log scale)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", output_path)
    return output_path


def plot_shap_summary(
    model,
    X: pd.DataFrame,
    title: str,
    output_path: Path,
    max_samples: int = 2_000,
) -> Path | None:
    """SHAP summary plot for a tree model, on a subsample for tractability."""
    try:
        import shap
    except ImportError:
        logger.warning("shap is not installed; skipping %s", title)
        return None

    sample = X.sample(n=min(max_samples, len(X)), random_state=config.RANDOM_STATE)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    plt.figure()
    shap.summary_plot(shap_values, sample, show=False)
    plt.title(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", output_path)
    return output_path


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the trained models.")
    parser.add_argument("--shap", action="store_true", help="compute SHAP summary plots")
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="evaluate on a random sample of N rows",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config.ensure_directories()

    df = load_dataset()
    if args.sample:
        df = df.sample(n=min(args.sample, len(df)), random_state=config.RANDOM_STATE)

    suit_model = joblib.load(config.SUITABILITY_MODEL_PATH)
    crop_model = joblib.load(config.CROP_MODEL_PATH)

    # --- Suitability ---
    X1 = features.build_suitability_features(df)
    y1 = df[config.SUITABILITY_COLUMN]
    _, X1_test, _, y1_test = train_test_split(
        X1, y1, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y1
    )
    y1_pred = suit_model.predict(X1_test)
    print("\n=== Suitability ===")
    print(classification_report(y1_test, y1_pred, zero_division=0))
    plot_confusion_matrix(
        y1_test,
        y1_pred,
        labels=[0, 1],
        title="Suitability prediction",
        output_path=config.FIGURES_DIR / "confusion_suitability.png",
    )

    # --- Recommendation ---
    X2 = features.build_recommendation_features(df)
    y2 = df[config.CROP_COLUMN]
    _, X2_test, _, y2_test = train_test_split(
        X2, y2, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y2
    )
    y2_pred = crop_model.predict(X2_test)
    print("\n=== Crop recommendation ===")
    print(classification_report(y2_test, y2_pred, zero_division=0))
    plot_confusion_matrix(
        y2_test,
        y2_pred,
        labels=list(crop_model.classes_),
        title="Crop recommendation",
        output_path=config.FIGURES_DIR / "confusion_recommendation.png",
    )

    plot_yield_distribution(df)

    if args.shap:
        plot_shap_summary(
            suit_model, X1_test, "Suitability SHAP", config.FIGURES_DIR / "shap_suitability.png"
        )
        plot_shap_summary(
            crop_model,
            X2_test,
            "Recommendation SHAP",
            config.FIGURES_DIR / "shap_recommendation.png",
        )


if __name__ == "__main__":
    main()
