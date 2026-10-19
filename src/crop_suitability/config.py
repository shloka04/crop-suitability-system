"""Central configuration: paths, feature definitions and labelling thresholds.

Every other module imports from here, so there are no hard-coded Colab paths
anywhere in the codebase.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

# <repo root>/src/crop_suitability/config.py -> parents[2] is the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Raw inputs
CROP_RECOMMENDATION_CSV = RAW_DATA_DIR / "Crop_recommendation.csv"
CROP_YIELD_CSV = RAW_DATA_DIR / "crop_yield.csv"
SUITABILITY_REFERENCE_CSV = RAW_DATA_DIR / "final_suitability_dataset.csv"

# Derived artefacts (git-ignored: regenerate with `python -m crop_suitability.data`)
MERGED_DATASET_CSV = PROCESSED_DATA_DIR / "final_merged_dataset.csv"

# Trained artefacts
SUITABILITY_MODEL_PATH = MODELS_DIR / "best_suitability_model.pkl"
CROP_MODEL_PATH = MODELS_DIR / "best_crop_recommendation_model.pkl"
FEATURE_SCHEMA_PATH = MODELS_DIR / "feature_schema.json"
METRICS_PATH = REPORTS_DIR / "metrics.json"

# --------------------------------------------------------------------------- #
# Features and targets
# --------------------------------------------------------------------------- #

#: Soil and weather measurements shared by both models.
BASE_FEATURES: list[str] = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall",
]

#: Crops retained after fuzzy-matching the two source datasets.
CROP_CATEGORIES: list[str] = ["banana", "coconut", "jute", "maize", "rice"]

#: One-hot column layout the suitability model was trained on.
SUITABILITY_FEATURES: list[str] = BASE_FEATURES + [
    f"crop_{crop}" for crop in CROP_CATEGORIES
]

CROP_COLUMN = "crop"
SUITABILITY_COLUMN = "Suitability"
YIELD_COLUMN = "Yield"

# --------------------------------------------------------------------------- #
# Labelling
# --------------------------------------------------------------------------- #

#: A crop/condition row is labelled suitable (1) when the recorded yield meets
#: or exceeds this per-crop threshold. Units are tonnes/hectare except coconut,
#: which the source dataset records in nuts/hectare.
#:
#: Sources:
#:   rice    - Economic Survey of India 2024-25
#:   maize   - Agmarknet maize crop profile
#:   banana  - National Horticulture Board banana report
#:   jute    - Directorate of Jute Development
#:   coconut - Coconut Development Board, 2025
YIELD_THRESHOLDS: dict[str, float] = {
    "rice": 2.5,
    "maize": 4.0,
    "banana": 40.0,
    "jute": 2.8,
    "coconut": 12_000.0,
}

#: Minimum fuzzy-match score (0-100) for mapping a crop name across datasets.
FUZZY_MATCH_THRESHOLD = 80

# --------------------------------------------------------------------------- #
# Modelling
# --------------------------------------------------------------------------- #

RANDOM_STATE = 42
TEST_SIZE = 0.2

LGBM_PARAMS: dict[str, object] = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": -1,
    "num_leaves": 31,
    "random_state": RANDOM_STATE,
    "verbose": -1,
}

EARLY_STOPPING_ROUNDS = 50

#: UI slider ranges for the Gradio app: (minimum, maximum, default).
INPUT_RANGES: dict[str, tuple[float, float, float]] = {
    "N": (0, 150, 90),
    "P": (0, 150, 42),
    "K": (0, 250, 43),
    "temperature": (10, 45, 25),
    "humidity": (20, 100, 80),
    "ph": (3.0, 10.0, 6.5),
    "rainfall": (20, 300, 210),
}


def ensure_directories() -> None:
    """Create the output directories the pipeline writes to."""
    for directory in (PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)
