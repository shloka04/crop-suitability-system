"""Dataset loading, crop-name reconciliation, merging and label construction.

Run as a script to regenerate the merged training table:

    python -m crop_suitability.data
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from . import config

logger = logging.getLogger(__name__)

# rapidfuzz is the maintained successor to fuzzywuzzy and needs no C extension.
try:  # pragma: no cover - import shim
    from rapidfuzz import fuzz, process

    _RAPIDFUZZ = True
except ImportError:  # pragma: no cover - fallback for older environments
    from fuzzywuzzy import fuzz, process  # type: ignore[no-redef]

    _RAPIDFUZZ = False


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_crop_recommendation(path: Path | None = None) -> pd.DataFrame:
    """Load the soil/weather dataset and normalise its crop column.

    The source file names the target column ``label``; it is renamed to
    ``crop`` so both datasets share a join key.
    """
    path = path or config.CROP_RECOMMENDATION_CSV
    df = pd.read_csv(path)
    df = df.rename(columns={"label": config.CROP_COLUMN})
    df[config.CROP_COLUMN] = df[config.CROP_COLUMN].str.lower().str.strip()
    logger.info("Loaded crop recommendation data: %s rows", len(df))
    return df


def load_crop_yield(path: Path | None = None) -> pd.DataFrame:
    """Load the district-level yield dataset and normalise its crop column."""
    path = path or config.CROP_YIELD_CSV
    df = pd.read_csv(path)
    df = df.rename(columns={"Crop": config.CROP_COLUMN})
    df[config.CROP_COLUMN] = df[config.CROP_COLUMN].str.lower().str.strip()
    # The source file carries stray whitespace in the Season labels.
    if "Season" in df.columns:
        df["Season"] = df["Season"].str.strip()
    logger.info("Loaded crop yield data: %s rows", len(df))
    return df


# --------------------------------------------------------------------------- #
# Crop-name reconciliation
# --------------------------------------------------------------------------- #


def build_crop_name_mapping(
    source_crops: list[str],
    target_crops: list[str],
    threshold: int = config.FUZZY_MATCH_THRESHOLD,
) -> dict[str, str]:
    """Map crop names in ``source_crops`` onto their closest ``target_crops``.

    Names scoring below ``threshold`` are dropped rather than guessed at.
    """
    mapping: dict[str, str] = {}
    for crop in source_crops:
        best = process.extractOne(crop, target_crops, scorer=fuzz.token_sort_ratio)
        if best is None:
            continue
        match, score = best[0], best[1]
        if score >= threshold:
            mapping[crop] = match
        else:
            logger.debug("No match above %s for crop %r (best %s)", threshold, crop, score)
    logger.info(
        "Matched %s of %s crop names (fuzzy backend: %s)",
        len(mapping),
        len(source_crops),
        "rapidfuzz" if _RAPIDFUZZ else "fuzzywuzzy",
    )
    return mapping


def align_crop_names(
    df_rec: pd.DataFrame,
    df_yield: pd.DataFrame,
    threshold: int = config.FUZZY_MATCH_THRESHOLD,
) -> pd.DataFrame:
    """Rewrite the recommendation dataset's crop names to the yield dataset's."""
    mapping = build_crop_name_mapping(
        sorted(df_rec[config.CROP_COLUMN].unique()),
        sorted(df_yield[config.CROP_COLUMN].unique()),
        threshold=threshold,
    )
    aligned = df_rec.copy()
    aligned[config.CROP_COLUMN] = aligned[config.CROP_COLUMN].map(mapping)
    return aligned.dropna(subset=[config.CROP_COLUMN])


# --------------------------------------------------------------------------- #
# Merge and label
# --------------------------------------------------------------------------- #


def merge_datasets(df_rec: pd.DataFrame, df_yield: pd.DataFrame) -> pd.DataFrame:
    """Inner-join the aligned datasets on crop name.

    Note: the join key is the crop name alone, so every soil/weather reading for
    a crop is paired with every yield record for that crop. This is a deliberate
    cross product that expands a few thousand rows into a few hundred thousand;
    see the "Known limitations" section of the README before reading much into
    the headline accuracy figures.
    """
    merged = pd.merge(df_rec, df_yield, on=config.CROP_COLUMN, how="inner")
    logger.info("Merged dataset: %s rows x %s columns", *merged.shape)
    return merged


def add_suitability_label(
    df: pd.DataFrame,
    thresholds: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Add the binary ``Suitability`` target derived from per-crop yield cut-offs.

    A crop with no configured threshold is labelled 0 (never suitable) rather
    than silently dropped, matching the original notebook's behaviour.
    """
    thresholds = thresholds or config.YIELD_THRESHOLDS
    labelled = df.copy()
    cutoffs = labelled[config.CROP_COLUMN].map(thresholds).fillna(float("inf"))
    labelled[config.SUITABILITY_COLUMN] = (
        labelled[config.YIELD_COLUMN] >= cutoffs
    ).astype(int)
    share = labelled[config.SUITABILITY_COLUMN].mean()
    logger.info("Labelled %.1f%% of rows as suitable", 100 * share)
    return labelled


def yield_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Per-crop yield summary, useful for sanity-checking the thresholds."""
    return (
        df.groupby(config.CROP_COLUMN)[config.YIELD_COLUMN]
        .agg(["count", "min", "max", "mean", "median", "std"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )


# --------------------------------------------------------------------------- #
# Pipeline entry point
# --------------------------------------------------------------------------- #


def build_dataset(save: bool = True) -> pd.DataFrame:
    """Run the full data pipeline and optionally persist the result."""
    df_rec = load_crop_recommendation()
    df_yield = load_crop_yield()
    aligned = align_crop_names(df_rec, df_yield)
    merged = merge_datasets(aligned, df_yield)
    labelled = add_suitability_label(merged)

    if save:
        config.ensure_directories()
        labelled.to_csv(config.MERGED_DATASET_CSV, index=False)
        logger.info("Wrote %s", config.MERGED_DATASET_CSV)
    return labelled


def load_dataset(rebuild: bool = False) -> pd.DataFrame:
    """Return the merged, labelled dataset, building it if it is missing."""
    if rebuild or not config.MERGED_DATASET_CSV.exists():
        return build_dataset(save=True)
    df = pd.read_csv(config.MERGED_DATASET_CSV)
    if config.SUITABILITY_COLUMN not in df.columns:
        df = add_suitability_label(df)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the merged crop dataset.")
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="build the dataset in memory without writing it to data/processed",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = build_dataset(save=not args.no_save)

    print("\nPer-crop yield statistics:")
    print(yield_statistics(df).to_string(index=False))
    print("\nSuitability distribution:")
    print(df[config.SUITABILITY_COLUMN].value_counts().to_string())


if __name__ == "__main__":
    main()
