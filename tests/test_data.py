"""Tests for crop-name reconciliation and merging."""

from __future__ import annotations

import pandas as pd

from crop_suitability import config
from crop_suitability.data import align_crop_names, build_crop_name_mapping, merge_datasets


def test_fuzzy_mapping_matches_close_names():
    mapping = build_crop_name_mapping(["rice", "maize"], ["rice", "maize", "arecanut"])
    assert mapping == {"rice": "rice", "maize": "maize"}


def test_fuzzy_mapping_rejects_unrelated_names():
    mapping = build_crop_name_mapping(["dragonfruit"], ["rice", "maize"])
    assert mapping == {}


def test_align_crop_names_drops_unmatched_rows():
    df_rec = pd.DataFrame({"crop": ["rice", "dragonfruit"], "N": [90, 10]})
    df_yield = pd.DataFrame({"crop": ["rice"], "Yield": [3.0]})
    aligned = align_crop_names(df_rec, df_yield)
    assert aligned["crop"].tolist() == ["rice"]


def test_merge_joins_on_crop():
    df_rec = pd.DataFrame({"crop": ["rice"], "N": [90]})
    df_yield = pd.DataFrame({"crop": ["rice", "rice"], "Yield": [3.0, 1.0]})
    merged = merge_datasets(df_rec, df_yield)
    # One reading joined to two yield records: the documented cross product.
    assert len(merged) == 2
    assert config.CROP_COLUMN in merged.columns
