# Raw data

These are the inputs, exactly as downloaded. Nothing in the code writes to this folder.

| File | Rows | What it is |
|---|---|---|
| `Crop_recommendation.csv` | 2,200 | Soil readings (N, P, K, pH) and weather (temperature, humidity, rainfall), each row tagged with a crop in the `label` column. Covers 22 crops. |
| `crop_yield.csv` | 19,689 | Indian district-year records: crop, year, season, state, area, production, annual rainfall, fertiliser, pesticide, yield. |
| `final_suitability_dataset.csv` | 2,200 | An earlier attempt that joined the two sources and added engineered columns like soil type and variety. Kept for reference. The current pipeline doesn't read it. |

## Rebuilding the merged dataset

`data/processed/final_merged_dataset.csv` is derived, so it isn't committed. Rebuild it with:

```bash
python -m crop_suitability.data
```

That produces 277,000 rows, which sounds like a lot more data than it is — the join
multiplies rows rather than adding information. The "Results" section of the main
README explains why that matters.

## Where these came from

Crop recommendation and soil data came from Kaggle and the Government of India Open
Data Portal. The suitability reference data draws on FAO soil and climate archives.
Check the licence terms on the originals before redistributing them.
