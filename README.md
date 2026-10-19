# Crop Suitability & Recommendation System

Pick the wrong crop for your field and you lose a whole season. Most farmers make that call from habit or local advice, often without a soil test. This project tries to help with that using data farmers can actually get hold of: soil nutrients, pH, temperature, humidity and rainfall.

There are two models behind it:

- **Suitability** — you name a crop and give your conditions, and it tells you whether that crop is likely to hit a decent yield there.
- **Recommendation** — same conditions, but it picks the crop that fits them best out of the five it knows.

Both run from Python, the command line, or a Gradio web app.

Crops covered: banana, coconut, jute, maize, rice.

## Getting started

```bash
git clone https://github.com/shloka04/crop-suitability-system.git
cd crop-suitability-system

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Trained models are already in `models/`, so you can go straight to predicting:

```bash
# Will rice do well here, and what would do better?
python -m crop_suitability.predict --crop rice \
    --N 90 --P 42 --K 43 --temperature 25 --humidity 80 --ph 6.5 --rainfall 210 --rank
```

```bash
# Web app on http://127.0.0.1:7860
python -m crop_suitability.app
```

To redo everything from the raw CSVs:

```bash
python -m crop_suitability.data           # merge the datasets and build the label
python -m crop_suitability.train --cv     # retrain both models
python -m crop_suitability.evaluate --shap   # reports and plots into reports/
```

## Using it from Python

```python
from crop_suitability.predict import CropAdvisor

advisor = CropAdvisor()

conditions = dict(N=90, P=42, K=43, temperature=25, humidity=80, ph=6.5, rainfall=210)

advisor.check_suitability("rice", **conditions)   # (True, 0.93)
advisor.recommend_crop(**conditions)              # ('rice', 0.88)
advisor.rank_crops(**conditions)                  # [('rice', 0.81), ('jute', 0.15), ...]
advisor.analyse("rice", **conditions)             # both answers at once
```

What it expects:

| Input | What it is | Usual range |
|---|---|---|
| `N` | Soil nitrogen | 0–150 mg/kg |
| `P` | Soil phosphorus | 0–150 mg/kg |
| `K` | Soil potassium | 0–250 mg/kg |
| `temperature` | Mean temperature | 10–45 °C |
| `humidity` | Relative humidity | 20–100 % |
| `ph` | Soil pH | 3.0–10.0 |
| `rainfall` | Rainfall | 20–300 mm |

## How it works

The two source datasets don't agree on how to spell crop names, so the first job is getting them to line up.

```
Crop_recommendation.csv    2,200 soil and weather readings, each tagged with a crop
crop_yield.csv            19,689 district-year yield records from Indian states
        |
        |  crop names lowercased, then fuzzy-matched (rapidfuzz, score >= 80)
        |  inner join on the matched name
        |  Suitability = 1 where the recorded yield clears that crop's threshold
        |
        +-- suitability model:    soil/weather + one-hot crop  ->  0 or 1
        +-- recommendation model: soil/weather only            ->  crop name
```

Both are LightGBM classifiers. SHAP is used to see which inputs drive a prediction — humidity, temperature, nitrogen and potassium turn out to matter most for suitability.

The yield thresholds come from Indian government and industry crop publications rather than being made up:

| Crop | Threshold | Unit | Source |
|---|---|---|---|
| rice | 2.5 | t/ha | Economic Survey of India 2024–25 |
| maize | 4.0 | t/ha | Agmarknet maize crop profile |
| banana | 40.0 | t/ha | National Horticulture Board banana report |
| jute | 2.8 | t/ha | Directorate of Jute Development |
| coconut | 12,000 | nuts/ha | Coconut Development Board, 2025 |

They live in `src/crop_suitability/config.py` if you want to change them.

## What's in here

```
data/
  raw/                  the source CSVs
  processed/            merged dataset, not committed (rebuild it with the data step)
models/                 trained .pkl files and the feature layout they expect
notebooks/
  01_exploration.ipynb  the exploratory work, tidied up
reports/figures/        confusion matrices, SHAP and yield plots (not committed)
src/crop_suitability/
  config.py             paths, features, thresholds, hyperparameters
  data.py               loading, crop-name matching, merge, labelling
  features.py           encoding, and keeping training and inference in step
  train.py              trains and saves both models
  evaluate.py           metrics, confusion matrices, SHAP
  predict.py            the CropAdvisor class and the CLI
  app.py                Gradio interface
tests/                  pytest suite
```

Tests:

```bash
pip install -r requirements-dev.txt
pytest
```

## Results, and what they actually mean

The suitability model gets about 76% accuracy with an F1 of 0.69. Recall is 0.98 on the "not suitable" class but only around 0.17 on "suitable", so it's good at warning you off a bad choice and much weaker at confirming a good one. For a tool meant to help farmers avoid losing a season, being cautious in that direction is the less harmful failure, but it's still a real limitation.

The recommendation model reports 100% accuracy and a macro-F1 of 1.00 across 5-fold cross-validation. **That number is too good, and it's worth being upfront about why.**

Joining the two datasets on crop name alone pairs every soil reading for a crop with every yield record for that crop. 2,200 readings and 19,689 yield records turn into about 277,000 rows, but there are only **500 distinct soil/weather combinations** in there, each repeated roughly 554 times. Split that randomly into train and test and identical rows land on both sides, so the model can score perfectly by memorising rows it has already seen. Cross-validation doesn't catch it, because the duplicates cross the fold boundaries too.

So the honest reading is: the recommendation model separates these five crops cleanly on the 500 readings it was given, and the 100% figure says nothing about how it would do on a field it hasn't seen.

Other things I'd fix given more time:

- The suitability label comes from the yield side of the join while the features come from the soil side. Within one crop the features repeat while the label moves around, so the model is partly learning each crop's yield distribution rather than a real link between conditions and outcome.
- The split should be grouped by source reading (or by state and year), not random.
- Only 5 of the 22 crops in the recommendation dataset survive matching, because the rest have no yield records to pair with.
- Coconut is measured in nuts per hectare while everything else is tonnes per hectare, which makes the yield plots awkward to read on one axis.

A better version would aggregate yields to one record per crop, state and season and join on those keys instead. That change would be contained to `data.py` and the split helper in `train.py`.

**This isn't agronomic advice.** Treat it as a first pass to sanity-check against a real soil test and local extension guidance.

## Data sources

- Crop recommendation and soil data: Kaggle, and the Government of India Open Data Portal.
- Crop yield records: district-level Indian state yield data.
- Suitability reference data: FAO soil and climate archives.

Check the licence on the originals before reusing or redistributing them.

One small change from the original notebook: trailing spaces in the `Season` column are stripped on load. Nothing else about the merge differs, and neither model uses `Season` as a feature.

## Licence

MIT, see [LICENSE](LICENSE).
