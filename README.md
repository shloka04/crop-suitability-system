# Crop Suitability & Recommendation System

Choose the wrong crop and you lose a season. Most smallholders make that decision from habit or local advice, often without a soil test, and a bad call costs a year of income rather than a bad quarter.

This project builds two models on soil and weather data that farmers can actually obtain — nitrogen, phosphorus, potassium, pH, temperature, humidity and rainfall:

- **Suitability** — name a crop and give your conditions, and it predicts whether that crop will reach a viable yield there.
- **Recommendation** — same conditions, and it ranks all five crops it knows by fit.

Both run from Python, a command line, or a Gradio web app.

Crops covered: banana, coconut, jute, maize, rice.

The headline accuracy of one of these models is misleading, and the most interesting part of this write-up is the section explaining why.

## Quick start

```bash
git clone https://github.com/shloka04/crop-suitability-system.git
cd crop-suitability-system

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Trained models ship with the repository, so predictions work immediately:

```bash
python -m crop_suitability.predict --crop rice \
    --N 90 --P 42 --K 43 --temperature 25 --humidity 80 --ph 6.5 --rainfall 210 --rank
```

```bash
python -m crop_suitability.app          # web app on http://127.0.0.1:7860
```

To rebuild everything from the raw CSVs:

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

advisor.check_suitability("rice", **conditions)   # (False, 0.651)
advisor.recommend_crop(**conditions)              # ('rice', 0.813)
advisor.rank_crops(**conditions)                  # [('rice', 0.813), ('jute', 0.154), ...]
advisor.analyse("rice", **conditions)             # both answers at once
```

| Input | Meaning | Range in the training data |
|---|---|---|
| `N` | Soil nitrogen | 0–120 mg/kg |
| `P` | Soil phosphorus | 5–95 mg/kg |
| `K` | Soil potassium | 15–55 mg/kg |
| `temperature` | Mean temperature | 18.0–29.9 °C |
| `humidity` | Relative humidity | 55.3–100 % |
| `ph` | Soil pH | 5.0–7.9 |
| `rainfall` | Rainfall | 60.7–298.6 mm |

These are the ranges the models actually saw. The Gradio sliders allow wider values — potassium goes to 250 where the data stops at 55 — so anything beyond the table is extrapolation and the confidence score should not be trusted there.

## How it works

The two source datasets disagree on how to spell crop names, so reconciling them comes first.

```
Crop_recommendation.csv    2,200 soil and weather readings, each tagged with a crop
crop_yield.csv            19,689 district-year yield records from Indian states
        |
        |  crop names lowercased, then fuzzy-matched (rapidfuzz, score >= 80)
        |  inner join on the matched name
        |  Suitability = 1 where recorded yield clears that crop's threshold
        |
        +-- suitability model:    soil/weather + one-hot crop  ->  0 or 1
        +-- recommendation model: soil/weather only            ->  crop name
```

Both are LightGBM classifiers.

The yield thresholds come from Indian government and industry publications rather than being invented:

| Crop | Threshold | Unit | Source |
|---|---|---|---|
| rice | 2.5 | t/ha | Economic Survey of India 2024–25 |
| maize | 4.0 | t/ha | Agmarknet maize crop profile |
| banana | 40.0 | t/ha | National Horticulture Board banana report |
| jute | 2.8 | t/ha | Directorate of Jute Development |
| coconut | 12,000 | nuts/ha | Coconut Development Board, 2025 |

They live in `config.py` and can be changed in one place.

## What SHAP says the models are doing

The two models behave very differently under inspection.

The **recommendation model** reads growing conditions, as intended. Rainfall and potassium dominate, then humidity, phosphorus and nitrogen.

The **suitability model** does not. Its SHAP values are led by the one-hot crop indicators by roughly two orders of magnitude:

```
crop_maize     0.6864
crop_jute      0.1920
crop_rice      0.1295
humidity       0.0062
ph             0.0054
temperature    0.0036
N              0.0035
```

It is deciding almost entirely on *which crop was selected*, barely on the conditions supplied. That is a real finding about the model, and it connects directly to the next section.

## Results, and what they actually mean

**Suitability model:** 75.6% accuracy, F1 0.69. Recall is 0.98 on the "not suitable" class and 0.17 on "suitable" — good at warning you off a bad choice, weak at confirming a good one. For a tool meant to stop farmers losing a season, erring in that direction is the less harmful failure, but it is a genuine limitation.

**Recommendation model:** 100% accuracy, macro-F1 1.00 across 5-fold cross-validation. **That number is wrong, and here is why.**

Joining the datasets on crop name alone pairs every soil reading for a crop with every yield record for that crop. 2,200 readings and 19,689 yield records expand to about 277,000 rows — but those rows contain only **500 distinct soil/weather combinations**, each repeated roughly 554 times. Split that randomly and byte-identical rows land in both train and test, so the model scores perfectly by recognising rows it has already seen. Cross-validation does not catch it either, because the duplicates cross fold boundaries.

So the honest reading is that the recommendation model separates these five crops cleanly on the 500 readings it was given, and the 100% figure says nothing about an unseen field.

Things I would fix with more time:

- The suitability label comes from the yield side of the join while the features come from the soil side. Within one crop the features repeat while the label varies, which is exactly what the SHAP values above show.
- The train/test split should be grouped by source reading, not random.
- Only 5 of the 22 crops in the recommendation dataset survive matching, because the rest have no yield records to pair with.
- Coconut is measured in nuts per hectare while everything else is tonnes, which makes the yield plots awkward on a single axis.

A better version would aggregate yields to one record per crop, state and season and join on those keys. That change is contained to `data.py` and the split helper in `train.py`.

## Engineering

```
src/crop_suitability/
  config.py       paths, features, thresholds, hyperparameters
  data.py         loading, crop-name matching, merge, labelling
  features.py     encoding, keeping training and inference in step
  train.py        trains and saves both models
  evaluate.py     metrics, confusion matrices, SHAP
  predict.py      CropAdvisor API and CLI
  app.py          Gradio interface
```

About 1,200 lines across eight modules, with a 15-test pytest suite and a GitHub Actions workflow that runs lint and tests on Python 3.10 and 3.12 on every push.

One design decision worth noting: training and inference share the same encoding path in `features.py`, so the column layout cannot silently drift between the two — a common and hard-to-find source of silent prediction bugs.

```bash
pip install -r requirements-dev.txt
pytest
```

## Data sources

Crop recommendation and soil data from Kaggle and the Government of India Open Data Portal. Suitability reference data draws on FAO soil and climate archives. Check the licence terms on the originals before redistributing.

One change from the original notebook: trailing whitespace in the `Season` column is stripped on load. Neither model uses `Season` as a feature.

**Not agronomic advice.** Treat the output as a first pass to check against a real soil test and local extension guidance.

## Licence

MIT — see [LICENSE](LICENSE).
