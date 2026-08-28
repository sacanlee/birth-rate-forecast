# Fertility Rate Forecasting

Machine-learning models that predict the **total fertility rate (TFR)** of 83 countries (2010 population > 10M) from structural development determinants, and forecast **future fertility decline** under user-defined scenarios.

The primary model predicts the **5-year TFR change** (ΔTFR) with a pure-numpy MLP (~11M parameters, CPU-only) trained on a 1950-2024 country-year panel — no PyTorch/TensorFlow required. On time extrapolation (2011-2024) it achieves **ΔTFR MAE 0.097, TFR-level R² 0.9925**, well above persistence (MAE ≈ 0.35) and linear (MAE 0.265) baselines.

## Key results

- **ΔTFR model (k=5, time test ≥2011)**: ΔTFR MAE 0.097 / R² 0.708 / |err|≤0.25 = 93.2%; TFR level MAE 0.097 / R² 0.9925
- **Top determinants of fertility change**: GDP per capita (ΔR² 0.267), physician density (0.218), manufacturing (0.192), mobile penetration (0.140), maternity leave (0.126)
- **Level model (k=1, 2011-2024 extrapolation)**: MAE 0.417 / R² 0.865; female education is the dominant determinant (ΔR² 0.42)
- Policy/event shocks (e.g. China's two-child policy rebound) are not predictable from structural features — this is an information floor, not a model defect

## Directory structure

```
birth-rate-forecast/
├── src/                      # All scripts (run from anywhere, paths are repo-relative)
│   ├── 01_fetch_sources.py   # Download all raw data sources -> data/raw/
│   ├── 02_build_panel.py     # Clean/merge/interpolate -> data/panel.csv
│   ├── 03_prepare_dataset.py # Features + splits (level model) -> data/splits.npz
│   ├── 04_train_nn.py        # Train level MLP (26 -> ... -> 1, ~11M params)
│   ├── 05_feature_select.py  # Permutation importance + minimal set (level model)
│   ├── 06_extract_formula.py # Generalized-logistic formula on the minimal set
│   ├── 07_build_delta_dataset.py  # ΔTFR windows (k=5 default) -> data/splits_delta_k5.npz
│   ├── 08_train_delta.py     # Train ΔTFR MLP (58 -> ... -> 1)
│   ├── 09_feature_select_delta.py # Feature selection for the ΔTFR model
│   ├── 10_formula_delta.py   # Linear ΔTFR increment equation (auxiliary expression)
│   ├── 11_predict_country.py # Predict one country's k-year decline + --eval metrics
│   ├── 12_predict_scenario.py# Scenario forecast: set future feature values, get TFR(T)
│   └── nn_common.py          # Shared transforms/model loading/country resolution
├── data/                     # Generated artifacts (gitignored: raw/, *.csv, *.xlsx, splits)
│   └── raw/                  # Downloaded sources (see docs/data_sources_and_cleaning.md)
├── models/                   # Trained weights + checkpoints (gitignored)
├── docs/                     # Reports: data cleaning, training reports, progress log
├── scraper_tmp/              # Local scraping artifacts (gitignored, never published)
├── requirements.txt
├── LICENSE                   # MIT
└── README.md
```

Large generated files (`data/raw/`, `data/*.csv`, `data/*.xlsx`, `data/splits*.npz`, `models/`) are gitignored — run the pipeline to regenerate them.

## Installation

```bash
git clone <repo-url>
cd birth-rate-forecast
pip install -r requirements.txt
```

Requires Python ≥ 3.8 and `pandas >= 2.2`. No GPU needed; training runs on CPU (the ~11M-param ΔTFR model takes ~2.5 h on a modest CPU).

## Usage

Run scripts from anywhere; all paths are resolved relative to the repo root:

### 1. Reproduce the full pipeline

```bash
python src/01_fetch_sources.py          # download sources (see docs for unreachable ones)
python src/02_build_panel.py            # -> data/panel.csv + coverage.json
python src/03_prepare_dataset.py        # level-model features/splits
python src/04_train_nn.py --epochs 400  # train level MLP (--resume to continue)
python src/05_feature_select.py         # importance + minimal feature set
python src/06_extract_formula.py        # compact logistic formula
python src/07_build_delta_dataset.py --k 5   # ΔTFR windows
python src/08_train_delta.py --k 5 --epochs 400   # train ΔTFR model (--big: ~30M params)
python src/09_feature_select_delta.py --k 5
python src/10_formula_delta.py --k 5
```

Each step is idempotent where possible (01 skips existing files; 04/08 support `--resume`).

### 2. Verify a known window (2019 -> 2024, China)

```bash
python src/11_predict_country.py --iso3 CHN --year 2019
```

### 3. Forward prediction with trend extrapolation

```bash
python src/11_predict_country.py --iso3 CHN --year 2024
# TFR 2024 -> predicted TFR 2029 (ΔX extrapolated by the recent 5-year trend)
```

### 4. Scenario forecasting

Set one or several features at a target year (original units); unset features keep their latest-year values. The model chains 5-year ΔTFR predictions forward and decomposes the reduction into per-parameter marginal contributions:

```bash
python src/12_predict_scenario.py --iso3 CHN --year 2035 --set '{"edu_f": 13.0, "mat_leave_days": 180.0}'
python src/12_predict_scenario.py --iso3 KOR --year 2050 --set '{"gdppc2000": 30000.0}'
python src/12_predict_scenario.py --list-countries   # all 83 countries
```

Country input is case-insensitive and accepts ISO3 codes, canonical names, common aliases and unique substrings (`--iso3 "viet nam"` == `--iso3 VNM` == `--iso3 vietnam`).

### 5. Re-run all performance metrics

```bash
python src/11_predict_country.py --eval
```

## Data

| Source | Variables |
|---|---|
| UN WPP 2024 (Medium) | TFR, population (83 countries, 1950-2024) |
| World Bank WDI | GDP per capita, manufacturing, electricity access, mobile, broadband, agriculture employment, literacy, physicians, urbanization, rail length |
| Our World in Data | electricity generation per capita |
| Barro-Lee v3 | female mean years of schooling (15+) |
| World Bank WBL | paid maternity/paternity leave (days) |

Cleaning: 10M-population country filter; per-capita conversions; 2000-equivalent USD conversion (×0.7473 via US GDP deflator); within-range interpolation; pre-introduction zeros for mobile/broadband; MAD-robust outlier correction (11 corrections). Unavailable variables (steel, refining, health workforce, road/rail panels) are documented, not imputed. Full details: [docs/data_sources_and_cleaning.md](docs/data_sources_and_cleaning.md).

## Models

- **Level model** (04): MLP `[26 -> 2048 -> 2048 -> 2048 -> 1024 -> 512 -> 1]`, 11,071,489 params. Predicts TFR(t) from 13 features + 13 missing masks. Adam, dropout 0.10, step LR, early stopping.
- **ΔTFR model** (08, primary): same architecture on 58 inputs `[TFR(t), 14 levels, 14 changes, 29 masks]`, predicting ΔTFR(t→t+5). A `--big` variant (≈28M params) is available.
- **Formula** (06): generalized logistic `TFR = Tmin + (Tmax−Tmin)/(1+exp(−z))` on the 10-feature minimal set (test R² 0.801) — an interpretable approximation of the level model.
- **Formula** (10): linear ΔTFR increment equation — auxiliary expression, weaker than the NN (MAE 0.265 vs 0.097).

Missing values are never fabricated: they are mean-filled at the z-score stage **and flagged with a mask column**, so the network learns "this value is unknown" rather than trusting the fill.

Reports: [docs/training_report.md](docs/training_report.md) (level model), [docs/training_report_delta.md](docs/training_report_delta.md) (ΔTFR model), [docs/PROGRESS.md](docs/PROGRESS.md) (development log).

## License

MIT — see [LICENSE](LICENSE).
