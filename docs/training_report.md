# Neural Network Training Report

Generated: 2026-08-27

## 1. Data

- 83 countries (2010 population > 10M, UN WPP 2024 basis) x 1950-2024 = **6,225 rows**
- **13 input features** (coverage >= 30%): urban, gdppc2000 (GDP per capita, 2000-equivalent USD), manuf_pc2000 (manufacturing value added per capita), elec_access, mobile, broadband, literacy, physicians (per 1000), elec_gen_pc (electricity per capita), edu_f (female mean years of schooling), mat_leave_days, pat_leave_days, nonagri_empl
- Missing handling: z-score mean fill + missing-mask columns (26-dim input), no fabricated data; cleaning rules in [data_sources_and_cleaning.md](data_sources_and_cleaning.md)
- Splits: random 70/20/10 (4357/1245/623) + time extrapolation (train <= 2010, test 2011-2024)

## 2. Model

- Pure-numpy MLP (local i5-2500 CPU, no AVX2, no torch): 26 -> 2048 -> 2048 -> 2048 -> 1024 -> 512 -> 1
- **11,071,489 parameters (11.1M)**
- Adam + weight decay 1e-4 + dropout 0.10 + step lr (1e-3 -> 3e-5) + early stopping (patience 120)
- Training: early stop at epoch 143, ~57 minutes
- Hyperparameter experiments: dropout 0.30 collapsed/divered in deep nets (best R2 0.02); 0.10 optimal (R2 0.813 @ 60 epochs); log target worse than raw; adding a year feature hurts; linear baseline R2 0.738-0.755

## 3. Results

| Set | MAE | R2 | \|err\|<=0.25 | \|err\|<=0.30 |
|---|---|---|---|---|
| Train | 0.6853 | 0.8227 | 26.9% | 31.9% |
| Valid | 0.6806 | 0.8322 | 27.6% | 32.9% |
| Test (random) | 0.7266 | **0.8089** | 22.0% | 28.4% |
| Test (2011-2024 time extrapolation) | 0.4173 | **0.8652** | 44.4% | 52.5% |

Linear baseline: train R2 0.743 / valid R2 0.755 / test R2 0.738 (MAE 0.89) - the NN clearly outperforms it.

## 4. 98% Target Status (honest assessment)

- User target: test "success rate" >= 98% (|predicted - actual| <= 0.25)
- Measured: test R2 0.809, |err|<=0.25 share 22%, **98% not reached**
- **Structural ceiling**: 240- and 400-epoch experiments consistently converge to valid R2 ~ 0.83; adding a year feature does not help - this is the explainable-variance ceiling of the 13 structural determinants under a random split
- Residual error sources: country-specific cultural differences and policy shocks not covered by the features (one-child policy, Iran's 2013 collapse, COVID years, etc.); these are not predictable from "determinants"
- Comparison to the V5 paper: MAE 0.05 there relies on **lagged TFR (an autoregressive term)**; this task's input is restricted to structural determinants (the user's 25 listed items) without lagged TFR. Adding lagged TFR or country fixed effects could approach R2 0.98, but that conflicts with the "determinant formula" research goal
- Time extrapolation (2011-2024) is strong: MAE 0.42 / R2 0.865 - good accuracy in the recent low-fertility range

## 5. Feature Selection (minimal set, >= 99% of full-model effect)

Permutation importance (valid R2 drop):

| Feature | dR2 | | Feature | dR2 |
|---|---|---|---|
| edu_f (female education) | **+0.422** | nonagri_empl | +0.021 |
| physicians (doctor density) | +0.250 | mat_leave_days | +0.020 |
| broadband | +0.106 | pat_leave_days | +0.019 |
| mobile | +0.105 | gdppc2000 | +0.016 |
| urban | +0.051 | manuf_pc2000 | +0.012 |
| elec_access | +0.043 | literacy | +0.011 |
| elec_gen_pc | +0.033 | | |

Greedy forward selection: **10 features reach 99.1% of full-model valid R2** (0.8244/0.8322), 98.8% on test (0.7994/0.8089)
**Minimal set**: urban, elec_access, mobile, broadband, physicians, elec_gen_pc, edu_f, pat_leave_days, gdppc2000, manuf_pc2000
Dropped: literacy, nonagri_empl, mat_leave_days

**Core finding**: female education is the overwhelmingly dominant determinant (dR2 = 0.42), followed by the health system (physicians dR2 = 0.25) and telecom infrastructure (broadband/mobile ~ 0.11); GDP per capita has the smallest marginal contribution (dR2 = 0.016) because its development effect is absorbed by education/health/telecom.

## 6. Mathematical Expression (generalized logistic, complete-case fit)

```
TFR = 1.17 + (7.40 - 1.17) / (1 + exp(-z))
z = -0.258 + 0.0095*urban - 0.020*elec_access + 0.012*mobile - 0.168*broadband
    - 0.332*physicians + 0.016*elec_gen_pc - 0.210*edu_f
    + 0.010*pat_leave_days + 0.626*ln(1+gdppc2000) - 0.487*ln(1+manuf_pc2000)
```

Fit on 1,184 complete cases (<= 2010), evaluated on 638 (> 2010): **R2 = 0.801, MAE = 0.449** (comparable to the 11M-param NN's time extrapolation of 0.417)
Note: multicollinearity makes some coefficient signs non-structural (e.g. gdppc2000 positive); structural conclusions should rely on the Section 5 importance ranking.

## 7. Reproduction

```bash
python src/01_fetch_sources.py   # download all sources -> data/raw/ (status: raw/source_log.json)
python src/02_build_panel.py     # clean/merge -> data/panel.csv + docs/data_sources_and_cleaning.md
python src/03_prepare_dataset.py # feature engineering/splits -> data/dataset.xlsx + data/splits.npz
python src/04_train_nn.py --epochs 400   # train -> models/ (--resume supported)
python src/05_feature_select.py  # feature selection -> data/feature_selection.json
python src/06_extract_formula.py # formula -> data/formula.json
```

Artifacts: data/panel.csv (6225x22) - data/dataset.xlsx - data/splits.npz - models/final_weights.npz (11.1M params) - models/config.json - models/train_history.csv - models/checkpoints/latest.npz - data/coverage.json - data/outliers.json - data/feature_selection.json - data/formula.json
