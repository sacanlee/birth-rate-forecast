# Fertility Determinants Research - Progress Log

Working directory: `C:\Users\Administrator\Desktop\zhihu\task\birth-rate-forecast\`
Start date: 2026-08-26

## Phases and Status (follow-up requirement iterations, 2026-08-27)

- [x] Supplementary data: worldsteel annual reports = subscription (sdv-downloads login page), Wayback/Wikipedia/mirrors all unreachable -> recorded honestly + drop-in interface left
- [x] Supplementary data: rail length IS.RRS.TOTL.KM (WDI/UIC, 1995+, 2,139 rows) OK; road has no global source; UIC Railisa = paid subscription
- [x] dTFR dataset restructure (07_build_delta_dataset.py): k=5 (5,810 windows) / k=10 (5,395 windows); input = [TFR(t), X levels 14, dX, 29 masks] = 58 dims
- [x] dTFR training k=5 (08_train_delta.py, 11M params) OK, early stop at epoch 290 (2026-08-27):
  - Time extrapolation (>=2011) 747 windows: dTFR MAE 0.0966 / R2 0.708 / |err|<=0.25 = 93.2%
  - **TFR level: MAE 0.097 / R2 0.9925 / |err|<=0.25 = 93.2%** (4x the accuracy of the round-1 absolute model's 0.42!)
  - Random split: MAE 0.139 / R2 0.622 / |err|<=0.25 = 85.9%
  - Baselines: persistence MAE ~ 0.35; linear (mean-filled) MAE 0.265 - the NN wins on all metrics
  - Chained (TFR 5 years ahead): most countries error < 0.1 (JPN/USA/BRA/MEX); China 2018-2023 overestimated (policy shock)
- [x] Feature selection k=5 (09): the dTFR model needs all 14 features (mobile completes 90.8% -> 100%); importance: gdppc 0.267 > physicians 0.218 > manuf 0.192 > mobile 0.140 > mat_leave 0.126
- [x] Formula k=5 (10): linear dTFR equation test MAE 0.265 (weaker than the NN's 0.097) -> NN is the primary model, the formula is an auxiliary expression
- [x] dTFR training k=10 (08, 11M) OK, early stop at epoch 352: time extrapolation dTFR MAE 0.144 / R2 0.797 / |err|<=0.25 = 84.6%; TFR level R2 0.982
- [x] Feature selection k=10 (09) OK: also needs all 14 features (mobile completes 95.8% -> 100%)
- [x] Report: docs/training_report_delta.md OK
- [x] Documentation wrap-up and context archiving OK (auto-memory updated)
- [x] Shared-module refactor + scenario-forecast scripts OK (2026-08-27): src/nn_common.py (shared by 11/12: transforms/loading/forward/panel) + src/12_predict_scenario.py (user sets several parameters at a future year -> predicts TFR + reduction + factor analysis; unset parameters keep latest-year values; 5-year chaining with linear scaling of the final partial window; zero-change baseline decomposition: structural baseline + per-parameter marginal contributions + interaction residual; fixed the 11 chained-index bug and the double-log bug in 11/12)
  - Verified: CHN 2024->2035 edu_f missing->13 + maternity 158->180: TFR 1.013 -> 0.543 (down 46.4%; zero-change baseline 29.2% + edu_f 70.7% + maternity 2.6%); KOR 2050 GDP +8%: TFR rises 0.108 instead (high-development reversal effect)
  - Note: some countries have missing parameters in the latest year (e.g. CHN edu_f/manuf, KOR literacy/edu_f); 'keeping' = the model treats them as unknown (mask)
  - Country resolution (nn_common.resolve_country): ISO3/canonical name/common aliases/unique substring, case-insensitive ("viet nam"/"vietnam" -> VNM, usa/uk/korea, etc.); substring matches attach a hint, multi-country matches raise an error; --list-countries lists all 83; both 11 and 12 use it

## Phases and Status

- [x] 0. Environment: pip install requests/openpyxl (note: this machine's pip needs `--proxy ""` to bypass the invalid system proxy 127.0.0.1:7890); directory structure created
- [x] 1. Data acquisition (01_fetch_sources.py) -> data/raw/ OK 2026-08-26
- [x] 2. Panel build (02_build_panel.py) -> data/panel.csv + docs/data_sources_and_cleaning.md OK 2026-08-26
- [x] 3. Dataset preparation (03_prepare_dataset.py) -> data/dataset.xlsx + data/splits.npz OK 2026-08-26
  - 13 features + 13 missing masks = 26 inputs; 6,225 rows; random split 4357/1245/623; time split 5063/1162
  - Missing handling: z-score mean fill + mask columns (no fabricated data)
- [x] 4. Neural network training (04_train_nn.py, ~10M params, numpy) -> models/ OK 2026-08-27
  - 11.07M params, early stop at epoch 143 (57 min); test R2 0.809 / MAE 0.727; time extrapolation R2 0.865 / MAE 0.417
  - Hyperparameter experiments: dropout 0.30 collapsed/divered / 0.10 optimal; log target and year feature both worse; linear baseline R2 0.74
  - Structural ceiling: valid R2 ~ 0.83 for 13 variables under a random split; the 98% success-rate target is structurally unreachable (documented honestly in the report)
- [x] 5. Feature selection (05_feature_select.py) OK: minimal set of 10 features = 99.1% of full-model valid R2; edu_f dR2 0.42 first
- [x] 6. Mathematical expression (06_extract_formula.py) OK: TFR = 1.17 + 6.23/(1+exp(-z)); time extrapolation R2 0.801 / MAE 0.449
- [x] 7. Documentation wrap-up OK: docs/training_report.md written; auto-memory updated (2026-08-26), final results pending

## Key Decisions

- Parameter count: reduced from 50M-500M to ~10M per user request (less overfitting)
- Sparse variables (18-25, etc.): recorded honestly with coverage flagged, no forced extrapolation; the NN only uses columns with coverage >= 30%
- Training engine: pure numpy (local i5-2500 has no AVX2; torch is risky)
- 2000-equivalent USD: WDI uses 2015 constant USD, converted via the US GDP deflator, ratio defl2000/defl2015 = 0.7473
- Pre-introduction years of mobile/broadband set to 0 (fact, not interpolation); maternity/paternity leave units = days (WBL basis, e.g. China 2019: 128/14 days)
- The large WPP file is rate-limited by population.un.org to ~13KB/s; downloaded in chunks with curl resume
- Network: urllib/curl need --noproxy / ProxyHandler({}) to bypass the invalid system proxy 127.0.0.1:7890

## Actual Results

- Data: 83 countries (2010 population > 10M) x 1950-2024 = 6,225 rows x 22 columns
- 15 variables successfully obtained (pop/tfr/urban/gdppc/manuf_pc/elec_access/mobile/broadband/agri_empl/literacy/physicians/elec_gen_pc/edu_f/mat_leave_days/pat_leave_days)
- Unavailable: steel_pc (OWID slug obsolete / UNdata 500), refining (EI 403), nurses (GHO domain DNS unreachable), road/rail (OECD/IRF/UIC paywall/registration), house-price-to-income/rent-to-income/childcare/benefits/tuition (no global historical panel)
- 11 outlier corrections (data/outliers.json); coverage: tfr 1.00, urban 0.92, gdppc 0.84, edu_f 0.83, physicians 0.86, elec_access 0.48, literacy 0.43, mat/pat_leave 0.69
