# Calibrated 2045 Fertility Forecasts: Nigeria, Niger and DR Congo

This report replaces all earlier scenario rounds. It answers one question: **which
reasonable 2045 development configurations bring the model's predicted total
fertility rate (TFR) to ~2.5 for Nigeria, ~3.5 for DR Congo and ~3.2 for Niger** —
and what those forecasts look like in detail.

Method: a grid/random search (`scraper_tmp/calib_search.py`, gitignored) over
plausible 2045 target values derived from real country benchmarks (India 2010/2024,
ILO and mid-income standards), evaluated with the repository's chained ΔTFR neural
network (`src/12_predict_scenario.py`, model `delta_k5`). The search evaluator
reproduces the official CLI to the third decimal (verified on known scenarios), and
every final configuration below was re-run through the official CLI.

Date: 2026-09-05. All inputs are taken from `data/panel.csv` (83-country, 1950-2024
panel); the trained model is `models/final_weights_delta_k5.npz`.

## 1. Final parameter configurations (reached linearly by 2045)

| Variable (units) | Nigeria (2024 → 2045) | Niger (2024 → 2045) | DR Congo (2024 → 2045) |
|---|---|---|---|
| Adult literacy (%) | 70.4 → **80** | 35.6 → **68.3** | 68.5 → **80** |
| Female schooling, `edu_f` (years) | missing → **6.1** | n/a (unset) | missing → **7.0** |
| GDP p.c., `gdppc2000` (2000-equiv. USD) | 1,737 → **2,000** | 440 → **1,450** | 414 → **1,450** |
| Manuf. value added p.c., `manuf_pc2000` (2000-equiv. USD) | 138 → **200** | n/a | 62 → **141** |
| Urbanization (%) | 63.0 → **80** | n/a | n/a |
| Non-agricultural employment (%) | n/a | 26.5 → **48.9** | 41.1 → **60** |
| Electricity generation p.c. (kWh) | n/a | 35 → **753** | n/a |
| Mobile subscriptions (per 100) | n/a | n/a | 58.5 → **95** |
| Physicians (per 1,000) | n/a | n/a | 0.21 → **0.60** |
| Paid maternity leave (days) | n/a | n/a | missing → **98** |

All other model features are held at their 2024 values (ceteris paribus).

### 1.1 Why these values are "reasonable" (benchmarks)

- **Nigeria** — literacy 80 ≈ India-2024 level (78.2); female schooling 6.1 years =
  India-2024 (Barro-Lee 2020: 6.12); GDP p.c. 2,000 (2000-equivalent USD) ≈ 3,450
  USD in 2024 prices, i.e. ~2.5% p.a. real growth from 2024; manufacturing p.c. 200 =
  India mid-2010s (India 141 in 2010, 275 in 2024); urbanization 80 ≈ Brazil/Malaysia
  today, from 63 today under continued fast urbanization.
- **Niger** — literacy, non-agricultural employment and electricity generation are
  exactly India's 2010 levels (68.3%, 48.9%, 753 kWh); GDP p.c. 1,450
  (2000-equivalent USD) = **2,500 USD in 2024 prices** (deflated by the US GDP
  deflator: ×69.011/119.027 = ×0.5798), ≈5.8% p.a. real growth.
- **DR Congo** — literacy 80 ≈ India-2024; female schooling 7.0 years (DRC's observed
  level was already 5.5 years in 2020, so +1.5 years in 25 years is moderate);
  non-agricultural employment 60 ≈ India-2024 (57.6); manufacturing p.c. 141 = India
  2010; physicians 0.6/1,000 (from 0.21; India-2024 = 0.72) reflects sustained health
  investment; 98 days of paid maternity leave ≈ the ILO 14-week standard; GDP p.c.
  1,450 = **2,500 USD in 2024 prices** (≈6.1% p.a. real growth, in line with DRC's
  recent mining-driven trajectory).

Deliberately *not* included (and why): in this model, single-dimension infrastructure
leaps applied to an already-declining country — large electricity-generation/access or
urbanization jumps — carry *positive* marginal offsets (see §4), so hitting the
targets cleanly favours broad but moderate packages over extreme one-dimensional
ones. This is a statistical property of the trained network (see §5).

## 2. Method and data processing

**Model.** The repository's primary model predicts 5-year ΔTFR with a pure-numpy MLP
(~11M parameters) trained on the 1950-2024 panel of 83 countries (58 inputs: TFR
level, 14 feature levels, 14 five-year changes, 29 missingness masks). Time
extrapolation (2011-2024): ΔTFR MAE 0.097, TFR-level R² 0.9925. Forecasts chain
5-year steps from 2024 to 2045 (final 1-year partial step scaled linearly). Set
features ramp linearly from 2024 to their 2045 targets in transform space
(log-space for log features); unset features stay at 2024 values; missing features
are flagged by masks exactly as in training.

**Calibration search.** For each country, 6,000 random configurations were drawn from
per-feature candidate lists (each candidate chosen from the benchmark-derived ranges
in §1.1), evaluated at TFR(2045), and ranked by distance to the target (Nigeria 2.5,
Niger 3.2, DR Congo 3.5). Final configurations were then selected from the best
results for interpretability (few features, round benchmark values) and verified with
the official CLI.

**Data-handling notes.**
- GDP targets in 2024 USD are deflated to the model's constant-2000-equivalent USD
  with the project's deflator chain: 2,500 × 0.5798 = 1,449.5; (Nigeria's 2,000
  model units correspond to ≈3,450 in 2024 dollars).
- `edu_f` (Barro-Lee) and, for DRC, `mat_leave_days` are missing in the 2024 baseline
  row (series end earlier; Nigeria's schooling series is entirely absent from the
  panel). By the scenario tool's convention, set features missing in the latest year
  are applied **from 2024 onward** — the model "knows" the value throughout the
  forecast. For DRC this is mild (observed schooling 5.5 years in 2020 vs. 7.0 set);
  treat the timing of these two conditions as optimistic.
- India 2010 benchmark values used: literacy 68.33, non-agricultural employment 48.94,
  electricity generation 753.0 kWh p.c., manufacturing p.c. 141.5, female schooling
  5.26 years. India 2024 (or last observed): literacy 78.2, female schooling 6.12,
  physicians 0.72/1,000, non-agricultural employment 57.6.

## 3. Results

| Country | TFR 2024 | **TFR 2045** | Reduction | Zero-change baseline (2045) | Target |
|---|---|---|---|---|---|
| Nigeria | 4.382 | **2.48** | −1.91 (−43.5%) | 2.67 | ~2.5 |
| Niger | 5.935 | **3.27** | −2.67 (−45.0%) | 4.15 | ~3.2 |
| DR Congo | 5.981 | **3.50** | −2.48 (−41.4%) | 5.17 | ~3.5 |

Chained TFR paths (2024 → 2029 → 2034 → 2039 → 2044 → 2045):

- **Nigeria:** 4.382 → 3.916 → 3.458 → 3.009 → 2.561 → **2.475**
- **Niger:** 5.935 → 5.254 → 4.589 → 3.949 → 3.337 → **3.265**
- **DR Congo:** 5.981 → 5.539 → 4.997 → 4.357 → 3.636 → **3.504**

### 3.1 Decomposition of the decline (ΔTFR 2024-2045, marginal contributions)

| Component | Nigeria | Niger | DR Congo |
|---|---|---|---|
| Zero-change baseline (endogenous path) | −1.711 | −1.785 | −0.810 |
| `literacy` | −0.088 | −0.487 | −0.051 |
| `edu_f` (female schooling) | −0.025 | — | −0.741 |
| `gdppc2000` | −0.019 | −0.080 | −0.484 |
| `manuf_pc2000` | +0.002 | — | −0.081 |
| `urban` | +0.178 | — | — |
| `nonagri_empl` | — | −0.331 | −0.796 |
| `elec_gen_pc` | — | −0.413 | — |
| `mobile` | — | — | +0.029 |
| `physicians` | — | — | +0.076 |
| `mat_leave_days` | — | — | −0.510 |
| Interaction residual | −0.244 | +0.426 | +0.891 |
| **Total ΔTFR** | **−1.906** | **−2.670** | **−2.477** |

Positive entries offset part of the decline (the network's marginal responses are
state-dependent and overlap, so per-feature numbers are indicative, not causal).

## 4. What delivers the decline

- **Nigeria (2.48).** The decline is dominated by the model's endogenous convergence
  path (−1.71 of −1.91; baseline alone reaches 2.67). The additional conditions close
  the last ~0.2: literacy to 80 (−0.09) is the largest explicit lever, interacting
  with the schooling/GDP/urbanization package (net interaction −0.24). Results hover
  near 2.5 for a wide range of stronger packages (2.4-2.6), i.e. the model sees
  Nigeria converging to a floor of roughly 2.4-2.6 by 2045 once literacy is high —
  further single-dimension jumps (urbanization to 80 already offsets +0.18 in
  isolation) add little beyond this plateau.
- **Niger (3.27).** All four conditions contribute directly: literacy to India-2010
  (−0.49), electricity generation to India-2010 (−0.41), employment structure
  (−0.33) and GDP to 2,500 USD-2024 (−0.08), on top of the baseline (−1.79). The
  structural package roughly doubles the decline the endogenous path would produce
  alone (4.15 → 3.27). Niger still finishes above replacement because its other
  features (female schooling, physicians, urbanization) are held at today's very low
  levels.
- **DR Congo (3.50).** Unlike Nigeria, the conditions — not the endogenous path — do
  most of the work (baseline covers only −0.81 of −2.48). The biggest levers are
  non-agricultural employment to India-2024 (60%, −0.80), female schooling (−0.74),
  paid maternity leave of 14 weeks (−0.51) and GDP (−0.48); literacy, manufacturing
  and mobile add the rest. A large positive interaction (+0.89) means the gains
  overlap and are partly redundant — which is why the search rejected even richer
  packages (e.g. adding electricity generation or urbanization targets *raises* the
  predicted TFR in this model; §5).

## 5. Caveats and interpretation

- **Model behaviour on infrastructure leaps.** The network was trained on historical
  5-year windows in which rapid electricity/urbanization expansion at still-high TFR
  (energy booms, sub-Saharan Africa 2000s-2020s) coincided with stalled fertility.
  Consequently, marginal effects of such leaps are small or positive when applied on
  top of a broad package, and the model's forecasts are insensitive (or perversely
  responsive) to them. The calibrated configurations therefore emphasize education,
  employment structure, income and health/family-policy features — the channels the
  model associates with real fertility decline.
- **Uncertainty.** Per-step ΔTFR MAE ≈ 0.1; chaining four 5-year windows compounds to
  an informal band of roughly ±0.2-0.3 births around each headline (before
  structural assumptions). Read Nigeria 2.5, Niger 3.3 and DRC 3.5 as bands,
  e.g. Nigeria 2.3-2.7, Niger 3.0-3.5, DRC 3.3-3.8.
- **Ceteris paribus.** Features not in a country's configuration stay at 2024 values
  for 21 years (e.g. physicians 0.04 for Niger, electricity access 21-62%, female
  schooling for Niger treated as unknown because unset).
- **Timing conventions.** Set features ramp linearly 2024→2045; missing-in-2024
  features (`edu_f`, DRC `mat_leave_days`) are applied from 2024 (optimistic); the
  final 1-year step is linearly scaled.
- **Structural model limits.** Policy shocks, conflict and discontinuities are not
  predictable from structural features; Niger and DRC carry conflict/migration
  histories the model cannot encode.

## 6. Reproduction

```bash
# Nigeria -> TFR(2045) = 2.475
python src/12_predict_scenario.py --iso3 NGA --year 2045 \
  --set '{"literacy": 80.0, "edu_f": 6.1, "gdppc2000": 2000.0, \
          "manuf_pc2000": 200.0, "urban": 80.0}'

# Niger -> TFR(2045) = 3.265
python src/12_predict_scenario.py --iso3 NER --year 2045 \
  --set '{"literacy": 68.326, "nonagri_empl": 48.944, "elec_gen_pc": 753.04, \
          "gdppc2000": 1449.5}'

# DR Congo -> TFR(2045) = 3.504
python src/12_predict_scenario.py --iso3 COD --year 2045 \
  --set '{"literacy": 80.0, "edu_f": 7.0, "manuf_pc2000": 141.0, \
          "nonagri_empl": 60.0, "mobile": 95.0, "physicians": 0.6, \
          "mat_leave_days": 98.0, "gdppc2000": 1450.0}'
```

The calibration search itself lives in `scraper_tmp/calib_search.py` (gitignored):
6,000 random draws per country over the benchmark-derived candidate grids of §1.1,
scored by |TFR(2045) − target|; the evaluator was validated against the official CLI
on three known scenarios (2.682 / 3.257 / 4.794 reproduced exactly).
