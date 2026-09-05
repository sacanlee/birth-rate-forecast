# Scenario Forecasts to 2045: Nigeria, Niger and DR Congo

What would the total fertility rate (TFR) of Nigeria, Niger and the Democratic Republic
of the Congo (DRC) be in **2045** if each country reached India's **2010** structural
development levels, together with a GDP-per-capita milestone expressed in **2024 US
dollars**? This report answers the three scenarios using the repository's trained ΔTFR
neural-network model (`src/12_predict_scenario.py`).

Date: 2026-09-05. All inputs are taken from `data/panel.csv` (the same cleaned
1950-2024 panel of 83 countries used for training); the trained model is
`models/final_weights_delta_k5.npz`.

## 1. Scenario definitions

| # | Country | Targets to reach by 2045 (India-2010 levels) | GDP per capita target |
|---|---|---|---|
| 1 | Nigeria (NGA) | adult literacy, electricity generation p.c. | 2,500 USD (2024 dollars) |
| 2 | Niger (NER) | adult literacy, non-agricultural employment share, electricity generation p.c. | 2,000 USD (2024 dollars) |
| 3 | DR Congo (COD) | adult literacy, non-agricultural employment share, electricity generation p.c. | 2,000 USD (2024 dollars) |

## 2. Data and processing

### 2.1 Target values — India 2010 (from the panel)

| Variable | India 2010 | Units / source |
|---|---|---|
| `literacy` | **68.33** | Adult literacy rate, % of population 15+ (WDI SE.ADT.LITR.ZS) |
| `elec_gen_pc` | **753.04** | Electricity generation per capita, kWh (Our World in Data / Ember-UN) |
| `nonagri_empl` | **48.94** | Non-agricultural employment, % of total employment (= 100 − `agri_empl`, WDI SL.AGR.EMPL.ZS) |

Reference (context): India's TFR was 2.60 in 2010 and its GDP per capita was 923
(2000-equivalent USD).

### 2.2 Conversion of the GDP target into model units

The model's GDP feature `gdppc2000` is GDP per capita in **constant-2000-equivalent
USD** (WDI constant-2015 USD × 0.7473, where 0.7473 = US GDP deflator
2000/2015 = 69.011/92.350). Because the scenarios are stated in **2024 current
dollars**, each target is deflated with the same US-deflator chain:

```
gdppc2000 target = USD_2024 × defl(2000)/defl(2024) = USD_2024 × 69.011/119.027
                 = USD_2024 × 0.5798
```

| Scenario | Target (2024 USD) | → `gdppc2000` target | Current `gdppc2000` (2024) |
|---|---|---|---|
| NGA | 2,500 | **1,449.5** | 1,737.1 |
| NER / COD | 2,000 | **1,159.6** | 440.2 / 414.1 |

Important caveat: Nigeria's model-unit GDP (1,737) already **exceeds** the converted
target (1,449.5). WDI's constant-USD series is based on the 2015 exchange rate, so the
2023 naira devaluation (which depressed *nominal* US-dollar GDP per capita) does not
show up in the real series. In real terms, therefore, Nigeria is already past the
"2,500 in 2024 dollars" threshold and the GDP condition is effectively non-binding —
the model receives a small GDP *decline* (1,737 → 1,450). This is a data-definition
artifact, not an assumption about economic decline, and is discussed in §5.

### 2.3 Current (2024) values of the three countries

| Variable | NGA | NER | COD |
|---|---|---|---|
| TFR (UN WPP 2024) | 4.382 | 5.935 | 5.981 |
| Literacy (%) | 70.41 | 35.61 | 68.50 |
| Electricity p.c. (kWh) | 161.4 | 35.1 | 145.7 |
| Non-agricultural employment (%) | 65.90 | 26.54 | 41.10 |
| `gdppc2000` | 1,737.1 | 440.2 | 414.1 |

Two of the "India-2010" conditions are already satisfied today:

- Nigeria's literacy (70.4%) is already above India's 2010 level (68.3%);
- DRC's literacy (68.5%) is already at India's 2010 level.

The genuinely binding conditions are electricity (all three countries: 3-10× short),
GDP (Niger, DRC: ~2.6-2.8× short in model units) and, for Niger, literacy (35.6 → 68.3)
and the employment structure (26.5 → 48.9).

## 3. Method

**Model.** The primary model of this repository predicts the 5-year TFR *change*
(ΔTFR) with a pure-numpy MLP (~11M parameters) trained on the 1950-2024 panel of 83
countries. Inputs are 58 values: TFR level, 14 structural feature levels, 14 five-year
changes and 29 missingness masks. On time extrapolation (2011-2024) it achieves ΔTFR
MAE 0.097 and TFR-level R² 0.9925 — i.e. a typical 5-year step is accurate to roughly
±0.1 births.

**Forecast.** Starting from each country's latest observed year (2024), the model is
chained forward in 5-year steps to 2045 (four full windows 2024→2029→…→2044 plus a
1-year partial window, which is scaled linearly — an approximation). The set features
ramp **linearly from their 2024 values to the scenario targets in 2045** and stay at
the target afterwards; all unset features (urbanization, physicians, mobile
penetration, female schooling `edu_f`, maternity leave, etc.) are held at their 2024
levels — a strict *ceteris paribus* assumption. Variables that lack 2024 data in the
panel (`edu_f`, `rail_pc` for all three countries; `mat_leave_days`/`pat_leave_days`
for COD) are handled by the model's missingness masks (treated as "unknown", exactly
as in training). Features are log-transformed where applicable
(`elec_gen_pc`, `gdppc2000`) and z-scored with training-set statistics.

**Counterfactual decomposition.** The marginal contribution of each scenario variable
is the difference between the full-scenario TFR(2045) and the TFR(2045) obtained when
that single variable is held at its 2024 value; the "zero-change baseline" holds all
variables at their 2024 values. Because the model is nonlinear, the sum of marginal
effects differs from the total by an interaction residual.

## 4. Results

### 4.1 Headline TFR(2045)

| Scenario | TFR 2024 | **TFR 2045 (scenario)** | Reduction | Zero-change baseline TFR 2045 |
|---|---|---|---|---|
| 1. Nigeria (India-2010 literacy + electricity; GDP 2,500 USD-2024) | 4.382 | **2.80** | −1.58 (−36%) | 2.67 |
| 2. Niger (India-2010 literacy + non-agri empl + electricity; GDP 2,000 USD-2024) | 5.935 | **3.26** | −2.68 (−45%) | 4.15 |
| 3. DR Congo (same targets as Niger) | 5.981 | **4.91** | −1.07 (−18%) | 5.17 |

Chained TFR paths (2024 → 2029 → 2034 → 2039 → 2044 → 2045):

- **Nigeria:** 4.382 → 3.956 → 3.563 → 3.210 → 2.871 → **2.804**
- **Niger:** 5.935 → 5.248 → 4.576 → 3.933 → 3.325 → **3.257**
- **DR Congo:** 5.981 → 5.682 → 5.385 → 5.121 → 4.951 → **4.909**

### 4.2 Marginal contribution of each scenario condition (ΔTFR over 2024-2045)

| Condition | Nigeria | Niger | DR Congo |
|---|---|---|---|
| Zero-change baseline (endogenous path) | −1.711 | −1.785 | −0.810 |
| `literacy` → 68.3% | +0.015 | **−0.517** | +0.004 |
| `nonagri_empl` → 48.9% | — | **−0.339** | +0.022 |
| `elec_gen_pc` → 753 kWh | +0.130 | **−0.443** | **−0.112** |
| `gdppc2000` → target | +0.005 | −0.088 | **−0.409** |
| Interaction residual | −0.016 | +0.494 | +0.232 |
| **Total ΔTFR** | **−1.578** | **−2.679** | **−1.072** |

Positive values mean the condition *offsets* part of the decline (e.g. a condition
that is already met, a GDP target below the current real level, or nonlinear
interactions between simultaneous changes).

## 5. Interpretation

**Nigeria (TFR 2024 4.38 → 2.80 by 2045).** Almost the entire decline is the model's
endogenous convergence path: even with *no* structural change at all, the chained
model reaches 2.67 by 2045 (the model's learned persistence implies continuing decline
from current levels regardless). The scenario conditions add essentially nothing
beyond that — indeed the scenario TFR (2.80) is slightly *above* the zero-change path
(2.67) because, in the model's units, two of the three conditions are non-binding
(GDP: current real level already above the converted 2,500-USD-2024 threshold;
literacy: 70.4% > 68.3%) and the electricity expansion 161→753 kWh has a small
positive marginal effect in this configuration. The headline answer for Nigeria is
therefore **TFR ≈ 2.8**, roughly the level the model associates with Nigeria's current
structure once the ongoing transition plays out — the "India-2010" conditions as
stated do not push Nigeria below its structural path. If the GDP target were instead
read as 2,500 in *constant* USD (i.e. a real 44% rise over 2024), a further reduction
of order 0.1-0.2 can be expected; the sign convention of the target changes little for
Nigeria because GDP's marginal effect at these levels is small.

**Niger (TFR 2024 5.94 → 3.26 by 2045).** This is the strongest response of the three.
Reaching India-2010 structural levels roughly *doubles* the decline relative to the
zero-change baseline (4.15). Literacy (→68.3%, +19% of the total reduction) and
electricity (35→753 kWh, +16%) are the largest single drivers, followed by the shift
of employment out of agriculture (26.5→48.9%, +13%); the GDP rise to 2,000-USD-2024
contributes only modestly (+3%) because the marginal effect of GDP is small relative
to the education/energy/employment channels. Note the large positive interaction
residual (+0.49): the marginal effects overlap, so the whole is less than the sum of
its parts. Niger remains well above replacement because its other structural features
(urbanization 18%, female schooling, health infrastructure) are kept at today's very
low levels.

**DR Congo (TFR 2024 5.98 → 4.91 by 2045).** The weakest response. DRC already meets
the literacy condition (68.5% ≈ India-2010 68.3%) and its non-agricultural employment
share (41%) is close; the binding conditions are GDP (414 → 1,160, the single largest
driver at −0.41) and electricity (146 → 753 kWh, −0.11). Yet the model's structural
path for DRC is shallow (−0.81 baseline) and even the full scenario leaves TFR near
4.9 — more than 2 births above replacement. In the model's view, DRC's fertility
decline is held back by conditions *not* varied in the scenario (very low physicians,
female schooling, urbanization around 45%, minimal electrification *access* relative
to population growth), so meeting the India-2010 electricity/income benchmarks alone
does not reproduce India's fertility transition by mid-century.

**Cross-scenario observations.** (i) Electricity access/generation is a genuine
driver for the poorest countries but its marginal effect saturates — it added little
for already-urbanized Nigeria. (ii) Income alone (GDP marginal effects) is a weaker
lever in this model than education/employment structure; the model's permutation
importance ranks GDP first overall (ΔR² 0.27), but at the levels and combinations
considered here the education/employment channels dominate for Niger. (iii) Literacy
matters most where it is still low (Niger); Nigeria and DRC are already at or above
the India-2010 benchmark, so the condition is a formality for them. (iv) All three
forecasts are dominated or co-dominated by the endogenous continuation path: the
model "knows" that high-TFR countries with today's structures keep declining for
decades.

## 6. Caveats

- **Uncertainty.** Per-step ΔTFR MAE is ≈0.1; chaining four 5-year windows compounds
  error, so TFR(2045) carries an uncertainty band of roughly **±0.2-0.3 births**
  (informal estimate), plus structural uncertainty from the ceteris-paribus
  assumption.
- **Ceteris paribus.** Features not in each scenario (urbanization, physicians,
  female schooling, mobile/broadband, leave policies, electrification *access*,
  manufacturing) are frozen at 2024 levels for 21 years. In reality they would keep
  evolving; scenarios that fixed India-2010 levels for *all* features would produce
  lower TFRs (female schooling is the level model's dominant determinant).
- **GDP conversion.** The 2024-dollar targets are converted with the US GDP deflator
  (the project's convention). Nigeria's converted target lies below its current
  constant-USD value because of the 2023 naira devaluation (nominal vs real USD gap);
  results for Nigeria should be read as "non-binding GDP target".
- **Model floor.** Policy/event shocks, conflict, and one-off discontinuities are not
  predictable from structural features (see README); DRC and Niger both carry
  conflict/migration histories that the structural model cannot encode.
- **Reach profile.** Set features ramp linearly 2024→2045 and stay constant after;
  the final 1-year step is linearly scaled from the 5-year prediction.

## 7. Reproduction

```bash
# Scenario 1 - Nigeria: India-2010 literacy + electricity; GDP 2,500 USD-2024 (=1,449.5 model units)
python src/12_predict_scenario.py --iso3 NGA --year 2045 \
  --set '{"literacy": 68.326, "elec_gen_pc": 753.04, "gdppc2000": 1449.5}'

# Scenario 2 - Niger: + non-agri employment; GDP 2,000 USD-2024 (=1,159.6 model units)
python src/12_predict_scenario.py --iso3 NER --year 2045 \
  --set '{"literacy": 68.326, "nonagri_empl": 48.944, "elec_gen_pc": 753.04, "gdppc2000": 1159.6}'

# Scenario 3 - DR Congo (same targets as Niger)
python src/12_predict_scenario.py --iso3 COD --year 2045 \
  --set '{"literacy": 68.326, "nonagri_empl": 48.944, "elec_gen_pc": 753.04, "gdppc2000": 1159.6}'
```

Panel values used above: `literacy` 68.3258 (India 2010), `elec_gen_pc` 753.039
(India 2010), `nonagri_empl` 48.944 (India 2010); deflator ratio
69.011230/119.027387 = 0.5798 (US GDP deflator 2000/2024, base 2020 = 100).

---

# Round 2 — Expanded scenarios (incl. manufacturing, urbanization, electricity access)

Second request (2026-09-05): the same three countries, now also asked to reach
India-2010 **manufacturing value added per capita** (and, for Niger/DRC,
non-agricultural employment), plus higher GDP targets and **absolute** urbanization
and electricity-access conditions. Predicted with the same model and method as Round 1.

## 8. Round-2 scenario definitions

| # | Country | India-2010 targets | Additional conditions by 2045 |
|---|---|---|---|
| 1 | Nigeria | literacy, manuf. p.c., electricity gen. p.c. | GDP 3,000 USD-2024; urban 70%; elec. access 90%+ |
| 2 | Niger | literacy, manuf. p.c., non-agri empl., electricity gen. p.c. | GDP 2,500 USD-2024; urban 30%; elec. access 90%+ |
| 3 | DR Congo | literacy, manuf. p.c., non-agri empl., electricity gen. p.c. | GDP 2,500 USD-2024; urban 60%; elec. access 90%+ |

### 8.1 Target values

India-2010 benchmark added in this round: **`manuf_pc2000` = 141.48** (manufacturing
value added per capita, 2000-equivalent USD; India's 2024 value is 274.9).

GDP conversion (2024 current USD → constant-2000-equivalent USD, ×0.5798, same
deflator chain as Round 1): Nigeria 3,000 → **1,739.4**; Niger/DRC 2,500 → **1,449.5**.

| Condition | NGA (2024 → target) | NER (2024 → target) | COD (2024 → target) |
|---|---|---|---|
| Literacy (%) | 70.41 → 68.33 (met) | 35.61 → 68.33 | 68.50 → 68.33 (met) |
| Manuf. p.c. (2000-USD) | 138.1 → 141.5 (met) | 31.3 → 141.5 | 61.7 → 141.5 |
| Electricity gen. p.c. (kWh) | 161.4 → 753.0 | 35.1 → 753.0 | 145.7 → 753.0 |
| Non-agricultural empl. (%) | — | 26.5 → 48.9 | 41.1 → 48.9 |
| GDP p.c. (2000-USD) | 1,737.1 → 1,739.4 (met) | 440.2 → 1,449.5 | 414.1 → 1,449.5 |
| Urbanization (%) | 63.0 → 70.0 | 18.0 → 30.0 | 44.7 → 60.0 |
| Electricity access (%) | 62.5 → 90.0 | 21.3 → 90.0 | 22.5 → 90.0 |

As in Round 1, Nigeria's GDP condition is non-binding after deflation (its real GDP
per capita already stands at the converted level), and its literacy/manufacturing
targets are already satisfied; Niger and DRC face large jumps on every dimension.

## 9. Round-2 results

### 9.1 Headline TFR(2045)

| Scenario | TFR 2024 | **TFR 2045 (scenario)** | Reduction | Zero-change baseline 2045 | Round-1 result (same countries, fewer conditions) |
|---|---|---|---|---|---|
| 1. Nigeria | 4.382 | **2.91** | −1.47 (−33%) | 2.67 | 2.80 |
| 2. Niger | 5.935 | **4.47** | −1.47 (−25%) | 4.15 | 3.26 |
| 3. DR Congo | 5.981 | **5.06** | −0.92 (−15%) | 5.17 | 4.91 |

Chained TFR paths (2024 → 2029 → 2034 → 2039 → 2044 → 2045):

- **Nigeria:** 4.382 → 4.008 → 3.651 → 3.312 → 2.986 → **2.914**
- **Niger:** 5.935 → 5.445 → 4.980 → 4.667 → 4.538 → **4.470**
- **DR Congo:** 5.981 → 5.709 → 5.442 → 5.225 → 5.070 → **5.057**

Note the striking pattern: adding the manufacturing / urbanization / electricity-
access conditions does **not** lower the predicted TFR relative to Round 1 — for Niger
and DR Congo the forecast is *higher* (4.47 vs 3.26; 5.06 vs 4.91), and Nigeria's is
similar (2.91 vs 2.80). Section 10 explains why.

### 9.2 Marginal contributions of each condition (ΔTFR over 2024-2045)

| Condition | Nigeria | Niger | DR Congo |
|---|---|---|---|
| Zero-change baseline (endogenous path) | −1.711 | −1.785 | −0.810 |
| `literacy` → 68.3% | +0.014 | **−0.227** | +0.000 |
| `manuf_pc2000` → 141.5 | +0.001 | +0.477 | +0.163 |
| `nonagri_empl` → 48.9% | — | +0.038 | +0.073 |
| `elec_gen_pc` → 753 kWh | +0.323 | +0.321 | +0.211 |
| `gdppc2000` → target | −0.000 | **−0.218** | +0.247 |
| `urban` → target | +0.016 | +0.104 | **−0.125** |
| `elec_access` → 90% | +0.093 | +0.903 | +0.269 |
| Interaction residual | −0.204 | −1.080 | −0.952 |
| **Total ΔTFR** | **−1.468** | **−1.466** | **−0.924** |

Positive values offset part of the decline. For Niger, the reductions come from
literacy (−0.23) and GDP (−0.22); for DR Congo only urbanization (−0.13) clearly
reduces TFR while GDP's marginal contribution flips sign relative to Round 1 (+0.25
here vs −0.41 there). Electricity-access, generation and manufacturing show *positive*
offsets in every country when applied simultaneously.

## 10. Interpretation

**Headline answers: Nigeria TFR ≈ 2.9; Niger TFR ≈ 4.5; DR Congo TFR ≈ 5.1 by 2045.**
Nigeria is again dominated by its endogenous convergence path (2.67 baseline), because
after conversion almost every condition is already met or near-met — only
urbanization 63→70, access 62.5→90 and electricity generation 161→753 kWh are genuine
new developments, and the model attributes little fertility change to them at
Nigeria's current TFR level. Niger and DR Congo, where all conditions are real leaps,
still land far higher than their Round-1 forecasts. Three points explain this:

**1. In the training record, "infrastructure catch-up" at high fertility did not
reduce fertility.** The ΔTFR model was trained on historical 5-year windows
(1950-2024). Countries that expanded electricity generation, access or manufacturing
rapidly *while TFR was still 5-6* (energy booms, the 1950s-60s, much of 2000s-2020s
sub-Saharan Africa) typically had flat or slowly-declining fertility; in the data,
fertility fell when education, income and urbanization moved *together* with
infrastructure. The model therefore associates fast electrification at a high current
TFR with little — even slightly positive — ΔTFR, and assigns the decline to the
education/income/urban channels instead. This is a statistical regularity of the
historical sample, not a policy claim that electricity "raises" fertility.

**2. Marginal effects are state-dependent and interact strongly.** Because the
network is nonlinear (11M parameters), the marginal sign of one feature can flip with
the rest of the state: DR Congo's GDP gain reduced TFR by −0.41 in Round 1 (fewer
simultaneous changes) but offsets +0.25 in Round 2. Isolated diagnostics confirm the
interactions drive the result: setting Niger's electricity access to 90% *alone*
changes TFR(2045) by only +0.03 (4.183 vs 4.150 baseline), yet the same condition
shows +0.90 in the full seven-condition scenario, with an interaction residual of
−1.08 absorbing most of the difference. Large simultaneous jumps from a low base have
few in-sample 5-year analogues, so per-condition decompositions are unreliable there;
the chained forecasts in §9.1 are the meaningful outputs.

**3. Education and income remain the strongest levers, and they are capped or
missing here.** Literacy is the main explicit education channel that moves (Niger
−0.23); female mean schooling (`edu_f`, Barro-Lee) — the level model's dominant
determinant — is *not* part of any scenario condition and is treated as unknown by the
model for all three countries because the series ends before 2024. Urbanization is
only raised to 30% (Niger) / 60% (DRC) / 70% (Nigeria) — modest levels that the model
does not consider transitional by themselves.

**Caveats.** (i) Uncertainty band ≈ ±0.2-0.3 births per scenario from chained
per-step error (MAE ≈ 0.1), before structural assumptions. (ii) Ceteris paribus:
physicians, female schooling, mobile/broadband, leave policies stay at 2024 levels.
(iii) The 90%-access condition would realistically be reached well before 2045 by
Niger/DRC (they start at ~21%); the linear ramp spreads it over 21 years, and the
final year is linearly scaled. (iv) Interactions make the marginal decomposition
illustrative only — the composition percentages printed by the script should not be
read as causal shares for these extreme scenarios.

**Practical reading for scenario design.** In this model, fertility responds to
education, income and (for COD) urbanization; energy and manufacturing expansions
matter mostly as correlates of those. A scenario that wants a lower 2045 TFR would
need to condition on female schooling (e.g., India-2010 `edu_f` = 5.26 years),
steeper urbanization or higher income — not on electricity metrics alone.

## 11. Reproduction (Round 2)

```bash
# Nigeria: + manuf, GDP 3,000 USD-2024 (=1,739.4), urban 70, elec access 90
python src/12_predict_scenario.py --iso3 NGA --year 2045 \
  --set '{"literacy": 68.326, "manuf_pc2000": 141.48, "elec_gen_pc": 753.04, \
          "gdppc2000": 1739.4, "urban": 70.0, "elec_access": 90.0}'

# Niger: + manuf, non-agri empl, GDP 2,500 USD-2024 (=1,449.5), urban 30, access 90
python src/12_predict_scenario.py --iso3 NER --year 2045 \
  --set '{"literacy": 68.326, "manuf_pc2000": 141.48, "nonagri_empl": 48.944, \
          "elec_gen_pc": 753.04, "gdppc2000": 1449.5, "urban": 30.0, "elec_access": 90.0}'

# DR Congo: same as Niger but urban 60
python src/12_predict_scenario.py --iso3 COD --year 2045 \
  --set '{"literacy": 68.326, "manuf_pc2000": 141.48, "nonagri_empl": 48.944, \
          "elec_gen_pc": 753.04, "gdppc2000": 1449.5, "urban": 60.0, "elec_access": 90.0}'
```

New India-2010 benchmark: `manuf_pc2000` 141.481. Other conversions as in §7.
