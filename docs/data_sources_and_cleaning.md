# Data Sources and Cleaning Record

Generated: 2026-08-28 22:15:44

Scope: countries with population > 10M in 2010 (UN WPP 2024 basis), **83 countries**, 1950-2024, 6225 panel rows.

## 1. Data Sources

| Variable | Name | Source | Note |
|---|---|---|---|
| pop | Total population (thousands) | UN WPP 2024 (Medium) | WPP2024_Demographic_Indicators_Medium.csv.gz, TPopulation1July (thousands) |
| tfr | Total fertility rate | UN WPP 2024 (Medium) | same file, TFR indicator |
| urban | Urbanization rate (%) | World Bank WDI | SP.URB.TOTL.IN.ZS, from 1960 |
| gdppc | GDP per capita (2015 constant USD) | World Bank WDI | NY.GDP.PCAP.KD, 2015 constant USD, from 1960 |
| manuf_pc | Manufacturing value added per capita (2015 constant USD) | World Bank WDI | NV.IND.MANF.KD (total) / population, 2015 constant USD |
| elec_access | Access to electricity (%) | World Bank WDI | EG.ELC.ACCS.ZS, from ~1990 |
| mobile | Mobile subscriptions (per 100) | World Bank WDI | IT.CEL.SETS.P2, pre-introduction years set to 0 (fact, not interpolation) |
| broadband | Broadband subscriptions (per 100) | World Bank WDI | IT.NET.BBND.P2, pre-introduction years set to 0 |
| agri_empl | Employment in agriculture (%) | World Bank WDI | SL.AGR.EMPL.ZS |
| literacy | Adult literacy rate (%) | World Bank WDI | SE.ADT.LITR.ZS (adult literacy, sparse years) |
| physicians | Physicians (per 1000) | World Bank WDI | SH.MED.PHYS.ZS (per 1000, sparse years) |
| elec_gen_pc | Electricity generation per capita (kWh) | Our World in Data (Ember/UN) | electricity generation per capita kWh, from 1900 |
| edu_f | Female mean years of schooling (years) | Barro-Lee education database v3 | BL_v3_F.csv, 1950-2015 every 5 years, 15+ age groups population-weighted |
| mat_leave_days | Paid maternity leave (days) | World Bank Women Business and Law | WBL2024-1-0-Historical-Panel-Data.xlsx, annual 1971-2024, unit: days |
| pat_leave_days | Paid paternity leave (days) | World Bank Women Business and Law | same file, 'Length of paid paternity leave', unit: days |
| steel_pc | Steel production per capita (kg) | unavailable (see doc) | OWID slug obsolete; worldsteel yearbooks only recent 10 years; UNdata API 500 |
| nurses | Nursing/midwifery density (per 10000) | WHO GHO (unreachable) | DNS cannot resolve azureedge.us on this network; not included |
| refining_pc | Refining capacity per capita (barrels/person/day) | Energy Institute (403 refused) | official site requires interactive download; 403 in this environment |
| rail_km | Rail length (km) | World Bank WDI | IS.RRS.TOTL.KM total rail length km (source UIC), from ~1995 |

## 2. Coverage

| Variable | Non-null share | Valid country-years |
|---|---|---|
| pop (Total population (thousands)) | 100.0% | 6225 |
| tfr (Total fertility rate) | 100.0% | 6225 |
| urban (Urbanization rate (%)) | 92.2% | 5740 |
| gdppc (GDP per capita (2015 constant USD)) | 84.5% | 5259 |
| manuf_pc (Manufacturing value added per capita (2015 constant USD)) | 58.1% | 3619 |
| elec_access (Access to electricity (%)) | 47.5% | 2960 |
| mobile (Mobile subscriptions (per 100)) | 100.0% | 6225 |
| broadband (Broadband subscriptions (per 100)) | 100.0% | 6225 |
| agri_empl (Employment in agriculture (%)) | 51.4% | 3198 |
| literacy (Adult literacy rate (%)) | 43.0% | 2678 |
| physicians (Physicians (per 1000)) | 85.9% | 5348 |
| elec_gen_pc (Electricity generation per capita (kWh)) | 52.0% | 3240 |
| edu_f (Female mean years of schooling (years)) | 83.3% | 5183 |
| mat_leave_days (Paid maternity leave (days)) | 68.5% | 4266 |
| pat_leave_days (Paid paternity leave (days)) | 68.5% | 4266 |
| rail_km (Rail length (km)) | 32.0% | 1994 |
| rail_pc (Rail length per capita (km/person)) | 19.5% | 1216 |
| nonagri_empl (Non-agricultural employment (%)) | 51.4% | 3198 |
| phys_pc (Physicians per capita (per 1000)) | 85.9% | 5348 |
| gdppc2000 (GDP per capita (2000-equivalent USD)) | 84.5% | 5259 |
| manuf_pc2000 (Manufacturing value added per capita (2000-equivalent USD)) | 58.1% | 3619 |

## 3. Cleaning Rules

1. **Country filter**: UN WPP 2024 countries with population > 10M on 1 July 2010, 83 countries total.
2. **Per-capita conversion**: manufacturing value added divided by year population; (steel/refining not included: sources unavailable).
3. **2000-equivalent USD**: WDI uses 2015 constant USD; converted via the US GDP deflator: 2000 value x [defl(2000)/defl(2015) = 69.011/92.350 = 0.7473].
4. **Interpolation**: linear interpolation within the data range only; at most 5 years of edge extrapolation; mobile/broadband set to 0 for years before technology introduction (fact, not interpolation); maternity/paternity leave are policy variables, carried forward from observations only (no backfill before 1971).
5. **Outlier correction**: isolated spikes with |z| (MAD-robust) > 5 in a single year with normal adjacent years are replaced by the mean of the two neighbors, 11 corrections in total; details in data/outliers.json.
6. **Unavailable variables** (recorded honestly, not imputed): steel per capita (OWID slug obsolete / UNdata 500), refining per capita (Energy Institute 403, interactive download), health workforce density (WHO GHO domain unreachable on this network), road/rail length (OECD/IRF/UIC registration or paywall; no historical panel), house price-to-income / rent-to-income (OECD only from 2010, 35 countries), childcare costs / child benefits (OECD from 2004, 38 countries), tuition-to-income (no systematic global source).

## 4. Data Acquisition Code

- `src/01_fetch_sources.py` - downloads all sources into data/raw/, source status in data/raw/source_log.json
- `src/02_build_panel.py` - cleaning/merging/interpolation/outlier correction; outputs data/panel.csv