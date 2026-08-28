# -*- coding: utf-8 -*-
"""
02_build_panel.py - build the country-year panel data/panel.csv

Pipeline:
  1. Read each source in data/raw/; every source adapter yields (iso3, year, value) series
  2. Country filter: WPP population > 10M in 2010
  3. Per-capita conversion (steel/electricity/refining/health/road/rail divided by population),
     non-agricultural employment = 100 - agriculture, manufacturing per capita
  4. Convert to 2000-equivalent USD (2015 constant USD x US deflator(2000)/deflator(2015))
  5. Interpolation (within data range only) + pre-introduction zeros for mobile/broadband +
     step-constant policy variables
  6. Outlier correction (|z| > 5 and violating trend), logged to the cleaning doc
  7. Output data/panel.csv + data/coverage.json + docs/data_sources_and_cleaning.md

Usage: python 02_build_panel.py
"""
import json
import os
import re
import sys
import time

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(BASE, "data", "raw")
OUT = os.path.join(BASE, "data")
DOCS = os.path.join(BASE, "docs")
os.makedirs(DOCS, exist_ok=True)

YEARS = list(range(1950, 2025))

# ============ Variable definitions ============
# name: (source file, description)
VARS = {
    "pop":            ("wpp2024_demo.csv", "total population (thousands, UN WPP 2024)"),
    "tfr":            ("wpp2024_demo.csv", "total fertility rate (UN WPP 2024)"),
    "urban":          ("wdi_urban.csv", "urbanization rate % (WDI)"),
    "gdppc":          ("wdi_gdppc_kd.csv", "GDP per capita, 2015 constant USD (WDI)"),
    "manuf_pc":       ("wdi_manuf_kd.csv", "manufacturing value added per capita, 2015 constant USD (WDI)"),
    "elec_access":    ("wdi_elec_access.csv", "access to electricity % (WDI)"),
    "mobile":         ("wdi_mobile.csv", "mobile subscriptions per 100 people (WDI)"),
    "broadband":      ("wdi_broadband.csv", "broadband subscriptions per 100 people (WDI)"),
    "agri_empl":      ("wdi_agri_empl.csv", "employment in agriculture % (WDI)"),
    "literacy":       ("wdi_literacy.csv", "adult literacy rate % (WDI)"),
    "physicians":     ("wdi_physicians.csv", "physicians per 1000 people (WDI)"),
    "elec_gen_pc":    ("owid_elec_gen_pc.csv", "electricity generation per capita kWh (OWID)"),
    "steel_pc":       ("owid_steel.csv", "steel production (tonnes, OWID/worldsteel)"),
    "edu_f":          ("barro_lee.csv", "female mean years of schooling (Barro-Lee)"),
    "mat_leave_days": ("wbl_panel.xlsx", "paid maternity leave (days, WBL)"),
    "pat_leave_days": ("wbl_panel.xlsx", "paid paternity leave (days, WBL)"),
    "nurses":         ("gho_nurses.csv", "nursing/midwifery density per 10000 (GHO)"),
    "refining_pc":    ("energy_inst_refining.xlsx", "refining capacity (thousand barrels/day, EI)"),
    "rail_km":        ("wdi_rail_km.csv", "rail length (km, WDI/UIC)"),
}

# Interpolation rules: "linear" in-range linear; "zero_before" zeros before the first
# observation (technology diffusion vars); "step" step-constant (policy vars)
INTERP_RULE = {
    "pop": "linear", "tfr": "linear", "urban": "linear",
    "gdppc": "linear", "manuf_pc": "linear", "elec_access": "linear",
    "mobile": "zero_before", "broadband": "zero_before",
    "agri_empl": "linear", "literacy": "linear", "physicians": "linear",
    "elec_gen_pc": "linear", "steel_pc": "linear", "edu_f": "linear",
    "mat_leave_days": "step", "pat_leave_days": "step", "nurses": "linear",
    "refining_pc": "linear",
    "rail_km": "linear",
}

RAW_TO_PANEL = {  # panel column name -> source variable column name
    "gdppc": "gdppc", "manuf_pc": "manuf_pc", "elec_access": "elec_access",
    "mobile": "mobile", "broadband": "broadband", "agri_empl": "agri_empl",
    "literacy": "literacy", "physicians": "physicians",
    "elec_gen_pc": "elec_gen_pc", "steel_pc": "steel_pc", "edu_f": "edu_f",
    "mat_leave_days": "mat_leave_days", "pat_leave_days": "pat_leave_days", "nurses": "nurses",
    "refining_pc": "refining_pc",
    "rail_km": "rail_km",
}

clean_log = []  # (iso3, var, year, old value, new value, reason)


def log_clean(iso3, var, year, old, new, reason):
    clean_log.append({"iso3": iso3, "var": var, "year": year,
                      "old": old, "new": new, "reason": reason})


# ============ Source adapters ============
def load_wpp():
    """Return dict of pop/tfr Series keyed by (iso3, year); urbanization comes from WDI"""
    path = os.path.join(RAW, "wpp2024_demo.csv")
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df[df["Variant"] == "Medium"]
    df = df[df["LocTypeName"] == "Country/Area"]
    cols = {c: c.strip() for c in df.columns}
    df = df.rename(columns=cols)
    df["Time"] = df["Time"].astype(int)
    out = {}
    out["pop"] = df.set_index(["ISO3_code", "Time"])["TPopulation1July"]  # thousands
    out["tfr"] = df.set_index(["ISO3_code", "Time"])["TFR"]
    return out, df


def load_wdi(name):
    path = os.path.join(RAW, name)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["year"] = df["year"].astype(int)
    return df.set_index(["iso3", "year"])["value"]


def load_owid(name):
    """OWID full-format CSV: leading # comment lines, columns Entity/Code/Year/<value>"""
    path = os.path.join(RAW, name)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, comment="#")
    code_col = "Code" if "Code" in df.columns else "code"
    year_col = "Year" if "Year" in df.columns else "year"
    val_col = df.columns[3] if len(df.columns) > 3 else df.columns[-1]
    df = df.rename(columns={code_col: "code", year_col: "year", val_col: "value"})
    df["year"] = df["year"].astype(int)
    df = df[df["code"].notna()]
    return df.set_index(["code", "year"])["value"]


def load_barro_lee():
    """Barro-Lee BL_v3_F: female age groups, population-weighted mean years of schooling 15+"""
    path = os.path.join(RAW, "barro_lee.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df = df[df["sex"].astype(str).str.upper() == "F"]
    df["year"] = df["year"].astype(int)
    w = df.groupby(["WBcode", "year"]).apply(
        lambda g: float(np.average(g["yr_sch"], weights=g["pop"].fillna(0))),
        include_groups=False)
    w = w.dropna()
    return w


def load_wbl():
    """WBL Parenthood sheet: maternity/paternity leave weeks, annual 1971-2024"""
    path = os.path.join(RAW, "wbl_panel.xlsx")
    if not os.path.exists(path):
        return None
    df = pd.read_excel(path, sheet_name="Parenthood")
    df["year"] = df["Report Year"].astype(int)
    out = {}
    if "Length of paid maternity leave" in df.columns:
        out["mat_leave_days"] = df.set_index(["Economy Code", "year"])["Length of paid maternity leave"]
    if "Length of paid paternity leave" in df.columns:
        out["pat_leave_days"] = df.set_index(["Economy Code", "year"])["Length of paid paternity leave"]
    return out


def load_gho(name):
    return load_wdi(name)


def load_refining():
    path = os.path.join(RAW, "energy_inst_refining.xlsx")
    if not os.path.exists(path):
        return None
    df = pd.read_excel(path, sheet_name=None)
    # EI statistical review workbook layout: usually a 'Refinery throughput' sheet
    # with columns = countries, rows = years
    best = None
    for sn, sheet in df.items():
        if re.search(r"refin", sn, re.I):
            best = sheet
            break
    if best is None:
        best = next(iter(df.values()))
    # Try long or wide format
    cols = list(best.columns)
    if best.iloc[0, 0] in (1950, 1965) or str(best.iloc[0, 0]).isdigit():
        # Wide format: first column is year
        best = best.rename(columns={cols[0]: "year"})
        best = best.melt(id_vars="year", var_name="country", value_name="value")
    # Country name -> iso3 requires a mapping; best effort in stage 02
    return best


# ============ Main pipeline ============
def main():
    print("== Building panel ==")
    wpp, wpp_df = load_wpp()
    print(f"  WPP: variables {list(wpp.keys())}, rows {len(wpp_df)}")

    # 1) Base frame: all countries 1950-2024 x variables
    series = {}
    for k, v in wpp.items():
        series[k] = v
    for name, (fname, desc) in VARS.items():
        if name in ("pop", "tfr"):
            continue
        if fname.startswith("wdi_"):
            s = load_wdi(fname)
        elif fname.startswith("owid_"):
            s = load_owid(fname)
        elif fname == "barro_lee.csv" or fname == "barro_lee.xlsx":
            s = load_barro_lee()
        elif fname == "wbl_panel.xlsx":
            wbl = load_wbl()
            if wbl:
                series.update(wbl)
            s = None
        elif fname.startswith("gho_"):
            s = load_gho(fname)
        elif fname == "energy_inst_refining.xlsx":
            s = load_refining()
        else:
            s = None
        if s is not None:
            series[name] = s
            print(f"  {name}: {len(s)} observations ({desc})")
        else:
            print(f"  {name}: no data")

    # 2) Country filter: population > 10M in 2010 (TPopulation1July unit: thousands)
    pop10 = series["pop"]
    big = set()
    for (iso3, year), v in pop10.items():
        if year == 2010 and not pd.isna(v) and v > 10000:
            big.add(iso3)
    # Also keep countries missing in 2010 WPP but with population > 10M in any year
    # 1950-2024 (edge safety)
    if len(big) < 80:
        for (iso3, year), v in pop10.items():
            if not pd.isna(v) and v > 10000:
                big.add(iso3)
    print(f"  Filter: {len(big)} countries with 2010 population > 10M")

    # 3) Build country name map
    loc = wpp_df[["ISO3_code", "Location"]].drop_duplicates().set_index("ISO3_code")["Location"]
    big = sorted(big)

    # 4) Assemble panel
    idx = pd.MultiIndex.from_product([big, YEARS], names=["iso3", "year"])
    panel = pd.DataFrame(index=idx)
    panel["country"] = [loc.get(i, i) for i in panel.index.get_level_values("iso3")]
    for name, s in series.items():
        if s is None:
            continue
        panel[name] = s.reindex(panel.index) if hasattr(s, "reindex") else np.nan

    # 5) Per-capita conversion (raw totals)
    if "manuf_pc" in panel.columns:
        panel["manuf_pc"] = panel["manuf_pc"] / (panel["pop"] * 1000.0)  # total USD -> per capita
    if "steel_pc" in panel.columns:
        panel["steel_pc"] = panel["steel_pc"] * 1000.0 / (panel["pop"] * 1000.0)
    if "refining_pc" in panel.columns:
        panel["refining_pc"] = panel["refining_pc"] * 1000.0 / (panel["pop"] * 1000.0)
    if "rail_km" in panel.columns:
        panel["rail_pc"] = panel["rail_km"] / (panel["pop"] * 1000.0)  # km per capita

    # 6) Interpolation (linear: within-range with up to 5-year edge extrapolation;
    #    step: policy vars forward-only; zero_before: 0 before introduction)
    for name, rule in INTERP_RULE.items():
        if name not in panel.columns:
            continue
        dfv = panel[name].unstack("year")  # iso3 x year
        if rule == "linear":
            dfv = dfv.interpolate(axis=1, limit_direction="both", limit=5)
        elif rule == "zero_before":
            dfv = dfv.interpolate(axis=1, limit_direction="both", limit=5).fillna(0)
        elif rule == "step":
            dfv = dfv.ffill(axis=1)  # policy carries forward, no backfill before first observation
        panel[name] = dfv.stack().reindex(panel.index)

    # 7) Derived variables (after interpolation)
    if "agri_empl" in panel.columns:
        panel["nonagri_empl"] = 100.0 - panel["agri_empl"]
    if "physicians" in panel.columns:
        panel["phys_pc"] = panel["physicians"]  # already per 1000
    # 2000-equivalent USD: 2015 constant USD x US GDP deflator defl(2000)/defl(2015) = 0.7473
    defl = load_wdi("wdi_us_gdp_defl.csv")
    if defl is not None:
        try:
            r2000 = float(defl.loc[("USA", 2000)])
            r2015 = float(defl.loc[("USA", 2015)])
            ratio = r2000 / r2015
            if "gdppc" in panel.columns:
                panel["gdppc2000"] = panel["gdppc"] * ratio
            if "manuf_pc" in panel.columns:
                panel["manuf_pc2000"] = panel["manuf_pc"] * ratio
            print(f"  USD conversion: defl2000/defl2015 = {ratio:.4f}")
        except Exception as e:
            print(f"  conversion failed: {e}")

    # 8) Outlier correction: isolated spikes with |z| > 5 on the time series,
    #    replaced by the mean of the two neighbors
    for name in panel.columns:
        if name in ("iso3", "year", "country") or panel[name].dtype.kind not in "fi":
            continue
        dfv = panel[name].unstack("year")  # rows = countries, columns = years
        for c in dfv.index:
            s = dfv.loc[c].astype(float)
            med = s.median()
            if pd.isna(med):
                continue
            mad = (s - med).abs().median() * 1.4826
            if mad <= 0 or pd.isna(mad):
                continue
            z = (s - med) / mad
            # Only single-year isolated spikes: the two years on either side are not spikes
            for y in s.index:
                if abs(z[y]) > 5:
                    nz = sum(1 for y2 in (y - 2, y - 1, y + 1, y + 2)
                             if y2 in s.index and abs(z[y2]) > 5)
                    if nz == 0:
                        old = s[y]
                        neigh = [s[y2] for y2 in (y - 1, y + 1) if y2 in s.index and not pd.isna(s[y2])]
                        if neigh:
                            new = float(np.mean(neigh))
                            log_clean(c, name, y, old, new, "outlier (isolated spike, |z|>5)")
                            dfv.loc[c, y] = new
        panel[name] = dfv.stack().reindex(panel.index)

    # 9) Coverage statistics
    cov = {}
    for name in panel.columns:
        if name in ("iso3", "year", "country"):
            continue
        n = panel[name].notna().sum()
        cov[name] = {"nonnull": int(n), "share": round(float(n) / len(panel), 4)}
    with open(os.path.join(OUT, "coverage.json"), "w", encoding="utf-8") as f:
        json.dump(cov, f, ensure_ascii=False, indent=1)

    # 10) Output
    out_path = os.path.join(OUT, "panel.csv")
    panel = panel.reset_index()
    panel.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  panel.csv: {panel.shape}")
    print("  coverage: " + ", ".join(f"{k}={v['share']:.2f}" for k, v in cov.items()))
    if clean_log:
        with open(os.path.join(OUT, "outliers.json"), "w", encoding="utf-8") as f:
            json.dump(clean_log, f, ensure_ascii=False, indent=1)
        print(f"  outlier corrections: {len(clean_log)}")

    # 11) Cleaning doc
    write_cleaning_doc(cov, panel)
    print("== Done ==")


VAR_NAMES = {
    "pop": "Total population (thousands)", "tfr": "Total fertility rate",
    "urban": "Urbanization rate (%)",
    "gdppc": "GDP per capita (2015 constant USD)",
    "gdppc2000": "GDP per capita (2000-equivalent USD)",
    "manuf_pc": "Manufacturing value added per capita (2015 constant USD)",
    "manuf_pc2000": "Manufacturing value added per capita (2000-equivalent USD)",
    "elec_access": "Access to electricity (%)", "mobile": "Mobile subscriptions (per 100)",
    "broadband": "Broadband subscriptions (per 100)",
    "agri_empl": "Employment in agriculture (%)",
    "nonagri_empl": "Non-agricultural employment (%)",
    "literacy": "Adult literacy rate (%)",
    "physicians": "Physicians (per 1000)", "phys_pc": "Physicians per capita (per 1000)",
    "elec_gen_pc": "Electricity generation per capita (kWh)",
    "steel_pc": "Steel production per capita (kg)",
    "edu_f": "Female mean years of schooling (years)",
    "mat_leave_days": "Paid maternity leave (days)",
    "pat_leave_days": "Paid paternity leave (days)",
    "nurses": "Nursing/midwifery density (per 10000)",
    "refining_pc": "Refining capacity per capita (barrels/person/day)",
    "rail_km": "Rail length (km)",
    "rail_pc": "Rail length per capita (km/person)",
}

SRC_INFO = {
    "pop": ("UN WPP 2024 (Medium)", "https://population.un.org/wpp/",
            "WPP2024_Demographic_Indicators_Medium.csv.gz, TPopulation1July (thousands)"),
    "tfr": ("UN WPP 2024 (Medium)", "https://population.un.org/wpp/",
            "same file, TFR indicator"),
    "urban": ("World Bank WDI", "https://api.worldbank.org/v2/country/ALL/indicator/SP.URB.TOTL.IN.ZS",
              "SP.URB.TOTL.IN.ZS, from 1960"),
    "gdppc": ("World Bank WDI", "https://api.worldbank.org/v2/country/ALL/indicator/NY.GDP.PCAP.KD",
              "NY.GDP.PCAP.KD, 2015 constant USD, from 1960"),
    "manuf_pc": ("World Bank WDI", "https://api.worldbank.org/v2/country/ALL/indicator/NV.IND.MANF.KD",
                 "NV.IND.MANF.KD (total) / population, 2015 constant USD"),
    "elec_access": ("World Bank WDI", "https://api.worldbank.org/v2/country/ALL/indicator/EG.ELC.ACCS.ZS",
                    "EG.ELC.ACCS.ZS, from ~1990"),
    "mobile": ("World Bank WDI", "https://api.worldbank.org/v2/country/ALL/indicator/IT.CEL.SETS.P2",
               "IT.CEL.SETS.P2, pre-introduction years set to 0 (fact, not interpolation)"),
    "broadband": ("World Bank WDI", "https://api.worldbank.org/v2/country/ALL/indicator/IT.NET.BBND.P2",
                  "IT.NET.BBND.P2, pre-introduction years set to 0"),
    "agri_empl": ("World Bank WDI", "https://api.worldbank.org/v2/country/ALL/indicator/SL.AGR.EMPL.ZS",
                  "SL.AGR.EMPL.ZS"),
    "literacy": ("World Bank WDI", "https://api.worldbank.org/v2/country/ALL/indicator/SE.ADT.LITR.ZS",
                 "SE.ADT.LITR.ZS (adult literacy, sparse years)"),
    "physicians": ("World Bank WDI", "https://api.worldbank.org/v2/country/ALL/indicator/SH.MED.PHYS.ZS",
                   "SH.MED.PHYS.ZS (per 1000, sparse years)"),
    "elec_gen_pc": ("Our World in Data (Ember/UN)", "https://ourworldindata.org/grapher/per-capita-electricity-generation.csv",
                    "electricity generation per capita kWh, from 1900"),
    "edu_f": ("Barro-Lee education database v3", "https://github.com/barrolee/BarroLeeDataSet",
              "BL_v3_F.csv, 1950-2015 every 5 years, 15+ age groups population-weighted"),
    "mat_leave_days": ("World Bank Women Business and Law", "https://wbl.worldbank.org/",
                       "WBL2024-1-0-Historical-Panel-Data.xlsx, annual 1971-2024, unit: days"),
    "pat_leave_days": ("World Bank Women Business and Law", "https://wbl.worldbank.org/",
                       "same file, 'Length of paid paternity leave', unit: days"),
    "steel_pc": ("unavailable (see doc)", "-", "OWID slug obsolete; worldsteel yearbooks only recent 10 years; UNdata API 500"),
    "nurses": ("WHO GHO (unreachable)", "https://ghoapi.azureedge.us/api/",
               "DNS cannot resolve azureedge.us on this network; not included"),
    "refining_pc": ("Energy Institute (403 refused)", "https://www.energyinst.org/statistical-review",
                    "official site requires interactive download; 403 in this environment"),
    "rail_km": ("World Bank WDI", "https://api.worldbank.org/v2/country/ALL/indicator/IS.RRS.TOTL.KM",
                "IS.RRS.TOTL.KM total rail length km (source UIC), from ~1995"),
}


def write_cleaning_doc(cov, panel):
    lines = []
    lines.append("# Data Sources and Cleaning Record")
    lines.append("")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append(f"Scope: countries with population > 10M in 2010 (UN WPP 2024 basis), "
                 f"**{panel['iso3'].nunique()} countries**, 1950-2024, {len(panel)} panel rows.")
    lines.append("")
    lines.append("## 1. Data Sources")
    lines.append("")
    lines.append("| Variable | Name | Source | Note |")
    lines.append("|---|---|---|---|")
    for name, (src, url, note) in SRC_INFO.items():
        lines.append(f"| {name} | {VAR_NAMES.get(name, name)} | {src} | {note} |")
    lines.append("")
    lines.append("## 2. Coverage")
    lines.append("")
    lines.append("| Variable | Non-null share | Valid country-years |")
    lines.append("|---|---|---|")
    for name, v in cov.items():
        lines.append(f"| {name} ({VAR_NAMES.get(name, '')}) | {v['share']:.1%} | {v['nonnull']} |")
    lines.append("")
    lines.append("## 3. Cleaning Rules")
    lines.append("")
    lines.append("1. **Country filter**: UN WPP 2024 countries with population > 10M on "
                 "1 July 2010, 83 countries total.")
    lines.append("2. **Per-capita conversion**: manufacturing value added divided by "
                 "year population; (steel/refining not included: sources unavailable).")
    lines.append("3. **2000-equivalent USD**: WDI uses 2015 constant USD; converted via the "
                 "US GDP deflator: 2000 value x [defl(2000)/defl(2015) = 69.011/92.350 = 0.7473].")
    lines.append("4. **Interpolation**: linear interpolation within the data range only; at "
                 "most 5 years of edge extrapolation; mobile/broadband set to 0 for years "
                 "before technology introduction (fact, not interpolation); maternity/"
                 "paternity leave are policy variables, carried forward from observations "
                 "only (no backfill before 1971).")
    lines.append("5. **Outlier correction**: isolated spikes with |z| (MAD-robust) > 5 in a "
                 "single year with normal adjacent years are replaced by the mean of the two "
                 "neighbors, %d corrections in total; details in data/outliers.json."
                 % len(clean_log))
    lines.append("6. **Unavailable variables** (recorded honestly, not imputed): steel per "
                 "capita (OWID slug obsolete / UNdata 500), refining per capita (Energy "
                 "Institute 403, interactive download), health workforce density (WHO GHO "
                 "domain unreachable on this network), road/rail length (OECD/IRF/UIC "
                 "registration or paywall; no historical panel), house price-to-income / "
                 "rent-to-income (OECD only from 2010, 35 countries), childcare costs / "
                 "child benefits (OECD from 2004, 38 countries), tuition-to-income (no "
                 "systematic global source).")
    lines.append("")
    lines.append("## 4. Data Acquisition Code")
    lines.append("")
    lines.append("- `src/01_fetch_sources.py` - downloads all sources into data/raw/, "
                 "source status in data/raw/source_log.json")
    lines.append("- `src/02_build_panel.py` - cleaning/merging/interpolation/outlier "
                 "correction; outputs data/panel.csv")
    doc_path = os.path.join(DOCS, "data_sources_and_cleaning.md")
    with open(doc_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  cleaning doc -> {doc_path}")


if __name__ == "__main__":
    main()
